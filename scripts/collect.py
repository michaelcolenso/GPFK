#!/usr/bin/env python3
"""
Collection daemon: pull fresh signals for all tracked companies.

Usage:
  python scripts/collect.py             # one-shot collection
  python scripts/collect.py --daemon    # run continuously on schedule
  python scripts/collect.py --ticker AAPL  # single ticker
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import typer
import structlog
from rich.console import Console
from rich.progress import track

from src.db.models import Company
from src.db.session import init_db, SessionLocal
from src.collectors.sec_edgar import EdgarCollector
from src.collectors.flight_tracker import FlightTracker
from src.collectors.job_postings import JobPostingCollector
from src.collectors.patent_signals import PatentSignalCollector
from src.collectors.lobbying import LobbyingCollector
from src.collectors.options_flow import OptionsFlowCollector
from src.collectors.trademark import TrademarkCollector
from src.collectors.exec_departures import ExecDepartureCollector
from src.collectors.employee_sentiment import EmployeeSentimentCollector
from src.db.models import (
    SecFiling, FlightRecord, JobPostingSnapshot, PatentAssignment,
    LobbyingRecord, OptionsSnapshot, TrademarkFiling,
    ExecutiveDeparture, EmployeeSentimentSnapshot,
)

logger = structlog.get_logger()
console = Console()
app = typer.Typer()


def collect_for_ticker(db, ticker: str, company_name: str, jet_codes: list[str]):
    """Run all collectors for one ticker and persist results."""

    # SEC EDGAR
    edgar = EdgarCollector()
    sec_records = edgar.collect(ticker, days_back=30)
    for r in sec_records:
        existing = (
            db.query(SecFiling)
            .filter(SecFiling.accession_number == r["accession_number"])
            .first()
        )
        if not existing:
            db.add(SecFiling(
                ticker=r["ticker"],
                cik=r["cik"],
                form_type=r["form_type"],
                accession_number=r["accession_number"],
                filed_at=r["filed_at"],
                document_url=r["document_url"],
                mentions_acquisition=r.get("mentions_acquisition", False),
                mentions_merger=r.get("mentions_merger", False),
                counterparty_ticker=r.get("counterparty_ticker"),
            ))
    db.commit()

    # Corporate jets
    if jet_codes:
        flight_tracker = FlightTracker()
        flight_records = flight_tracker.collect(ticker, icao24_codes=jet_codes, days_back=30)
        for r in flight_records:
            if r.get("first_seen"):
                existing = (
                    db.query(FlightRecord)
                    .filter(
                        FlightRecord.icao24 == r["icao24"],
                        FlightRecord.first_seen == r["first_seen"],
                    )
                    .first()
                )
                if not existing:
                    db.add(FlightRecord(
                        icao24=r["icao24"],
                        owner_ticker=r["owner_ticker"],
                        callsign=r.get("callsign"),
                        departure_airport=r.get("departure_airport"),
                        arrival_airport=r.get("arrival_airport"),
                        first_seen=r.get("first_seen"),
                        last_seen=r.get("last_seen"),
                        anomaly_score=r.get("anomaly_score"),
                    ))
        db.commit()

    # Job postings
    job_collector = JobPostingCollector()
    job_records = job_collector.collect(ticker, company_name=company_name)
    for r in job_records:
        existing = (
            db.query(JobPostingSnapshot)
            .filter(
                JobPostingSnapshot.ticker == r["ticker"],
                JobPostingSnapshot.snapshot_date == r["snapshot_date"],
            )
            .first()
        )
        if not existing:
            db.add(JobPostingSnapshot(
                ticker=r["ticker"],
                snapshot_date=r["snapshot_date"],
                posting_count=r["posting_count"],
            ))
    db.commit()

    # Patents
    patent_collector = PatentSignalCollector()
    patent_records = patent_collector.collect(ticker, company_name=company_name, days_back=30)
    for r in patent_records:
        if r.get("reel_frame"):
            existing = (
                db.query(PatentAssignment)
                .filter(PatentAssignment.reel_frame == r["reel_frame"])
                .first()
            )
            if not existing:
                db.add(PatentAssignment(
                    assignor=r["assignor"],
                    assignee=r["assignee"],
                    assignor_ticker=r.get("assignor_ticker"),
                    assignee_ticker=r.get("assignee_ticker"),
                    assignment_date=r["assignment_date"],
                    patent_count=r.get("patent_count", 1),
                    reel_frame=r.get("reel_frame"),
                ))
    db.commit()

    # Lobbying disclosures (Senate LDA)
    lobby_collector = LobbyingCollector()
    lobby_records = lobby_collector.collect(ticker, company_name=company_name, days_back=90)
    for r in lobby_records:
        if r.get("filing_uuid"):
            existing = (
                db.query(LobbyingRecord)
                .filter(LobbyingRecord.filing_uuid == r["filing_uuid"])
                .first()
            )
            if not existing:
                from datetime import datetime as _dt
                filed_at = None
                if r.get("filed_at_str"):
                    try:
                        filed_at = _dt.strptime(r["filed_at_str"][:10], "%Y-%m-%d")
                    except ValueError:
                        pass
                db.add(LobbyingRecord(
                    ticker=ticker,
                    filing_uuid=r["filing_uuid"],
                    filing_type=r.get("filing_type"),
                    filing_year=r.get("filing_year"),
                    period_display=r.get("period_display"),
                    filed_at=filed_at,
                    registrant_name=r.get("registrant_name"),
                    client_name=r.get("client_name"),
                    income=r.get("income"),
                    expenses=r.get("expenses"),
                    issue_codes=",".join(r.get("issue_codes", [])),
                    descriptions=r.get("descriptions"),
                    is_ma_signal=r.get("is_ma_signal", False),
                ))
    db.commit()

    # Options flow (yfinance)
    options_collector = OptionsFlowCollector()
    options_records = options_collector.collect(ticker)
    for r in options_records:
        existing = (
            db.query(OptionsSnapshot)
            .filter(
                OptionsSnapshot.ticker == r["ticker"],
                OptionsSnapshot.snapshot_date == r["snapshot_date"],
            )
            .first()
        )
        if not existing:
            # Compute anomaly score from history
            history = (
                db.query(OptionsSnapshot)
                .filter(OptionsSnapshot.ticker == ticker)
                .order_by(OptionsSnapshot.snapshot_date)
                .all()
            )
            hist_dicts = [
                {
                    "otm_call_volume": s.otm_call_volume,
                    "put_call_ratio": s.put_call_ratio or 1.0,
                    "volume_oi_ratio": s.volume_oi_ratio or 0.0,
                }
                for s in history
            ]
            anomaly = options_collector.compute_anomaly_score(hist_dicts + [r])
            db.add(OptionsSnapshot(
                ticker=r["ticker"],
                snapshot_date=r["snapshot_date"],
                current_price=r.get("current_price"),
                otm_call_volume=r.get("otm_call_volume", 0),
                otm_call_oi=r.get("otm_call_oi", 0),
                total_call_volume=r.get("total_call_volume", 0),
                total_put_volume=r.get("total_put_volume", 0),
                put_call_ratio=r.get("put_call_ratio"),
                volume_oi_ratio=r.get("volume_oi_ratio"),
                max_volume_oi_ratio=r.get("max_volume_oi_ratio"),
                avg_otm_iv=r.get("avg_otm_iv"),
                anomaly_score=anomaly,
            ))
    db.commit()

    # Trademark registrations (USPTO)
    tm_collector = TrademarkCollector()
    tm_records = tm_collector.collect(ticker, company_name=company_name, days_back=90)
    for r in tm_records:
        serial = r.get("serial_number", "")
        if serial:
            existing = (
                db.query(TrademarkFiling)
                .filter(TrademarkFiling.serial_number == serial)
                .first()
            )
            if not existing:
                db.add(TrademarkFiling(
                    ticker=ticker,
                    serial_number=serial,
                    mark_text=r.get("mark_text"),
                    filing_date=r.get("filing_date"),
                    status=r.get("status"),
                    is_merger_signal=r.get("is_merger_signal", False),
                    signal_type=r.get("signal_type"),
                    source=r.get("source"),
                ))
    db.commit()

    # Executive departures (8-K Item 5.02)
    dep_collector = ExecDepartureCollector()
    dep_records = dep_collector.collect(ticker, days_back=90)
    for r in dep_records:
        if r.get("accession_number"):
            existing = (
                db.query(ExecutiveDeparture)
                .filter(ExecutiveDeparture.accession_number == r["accession_number"])
                .first()
            )
            if not existing:
                db.add(ExecutiveDeparture(
                    ticker=ticker,
                    cik=r.get("cik"),
                    accession_number=r["accession_number"],
                    filed_at=r["filed_at"],
                    officer_name=r.get("officer_name"),
                    title=r.get("title"),
                    departure_type=r.get("departure_type"),
                    officer_weight=r.get("officer_weight", 0.2),
                    is_senior=r.get("is_senior", False),
                ))
    db.commit()

    # Employee sentiment (Indeed reviews)
    sentiment_collector = EmployeeSentimentCollector()
    sentiment_records = sentiment_collector.collect(ticker, company_name=company_name)
    for r in sentiment_records:
        existing = (
            db.query(EmployeeSentimentSnapshot)
            .filter(
                EmployeeSentimentSnapshot.ticker == r["ticker"],
                EmployeeSentimentSnapshot.snapshot_date == r["snapshot_date"],
                EmployeeSentimentSnapshot.source == r.get("source", "indeed"),
            )
            .first()
        )
        if not existing:
            db.add(EmployeeSentimentSnapshot(
                ticker=ticker,
                snapshot_date=r["snapshot_date"],
                source=r.get("source", "indeed"),
                review_count=r.get("review_count", 0),
                ma_mention_count=r.get("ma_mention_count", 0),
                uncertainty_mention_count=r.get("uncertainty_mention_count", 0),
                avg_sentiment=r.get("avg_sentiment"),
                avg_rating=r.get("avg_rating"),
                ma_mention_rate=r.get("ma_mention_rate"),
                sentiment_drop_flag=r.get("sentiment_drop_flag", False),
            ))
    db.commit()


@app.command()
def main(
    ticker: str | None = typer.Option(None, help="Single ticker to collect (default: all)"),
    daemon: bool = typer.Option(False, help="Run continuously on schedule"),
    interval_minutes: int = typer.Option(240, help="Collection interval in daemon mode"),
):
    console.print("[bold cyan]HARBINGER[/bold cyan] — Signal Collector")

    init_db()

    def run_once():
        db = SessionLocal()
        try:
            if ticker:
                companies = db.query(Company).filter(Company.ticker == ticker.upper()).all()
            else:
                companies = db.query(Company).all()

            console.print(f"Collecting signals for {len(companies)} companies...")

            for company in track(companies, description="Collecting..."):
                jet_codes = (
                    company.jet_icao24.split(",")
                    if company.jet_icao24
                    else []
                )
                try:
                    collect_for_ticker(db, company.ticker, company.name, jet_codes)
                except Exception as e:
                    logger.error("collection_error", ticker=company.ticker, error=str(e))

            console.print(f"[green]✓ Collection complete[/green]")
        finally:
            db.close()

    if daemon:
        console.print(f"Running in daemon mode (every {interval_minutes} minutes)")
        while True:
            run_once()
            console.print(f"Sleeping {interval_minutes}m until next collection...")
            time.sleep(interval_minutes * 60)
    else:
        run_once()


if __name__ == "__main__":
    app()
