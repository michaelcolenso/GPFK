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
from src.db.models import SecFiling, FlightRecord, JobPostingSnapshot, PatentAssignment

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
