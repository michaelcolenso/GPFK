"""
Raw signal data routes.

GET /signals/{ticker}          - All raw signals for a ticker
GET /signals/{ticker}/sec      - SEC filing history
GET /signals/{ticker}/flights  - Corporate jet flight records
GET /signals/{ticker}/jobs     - Job posting trend
GET /signals/{ticker}/patents  - Patent assignment activity
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.models import FlightRecord, JobPostingSnapshot, PatentAssignment, SecFiling
from src.db.session import get_db

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/{ticker}")
def get_all_signals(
    ticker: str,
    days_back: int = Query(default=90, ge=7, le=365),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    sec = (
        db.query(SecFiling)
        .filter(SecFiling.ticker == ticker, SecFiling.filed_at >= cutoff)
        .order_by(SecFiling.filed_at.desc())
        .limit(50)
        .all()
    )
    flights = (
        db.query(FlightRecord)
        .filter(FlightRecord.owner_ticker == ticker, FlightRecord.first_seen >= cutoff)
        .order_by(FlightRecord.first_seen.desc())
        .limit(100)
        .all()
    )
    jobs = (
        db.query(JobPostingSnapshot)
        .filter(
            JobPostingSnapshot.ticker == ticker,
            JobPostingSnapshot.snapshot_date >= cutoff,
        )
        .order_by(JobPostingSnapshot.snapshot_date.desc())
        .all()
    )
    patents = (
        db.query(PatentAssignment)
        .filter(
            (PatentAssignment.assignor_ticker == ticker)
            | (PatentAssignment.assignee_ticker == ticker),
            PatentAssignment.assignment_date >= cutoff,
        )
        .order_by(PatentAssignment.assignment_date.desc())
        .limit(50)
        .all()
    )

    return {
        "ticker": ticker,
        "days_back": days_back,
        "sec_filings": [
            {
                "form_type": f.form_type,
                "filed_at": f.filed_at.isoformat(),
                "mentions_acquisition": f.mentions_acquisition,
                "accession_number": f.accession_number,
                "document_url": f.document_url,
            }
            for f in sec
        ],
        "flight_records": [
            {
                "icao24": r.icao24,
                "departure": r.departure_airport,
                "arrival": r.arrival_airport,
                "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                "anomaly_score": r.anomaly_score,
            }
            for r in flights
        ],
        "job_postings": [
            {
                "date": s.snapshot_date.isoformat(),
                "count": s.posting_count,
                "pct_change_30d": s.pct_change_30d,
            }
            for s in jobs
        ],
        "patent_assignments": [
            {
                "direction": "outbound"
                if p.assignor_ticker == ticker
                else "inbound",
                "counterparty": p.assignee
                if p.assignor_ticker == ticker
                else p.assignor,
                "date": p.assignment_date.isoformat(),
                "patent_count": p.patent_count,
            }
            for p in patents
        ],
        "counts": {
            "sec_filings": len(sec),
            "flight_records": len(flights),
            "job_snapshots": len(jobs),
            "patent_assignments": len(patents),
        },
    }


@router.get("/{ticker}/sec")
def get_sec_signals(
    ticker: str,
    days_back: int = Query(default=90, ge=7, le=365),
    db: Session = Depends(get_db),
) -> list[dict]:
    ticker = ticker.upper().strip()
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    filings = (
        db.query(SecFiling)
        .filter(SecFiling.ticker == ticker, SecFiling.filed_at >= cutoff)
        .order_by(SecFiling.filed_at.desc())
        .all()
    )
    return [
        {
            "form_type": f.form_type,
            "filed_at": f.filed_at.isoformat(),
            "mentions_acquisition": f.mentions_acquisition,
            "counterparty_ticker": f.counterparty_ticker,
            "document_url": f.document_url,
        }
        for f in filings
    ]
