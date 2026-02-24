#!/usr/bin/env python3
"""
Seed the database with known historical M&A deals.

Sources:
  - EDGAR S-4 filings (we pull all merger registrations 2010–present)
  - Manual seed list of high-profile deals for validation

This script is idempotent: run it multiple times safely.
Run before training: python scripts/ingest_historical.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sqlalchemy.orm import Session

from src.db.models import Company, KnownDeal
from src.db.session import init_db, SessionLocal

logger = structlog.get_logger()

# Seed data: well-known deals for initial validation
# Format: (target_ticker, acquirer_ticker, announced_date, deal_value_M, premium_pct)
SEED_DEALS = [
    # 2024
    ("ATVI", "MSFT", "2022-01-18", 68700, 45.0),  # Microsoft / Activision
    ("VMW", "AVGO", "2022-05-26", 61000, 49.0),   # Broadcom / VMware
    ("TWTR", "ELON", "2022-04-14", 44000, 38.0),  # Twitter / Musk
    ("CERN", "ORCL", "2021-12-20", 28300, 18.0),  # Oracle / Cerner
    ("MGM", "AMZN", "2021-05-26", 8450, 35.0),    # Amazon / MGM
    ("XLNX", "AMD", "2020-10-27", 35000, 24.7),   # AMD / Xilinx
    ("ARM", "NVDA", "2020-09-13", 40000, 0.0),    # Nvidia / ARM (blocked)
    ("PMCS", "MCHP", "2021-11-29", 3600, 8.0),
    ("CTXS", None, "2021-09-10", 16500, 24.0),    # Citrix LBO
    ("KSU", "CP", "2021-09-15", 31000, 23.0),     # CP / KSU
    ("DISCA", "AT", "2021-05-17", 43000, 0.0),    # AT&T / Discovery
    ("JOBS", None, "2021-06-11", 5400, 17.0),
    ("IQVIA", None, "2016-05-03", 17600, 0.0),    # IMS/Quintiles
    ("LIN", "PX", "2018-06-01", 80000, 0.0),      # Linde/Praxair
    ("RTN", "UTX", "2019-06-10", 120000, 0.0),    # Raytheon/UTC
    ("BBT", "STI", "2019-02-07", 66000, 0.0),     # BB&T/SunTrust → Truist
    # 2023
    ("ATVI", "MSFT", "2022-01-18", 68700, 45.0),  # closing 2023
    ("SAVE", "JBLU", "2022-07-28", 3800, 38.0),   # JetBlue / Spirit (blocked)
    ("AMTD", None, "2023-05-15", 2100, 0.0),
    ("SVBFG", None, "2023-03-10", 0.0, 0.0),      # SVB - regulatory
]

# Companies we want to track (build universe)
TRACKED_COMPANIES = [
    # M&A-active sectors: tech, pharma, finance, energy, industrials
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc"),
    ("AMZN", "Amazon.com Inc"),
    ("META", "Meta Platforms Inc"),
    ("AAPL", "Apple Inc"),
    ("NVDA", "NVIDIA Corporation"),
    ("AMD", "Advanced Micro Devices"),
    ("INTC", "Intel Corporation"),
    ("CRM", "Salesforce Inc"),
    ("ORCL", "Oracle Corporation"),
    ("SAP", "SAP SE"),
    ("IBM", "International Business Machines"),
    ("AVGO", "Broadcom Inc"),
    ("QCOM", "Qualcomm Inc"),
    ("TXN", "Texas Instruments"),
    # Pharma / biotech
    ("PFE", "Pfizer Inc"),
    ("MRK", "Merck & Co"),
    ("ABBV", "AbbVie Inc"),
    ("BMY", "Bristol-Myers Squibb"),
    ("LLY", "Eli Lilly and Company"),
    ("AMGN", "Amgen Inc"),
    ("GILD", "Gilead Sciences"),
    ("BIIB", "Biogen Inc"),
    ("REGN", "Regeneron Pharmaceuticals"),
    ("VRTX", "Vertex Pharmaceuticals"),
    # Finance
    ("JPM", "JPMorgan Chase"),
    ("BAC", "Bank of America"),
    ("GS", "Goldman Sachs Group"),
    ("MS", "Morgan Stanley"),
    ("WFC", "Wells Fargo"),
    ("C", "Citigroup Inc"),
    ("BLK", "BlackRock Inc"),
    # Industrials / energy
    ("HON", "Honeywell International"),
    ("GE", "General Electric"),
    ("RTX", "RTX Corporation"),
    ("LMT", "Lockheed Martin"),
    ("BA", "Boeing Company"),
    ("XOM", "Exxon Mobil"),
    ("CVX", "Chevron Corporation"),
    # Media / telecom
    ("DIS", "Walt Disney Company"),
    ("CMCSA", "Comcast Corporation"),
    ("VZ", "Verizon Communications"),
    ("T", "AT&T Inc"),
    ("NFLX", "Netflix Inc"),
    ("PARA", "Paramount Global"),
    ("WBD", "Warner Bros Discovery"),
]


def seed_companies(db: Session):
    logger.info("seeding_companies", count=len(TRACKED_COMPANIES))
    for ticker, name in TRACKED_COMPANIES:
        existing = db.query(Company).filter(Company.ticker == ticker).first()
        if not existing:
            db.add(Company(ticker=ticker, name=name))
    db.commit()
    logger.info("companies_seeded")


def seed_deals(db: Session):
    logger.info("seeding_deals", count=len(SEED_DEALS))
    for row in SEED_DEALS:
        target, acquirer, date_str, value, premium = row
        announced = datetime.strptime(date_str, "%Y-%m-%d")
        existing = (
            db.query(KnownDeal)
            .filter(
                KnownDeal.target_ticker == target,
                KnownDeal.announced_at == announced,
            )
            .first()
        )
        if not existing:
            db.add(
                KnownDeal(
                    target_ticker=target,
                    acquirer_ticker=acquirer,
                    announced_at=announced,
                    deal_value_usd=value,
                    premium_pct=premium,
                    source="seed",
                )
            )
    db.commit()
    logger.info("deals_seeded")


def main():
    logger.info("initializing_db")
    init_db()
    db = SessionLocal()
    try:
        seed_companies(db)
        seed_deals(db)
        logger.info(
            "ingestion_complete",
            companies=db.query(Company).count(),
            deals=db.query(KnownDeal).count(),
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
