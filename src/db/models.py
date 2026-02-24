"""
SQLAlchemy ORM models.
Every raw signal lands in a typed table. The signals layer reads from these
to build feature vectors. Keeping raw storage separate from features means
we can re-derive features without re-collecting data.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Company(Base):
    """Master company registry. Keyed by ticker for simplicity."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    cik: Mapped[str | None] = mapped_column(String(20))  # SEC CIK
    sic_code: Mapped[str | None] = mapped_column(String(10))
    # Known corporate jet ICAO24 hex codes (comma-separated)
    jet_icao24: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SecFiling(Base):
    """Raw SEC EDGAR filing records."""

    __tablename__ = "sec_filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False)
    form_type: Mapped[str] = mapped_column(String(30), nullable=False)
    accession_number: Mapped[str] = mapped_column(String(50), unique=True)
    filed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    period_of_report: Mapped[datetime | None] = mapped_column(DateTime)
    # Parsed signal fields
    mentions_acquisition: Mapped[bool] = mapped_column(Boolean, default=False)
    mentions_merger: Mapped[bool] = mapped_column(Boolean, default=False)
    counterparty_ticker: Mapped[str | None] = mapped_column(String(20))
    document_url: Mapped[str] = mapped_column(Text)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_sec_filings_ticker_form", "ticker", "form_type"),)


class FlightRecord(Base):
    """Corporate jet movement record from OpenSky Network."""

    __tablename__ = "flight_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    icao24: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    owner_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    callsign: Mapped[str | None] = mapped_column(String(20))
    departure_airport: Mapped[str | None] = mapped_column(String(10))
    arrival_airport: Mapped[str | None] = mapped_column(String(10))
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # Derived: was this an unusual destination vs. baseline?
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("icao24", "first_seen", name="uq_flight_icao_time"),
    )


class JobPostingSnapshot(Base):
    """Daily job posting count snapshot per company."""

    __tablename__ = "job_posting_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    posting_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Rolling baselines
    rolling_30d_avg: Mapped[float | None] = mapped_column(Float)
    rolling_90d_avg: Mapped[float | None] = mapped_column(Float)
    # Derived
    pct_change_30d: Mapped[float | None] = mapped_column(Float)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "snapshot_date", name="uq_jobs_ticker_date"),
    )


class PatentAssignment(Base):
    """USPTO patent assignment filings."""

    __tablename__ = "patent_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assignor: Mapped[str] = mapped_column(String(255), nullable=False)
    assignee: Mapped[str] = mapped_column(String(255), nullable=False)
    assignor_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    assignee_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    assignment_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    patent_count: Mapped[int] = mapped_column(Integer, default=1)
    reel_frame: Mapped[str | None] = mapped_column(String(30))
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnownDeal(Base):
    """
    Ground truth: confirmed M&A announcements.
    Used for training and backtesting.
    Target = company being acquired. Acquirer = buyer.
    """

    __tablename__ = "known_deals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    acquirer_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    announced_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deal_value_usd: Mapped[float | None] = mapped_column(Float)  # millions
    premium_pct: Mapped[float | None] = mapped_column(Float)
    deal_type: Mapped[str | None] = mapped_column(String(50))  # merger, tender, etc.
    completed: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str | None] = mapped_column(String(50))


class SignalVector(Base):
    """
    Computed feature vector per company per day.
    This is what the model consumes. Rebuilt nightly from raw tables.
    """

    __tablename__ = "signal_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    as_of_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # SEC filing signals
    s4_filings_30d: Mapped[int] = mapped_column(Integer, default=0)
    sc_to_t_filings_30d: Mapped[int] = mapped_column(Integer, default=0)
    sc13d_amendments_30d: Mapped[int] = mapped_column(Integer, default=0)
    unusual_8k_count_30d: Mapped[int] = mapped_column(Integer, default=0)

    # Flight signals
    flight_anomaly_score_30d: Mapped[float] = mapped_column(Float, default=0.0)
    unique_destinations_30d: Mapped[int] = mapped_column(Integer, default=0)
    flights_to_financial_hubs_30d: Mapped[int] = mapped_column(Integer, default=0)

    # Job signals
    job_posting_pct_change_30d: Mapped[float | None] = mapped_column(Float)
    job_posting_pct_change_60d: Mapped[float | None] = mapped_column(Float)
    job_freeze_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    # Patent signals
    patent_assignments_outbound_30d: Mapped[int] = mapped_column(Integer, default=0)
    patent_assignments_inbound_30d: Mapped[int] = mapped_column(Integer, default=0)

    # Composite
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    model_probability: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "as_of_date", name="uq_signal_ticker_date"),
        Index("ix_signal_score", "composite_score"),
    )
