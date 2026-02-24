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


class LobbyingRecord(Base):
    """Senate LDA lobbying filing where a tracked company is the client."""

    __tablename__ = "lobbying_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    filing_uuid: Mapped[str] = mapped_column(String(64), unique=True)
    filing_type: Mapped[str | None] = mapped_column(String(20))
    filing_year: Mapped[int | None] = mapped_column(Integer)
    period_display: Mapped[str | None] = mapped_column(String(20))
    filed_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    registrant_name: Mapped[str | None] = mapped_column(String(255))
    client_name: Mapped[str | None] = mapped_column(String(255))
    income: Mapped[float | None] = mapped_column(Float)
    expenses: Mapped[float | None] = mapped_column(Float)
    issue_codes: Mapped[str | None] = mapped_column(Text)  # comma-separated
    descriptions: Mapped[str | None] = mapped_column(Text)
    is_ma_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OptionsSnapshot(Base):
    """Daily options market snapshot for a ticker."""

    __tablename__ = "options_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    current_price: Mapped[float | None] = mapped_column(Float)
    otm_call_volume: Mapped[int] = mapped_column(Integer, default=0)
    otm_call_oi: Mapped[int] = mapped_column(Integer, default=0)
    total_call_volume: Mapped[int] = mapped_column(Integer, default=0)
    total_put_volume: Mapped[int] = mapped_column(Integer, default=0)
    put_call_ratio: Mapped[float | None] = mapped_column(Float)
    volume_oi_ratio: Mapped[float | None] = mapped_column(Float)
    max_volume_oi_ratio: Mapped[float | None] = mapped_column(Float)
    avg_otm_iv: Mapped[float | None] = mapped_column(Float)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "snapshot_date", name="uq_options_ticker_date"),
    )


class TrademarkFiling(Base):
    """USPTO trademark application filed by a tracked company."""

    __tablename__ = "trademark_filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    serial_number: Mapped[str] = mapped_column(String(30), unique=True)
    mark_text: Mapped[str | None] = mapped_column(Text)
    filing_date: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    status: Mapped[str | None] = mapped_column(String(50))
    is_merger_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    signal_type: Mapped[str | None] = mapped_column(String(50))
    source: Mapped[str | None] = mapped_column(String(30))
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExecutiveDeparture(Base):
    """SEC 8-K Item 5.02 executive departure filing."""

    __tablename__ = "executive_departures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    cik: Mapped[str | None] = mapped_column(String(20))
    accession_number: Mapped[str] = mapped_column(String(50), unique=True)
    filed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    officer_name: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(255))
    departure_type: Mapped[str | None] = mapped_column(String(50))
    officer_weight: Mapped[float] = mapped_column(Float, default=0.2)
    is_senior: Mapped[bool] = mapped_column(Boolean, default=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EmployeeSentimentSnapshot(Base):
    """Aggregated employee review sentiment snapshot."""

    __tablename__ = "employee_sentiment_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="indeed")
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    ma_mention_count: Mapped[int] = mapped_column(Integer, default=0)
    uncertainty_mention_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_sentiment: Mapped[float | None] = mapped_column(Float)
    avg_rating: Mapped[float | None] = mapped_column(Float)
    ma_mention_rate: Mapped[float | None] = mapped_column(Float)
    sentiment_drop_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "snapshot_date", "source",
            name="uq_sentiment_ticker_date_source",
        ),
    )


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

    # Lobbying signals
    lobbying_ma_filings_30d: Mapped[int] = mapped_column(Integer, default=0)
    lobbying_ma_filings_90d: Mapped[int] = mapped_column(Integer, default=0)
    lobbying_ma_spend_30d: Mapped[float] = mapped_column(Float, default=0.0)

    # Options flow signals
    options_otm_call_volume: Mapped[int] = mapped_column(Integer, default=0)
    options_put_call_ratio: Mapped[float | None] = mapped_column(Float)
    options_volume_oi_ratio: Mapped[float | None] = mapped_column(Float)
    options_anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    options_fresh_buying_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    # Trademark signals
    trademark_filings_30d: Mapped[int] = mapped_column(Integer, default=0)
    trademark_merger_signals_30d: Mapped[int] = mapped_column(Integer, default=0)
    trademark_domain_registrations: Mapped[int] = mapped_column(Integer, default=0)

    # Executive departure signals
    exec_departures_30d: Mapped[int] = mapped_column(Integer, default=0)
    senior_exec_departures_30d: Mapped[int] = mapped_column(Integer, default=0)
    exec_departure_weighted_score: Mapped[float] = mapped_column(Float, default=0.0)
    ceo_departed_90d: Mapped[bool] = mapped_column(Boolean, default=False)
    cfo_departed_90d: Mapped[bool] = mapped_column(Boolean, default=False)

    # Employee sentiment signals
    employee_ma_mention_rate: Mapped[float] = mapped_column(Float, default=0.0)
    employee_avg_sentiment: Mapped[float | None] = mapped_column(Float)
    employee_uncertainty_count: Mapped[int] = mapped_column(Integer, default=0)
    employee_sentiment_drop_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    # Composite
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    model_probability: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "as_of_date", name="uq_signal_ticker_date"),
        Index("ix_signal_score", "composite_score"),
    )
