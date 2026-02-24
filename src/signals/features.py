"""
Feature engineering layer.

Takes raw signal data from DB tables and computes a normalized
feature vector per (ticker, date) that the ML model consumes.

Feature groups:
  1. SEC filing velocity and type composition
  2. Flight anomaly scores and destination patterns
  3. Job posting trend signals
  4. Patent transfer activity
  5. Cross-signal interactions (e.g., SEC spike AND flight spike)

All features are normalized to [0, 1] before model input.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import (
    FlightRecord,
    JobPostingSnapshot,
    PatentAssignment,
    SecFiling,
)

# Feature names in canonical order (must match model training)
FEATURE_NAMES = [
    # SEC
    "s4_30d",
    "sc_to_t_30d",
    "sc13d_amend_30d",
    "merger_proxy_30d",
    "counterparty_mention_30d",
    "sec_filing_velocity_ratio",  # 30d count / 90d count
    # Flight
    "flight_anomaly_score",
    "unique_destinations_30d",
    "financial_hub_pct",
    "flight_frequency_zscore",
    # Jobs
    "job_pct_change_30d",
    "job_pct_change_60d",
    "job_freeze_flag",
    "job_velocity_zscore",
    # Patents
    "patent_outbound_30d",
    "patent_inbound_30d",
    "patent_net_flow",  # inbound - outbound (negative = shedding IP)
    # Cross-signal
    "sec_and_flight_spike",  # both elevated
    "sec_and_job_freeze",    # filing spike with job freeze
    "all_signals_elevated",  # composite boolean
]


class FeatureBuilder:
    """
    Builds feature vectors from raw DB rows.
    One FeatureBuilder per session/request.
    """

    def __init__(self, db: Session):
        self.db = db

    def _sec_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)
        cutoff_90d = as_of - timedelta(days=90)

        def count_filings(form_types: list[str], since: datetime) -> int:
            return (
                self.db.query(SecFiling)
                .filter(
                    SecFiling.ticker == ticker,
                    SecFiling.form_type.in_(form_types),
                    SecFiling.filed_at >= since,
                    SecFiling.filed_at <= as_of,
                )
                .count()
            )

        s4_30d = count_filings(["S-4", "S-4/A"], cutoff_30d)
        sc_to_t_30d = count_filings(["SC TO-T", "SC TO-T/A"], cutoff_30d)
        sc13d_30d = count_filings(["SC 13D/A"], cutoff_30d)
        proxy_30d = count_filings(["PREM14A", "DEFM14A"], cutoff_30d)

        total_30d = (
            self.db.query(SecFiling)
            .filter(
                SecFiling.ticker == ticker,
                SecFiling.filed_at >= cutoff_30d,
                SecFiling.filed_at <= as_of,
            )
            .count()
        )
        total_90d = (
            self.db.query(SecFiling)
            .filter(
                SecFiling.ticker == ticker,
                SecFiling.filed_at >= cutoff_90d,
                SecFiling.filed_at <= as_of,
            )
            .count()
        )

        counterparty_30d = (
            self.db.query(SecFiling)
            .filter(
                SecFiling.ticker == ticker,
                SecFiling.filed_at >= cutoff_30d,
                SecFiling.filed_at <= as_of,
                SecFiling.counterparty_ticker.isnot(None),
            )
            .count()
        )

        # Velocity ratio: if filings accelerated in last 30d vs prior 60d
        prior_60d_rate = (total_90d - total_30d) / 60 if total_90d > total_30d else 0
        recent_30d_rate = total_30d / 30
        velocity_ratio = (recent_30d_rate / prior_60d_rate) if prior_60d_rate > 0 else 1.0

        return {
            "s4_30d": min(s4_30d / 3, 1.0),
            "sc_to_t_30d": min(sc_to_t_30d / 3, 1.0),
            "sc13d_amend_30d": min(sc13d_30d / 5, 1.0),
            "merger_proxy_30d": min(proxy_30d / 2, 1.0),
            "counterparty_mention_30d": min(counterparty_30d / 5, 1.0),
            "sec_filing_velocity_ratio": min(velocity_ratio / 5, 1.0),
        }

    def _flight_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)
        cutoff_90d = as_of - timedelta(days=90)

        records_30d = (
            self.db.query(FlightRecord)
            .filter(
                FlightRecord.owner_ticker == ticker,
                FlightRecord.first_seen >= cutoff_30d,
                FlightRecord.first_seen <= as_of,
            )
            .all()
        )

        records_90d = (
            self.db.query(FlightRecord)
            .filter(
                FlightRecord.owner_ticker == ticker,
                FlightRecord.first_seen >= cutoff_90d,
                FlightRecord.first_seen <= as_of,
            )
            .all()
        )

        if not records_30d:
            return {
                "flight_anomaly_score": 0.0,
                "unique_destinations_30d": 0.0,
                "financial_hub_pct": 0.0,
                "flight_frequency_zscore": 0.0,
            }

        from src.collectors.flight_tracker import FINANCIAL_HUB_AIRPORTS

        destinations_30d = [
            r.arrival_airport for r in records_30d if r.arrival_airport
        ]
        financial_hub_count = sum(
            1 for d in destinations_30d if d in FINANCIAL_HUB_AIRPORTS
        )
        financial_hub_pct = (
            financial_hub_count / len(destinations_30d) if destinations_30d else 0
        )

        # Flight frequency z-score relative to 90d baseline
        weekly_90d = len(records_90d) / 13  # 90 days / 7
        weekly_30d = len(records_30d) / 4
        freq_zscore = (
            (weekly_30d - weekly_90d) / max(weekly_90d**0.5, 0.5)
            if weekly_90d > 0
            else 0
        )

        avg_anomaly = (
            sum(r.anomaly_score or 0 for r in records_30d) / len(records_30d)
            if records_30d
            else 0
        )

        return {
            "flight_anomaly_score": min(avg_anomaly, 1.0),
            "unique_destinations_30d": min(len(set(destinations_30d)) / 10, 1.0),
            "financial_hub_pct": min(financial_hub_pct, 1.0),
            "flight_frequency_zscore": min(max(freq_zscore / 3, 0), 1.0),
        }

    def _job_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_90d = as_of - timedelta(days=90)
        snapshots = (
            self.db.query(JobPostingSnapshot)
            .filter(
                JobPostingSnapshot.ticker == ticker,
                JobPostingSnapshot.snapshot_date >= cutoff_90d,
                JobPostingSnapshot.snapshot_date <= as_of,
            )
            .order_by(JobPostingSnapshot.snapshot_date)
            .all()
        )

        if len(snapshots) < 7:
            return {
                "job_pct_change_30d": 0.0,
                "job_pct_change_60d": 0.0,
                "job_freeze_flag": 0.0,
                "job_velocity_zscore": 0.0,
            }

        counts = np.array([s.posting_count for s in snapshots], dtype=float)
        dates = [s.snapshot_date for s in snapshots]

        # Split into windows
        now = as_of
        recent_mask = np.array([(now - d).days <= 30 for d in dates])
        mid_mask = np.array([30 < (now - d).days <= 60 for d in dates])
        old_mask = np.array([60 < (now - d).days for d in dates])

        avg_recent = counts[recent_mask].mean() if recent_mask.any() else counts.mean()
        avg_mid = counts[mid_mask].mean() if mid_mask.any() else avg_recent
        avg_old = counts[old_mask].mean() if old_mask.any() else avg_recent

        pct_30d = (avg_recent - avg_mid) / avg_mid * 100 if avg_mid > 0 else 0
        pct_60d = (avg_recent - avg_old) / avg_old * 100 if avg_old > 0 else 0

        freeze_flag = pct_30d < -30 and avg_recent < avg_mid * 0.7

        # Z-score of recent count vs all
        global_std = counts.std() if counts.std() > 0 else 1
        velocity_z = (avg_recent - counts.mean()) / global_std

        return {
            "job_pct_change_30d": np.clip((-pct_30d) / 100, 0, 1),  # flip: drop = high signal
            "job_pct_change_60d": np.clip((-pct_60d) / 100, 0, 1),
            "job_freeze_flag": 1.0 if freeze_flag else 0.0,
            "job_velocity_zscore": np.clip(-velocity_z / 3, 0, 1),
        }

    def _patent_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)

        outbound = (
            self.db.query(PatentAssignment)
            .filter(
                PatentAssignment.assignor_ticker == ticker,
                PatentAssignment.assignment_date >= cutoff_30d,
                PatentAssignment.assignment_date <= as_of,
            )
            .count()
        )

        inbound = (
            self.db.query(PatentAssignment)
            .filter(
                PatentAssignment.assignee_ticker == ticker,
                PatentAssignment.assignment_date >= cutoff_30d,
                PatentAssignment.assignment_date <= as_of,
            )
            .count()
        )

        net_flow = inbound - outbound  # positive = receiving IP; negative = shedding

        return {
            "patent_outbound_30d": min(outbound / 10, 1.0),
            "patent_inbound_30d": min(inbound / 10, 1.0),
            "patent_net_flow": np.clip(-net_flow / 10, 0, 1),  # shedding IP = high signal
        }

    def build(self, ticker: str, as_of: datetime | None = None) -> dict[str, float]:
        """
        Build the full feature vector for a ticker as of a given date.
        Returns a dict keyed by FEATURE_NAMES.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        sec = self._sec_features(ticker, as_of)
        flight = self._flight_features(ticker, as_of)
        job = self._job_features(ticker, as_of)
        patent = self._patent_features(ticker, as_of)

        # Cross-signal interaction features
        sec_elevated = sec["s4_30d"] > 0.3 or sec["sc_to_t_30d"] > 0.3
        flight_elevated = flight["flight_anomaly_score"] > 0.4
        job_frozen = job["job_freeze_flag"] > 0.5

        features = {
            **sec,
            **flight,
            **job,
            **patent,
            "sec_and_flight_spike": 1.0 if (sec_elevated and flight_elevated) else 0.0,
            "sec_and_job_freeze": 1.0 if (sec_elevated and job_frozen) else 0.0,
            "all_signals_elevated": 1.0
            if (sec_elevated and flight_elevated and job_frozen)
            else 0.0,
        }

        # Ensure canonical order
        return {k: features.get(k, 0.0) for k in FEATURE_NAMES}

    def build_vector(self, ticker: str, as_of: datetime | None = None) -> np.ndarray:
        """Returns features as a numpy array in FEATURE_NAMES order."""
        feat_dict = self.build(ticker, as_of)
        return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
