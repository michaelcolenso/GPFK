"""
Feature engineering layer.

Takes raw signal data from DB tables and computes a normalized
feature vector per (ticker, date) that the ML model consumes.

Feature groups:
  1. SEC filing velocity and type composition
  2. Flight anomaly scores and destination patterns
  3. Job posting trend signals
  4. Patent transfer activity
  5. Lobbying disclosure patterns
  6. Options flow unusual activity
  7. Trademark / brand registration signals
  8. Executive departure signals
  9. Employee review sentiment
  10. Cross-signal interactions

All features normalized to [0, 1] before model input.
"""

from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from src.db.models import (
    EmployeeSentimentSnapshot,
    ExecutiveDeparture,
    FlightRecord,
    JobPostingSnapshot,
    LobbyingRecord,
    OptionsSnapshot,
    PatentAssignment,
    SecFiling,
    TrademarkFiling,
)

# Feature names in canonical order (must match model training)
FEATURE_NAMES = [
    # --- SEC (6) ---
    "s4_30d",
    "sc_to_t_30d",
    "sc13d_amend_30d",
    "merger_proxy_30d",
    "counterparty_mention_30d",
    "sec_filing_velocity_ratio",
    # --- Flight (4) ---
    "flight_anomaly_score",
    "unique_destinations_30d",
    "financial_hub_pct",
    "flight_frequency_zscore",
    # --- Jobs (4) ---
    "job_pct_change_30d",
    "job_pct_change_60d",
    "job_freeze_flag",
    "job_velocity_zscore",
    # --- Patents (3) ---
    "patent_outbound_30d",
    "patent_inbound_30d",
    "patent_net_flow",
    # --- Lobbying (3) ---
    "lobbying_ma_filings_30d",
    "lobbying_ma_filings_90d",
    "lobbying_ma_spend_normalized",
    # --- Options (4) ---
    "options_anomaly_score",
    "options_pc_ratio_inverted",   # low PC ratio = bullish = high signal
    "options_volume_oi_ratio",
    "options_fresh_buying_flag",
    # --- Trademarks (3) ---
    "trademark_filings_30d",
    "trademark_merger_signals_30d",
    "trademark_domain_registrations",
    # --- Exec departures (4) ---
    "exec_departures_30d",
    "senior_exec_departures_30d",
    "exec_departure_weighted_score",
    "c_suite_departure_flag",       # CEO or CFO departed
    # --- Employee sentiment (4) ---
    "employee_ma_mention_rate",
    "employee_sentiment_drop",      # negative sentiment = high signal
    "employee_uncertainty_rate",
    "employee_sentiment_flag",
    # --- Cross-signal interactions (5) ---
    "sec_and_flight_spike",
    "sec_and_job_freeze",
    "lobbying_and_options_spike",
    "exec_departure_and_sec",
    "all_signals_elevated",
]


class FeatureBuilder:
    """
    Builds feature vectors from raw DB rows.
    One FeatureBuilder per session/request.
    """

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ #
    #  SEC                                                                 #
    # ------------------------------------------------------------------ #
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

        prior_rate = (total_90d - total_30d) / 60 if total_90d > total_30d else 0
        recent_rate = total_30d / 30
        velocity_ratio = (recent_rate / prior_rate) if prior_rate > 0 else 1.0

        return {
            "s4_30d": min(s4_30d / 3, 1.0),
            "sc_to_t_30d": min(sc_to_t_30d / 3, 1.0),
            "sc13d_amend_30d": min(sc13d_30d / 5, 1.0),
            "merger_proxy_30d": min(proxy_30d / 2, 1.0),
            "counterparty_mention_30d": min(counterparty_30d / 5, 1.0),
            "sec_filing_velocity_ratio": min(velocity_ratio / 5, 1.0),
        }

    # ------------------------------------------------------------------ #
    #  Flight                                                              #
    # ------------------------------------------------------------------ #
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

        dests_30d = [r.arrival_airport for r in records_30d if r.arrival_airport]
        fin_hub_count = sum(1 for d in dests_30d if d in FINANCIAL_HUB_AIRPORTS)
        fin_hub_pct = fin_hub_count / len(dests_30d) if dests_30d else 0

        weekly_90d = len(records_90d) / 13
        weekly_30d = len(records_30d) / 4
        freq_zscore = (
            (weekly_30d - weekly_90d) / max(weekly_90d**0.5, 0.5)
            if weekly_90d > 0
            else 0
        )

        avg_anomaly = (
            sum(r.anomaly_score or 0 for r in records_30d) / len(records_30d)
        )

        return {
            "flight_anomaly_score": min(avg_anomaly, 1.0),
            "unique_destinations_30d": min(len(set(dests_30d)) / 10, 1.0),
            "financial_hub_pct": min(fin_hub_pct, 1.0),
            "flight_frequency_zscore": float(np.clip(freq_zscore / 3, 0, 1)),
        }

    # ------------------------------------------------------------------ #
    #  Jobs                                                                #
    # ------------------------------------------------------------------ #
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

        global_std = counts.std() if counts.std() > 0 else 1
        velocity_z = (avg_recent - counts.mean()) / global_std

        return {
            "job_pct_change_30d": float(np.clip(-pct_30d / 100, 0, 1)),
            "job_pct_change_60d": float(np.clip(-pct_60d / 100, 0, 1)),
            "job_freeze_flag": 1.0 if freeze_flag else 0.0,
            "job_velocity_zscore": float(np.clip(-velocity_z / 3, 0, 1)),
        }

    # ------------------------------------------------------------------ #
    #  Patents                                                             #
    # ------------------------------------------------------------------ #
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
        net_flow = inbound - outbound

        return {
            "patent_outbound_30d": min(outbound / 10, 1.0),
            "patent_inbound_30d": min(inbound / 10, 1.0),
            "patent_net_flow": float(np.clip(-net_flow / 10, 0, 1)),
        }

    # ------------------------------------------------------------------ #
    #  Lobbying                                                            #
    # ------------------------------------------------------------------ #
    def _lobbying_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)
        cutoff_90d = as_of - timedelta(days=90)

        ma_30d = (
            self.db.query(LobbyingRecord)
            .filter(
                LobbyingRecord.ticker == ticker,
                LobbyingRecord.is_ma_signal == True,  # noqa: E712
                LobbyingRecord.filed_at >= cutoff_30d,
                LobbyingRecord.filed_at <= as_of,
            )
            .count()
        )
        ma_90d = (
            self.db.query(LobbyingRecord)
            .filter(
                LobbyingRecord.ticker == ticker,
                LobbyingRecord.is_ma_signal == True,  # noqa: E712
                LobbyingRecord.filed_at >= cutoff_90d,
                LobbyingRecord.filed_at <= as_of,
            )
            .count()
        )

        spend_rows = (
            self.db.query(LobbyingRecord.expenses, LobbyingRecord.income)
            .filter(
                LobbyingRecord.ticker == ticker,
                LobbyingRecord.is_ma_signal == True,  # noqa: E712
                LobbyingRecord.filed_at >= cutoff_30d,
                LobbyingRecord.filed_at <= as_of,
            )
            .all()
        )
        ma_spend = sum((r.expenses or 0) + (r.income or 0) for r in spend_rows)

        return {
            "lobbying_ma_filings_30d": min(ma_30d / 3, 1.0),
            "lobbying_ma_filings_90d": min(ma_90d / 8, 1.0),
            "lobbying_ma_spend_normalized": min(ma_spend / 500_000, 1.0),
        }

    # ------------------------------------------------------------------ #
    #  Options                                                             #
    # ------------------------------------------------------------------ #
    def _options_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)

        snapshots = (
            self.db.query(OptionsSnapshot)
            .filter(
                OptionsSnapshot.ticker == ticker,
                OptionsSnapshot.snapshot_date >= cutoff_30d,
                OptionsSnapshot.snapshot_date <= as_of,
            )
            .order_by(OptionsSnapshot.snapshot_date)
            .all()
        )

        if not snapshots:
            return {
                "options_anomaly_score": 0.0,
                "options_pc_ratio_inverted": 0.0,
                "options_volume_oi_ratio": 0.0,
                "options_fresh_buying_flag": 0.0,
            }

        latest = snapshots[-1]
        anomaly = latest.anomaly_score or 0.0
        pc = latest.put_call_ratio or 1.0
        pc_inverted = float(np.clip((1.5 - pc) / 1.5, 0, 1))
        voi = float(np.clip((latest.volume_oi_ratio or 0) / 2.0, 0, 1))
        fresh = 1.0 if (latest.volume_oi_ratio or 0) > 0.5 else 0.0

        return {
            "options_anomaly_score": min(anomaly, 1.0),
            "options_pc_ratio_inverted": pc_inverted,
            "options_volume_oi_ratio": voi,
            "options_fresh_buying_flag": fresh,
        }

    # ------------------------------------------------------------------ #
    #  Trademarks                                                          #
    # ------------------------------------------------------------------ #
    def _trademark_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)

        all_30d = (
            self.db.query(TrademarkFiling)
            .filter(
                TrademarkFiling.ticker == ticker,
                TrademarkFiling.filing_date >= cutoff_30d,
                TrademarkFiling.filing_date <= as_of,
            )
            .count()
        )
        merger_signals = (
            self.db.query(TrademarkFiling)
            .filter(
                TrademarkFiling.ticker == ticker,
                TrademarkFiling.is_merger_signal == True,  # noqa: E712
                TrademarkFiling.filing_date >= cutoff_30d,
                TrademarkFiling.filing_date <= as_of,
            )
            .count()
        )
        domains = (
            self.db.query(TrademarkFiling)
            .filter(
                TrademarkFiling.ticker == ticker,
                TrademarkFiling.signal_type == "domain_registration",
            )
            .count()
        )

        return {
            "trademark_filings_30d": min(all_30d / 5, 1.0),
            "trademark_merger_signals_30d": min(merger_signals / 3, 1.0),
            "trademark_domain_registrations": min(domains / 2, 1.0),
        }

    # ------------------------------------------------------------------ #
    #  Executive Departures                                                #
    # ------------------------------------------------------------------ #
    def _exec_departure_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)
        cutoff_90d = as_of - timedelta(days=90)

        all_30d = (
            self.db.query(ExecutiveDeparture)
            .filter(
                ExecutiveDeparture.ticker == ticker,
                ExecutiveDeparture.filed_at >= cutoff_30d,
                ExecutiveDeparture.filed_at <= as_of,
            )
            .all()
        )
        senior_30d = [r for r in all_30d if r.is_senior]
        weighted_score = sum(r.officer_weight for r in senior_30d)

        all_90d = (
            self.db.query(ExecutiveDeparture)
            .filter(
                ExecutiveDeparture.ticker == ticker,
                ExecutiveDeparture.filed_at >= cutoff_90d,
                ExecutiveDeparture.filed_at <= as_of,
            )
            .all()
        )
        ceo_departed = any(r.officer_weight == 1.0 for r in all_90d)
        cfo_departed = any(r.officer_weight == 0.9 for r in all_90d)

        return {
            "exec_departures_30d": min(len(all_30d) / 4, 1.0),
            "senior_exec_departures_30d": min(len(senior_30d) / 2, 1.0),
            "exec_departure_weighted_score": min(weighted_score / 3, 1.0),
            "c_suite_departure_flag": 1.0 if (ceo_departed or cfo_departed) else 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Employee Sentiment                                                  #
    # ------------------------------------------------------------------ #
    def _sentiment_features(self, ticker: str, as_of: datetime) -> dict[str, float]:
        cutoff_30d = as_of - timedelta(days=30)

        snapshots = (
            self.db.query(EmployeeSentimentSnapshot)
            .filter(
                EmployeeSentimentSnapshot.ticker == ticker,
                EmployeeSentimentSnapshot.snapshot_date >= cutoff_30d,
                EmployeeSentimentSnapshot.snapshot_date <= as_of,
            )
            .order_by(EmployeeSentimentSnapshot.snapshot_date.desc())
            .limit(1)
            .all()
        )

        if not snapshots:
            return {
                "employee_ma_mention_rate": 0.0,
                "employee_sentiment_drop": 0.0,
                "employee_uncertainty_rate": 0.0,
                "employee_sentiment_flag": 0.0,
            }

        s = snapshots[0]
        sentiment = s.avg_sentiment or 0.0
        sentiment_drop = float(np.clip(-sentiment, 0, 1))
        review_count = max(s.review_count or 1, 1)
        uncertainty_rate = min((s.uncertainty_mention_count or 0) / review_count, 1.0)

        return {
            "employee_ma_mention_rate": min(s.ma_mention_rate or 0.0, 1.0),
            "employee_sentiment_drop": sentiment_drop,
            "employee_uncertainty_rate": uncertainty_rate,
            "employee_sentiment_flag": 1.0 if s.sentiment_drop_flag else 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #
    def build(self, ticker: str, as_of: datetime | None = None) -> dict[str, float]:
        if as_of is None:
            as_of = datetime.utcnow()

        sec = self._sec_features(ticker, as_of)
        flight = self._flight_features(ticker, as_of)
        job = self._job_features(ticker, as_of)
        patent = self._patent_features(ticker, as_of)
        lobbying = self._lobbying_features(ticker, as_of)
        options = self._options_features(ticker, as_of)
        trademark = self._trademark_features(ticker, as_of)
        exec_dep = self._exec_departure_features(ticker, as_of)
        sentiment = self._sentiment_features(ticker, as_of)

        sec_elevated = sec["s4_30d"] > 0.3 or sec["sc_to_t_30d"] > 0.3
        flight_elevated = flight["flight_anomaly_score"] > 0.4
        job_frozen = job["job_freeze_flag"] > 0.5
        lobbying_elevated = lobbying["lobbying_ma_filings_30d"] > 0.3
        options_elevated = options["options_anomaly_score"] > 0.4
        exec_departed = exec_dep["c_suite_departure_flag"] > 0.5

        cross = {
            "sec_and_flight_spike": 1.0 if (sec_elevated and flight_elevated) else 0.0,
            "sec_and_job_freeze": 1.0 if (sec_elevated and job_frozen) else 0.0,
            "lobbying_and_options_spike": 1.0 if (lobbying_elevated and options_elevated) else 0.0,
            "exec_departure_and_sec": 1.0 if (exec_departed and sec_elevated) else 0.0,
            "all_signals_elevated": 1.0
            if sum([sec_elevated, flight_elevated, job_frozen, lobbying_elevated, options_elevated]) >= 3
            else 0.0,
        }

        features = {
            **sec, **flight, **job, **patent,
            **lobbying, **options, **trademark,
            **exec_dep, **sentiment, **cross,
        }

        return {k: features.get(k, 0.0) for k in FEATURE_NAMES}

    def build_vector(self, ticker: str, as_of: datetime | None = None) -> np.ndarray:
        feat_dict = self.build(ticker, as_of)
        return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
