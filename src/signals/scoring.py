"""
Composite signal scoring.

Two layers:
  1. Rule-based composite score (0–1): fast, interpretable, no training data needed.
     Used for watchlist ranking and alerting when no model is available.

  2. Model probability: XGBoost output P(deal within 60 days | features).
     Higher accuracy, requires training.

The composite score is a weighted sum of normalized feature groups.
Weights reflect empirical signal quality from M&A research literature
and have been expanded to cover all 9 signal categories.
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.signals.features import FEATURE_NAMES, FeatureBuilder

# Weights for composite score (must sum to 1.0)
# Ordered to match FEATURE_NAMES exactly.
SIGNAL_WEIGHTS = {
    # --- SEC (total: 0.25) --- most direct legal paper trail
    "s4_30d":                       0.09,
    "sc_to_t_30d":                  0.08,
    "sc13d_amend_30d":              0.03,
    "merger_proxy_30d":             0.03,
    "counterparty_mention_30d":     0.01,
    "sec_filing_velocity_ratio":    0.01,
    # --- Flight (total: 0.08) --- behavioral, hard to fake
    "flight_anomaly_score":         0.04,
    "unique_destinations_30d":      0.01,
    "financial_hub_pct":            0.02,
    "flight_frequency_zscore":      0.01,
    # --- Jobs (total: 0.08) --- operational consequence of deal
    "job_pct_change_30d":           0.03,
    "job_pct_change_60d":           0.01,
    "job_freeze_flag":              0.03,
    "job_velocity_zscore":          0.01,
    # --- Patents (total: 0.04) --- lower signal-to-noise but unique
    "patent_outbound_30d":          0.02,
    "patent_inbound_30d":           0.01,
    "patent_net_flow":              0.01,
    # --- Lobbying (total: 0.10) --- long lead time, high precision
    "lobbying_ma_filings_30d":      0.04,
    "lobbying_ma_filings_90d":      0.04,
    "lobbying_ma_spend_normalized": 0.02,
    # --- Options (total: 0.12) --- strongest raw predictive signal
    "options_anomaly_score":        0.05,
    "options_pc_ratio_inverted":    0.03,
    "options_volume_oi_ratio":      0.02,
    "options_fresh_buying_flag":    0.02,
    # --- Trademarks (total: 0.05) --- very high precision, low recall
    "trademark_filings_30d":        0.01,
    "trademark_merger_signals_30d": 0.03,
    "trademark_domain_registrations": 0.01,
    # --- Exec departures (total: 0.10) --- strong when it fires
    "exec_departures_30d":          0.01,
    "senior_exec_departures_30d":   0.02,
    "exec_departure_weighted_score": 0.03,
    "c_suite_departure_flag":       0.04,
    # --- Employee sentiment (total: 0.05) --- noisy but additive
    "employee_ma_mention_rate":     0.02,
    "employee_sentiment_drop":      0.01,
    "employee_uncertainty_rate":    0.01,
    "employee_sentiment_flag":      0.01,
    # --- Cross-signal interactions (total: 0.13) ---
    "sec_and_flight_spike":         0.02,
    "sec_and_job_freeze":           0.02,
    "lobbying_and_options_spike":   0.03,
    "exec_departure_and_sec":       0.03,
    "all_signals_elevated":         0.03,
}

_total = sum(SIGNAL_WEIGHTS.values())
assert abs(_total - 1.0) < 1e-6, f"Weights sum to {_total:.6f}, expected 1.0"

# Group boundaries for breakdown reporting
_GROUPS = {
    "sec": [
        "s4_30d", "sc_to_t_30d", "sc13d_amend_30d", "merger_proxy_30d",
        "counterparty_mention_30d", "sec_filing_velocity_ratio",
    ],
    "flight": [
        "flight_anomaly_score", "unique_destinations_30d",
        "financial_hub_pct", "flight_frequency_zscore",
    ],
    "jobs": [
        "job_pct_change_30d", "job_pct_change_60d",
        "job_freeze_flag", "job_velocity_zscore",
    ],
    "patents": [
        "patent_outbound_30d", "patent_inbound_30d", "patent_net_flow",
    ],
    "lobbying": [
        "lobbying_ma_filings_30d", "lobbying_ma_filings_90d",
        "lobbying_ma_spend_normalized",
    ],
    "options": [
        "options_anomaly_score", "options_pc_ratio_inverted",
        "options_volume_oi_ratio", "options_fresh_buying_flag",
    ],
    "trademarks": [
        "trademark_filings_30d", "trademark_merger_signals_30d",
        "trademark_domain_registrations",
    ],
    "exec_departures": [
        "exec_departures_30d", "senior_exec_departures_30d",
        "exec_departure_weighted_score", "c_suite_departure_flag",
    ],
    "employee_sentiment": [
        "employee_ma_mention_rate", "employee_sentiment_drop",
        "employee_uncertainty_rate", "employee_sentiment_flag",
    ],
}


class SignalScorer:
    """Computes composite and model-based scores for a given ticker."""

    def __init__(self, db, model=None):
        self.db = db
        self.model = model
        self.builder = FeatureBuilder(db)

    def score(self, ticker: str, as_of: datetime | None = None) -> dict[str, Any]:
        if as_of is None:
            as_of = datetime.utcnow()

        features = self.builder.build(ticker, as_of)

        composite = sum(
            features[feat] * weight for feat, weight in SIGNAL_WEIGHTS.items()
        )
        composite = float(np.clip(composite, 0, 1))

        breakdown = {
            group: round(
                sum(features[f] * SIGNAL_WEIGHTS[f] for f in feats), 4
            )
            for group, feats in _GROUPS.items()
        }

        model_prob = None
        if self.model is not None:
            vector = np.array(
                [features[k] for k in FEATURE_NAMES], dtype=np.float32
            ).reshape(1, -1)
            model_prob = float(self.model.predict_proba(vector)[0])

        score_for_alert = model_prob if model_prob is not None else composite
        if score_for_alert >= 0.75:
            alert_level = "high"
        elif score_for_alert >= 0.50:
            alert_level = "elevated"
        elif score_for_alert >= 0.25:
            alert_level = "watch"
        else:
            alert_level = "none"

        return {
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "features": features,
            "composite_score": round(composite, 4),
            "model_probability": round(model_prob, 4) if model_prob is not None else None,
            "signal_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "alert_level": alert_level,
            "primary_driver": max(breakdown, key=breakdown.get),
        }

    def watchlist(
        self,
        tickers: list[str],
        top_n: int = 20,
        as_of: datetime | None = None,
    ) -> list[dict]:
        scores = []
        for ticker in tickers:
            try:
                result = self.score(ticker, as_of)
                scores.append(result)
            except Exception:
                pass

        key = (
            "model_probability"
            if any(s["model_probability"] is not None for s in scores)
            else "composite_score"
        )
        scores.sort(
            key=lambda s: s[key] if s[key] is not None else 0,
            reverse=True,
        )
        return scores[:top_n]
