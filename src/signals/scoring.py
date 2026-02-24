"""
Composite signal scoring.

Two layers:
  1. Rule-based composite score (0–1): fast, interpretable, no training data needed.
     Used for watchlist ranking and alerting when no model is available.

  2. Model probability: XGBoost output P(deal within 60 days | features).
     Higher accuracy, requires training.

The composite score is a weighted sum of normalized feature groups.
Weights reflect empirical signal quality from M&A research literature.
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.signals.features import FEATURE_NAMES, FeatureBuilder

# Weights for composite score (must sum to 1.0)
# These reflect signal quality relative to known M&A research
SIGNAL_WEIGHTS = {
    # SEC signals: most direct legal paper trail
    "s4_30d": 0.18,
    "sc_to_t_30d": 0.15,
    "sc13d_amend_30d": 0.08,
    "merger_proxy_30d": 0.12,
    "counterparty_mention_30d": 0.06,
    "sec_filing_velocity_ratio": 0.04,
    # Flight signals: behavioral, hard to fake
    "flight_anomaly_score": 0.07,
    "unique_destinations_30d": 0.02,
    "financial_hub_pct": 0.03,
    "flight_frequency_zscore": 0.02,
    # Job signals: operational consequence of deal
    "job_pct_change_30d": 0.06,
    "job_pct_change_60d": 0.03,
    "job_freeze_flag": 0.05,
    "job_velocity_zscore": 0.02,
    # Patent signals: lower signal-to-noise but unique
    "patent_outbound_30d": 0.02,
    "patent_inbound_30d": 0.01,
    "patent_net_flow": 0.01,
    # Cross-signal interactions: high precision when triggered
    "sec_and_flight_spike": 0.00,  # counted via components above
    "sec_and_job_freeze": 0.00,
    "all_signals_elevated": 0.03,  # bonus when all signals agree
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


class SignalScorer:
    """Computes composite and model-based scores for a given ticker."""

    def __init__(self, db, model=None):
        """
        db: SQLAlchemy session
        model: optional trained Predictor instance
        """
        self.db = db
        self.model = model
        self.builder = FeatureBuilder(db)

    def score(self, ticker: str, as_of: datetime | None = None) -> dict[str, Any]:
        """
        Full scoring pipeline. Returns a dict with:
          - features: raw feature dict
          - composite_score: rule-based 0–1
          - model_probability: ML 0–1 (or None if no model)
          - signal_breakdown: per-group contribution
          - alert_level: "none" | "watch" | "elevated" | "high"
        """
        if as_of is None:
            as_of = datetime.utcnow()

        features = self.builder.build(ticker, as_of)

        # Composite score
        composite = sum(
            features[feat] * weight for feat, weight in SIGNAL_WEIGHTS.items()
        )
        composite = float(np.clip(composite, 0, 1))

        # Group-level breakdown
        breakdown = {
            "sec": sum(
                features[f] * SIGNAL_WEIGHTS[f]
                for f in [
                    "s4_30d",
                    "sc_to_t_30d",
                    "sc13d_amend_30d",
                    "merger_proxy_30d",
                    "counterparty_mention_30d",
                    "sec_filing_velocity_ratio",
                ]
            ),
            "flight": sum(
                features[f] * SIGNAL_WEIGHTS[f]
                for f in [
                    "flight_anomaly_score",
                    "unique_destinations_30d",
                    "financial_hub_pct",
                    "flight_frequency_zscore",
                ]
            ),
            "jobs": sum(
                features[f] * SIGNAL_WEIGHTS[f]
                for f in [
                    "job_pct_change_30d",
                    "job_pct_change_60d",
                    "job_freeze_flag",
                    "job_velocity_zscore",
                ]
            ),
            "patents": sum(
                features[f] * SIGNAL_WEIGHTS[f]
                for f in [
                    "patent_outbound_30d",
                    "patent_inbound_30d",
                    "patent_net_flow",
                ]
            ),
        }

        # Model probability
        model_prob = None
        if self.model is not None:
            vector = np.array(
                [features[k] for k in FEATURE_NAMES], dtype=np.float32
            ).reshape(1, -1)
            model_prob = float(self.model.predict_proba(vector)[0])

        # Alert level
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
        """Score all tickers and return top N by score."""
        scores = []
        for ticker in tickers:
            try:
                result = self.score(ticker, as_of)
                scores.append(result)
            except Exception:
                pass

        key = "model_probability" if any(
            s["model_probability"] is not None for s in scores
        ) else "composite_score"

        scores.sort(
            key=lambda s: s[key] if s[key] is not None else 0,
            reverse=True,
        )
        return scores[:top_n]
