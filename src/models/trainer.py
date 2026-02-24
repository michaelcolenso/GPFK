"""
Model trainer.

Trains an XGBoost classifier on historical data.
Label: did this company receive an M&A offer within 60 days of this date?

Training data construction:
  - Positive examples: (ticker, date) pairs where date is 60–1 days before a known deal
  - Negative examples: (ticker, random_date) pairs where no deal followed in 60 days

Heavy class imbalance: deals are rare. We handle this with:
  - scale_pos_weight parameter in XGBoost
  - SMOTE oversampling for minority class
  - Careful threshold selection on validation set

Temporal split: NEVER use future data to predict past. Train on deals before
a cutoff date, validate on deals after. This prevents data leakage.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

from src.config import get_settings
from src.db.models import KnownDeal, SignalVector
from src.signals.features import FEATURE_NAMES

logger = structlog.get_logger()


class ModelTrainer:
    """
    Builds training dataset from DB and trains XGBoost M&A predictor.
    """

    def __init__(self, db):
        self.db = db
        self.settings = get_settings()
        self.model_path = Path(self.settings.model_cache_dir) / "xgb_ma_predictor.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def build_training_data(
        self,
        label_horizon_days: int = 60,
        negative_sample_ratio: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Construct (X, y, tickers) training arrays.

        Positive examples: signal vectors from 1–60 days before a deal announcement.
        Negative examples: signal vectors from companies with no deal in +60 days.

        Returns:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,) binary
            tickers: ticker for each sample (for temporal grouping)
        """
        deals = self.db.query(KnownDeal).all()
        logger.info("training_deals_found", count=len(deals))

        all_vectors: list[SignalVector] = self.db.query(SignalVector).all()
        if not all_vectors:
            raise ValueError(
                "No signal vectors in DB. Run scripts/ingest_historical.py first."
            )

        deal_index: dict[str, list[datetime]] = {}
        for d in deals:
            deal_index.setdefault(d.target_ticker, []).append(d.announced_at)

        X_rows = []
        y_labels = []
        tickers = []

        for sv in all_vectors:
            is_positive = False
            if sv.ticker in deal_index:
                for deal_date in deal_index[sv.ticker]:
                    days_to_deal = (deal_date - sv.as_of_date).days
                    if 1 <= days_to_deal <= label_horizon_days:
                        is_positive = True
                        break

            # Build feature vector from stored values
            features = [
                sv.s4_filings_30d / 3,
                sv.sc_to_t_filings_30d / 3,
                sv.sc13d_amendments_30d / 5,
                sv.unusual_8k_count_30d / 5,
                0.0,  # counterparty_mention_30d (stored separately)
                0.0,  # sec_filing_velocity_ratio
                sv.flight_anomaly_score_30d,
                sv.unique_destinations_30d / 10,
                0.0,  # financial_hub_pct
                0.0,  # flight_frequency_zscore
                (sv.job_posting_pct_change_30d or 0) / -100,
                (sv.job_posting_pct_change_60d or 0) / -100,
                1.0 if sv.job_freeze_flag else 0.0,
                0.0,  # job_velocity_zscore
                sv.patent_assignments_outbound_30d / 10,
                sv.patent_assignments_inbound_30d / 10,
                (sv.patent_assignments_inbound_30d - sv.patent_assignments_outbound_30d) / 10,
                0.0,  # sec_and_flight_spike
                0.0,  # sec_and_job_freeze
                0.0,  # all_signals_elevated
            ]
            features = np.clip(features, 0, 1).tolist()

            X_rows.append(features)
            y_labels.append(1 if is_positive else 0)
            tickers.append(sv.ticker)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int32)

        logger.info(
            "dataset_built",
            total=len(y),
            positives=int(y.sum()),
            negatives=int((1 - y).sum()),
        )
        return X, y, tickers

    def train(
        self,
        label_horizon_days: int = 60,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Train XGBoost model with time-series cross-validation.
        Returns dict of evaluation metrics.
        """
        if not XGB_AVAILABLE:
            raise ImportError(
                "xgboost not installed. Run: pip install xgboost"
            )

        X, y, tickers = self.build_training_data(
            label_horizon_days=label_horizon_days
        )

        n_pos = y.sum()
        n_neg = (1 - y).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )

        # Time-series split: train on past, validate on future
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        logger.info("training_start", n_samples=len(y), n_features=X.shape[1])

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if y_val.sum() == 0:
                logger.warning("no_positives_in_val_fold", fold=fold)
                continue

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            probs = model.predict_proba(X_val)[:, 1]
            auc_pr = average_precision_score(y_val, probs)
            auc_roc = roc_auc_score(y_val, probs)
            cv_scores.append({"fold": fold, "auc_pr": auc_pr, "auc_roc": auc_roc})
            logger.info("fold_complete", fold=fold, auc_pr=auc_pr, auc_roc=auc_roc)

        # Final fit on all data
        model.fit(X, y, verbose=False)

        # Find optimal decision threshold on last val set
        if len(tscv.split(X)) > 0:
            _, val_idx = list(tscv.split(X))[-1]
            probs = model.predict_proba(X[val_idx])[:, 1]
            precision, recall, thresholds = precision_recall_curve(y[val_idx], probs)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_threshold = float(thresholds[np.argmax(f1_scores)])
        else:
            best_threshold = 0.5

        # Save model and metadata
        payload = {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "best_threshold": best_threshold,
            "label_horizon_days": label_horizon_days,
            "trained_at": datetime.utcnow().isoformat(),
            "cv_scores": cv_scores,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(
            "model_saved",
            path=str(self.model_path),
            best_threshold=best_threshold,
        )

        metrics = {
            "cv_mean_auc_pr": np.mean([s["auc_pr"] for s in cv_scores]) if cv_scores else None,
            "cv_mean_auc_roc": np.mean([s["auc_roc"] for s in cv_scores]) if cv_scores else None,
            "best_threshold": best_threshold,
            "n_samples": len(y),
            "n_positives": int(n_pos),
        }
        return metrics

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances from saved model."""
        payload = self._load()
        model = payload["model"]
        scores = model.feature_importances_
        return dict(zip(FEATURE_NAMES, [float(s) for s in scores]))

    def _load(self) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                "Run: python scripts/train.py"
            )
        with open(self.model_path, "rb") as f:
            return pickle.load(f)
