"""
Prediction interface.
Loads the trained model and exposes predict_proba for use in scoring.
"""

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class Predictor:
    """
    Thin wrapper around the serialized XGBoost model.
    Thread-safe for concurrent API requests (XGBoost predict is GIL-free).
    """

    def __init__(self, model_path: str | None = None):
        settings = get_settings()
        path = Path(model_path or settings.model_cache_dir) / "xgb_ma_predictor.pkl"

        if not path.exists():
            logger.warning(
                "model_not_found",
                path=str(path),
                note="Scoring will use rule-based composite only",
            )
            self._model = None
            self._threshold = 0.5
            self._trained_at = None
            return

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self._model = payload["model"]
        self._threshold = payload.get("best_threshold", 0.5)
        self._trained_at = payload.get("trained_at")
        logger.info("model_loaded", trained_at=self._trained_at, threshold=self._threshold)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability of M&A event within label horizon for each row.
        X shape: (n_samples, n_features)
        Returns: (n_samples,) float array
        """
        if self._model is None:
            raise RuntimeError("No trained model available. Run scripts/train.py first.")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary classification at best_threshold."""
        probs = self.predict_proba(X)
        return (probs >= self._threshold).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def trained_at(self) -> str | None:
        return self._trained_at


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Cached singleton for use in FastAPI dependencies."""
    return Predictor()
