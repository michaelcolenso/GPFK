"""
Unit tests for signal scoring logic.
"""

import pytest
import numpy as np

from src.signals.scoring import SIGNAL_WEIGHTS


def test_weights_sum_to_one():
    total = sum(SIGNAL_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


def test_all_features_have_weights():
    from src.signals.features import FEATURE_NAMES
    for feat in FEATURE_NAMES:
        assert feat in SIGNAL_WEIGHTS, f"Feature {feat!r} has no weight"


def test_weights_non_negative():
    for feat, weight in SIGNAL_WEIGHTS.items():
        assert weight >= 0, f"Feature {feat!r} has negative weight {weight}"


def test_composite_score_zero_for_zero_features():
    """A company with no signals should score zero."""
    zero_features = {k: 0.0 for k in SIGNAL_WEIGHTS}
    composite = sum(
        zero_features[feat] * weight for feat, weight in SIGNAL_WEIGHTS.items()
    )
    assert composite == 0.0


def test_composite_score_one_for_all_ones():
    """A company with all features at max should score 1.0."""
    max_features = {k: 1.0 for k in SIGNAL_WEIGHTS}
    composite = sum(
        max_features[feat] * weight for feat, weight in SIGNAL_WEIGHTS.items()
    )
    assert abs(composite - 1.0) < 1e-6


def test_high_weight_features_drive_score():
    """S-4 filings (weight 0.18) should dominate when other signals are quiet."""
    features = {k: 0.0 for k in SIGNAL_WEIGHTS}
    features["s4_30d"] = 1.0  # only S-4 is elevated

    composite = sum(
        features[feat] * weight for feat, weight in SIGNAL_WEIGHTS.items()
    )
    # Should equal exactly the S-4 weight
    assert abs(composite - SIGNAL_WEIGHTS["s4_30d"]) < 1e-9
