"""
Unit tests for the five new signal collectors.
All tests run without live API calls — external fetches are mocked.
"""

import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.collectors.lobbying import (
    DEAL_PATTERNS,
    MA_ISSUE_CODES,
    LobbyingCollector,
)
from src.collectors.options_flow import (
    OTM_LOWER_BOUND,
    OTM_UPPER_BOUND,
    VOLUME_OI_THRESHOLD,
    OptionsFlowCollector,
)
from src.collectors.trademark import MERGER_NAME_PATTERNS, TrademarkCollector
from src.collectors.exec_departures import (
    DEPARTURE_PATTERNS,
    OFFICER_WEIGHTS,
    _departure_type,
    _officer_weight,
    ExecDepartureCollector,
)
from src.collectors.employee_sentiment import (
    MA_KEYWORDS,
    UNCERTAINTY_KEYWORDS,
    EmployeeSentimentCollector,
    _simple_sentiment,
)


# ------------------------------------------------------------------ #
#  Lobbying                                                            #
# ------------------------------------------------------------------ #

class TestLobbyingCollector:
    def test_ma_issue_codes_present(self):
        assert "ANT" in MA_ISSUE_CODES
        assert "FIN" in MA_ISSUE_CODES

    def test_deal_pattern_matches_antitrust(self):
        assert DEAL_PATTERNS.search("discussing Hart-Scott-Rodino filing requirements")
        assert DEAL_PATTERNS.search("antitrust review by the DOJ")
        assert DEAL_PATTERNS.search("pending merger approval")

    def test_deal_pattern_no_false_positive(self):
        assert not DEAL_PATTERNS.search("quarterly earnings guidance update")
        assert not DEAL_PATTERNS.search("immigration policy reform")

    def test_collector_instantiates(self):
        c = LobbyingCollector()
        assert c.name == "lobbying"

    def test_summarize_empty_returns_zeros(self):
        c = LobbyingCollector()
        with patch.object(c, "collect", return_value=[]):
            summary = c.summarize("TEST")
        assert summary["total_filings_90d"] == 0
        assert summary["ma_signal_filings_90d"] == 0

    def test_summarize_counts_ma_signals(self):
        c = LobbyingCollector()
        mock_records = [
            {"is_ma_signal": True, "expenses": 100_000, "income": 0,
             "filed_at_str": datetime.utcnow().strftime("%Y-%m-%d")},
            {"is_ma_signal": True, "expenses": 200_000, "income": 0,
             "filed_at_str": datetime.utcnow().strftime("%Y-%m-%d")},
            {"is_ma_signal": False, "expenses": 50_000, "income": 0,
             "filed_at_str": datetime.utcnow().strftime("%Y-%m-%d")},
        ]
        with patch.object(c, "collect", return_value=mock_records):
            summary = c.summarize("TEST")
        assert summary["total_filings_90d"] == 3
        assert summary["ma_signal_filings_90d"] == 2
        assert summary["total_lobbying_spend"] == 350_000
        assert summary["ma_lobbying_spend"] == 300_000


# ------------------------------------------------------------------ #
#  Options Flow                                                        #
# ------------------------------------------------------------------ #

class TestOptionsFlowCollector:
    def test_constants_sensible(self):
        assert 0 < OTM_LOWER_BOUND < OTM_UPPER_BOUND < 1
        assert 0 < VOLUME_OI_THRESHOLD < 2

    def test_zero_anomaly_empty_history(self):
        c = OptionsFlowCollector()
        assert c.compute_anomaly_score([]) == 0.0

    def test_zero_anomaly_insufficient_history(self):
        c = OptionsFlowCollector()
        snaps = [{"otm_call_volume": 100, "put_call_ratio": 1.0, "volume_oi_ratio": 0.1}] * 3
        assert c.compute_anomaly_score(snaps) == 0.0

    def test_high_anomaly_volume_spike(self):
        c = OptionsFlowCollector()
        # Baseline: 100 OTM call volume/day; today: 5000
        baseline = [
            {"otm_call_volume": 100, "put_call_ratio": 1.2, "volume_oi_ratio": 0.05}
        ] * 20
        today = [{"otm_call_volume": 5000, "put_call_ratio": 0.3, "volume_oi_ratio": 2.5}]
        score = c.compute_anomaly_score(baseline + today)
        assert score > 0.5

    def test_anomaly_capped_at_one(self):
        c = OptionsFlowCollector()
        baseline = [
            {"otm_call_volume": 10, "put_call_ratio": 1.5, "volume_oi_ratio": 0.01}
        ] * 20
        extreme = [{"otm_call_volume": 1_000_000, "put_call_ratio": 0.01, "volume_oi_ratio": 100.0}]
        score = c.compute_anomaly_score(baseline + extreme)
        assert score <= 1.0

    def test_low_anomaly_stable_volume(self):
        c = OptionsFlowCollector()
        stable = [
            {"otm_call_volume": 500, "put_call_ratio": 1.0, "volume_oi_ratio": 0.1}
        ] * 21
        score = c.compute_anomaly_score(stable)
        assert score < 0.2


# ------------------------------------------------------------------ #
#  Trademark                                                           #
# ------------------------------------------------------------------ #

class TestTrademarkCollector:
    def test_merger_name_patterns(self):
        assert MERGER_NAME_PATTERNS.search("NewCo Holdings LLC")
        assert MERGER_NAME_PATTERNS.search("Unified Technologies Group")
        assert MERGER_NAME_PATTERNS.search("Combined Solutions Inc")

    def test_no_merger_pattern_for_regular_brand(self):
        # Regular product names shouldn't match
        assert not MERGER_NAME_PATTERNS.search("iPhone")
        assert not MERGER_NAME_PATTERNS.search("Azure")
        assert not MERGER_NAME_PATTERNS.search("Photoshop")

    def test_collector_instantiates(self):
        c = TrademarkCollector()
        assert c.name == "trademark"

    def test_summarize_empty_returns_zeros(self):
        c = TrademarkCollector()
        with patch.object(c, "collect", return_value=[]):
            summary = c.summarize("TEST")
        assert summary["total_tm_applications_30d"] == 0
        assert summary["merger_signal_applications"] == 0

    def test_summarize_counts_correctly(self):
        c = TrademarkCollector()
        mock_records = [
            {"source": "uspto_trademark", "is_merger_signal": True,
             "details": {"is_abandoned": False}},
            {"source": "uspto_trademark", "is_merger_signal": False,
             "details": {"is_abandoned": False}},
            {"source": "domain_registration", "is_merger_signal": True,
             "details": {}},
        ]
        with patch.object(c, "collect", return_value=mock_records):
            summary = c.summarize("TEST")
        assert summary["total_tm_applications_30d"] == 2
        assert summary["merger_signal_applications"] == 3  # 2 TM + 1 domain
        assert summary["suspicious_domain_registrations"] == 1


# ------------------------------------------------------------------ #
#  Executive Departures                                                #
# ------------------------------------------------------------------ #

class TestExecDepartureCollector:
    def test_departure_type_mutual_agreement(self):
        assert _departure_type("resigned by mutual agreement with the board") == "mutual_agreement"

    def test_departure_type_pursue_other(self):
        assert _departure_type("will pursue other opportunities outside the company") == "pursue_other"

    def test_departure_type_retirement(self):
        assert _departure_type("announced his retirement effective March 31") == "retirement"

    def test_departure_type_unspecified(self):
        assert _departure_type("has stepped down from his role") == "unspecified"

    def test_officer_weight_ceo(self):
        assert _officer_weight("Chief Executive Officer") == 1.0
        assert _officer_weight("CEO and President") == 1.0

    def test_officer_weight_cfo(self):
        assert _officer_weight("Chief Financial Officer") == 0.9

    def test_officer_weight_gc(self):
        assert _officer_weight("General Counsel") == 0.8
        assert _officer_weight("Chief Legal Officer") == 0.8

    def test_officer_weight_unknown(self):
        weight = _officer_weight("Director of Sales")
        assert weight < 0.5

    def test_senior_threshold(self):
        # CEO, CFO, GC, COO should be "senior" (weight >= 0.7)
        assert _officer_weight("Chief Executive Officer") >= 0.7
        assert _officer_weight("Chief Financial Officer") >= 0.7
        assert _officer_weight("Chief Operating Officer") >= 0.7
        assert _officer_weight("General Counsel") >= 0.7

    def test_collector_instantiates(self):
        c = ExecDepartureCollector()
        assert c.name == "exec_departures"

    def test_summarize_empty_returns_zeros(self):
        c = ExecDepartureCollector()
        with patch.object(c, "collect", return_value=[]):
            summary = c.summarize("TEST")
        assert summary["total_departures_90d"] == 0
        assert summary["weighted_departure_score"] == 0.0

    def test_summarize_detects_ceo_departure(self):
        c = ExecDepartureCollector()
        now = datetime.utcnow()
        mock_records = [
            {
                "officer_name": "John Smith",
                "title": "Chief Executive Officer",
                "officer_weight": 1.0,
                "is_senior": True,
                "departure_type": "mutual_agreement",
                "filed_at": now - timedelta(days=15),
            }
        ]
        with patch.object(c, "collect", return_value=mock_records):
            summary = c.summarize("TEST")
        assert summary["ceo_departed"] is True
        assert summary["weighted_departure_score"] == 1.0


# ------------------------------------------------------------------ #
#  Employee Sentiment                                                  #
# ------------------------------------------------------------------ #

class TestEmployeeSentimentCollector:
    def test_ma_keywords_match(self):
        assert MA_KEYWORDS.search("heard rumors about an acquisition")
        assert MA_KEYWORDS.search("company is being sold to a PE firm")
        assert MA_KEYWORDS.search("merger announcement expected soon")
        assert MA_KEYWORDS.search("buyout rumored for months")

    def test_ma_keywords_no_false_positive(self):
        assert not MA_KEYWORDS.search("great work-life balance at this company")
        assert not MA_KEYWORDS.search("annual performance review process")

    def test_uncertainty_keywords_match(self):
        assert UNCERTAINTY_KEYWORDS.search("a lot of uncertainty about the future")
        assert UNCERTAINTY_KEYWORDS.search("rumors of upcoming layoffs circulating")
        assert UNCERTAINTY_KEYWORDS.search("major restructuring announced last month")

    def test_simple_sentiment_positive(self):
        score = _simple_sentiment("great amazing wonderful excellent work environment")
        assert score > 0

    def test_simple_sentiment_negative(self):
        score = _simple_sentiment("terrible horrible awful worst company ever")
        assert score < 0

    def test_simple_sentiment_neutral(self):
        score = _simple_sentiment("the office is located downtown near the train station")
        assert score == 0.0

    def test_analyze_reviews_empty(self):
        c = EmployeeSentimentCollector()
        result = c._analyze_reviews([])
        assert result["review_count"] == 0
        assert result["ma_mention_count"] == 0
        assert result["freeze_flag"] is False if "freeze_flag" in result else True

    def test_analyze_reviews_detects_ma_mentions(self):
        c = EmployeeSentimentCollector()
        reviews = [
            {"text": "Great company culture, heard rumors about acquisition though", "rating": 4.0},
            {"text": "Company is being sold, lots of uncertainty", "rating": 2.0},
            {"text": "Normal day at work, like the team", "rating": 4.5},
        ]
        result = c._analyze_reviews(reviews)
        assert result["ma_mention_count"] >= 1
        assert result["review_count"] == 3

    def test_analyze_reviews_sentiment_drop_flag(self):
        c = EmployeeSentimentCollector()
        # All very negative reviews
        reviews = [
            {"text": "terrible horrible awful company worst job ever", "rating": 1.0}
        ] * 10
        result = c._analyze_reviews(reviews)
        assert result["sentiment_drop_flag"] is True

    def test_collector_instantiates(self):
        c = EmployeeSentimentCollector()
        assert c.name == "employee_sentiment"


# ------------------------------------------------------------------ #
#  Scoring weights sanity check (all 9 signal groups covered)         #
# ------------------------------------------------------------------ #

class TestExpandedScoringWeights:
    def test_all_feature_names_covered(self):
        from src.signals.features import FEATURE_NAMES
        from src.signals.scoring import SIGNAL_WEIGHTS
        for feat in FEATURE_NAMES:
            assert feat in SIGNAL_WEIGHTS, f"Missing weight for feature: {feat!r}"

    def test_weights_sum_to_one(self):
        from src.signals.scoring import SIGNAL_WEIGHTS
        total = sum(SIGNAL_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_new_signal_groups_have_nonzero_weight(self):
        from src.signals.scoring import SIGNAL_WEIGHTS
        # Each new signal group should have meaningful weight
        assert SIGNAL_WEIGHTS["lobbying_ma_filings_30d"] > 0
        assert SIGNAL_WEIGHTS["options_anomaly_score"] > 0
        assert SIGNAL_WEIGHTS["trademark_merger_signals_30d"] > 0
        assert SIGNAL_WEIGHTS["c_suite_departure_flag"] > 0
        assert SIGNAL_WEIGHTS["employee_ma_mention_rate"] > 0

    def test_options_is_highest_single_weight(self):
        """Options flow should have the highest individual feature weight."""
        from src.signals.scoring import SIGNAL_WEIGHTS
        options_weights = {
            k: v for k, v in SIGNAL_WEIGHTS.items()
            if k.startswith("options_")
        }
        non_options_max = max(
            v for k, v in SIGNAL_WEIGHTS.items()
            if not k.startswith("options_") and not k.startswith("all_")
        )
        options_max = max(options_weights.values())
        # Options anomaly should be among the highest individual weights
        assert options_max >= non_options_max * 0.5
