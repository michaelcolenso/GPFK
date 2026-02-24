"""
Unit tests for collectors.
These test parsing and signal logic without hitting live APIs.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.collectors.flight_tracker import FINANCIAL_HUB_AIRPORTS, FlightTracker
from src.collectors.job_postings import JobPostingCollector
from src.collectors.sec_edgar import EdgarCollector, HIGH_SIGNAL_FORMS


class TestEdgarCollector:
    def test_high_signal_forms_not_empty(self):
        assert len(HIGH_SIGNAL_FORMS) > 0
        assert "S-4" in HIGH_SIGNAL_FORMS
        assert "SC TO-T" in HIGH_SIGNAL_FORMS

    def test_collector_instantiates(self):
        collector = EdgarCollector()
        assert collector.name == "edgar"

    def test_summarize_returns_expected_keys(self):
        collector = EdgarCollector()
        # Patch collect to return empty list
        with patch.object(collector, "collect", return_value=[]):
            summary = collector.summarize("TEST")
        expected_keys = {
            "s4_count",
            "sc_to_t_count",
            "sc13d_amendment_count",
            "merger_proxy_count",
            "counterparty_mention_count",
            "total_high_signal_filings",
        }
        assert set(summary.keys()) == expected_keys

    def test_summarize_counts_correctly(self):
        collector = EdgarCollector()
        mock_records = [
            {"form_type": "S-4", "is_counterparty_mention": False},
            {"form_type": "S-4/A", "is_counterparty_mention": True},
            {"form_type": "SC TO-T", "is_counterparty_mention": False},
            {"form_type": "SC 13D/A", "is_counterparty_mention": False},
            {"form_type": "DEFM14A", "is_counterparty_mention": False},
        ]
        with patch.object(collector, "collect", return_value=mock_records):
            summary = collector.summarize("TEST")

        assert summary["s4_count"] == 2
        assert summary["sc_to_t_count"] == 1
        assert summary["sc13d_amendment_count"] == 1
        assert summary["merger_proxy_count"] == 1
        assert summary["counterparty_mention_count"] == 1


class TestFlightTracker:
    def test_financial_hubs_not_empty(self):
        assert len(FINANCIAL_HUB_AIRPORTS) > 0
        assert "KTEB" in FINANCIAL_HUB_AIRPORTS  # Teterboro

    def test_zero_score_no_flights(self):
        tracker = FlightTracker()
        score = tracker.score_flight_anomaly([])
        assert score == 0.0

    def test_high_score_deal_city_visits(self):
        tracker = FlightTracker()
        now = datetime.utcnow()
        flights = [
            {
                "departure_airport": "KLAX",
                "arrival_airport": "KTEB",  # Teterboro = deal city
                "first_seen": now - timedelta(days=3),
                "last_seen": now - timedelta(days=3),
            },
            {
                "departure_airport": "KLAX",
                "arrival_airport": "KTEB",
                "first_seen": now - timedelta(days=7),
                "last_seen": now - timedelta(days=7),
            },
            {
                "departure_airport": "KLAX",
                "arrival_airport": "KTEB",
                "first_seen": now - timedelta(days=14),
                "last_seen": now - timedelta(days=14),
            },
        ]
        score = tracker.score_flight_anomaly(flights)
        assert score > 0.5  # Multiple Teterboro visits should score high

    def test_score_capped_at_one(self):
        tracker = FlightTracker()
        now = datetime.utcnow()
        # Many high-signal flights
        flights = [
            {
                "departure_airport": "KLAX",
                "arrival_airport": "KTEB",
                "first_seen": now - timedelta(days=i),
                "last_seen": now - timedelta(days=i),
            }
            for i in range(1, 20)
        ]
        score = tracker.score_flight_anomaly(flights)
        assert score <= 1.0

    def test_collect_empty_without_jets(self):
        tracker = FlightTracker()
        records = tracker.collect("TEST", icao24_codes=[])
        assert records == []


class TestJobPostingCollector:
    def test_freeze_detection_triggered(self):
        collector = JobPostingCollector()
        # 60 days of high counts, then sudden drop
        snapshots = [
            {"posting_count": 500} for _ in range(60)
        ] + [
            {"posting_count": 50} for _ in range(7)  # 90% drop
        ]
        result = collector.compute_freeze_signal(snapshots)
        assert result["freeze_flag"] is True
        assert result["pct_change_30d"] < -30

    def test_freeze_not_triggered_stable(self):
        collector = JobPostingCollector()
        snapshots = [{"posting_count": 500} for _ in range(60)]
        result = collector.compute_freeze_signal(snapshots)
        assert result["freeze_flag"] is False

    def test_freeze_not_triggered_insufficient_data(self):
        collector = JobPostingCollector()
        snapshots = [{"posting_count": 100} for _ in range(5)]
        result = collector.compute_freeze_signal(snapshots)
        assert result["freeze_flag"] is False

    def test_severity_proportional(self):
        collector = JobPostingCollector()
        # Moderate drop
        moderate = [{"posting_count": 500}] * 60 + [{"posting_count": 300}] * 7
        # Severe drop
        severe = [{"posting_count": 500}] * 60 + [{"posting_count": 10}] * 7

        r_moderate = collector.compute_freeze_signal(moderate)
        r_severe = collector.compute_freeze_signal(severe)

        # Severe should have higher severity (if both triggered)
        if r_moderate["freeze_flag"] and r_severe["freeze_flag"]:
            assert r_severe["severity"] >= r_moderate["severity"]
