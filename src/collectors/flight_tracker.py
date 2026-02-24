"""
Corporate jet tracker using OpenSky Network's public REST API.

OpenSky provides free ADS-B flight data for all aircraft.
We cross-reference aircraft ICAO24 codes with FAA ownership records
to identify corporate jets and flag unusual routing patterns.

API docs: https://openskynetwork.github.io/opensky-api/rest.html
FAA registry: https://registry.faa.gov/aircraftinquiry/

Key insight: Executives fly commercial to industry events (public).
They fly private to secret deal meetings. The jet route is the signal.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector
from src.config import get_settings

OPENSKY_BASE = "https://opensky-network.org/api"
FAA_REGISTRY_BASE = "https://registry.faa.gov/aircraftinquiry/Search"

# Major financial hub airports by ICAO code
# Unusual routing = corporate jet goes to these vs. normal routes
FINANCIAL_HUB_AIRPORTS = frozenset(
    {
        "KJFK",
        "KEWR",
        "KTEB",  # NYC metro (Teterboro = private jet hub)
        "KORD",
        "KMDW",  # Chicago
        "KSFO",
        "KSJC",
        "KOAK",  # Bay Area
        "KBOS",  # Boston
        "KDCA",
        "KIAD",
        "KBWI",  # DC metro
        "KLAX",
        "KVNY",  # LA (Van Nuys = private jet)
        "EGLL",
        "EGLC",  # London
        "LFPG",
        "LFPB",  # Paris
        "EDDF",  # Frankfurt
        "LSZH",  # Zurich
        "YSSY",
        "YMML",  # Australia
    }
)

# Law firm / advisor cities where unusual visits precede deals
DEAL_CITY_AIRPORTS = frozenset(
    {
        "KTEB",  # Teterboro - most common private jet entry to NYC
        "KCDW",  # Caldwell, NJ - another NYC metro private
        "KPWK",  # Palwaukee / Chicago Executive
        "KPAO",  # Palo Alto
        "KSQL",  # San Carlos (Silicon Valley)
    }
)


class FlightTracker(BaseCollector):
    name = "flight_tracker"
    request_delay = 1.0  # OpenSky asks for polite usage

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._auth = None
        if self.settings.opensky_username and self.settings.opensky_password:
            self._auth = (
                self.settings.opensky_username,
                self.settings.opensky_password,
            )

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{OPENSKY_BASE}{path}"
        kwargs: dict = {}
        if params:
            kwargs["params"] = params
        if self._auth:
            kwargs["auth"] = self._auth
        resp = self.fetch(url, **kwargs)
        return resp.json()

    def get_flights_for_aircraft(
        self,
        icao24: str,
        days_back: int = 30,
    ) -> list[dict]:
        """
        Fetch all recorded flights for an aircraft over the past N days.
        OpenSky free tier allows up to 30 days of historical data.
        """
        end_ts = int(datetime.utcnow().timestamp())
        begin_ts = int((datetime.utcnow() - timedelta(days=min(days_back, 30))).timestamp())

        try:
            data = self._get(
                "/flights/aircraft",
                params={
                    "icao24": icao24.lower(),
                    "begin": begin_ts,
                    "end": end_ts,
                },
            )
        except Exception as e:
            self.log.warning(
                "opensky_fetch_failed", icao24=icao24, error=str(e)
            )
            return []

        if not isinstance(data, list):
            return []

        flights = []
        for f in data:
            first_seen = f.get("firstSeen")
            last_seen = f.get("lastSeen")
            flights.append(
                {
                    "icao24": icao24,
                    "callsign": (f.get("callsign") or "").strip(),
                    "departure_airport": f.get("estDepartureAirport"),
                    "arrival_airport": f.get("estArrivalAirport"),
                    "first_seen": datetime.utcfromtimestamp(first_seen)
                    if first_seen
                    else None,
                    "last_seen": datetime.utcfromtimestamp(last_seen)
                    if last_seen
                    else None,
                }
            )

        return flights

    def score_flight_anomaly(
        self,
        flights: list[dict],
        baseline_airports: set[str] | None = None,
    ) -> float:
        """
        Score how anomalous recent flight activity is for an aircraft.

        Signals that elevate the score:
        - Flying to financial hubs not previously visited
        - Unusual frequency increase
        - Multiple short-turnaround trips (day trips to unknown cities)
        - Trips to known deal-city private airports
        """
        if not flights:
            return 0.0

        score = 0.0
        destinations = [f["arrival_airport"] for f in flights if f.get("arrival_airport")]

        # Financial hub visits
        fin_hub_visits = sum(
            1 for d in destinations if d in FINANCIAL_HUB_AIRPORTS
        )
        score += min(fin_hub_visits * 0.15, 0.45)

        # Deal city private airport visits (very high signal)
        deal_city_visits = sum(
            1 for d in destinations if d in DEAL_CITY_AIRPORTS
        )
        score += min(deal_city_visits * 0.25, 0.50)

        # Flights to airports outside baseline
        if baseline_airports:
            novel_destinations = set(destinations) - baseline_airports
            score += min(len(novel_destinations) * 0.10, 0.30)

        # High frequency: more than 2 flights/week is unusual for exec travel
        weeks = max(
            (
                (flights[-1]["last_seen"] - flights[0]["first_seen"]).days / 7
                if flights[-1].get("last_seen") and flights[0].get("first_seen")
                else 1
            ),
            1,
        )
        flight_rate = len(flights) / weeks
        if flight_rate > 2:
            score += min((flight_rate - 2) * 0.05, 0.20)

        return min(score, 1.0)

    def collect(
        self,
        ticker: str,
        icao24_codes: list[str] | None = None,
        days_back: int = 30,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Collect flight records for all known jets associated with a ticker.
        `icao24_codes` should come from Company.jet_icao24 in the DB.
        """
        if not icao24_codes:
            self.log.info("no_jets_registered", ticker=ticker)
            return []

        all_records = []

        for icao24 in icao24_codes:
            flights = self.get_flights_for_aircraft(icao24, days_back=days_back)
            anomaly_score = self.score_flight_anomaly(flights)

            for flight in flights:
                all_records.append(
                    {
                        "icao24": icao24,
                        "owner_ticker": ticker,
                        "callsign": flight.get("callsign"),
                        "departure_airport": flight.get("departure_airport"),
                        "arrival_airport": flight.get("arrival_airport"),
                        "first_seen": flight.get("first_seen"),
                        "last_seen": flight.get("last_seen"),
                        "anomaly_score": anomaly_score,
                    }
                )

        return all_records

    def summarize(
        self,
        ticker: str,
        icao24_codes: list[str],
        days_back: int = 30,
    ) -> dict:
        """Quick summary dict for signal scoring."""
        records = self.collect(ticker, icao24_codes=icao24_codes, days_back=days_back)
        if not records:
            return {
                "flight_count": 0,
                "anomaly_score": 0.0,
                "unique_destinations": 0,
                "financial_hub_visits": 0,
                "deal_city_visits": 0,
            }

        destinations = [
            r["arrival_airport"] for r in records if r.get("arrival_airport")
        ]
        return {
            "flight_count": len(records),
            "anomaly_score": records[0]["anomaly_score"],  # same for all in batch
            "unique_destinations": len(set(destinations)),
            "financial_hub_visits": sum(
                1 for d in destinations if d in FINANCIAL_HUB_AIRPORTS
            ),
            "deal_city_visits": sum(
                1 for d in destinations if d in DEAL_CITY_AIRPORTS
            ),
        }
