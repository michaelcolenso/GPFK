"""
Executive departure parser — SEC Form 8-K Item 5.02.

Item 5.02 = "Departure of Directors or Certain Officers; Election of Directors;
             Appointment of Certain Officers; Compensatory Arrangements of
             Certain Officers"

This is the most underused EDGAR signal. When a TARGET company's C-suite
departs 30–90 days before an acquisition announcement, the pattern is:

1. Deal is agreed in principle, subject to regulatory approval
2. Executives' employment agreements are triggered ("change of control" clauses)
3. They agree to step down as a condition of the deal
4. 8-K Item 5.02 is filed — legally required within 4 business days

Departure language to watch (strongest to weakest signal):
  - "mutually agreed" / "mutual agreement" — almost always means forced out
  - "to pursue other opportunities" — common euphemism
  - "personal reasons" after a very short tenure — suspicious
  - "retirement" for anyone under 60 — suspicious
  - Multiple C-suite departures within 90 days — very high signal

Officers we weight heavily:
  - CEO (weight 1.0) — departure nearly always tied to deal
  - CFO (weight 0.9) — critical for deal structuring; often first to leave
  - General Counsel / CLO (weight 0.8) — owns the deal paperwork
  - COO (weight 0.7)
  - All others (weight 0.3)
"""

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector
from src.config import get_settings

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE = "https://data.sec.gov"

# Departure language patterns — sorted by signal strength
DEPARTURE_PATTERNS = {
    "mutual_agreement": re.compile(
        r"mutual(?:ly)?\s+(?:agreed?|agreement|consent)", re.IGNORECASE
    ),
    "pursue_other": re.compile(
        r"pursue\s+(?:other\s+)?(?:opportunities|interests|endeavors)", re.IGNORECASE
    ),
    "personal_reasons": re.compile(r"personal\s+reasons?", re.IGNORECASE),
    "retirement": re.compile(r"\bretir(?:ed?|ement|ing)\b", re.IGNORECASE),
    "resignation": re.compile(r"\bresign(?:ed?|ation|ing)\b", re.IGNORECASE),
    "termination": re.compile(
        r"\bterminat(?:ed?|ion|ing)\b|\bdismiss(?:ed?|al)\b", re.IGNORECASE
    ),
}

# Weighting by officer title
OFFICER_WEIGHTS = {
    "chief executive": 1.0,
    "ceo": 1.0,
    "president and ceo": 1.0,
    "chief financial": 0.9,
    "cfo": 0.9,
    "general counsel": 0.8,
    "chief legal": 0.8,
    "clo": 0.8,
    "chief operating": 0.7,
    "coo": 0.7,
    "chief technology": 0.5,
    "cto": 0.5,
    "chief revenue": 0.5,
    "chief marketing": 0.4,
    "executive vice president": 0.4,
    "evp": 0.4,
    "senior vice president": 0.3,
    "svp": 0.3,
}


def _officer_weight(title: str) -> float:
    title_lower = title.lower()
    for key, weight in OFFICER_WEIGHTS.items():
        if key in title_lower:
            return weight
    return 0.2


def _departure_type(text: str) -> str:
    """Classify departure reason from filing text."""
    for dep_type, pattern in DEPARTURE_PATTERNS.items():
        if pattern.search(text):
            return dep_type
    return "unspecified"


class ExecDepartureCollector(BaseCollector):
    name = "exec_departures"
    request_delay = 0.15

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._headers = {"User-Agent": self.settings.edgar_user_agent}

    def _get_8k_filings(self, cik: str, days_back: int = 90) -> list[dict]:
        """Fetch recent 8-K filings for a company."""
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self.fetch(url, headers=self._headers)
            data = resp.json()
        except Exception as e:
            self.log.warning("edgar_fetch_failed", cik=cik, error=str(e))
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        items = recent.get("items", [""] * len(forms))

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        results = []

        for form, date_str, accession, doc, item in zip(
            forms, dates, accessions, docs, items
        ):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                filed_at = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filed_at < cutoff:
                continue

            # Item 5.02 is the departure item
            if "5.02" not in str(item):
                continue

            accession_clean = accession.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{doc}"
            )

            results.append(
                {
                    "accession_number": accession,
                    "filed_at": filed_at,
                    "document_url": doc_url,
                    "items": item,
                }
            )

        return results

    def _parse_departure_details(self, doc_url: str) -> dict | None:
        """
        Fetch and parse an 8-K document to extract departure details.
        Returns structured departure record or None if not parseable.
        """
        try:
            resp = self.fetch(doc_url, headers=self._headers)
            text = resp.text
        except Exception as e:
            self.log.warning("doc_fetch_failed", url=doc_url, error=str(e))
            return None

        # Extract the Item 5.02 section
        item_match = re.search(
            r"Item\s+5\.02.*?(?=Item\s+\d|$)", text, re.DOTALL | re.IGNORECASE
        )
        if not item_match:
            return None

        section = item_match.group(0)[:3000]  # cap at 3000 chars

        # Extract officer name (usually in quotes or follows "Mr."/"Ms.")
        name_match = re.search(
            r'(?:Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            section,
        )
        officer_name = name_match.group(1) if name_match else "Unknown"

        # Extract title
        title_match = re.search(
            r'(?:as|,\s*)((?:Chief\s+\w+\s*Officer|CEO|CFO|COO|CTO|CLO|'
            r'President|General\s+Counsel|Executive\s+Vice\s+President)'
            r'(?:\s+and\s+\w+\s+\w+)?)',
            section,
            re.IGNORECASE,
        )
        title = title_match.group(1).strip() if title_match else "Officer"

        departure_type = _departure_type(section)
        officer_weight = _officer_weight(title)

        return {
            "officer_name": officer_name,
            "title": title,
            "departure_type": departure_type,
            "officer_weight": officer_weight,
            "is_senior": officer_weight >= 0.7,
            "text_excerpt": section[:500],
        }

    def collect(
        self,
        ticker: str,
        cik: str | None = None,
        days_back: int = 90,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if not cik:
            # Resolve CIK — reuse EdgarCollector logic
            from src.collectors.sec_edgar import EdgarCollector
            edgar = EdgarCollector()
            cik = edgar.get_cik(ticker)
            if not cik:
                return []

        self.log.info("collecting_departures", ticker=ticker, cik=cik)
        filings = self._get_8k_filings(cik, days_back=days_back)

        results = []
        for filing in filings:
            details = self._parse_departure_details(filing["document_url"])
            if not details:
                continue

            results.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "accession_number": filing["accession_number"],
                    "filed_at": filing["filed_at"],
                    "officer_name": details["officer_name"],
                    "title": details["title"],
                    "departure_type": details["departure_type"],
                    "officer_weight": details["officer_weight"],
                    "is_senior": details["is_senior"],
                    "text_excerpt": details["text_excerpt"],
                }
            )

        self.log.info(
            "departures_collected",
            ticker=ticker,
            total=len(results),
            senior=sum(1 for r in results if r["is_senior"]),
        )
        return results

    def summarize(
        self,
        ticker: str,
        cik: str | None = None,
        days_back: int = 90,
    ) -> dict:
        records = self.collect(ticker, cik=cik, days_back=days_back)

        senior_30d = [
            r for r in records
            if r["is_senior"]
            and (datetime.utcnow() - r["filed_at"]).days <= 30
        ]
        mutual_agreement = [
            r for r in records
            if r["departure_type"] == "mutual_agreement"
        ]

        # Weighted departure score: sum of officer weights in last 30 days
        weighted_score = sum(r["officer_weight"] for r in senior_30d)

        return {
            "total_departures_90d": len(records),
            "senior_departures_30d": len(senior_30d),
            "ceo_departed": any(
                r["officer_weight"] == 1.0 for r in records
            ),
            "cfo_departed": any(
                r["officer_weight"] == 0.9 for r in records
            ),
            "mutual_agreement_count": len(mutual_agreement),
            "weighted_departure_score": round(weighted_score, 2),
        }
