"""
SEC EDGAR collector.

Uses EDGAR's public APIs (no auth required) to collect:
  - S-4 filings (registration statements for mergers)
  - SC TO-T (tender offer statements by third parties)
  - SC 13D / 13D/A (activist investor positions and amendments)
  - 14D-9 (target company responses to tender offers)
  - 8-K filings mentioning acquisitions

EDGAR rate limit: 10 requests/sec. We stay well under.

API docs: https://efts.sec.gov/LATEST/search-index (full-text search)
          https://data.sec.gov/submissions/CIK{cik:010d}.json (company filings)
"""

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector
from src.config import get_settings

logger = structlog.get_logger()

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# Form types that are strong pre-announcement signals
HIGH_SIGNAL_FORMS = {
    "S-4": "merger_registration",
    "S-4/A": "merger_registration_amendment",
    "SC TO-T": "tender_offer",
    "SC TO-T/A": "tender_offer_amendment",
    "SC 13D": "activist_position",
    "SC 13D/A": "activist_amendment",
    "14D-9": "target_response",
    "14D-9/A": "target_response_amendment",
    "PREM14A": "preliminary_merger_proxy",
    "DEFM14A": "definitive_merger_proxy",
}

# 8-K item codes that signal M&A activity
MA_8K_ITEMS = {
    "1.01",  # Entry into a Material Definitive Agreement
    "8.01",  # Other Events (often used for deal announcements)
}

ACQUISITION_KEYWORDS = re.compile(
    r"\b(acqui(?:re|sition|red)|merger|tender offer|going.private|"
    r"take.private|definitive agreement|combination|purchase agreement)\b",
    re.IGNORECASE,
)


class EdgarCollector(BaseCollector):
    name = "edgar"
    request_delay = 0.15  # ~7 req/sec, well under the 10/sec limit

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._headers = {"User-Agent": self.settings.edgar_user_agent}

    def get_cik(self, ticker: str) -> str | None:
        """Resolve ticker to SEC CIK via EDGAR company search."""
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K&hits.hits._source=period_of_report,file_num,period_of_report,biz_location,inc_states"
        # Better: use the EDGAR company tickers JSON
        try:
            resp = self.fetch(
                "https://www.sec.gov/files/company_tickers.json",
                headers=self._headers,
            )
            data = resp.json()
            # data is {str_index: {cik_str, ticker, title}}
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    return cik
        except Exception as e:
            self.log.warning("cik_lookup_failed", ticker=ticker, error=str(e))
        return None

    def get_recent_filings(
        self,
        cik: str,
        form_types: list[str] | None = None,
        days_back: int = 90,
    ) -> list[dict]:
        """
        Pull recent filings for a company by CIK.
        Returns raw filing metadata list.
        """
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = self.fetch(url, headers=self._headers)
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        results = []

        for form, date_str, accession, doc in zip(forms, dates, accessions, descriptions):
            try:
                filed_at = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            if filed_at < cutoff:
                continue

            if form_types and form not in form_types:
                continue

            accession_clean = accession.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{doc}"
            )

            results.append(
                {
                    "form_type": form,
                    "filed_at": filed_at,
                    "accession_number": accession,
                    "document_url": doc_url,
                    "signal_type": HIGH_SIGNAL_FORMS.get(form, "general"),
                }
            )

        return results

    def search_fulltext(
        self,
        ticker: str,
        form_types: list[str],
        days_back: int = 30,
    ) -> list[dict]:
        """
        EDGAR full-text search for filings mentioning a ticker.
        Catches counterparty references (e.g., target named in acquirer's S-4).
        """
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        forms_param = ",".join(form_types)

        url = (
            f"{EDGAR_SEARCH}?q=%22{ticker}%22"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
            f"&forms={forms_param}"
        )

        try:
            resp = self.fetch(url, headers=self._headers)
            data = resp.json()
        except Exception as e:
            self.log.warning("fulltext_search_failed", ticker=ticker, error=str(e))
            return []

        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            entity_names = src.get("entity_name", "")
            form_type = src.get("form_type", "")
            filed_at_str = src.get("file_date", "")

            try:
                filed_at = datetime.strptime(filed_at_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Only include if ticker is *not* the filer (it's the counterparty)
            filer_ticker = src.get("ticker", "").upper()
            is_counterparty_mention = filer_ticker != ticker.upper()

            results.append(
                {
                    "form_type": form_type,
                    "filed_at": filed_at,
                    "accession_number": src.get("accession_no", ""),
                    "document_url": hit.get("_id", ""),
                    "filer_name": entity_names,
                    "is_counterparty_mention": is_counterparty_mention,
                    "signal_type": HIGH_SIGNAL_FORMS.get(form_type, "general"),
                }
            )

        return results

    def collect(self, ticker: str, days_back: int = 90, **kwargs) -> list[dict[str, Any]]:
        """
        Full collection pipeline for one ticker.
        Returns list of filing records ready for DB upsert.
        """
        self.log.info("collecting_edgar", ticker=ticker, days_back=days_back)

        cik = self.get_cik(ticker)
        if not cik:
            self.log.warning("no_cik_found", ticker=ticker)
            return []

        # Direct filings by this company
        own_filings = self.get_recent_filings(
            cik,
            form_types=list(HIGH_SIGNAL_FORMS.keys()),
            days_back=days_back,
        )

        # Counterparty mentions (another company filed something naming this ticker)
        counterparty_hits = self.search_fulltext(
            ticker,
            form_types=["S-4", "SC TO-T", "PREM14A", "DEFM14A"],
            days_back=days_back,
        )

        all_records = []

        for filing in own_filings:
            all_records.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form_type": filing["form_type"],
                    "accession_number": filing["accession_number"],
                    "filed_at": filing["filed_at"],
                    "document_url": filing["document_url"],
                    "mentions_acquisition": filing["signal_type"]
                    in ("merger_registration", "tender_offer", "merger_registration_amendment"),
                    "mentions_merger": filing["signal_type"]
                    in ("merger_registration", "merger_registration_amendment"),
                    "counterparty_ticker": None,
                    "is_counterparty_mention": False,
                }
            )

        for hit in counterparty_hits:
            all_records.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form_type": hit["form_type"],
                    "accession_number": hit["accession_number"],
                    "filed_at": hit["filed_at"],
                    "document_url": hit["document_url"],
                    "mentions_acquisition": True,
                    "mentions_merger": hit["form_type"] in ("S-4", "DEFM14A"),
                    "counterparty_ticker": None,
                    "is_counterparty_mention": hit["is_counterparty_mention"],
                }
            )

        self.log.info(
            "edgar_collected",
            ticker=ticker,
            own_filings=len(own_filings),
            counterparty_hits=len(counterparty_hits),
        )
        return all_records

    def summarize(self, ticker: str, days_back: int = 30) -> dict:
        """
        Quick summary for use in signal scoring.
        Returns counts of each filing type.
        """
        records = self.collect(ticker, days_back=days_back)
        summary = {
            "s4_count": 0,
            "sc_to_t_count": 0,
            "sc13d_amendment_count": 0,
            "merger_proxy_count": 0,
            "counterparty_mention_count": 0,
            "total_high_signal_filings": 0,
        }

        for r in records:
            ft = r["form_type"]
            if ft in ("S-4", "S-4/A"):
                summary["s4_count"] += 1
            elif ft in ("SC TO-T", "SC TO-T/A"):
                summary["sc_to_t_count"] += 1
            elif ft in ("SC 13D/A",):
                summary["sc13d_amendment_count"] += 1
            elif ft in ("PREM14A", "DEFM14A"):
                summary["merger_proxy_count"] += 1
            if r.get("is_counterparty_mention"):
                summary["counterparty_mention_count"] += 1
            if ft in HIGH_SIGNAL_FORMS:
                summary["total_high_signal_filings"] += 1

        return summary
