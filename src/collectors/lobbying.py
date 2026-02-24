"""
Senate Lobbying Disclosure Act (LDA) collector.

Free REST API: https://lda.senate.gov/api/v1/
No auth required for read access.

Key insight: Companies preparing for a major acquisition must lobby regulators
on antitrust/merger issues 6-12 weeks before announcement. Hart-Scott-Rodino
filings (the mandatory pre-merger notification) require lobbying preparation.

Issue codes to watch:
  ANT - Antitrust / Competition
  MER - Mergers & Acquisitions (not an official code; watch ANT + text)
  FIN - Financial Institutions / Investments
  TAX - Tax (deal structuring)
  TRD - Trade (cross-border deals)

Signals:
  - New lobbying registrations by company on antitrust topics
  - Lobbying disclosures naming specific counterparty companies
  - Sudden increase in quarterly filing spend
  - Lobbying by *law firms* on behalf of a company (harder to see, but public)
"""

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector

LDA_BASE = "https://lda.senate.gov/api/v1"

# Issue codes that signal M&A preparation
MA_ISSUE_CODES = frozenset({"ANT", "FIN", "TRD", "TAX", "SMB"})

# Text patterns in filing descriptions that indicate deal activity
DEAL_PATTERNS = re.compile(
    r"\b(antitrust|hart.scott.rodino|HSR|merger|acquisition|"
    r"consolidation|market.concentration|competition|FTC|DOJ|"
    r"CFIUS|foreign investment|divestiture|consent decree)\b",
    re.IGNORECASE,
)


class LobbyingCollector(BaseCollector):
    name = "lobbying"
    request_delay = 0.5  # LDA API is lightly loaded; be respectful

    def _search_filings(
        self,
        company_name: str,
        days_back: int = 90,
        page_size: int = 50,
    ) -> list[dict]:
        """
        Search LDA filings where the given company is the client.
        LDA distinguishes registrant (lobbyist/firm) from client (the company paying).
        We want filings where our target company is the CLIENT.
        """
        filed_after = (datetime.utcnow() - timedelta(days=days_back)).strftime(
            "%Y-%m-%d"
        )

        results = []
        url = f"{LDA_BASE}/filings/"
        params = {
            "client_name": company_name,
            "filing_dt_posted_after": filed_after,
            "page_size": page_size,
            "ordering": "-dt_posted",
        }

        while url:
            try:
                resp = self.fetch(url, params=params if url == f"{LDA_BASE}/filings/" else {})
                data = resp.json()
            except Exception as e:
                self.log.warning("lda_fetch_failed", company=company_name, error=str(e))
                break

            for filing in data.get("results", []):
                # Extract lobbying activities and their issue codes
                activities = filing.get("lobbying_activities", [])
                issue_codes = [a.get("general_issue_code", "") for a in activities]
                descriptions = " ".join(
                    a.get("description", "") for a in activities
                )

                ma_issue_hit = bool(MA_ISSUE_CODES & set(issue_codes))
                deal_text_hit = bool(DEAL_PATTERNS.search(descriptions))

                results.append(
                    {
                        "filing_uuid": filing.get("filing_uuid", ""),
                        "filing_type": filing.get("filing_type", ""),
                        "filing_year": filing.get("filing_year"),
                        "period_display": filing.get("filing_period_display", ""),
                        "filed_at_str": filing.get("dt_posted", ""),
                        "registrant_name": filing.get("registrant", {}).get(
                            "name", ""
                        ),
                        "client_name": filing.get("client", {}).get("name", ""),
                        "income": filing.get("income"),
                        "expenses": filing.get("expenses"),
                        "issue_codes": issue_codes,
                        "descriptions": descriptions[:500],
                        "ma_issue_hit": ma_issue_hit,
                        "deal_text_hit": deal_text_hit,
                        "is_ma_signal": ma_issue_hit or deal_text_hit,
                    }
                )

            # Paginate
            url = data.get("next")
            params = {}  # pagination URL already contains params

        return results

    def collect(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 90,
        **kwargs,
    ) -> list[dict[str, Any]]:
        name = company_name or ticker
        self.log.info("collecting_lobbying", ticker=ticker, company=name)

        filings = self._search_filings(name, days_back=days_back)
        self.log.info(
            "lobbying_collected",
            ticker=ticker,
            total=len(filings),
            ma_signals=sum(1 for f in filings if f["is_ma_signal"]),
        )

        # Attach ticker to all records
        for f in filings:
            f["ticker"] = ticker

        return filings

    def summarize(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 90,
    ) -> dict:
        filings = self.collect(ticker, company_name=company_name, days_back=days_back)
        ma_filings = [f for f in filings if f["is_ma_signal"]]

        total_spend = sum(
            float(f["expenses"] or 0) + float(f["income"] or 0)
            for f in filings
        )
        ma_spend = sum(
            float(f["expenses"] or 0) + float(f["income"] or 0)
            for f in ma_filings
        )

        return {
            "total_filings_90d": len(filings),
            "ma_signal_filings_90d": len(ma_filings),
            "total_lobbying_spend": total_spend,
            "ma_lobbying_spend": ma_spend,
            "antitrust_filings_30d": sum(
                1
                for f in filings
                if f["is_ma_signal"]
                and f.get("filed_at_str", "") >= (
                    datetime.utcnow() - timedelta(days=30)
                ).strftime("%Y-%m-%d")
            ),
        }
