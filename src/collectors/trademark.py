"""
USPTO Trademark application signal collector.

Free API: https://developer.uspto.gov/api-catalog/trademark-search-api

Key insight: Companies register new trademarks for the COMBINED entity before
announcing a merger. This includes:
  - New holding company names ("NewHoldco Corp")
  - Combined product/brand names
  - Logo marks that blend both companies' visual identities
  - Domain-like wordmarks for the merged entity

These are filed weeks to months before announcement because trademark
prosecution takes time and deals can collapse if brand registration fails.

Secondary signal: A company ABANDONING trademark applications suggests
a deal fell through (the brand work stops).

Search strategy:
  1. Search recent applications by the target company as applicant
  2. Flag applications for names that don't match existing products
  3. Flag cross-filing patterns (both companies file related marks same period)
  4. Check for "Holdings", "Group", "Merged", "Combined" in new applications
"""

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector

USPTO_TM_BASE = "https://developer.uspto.gov/trademark-search/v1"
USPTO_TM_STATUS = "https://tsdrapi.uspto.gov/ts/cd/casestatus"

# Wordmarks that suggest new entity creation (post-merger naming)
MERGER_NAME_PATTERNS = re.compile(
    r"\b(holdings?|group|combined|merged|unified|global|"
    r"solutions|technologies|enterprises?|partners?)\b",
    re.IGNORECASE,
)

# Suspicious: applications filed under unusual law firm (M&A IP counsel)
MA_IP_FIRMS = frozenset(
    {
        "KIRKLAND & ELLIS",
        "SKADDEN",
        "SULLIVAN & CROMWELL",
        "WACHTELL",
        "SIMPSON THACHER",
        "LATHAM & WATKINS",
        "DAVIS POLK",
        "CLEARY GOTTLIEB",
    }
)


class TrademarkCollector(BaseCollector):
    name = "trademark"
    request_delay = 0.5

    def _search_by_owner(
        self,
        company_name: str,
        days_back: int = 90,
    ) -> list[dict]:
        """
        Search USPTO trademark applications where the company is the owner/applicant.
        Uses the USPTO Trademark Search API.
        """
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime(
            "%Y%m%d"
        )
        end_date = datetime.utcnow().strftime("%Y%m%d")

        try:
            resp = self.fetch(
                f"{USPTO_TM_BASE}/trademark",
                params={
                    "q": f'ownerName:"{company_name}"',
                    "dateRange": f"{start_date}-{end_date}",
                    "rows": 50,
                    "start": 0,
                    "fields": "wordMark,ownerName,filingDate,statusCode,serialNumber,attorneyName",
                },
            )
            data = resp.json()
        except Exception as e:
            self.log.warning("tm_search_failed", company=company_name, error=str(e))
            return []

        results = []
        for doc in data.get("docs", data.get("trademarks", [])):
            word_mark = doc.get("wordMark", doc.get("mark", ""))
            filing_date_str = doc.get("filingDate", "")
            status = doc.get("statusCode", doc.get("status", ""))
            serial = doc.get("serialNumber", "")
            attorney = (doc.get("attorneyName") or "").upper()

            try:
                filing_date = datetime.strptime(filing_date_str[:8], "%Y%m%d")
            except (ValueError, TypeError):
                filing_date = None

            # Score the mark for merger-signal characteristics
            is_merger_name_pattern = bool(
                MERGER_NAME_PATTERNS.search(word_mark)
            ) if word_mark else False
            is_ma_counsel = any(firm in attorney for firm in MA_IP_FIRMS)
            is_abandoned = "ABANDONED" in str(status).upper()

            results.append(
                {
                    "serial_number": serial,
                    "word_mark": word_mark,
                    "filing_date": filing_date,
                    "status": status,
                    "attorney": attorney,
                    "is_merger_name_pattern": is_merger_name_pattern,
                    "is_ma_counsel": is_ma_counsel,
                    "is_abandoned": is_abandoned,
                    "signal_strength": (
                        (1 if is_merger_name_pattern else 0)
                        + (1 if is_ma_counsel else 0)
                    ),
                }
            )

        return results

    def _check_domain_registrations(self, company_name: str) -> list[dict]:
        """
        Check ICANN WHOIS for recently registered domains combining company names.
        Very high precision signal: nobody registers mergedco.com by accident.

        Uses RDAP (Registration Data Access Protocol) — the modern WHOIS replacement.
        https://www.rdap.org/
        """
        # Build candidate domain patterns from company name
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", company_name).lower()
        candidates = [
            f"{clean_name}holdings.com",
            f"{clean_name}group.com",
            f"new{clean_name}.com",
        ]

        registered = []
        for domain in candidates:
            try:
                resp = self.fetch(
                    f"https://rdap.org/domain/{domain}",
                    headers={"Accept": "application/rdap+json"},
                )
                data = resp.json()
                # Domain exists if we get a 200 response
                registered_date = None
                for event in data.get("events", []):
                    if event.get("eventAction") == "registration":
                        registered_date = event.get("eventDate")
                        break

                registered.append(
                    {
                        "domain": domain,
                        "registered_date": registered_date,
                        "registrar": data.get("entities", [{}])[0]
                        .get("vcardArray", [[]])[1] if data.get("entities") else None,
                    }
                )
            except Exception:
                # 404 = domain not registered, which is the normal case
                pass

        return registered

    def collect(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 90,
        **kwargs,
    ) -> list[dict[str, Any]]:
        name = company_name or ticker
        self.log.info("collecting_trademarks", ticker=ticker, company=name)

        tm_records = self._search_by_owner(name, days_back=days_back)
        domain_records = self._check_domain_registrations(name)

        results = []

        for tm in tm_records:
            results.append(
                {
                    "ticker": ticker,
                    "source": "uspto_trademark",
                    "serial_number": tm["serial_number"],
                    "mark_text": tm["word_mark"],
                    "filing_date": tm["filing_date"],
                    "status": tm["status"],
                    "is_merger_signal": tm["signal_strength"] > 0,
                    "signal_type": "trademark_application",
                    "details": {
                        "merger_name_pattern": tm["is_merger_name_pattern"],
                        "ma_counsel": tm["is_ma_counsel"],
                        "is_abandoned": tm["is_abandoned"],
                    },
                }
            )

        for domain in domain_records:
            results.append(
                {
                    "ticker": ticker,
                    "source": "domain_registration",
                    "serial_number": domain["domain"],
                    "mark_text": domain["domain"],
                    "filing_date": None,
                    "status": "REGISTERED",
                    "is_merger_signal": True,
                    "signal_type": "domain_registration",
                    "details": {
                        "registered_date": domain["registered_date"],
                    },
                }
            )

        self.log.info(
            "trademark_collected",
            ticker=ticker,
            trademarks=len(tm_records),
            domains=len(domain_records),
            signals=sum(1 for r in results if r["is_merger_signal"]),
        )
        return results

    def summarize(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 30,
    ) -> dict:
        records = self.collect(ticker, company_name=company_name, days_back=days_back)
        return {
            "total_tm_applications_30d": len(
                [r for r in records if r["source"] == "uspto_trademark"]
            ),
            "merger_signal_applications": sum(
                1 for r in records if r["is_merger_signal"]
            ),
            "suspicious_domain_registrations": len(
                [r for r in records if r["source"] == "domain_registration"]
            ),
            "abandoned_applications": sum(
                1
                for r in records
                if r.get("details", {}).get("is_abandoned")
            ),
        }
