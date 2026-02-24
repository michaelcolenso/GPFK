"""
USPTO patent assignment signal collector.

Patent bulk-transfer between companies (assignor → assignee) often precedes
full acquisitions. IP is typically moved first as part of deal structuring.

Cross-licensing agreements are even stronger: companies that are about to
compete don't cross-license. Companies that are about to merge do.

USPTO Public Patent Data API:
  https://developer.uspto.gov/api-catalog/bulk-search-and-download

Patent Assignment search API (no auth required):
  https://developer.uspto.gov/api-catalog/bulk-search-and-download#tab-endpoint3
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from src.collectors.base import BaseCollector

USPTO_ASSIGNMENT_API = "https://developer.uspto.gov/assignments/api/search.json"


class PatentSignalCollector(BaseCollector):
    name = "patent_signals"
    request_delay = 0.5

    def _search_assignments(
        self,
        company_name: str,
        role: str = "assignor",  # or "assignee"
        days_back: int = 90,
    ) -> list[dict]:
        """
        Search USPTO patent assignments for a company as assignor or assignee.
        """
        end_date = datetime.utcnow().strftime("%Y%m%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")

        params = {
            f"{role}Name": company_name,
            "dateRange": f"{start_date}-{end_date}",
            "rows": 50,
            "start": 0,
        }

        try:
            resp = self.fetch(USPTO_ASSIGNMENT_API, params=params)
            data = resp.json()
        except Exception as e:
            self.log.warning(
                "uspto_fetch_failed",
                company=company_name,
                role=role,
                error=str(e),
            )
            return []

        assignments = data.get("docs", [])
        results = []

        for a in assignments:
            assignment_date_str = a.get("assignmentRecordedDate", "")
            try:
                assignment_date = datetime.strptime(assignment_date_str, "%Y-%m-%d")
            except ValueError:
                continue

            results.append(
                {
                    "assignor": a.get("assignorName", ""),
                    "assignee": a.get("assigneeName", ""),
                    "assignment_date": assignment_date,
                    "reel_frame": a.get("reelNo", "") + "/" + a.get("frameNo", ""),
                    "patent_count": len(a.get("patents", [])),
                }
            )

        return results

    def collect(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 90,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Collect outbound (assignor) and inbound (assignee) patent assignments.
        Both directions are interesting:
        - Outbound: company shedding IP → acquisition of remainder
        - Inbound: company accumulating IP from target as pre-deal transfer
        """
        if not company_name:
            company_name = ticker

        outbound = self._search_assignments(company_name, role="assignor", days_back=days_back)
        inbound = self._search_assignments(company_name, role="assignee", days_back=days_back)

        records = []

        for a in outbound:
            records.append(
                {
                    "assignor": a["assignor"],
                    "assignee": a["assignee"],
                    "assignor_ticker": ticker,
                    "assignee_ticker": None,  # would need CIK lookup
                    "assignment_date": a["assignment_date"],
                    "patent_count": a["patent_count"],
                    "reel_frame": a["reel_frame"],
                    "direction": "outbound",
                }
            )

        for a in inbound:
            records.append(
                {
                    "assignor": a["assignor"],
                    "assignee": a["assignee"],
                    "assignor_ticker": None,
                    "assignee_ticker": ticker,
                    "assignment_date": a["assignment_date"],
                    "patent_count": a["patent_count"],
                    "reel_frame": a["reel_frame"],
                    "direction": "inbound",
                }
            )

        self.log.info(
            "patent_collected",
            ticker=ticker,
            outbound=len(outbound),
            inbound=len(inbound),
        )
        return records

    def summarize(
        self,
        ticker: str,
        company_name: str | None = None,
        days_back: int = 30,
    ) -> dict:
        records = self.collect(ticker, company_name=company_name, days_back=days_back)
        return {
            "outbound_count": sum(1 for r in records if r["direction"] == "outbound"),
            "inbound_count": sum(1 for r in records if r["direction"] == "inbound"),
            "total_patents_transferred": sum(r["patent_count"] for r in records),
        }
