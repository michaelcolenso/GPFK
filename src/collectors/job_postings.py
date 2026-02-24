"""
Job posting freeze detector.

When a company is being acquired, hiring typically freezes 3–8 weeks
before announcement. The acquirer's HR team tells the target to stop
making offers — legal/integration planning reasons.

We track daily job posting counts per company and flag:
  - Sudden count drops > 30% from 30-day average
  - Sustained freeze (count near zero for >2 weeks)
  - Spikes in "integration" or "transition" role postings (acquirer side)

Data source: Indeed's public search results (no auth needed for counts)
             LinkedIn public job search (count visible without auth)
"""

import re
from datetime import datetime
from typing import Any

import structlog
from bs4 import BeautifulSoup

from src.collectors.base import BaseCollector

# Indeed search URL — returns job count in page meta
INDEED_SEARCH = "https://www.indeed.com/jobs"

# Keywords that suggest integration prep (acquirer-side signal)
INTEGRATION_KEYWORDS = re.compile(
    r"\b(integration|transition|change management|"
    r"merger integration|post.merger|PMO|program management office)\b",
    re.IGNORECASE,
)


class JobPostingCollector(BaseCollector):
    name = "job_postings"
    request_delay = 2.0  # Be polite to job boards

    def _fetch_indeed_count(self, company_name: str) -> int | None:
        """
        Scrape Indeed job count for a company.
        Returns the integer count shown in search results, or None on failure.
        """
        params = {"q": f'"{company_name}"', "l": "", "sort": "date"}
        try:
            resp = self.fetch(
                INDEED_SEARCH,
                params=params,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        except Exception as e:
            self.log.warning("indeed_fetch_failed", company=company_name, error=str(e))
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Indeed shows job count in a <div data-testid="jobsearch-JobCountAndSortPane-jobCount">
        count_el = soup.find(attrs={"data-testid": "jobsearch-JobCountAndSortPane-jobCount"})
        if count_el:
            text = count_el.get_text(strip=True)
            # Extract first number from strings like "1,234 jobs" or "Page 1 of 200 jobs"
            nums = re.findall(r"[\d,]+", text)
            if nums:
                try:
                    return int(nums[0].replace(",", ""))
                except ValueError:
                    pass

        # Fallback: look for "N jobs" pattern anywhere in page
        match = re.search(r'([\d,]+)\s+jobs?\b', resp.text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                pass

        return None

    def _check_integration_roles(self, company_name: str) -> int:
        """
        Count integration/PMO role postings — a signal on the ACQUIRER side.
        An acquirer hiring integration managers suggests an imminent deal.
        """
        params = {
            "q": f'"{company_name}" (integration OR "change management" OR "PMO")',
            "l": "",
            "sort": "date",
        }
        try:
            resp = self.fetch(
                INDEED_SEARCH,
                params=params,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36"
                    )
                },
            )
            match = re.search(r"([\d,]+)\s+jobs?\b", resp.text, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
        except Exception as e:
            self.log.warning(
                "integration_check_failed", company=company_name, error=str(e)
            )
        return 0

    def collect(
        self,
        ticker: str,
        company_name: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Collect a single job posting snapshot for the given company.
        In production this runs daily; the signal is the trend, not the point.
        """
        if not company_name:
            # Use ticker as fallback (less accurate)
            company_name = ticker

        count = self._fetch_indeed_count(company_name)
        integration_count = self._check_integration_roles(company_name)

        if count is None:
            self.log.warning("job_count_unavailable", ticker=ticker)
            return []

        return [
            {
                "ticker": ticker,
                "snapshot_date": datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                "posting_count": count,
                "integration_role_count": integration_count,
                "source": "indeed",
                # rolling averages computed by signal layer from historical snapshots
                "rolling_30d_avg": None,
                "rolling_90d_avg": None,
                "pct_change_30d": None,
            }
        ]

    def compute_freeze_signal(
        self,
        snapshots: list[dict],  # ordered oldest → newest
    ) -> dict:
        """
        Given a time series of snapshots, compute the freeze signal.
        Returns a dict with freeze_flag, severity, and pct_change.
        """
        if len(snapshots) < 7:
            return {"freeze_flag": False, "pct_change_30d": None, "severity": 0.0}

        counts = [s["posting_count"] for s in snapshots]
        recent = counts[-7:]  # last week
        baseline = counts[:-7]  # everything before

        avg_recent = sum(recent) / len(recent)
        avg_baseline = sum(baseline) / len(baseline) if baseline else avg_recent

        pct_change = (
            (avg_recent - avg_baseline) / avg_baseline * 100
            if avg_baseline > 0
            else 0
        )

        # Freeze: >30% drop sustained
        freeze_flag = pct_change < -30 and avg_recent < avg_baseline * 0.7

        # Severity: 0–1 scale
        severity = min(abs(pct_change) / 100, 1.0) if freeze_flag else 0.0

        return {
            "freeze_flag": freeze_flag,
            "pct_change_30d": pct_change,
            "avg_recent": avg_recent,
            "avg_baseline": avg_baseline,
            "severity": severity,
        }
