"""
Base collector class. All collectors inherit from this.
Handles rate limiting, retries, and structured logging.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger()


class BaseCollector(ABC):
    """
    Abstract base for all data collectors.

    Subclasses implement `collect()` which should return a list of dicts
    that will be upserted into the database.
    """

    name: str = "base"
    # Seconds between requests. EDGAR allows 10/sec; we're conservative.
    request_delay: float = 0.2

    def __init__(self):
        self.log = structlog.get_logger(collector=self.name)
        self._last_request_time: float = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def fetch(self, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        headers = kwargs.pop("headers", {})
        headers.setdefault("User-Agent", "Harbinger Research contact@harbinger.local")
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    async def fetch_async(self, url: str, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        headers.setdefault("User-Agent", "Harbinger Research contact@harbinger.local")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp

    @abstractmethod
    def collect(self, ticker: str, **kwargs) -> list[dict[str, Any]]:
        """Collect signals for a given ticker. Returns list of raw records."""
        ...

    def collect_batch(
        self, tickers: list[str], **kwargs
    ) -> dict[str, list[dict[str, Any]]]:
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.collect(ticker, **kwargs)
                self.log.info("collected", ticker=ticker, count=len(results[ticker]))
            except Exception as e:
                self.log.error("collection_failed", ticker=ticker, error=str(e))
                results[ticker] = []
        return results
