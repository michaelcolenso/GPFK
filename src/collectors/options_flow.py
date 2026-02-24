"""
Options flow unusual activity detector.

Data source: Yahoo Finance options chains via yfinance (free, no auth).

The signal: abnormal OTM call buying before M&A announcements.
When someone knows a deal is coming at a 30% premium, they buy OTM calls
at strikes near the expected acquisition price. This is the first thing
the SEC looks at post-announcement.

We detect the PATTERN from public data — not from tips.

What we measure:
  1. OTM call volume / open interest ratio  (fresh buying vs. existing positions)
  2. OTM call volume relative to 30d baseline  (is today unusual?)
  3. Put/call ratio drop  (shift toward calls = bullish positioning)
  4. Implied volatility skew  (OTM calls more expensive = someone pricing in upside)
  5. Short-dated vs. long-dated call ratio  (near-term bets = event-driven)

NOTE: This detects public market patterns. All data is from public exchanges.
This is identical methodology to academic M&A detection papers (Cao et al., 2005;
Augustin et al., 2019). Trading on public options data is not insider trading.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from src.collectors.base import BaseCollector

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    yf = None

# OTM threshold: calls we consider "acquisition-bet" strikes
# 5% to 50% above current price — range that covers typical 20-40% deal premiums
OTM_LOWER_BOUND = 0.05
OTM_UPPER_BOUND = 0.50

# Expiry window: 1 to 8 weeks out is the "deal bet" sweet spot
MIN_EXPIRY_DAYS = 7
MAX_EXPIRY_DAYS = 60

# Volume/OI ratio threshold for "fresh buying" flag
VOLUME_OI_THRESHOLD = 0.5


class OptionsFlowCollector(BaseCollector):
    name = "options_flow"
    request_delay = 1.0  # Yahoo Finance rate limits aggressively

    def _get_options_snapshot(self, ticker: str) -> dict | None:
        """
        Pull current options chain for a ticker.
        Returns structured snapshot or None on failure.
        """
        if not YF_AVAILABLE:
            self.log.warning("yfinance_not_installed", ticker=ticker)
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            current_price = getattr(info, "last_price", None)
            if not current_price:
                return None

            expiry_dates = stock.options
            if not expiry_dates:
                return None

        except Exception as e:
            self.log.warning("yf_fetch_failed", ticker=ticker, error=str(e))
            return None

        now = datetime.utcnow()
        target_expiries = []
        for expiry_str in expiry_dates:
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
                days_out = (expiry_dt - now).days
                if MIN_EXPIRY_DAYS <= days_out <= MAX_EXPIRY_DAYS:
                    target_expiries.append(expiry_str)
            except ValueError:
                continue

        if not target_expiries:
            return None

        total_otm_call_volume = 0
        total_otm_call_oi = 0
        total_call_volume = 0
        total_put_volume = 0
        max_volume_oi_ratio = 0.0
        iv_skew_calls = []

        for expiry in target_expiries[:3]:  # cap at 3 expiries to avoid rate limits
            try:
                chain = stock.option_chain(expiry)
            except Exception as e:
                self.log.warning("chain_fetch_failed", ticker=ticker, expiry=expiry, error=str(e))
                continue

            calls = chain.calls
            puts = chain.puts

            if calls.empty:
                continue

            # OTM calls: strike between 5% and 50% above current price
            otm_mask = (
                (calls["strike"] > current_price * (1 + OTM_LOWER_BOUND))
                & (calls["strike"] < current_price * (1 + OTM_UPPER_BOUND))
            )
            otm_calls = calls[otm_mask]

            otm_vol = int(otm_calls["volume"].fillna(0).sum())
            otm_oi = int(otm_calls["openInterest"].fillna(0).sum())

            total_otm_call_volume += otm_vol
            total_otm_call_oi += max(otm_oi, 1)
            total_call_volume += int(calls["volume"].fillna(0).sum())
            total_put_volume += int(puts["volume"].fillna(0).sum()) if not puts.empty else 0

            # Volume/OI ratio for OTM calls
            if otm_oi > 0 and otm_vol > 0:
                ratio = otm_vol / otm_oi
                max_volume_oi_ratio = max(max_volume_oi_ratio, ratio)

            # IV skew: average IV of OTM calls
            if "impliedVolatility" in otm_calls.columns:
                iv_vals = otm_calls["impliedVolatility"].dropna().tolist()
                iv_skew_calls.extend(iv_vals)

        pc_ratio = (
            total_put_volume / total_call_volume
            if total_call_volume > 0
            else 1.0
        )
        avg_iv_skew = float(np.mean(iv_skew_calls)) if iv_skew_calls else 0.0
        volume_oi_ratio = total_otm_call_volume / total_otm_call_oi

        return {
            "ticker": ticker,
            "snapshot_date": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "current_price": current_price,
            "otm_call_volume": total_otm_call_volume,
            "otm_call_oi": total_otm_call_oi,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "put_call_ratio": round(pc_ratio, 4),
            "volume_oi_ratio": round(volume_oi_ratio, 4),
            "max_volume_oi_ratio": round(max_volume_oi_ratio, 4),
            "avg_otm_iv": round(avg_iv_skew, 4),
            "expiries_checked": len(target_expiries[:3]),
        }

    def compute_anomaly_score(
        self,
        snapshots: list[dict],  # ordered oldest → newest
    ) -> float:
        """
        Given a time series of daily snapshots, score how anomalous today is.
        Uses z-score of OTM call volume vs. 30-day rolling baseline.
        Returns 0.0–1.0.
        """
        if len(snapshots) < 5:
            return 0.0

        volumes = np.array([s["otm_call_volume"] for s in snapshots], dtype=float)
        pc_ratios = np.array([s["put_call_ratio"] for s in snapshots], dtype=float)
        vol_oi_ratios = np.array([s["volume_oi_ratio"] for s in snapshots], dtype=float)

        # Z-score of most recent day vs prior baseline
        baseline_vol = volumes[:-1]
        today_vol = volumes[-1]

        mean_vol = baseline_vol.mean()
        std_vol = baseline_vol.std() if baseline_vol.std() > 0 else 1.0
        vol_zscore = (today_vol - mean_vol) / std_vol

        # P/C ratio: low ratio = more calls vs. puts = bullish signal
        avg_pc = pc_ratios[:-1].mean()
        today_pc = pc_ratios[-1]
        pc_drop = max(avg_pc - today_pc, 0) / avg_pc if avg_pc > 0 else 0

        # Volume/OI ratio today
        today_voi = vol_oi_ratios[-1]
        voi_signal = min(today_voi / 2.0, 1.0)  # normalize: 2x = max signal

        # Composite
        score = (
            np.clip(vol_zscore / 4, 0, 1) * 0.50
            + pc_drop * 0.30
            + voi_signal * 0.20
        )
        return float(np.clip(score, 0, 1))

    def collect(
        self,
        ticker: str,
        **kwargs,
    ) -> list[dict[str, Any]]:
        snapshot = self._get_options_snapshot(ticker)
        if snapshot is None:
            return []
        return [snapshot]

    def summarize(self, ticker: str, snapshots: list[dict] | None = None) -> dict:
        """
        Summarize using stored snapshot history.
        `snapshots` should be fetched from DB by the caller.
        """
        current = self._get_options_snapshot(ticker)
        if not current:
            return {
                "otm_call_volume": 0,
                "put_call_ratio": 1.0,
                "volume_oi_ratio": 0.0,
                "anomaly_score": 0.0,
                "fresh_buying_flag": False,
            }

        all_snaps = (snapshots or []) + [current]
        anomaly = self.compute_anomaly_score(all_snaps)

        return {
            "otm_call_volume": current["otm_call_volume"],
            "put_call_ratio": current["put_call_ratio"],
            "volume_oi_ratio": current["volume_oi_ratio"],
            "anomaly_score": anomaly,
            "fresh_buying_flag": current["volume_oi_ratio"] > VOLUME_OI_THRESHOLD,
        }
