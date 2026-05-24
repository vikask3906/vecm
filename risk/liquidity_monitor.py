# risk/liquidity_monitor.py
"""
Liquidity monitoring, halt detection, and hard-to-borrow flagging.

Monitors real-time and historical volume data to detect:
    - Trading halts (zero volume or exchange-flagged halts)
    - Hard-to-borrow conditions (elevated borrow costs)
    - Low liquidity (volume drops below ADV thresholds)
    - Wide spreads (bid-ask spread exceeds normal levels)

Emits LiquidityEvent objects that the Rebalancer consumes to trigger
hedge re-optimization or position reduction.
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import config


class LiquidityEventType(Enum):
    """Types of liquidity events."""
    HALT = "halt"                      # Trading halt
    HARD_TO_BORROW = "hard_to_borrow"  # Elevated borrow cost
    LOW_VOLUME = "low_volume"          # Volume below ADV threshold
    WIDE_SPREAD = "wide_spread"        # Abnormal bid-ask spread
    RESUMED = "resumed"                # Halt lifted / liquidity restored


class EventSeverity(Enum):
    """Severity levels for liquidity events."""
    INFO = "info"           # Informational only
    WARNING = "warning"     # Requires monitoring
    CRITICAL = "critical"   # Requires immediate action (rebalance or liquidate)


@dataclass
class LiquidityEvent:
    """A liquidity event for a specific asset."""
    event_type: LiquidityEventType
    severity: EventSeverity
    ticker: str
    timestamp: float
    message: str
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class LiquidityMonitor:
    """
    Monitors liquidity conditions across all tracked assets.

    Usage:
        monitor = LiquidityMonitor(tickers=ASSETS)
        monitor.update_volume("XLE", 1_200_000)
        monitor.update_spread_bps("XLE", 3.5)
        events = monitor.check_all()
        for event in events:
            print(event.message)
    """

    def __init__(
        self,
        tickers: list[str],
        min_adv: int = None,
        warning_pct: float = None,
        halt_pct: float = None,
        max_spread_bps: float = 10.0,
    ):
        """
        Parameters
        ----------
        tickers : list[str]
            Assets to monitor.
        min_adv : int, optional
            Minimum average daily volume. Defaults to config.
        warning_pct : float, optional
            Warning threshold as fraction of ADV. Defaults to config.
        halt_pct : float, optional
            Halt threshold as fraction of ADV. Defaults to config.
        max_spread_bps : float
            Maximum acceptable bid-ask spread in bps.
        """
        self.tickers = tickers
        self.min_adv = min_adv or config.MIN_ADV_SHARES
        self.warning_pct = warning_pct or config.LIQUIDITY_WARNING_PCT
        self.halt_pct = halt_pct or config.LIQUIDITY_HALT_PCT
        self.max_spread_bps = max_spread_bps

        # State tracking
        self._volumes: dict[str, list[float]] = {t: [] for t in tickers}
        self._current_volume: dict[str, float] = {t: 0 for t in tickers}
        self._spread_bps: dict[str, float] = {t: 0 for t in tickers}
        self._is_halted: dict[str, bool] = {t: False for t in tickers}
        self._is_htb: dict[str, bool] = {t: False for t in tickers}
        self._event_history: list[LiquidityEvent] = []

    def update_volume(self, ticker: str, volume: float) -> None:
        """Update the current volume for an asset."""
        if ticker in self._volumes:
            self._current_volume[ticker] = volume
            self._volumes[ticker].append(volume)
            # Keep only last N days for ADV calculation
            if len(self._volumes[ticker]) > config.ADV_LOOKBACK:
                self._volumes[ticker] = self._volumes[ticker][-config.ADV_LOOKBACK:]

    def update_spread_bps(self, ticker: str, spread_bps: float) -> None:
        """Update the current bid-ask spread in bps."""
        if ticker in self._spread_bps:
            self._spread_bps[ticker] = spread_bps

    def set_halt(self, ticker: str, halted: bool) -> None:
        """Set trading halt status for an asset."""
        if ticker in self._is_halted:
            self._is_halted[ticker] = halted

    def set_hard_to_borrow(self, ticker: str, htb: bool) -> None:
        """Set hard-to-borrow flag for an asset."""
        if ticker in self._is_htb:
            self._is_htb[ticker] = htb

    def get_adv(self, ticker: str) -> float:
        """Get the average daily volume for an asset."""
        vols = self._volumes.get(ticker, [])
        return np.mean(vols) if vols else 0.0

    def check_all(self) -> list[LiquidityEvent]:
        """
        Check all assets for liquidity events.

        Returns
        -------
        list[LiquidityEvent]
            List of events detected in this check cycle.
        """
        events = []
        now = time.time()

        for ticker in self.tickers:
            # Check for halt
            if self._is_halted[ticker]:
                events.append(LiquidityEvent(
                    event_type=LiquidityEventType.HALT,
                    severity=EventSeverity.CRITICAL,
                    ticker=ticker,
                    timestamp=now,
                    message=f"{ticker}: TRADING HALTED — immediate rebalance required",
                ))

            # Check hard-to-borrow
            if self._is_htb[ticker]:
                events.append(LiquidityEvent(
                    event_type=LiquidityEventType.HARD_TO_BORROW,
                    severity=EventSeverity.WARNING,
                    ticker=ticker,
                    timestamp=now,
                    message=f"{ticker}: Hard-to-borrow — elevated borrow costs",
                ))

            # Check volume vs ADV
            adv = self.get_adv(ticker)
            current_vol = self._current_volume[ticker]
            if adv > 0 and current_vol > 0:
                vol_ratio = current_vol / adv
                if vol_ratio < self.halt_pct:
                    events.append(LiquidityEvent(
                        event_type=LiquidityEventType.LOW_VOLUME,
                        severity=EventSeverity.CRITICAL,
                        ticker=ticker,
                        timestamp=now,
                        message=f"{ticker}: Volume critically low ({vol_ratio:.1%} of ADV)",
                        details={"adv": adv, "current_volume": current_vol,
                                 "ratio": vol_ratio},
                    ))
                elif vol_ratio < self.warning_pct:
                    events.append(LiquidityEvent(
                        event_type=LiquidityEventType.LOW_VOLUME,
                        severity=EventSeverity.WARNING,
                        ticker=ticker,
                        timestamp=now,
                        message=f"{ticker}: Volume below warning ({vol_ratio:.1%} of ADV)",
                        details={"adv": adv, "current_volume": current_vol,
                                 "ratio": vol_ratio},
                    ))

            # Check bid-ask spread
            spread = self._spread_bps.get(ticker, 0)
            if spread > self.max_spread_bps:
                events.append(LiquidityEvent(
                    event_type=LiquidityEventType.WIDE_SPREAD,
                    severity=EventSeverity.WARNING,
                    ticker=ticker,
                    timestamp=now,
                    message=f"{ticker}: Wide spread ({spread:.1f} bps > {self.max_spread_bps} bps)",
                    details={"spread_bps": spread},
                ))

        self._event_history.extend(events)
        return events

    def get_excluded_tickers(self) -> list[str]:
        """Get tickers that should be excluded from trading due to critical events."""
        return [t for t in self.tickers
                if self._is_halted[t] or
                (self.get_adv(t) > 0 and
                 self._current_volume[t] / self.get_adv(t) < self.halt_pct)]

    def get_excluded_indices(self, all_tickers: list[str]) -> list[int]:
        """Get indices of excluded tickers within the full ticker list."""
        excluded = set(self.get_excluded_tickers())
        return [i for i, t in enumerate(all_tickers) if t in excluded]

    @property
    def event_history(self) -> list[LiquidityEvent]:
        return self._event_history
