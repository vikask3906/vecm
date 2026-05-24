# core/cointegration_monitor.py
"""
Rolling cointegration monitoring and breakdown detection (Phase 5).

This module implements a rolling Johansen test that continuously monitors
the health of the cointegrating relationship. When the rank drops below
the expected level, it triggers liquidation events.

Key concepts:
    - Rolling window Johansen test on a sliding window of prices
    - Rank-drop detection with configurable grace period
    - Cooldown period after breakdown before re-entry is allowed
    - Eigenvalue gap monitoring for early warning signals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from core.johansen import run_johansen
from core.ml_oracle import MLRegimeOracle
from core.spread import construct_spread


class BreakdownSeverity(Enum):
    """Severity levels for cointegration breakdown events."""
    WARNING = "warning"         # Eigenvalue gap narrowing
    PARTIAL = "partial"         # Rank decreased but > 0
    FULL = "full"               # Rank dropped to 0


@dataclass
class RankDropEvent:
    """Event emitted when cointegration rank degrades."""
    timestamp: pd.Timestamp
    old_rank: int
    new_rank: int
    severity: BreakdownSeverity
    eigenvalue_gap: float       # Gap between last significant and first insignificant eigenvalue
    trace_stats: np.ndarray
    message: str


@dataclass
class MonitorState:
    """Internal state of the cointegration monitor."""
    current_rank: int = 0
    consecutive_drops: int = 0
    in_cooldown: bool = False
    cooldown_remaining: int = 0
    last_check: Optional[pd.Timestamp] = None
    rank_history: list = field(default_factory=list)
    eigenvalue_history: list = field(default_factory=list)


class CointegrationMonitor:
    """
    Rolling Johansen test monitor for cointegration breakdown detection.

    Usage:
        monitor = CointegrationMonitor(expected_rank=1)
        for date, window_prices in rolling_windows:
            event = monitor.check(window_prices, date)
            if event and event.severity == BreakdownSeverity.FULL:
                # Trigger liquidation
                ...
    """

    def __init__(
        self,
        expected_rank: int = 1,
        window_size: int = 252,
        recheck_freq: int = 21,
        grace_period: int = 5,
        cooldown_days: int = 42,
        det_order: int = 0,
        k_ar_diff: int = 2,
        eigenvalue_warning_threshold: float = 0.01,
        enable_ml_oracle: bool = True,
    ):
        """
        Parameters
        ----------
        expected_rank : int
            The expected cointegration rank (typically 1).
        window_size : int
            Rolling window size in trading days for the Johansen test.
        recheck_freq : int
            Number of trading days between rechecks.
        grace_period : int
            Consecutive rank-drop observations before triggering liquidation.
        cooldown_days : int
            Days to wait after a full breakdown before allowing re-entry.
        det_order : int
            Deterministic term for Johansen test.
        k_ar_diff : int
            Lagged differences for Johansen test.
        eigenvalue_warning_threshold : float
            Emit WARNING if the eigenvalue gap narrows below this.
        """
        self.expected_rank = expected_rank
        self.window_size = window_size
        self.recheck_freq = recheck_freq
        self.grace_period = grace_period
        self.cooldown_days = cooldown_days
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        self.eigenvalue_warning_threshold = eigenvalue_warning_threshold
        
        try:
            self.ml_oracle = MLRegimeOracle() if enable_ml_oracle else None
        except FileNotFoundError:
            print("WARNING: ML Oracle model not found. Running purely on Johansen.")
            self.ml_oracle = None

        self.state = MonitorState(current_rank=expected_rank)
        self._days_since_check = 0

    def check(self, prices: pd.DataFrame,
              current_date: pd.Timestamp) -> Optional[RankDropEvent]:
        """
        Check cointegration health on the given price window.

        Parameters
        ----------
        prices : pd.DataFrame
            Rolling window of log prices (last `window_size` observations).
        current_date : pd.Timestamp
            The current date for event timestamping.

        Returns
        -------
        Optional[RankDropEvent]
            Event if rank degradation detected, None otherwise.
        """
        # Handle cooldown
        if self.state.in_cooldown:
            self.state.cooldown_remaining -= 1
            if self.state.cooldown_remaining <= 0:
                self.state.in_cooldown = False
                self.state.consecutive_drops = 0
            return None

        # Only recheck at the configured frequency
        self._days_since_check += 1
        if self._days_since_check < self.recheck_freq:
            return None
        self._days_since_check = 0

        # Ensure we have enough data
        if len(prices) < self.window_size:
            return None

        # Use the last window_size observations
        window_prices = prices.iloc[-self.window_size:]

        # Run Johansen test
        try:
            jr = run_johansen(window_prices, det_order=self.det_order,
                              k_ar_diff=self.k_ar_diff)
        except Exception:
            return None

        new_rank = jr.rank
        old_rank = self.state.current_rank

        # Record history
        self.state.rank_history.append((current_date, new_rank))
        self.state.eigenvalue_history.append((current_date, jr.eigenvalues.copy()))
        self.state.last_check = current_date

        # Compute eigenvalue gap (between last significant and first insignificant)
        if new_rank < len(jr.eigenvalues):
            if new_rank > 0:
                gap = jr.eigenvalues[new_rank - 1] - jr.eigenvalues[new_rank]
            else:
                gap = 0.0
        else:
            gap = jr.eigenvalues[-1]

        # ─── ML ORACLE REGIME DETECTION ───
        ml_regime_crisis = False
        if self.ml_oracle and new_rank > 0:
            # We use the rolling window's beta to construct the spread
            spread = construct_spread(window_prices, jr.beta, vec_idx=0)
            regime = self.ml_oracle.predict_regime(spread)
            if regime == MLRegimeOracle.REGIME_CRISIS:
                ml_regime_crisis = True

        # ─── Breakdown detection ───
        event = None

        if new_rank < self.expected_rank or ml_regime_crisis:
            self.state.consecutive_drops += 1
            
            if ml_regime_crisis:
                severity = BreakdownSeverity.FULL
                msg = (f"ML ORACLE CRISIS: HMM detected regime shift to unpredictable blowout state. "
                       f"Consecutive observations: {self.state.consecutive_drops}/{self.grace_period}")
            elif new_rank == 0:
                severity = BreakdownSeverity.FULL
                msg = (f"FULL BREAKDOWN: Johansen rank dropped from {old_rank} to 0. "
                       f"Consecutive observations: {self.state.consecutive_drops}/{self.grace_period}")
            else:
                severity = BreakdownSeverity.PARTIAL
                msg = (f"PARTIAL DEGRADATION: Johansen rank dropped from {self.expected_rank} to {new_rank}. "
                       f"Consecutive observations: {self.state.consecutive_drops}/{self.grace_period}")

            event = RankDropEvent(
                timestamp=current_date,
                old_rank=old_rank,
                new_rank=new_rank,
                severity=severity,
                eigenvalue_gap=gap,
                trace_stats=jr.trace_stats,
                message=msg,
            )

            # Trigger liquidation if grace period exceeded
            if self.state.consecutive_drops >= self.grace_period:
                self.state.in_cooldown = True
                self.state.cooldown_remaining = self.cooldown_days
                event.message += " -> LIQUIDATION TRIGGERED. Entering cooldown."

        elif gap < self.eigenvalue_warning_threshold and new_rank == self.expected_rank:
            # Eigenvalue gap warning — cointegration is weakening
            self.state.consecutive_drops = 0
            event = RankDropEvent(
                timestamp=current_date,
                old_rank=old_rank,
                new_rank=new_rank,
                severity=BreakdownSeverity.WARNING,
                eigenvalue_gap=gap,
                trace_stats=jr.trace_stats,
                message=f"WARNING: Eigenvalue gap narrowing ({gap:.6f}). "
                        f"Cointegration may be weakening.",
            )
        else:
            # Rank is stable — reset consecutive drops
            self.state.consecutive_drops = 0

        self.state.current_rank = new_rank
        return event

    @property
    def is_safe(self) -> bool:
        """True if cointegration rank is at expected level and not in cooldown."""
        return (self.state.current_rank >= self.expected_rank
                and not self.state.in_cooldown)

    @property
    def is_in_cooldown(self) -> bool:
        """True if monitor is in post-breakdown cooldown period."""
        return self.state.in_cooldown

    def get_rank_history(self) -> pd.DataFrame:
        """Return rank history as a DataFrame."""
        if not self.state.rank_history:
            return pd.DataFrame(columns=["date", "rank"])
        dates, ranks = zip(*self.state.rank_history)
        return pd.DataFrame({"date": dates, "rank": ranks})

    def reset(self, expected_rank: int = None) -> None:
        """Reset monitor state (e.g., after re-estimating the model)."""
        if expected_rank is not None:
            self.expected_rank = expected_rank
        self.state = MonitorState(current_rank=self.expected_rank)
        self._days_since_check = 0
