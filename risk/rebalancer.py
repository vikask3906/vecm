# risk/rebalancer.py
"""
Rebalancer — Triggers hedge re-optimization on liquidity events.

Listens for liquidity events from the LiquidityMonitor and decides
whether to rebalance, reduce, or liquidate positions. Implements
graceful degradation: when an asset becomes illiquid, it's excluded
from the hedge optimization and positions are scaled down proportionally.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from risk.hedge_optimizer import optimize_hedge, HedgeResult
from risk.liquidity_monitor import (
    LiquidityMonitor, LiquidityEvent,
    LiquidityEventType, EventSeverity,
)
from risk.position_manager import PositionManager
import config


@dataclass
class RebalanceAction:
    """Describes a rebalancing action to execute."""
    action_type: str        # "rebalance", "reduce", "liquidate", "hold"
    target_weights: np.ndarray
    current_weights: np.ndarray
    trade_weights: np.ndarray   # target - current = what to trade
    reason: str
    excluded_assets: list[str]
    scale_factor: float = 1.0   # 1.0 = full size, <1.0 = scaled down


class Rebalancer:
    """
    Monitors liquidity events and triggers hedge re-optimization.

    The rebalancer sits between the LiquidityMonitor and the
    PositionManager, deciding when and how to adjust positions.

    Degradation hierarchy:
        1. Normal: Full β-tracking with all assets
        2. Reduced: Exclude illiquid assets, reoptimize
        3. Scaled: Reduce position sizes proportionally
        4. Emergency: Full liquidation on critical failure
    """

    def __init__(
        self,
        tickers: list[str],
        liquidity_monitor: LiquidityMonitor,
        position_manager: PositionManager,
        max_rebalance_freq: int = 1,  # Minimum days between rebalances
    ):
        self.tickers = tickers
        self.liq_monitor = liquidity_monitor
        self.pos_manager = position_manager
        self.max_rebalance_freq = max_rebalance_freq
        self._days_since_rebalance = 0
        self._rebalance_history: list[RebalanceAction] = []

    def evaluate(
        self,
        beta: np.ndarray,
        current_prices: dict[str, float],
        vec_idx: int = 0,
    ) -> Optional[RebalanceAction]:
        """
        Evaluate whether rebalancing is needed based on current conditions.

        Parameters
        ----------
        beta : np.ndarray
            (k, r) cointegrating vector matrix.
        current_prices : dict[str, float]
            Current prices for all assets.
        vec_idx : int
            Which cointegrating vector to track.

        Returns
        -------
        Optional[RebalanceAction]
            Rebalancing action to execute, or None if no action needed.
        """
        # Check for liquidity events
        events = self.liq_monitor.check_all()

        # Classify severity
        critical_events = [e for e in events if e.severity == EventSeverity.CRITICAL]
        warning_events = [e for e in events if e.severity == EventSeverity.WARNING]

        # Get excluded assets
        excluded_tickers = self.liq_monitor.get_excluded_tickers()
        excluded_indices = [self.tickers.index(t) for t in excluded_tickers
                           if t in self.tickers]

        # Compute current weights from positions
        equity = self.pos_manager.equity
        current_weights = np.zeros(len(self.tickers))
        for i, ticker in enumerate(self.tickers):
            if ticker in self.pos_manager.positions:
                pos = self.pos_manager.positions[ticker]
                current_weights[i] = pos.market_value / equity if equity > 0 else 0

        # ─── Decision logic ───

        # Emergency: multiple critical events or halts
        halt_count = sum(1 for e in critical_events
                        if e.event_type == LiquidityEventType.HALT)
        if halt_count >= 2:
            return RebalanceAction(
                action_type="liquidate",
                target_weights=np.zeros(len(self.tickers)),
                current_weights=current_weights,
                trade_weights=-current_weights,
                reason=f"Emergency liquidation: {halt_count} assets halted",
                excluded_assets=excluded_tickers,
                scale_factor=0.0,
            )

        # Critical events: reoptimize with exclusions
        if critical_events:
            hr = optimize_hedge(beta, vec_idx=vec_idx,
                                excluded_indices=excluded_indices)
            # Scale down based on number of excluded assets
            n_excluded = len(excluded_indices)
            n_total = len(self.tickers)
            scale = max(0.5, 1.0 - n_excluded / n_total)

            target = hr.weights * scale
            trade = target - current_weights

            return RebalanceAction(
                action_type="reduce",
                target_weights=target,
                current_weights=current_weights,
                trade_weights=trade,
                reason=f"Critical events on {[e.ticker for e in critical_events]}",
                excluded_assets=excluded_tickers,
                scale_factor=scale,
            )

        # Warning events: reoptimize but maintain full size
        if warning_events and self._days_since_rebalance >= self.max_rebalance_freq:
            hr = optimize_hedge(beta, vec_idx=vec_idx,
                                excluded_indices=excluded_indices)
            trade = hr.weights - current_weights

            self._days_since_rebalance = 0
            return RebalanceAction(
                action_type="rebalance",
                target_weights=hr.weights,
                current_weights=current_weights,
                trade_weights=trade,
                reason=f"Rebalance on warnings: {[e.ticker for e in warning_events]}",
                excluded_assets=excluded_tickers,
                scale_factor=1.0,
            )

        self._days_since_rebalance += 1
        return None

    def execute_action(self, action: RebalanceAction,
                       prices: dict[str, float]) -> None:
        """
        Execute a rebalancing action by adjusting positions.

        Parameters
        ----------
        action : RebalanceAction
            The action to execute.
        prices : dict[str, float]
            Current prices for execution.
        """
        equity = self.pos_manager.equity

        if action.action_type == "liquidate":
            self.pos_manager.close_all(prices)
            print(f"[Rebalancer] LIQUIDATED all positions: {action.reason}")
        else:
            for i, ticker in enumerate(self.tickers):
                target_value = action.target_weights[i] * equity
                current_value = action.current_weights[i] * equity
                trade_value = target_value - current_value

                if abs(trade_value) < 100:  # Minimum trade size
                    continue

                price = prices.get(ticker, 0)
                if price <= 0:
                    continue

                trade_qty = trade_value / price
                self.pos_manager.open_position(ticker, trade_qty, price)

            print(f"[Rebalancer] {action.action_type.upper()}: {action.reason}")
            print(f"  Scale: {action.scale_factor:.1%}, "
                  f"Excluded: {action.excluded_assets}")

        self._rebalance_history.append(action)

    @property
    def rebalance_history(self) -> list[RebalanceAction]:
        return self._rebalance_history
