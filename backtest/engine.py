# backtest/engine.py
"""
Main backtest engine for the VECM arbitrage system.

Event-driven backtesting loop that integrates:
    - Phase 1: VECM math engine (signals)
    - Phase 3: Hedge optimization (weights)
    - Phase 4: Transaction costs + market impact
    - Phase 5: Cointegration monitoring (breakdown protocol)

The engine processes daily bars, generates signals from the VECM spread,
executes trades with realistic costs, and tracks full P&L.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from core.johansen import run_johansen
from core.vecm import fit_vecm
from core.spread import construct_spread, compute_zscore, generate_signals
from core.cointegration_monitor import CointegrationMonitor, BreakdownSeverity
from risk.hedge_optimizer import optimize_hedge, normalize_beta
from risk.position_manager import PositionManager
from backtest.transaction_costs import TransactionCostModel
from backtest.market_impact import (
    SquareRootImpactModel, compute_adv_from_volume, compute_daily_volatility
)
import config


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = config.INITIAL_CAPITAL
    entry_zscore: float = config.ENTRY_ZSCORE
    exit_zscore: float = config.EXIT_ZSCORE
    stop_loss_zscore: float = config.STOP_LOSS_ZSCORE
    zscore_window: int = config.ZSCORE_WINDOW
    johansen_det_order: int = config.JOHANSEN_DET_ORDER
    k_ar_diff: int = config.JOHANSEN_K_AR_DIFF
    reestimation_freq: int = 63     # Re-estimate VECM every ~3 months
    warmup_period: int = 252        # 1 year warmup for estimation
    enable_impact: bool = True      # Include market impact
    enable_breakdown: bool = True   # Enable cointegration breakdown protocol
    monitor_window: int = config.MONITOR_WINDOW
    monitor_recheck_freq: int = config.MONITOR_RECHECK_FREQ
    monitor_grace_period: int = config.MONITOR_GRACE_PERIOD
    monitor_cooldown: int = config.MONITOR_COOLDOWN


@dataclass
class DailyRecord:
    """Daily P&L and state record for the backtest."""
    date: pd.Timestamp
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    signal: float
    zscore: float
    spread: float
    daily_pnl: float
    cumulative_pnl: float
    transaction_costs: float
    impact_costs: float
    borrow_costs: float
    coint_rank: int
    half_life: float
    n_positions: int


class BacktestEngine:
    """
    Main backtest engine for VECM statistical arbitrage.

    Runs a daily bar-by-bar simulation with:
        - Rolling VECM estimation
        - Signal generation from z-score thresholds
        - Hedge weight optimization via CVXPY
        - Transaction cost + market impact deduction
        - Cointegration breakdown monitoring and auto-liquidation

    Usage:
        engine = BacktestEngine(log_prices, volume)
        results = engine.run()
        print(f"Sharpe: {results.sharpe:.2f}")
    """

    def __init__(
        self,
        log_prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        bt_config: BacktestConfig = None,
    ):
        """
        Parameters
        ----------
        log_prices : pd.DataFrame
            Full log price series for all assets.
        volume : pd.DataFrame, optional
            Volume data for market impact estimation.
        bt_config : BacktestConfig, optional
            Backtest configuration. Uses defaults if None.
        """
        self.log_prices = log_prices
        self.volume = volume
        self.config = bt_config or BacktestConfig()
        self.tickers = list(log_prices.columns)

        # Components
        self.pos_manager = PositionManager(capital=self.config.initial_capital)
        self.cost_model = TransactionCostModel()
        self.impact_model = SquareRootImpactModel()
        self.coint_monitor = CointegrationMonitor(
            expected_rank=1,
            window_size=self.config.monitor_window,
            recheck_freq=self.config.monitor_recheck_freq,
            grace_period=self.config.monitor_grace_period,
            cooldown_days=self.config.monitor_cooldown,
        ) if self.config.enable_breakdown else None

        # State
        self._current_beta = None
        self._current_rank = 0
        self._current_half_life = np.nan
        self._current_signal = 0.0
        self._days_since_estimation = 0
        self._total_tcost = 0.0
        self._total_impact = 0.0
        self._total_borrow = 0.0

        # Results
        self.daily_records: list[DailyRecord] = []

    def run(self) -> "BacktestResults":
        """
        Run the backtest.

        Returns
        -------
        BacktestResults
            Complete results with daily records and summary statistics.
        """
        dates = self.log_prices.index
        n = len(dates)
        warmup = self.config.warmup_period

        # Precompute ADV and volatility if volume data available
        adv_df = None
        vol_df = None
        if self.volume is not None and self.config.enable_impact:
            adv_df = compute_adv_from_volume(self.volume)
            vol_df = compute_daily_volatility(self.log_prices)

        print(f"[Backtest] Running {n} trading days, warmup={warmup}...")
        print(f"[Backtest] Assets: {self.tickers}")

        prev_equity = self.config.initial_capital

        for i in range(warmup, n):
            date = dates[i]
            window = self.log_prices.iloc[:i + 1]

            # ─── VECM re-estimation ───
            if self._days_since_estimation >= self.config.reestimation_freq or self._current_beta is None:
                self._reestimate(window)
                self._days_since_estimation = 0
            self._days_since_estimation += 1

            # ─── Cointegration monitoring ───
            if self.coint_monitor and self._current_beta is not None:
                event = self.coint_monitor.check(window, date)
                if event:
                    if event.severity == BreakdownSeverity.FULL:
                        print(f"[Backtest] {date.date()}: {event.message}")
                        # Liquidate on full breakdown
                        prices_now = np.exp(self.log_prices.iloc[i])
                        price_dict = {t: prices_now[t] for t in self.tickers}
                        self.pos_manager.close_all(price_dict)
                        self._current_signal = 0.0
                    elif event.severity == BreakdownSeverity.PARTIAL:
                        print(f"[Backtest] {date.date()}: {event.message}")

            # Skip if no valid model
            if self._current_beta is None or self._current_rank == 0:
                self._record_day(date, 0, 0, 0, prev_equity)
                continue

            # ─── Signal generation ───
            spread = construct_spread(window, self._current_beta, vec_idx=0)
            zscore = compute_zscore(spread, window=self.config.zscore_window)
            signals = generate_signals(
                zscore,
                entry_z=self.config.entry_zscore,
                exit_z=self.config.exit_zscore,
                stop_loss_z=self.config.stop_loss_zscore,
            )

            current_zscore = zscore.iloc[-1] if not zscore.empty else 0
            current_signal = signals.iloc[-1] if not signals.empty else 0
            current_spread = spread.iloc[-1] if not spread.empty else 0

            # ─── Trade execution ───
            if current_signal != self._current_signal:
                self._execute_signal_change(
                    current_signal, i, adv_df, vol_df
                )

            self._current_signal = current_signal

            # ─── Daily borrow costs ───
            daily_borrow = self._accrue_borrow_costs()

            # ─── Update prices and record ───
            prices_now = np.exp(self.log_prices.iloc[i])
            for t in self.tickers:
                self.pos_manager.update_price(t, prices_now[t])

            equity = self.pos_manager.equity
            daily_pnl = equity - prev_equity

            self._record_day(
                date, current_signal, current_zscore, current_spread,
                equity, daily_pnl, daily_borrow
            )
            prev_equity = equity

            # Progress
            if (i - warmup) % 252 == 0 and i > warmup:
                years = (i - warmup) / 252
                print(f"[Backtest] {date.date()}: Year {years:.0f}, "
                      f"equity=${equity:,.0f}, "
                      f"return={self.pos_manager.total_return_pct:+.1f}%")

        print(f"[Backtest] Complete. Final equity: ${self.pos_manager.equity:,.0f}")

        return BacktestResults(
            daily_records=self.daily_records,
            initial_capital=self.config.initial_capital,
            final_equity=self.pos_manager.equity,
            total_transaction_costs=self._total_tcost,
            total_impact_costs=self._total_impact,
            total_borrow_costs=self._total_borrow,
            tickers=self.tickers,
        )

    def _reestimate(self, window: pd.DataFrame) -> None:
        """Re-estimate the VECM model on the current window."""
        try:
            # Use last warmup_period observations for estimation
            est_window = window.iloc[-self.config.warmup_period:]

            jr = run_johansen(est_window, det_order=self.config.johansen_det_order,
                              k_ar_diff=self.config.k_ar_diff)
            self._current_rank = jr.rank

            if jr.rank > 0:
                vr = fit_vecm(est_window, rank=jr.rank,
                              k_ar_diff=self.config.k_ar_diff)
                self._current_beta = vr.beta
                self._current_half_life = vr.half_lives[0]

                if self.coint_monitor:
                    self.coint_monitor.expected_rank = jr.rank
            else:
                self._current_beta = None
                self._current_half_life = np.nan

        except Exception as e:
            # Keep previous estimates on failure
            pass

    def _execute_signal_change(self, new_signal: float, bar_idx: int,
                                adv_df: pd.DataFrame, vol_df: pd.DataFrame) -> None:
        """Execute trades for a signal change."""
        prices_now = np.exp(self.log_prices.iloc[bar_idx])
        equity = self.pos_manager.equity

        if new_signal == 0:
            # Flatten all positions
            price_dict = {t: prices_now[t] for t in self.tickers}

            # Compute transaction costs for closing
            for t in self.tickers:
                if t in self.pos_manager.positions:
                    pos = self.pos_manager.positions[t]
                    notional = abs(pos.market_value)
                    cb = self.cost_model.compute_trade_cost(
                        t, notional, is_short=pos.is_short)
                    self._total_tcost += cb.total_cost

            self.pos_manager.close_all(price_dict)
        else:
            # First close existing positions if direction changed
            if self._current_signal != 0 and self._current_signal != new_signal:
                price_dict = {t: prices_now[t] for t in self.tickers}
                self.pos_manager.close_all(price_dict)

            # Open new positions based on hedge weights
            if self._current_beta is not None:
                weights = normalize_beta(self._current_beta, vec_idx=0)

                # Signal direction: +1 = long spread, -1 = short spread
                weights = weights * new_signal

                for idx, ticker in enumerate(self.tickers):
                    target_notional = weights[idx] * equity
                    price = prices_now[ticker]
                    qty = target_notional / price

                    # Transaction cost
                    cb = self.cost_model.compute_trade_cost(
                        ticker, target_notional, is_short=(qty < 0))
                    self._total_tcost += cb.total_cost

                    # Market impact
                    impact_cost = 0
                    if self.config.enable_impact and adv_df is not None and vol_df is not None:
                        adv = adv_df.iloc[bar_idx].get(ticker, 1_000_000)
                        volatility = vol_df.iloc[bar_idx].get(ticker, 0.02)
                        if not np.isnan(adv) and not np.isnan(volatility) and adv > 0:
                            impact_est = self.impact_model.estimate_impact(
                                ticker, abs(qty), price, adv, volatility)
                            impact_cost = impact_est.impact_dollars
                            self._total_impact += impact_cost

                    # Execute with slippage from impact
                    exec_price = price * (1 + impact_cost / (abs(qty) * price + 1e-10))
                    total_fees = cb.total_cost + impact_cost
                    self.pos_manager.open_position(ticker, qty, exec_price, fees=total_fees)

    def _accrue_borrow_costs(self) -> float:
        """Accrue daily borrow costs for short positions."""
        daily_borrow = 0.0
        for pos in self.pos_manager.positions.values():
            if pos.is_short:
                daily = abs(pos.market_value) * config.BORROW_COST_ANNUAL_BPS / 10_000 / 252
                daily_borrow += daily
        self._total_borrow += daily_borrow
        self.pos_manager.cash -= daily_borrow
        return daily_borrow

    def _record_day(self, date, signal, zscore, spread,
                    equity, daily_pnl=0, borrow=0):
        """Record daily state."""
        self.daily_records.append(DailyRecord(
            date=date,
            equity=equity,
            cash=self.pos_manager.cash,
            gross_exposure=self.pos_manager.gross_exposure,
            net_exposure=self.pos_manager.net_exposure,
            signal=signal,
            zscore=zscore if not np.isnan(zscore) else 0,
            spread=spread,
            daily_pnl=daily_pnl,
            cumulative_pnl=equity - self.config.initial_capital,
            transaction_costs=self._total_tcost,
            impact_costs=self._total_impact,
            borrow_costs=self._total_borrow,
            coint_rank=self._current_rank,
            half_life=self._current_half_life if not np.isnan(self._current_half_life) else 0,
            n_positions=len(self.pos_manager.positions),
        ))


@dataclass
class BacktestResults:
    """Complete results from a backtest run."""
    daily_records: list[DailyRecord]
    initial_capital: float
    final_equity: float
    total_transaction_costs: float
    total_impact_costs: float
    total_borrow_costs: float
    tickers: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert daily records to a DataFrame."""
        records = [{
            "date": r.date,
            "equity": r.equity,
            "cash": r.cash,
            "gross_exposure": r.gross_exposure,
            "net_exposure": r.net_exposure,
            "signal": r.signal,
            "zscore": r.zscore,
            "spread": r.spread,
            "daily_pnl": r.daily_pnl,
            "cumulative_pnl": r.cumulative_pnl,
            "coint_rank": r.coint_rank,
            "half_life": r.half_life,
            "n_positions": r.n_positions,
        } for r in self.daily_records]
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("date", inplace=True)
        return df

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100

    @property
    def total_costs(self) -> float:
        return self.total_transaction_costs + self.total_impact_costs + self.total_borrow_costs
