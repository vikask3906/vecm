# backtest/tearsheet.py
"""
Performance analytics and tearsheet generation.

Computes key performance metrics:
    - Sharpe ratio (annualized)
    - Maximum drawdown (depth, duration, recovery)
    - Calmar ratio (return / max drawdown)
    - Win rate, profit factor
    - Capacity analysis (alpha decay vs. notional)
    - Cost attribution (fees, impact, borrow)

Outputs both console-formatted tables and matplotlib charts.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TearsheetMetrics:
    """Complete performance metrics for a backtest."""
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    annualized_volatility: float
    downside_volatility: float

    # Trading
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return_pct: float
    avg_trade_duration_days: float
    max_consecutive_losses: int

    # Costs
    total_costs: float
    total_transaction_costs: float
    total_impact_costs: float
    total_borrow_costs: float
    costs_as_pct_of_gross: float

    # Exposure
    avg_gross_leverage: float
    avg_net_leverage: float
    max_gross_leverage: float
    pct_time_invested: float


def compute_tearsheet(results, risk_free_rate: float = 0.05) -> TearsheetMetrics:
    """
    Compute comprehensive performance metrics from backtest results.

    Parameters
    ----------
    results : BacktestResults
        Output from BacktestEngine.run().
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio calculation.

    Returns
    -------
    TearsheetMetrics
        Complete performance metrics.
    """
    df = results.to_dataframe()
    if df.empty:
        return _empty_metrics()

    # ─── Daily returns ───
    equity = df["equity"]
    daily_returns = equity.pct_change().dropna()
    n_days = len(daily_returns)
    n_years = n_days / 252

    # ─── Return metrics ───
    total_return = results.total_return_pct / 100
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # ─── Volatility ───
    ann_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0

    # ─── Sharpe & Sortino ───
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
              if excess_returns.std() > 0 else 0)
    sortino = ((daily_returns.mean() - daily_rf) / downside_vol
               if downside_vol > 0 else 0) * np.sqrt(252)

    # ─── Drawdown ───
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = abs(drawdown.min()) * 100

    # Drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # ─── Calmar ───
    calmar = ann_return / (max_dd / 100) if max_dd > 0 else 0

    # ─── Trade analysis ───
    signals = df["signal"]
    # Count trades (signal transitions from 0 to non-zero)
    trade_entries = ((signals != 0) & (signals.shift(1).fillna(0) == 0))
    num_trades = trade_entries.sum()

    # Segment P&L by trades
    trade_pnls = []
    trade_durations = []
    current_pnl = 0
    current_duration = 0
    in_trade = False
    current_sig = 0

    for i in range(len(df)):
        sig = df["signal"].iloc[i]
        pnl = df["daily_pnl"].iloc[i]

        if sig != 0:
            if not in_trade:
                in_trade = True
                current_pnl = pnl
                current_duration = 1
                current_sig = sig
            else:
                if sig != current_sig:
                    # Trade flipped directly (e.g. Long to Short). 
                    # Close old trade, assigning today's PNL to the exit of the old trade.
                    current_pnl += pnl
                    trade_pnls.append(current_pnl)
                    trade_durations.append(current_duration + 1)
                    
                    # Start tracking the new trade
                    current_pnl = 0
                    current_duration = 0
                    current_sig = sig
                else:
                    current_pnl += pnl
                    current_duration += 1
        elif in_trade:
            # Trade has closed. Accumulate the exit day PNL (includes exit fees and convergence jump).
            current_pnl += pnl
            current_duration += 1
            trade_pnls.append(current_pnl)
            trade_durations.append(current_duration)
            in_trade = False
            current_sig = 0

    if in_trade:
        trade_pnls.append(current_pnl)
        trade_durations.append(current_duration)

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / max(len(trade_pnls), 1) * 100
    profit_factor = (sum(wins) / abs(sum(losses)) if sum(losses) != 0 else
                     float("inf") if sum(wins) > 0 else 0)

    avg_trade_return = np.mean(trade_pnls) / results.initial_capital * 100 if trade_pnls else 0
    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

    # Consecutive losses
    max_consec_losses = 0
    current_consec = 0
    for pnl in trade_pnls:
        if pnl <= 0:
            current_consec += 1
            max_consec_losses = max(max_consec_losses, current_consec)
        else:
            current_consec = 0

    # ─── Exposure ───
    avg_gross = df["gross_exposure"].mean() / results.initial_capital if results.initial_capital > 0 else 0
    avg_net = df["net_exposure"].mean() / results.initial_capital if results.initial_capital > 0 else 0
    max_gross = df["gross_exposure"].max() / results.initial_capital if results.initial_capital > 0 else 0
    pct_invested = (df["signal"] != 0).sum() / len(df) * 100

    # ─── Cost attribution ───
    total_gross_traded = df["gross_exposure"].sum()
    costs_pct = results.total_costs / max(total_gross_traded, 1) * 100

    return TearsheetMetrics(
        total_return_pct=results.total_return_pct,
        annualized_return_pct=ann_return * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        annualized_volatility=ann_vol * 100,
        downside_volatility=downside_vol * 100,
        num_trades=int(num_trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return_pct=avg_trade_return,
        avg_trade_duration_days=avg_trade_duration,
        max_consecutive_losses=max_consec_losses,
        total_costs=results.total_costs,
        total_transaction_costs=results.total_transaction_costs,
        total_impact_costs=results.total_impact_costs,
        total_borrow_costs=results.total_borrow_costs,
        costs_as_pct_of_gross=costs_pct,
        avg_gross_leverage=avg_gross,
        avg_net_leverage=avg_net,
        max_gross_leverage=max_gross,
        pct_time_invested=pct_invested,
    )


def print_tearsheet(metrics: TearsheetMetrics) -> None:
    """Print a formatted tearsheet to the console."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST TEARSHEET")
    print(f"{'='*60}")

    print(f"\n  -- Returns --")
    print(f"  {'Total Return':<30} {metrics.total_return_pct:>+10.2f}%")
    print(f"  {'Annualized Return':<30} {metrics.annualized_return_pct:>+10.2f}%")
    print(f"  {'Sharpe Ratio':<30} {metrics.sharpe_ratio:>10.2f}")
    print(f"  {'Sortino Ratio':<30} {metrics.sortino_ratio:>10.2f}")
    print(f"  {'Calmar Ratio':<30} {metrics.calmar_ratio:>10.2f}")

    print(f"\n  -- Risk --")
    print(f"  {'Annualized Volatility':<30} {metrics.annualized_volatility:>10.2f}%")
    print(f"  {'Downside Volatility':<30} {metrics.downside_volatility:>10.2f}%")
    print(f"  {'Max Drawdown':<30} {metrics.max_drawdown_pct:>10.2f}%")
    print(f"  {'Max DD Duration':<30} {metrics.max_drawdown_duration_days:>10d} days")

    print(f"\n  -- Trading --")
    print(f"  {'Number of Trades':<30} {metrics.num_trades:>10d}")
    print(f"  {'Win Rate':<30} {metrics.win_rate:>10.1f}%")
    print(f"  {'Profit Factor':<30} {metrics.profit_factor:>10.2f}")
    print(f"  {'Avg Trade Return':<30} {metrics.avg_trade_return_pct:>+10.3f}%")
    print(f"  {'Avg Trade Duration':<30} {metrics.avg_trade_duration_days:>10.1f} days")
    print(f"  {'Max Consecutive Losses':<30} {metrics.max_consecutive_losses:>10d}")

    print(f"\n  -- Costs --")
    print(f"  {'Total Costs':<30} ${metrics.total_costs:>10,.2f}")
    print(f"  {'  Transaction Fees':<30} ${metrics.total_transaction_costs:>10,.2f}")
    print(f"  {'  Market Impact':<30} ${metrics.total_impact_costs:>10,.2f}")
    print(f"  {'  Borrow Costs':<30} ${metrics.total_borrow_costs:>10,.2f}")
    print(f"  {'Costs / Gross Traded':<30} {metrics.costs_as_pct_of_gross:>10.4f}%")

    print(f"\n  -- Exposure --")
    print(f"  {'Avg Gross Leverage':<30} {metrics.avg_gross_leverage:>10.2f}x")
    print(f"  {'Avg Net Leverage':<30} {metrics.avg_net_leverage:>+10.4f}x")
    print(f"  {'Max Gross Leverage':<30} {metrics.max_gross_leverage:>10.2f}x")
    print(f"  {'% Time Invested':<30} {metrics.pct_time_invested:>10.1f}%")

    print(f"{'='*60}\n")


def plot_tearsheet(results, metrics: TearsheetMetrics,
                   save_path: str = None) -> None:
    """
    Generate a multi-panel matplotlib tearsheet.

    Panels:
        1. Equity curve
        2. Drawdown
        3. Z-score with entry/exit thresholds
        4. Cointegration rank over time
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available for plotting. Skipping chart generation.")
        return

    df = results.to_dataframe()
    if df.empty:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 2, 1]})
    fig.suptitle("VECM Arbitrage Backtest Tearsheet", fontsize=16, fontweight="bold", y=0.98)

    # 1. Equity Curve
    ax1 = axes[0]
    ax1.plot(df.index, df["equity"], color="#2196F3", linewidth=1.5, label="Equity")
    ax1.axhline(y=results.initial_capital, color="gray", linestyle="--", alpha=0.5,
                label=f"Initial (${results.initial_capital:,.0f})")
    ax1.fill_between(df.index, results.initial_capital, df["equity"],
                     where=df["equity"] >= results.initial_capital,
                     alpha=0.15, color="green")
    ax1.fill_between(df.index, results.initial_capital, df["equity"],
                     where=df["equity"] < results.initial_capital,
                     alpha=0.15, color="red")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"Equity Curve  |  Sharpe={metrics.sharpe_ratio:.2f}  "
                  f"Return={metrics.total_return_pct:+.1f}%  "
                  f"MaxDD={metrics.max_drawdown_pct:.1f}%")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    cummax = df["equity"].cummax()
    drawdown = (df["equity"] - cummax) / cummax * 100
    ax2.fill_between(df.index, 0, drawdown, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    # 3. Z-score with thresholds
    ax3 = axes[2]
    ax3.plot(df.index, df["zscore"], color="#FF9800", linewidth=0.8, label="Z-score")
    ax3.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Entry (±2.0)")
    ax3.axhline(y=-2.0, color="red", linestyle="--", alpha=0.5)
    ax3.axhline(y=0.5, color="green", linestyle=":", alpha=0.5, label="Exit (±0.5)")
    ax3.axhline(y=-0.5, color="green", linestyle=":", alpha=0.5)
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Color background based on signal
    for i in range(len(df) - 1):
        if df["signal"].iloc[i] > 0:
            ax3.axvspan(df.index[i], df.index[i + 1], alpha=0.08, color="green")
        elif df["signal"].iloc[i] < 0:
            ax3.axvspan(df.index[i], df.index[i + 1], alpha=0.08, color="red")

    ax3.set_ylabel("Z-Score")
    ax3.set_title("Spread Z-Score & Trading Signals")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 5)

    # 4. Cointegration Rank
    ax4 = axes[3]
    ax4.step(df.index, df["coint_rank"], color="#9C27B0", linewidth=1.5, where="post")
    ax4.set_ylabel("Coint. Rank")
    ax4.set_title("Cointegration Rank (Johansen)")
    ax4.set_ylim(-0.5, max(df["coint_rank"].max() + 1, 3))
    ax4.grid(True, alpha=0.3)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Tearsheet saved to: {save_path}")
    else:
        plt.savefig("backtest_tearsheet.png", dpi=150, bbox_inches="tight")
        print(f"  Tearsheet saved to: backtest_tearsheet.png")

    plt.close()


def _empty_metrics() -> TearsheetMetrics:
    """Return empty metrics when no data is available."""
    return TearsheetMetrics(
        total_return_pct=0, annualized_return_pct=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown_pct=0, max_drawdown_duration_days=0,
        annualized_volatility=0, downside_volatility=0,
        num_trades=0, win_rate=0, profit_factor=0,
        avg_trade_return_pct=0, avg_trade_duration_days=0,
        max_consecutive_losses=0,
        total_costs=0, total_transaction_costs=0,
        total_impact_costs=0, total_borrow_costs=0,
        costs_as_pct_of_gross=0,
        avg_gross_leverage=0, avg_net_leverage=0,
        max_gross_leverage=0, pct_time_invested=0,
    )
