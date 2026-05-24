# run_backtest.py
"""
Full Backtest Driver — Runs the VECM arbitrage strategy with all phases integrated.

Executes:
    1. Fetches historical price and volume data
    2. Runs the backtest engine (VECM signals + hedge optimization + costs + breakdown)
    3. Computes and prints the tearsheet
    4. Generates tearsheet charts
"""

from core.data_loader import fetch_prices, fetch_prices_with_volume, ASSETS
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.tearsheet import compute_tearsheet, print_tearsheet, plot_tearsheet
import config


def main():
    print("=" * 60)
    print("  VECM ARBITRAGE SYSTEM — FULL BACKTEST")
    print("=" * 60)

    # ─── Fetch data ───
    print("\n[1/3] Fetching historical data...")
    log_prices, volume = fetch_prices_with_volume(
        config.ASSETS, start="2016-01-01", end="2024-01-01"
    )
    print(f"  Assets: {list(log_prices.columns)}")
    print(f"  Period: {log_prices.index[0].date()} -> {log_prices.index[-1].date()}")
    print(f"  Trading days: {len(log_prices)}")

    # ─── Configure backtest ───
    bt_config = BacktestConfig(
        initial_capital=config.INITIAL_CAPITAL,
        entry_zscore=config.ENTRY_ZSCORE,
        exit_zscore=config.EXIT_ZSCORE,
        stop_loss_zscore=config.STOP_LOSS_ZSCORE,
        zscore_window=config.ZSCORE_WINDOW,
        reestimation_freq=63,         # Re-estimate every quarter
        warmup_period=252,             # 1 year warmup
        enable_impact=True,
        enable_breakdown=True,
        monitor_window=config.MONITOR_WINDOW,
        monitor_recheck_freq=config.MONITOR_RECHECK_FREQ,
        monitor_grace_period=config.MONITOR_GRACE_PERIOD,
        monitor_cooldown=config.MONITOR_COOLDOWN,
    )

    # ─── Run backtest ───
    print(f"\n[2/3] Running backtest...")
    print(f"  Config: entry_z={bt_config.entry_zscore}, exit_z={bt_config.exit_zscore}, "
          f"stop_z={bt_config.stop_loss_zscore}")
    print(f"  Capital: ${bt_config.initial_capital:,.0f}")
    print(f"  Re-estimation: every {bt_config.reestimation_freq} days")
    print(f"  Breakdown protocol: {'ON' if bt_config.enable_breakdown else 'OFF'}")
    print(f"  Market impact: {'ON' if bt_config.enable_impact else 'OFF'}")

    engine = BacktestEngine(log_prices, volume, bt_config)
    results = engine.run()

    # ─── Compute & print tearsheet ───
    print(f"\n[3/3] Computing tearsheet...")
    metrics = compute_tearsheet(results)
    print_tearsheet(metrics)

    # ─── Generate charts ───
    print("  Generating tearsheet chart...")
    plot_tearsheet(results, metrics, save_path="backtest_tearsheet.png")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE")
    print(f"  Final equity:     ${results.final_equity:>14,.2f}")
    print(f"  Total return:     {results.total_return_pct:>+13.2f}%")
    print(f"  Sharpe ratio:     {metrics.sharpe_ratio:>13.2f}")
    print(f"  Max drawdown:     {metrics.max_drawdown_pct:>13.2f}%")
    print(f"  Total costs:      ${results.total_costs:>14,.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
