# grid_search_backtest.py
"""
VECM Arbitrage Parameter Sweeper

Performs an automated grid search across z-score thresholds and cooldown periods
to optimize the Sharpe Ratio, Annualized Return, and Drawdown profile.
Runs backtests in parallel using ProcessPoolExecutor.
"""

import sys
import os
import time
import warnings
from itertools import product
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ignore warnings from dep libraries to keep output clean
warnings.filterwarnings("ignore")

# Import system components
from core.data_loader import fetch_prices_with_volume, ASSETS
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.tearsheet import compute_tearsheet
import config

# Define parameter ranges to sweep
ENTRY_Z_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
EXIT_Z_GRID = [0.2, 0.5, 0.8]
COOLDOWN_GRID = [10, 15, 21, 30, 42]

@dataclass
class SearchResult:
    entry_z: float
    exit_z: float
    cooldown: int
    total_return: float
    ann_return: float
    sharpe: float
    max_dd: float
    num_trades: int
    win_rate: float
    profit_factor: float
    pct_time_invested: float
    total_costs: float

def run_single_backtest(entry_z, exit_z, cooldown, log_prices, volume):
    """Worker function to run a single configuration."""
    try:
        # Enforce exit < entry
        if exit_z >= entry_z:
            return None

        # Build config
        bt_config = BacktestConfig(
            initial_capital=config.INITIAL_CAPITAL,
            entry_zscore=entry_z,
            exit_zscore=exit_z,
            stop_loss_zscore=config.STOP_LOSS_ZSCORE,
            zscore_window=config.ZSCORE_WINDOW,
            reestimation_freq=63,
            warmup_period=252,
            enable_impact=True,
            enable_breakdown=True,
            monitor_window=config.MONITOR_WINDOW,
            monitor_recheck_freq=config.MONITOR_RECHECK_FREQ,
            monitor_grace_period=config.MONITOR_GRACE_PERIOD,
            monitor_cooldown=cooldown,
        )

        # Run engine
        engine = BacktestEngine(log_prices, volume, bt_config)
        results = engine.run()
        metrics = compute_tearsheet(results)

        return SearchResult(
            entry_z=entry_z,
            exit_z=exit_z,
            cooldown=cooldown,
            total_return=results.total_return_pct,
            ann_return=metrics.annualized_return_pct,
            sharpe=metrics.sharpe_ratio,
            max_dd=metrics.max_drawdown_pct,
            num_trades=metrics.num_trades,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            pct_time_invested=metrics.pct_time_invested,
            total_costs=results.total_costs
        )
    except Exception as e:
        print(f"Error running configuration Entry={entry_z}, Exit={exit_z}, Cooldown={cooldown}: {e}")
        return None

def main():
    print("=" * 70)
    print("  VECM ARBITRAGE SYSTEM — AUTOMATED PARAMETER GRID SWEEP")
    print("=" * 70)

    # ─── 1. Fetch data once ───
    print("\n[1/3] Fetching historical prices & volumes...")
    log_prices, volume = fetch_prices_with_volume(
        config.ASSETS, start="2016-01-01", end="2024-01-01"
    )
    print(f"  Universe: {list(log_prices.columns)}")
    print(f"  Period:   {log_prices.index[0].date()} -> {log_prices.index[-1].date()}")
    print(f"  Days:     {len(log_prices)} trading days")

    # ─── 2. Build grid combinations ───
    param_combinations = []
    for entry_z, exit_z, cooldown in product(ENTRY_Z_GRID, EXIT_Z_GRID, COOLDOWN_GRID):
        if exit_z < entry_z:
            param_combinations.append((entry_z, exit_z, cooldown))

    n_runs = len(param_combinations)
    print(f"\n[2/3] Generated {n_runs} valid parameter combinations to sweep.")
    print("  Grid settings:")
    print(f"    Entry Z:  {ENTRY_Z_GRID}")
    print(f"    Exit Z:   {EXIT_Z_GRID}")
    print(f"    Cooldown: {COOLDOWN_GRID} trading days")
    print("\nStarting parallel sweep (using ProcessPoolExecutor)...")

    start_time = time.time()
    results_list = []

    # Run in parallel using ProcessPoolExecutor
    # Note: On Windows, passing dataframes to subprocesses is fine since they are pickled.
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"  Running on {max_workers} CPU cores...")

    completed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                run_single_backtest, entry_z, exit_z, cooldown, log_prices, volume
            ): (entry_z, exit_z, cooldown)
            for entry_z, exit_z, cooldown in param_combinations
        }

        # Process as they complete
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            if res is not None:
                results_list.append(res)
                # Print micro progress update
                sys.stdout.write(
                    f"\r  Progress: {completed_count}/{n_runs} runs complete... "
                    f"Latest Sharpe: {res.sharpe:>+5.2f} (Entry={res.entry_z}, Exit={res.exit_z}, Cooldown={res.cooldown})"
                )
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n\n[3/3] Sweep completed in {elapsed:.1f} seconds.")

    if not results_list:
        print("Error: No successful backtest runs.")
        return

    # ─── 3. Compile and analyze results ───
    df_results = pd.DataFrame([res.__dict__ for res in results_list])
    
    # Save complete findings to CSV
    csv_filename = "grid_search_results.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"  Saved full raw results ({len(df_results)} rows) to: {csv_filename}")

    # Display Top 10 Configurations sorted by Sharpe ratio
    print("\n" + "=" * 80)
    print(f"                   TOP 10 CONFIGURATIONS (BY SHARPE RATIO)")
    print("=" * 80)
    
    # Sort and slice
    df_sorted = df_results.sort_values(by="sharpe", ascending=False).reset_index(drop=True)
    
    # Format printing
    print(f"{'Rank':<5} {'EntryZ':<8} {'ExitZ':<8} {'Cooldown':<10} {'Tot Ret':<10} {'Ann Ret':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Win%':<6} {'Invest%':<8}")
    print("-" * 95)
    for idx, row in df_sorted.head(10).iterrows():
        print(
            f"{idx+1:<5} "
            f"{row['entry_z']:<8.1f} "
            f"{row['exit_z']:<8.1f} "
            f"{int(row['cooldown']):<10d} "
            f"{row['total_return']:>+8.2f}% "
            f"{row['ann_return']:>+8.2f}% "
            f"{row['sharpe']:>8.2f} "
            f"{row['max_dd']:>7.2f}% "
            f"{int(row['num_trades']):>8d} "
            f"{row['win_rate']:>5.1f}% "
            f"{row['pct_time_invested']:>7.1f}%"
        )
    print("=" * 95)

    # ─── 4. Deliver recommendations ───
    best = df_sorted.iloc[0]
    baseline = df_results[(df_results["entry_z"] == 2.0) & (df_results["exit_z"] == 0.5) & (df_results["cooldown"] == 42)]
    
    print("\n" + "*" * 80)
    print("  DECISION ENGINE RECOMMENDATION")
    print("*" * 80)
    print(f"  Optimal Configuration found:")
    print(f"    - Entry Z-Score:      {best['entry_z']:.2f} (from 2.0)")
    print(f"    - Exit Z-Score:       {best['exit_z']:.2f} (from 0.5)")
    print(f"    - Cooldown Period:    {int(best['cooldown'])} trading days (from 42)")
    print(f"\n  Projected Metrics Improvement:")
    print(f"    - Total Return:       {best['total_return']:>+7.2f}%  (Baseline: +4.13%)")
    print(f"    - Annualized Return:  {best['ann_return']:>+7.2f}%  (Baseline: +0.58%)")
    print(f"    - Sharpe Ratio:       {best['sharpe']:>7.2f}  (Baseline: -0.92)")
    print(f"    - Max Drawdown:       {best['max_dd']:>7.2f}%  (Baseline: 13.69%)")
    print(f"    - Total Trades:       {int(best['num_trades'])} trades  (Baseline: 27)")
    print(f"    - % Time Invested:    {best['pct_time_invested']:>6.1f}%  (Baseline: 25.8%)")
    print("*" * 80)
    print("\nTo apply these parameters, update the constants in config.py.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
