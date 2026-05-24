# scan_triads.py
"""
Phase 6 Triad Cointegration Scanner

Generates all 3-asset combinations from the Energy Sector universe,
verifies Johansen cointegration, calculates spread half-lives,
runs full backtests on each triad in parallel, and ranks them
by Sharpe ratio and return.
"""

import sys
import os
import time
import warnings
from itertools import combinations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ignore warnings to keep terminal output clean
warnings.filterwarnings("ignore")

# Import systems architecture
from core.data_loader import fetch_prices_with_volume
from core.johansen import run_johansen
from core.vecm import fit_vecm
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.tearsheet import compute_tearsheet
import config

# Triad scanning range
START_DATE = "2016-01-01"
END_DATE = "2024-01-01"

@dataclass
class TriadResult:
    assets: list[str]
    rank: int
    half_life: float
    total_return: float
    ann_return: float
    sharpe: float
    max_dd: float
    num_trades: int
    win_rate: float
    pct_time_invested: float
    total_costs: float

def evaluate_single_triad(assets, log_prices_full, volume_full):
    """
    Worker function to evaluate cointegration and backtest a single 3-asset triad.
    """
    try:
        # Extract subset of assets
        log_prices = log_prices_full[list(assets)]
        volume = volume_full[list(assets)]

        # ─── 1. Run Johansen Test ───
        jr = run_johansen(
            log_prices,
            det_order=config.JOHANSEN_DET_ORDER,
            k_ar_diff=config.JOHANSEN_K_AR_DIFF,
            confidence=config.JOHANSEN_CONFIDENCE,
        )

        # Skip if no cointegration rank detected on full period
        if jr.rank == 0:
            return TriadResult(
                assets=list(assets),
                rank=0,
                half_life=np.nan,
                total_return=0.0,
                ann_return=0.0,
                sharpe=-9.99,
                max_dd=0.0,
                num_trades=0,
                win_rate=0.0,
                pct_time_invested=0.0,
                total_costs=0.0
            )

        # ─── 2. Fit VECM & Extract Half-life ───
        vr = fit_vecm(
            log_prices,
            rank=jr.rank,
            k_ar_diff=config.JOHANSEN_K_AR_DIFF,
        )
        hl = vr.half_lives[0] if len(vr.half_lives) > 0 else np.nan

        # ─── 3. Run Backtest with Swept Parameters ───
        # Using Entry=2.0, Exit=0.2, Cooldown=15
        bt_config = BacktestConfig(
            initial_capital=config.INITIAL_CAPITAL,
            entry_zscore=2.0,
            exit_zscore=0.2,
            stop_loss_zscore=config.STOP_LOSS_ZSCORE,
            zscore_window=config.ZSCORE_WINDOW,
            reestimation_freq=63,
            warmup_period=252,
            enable_impact=True,
            enable_breakdown=True,
            monitor_window=config.MONITOR_WINDOW,
            monitor_recheck_freq=config.MONITOR_RECHECK_FREQ,
            monitor_grace_period=config.MONITOR_GRACE_PERIOD,
            monitor_cooldown=15, # Optimized safety cooldown
        )

        engine = BacktestEngine(log_prices, volume, bt_config)
        results = engine.run()
        metrics = compute_tearsheet(results)

        return TriadResult(
            assets=list(assets),
            rank=jr.rank,
            half_life=hl,
            total_return=results.total_return_pct,
            ann_return=metrics.annualized_return_pct,
            sharpe=metrics.sharpe_ratio,
            max_dd=metrics.max_drawdown_pct,
            num_trades=metrics.num_trades,
            win_rate=metrics.win_rate,
            pct_time_invested=metrics.pct_time_invested,
            total_costs=results.total_costs
        )

    except Exception as e:
        print(f"\nError processing triad {assets}: {e}")
        return None

def main():
    print("=" * 80)
    print("  VECM ARBITRAGE SYSTEM — MULTI-ASSET TRIAD COINTEGRATION SCANNER")
    print("=" * 80)

    # ─── 1. Fetch entire historical dataset ───
    print(f"\n[1/3] Fetching adjusting close & volumes for universe...")
    log_prices, volume = fetch_prices_with_volume(
        config.ASSETS, start=START_DATE, end=END_DATE
    )
    tickers = list(log_prices.columns)
    print(f"  Complete Asset Pool: {tickers}")
    print(f"  Trading Days:        {len(log_prices)} ({START_DATE} -> {END_DATE})")

    # ─── 2. Generate Triad combinations ───
    triads = list(combinations(tickers, 3))
    n_triads = len(triads)
    print(f"\n[2/3] Generated {n_triads} unique 3-asset combinations.")
    print("  Starting parallel evaluation (cointegration check + full backtest)...")

    start_time = time.time()
    results = []

    max_workers = min(os.cpu_count() or 4, 8)
    print(f"  Running on {max_workers} CPU cores...")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all combinations
        futures = {
            executor.submit(evaluate_single_triad, triad, log_prices, volume): triad
            for triad in triads
        }

        # Collect results dynamically
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res is not None:
                results.append(res)
                # Print progress update
                sys.stdout.write(
                    f"\r  Progress: {completed}/{n_triads} triads evaluated... "
                    f"Latest Triad: {res.assets} | Sharpe: {res.sharpe:>+5.2f}"
                )
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n\n[3/3] Scanning completed in {elapsed:.1f} seconds.")

    # ─── 3. Analyze and Compile Leadboard ───
    df_results = pd.DataFrame([res.__dict__ for res in results])
    
    # Save raw results
    csv_filename = "triad_scan_results.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"  Saved full diagnostics ({len(df_results)} triads) to: {csv_filename}")

    # Cointegrated subsets
    df_coint = df_results[df_results["rank"] > 0]
    n_coint = len(df_coint)
    print(f"  Cointegrated Triads detected: {n_coint} / {n_triads} ({n_coint/n_triads*100:.1f}%)")

    if df_coint.empty:
        print("\n⚠ No cointegrated triads found over the period. Consider expanding the universe.")
        return

    # Sort by Sharpe Ratio (Highest to Lowest)
    df_sorted = df_coint.sort_values(by="sharpe", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 105)
    print(f"                                   TOP 10 TRIAD LEADERBOARD (BY SHARPE RATIO)")
    print("=" * 105)
    print(f"{'Rank':<5} {'Triad Assets':<28} {'Rank':<6} {'Half-Life':<10} {'Tot Ret':<10} {'Ann Ret':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Win%':<6} {'Invest%':<8}")
    print("-" * 115)
    for idx, row in df_sorted.head(10).iterrows():
        assets_str = f"[{', '.join(row['assets'])}]"
        print(
            f"{idx+1:<5} "
            f"{assets_str:<28} "
            f"{int(row['rank']):<6d} "
            f"{row['half_life']:>8.1f}d "
            f"{row['total_return']:>+8.2f}% "
            f"{row['ann_return']:>+8.2f}% "
            f"{row['sharpe']:>8.2f} "
            f"{row['max_dd']:>7.2f}% "
            f"{int(row['num_trades']):>8d} "
            f"{row['win_rate']:>5.1f}% "
            f"{row['pct_time_invested']:>7.1f}%"
        )
    print("=" * 115)

    # ─── 4. Recommendation ───
    best = df_sorted.iloc[0]
    print("\n" + "*" * 90)
    print("  OPTIMAL ALPHA MULTIPLIER RECOMMENDATION")
    print("*" * 90)
    print(f"  The #1 Cointegrated Triad discovered is: {best['assets']}")
    print(f"    - Cointegration Rank:  {int(best['rank'])}")
    print(f"    - Spread Half-Life:    {best['half_life']:.1f} trading days (highly tradeable)")
    print(f"\n  Projected Backtest Performance (2016-2023):")
    print(f"    - Total Return:        {best['total_return']:>+7.2f}%  (7-Asset Baseline: +7.12%)")
    print(f"    - Annualized Return:   {best['ann_return']:>+7.2f}%  (7-Asset Baseline: +0.99%)")
    print(f"    - Sharpe Ratio (Rf=5%):{best['sharpe']:>7.2f}  (7-Asset Baseline: -0.83)")
    print(f"    - Max Drawdown:        {best['max_dd']:>7.2f}%  (7-Asset Baseline: 13.69%)")
    print(f"    - Number of Trades:    {int(best['num_trades'])} trades  (7-Asset Baseline: 27)")
    print(f"    - % Time Invested:     {best['pct_time_invested']:>6.1f}%  (7-Asset Baseline: 28.3%)")
    print(f"    - Total Costs:         ${best['total_costs']:,.2f}  (7-Asset Baseline: $21,384.28)")
    print("*" * 90)
    print("\n  NEXT STEPS:")
    print(f"  1. Update config.py with: config.ASSETS = {best['assets']}")
    print("  2. Run the main backtest to visualize this triad's tearsheet and performance.")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
