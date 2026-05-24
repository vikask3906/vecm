# run_phase3_demo.py
"""
Phase 3 Driver — Dynamic Hedging & Risk Engine Demo

Demonstrates the risk engine by:
1. Running Phase 1 to get VECM beta vector
2. Optimizing hedge weights via CVXPY
3. Simulating liquidity events (halt, HTB)
4. Showing rebalancer response (re-optimization with exclusions)
"""

import numpy as np
from core.data_loader import fetch_prices, ASSETS
from core.johansen import run_johansen
from core.vecm import fit_vecm
from risk.hedge_optimizer import optimize_hedge, print_hedge_summary
from risk.liquidity_monitor import LiquidityMonitor
from risk.position_manager import PositionManager
from risk.rebalancer import Rebalancer
import config


def main():
    print("=" * 60)
    print("  VECM ARBITRAGE SYSTEM — PHASE 3: RISK ENGINE DEMO")
    print("=" * 60)

    # ─── Step 1: Get VECM beta from Phase 1 ───
    print("\n[1/4] Running Phase 1 pipeline...")
    prices = fetch_prices(ASSETS, start="2020-01-01", end="2024-01-01")
    jr = run_johansen(prices, det_order=config.JOHANSEN_DET_ORDER,
                      k_ar_diff=config.JOHANSEN_K_AR_DIFF)

    if jr.rank == 0:
        print("  No cointegration found. Exiting.")
        return

    vr = fit_vecm(prices, rank=jr.rank, k_ar_diff=config.JOHANSEN_K_AR_DIFF)
    asset_names = list(prices.columns)
    print(f"  Rank: {jr.rank}, Half-life: {vr.half_lives[0]:.1f} days")

    # ─── Step 2: Optimize hedge weights (unconstrained) ───
    print("\n[2/4] Optimizing hedge weights (no exclusions)...")
    hr_full = optimize_hedge(vr.beta, vec_idx=0)
    print_hedge_summary(hr_full, asset_names)

    # ─── Step 3: Simulate liquidity events ───
    print("[3/4] Simulating liquidity events...")
    liq_monitor = LiquidityMonitor(tickers=asset_names)

    # Simulate: SLB gets halted, MRO becomes hard-to-borrow
    liq_monitor.set_halt("SLB", True)
    liq_monitor.set_hard_to_borrow("MRO", True)

    # Add some volume data
    for ticker in asset_names:
        liq_monitor.update_volume(ticker, 2_000_000)
    liq_monitor.update_volume("DVN", 100_000)  # Low volume

    events = liq_monitor.check_all()
    print(f"\n  Detected {len(events)} liquidity events:")
    for e in events:
        print(f"    [{e.severity.value:>8}] {e.message}")

    # ─── Step 4: Reoptimize with exclusions ───
    print("\n[4/4] Re-optimizing with liquidity exclusions...")
    excluded = liq_monitor.get_excluded_indices(asset_names)
    excluded_names = [asset_names[i] for i in excluded]
    print(f"  Excluding: {excluded_names}")

    hr_constrained = optimize_hedge(vr.beta, vec_idx=0,
                                     excluded_indices=excluded)
    print_hedge_summary(hr_constrained, asset_names)

    # Show the portfolio impact
    pm = PositionManager(capital=1_000_000)
    # Open positions based on optimized weights
    latest_prices = np.exp(prices.iloc[-1])  # Convert back from log prices
    for i, ticker in enumerate(asset_names):
        weight = hr_constrained.weights[i]
        if abs(weight) > 0.001:
            notional = weight * pm.initial_capital
            qty = notional / latest_prices.iloc[i]
            pm.open_position(ticker, qty, latest_prices.iloc[i])

    # Update with current prices
    for i, ticker in enumerate(asset_names):
        pm.update_price(ticker, latest_prices.iloc[i])

    print(pm.summary())

    print(f"{'='*60}")
    print(f"  PHASE 3 DEMO COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
