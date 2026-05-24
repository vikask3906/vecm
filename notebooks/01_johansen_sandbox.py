#!/usr/bin/env python3
"""
01_johansen_sandbox.py — Offline Math Verification (Strategy Domain)

This script is the sandbox equivalent of a Jupyter notebook.
It pulls historical data for the energy sector basket and runs
the full Johansen → VECM → Spread → Signal pipeline offline
to verify the math BEFORE trusting any infrastructure.

Domain: Strategy (Pure Math)
Machine Learning: NONE
Dev: Minimal (just yfinance fetch)

Run:
    python notebooks/01_johansen_sandbox.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

from core.data_loader import fetch_prices, ASSETS
from core.johansen import check_stationarity, run_johansen, interpret_results
from core.vecm import fit_vecm, print_vecm_summary
from core.spread import construct_spread, compute_zscore, generate_signals, compute_signal_stats
import config


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


def main():
    # ─────────────────────────────────────────────
    # 1. DATA ACQUISITION
    # ─────────────────────────────────────────────
    section("STEP 1: Fetching Historical Data")

    start_date = "2020-01-01"
    end_date = "2025-12-31"
    tickers = config.ASSETS

    print(f"  Basket: {tickers}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Fetching log prices from yfinance...")

    log_prices = fetch_prices(tickers, start=start_date, end=end_date)
    print(f"  ✓ Received {len(log_prices)} trading days × {log_prices.shape[1]} assets")
    print(f"  Date range: {log_prices.index[0].date()} → {log_prices.index[-1].date()}")
    print(f"\n  Log price sample (first 5 rows):")
    print(log_prices.head().to_string(float_format=lambda x: f"{x:.4f}"))

    # ─────────────────────────────────────────────
    # 2. ADF STATIONARITY TESTS
    # ─────────────────────────────────────────────
    section("STEP 2: ADF Stationarity Tests (Confirm I(1))")
    print("  For cointegration to work, all series must be I(1):")
    print("    → Non-stationary in LEVELS (p > 0.05)")
    print("    → Stationary in FIRST DIFFERENCES (p < 0.05)\n")

    stationarity = check_stationarity(log_prices, alpha=config.ADF_ALPHA)

    all_i1 = True
    print(f"  {'Asset':<8} {'Level p-val':>12} {'Diff p-val':>12} {'I(1)?':>8}")
    print(f"  {'-'*44}")
    for asset, result in stationarity.items():
        status = "✓ YES" if result["is_I1"] else "✗ NO"
        if not result["is_I1"]:
            all_i1 = False
        print(f"  {asset:<8} {result['level_pvalue']:>12.6f} {result['diff_pvalue']:>12.6f} {status:>8}")

    if all_i1:
        print(f"\n  ✓ ALL {len(tickers)} assets are I(1). Johansen test is valid.")
    else:
        print(f"\n  ⚠ WARNING: Not all assets are I(1). Results may be unreliable.")

    # ─────────────────────────────────────────────
    # 3. JOHANSEN COINTEGRATION TEST
    # ─────────────────────────────────────────────
    section("STEP 3: Johansen Cointegration Test")

    jr = run_johansen(
        log_prices,
        det_order=config.JOHANSEN_DET_ORDER,
        k_ar_diff=config.JOHANSEN_K_AR_DIFF,
        confidence=config.JOHANSEN_CONFIDENCE,
    )
    interpret_results(jr, tickers)

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  VERDICT: Cointegration Rank r = {jr.rank}")
    print(f"  ═══════════════════════════════════════════")

    if jr.rank == 0:
        print("  ⚠ No cointegration found. The strategy has no edge on this basket/period.")
        print("  Consider: different assets, longer lookback, or alternative det_order.")
        return

    # ─────────────────────────────────────────────
    # 4. VECM ESTIMATION (Half-Life & Adjustment Speeds)
    # ─────────────────────────────────────────────
    section("STEP 4: VECM Maximum Likelihood Estimation")

    vr = fit_vecm(
        log_prices,
        rank=jr.rank,
        k_ar_diff=config.JOHANSEN_K_AR_DIFF,
    )
    print_vecm_summary(vr, tickers)

    # Half-life interpretation
    for i, hl in enumerate(vr.half_lives):
        if np.isnan(hl):
            print(f"  Spread {i+1}: Half-life = NaN (non-stationary root)")
        elif hl < config.MIN_HALF_LIFE:
            print(f"  Spread {i+1}: Half-life = {hl:.1f}d → ⚠ TOO FAST (likely noise)")
        elif hl > config.MAX_HALF_LIFE:
            print(f"  Spread {i+1}: Half-life = {hl:.1f}d → ⚠ TOO SLOW (unprofitable after costs)")
        else:
            print(f"  Spread {i+1}: Half-life = {hl:.1f}d → ✓ TRADEABLE SWEET SPOT")

    # ─────────────────────────────────────────────
    # 5. SPREAD CONSTRUCTION & SIGNAL GENERATION
    # ─────────────────────────────────────────────
    section("STEP 5: Spread Construction & Z-Score Signals")

    spread = construct_spread(log_prices, vr.beta, vec_idx=0)
    zscore = compute_zscore(spread, window=config.ZSCORE_WINDOW)
    signals = generate_signals(
        zscore,
        entry_z=config.ENTRY_ZSCORE,
        exit_z=config.EXIT_ZSCORE,
        stop_loss_z=config.STOP_LOSS_ZSCORE,
    )

    stats = compute_signal_stats(signals, zscore)
    print(f"\n  Signal Summary:")
    print(f"    Total trading days:     {stats['total_days']}")
    print(f"    Active (in-trade) days: {stats['long_days'] + stats['short_days']} ({stats['active_pct']:.1f}%)")
    print(f"    Long days:              {stats['long_days']}")
    print(f"    Short days:             {stats['short_days']}")
    print(f"    Flat days:              {stats['flat_days']}")
    print(f"    Number of trades:       {stats['num_trades']}")
    print(f"    Avg trade duration:     {stats['avg_trade_duration']:.1f} days")
    print(f"    Max z-score:            {stats['max_zscore']:.2f}")
    print(f"    Min z-score:            {stats['min_zscore']:.2f}")

    # ─────────────────────────────────────────────
    # 6. VISUALIZATION
    # ─────────────────────────────────────────────
    section("STEP 6: Generating Diagnostic Plots")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("VECM Arbitrage — Sandbox Math Verification", fontsize=14, fontweight="bold")

    # Panel 1: Log prices (normalized to start = 0)
    norm_prices = log_prices - log_prices.iloc[0]
    norm_prices.plot(ax=axes[0], alpha=0.7, linewidth=0.8)
    axes[0].set_title("Normalized Log Prices (relative to start)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_ylabel("Log return from start")

    # Panel 2: Cointegrating spread
    axes[1].plot(spread.index, spread.values, color="navy", linewidth=0.8)
    axes[1].axhline(spread.mean(), color="gray", linestyle="--", alpha=0.5, label="Mean")
    axes[1].set_title(f"Cointegrating Spread (β'Pₜ)")
    axes[1].set_ylabel("Spread value")
    axes[1].legend(fontsize=8)

    # Panel 3: Z-score with entry/exit thresholds
    axes[2].plot(zscore.index, zscore.values, color="darkblue", linewidth=0.7)
    axes[2].axhline(config.ENTRY_ZSCORE, color="red", linestyle="--", alpha=0.7, label=f"Entry ±{config.ENTRY_ZSCORE}")
    axes[2].axhline(-config.ENTRY_ZSCORE, color="red", linestyle="--", alpha=0.7)
    axes[2].axhline(config.EXIT_ZSCORE, color="green", linestyle=":", alpha=0.5, label=f"Exit ±{config.EXIT_ZSCORE}")
    axes[2].axhline(-config.EXIT_ZSCORE, color="green", linestyle=":", alpha=0.5)
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].fill_between(zscore.index, config.ENTRY_ZSCORE, config.STOP_LOSS_ZSCORE,
                         alpha=0.05, color="red")
    axes[2].fill_between(zscore.index, -config.ENTRY_ZSCORE, -config.STOP_LOSS_ZSCORE,
                         alpha=0.05, color="red")
    axes[2].set_title("Z-Score with Entry/Exit Thresholds")
    axes[2].set_ylabel("Z-score")
    axes[2].legend(fontsize=8)

    # Panel 4: Trading signals
    axes[3].fill_between(signals.index, signals.values, 0, alpha=0.4,
                         where=signals > 0, color="green", label="Long")
    axes[3].fill_between(signals.index, signals.values, 0, alpha=0.4,
                         where=signals < 0, color="red", label="Short")
    axes[3].set_title("Trading Signals (+1 Long / -1 Short / 0 Flat)")
    axes[3].set_ylabel("Signal")
    axes[3].set_xlabel("Date")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "01_sandbox_diagnostic.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved diagnostic plot to: {output_path}")
    plt.close()

    # ─────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────
    section("SANDBOX VERDICT")
    print(f"  Cointegration rank:  r = {jr.rank}")
    print(f"  Half-life:           {vr.half_lives[0]:.1f} trading days")
    print(f"  All assets I(1):     {'YES' if all_i1 else 'NO'}")
    print(f"  Active trade ratio:  {stats['active_pct']:.1f}%")
    print(f"  Number of trades:    {stats['num_trades']}")

    tradeable = (
        jr.rank >= 1
        and all_i1
        and config.MIN_HALF_LIFE <= vr.half_lives[0] <= config.MAX_HALF_LIFE
    )

    if tradeable:
        print(f"\n  ✓ MATH VERIFIED: This basket has a tradeable cointegrating relationship.")
        print(f"    The infrastructure (Phases 2-5) can be trusted for this basket.")
    else:
        print(f"\n  ✗ MATH INCONCLUSIVE: Review the diagnostics above before proceeding.")

    print(f"\n{'━'*60}\n")


if __name__ == "__main__":
    main()
