# run_phase1.py
"""
Phase 1 Driver — Core Math Engine

Runs the full Johansen → VECM → Spread → Signal pipeline on the
energy sector equity basket.
"""

from core.data_loader import fetch_prices, ASSETS
from core.johansen import check_stationarity, run_johansen, interpret_results
from core.vecm import fit_vecm, print_vecm_summary
from core.spread import construct_spread, compute_zscore, generate_signals, compute_signal_stats
import config


def main():
    print("=" * 60)
    print("  VECM ARBITRAGE SYSTEM — PHASE 1: CORE MATH ENGINE")
    print("=" * 60)

    # ─── Step 1: Fetch prices ───
    print("\n[1/5] Fetching prices...")
    prices = fetch_prices(ASSETS, start="2018-01-01", end="2024-01-01")
    print(f"  Shape: {prices.shape}")
    print(f"  Assets: {list(prices.columns)}")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # ─── Step 2: Stationarity check ───
    print("\n[2/5] Checking stationarity (ADF test)...")
    stat_results = check_stationarity(prices, alpha=config.ADF_ALPHA)
    print(f"  {'Asset':<8} {'Level p':>10} {'Diff p':>10} {'Status':>12}")
    print(f"  {'-'*42}")
    for asset, res in stat_results.items():
        tag = "I(1) ✓" if res["is_I1"] else "NOT I(1) ⚠"
        print(f"  {asset:<8} {res['level_pvalue']:>10.4f} {res['diff_pvalue']:>10.4f} {tag:>12}")

    # Filter to I(1) assets only
    i1_assets = [a for a, r in stat_results.items() if r["is_I1"]]
    non_i1 = [a for a in stat_results if a not in i1_assets]
    if non_i1:
        print(f"\n  ⚠ Excluding non-I(1) assets: {non_i1}")
    prices_i1 = prices[i1_assets]
    print(f"\n  Proceeding with {len(i1_assets)} I(1) assets: {i1_assets}")

    # ─── Step 3: Johansen test ───
    print(f"\n[3/5] Running Johansen cointegration test...")
    print(f"  det_order={config.JOHANSEN_DET_ORDER}, k_ar_diff={config.JOHANSEN_K_AR_DIFF}")
    jr = run_johansen(prices_i1, det_order=config.JOHANSEN_DET_ORDER,
                      k_ar_diff=config.JOHANSEN_K_AR_DIFF)
    interpret_results(jr, i1_assets)

    if jr.rank == 0:
        print("\n  ⚠ No cointegration found. Try different assets or time period.")
        return

    # ─── Step 4: VECM estimation ───
    print(f"[4/5] Fitting VECM (rank={jr.rank})...")
    vr = fit_vecm(prices_i1, rank=jr.rank, k_ar_diff=config.JOHANSEN_K_AR_DIFF)
    print_vecm_summary(vr, i1_assets)

    # Half-life sanity check
    hl = vr.half_lives[0]
    if hl < config.MIN_HALF_LIFE:
        print(f"  ⚠ Half-life ({hl:.1f}d) is below {config.MIN_HALF_LIFE}d — may be noise.")
    elif hl > config.MAX_HALF_LIFE:
        print(f"  ⚠ Half-life ({hl:.1f}d) exceeds {config.MAX_HALF_LIFE}d — may be too slow.")
    else:
        print(f"  ✓ Half-life ({hl:.1f}d) is in the tradeable range [{config.MIN_HALF_LIFE}, {config.MAX_HALF_LIFE}].")

    # ─── Step 5: Spread & signals ───
    print(f"\n[5/5] Constructing spread and generating signals...")
    spread = construct_spread(prices_i1, vr.beta, vec_idx=0)
    zscore = compute_zscore(spread, window=config.ZSCORE_WINDOW)
    signals = generate_signals(zscore, entry_z=config.ENTRY_ZSCORE,
                               exit_z=config.EXIT_ZSCORE,
                               stop_loss_z=config.STOP_LOSS_ZSCORE)

    stats = compute_signal_stats(signals, zscore)
    print(f"\n  Signal Statistics:")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'Total trading days':<25} {stats['total_days']:>10}")
    print(f"  {'Long days':<25} {stats['long_days']:>10}")
    print(f"  {'Short days':<25} {stats['short_days']:>10}")
    print(f"  {'Flat days':<25} {stats['flat_days']:>10}")
    print(f"  {'Active %':<25} {stats['active_pct']:>9.1f}%")
    print(f"  {'Number of trades':<25} {stats['num_trades']:>10}")
    print(f"  {'Avg trade duration':<25} {stats['avg_trade_duration']:>8.1f}d")
    print(f"  {'Max z-score':<25} {stats['max_zscore']:>10.2f}")
    print(f"  {'Min z-score':<25} {stats['min_zscore']:>10.2f}")

    print(f"\n{'='*60}")
    print(f"  PHASE 1 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
