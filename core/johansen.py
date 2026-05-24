# core/johansen.py
"""
Johansen cointegration test, rank selection, and eigenvector extraction.

The Johansen procedure tests for cointegration in a multivariate system
by reformulating a VAR(p) model as a VECM and solving an eigenvalue problem
on residual cross-moment matrices.

Key outputs:
    - Cointegration rank r (number of independent stationary linear combinations)
    - Beta matrix (k×r) — cointegrating vectors = hedge ratios
    - Eigenvalues — strength of each cointegrating relationship
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass


@dataclass
class JohansenResult:
    """Results from the Johansen cointegration test."""
    rank: int                    # cointegration rank r
    beta: np.ndarray             # shape (k, r) — hedge ratio matrix
    alpha: np.ndarray            # shape (k, r) — speed of adjustment
    eigenvalues: np.ndarray      # all eigenvalues from the procedure
    trace_stats: np.ndarray      # trace test statistics
    trace_cvals: np.ndarray      # critical values at 90/95/99%
    max_eig_stats: np.ndarray    # max-eigenvalue test statistics
    max_eig_cvals: np.ndarray    # max-eigenvalue critical values


def check_stationarity(prices: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    ADF test on each series in levels and first differences.

    For cointegration analysis, we need all series to be I(1):
    non-stationary in levels, stationary in first differences.

    Parameters
    ----------
    prices : pd.DataFrame
        Log price series (each column = one asset).
    alpha : float
        Significance level for the ADF test.

    Returns
    -------
    dict
        Per-asset results with level/diff p-values and I(1) classification.
    """
    results = {}
    for col in prices.columns:
        adf_level = adfuller(prices[col], autolag="AIC")
        adf_diff = adfuller(prices[col].diff().dropna(), autolag="AIC")
        results[col] = {
            "level_pvalue": adf_level[1],
            "diff_pvalue": adf_diff[1],
            "is_I1": (adf_level[1] > alpha) and (adf_diff[1] < alpha)
        }
    return results


def run_johansen(prices: pd.DataFrame, det_order: int = 0,
                 k_ar_diff: int = 1, confidence: int = 1) -> JohansenResult:
    """
    Run the Johansen cointegration test.

    Parameters
    ----------
    prices : pd.DataFrame
        Log price series for k assets.
    det_order : int
        Deterministic term specification:
            -1 = no constant/trend
             0 = constant in cointegrating relation (most common for prices)
             1 = constant + linear trend
    k_ar_diff : int
        Number of lagged differences in the VECM (= VAR lag order - 1).
        Use information criteria (AIC/BIC) on a VAR to choose optimally.
    confidence : int
        Critical value column: 0=90%, 1=95%, 2=99%.

    Returns
    -------
    JohansenResult
        Contains rank, beta (hedge ratios), alpha, eigenvalues, test statistics.
    """
    result = coint_johansen(prices.values, det_order=det_order, k_ar_diff=k_ar_diff)

    # Determine rank r using trace statistic at the specified confidence level
    # result.cvt columns: [90%, 95%, 99%]
    rank = 0
    for i in range(len(result.lr1)):          # lr1 = trace statistics
        if result.lr1[i] > result.cvt[i, confidence]:
            rank += 1
        else:
            break

    # Eigenvectors (columns of evec) are the cointegrating vectors = beta
    # statsmodels returns them as shape (k, k); we take first r columns
    beta = result.evec[:, :max(rank, 1)]     # (k, r) — hedge ratios; keep at least 1
    alpha = result.eig[:max(rank, 1)]        # raw eigenvalues

    return JohansenResult(
        rank=rank,
        beta=beta,
        alpha=alpha,
        eigenvalues=result.eig,
        trace_stats=result.lr1,
        trace_cvals=result.cvt,
        max_eig_stats=result.lr2,
        max_eig_cvals=result.cvm,
    )


def interpret_results(jr: JohansenResult, asset_names: list[str]) -> None:
    """
    Print human-readable interpretation of the Johansen test results.

    Parameters
    ----------
    jr : JohansenResult
        Output from run_johansen().
    asset_names : list[str]
        Ticker symbols corresponding to columns of the price matrix.
    """
    print(f"\n{'='*50}")
    print(f"  JOHANSEN COINTEGRATION TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Cointegration rank r = {jr.rank}")

    print(f"\n  Trace statistics vs 95% critical values:")
    print(f"  {'H0':<15} {'Stat':>10} {'CV(95%)':>10} {'Result':>10}")
    print(f"  {'-'*45}")
    for i, (stat, cv) in enumerate(zip(jr.trace_stats, jr.trace_cvals[:, 1])):
        sig = "REJECT" if stat > cv else "accept"
        print(f"  rank <= {i:<6} {stat:>10.2f} {cv:>10.2f} {sig:>10}")

    print(f"\n  Max-eigenvalue statistics vs 95% critical values:")
    print(f"  {'H0':<15} {'Stat':>10} {'CV(95%)':>10} {'Result':>10}")
    print(f"  {'-'*45}")
    for i, (stat, cv) in enumerate(zip(jr.max_eig_stats, jr.max_eig_cvals[:, 1])):
        sig = "REJECT" if stat > cv else "accept"
        print(f"  rank = {i:<7} {stat:>10.2f} {cv:>10.2f} {sig:>10}")

    if jr.rank > 0:
        print(f"\n  Cointegrating vector(s) — normalized hedge ratios:")
        for vec_idx in range(jr.rank):
            beta_norm = jr.beta[:, vec_idx] / jr.beta[0, vec_idx]
            print(f"\n  Vector {vec_idx + 1} (normalized to {asset_names[0]} = 1.0):")
            for asset, w in zip(asset_names, beta_norm):
                print(f"    {asset:6s}: {w:+.6f}")

        print(f"\n  Eigenvalues: {jr.eigenvalues[:jr.rank]}")
    else:
        print("\n  ⚠ No cointegration detected at 95% confidence.")
    print(f"{'='*50}\n")
