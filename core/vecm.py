# core/vecm.py
"""
Vector Error Correction Model (VECM) estimation.

Given a cointegration rank r from the Johansen test, this module fits the
full VECM via maximum likelihood to obtain:
    - β (beta): cointegrating vectors = hedge ratios
    - α (alpha): speed-of-adjustment coefficients (error correction)
    - Half-life of mean reversion (from companion matrix eigenvalues)
    - Γ (gamma): short-run dynamics coefficients

The VECM representation:
    ΔPₜ = αβ'Pₜ₋₁ + Γ₁ΔPₜ₋₁ + ... + Γₚ₋₁ΔPₜ₋ₚ₊₁ + εₜ

where Π = αβ' is the long-run impact matrix with rank r.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
from dataclasses import dataclass


@dataclass
class VECMResult:
    """Results from VECM maximum likelihood estimation."""
    beta: np.ndarray        # (k, r) cointegrating vectors (hedge ratios)
    alpha: np.ndarray       # (k, r) adjustment coefficients
    half_lives: np.ndarray  # (r,) half-lives in trading days
    gamma: np.ndarray       # short-run dynamics coefficients
    residuals: np.ndarray   # model residuals for diagnostic tests
    sigma_u: np.ndarray     # residual covariance matrix
    aic: float              # Akaike information criterion
    bic: float              # Bayesian information criterion


def fit_vecm(prices: pd.DataFrame, rank: int, k_ar_diff: int = 1,
             det_order: str = "ci") -> VECMResult:
    """
    Fit full VECM via maximum likelihood estimation.

    This gives proper MLE estimates of alpha (speed of adjustment), which
    is more reliable than Johansen eigenvalues alone for position sizing
    and half-life estimation.

    Parameters
    ----------
    prices : pd.DataFrame
        Log price series for k assets.
    rank : int
        Cointegration rank r (from Johansen test).
    k_ar_diff : int
        Number of lagged differences (VAR lag order - 1).
    det_order : str
        Deterministic term specification:
            "ci"  = constant in cointegrating relation (Johansen det_order=0)
            "n"   = no deterministic terms
            "lo"  = constant outside cointegrating relation
            "li"  = constant + linear trend inside cointegrating relation
            "coli"= constant outside + linear trend inside

    Returns
    -------
    VECMResult
        Contains beta, alpha, half_lives, gamma, residuals, and model fit stats.
    """
    k = prices.shape[1]

    model = VECM(prices.values, k_ar_diff=k_ar_diff, coint_rank=rank,
                 deterministic=det_order)
    fitted = model.fit(method="ml")

    # α (alpha): (k, r) — adjustment coefficients for error correction
    # Each column corresponds to one cointegrating relationship
    alpha = fitted.alpha                 # shape (k, r)

    # β (beta): (k + det_terms, r) — first k rows are the price betas
    beta_full = fitted.beta
    beta = beta_full[:k, :]              # shape (k, r) — hedge ratios only

    # Γ (gamma): short-run dynamics coefficients
    gamma = fitted.gamma

    # Residuals and covariance
    residuals = fitted.resid
    sigma_u = fitted.sigma_u

    # Information criteria
    aic = getattr(fitted, 'aic', float('nan'))
    bic = getattr(fitted, 'bic', float('nan'))

    # ─── Half-life computation ───
    # For the multivariate case, we compute the half-life from the
    # companion matrix eigenvalues: companion = I + α @ β'
    # The dominant eigenvalue determines the system's slowest decay rate.
    companion = np.eye(k) + alpha @ beta.T
    eigvals = np.linalg.eigvals(companion)

    # Take eigenvalues inside the unit circle (stationary roots)
    stationary_eigs = eigvals[np.abs(eigvals) < 1.0]

    if len(stationary_eigs) > 0:
        # The dominant (largest absolute value) stationary eigenvalue
        # determines the slowest mean-reversion speed
        dominant = stationary_eigs[np.argmax(np.abs(stationary_eigs))]
        half_life = np.array([-np.log(2) / np.log(np.abs(dominant))])
    else:
        half_life = np.array([np.nan])

    # Also compute per-spread half-lives using univariate approximation
    # HL_i = -log(2) / log(1 + α_ii) for each cointegrating vector
    per_spread_hl = []
    for r_idx in range(rank):
        # Effective speed for this spread: α[:, r_idx]' @ β[:, r_idx]
        speed = alpha[:, r_idx].T @ beta[:, r_idx]
        if speed < 0:
            hl = -np.log(2) / np.log(1 + speed)
            per_spread_hl.append(hl)
        else:
            per_spread_hl.append(np.nan)

    half_lives = np.array(per_spread_hl) if per_spread_hl else half_life

    return VECMResult(
        beta=beta,
        alpha=alpha,
        half_lives=half_lives,
        gamma=gamma,
        residuals=residuals,
        sigma_u=sigma_u,
        aic=aic,
        bic=bic,
    )


def print_vecm_summary(vr: VECMResult, asset_names: list[str]) -> None:
    """
    Print a formatted summary of the VECM estimation results.

    Parameters
    ----------
    vr : VECMResult
        Output from fit_vecm().
    asset_names : list[str]
        Ticker symbols corresponding to the assets.
    """
    k, r = vr.alpha.shape

    print(f"\n{'='*50}")
    print(f"  VECM ESTIMATION RESULTS")
    print(f"{'='*50}")
    print(f"  Assets:    {k}")
    print(f"  Rank:      {r}")
    print(f"  AIC:       {vr.aic:.2f}")
    print(f"  BIC:       {vr.bic:.2f}")

    for vec_idx in range(r):
        print(f"\n  ── Cointegrating Relation {vec_idx + 1} ──")
        print(f"  Half-life: {vr.half_lives[vec_idx]:.1f} trading days")

        # Normalized beta
        beta_norm = vr.beta[:, vec_idx] / vr.beta[0, vec_idx]
        print(f"\n  Beta (normalized to {asset_names[0]} = 1.0):")
        for name, b in zip(asset_names, beta_norm):
            print(f"    {name:6s}: {b:+.6f}")

        print(f"\n  Alpha (adjustment speeds):")
        for name, a in zip(asset_names, vr.alpha[:, vec_idx]):
            direction = "correcting" if a < 0 else "DIVERGING ⚠"
            print(f"    {name:6s}: {a:+.6f}  ({direction})")

    print(f"{'='*50}\n")
