# risk/hedge_optimizer.py
"""
CVXPY-based convex optimizer for hedge weights.

Given a target cointegrating vector β and a set of constraints
(dollar-neutral, position limits, exposure caps), this module solves
for the optimal portfolio weights that track β as closely as possible
while respecting risk and liquidity constraints.

The optimization problem:
    minimize    ||w - β_norm||²  +  λ * ||w||²
    subject to  Σ wᵢ = 0          (dollar-neutral)
                |wᵢ| ≤ w_max       (position limits)
                Σ |wᵢ| ≤ L_max     (gross leverage cap)
                wᵢ = 0  ∀ i ∈ excluded_assets  (liquidity exclusions)
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class HedgeResult:
    """Results from the hedge optimization."""
    weights: np.ndarray          # (k,) optimal weights
    target_weights: np.ndarray   # (k,) target beta-normalized weights
    tracking_error: float        # ||w - β_norm||²
    gross_exposure: float        # Σ|wᵢ|
    net_exposure: float          # Σwᵢ
    is_feasible: bool
    solver_status: str
    excluded_assets: list[int]   # Indices of excluded (illiquid) assets


def normalize_beta(beta: np.ndarray, vec_idx: int = 0) -> np.ndarray:
    """
    Normalize a cointegrating vector to sum of absolute values = 1.

    This converts the raw β vector into portfolio weights that are
    interpretable as dollar allocations.

    Parameters
    ----------
    beta : np.ndarray
        (k, r) cointegrating vector matrix.
    vec_idx : int
        Which vector to normalize.

    Returns
    -------
    np.ndarray
        (k,) normalized weights summing to approximately 0 (dollar-neutral).
    """
    b = beta[:, vec_idx]
    return b / np.sum(np.abs(b))


def optimize_hedge(
    beta: np.ndarray,
    vec_idx: int = 0,
    excluded_indices: Optional[list[int]] = None,
    max_position_pct: float = None,
    max_gross_exposure: float = None,
    max_net_exposure: float = None,
    regularization: float = 0.01,
) -> HedgeResult:
    """
    Solve for optimal hedge weights using convex optimization.

    Parameters
    ----------
    beta : np.ndarray
        (k, r) cointegrating vector matrix from VECM estimation.
    vec_idx : int
        Which cointegrating vector to track.
    excluded_indices : list[int], optional
        Indices of assets to exclude (e.g., halted or illiquid).
    max_position_pct : float, optional
        Max weight per asset. Defaults to config.MAX_POSITION_SIZE_PCT.
    max_gross_exposure : float, optional
        Max sum of absolute weights. Defaults to config.MAX_GROSS_EXPOSURE.
    max_net_exposure : float, optional
        Max absolute net exposure. Defaults to config.MAX_NET_EXPOSURE.
    regularization : float
        L2 regularization coefficient (shrinks weights toward zero).

    Returns
    -------
    HedgeResult
        Optimal weights and optimization diagnostics.
    """
    k = beta.shape[0]
    target = normalize_beta(beta, vec_idx)
    excluded = excluded_indices or []

    # Defaults from config
    w_max = max_position_pct or config.MAX_POSITION_SIZE_PCT
    l_max = max_gross_exposure or config.MAX_GROSS_EXPOSURE
    net_max = max_net_exposure or config.MAX_NET_EXPOSURE

    # Decision variable
    w = cp.Variable(k)

    # Objective: minimize tracking error + regularization
    objective = cp.Minimize(
        cp.sum_squares(w - target) + regularization * cp.sum_squares(w)
    )

    # Constraints
    constraints = [
        # Dollar-neutral: net exposure ≈ 0
        cp.abs(cp.sum(w)) <= net_max,
        # Position limits
        cp.abs(w) <= w_max,
        # Gross leverage cap
        cp.norm(w, 1) <= l_max,
    ]

    # Exclude illiquid/halted assets
    for idx in excluded:
        constraints.append(w[idx] == 0)

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        # Fallback to SCS if ECOS fails
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            return HedgeResult(
                weights=target,
                target_weights=target,
                tracking_error=0.0,
                gross_exposure=float(np.sum(np.abs(target))),
                net_exposure=float(np.sum(target)),
                is_feasible=False,
                solver_status="failed",
                excluded_assets=excluded,
            )

    if problem.status in ("optimal", "optimal_inaccurate"):
        opt_w = w.value
        return HedgeResult(
            weights=opt_w,
            target_weights=target,
            tracking_error=float(np.sum((opt_w - target) ** 2)),
            gross_exposure=float(np.sum(np.abs(opt_w))),
            net_exposure=float(np.sum(opt_w)),
            is_feasible=True,
            solver_status=problem.status,
            excluded_assets=excluded,
        )
    else:
        # Infeasible — return the target weights clamped
        return HedgeResult(
            weights=target,
            target_weights=target,
            tracking_error=0.0,
            gross_exposure=float(np.sum(np.abs(target))),
            net_exposure=float(np.sum(target)),
            is_feasible=False,
            solver_status=problem.status,
            excluded_assets=excluded,
        )


def print_hedge_summary(hr: HedgeResult, asset_names: list[str]) -> None:
    """Print formatted hedge optimization results."""
    print(f"\n{'='*50}")
    print(f"  HEDGE OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"  Status:         {hr.solver_status}")
    print(f"  Feasible:       {'Yes' if hr.is_feasible else 'No ⚠'}")
    print(f"  Tracking error: {hr.tracking_error:.6f}")
    print(f"  Gross exposure: {hr.gross_exposure:.4f}")
    print(f"  Net exposure:   {hr.net_exposure:+.6f}")

    if hr.excluded_assets:
        excluded_names = [asset_names[i] for i in hr.excluded_assets]
        print(f"  Excluded:       {excluded_names}")

    print(f"\n  {'Asset':<8} {'Target':>10} {'Optimal':>10} {'Diff':>10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(asset_names):
        t = hr.target_weights[i]
        o = hr.weights[i]
        print(f"  {name:<8} {t:>+10.4f} {o:>+10.4f} {o-t:>+10.4f}")
    print(f"{'='*50}\n")
