# tests/test_hedge_optimizer.py
"""
Tests for the CVXPY hedge optimizer.

Tests include:
    - Dollar-neutral constraint satisfaction
    - Position limit enforcement
    - Asset exclusion handling
    - Feasibility on various inputs
"""

import numpy as np
import pytest
from risk.hedge_optimizer import optimize_hedge, normalize_beta, HedgeResult


class TestNormalizeBeta:
    """Tests for beta normalization."""

    def test_normalization_sums_to_one(self):
        """Normalized beta should have absolute values summing to 1."""
        beta = np.array([[1.0], [-0.5], [0.3], [-0.8]])
        norm = normalize_beta(beta, vec_idx=0)
        assert abs(np.sum(np.abs(norm)) - 1.0) < 1e-10

    def test_normalization_preserves_signs(self):
        """Normalization should preserve the sign structure."""
        beta = np.array([[1.0], [-0.5], [0.3]])
        norm = normalize_beta(beta, vec_idx=0)
        assert norm[0] > 0
        assert norm[1] < 0
        assert norm[2] > 0

    def test_normalization_multiple_vectors(self):
        """Should correctly normalize a specific vector from multiple."""
        beta = np.array([[1.0, 2.0], [-0.5, 0.3], [0.3, -1.0]])
        norm_0 = normalize_beta(beta, vec_idx=0)
        norm_1 = normalize_beta(beta, vec_idx=1)

        # Different vectors should give different normalizations
        assert not np.allclose(norm_0, norm_1)


class TestOptimizeHedge:
    """Tests for the hedge optimization."""

    def test_feasibility(self):
        """Optimizer should find a feasible solution for typical inputs."""
        beta = np.array([[1.0], [-0.5], [0.3], [-0.2], [0.1]])
        hr = optimize_hedge(beta, vec_idx=0)

        assert isinstance(hr, HedgeResult)
        assert hr.is_feasible, f"Should be feasible, got status: {hr.solver_status}"

    def test_dollar_neutral(self):
        """Optimal weights should be approximately dollar-neutral."""
        beta = np.array([[1.0], [-0.5], [0.3], [-0.8]])
        hr = optimize_hedge(beta, vec_idx=0, max_net_exposure=0.05)

        assert hr.is_feasible
        assert abs(hr.net_exposure) < 0.05 + 1e-3, \
            f"Net exposure {hr.net_exposure} exceeds limit 0.05 (with solver tolerance)"

    def test_position_limits(self):
        """No weight should exceed the position limit."""
        beta = np.array([[1.0], [-0.5], [0.3]])
        max_pos = 0.15
        hr = optimize_hedge(beta, vec_idx=0, max_position_pct=max_pos)

        assert hr.is_feasible
        for w in hr.weights:
            assert abs(w) <= max_pos + 1e-6, \
                f"Weight {w} exceeds position limit {max_pos}"

    def test_gross_leverage_cap(self):
        """Gross exposure should respect the leverage cap."""
        beta = np.array([[1.0], [-0.5], [0.3], [-0.2]])
        max_gross = 1.5
        hr = optimize_hedge(beta, vec_idx=0, max_gross_exposure=max_gross)

        assert hr.is_feasible
        assert hr.gross_exposure <= max_gross + 1e-6, \
            f"Gross exposure {hr.gross_exposure} exceeds cap {max_gross}"

    def test_asset_exclusion(self):
        """Excluded assets should have zero weight."""
        beta = np.array([[1.0], [-0.5], [0.3], [-0.8], [0.2]])
        excluded = [1, 3]  # Exclude assets 1 and 3
        hr = optimize_hedge(beta, vec_idx=0, excluded_indices=excluded)

        assert hr.is_feasible
        for idx in excluded:
            assert abs(hr.weights[idx]) < 1e-8, \
                f"Excluded asset {idx} should have zero weight, got {hr.weights[idx]}"

    def test_weights_shape(self):
        """Output weights should match input dimensions."""
        k = 6
        beta = np.random.randn(k, 1)
        hr = optimize_hedge(beta, vec_idx=0)

        assert hr.weights.shape == (k,), \
            f"Weights shape should be ({k},), got {hr.weights.shape}"

    def test_tracking_error_bounded(self):
        """Tracking error should be reasonably small."""
        beta = np.array([[1.0], [-0.5], [0.3]])
        hr = optimize_hedge(beta, vec_idx=0, regularization=0.001)

        assert hr.is_feasible
        assert hr.tracking_error < 0.5, \
            f"Tracking error {hr.tracking_error} is too large (position limits constrain away from target)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
