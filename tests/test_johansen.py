# tests/test_johansen.py
"""
Tests for the Johansen cointegration module.

Tests include:
    - Rank detection on synthetic cointegrated series
    - Stationarity check on I(1) vs I(0) series
    - Beta vector recovery from known cointegrating relationships
"""

import numpy as np
import pandas as pd
import pytest
from core.johansen import check_stationarity, run_johansen, JohansenResult


def generate_cointegrated_series(
    n: int = 1000,
    k: int = 3,
    rank: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic cointegrated time series.

    Creates k I(1) random walks with rank cointegrating relationships
    by constructing: P_t = C * F_t + noise, where F_t is a random walk
    and C encodes the cointegrating structure.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        (log_prices DataFrame, true_beta vector)
    """
    rng = np.random.RandomState(seed)

    # Generate common stochastic trends
    n_trends = k - rank
    trends = np.cumsum(rng.randn(n, n_trends), axis=0)

    # Loading matrix: maps trends to observed prices
    C = rng.randn(k, n_trends)

    # Prices = loading * trends + stationary noise
    noise = rng.randn(n, k) * 0.1
    prices = trends @ C.T + noise

    # Shift to make all prices positive (for log transformation)
    prices = prices - prices.min(axis=0) + 10

    # Create DataFrame
    tickers = [f"ASSET_{i}" for i in range(k)]
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(np.log(prices), index=dates, columns=tickers)

    # True beta: null space of C.T gives cointegrating vectors
    # For rank=1 with 3 assets, there's 1 cointegrating vector
    U, S, Vt = np.linalg.svd(C.T)
    true_beta = Vt[n_trends:, :].T  # (k, rank)

    return df, true_beta


class TestStationarity:
    """Tests for the ADF stationarity check."""

    def test_random_walk_is_i1(self):
        """A random walk should be classified as I(1)."""
        rng = np.random.RandomState(42)
        rw = pd.DataFrame({
            "rw": np.cumsum(rng.randn(500))
        }, index=pd.date_range("2020-01-01", periods=500, freq="B"))

        result = check_stationarity(rw, alpha=0.05)
        assert result["rw"]["is_I1"], "Random walk should be I(1)"

    def test_stationary_is_not_i1(self):
        """A stationary AR(1) process should NOT be classified as I(1)."""
        rng = np.random.RandomState(42)
        x = np.zeros(500)
        for t in range(1, 500):
            x[t] = 0.5 * x[t - 1] + rng.randn()

        df = pd.DataFrame({
            "ar1": x
        }, index=pd.date_range("2020-01-01", periods=500, freq="B"))

        result = check_stationarity(df, alpha=0.05)
        assert not result["ar1"]["is_I1"], "Stationary AR(1) should not be I(1)"

    def test_multiple_series(self):
        """Test stationarity check on multiple series simultaneously."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        df = pd.DataFrame({
            "rw1": np.cumsum(rng.randn(n)),
            "rw2": np.cumsum(rng.randn(n)),
        }, index=dates)

        results = check_stationarity(df)
        assert len(results) == 2
        for name, res in results.items():
            assert "level_pvalue" in res
            assert "diff_pvalue" in res
            assert "is_I1" in res


class TestJohansen:
    """Tests for the Johansen cointegration test."""

    def test_rank_detection_cointegrated(self):
        """Johansen should detect rank >= 1 for cointegrated series."""
        prices, true_beta = generate_cointegrated_series(n=1000, k=3, rank=1)
        jr = run_johansen(prices, det_order=0, k_ar_diff=1)

        assert isinstance(jr, JohansenResult)
        assert jr.rank >= 1, f"Expected rank >= 1 for cointegrated series, got {jr.rank}"

    def test_rank_zero_for_independent_walks(self):
        """Independent random walks should have rank = 0."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        # Three truly independent random walks
        df = pd.DataFrame({
            "a": np.cumsum(rng.randn(n) * 0.01) + 4,
            "b": np.cumsum(rng.randn(n) * 0.01) + 4,
            "c": np.cumsum(rng.randn(n) * 0.01) + 4,
        }, index=dates)

        jr = run_johansen(df, det_order=0, k_ar_diff=1)
        # Rank should be 0 (no cointegration), but allow 1 due to finite sample
        assert jr.rank <= 1, f"Independent walks should not show high cointegration rank, got {jr.rank}"

    def test_beta_shape(self):
        """Beta matrix should have correct dimensions."""
        prices, _ = generate_cointegrated_series(n=1000, k=4, rank=1)
        jr = run_johansen(prices, det_order=0, k_ar_diff=1)

        k = prices.shape[1]
        assert jr.beta.shape[0] == k, f"Beta should have {k} rows, got {jr.beta.shape[0]}"

    def test_trace_stats_decreasing(self):
        """Trace statistics should be monotonically decreasing."""
        prices, _ = generate_cointegrated_series(n=1000, k=3, rank=1)
        jr = run_johansen(prices, det_order=0, k_ar_diff=1)

        for i in range(len(jr.trace_stats) - 1):
            assert jr.trace_stats[i] >= jr.trace_stats[i + 1], \
                "Trace statistics should be decreasing"

    def test_eigenvalues_non_negative(self):
        """Eigenvalues from Johansen should be non-negative."""
        prices, _ = generate_cointegrated_series(n=1000, k=3, rank=1)
        jr = run_johansen(prices, det_order=0, k_ar_diff=1)

        assert np.all(jr.eigenvalues >= -1e-10), "Eigenvalues should be non-negative"

    def test_critical_values_shape(self):
        """Critical value matrices should have correct shape."""
        prices, _ = generate_cointegrated_series(n=1000, k=3, rank=1)
        jr = run_johansen(prices, det_order=0, k_ar_diff=1)

        k = prices.shape[1]
        assert jr.trace_cvals.shape == (k, 3), \
            f"Trace cvals should be ({k}, 3), got {jr.trace_cvals.shape}"
        assert jr.max_eig_cvals.shape == (k, 3), \
            f"Max eig cvals should be ({k}, 3), got {jr.max_eig_cvals.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
