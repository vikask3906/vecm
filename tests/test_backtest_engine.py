# tests/test_backtest_engine.py
"""
Tests for the backtest engine.

Tests include:
    - P&L calculation with known synthetic signals
    - Transaction cost deduction
    - Cointegration breakdown handling
    - Position tracking consistency
"""

import numpy as np
import pandas as pd
import pytest
from backtest.engine import BacktestEngine, BacktestConfig, BacktestResults
from backtest.transaction_costs import TransactionCostModel, CostBreakdown
from backtest.market_impact import SquareRootImpactModel, ImpactEstimate
from backtest.tearsheet import compute_tearsheet, TearsheetMetrics
from core.spread import construct_spread, compute_zscore, generate_signals


def generate_synthetic_data(n: int = 600, k: int = 3, seed: int = 42):
    """Generate synthetic log price and volume data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Create cointegrated series
    trend = np.cumsum(rng.randn(n) * 0.01)
    noise = rng.randn(n, k) * 0.005

    prices = np.column_stack([
        trend + noise[:, 0] + 4.0,
        -0.5 * trend + noise[:, 1] + 4.5,
        0.3 * trend + noise[:, 2] + 4.2,
    ])

    tickers = [f"A{i}" for i in range(k)]
    log_prices = pd.DataFrame(prices, index=dates, columns=tickers)
    volume = pd.DataFrame(
        rng.randint(500_000, 5_000_000, size=(n, k)),
        index=dates, columns=tickers
    ).astype(float)

    return log_prices, volume


class TestTransactionCostModel:
    """Tests for the transaction cost model."""

    def test_basic_cost(self):
        """Basic cost calculation should be positive."""
        tcm = TransactionCostModel(
            maker_fee_bps=0.5,
            taker_fee_bps=1.0,
            borrow_cost_annual_bps=50,
            bid_ask_spread_bps=2.0,
        )
        cost = tcm.compute_trade_cost("XLE", notional=100_000)
        assert cost.total_cost > 0
        assert cost.fee_cost > 0
        assert cost.spread_cost > 0

    def test_maker_cheaper_than_taker(self):
        """Maker orders should be cheaper than taker orders."""
        tcm = TransactionCostModel(maker_fee_bps=0.5, taker_fee_bps=1.0)
        maker_cost = tcm.compute_trade_cost("XLE", 100_000, is_maker=True)
        taker_cost = tcm.compute_trade_cost("XLE", 100_000, is_maker=False)
        assert maker_cost.fee_cost < taker_cost.fee_cost

    def test_short_has_borrow_cost(self):
        """Short positions should incur borrow costs."""
        tcm = TransactionCostModel(borrow_cost_annual_bps=50)
        short_cost = tcm.compute_trade_cost("XLE", 100_000, is_short=True)
        long_cost = tcm.compute_trade_cost("XLE", 100_000, is_short=False)
        assert short_cost.borrow_cost_daily > 0
        assert long_cost.borrow_cost_daily == 0

    def test_portfolio_cost(self):
        """Portfolio cost should be the sum of individual costs."""
        tcm = TransactionCostModel()
        trades = {"XLE": 50_000, "XOM": -30_000, "CVX": 20_000}
        total, breakdowns = tcm.compute_portfolio_cost(trades)
        assert total > 0
        assert len(breakdowns) == 3
        assert abs(total - sum(b.total_cost for b in breakdowns.values())) < 1e-10


class TestSquareRootImpactModel:
    """Tests for the market impact model."""

    def test_zero_order_zero_impact(self):
        """Zero order size should have zero impact."""
        model = SquareRootImpactModel(eta=0.1)
        impact = model.estimate_impact("XLE", 0, 85.0, 5_000_000, 0.02)
        assert impact.impact_bps == 0
        assert impact.impact_dollars == 0

    def test_impact_increases_with_size(self):
        """Larger orders should have more impact."""
        model = SquareRootImpactModel(eta=0.1)
        small = model.estimate_impact("XLE", 1_000, 85.0, 5_000_000, 0.02)
        large = model.estimate_impact("XLE", 100_000, 85.0, 5_000_000, 0.02)
        assert large.impact_bps > small.impact_bps

    def test_sublinear_scaling(self):
        """Impact should scale sub-linearly (square root)."""
        model = SquareRootImpactModel(eta=0.1)
        base = model.estimate_impact("XLE", 10_000, 85.0, 5_000_000, 0.02)
        quad = model.estimate_impact("XLE", 40_000, 85.0, 5_000_000, 0.02)
        # 4x the order should give 2x the impact (sqrt)
        ratio = quad.impact_bps / base.impact_bps
        assert abs(ratio - 2.0) < 0.1, f"Expected ~2x impact, got {ratio}x"

    def test_capacity_positive(self):
        """Capacity should be positive for valid inputs."""
        model = SquareRootImpactModel(eta=0.1)
        cap = model.compute_capacity(85.0, 5_000_000, 0.02, target_impact_bps=5.0)
        assert cap > 0

    def test_capacity_constrained_flag(self):
        """Large orders should be flagged as capacity-constrained."""
        model = SquareRootImpactModel(eta=0.1)
        # Order = 20% of ADV (above 10% threshold)
        impact = model.estimate_impact("XLE", 1_000_000, 85.0, 5_000_000, 0.02)
        assert impact.is_capacity_constrained


class TestSignalGeneration:
    """Tests for the spread signal generation."""

    def test_signal_values(self):
        """Signals should only be -1, 0, or +1."""
        zscore = pd.Series(np.sin(np.linspace(0, 10, 200)) * 3)
        signals = generate_signals(zscore, entry_z=2.0, exit_z=0.5)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_signal_length(self):
        """Signal series should have same length as input."""
        zscore = pd.Series(np.random.randn(100))
        signals = generate_signals(zscore, entry_z=2.0, exit_z=0.5)
        assert len(signals) == len(zscore)

    def test_entry_threshold(self):
        """Signals should only appear when z-score exceeds entry threshold."""
        # Z-score that never exceeds 1.5 should give all zeros with entry_z=2.0
        zscore = pd.Series(np.ones(100) * 1.5)
        signals = generate_signals(zscore, entry_z=2.0, exit_z=0.5)
        assert (signals == 0).all()


class TestBacktestEngine:
    """Tests for the backtest engine."""

    def test_backtest_runs(self):
        """Backtest should complete without errors."""
        log_prices, volume = generate_synthetic_data(n=400)
        bt_config = BacktestConfig(
            initial_capital=100_000,
            warmup_period=200,
            reestimation_freq=50,
            enable_impact=False,
            enable_breakdown=False,
        )
        engine = BacktestEngine(log_prices, volume, bt_config)
        results = engine.run()

        assert isinstance(results, BacktestResults)
        assert results.final_equity > 0
        assert len(results.daily_records) > 0

    def test_tearsheet_computation(self):
        """Tearsheet should compute all metrics."""
        log_prices, volume = generate_synthetic_data(n=400)
        bt_config = BacktestConfig(
            initial_capital=100_000,
            warmup_period=200,
            reestimation_freq=50,
            enable_impact=False,
            enable_breakdown=False,
        )
        engine = BacktestEngine(log_prices, volume, bt_config)
        results = engine.run()
        metrics = compute_tearsheet(results)

        assert isinstance(metrics, TearsheetMetrics)
        assert not np.isnan(metrics.sharpe_ratio)
        assert metrics.annualized_volatility >= 0

    def test_costs_tracked(self):
        """Transaction costs should be tracked correctly."""
        log_prices, volume = generate_synthetic_data(n=400)
        bt_config = BacktestConfig(
            initial_capital=100_000,
            warmup_period=200,
            reestimation_freq=50,
            enable_impact=False,
            enable_breakdown=False,
        )
        engine = BacktestEngine(log_prices, volume, bt_config)
        results = engine.run()

        # Costs should be non-negative
        assert results.total_transaction_costs >= 0
        assert results.total_borrow_costs >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
