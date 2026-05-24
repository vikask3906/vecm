# backtest/market_impact.py
"""
Market impact modeling for the backtester.

Implements the square-root market impact model, which is the industry
standard for equity markets:

    Impact = η × σ × √(Q / ADV)

where:
    η (eta)  = impact coefficient (calibrated, typically 0.05–0.15)
    σ (sigma) = daily volatility of the asset
    Q        = order size (shares)
    ADV      = average daily volume (shares)

This model captures the empirical observation that market impact grows
sub-linearly with order size — doubling the order doesn't double the cost.

References:
    - Almgren & Chriss (2001), "Optimal execution of portfolio transactions"
    - Barra Market Impact Model
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
import config


@dataclass
class ImpactEstimate:
    """Market impact estimate for a single trade."""
    ticker: str
    order_size_shares: float
    adv: float
    volatility: float
    participation_rate: float   # Q / ADV
    impact_bps: float           # Estimated impact in basis points
    impact_dollars: float       # Absolute dollar impact
    is_capacity_constrained: bool


class SquareRootImpactModel:
    """
    Square-root market impact model.

    Usage:
        model = SquareRootImpactModel()
        impact = model.estimate_impact("XLE", order_shares=10000,
                                        price=85.0, adv=5_000_000,
                                        volatility=0.02)
    """

    def __init__(
        self,
        eta: float = None,
        adv_lookback: int = None,
    ):
        """
        Parameters
        ----------
        eta : float, optional
            Impact coefficient. Defaults to config.IMPACT_COEFFICIENT.
        adv_lookback : int, optional
            Days for ADV calculation. Defaults to config.ADV_LOOKBACK.
        """
        self.eta = eta or config.IMPACT_COEFFICIENT
        self.adv_lookback = adv_lookback or config.ADV_LOOKBACK

    def estimate_impact(
        self,
        ticker: str,
        order_shares: float,
        price: float,
        adv: float,
        volatility: float,
    ) -> ImpactEstimate:
        """
        Estimate market impact for a single order.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        order_shares : float
            Order size in shares (absolute value).
        price : float
            Current price per share.
        adv : float
            Average daily volume in shares.
        volatility : float
            Daily return volatility (standard deviation).

        Returns
        -------
        ImpactEstimate
            Impact estimate with all components.
        """
        abs_shares = abs(order_shares)

        if adv <= 0 or volatility <= 0:
            return ImpactEstimate(
                ticker=ticker,
                order_size_shares=abs_shares,
                adv=adv,
                volatility=volatility,
                participation_rate=0,
                impact_bps=0,
                impact_dollars=0,
                is_capacity_constrained=True,
            )

        participation_rate = abs_shares / adv

        # Square-root impact formula
        impact_pct = self.eta * volatility * np.sqrt(participation_rate)
        impact_bps = impact_pct * 10_000
        impact_dollars = impact_pct * abs_shares * price

        # Capacity constraint: flag if participation > 10%
        is_constrained = participation_rate > 0.10

        return ImpactEstimate(
            ticker=ticker,
            order_size_shares=abs_shares,
            adv=adv,
            volatility=volatility,
            participation_rate=participation_rate,
            impact_bps=impact_bps,
            impact_dollars=impact_dollars,
            is_capacity_constrained=is_constrained,
        )

    def estimate_portfolio_impact(
        self,
        orders: dict[str, float],
        prices: dict[str, float],
        advs: dict[str, float],
        volatilities: dict[str, float],
    ) -> tuple[float, dict[str, ImpactEstimate]]:
        """
        Estimate total portfolio market impact.

        Returns
        -------
        tuple[float, dict[str, ImpactEstimate]]
            (total_impact_dollars, per_asset_estimates)
        """
        estimates = {}
        total_impact = 0.0

        for ticker, shares in orders.items():
            est = self.estimate_impact(
                ticker=ticker,
                order_shares=shares,
                price=prices.get(ticker, 0),
                adv=advs.get(ticker, 0),
                volatility=volatilities.get(ticker, 0.02),
            )
            estimates[ticker] = est
            total_impact += est.impact_dollars

        return total_impact, estimates

    def compute_capacity(
        self,
        price: float,
        adv: float,
        volatility: float,
        target_impact_bps: float = 5.0,
    ) -> float:
        """
        Compute the maximum order size before impact exceeds the target.

        Solves: target_bps = η × σ × √(Q / ADV) × 10000
        For Q:  Q = ADV × (target_bps / (η × σ × 10000))²

        Parameters
        ----------
        price : float
            Current price per share.
        adv : float
            Average daily volume in shares.
        volatility : float
            Daily return volatility.
        target_impact_bps : float
            Maximum acceptable impact in basis points.

        Returns
        -------
        float
            Maximum notional value (in dollars) that can be traded.
        """
        if self.eta <= 0 or volatility <= 0 or adv <= 0:
            return 0.0

        target_pct = target_impact_bps / 10_000
        max_participation = (target_pct / (self.eta * volatility)) ** 2
        max_shares = max_participation * adv
        max_notional = max_shares * price

        return max_notional


def compute_adv_from_volume(volume: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute rolling average daily volume."""
    return volume.rolling(window=lookback).mean()


def compute_daily_volatility(log_prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute rolling daily return volatility from log prices."""
    returns = log_prices.diff()
    return returns.rolling(window=lookback).std()
