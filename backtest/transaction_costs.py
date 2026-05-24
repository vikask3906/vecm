# backtest/transaction_costs.py
"""
Transaction cost modeling for the backtester.

Implements a realistic multi-component cost model:
    1. Maker/taker fees (exchange fees, typically 0.5–1.0 bps)
    2. Bid-ask spread cost (half-spread per leg)
    3. Borrow cost for short positions (annualized, accrues daily)
    4. Clearing/settlement fees (fixed per trade)

The total transaction cost significantly affects strategy profitability,
especially for high-frequency rebalancing strategies.
"""

import numpy as np
from dataclasses import dataclass
import config


@dataclass
class CostBreakdown:
    """Breakdown of transaction costs for a single trade."""
    ticker: str
    notional: float
    fee_cost: float            # Exchange maker/taker fee
    spread_cost: float         # Half bid-ask spread cost
    borrow_cost_daily: float   # Daily borrow cost (shorts only)
    total_cost: float          # Sum of all components


class TransactionCostModel:
    """
    Multi-component transaction cost model.

    Computes realistic trading costs for each leg of a spread trade,
    including exchange fees, bid-ask spread costs, and short borrow costs.

    Usage:
        tcm = TransactionCostModel()
        cost = tcm.compute_trade_cost("XLE", notional=50000, is_maker=True)
        print(f"Total cost: ${cost.total_cost:.2f}")
    """

    def __init__(
        self,
        maker_fee_bps: float = None,
        taker_fee_bps: float = None,
        borrow_cost_annual_bps: float = None,
        bid_ask_spread_bps: float = None,
    ):
        """
        Parameters
        ----------
        maker_fee_bps : float, optional
            Maker fee in basis points. Defaults to config.
        taker_fee_bps : float, optional
            Taker fee in basis points. Defaults to config.
        borrow_cost_annual_bps : float, optional
            Annualized borrow cost in bps (for shorts). Defaults to config.
        bid_ask_spread_bps : float, optional
            Average bid-ask spread in bps. Defaults to config.
        """
        self.maker_fee_bps = maker_fee_bps or config.MAKER_FEE_BPS
        self.taker_fee_bps = taker_fee_bps or config.TAKER_FEE_BPS
        self.borrow_cost_annual_bps = borrow_cost_annual_bps or config.BORROW_COST_ANNUAL_BPS
        self.bid_ask_spread_bps = bid_ask_spread_bps or config.BID_ASK_SPREAD_BPS

    def compute_trade_cost(
        self,
        ticker: str,
        notional: float,
        is_maker: bool = False,
        is_short: bool = False,
        custom_spread_bps: float = None,
    ) -> CostBreakdown:
        """
        Compute the total transaction cost for a trade.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        notional : float
            Absolute notional value of the trade.
        is_maker : bool
            True if this is a maker (passive) order.
        is_short : bool
            True if this is a short sale.
        custom_spread_bps : float, optional
            Override for the bid-ask spread in bps.

        Returns
        -------
        CostBreakdown
            Breakdown of all cost components.
        """
        abs_notional = abs(notional)

        # 1. Exchange fee
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee_cost = abs_notional * fee_bps / 10_000

        # 2. Bid-ask spread cost (half-spread, you cross the spread)
        spread_bps = custom_spread_bps or self.bid_ask_spread_bps
        spread_cost = abs_notional * (spread_bps / 2) / 10_000

        # 3. Borrow cost (shorts only, daily accrual)
        borrow_daily = 0.0
        if is_short:
            borrow_daily = abs_notional * self.borrow_cost_annual_bps / 10_000 / 252

        total = fee_cost + spread_cost + borrow_daily

        return CostBreakdown(
            ticker=ticker,
            notional=abs_notional,
            fee_cost=fee_cost,
            spread_cost=spread_cost,
            borrow_cost_daily=borrow_daily,
            total_cost=total,
        )

    def compute_portfolio_cost(
        self,
        trade_notionals: dict[str, float],
        is_maker: bool = False,
    ) -> tuple[float, dict[str, CostBreakdown]]:
        """
        Compute total cost for a portfolio trade (multiple assets).

        Parameters
        ----------
        trade_notionals : dict[str, float]
            Mapping of ticker → signed notional (negative = short).

        Returns
        -------
        tuple[float, dict[str, CostBreakdown]]
            (total_cost, per_asset_breakdown)
        """
        breakdowns = {}
        total = 0.0

        for ticker, notional in trade_notionals.items():
            cb = self.compute_trade_cost(
                ticker=ticker,
                notional=notional,
                is_maker=is_maker,
                is_short=(notional < 0),
            )
            breakdowns[ticker] = cb
            total += cb.total_cost

        return total, breakdowns

    def compute_holding_cost(
        self,
        short_notionals: dict[str, float],
        days: int = 1,
    ) -> float:
        """
        Compute daily borrow cost for all short positions.

        Parameters
        ----------
        short_notionals : dict[str, float]
            Mapping of ticker → absolute notional for short positions.
        days : int
            Number of days to compute cost for.

        Returns
        -------
        float
            Total borrow cost over the specified period.
        """
        total = 0.0
        for ticker, notional in short_notionals.items():
            daily = abs(notional) * self.borrow_cost_annual_bps / 10_000 / 252
            total += daily * days
        return total
