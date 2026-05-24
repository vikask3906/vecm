# risk/position_manager.py
"""
Position tracking, P&L, delta, and exposure management.

Tracks all open positions across the portfolio, computing real-time
metrics like unrealized P&L, gross/net exposure, and per-asset delta.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class Position:
    """A single asset position."""
    ticker: str
    quantity: float           # Positive = long, negative = short
    entry_price: float        # Average entry price
    current_price: float = 0.0
    cost_basis: float = 0.0   # Total cost including fees

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as a percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / abs(self.cost_basis) * 100

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def direction(self) -> str:
        if self.quantity > 0:
            return "LONG"
        elif self.quantity < 0:
            return "SHORT"
        return "FLAT"


class PositionManager:
    """
    Manages portfolio positions and computes aggregate risk metrics.

    Usage:
        pm = PositionManager(capital=1_000_000)
        pm.open_position("XLE", quantity=1000, price=85.0, fees=4.25)
        pm.open_position("XOM", quantity=-500, price=110.0, fees=5.50)
        pm.update_price("XLE", 86.0)
        pm.update_price("XOM", 109.0)
        print(pm.summary())
    """

    def __init__(self, capital: float = None):
        self.initial_capital = capital or config.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self._trade_history: list[dict] = []

    def open_position(self, ticker: str, quantity: float,
                      price: float, fees: float = 0.0) -> None:
        """
        Open or add to a position.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        quantity : float
            Shares to buy (positive) or sell short (negative).
        price : float
            Execution price per share.
        fees : float
            Total transaction fees for this trade.
        """
        cost = quantity * price + fees

        if ticker in self.positions:
            pos = self.positions[ticker]
            # Compute new average entry price
            total_qty = pos.quantity + quantity
            if abs(total_qty) > 1e-10:
                if (pos.quantity >= 0 and quantity >= 0) or (pos.quantity <= 0 and quantity <= 0):
                    # Adding to position — weighted average entry
                    pos.entry_price = (pos.quantity * pos.entry_price + quantity * price) / total_qty
                else:
                    # Reducing position — realize P&L on closed portion
                    closed_qty = min(abs(quantity), abs(pos.quantity)) * np.sign(-pos.quantity)
                    pnl = closed_qty * (price - pos.entry_price)
                    self.realized_pnl += pnl
                pos.quantity = total_qty
                pos.cost_basis += cost
            else:
                # Position fully closed
                pnl = pos.quantity * (price - pos.entry_price)
                self.realized_pnl += pnl
                del self.positions[ticker]
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                cost_basis=cost,
            )

        self.cash -= cost

        self._trade_history.append({
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "fees": fees,
        })

    def close_position(self, ticker: str, price: float, fees: float = 0.0) -> float:
        """
        Close an entire position.

        Returns the realized P&L.
        """
        if ticker not in self.positions:
            return 0.0

        pos = self.positions[ticker]
        pnl = pos.quantity * (price - pos.entry_price) - fees
        self.realized_pnl += pnl
        self.cash += pos.quantity * price - fees
        del self.positions[ticker]
        return pnl

    def close_all(self, prices: dict[str, float], fees_per_trade: float = 0.0) -> float:
        """Close all positions at the given prices."""
        total_pnl = 0.0
        for ticker in list(self.positions.keys()):
            if ticker in prices:
                total_pnl += self.close_position(ticker, prices[ticker], fees_per_trade)
        return total_pnl

    def update_price(self, ticker: str, price: float) -> None:
        """Update the current market price for a position."""
        if ticker in self.positions:
            self.positions[ticker].current_price = price

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update prices for multiple assets."""
        for ticker, price in prices.items():
            self.update_price(ticker, price)

    # ─── Portfolio-level metrics ───

    @property
    def total_market_value(self) -> float:
        """Total market value of all positions (long - short)."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position values."""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Net long/short exposure."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def gross_leverage(self) -> float:
        """Gross exposure / equity."""
        equity = self.equity
        return self.gross_exposure / equity if equity > 0 else 0.0

    @property
    def net_leverage(self) -> float:
        """Net exposure / equity."""
        equity = self.equity
        return self.net_exposure / equity if equity > 0 else 0.0

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized P&L."""
        return self.cash + self.total_market_value

    @property
    def total_pnl(self) -> float:
        """Total P&L = realized + unrealized."""
        return self.realized_pnl + self.total_unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage of initial capital."""
        return (self.equity - self.initial_capital) / self.initial_capital * 100

    def summary(self) -> str:
        """Return a formatted portfolio summary."""
        lines = [
            f"\n{'='*60}",
            f"  PORTFOLIO SUMMARY",
            f"{'='*60}",
            f"  Initial Capital:    ${self.initial_capital:>14,.2f}",
            f"  Cash:               ${self.cash:>14,.2f}",
            f"  Equity:             ${self.equity:>14,.2f}",
            f"  Gross Exposure:     ${self.gross_exposure:>14,.2f}",
            f"  Net Exposure:       ${self.net_exposure:>+14,.2f}",
            f"  Gross Leverage:      {self.gross_leverage:>13.2f}x",
            f"  Net Leverage:        {self.net_leverage:>+13.4f}x",
            f"  Realized P&L:       ${self.realized_pnl:>+14,.2f}",
            f"  Unrealized P&L:     ${self.total_unrealized_pnl:>+14,.2f}",
            f"  Total P&L:          ${self.total_pnl:>+14,.2f}",
            f"  Total Return:        {self.total_return_pct:>+13.2f}%",
        ]

        if self.positions:
            lines.append(f"\n  {'Ticker':<8} {'Dir':<6} {'Qty':>10} {'Entry':>10} "
                         f"{'Current':>10} {'P&L':>12} {'P&L%':>8}")
            lines.append(f"  {'-'*66}")
            for p in sorted(self.positions.values(), key=lambda x: x.ticker):
                lines.append(
                    f"  {p.ticker:<8} {p.direction:<6} {p.quantity:>10,.0f} "
                    f"${p.entry_price:>9.2f} ${p.current_price:>9.2f} "
                    f"${p.unrealized_pnl:>+11,.2f} {p.unrealized_pnl_pct:>+7.1f}%"
                )

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
