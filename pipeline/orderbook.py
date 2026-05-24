# pipeline/orderbook.py
"""
Orderbook state management for the distributed pipeline.

Maintains per-asset orderbook state (bid/ask/mid/spread) from
incoming price ticks. Used by the risk engine and backtester to
compute realistic execution prices and transaction costs.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from pipeline.broker import PriceTick


@dataclass
class OrderBookLevel:
    """A single price level in the orderbook."""
    price: float
    size: float
    timestamp: float


@dataclass
class OrderBookState:
    """Current state of a single asset's orderbook."""
    ticker: str
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: float = 0.0
    last_volume: float = 0.0
    last_update: float = 0.0

    @property
    def mid_price(self) -> float:
        """Mid-price (average of bid and ask)."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute terms."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0

    @property
    def is_stale(self) -> bool:
        """True if last update was more than 60 seconds ago."""
        return (time.time() - self.last_update) > 60


class OrderBookManager:
    """
    Manages orderbook state for all tracked assets.

    Usage:
        manager = OrderBookManager(tickers=["XLE", "XOM", "CVX"])
        manager.update(tick)
        state = manager.get("XLE")
        print(f"Mid: {state.mid_price}, Spread: {state.spread_bps:.1f}bps")
    """

    def __init__(self, tickers: list[str]):
        self.books: dict[str, OrderBookState] = {
            t: OrderBookState(ticker=t) for t in tickers
        }
        self._update_count = 0

    def update(self, tick: PriceTick) -> None:
        """
        Update orderbook state from a price tick.

        Parameters
        ----------
        tick : PriceTick
            Incoming price tick with bid/ask data.
        """
        if tick.ticker not in self.books:
            self.books[tick.ticker] = OrderBookState(ticker=tick.ticker)

        book = self.books[tick.ticker]
        book.last_price = tick.price
        book.last_volume = tick.volume
        book.last_update = tick.timestamp or time.time()

        if tick.bid > 0:
            book.bid = tick.bid
        if tick.ask > 0:
            book.ask = tick.ask

        self._update_count += 1

    def get(self, ticker: str) -> Optional[OrderBookState]:
        """Get the current orderbook state for a ticker."""
        return self.books.get(ticker)

    def get_all_mid_prices(self) -> dict[str, float]:
        """Get mid-prices for all tracked assets."""
        return {t: book.mid_price for t, book in self.books.items()}

    def get_all_spreads_bps(self) -> dict[str, float]:
        """Get bid-ask spreads in bps for all tracked assets."""
        return {t: book.spread_bps for t, book in self.books.items()}

    def get_stale_tickers(self) -> list[str]:
        """Get tickers with stale orderbook data."""
        return [t for t, book in self.books.items() if book.is_stale]

    @property
    def update_count(self) -> int:
        return self._update_count

    def summary(self) -> str:
        """Return a formatted summary of all orderbooks."""
        lines = [f"{'Ticker':<8} {'Bid':>10} {'Ask':>10} {'Mid':>10} {'Spread(bps)':>12}"]
        lines.append("-" * 52)
        for t, book in sorted(self.books.items()):
            lines.append(
                f"{t:<8} {book.bid:>10.2f} {book.ask:>10.2f} "
                f"{book.mid_price:>10.2f} {book.spread_bps:>11.1f}"
            )
        return "\n".join(lines)
