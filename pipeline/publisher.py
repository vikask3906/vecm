# pipeline/publisher.py
"""
Price tick publisher for the distributed pipeline.

Supports two modes:
    1. Historical replay: Replays yfinance data as simulated ticks
    2. Live streaming: Connects to Alpaca API for real-time price feeds

The publisher pushes price ticks to the message broker (ZMQ or Redis),
which fans them out to all connected subscribers.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional

from pipeline.broker import MessageBroker, PriceTick, create_broker
from core.data_loader import fetch_prices, ASSETS
import config


class PricePublisher:
    """
    Publishes price ticks to the message bus.

    Usage:
        pub = PricePublisher()
        pub.replay_historical(start="2023-01-01", end="2024-01-01", speed=100)
    """

    def __init__(self, broker: Optional[MessageBroker] = None,
                 tickers: list[str] = None):
        """
        Parameters
        ----------
        broker : MessageBroker, optional
            Pre-configured broker. Creates a new PUB broker if None.
        tickers : list[str], optional
            Assets to publish. Defaults to config.ASSETS.
        """
        self.broker = broker or create_broker(mode="pub")
        self.tickers = tickers or ASSETS
        self._running = False

    def replay_historical(self, start: str, end: str,
                          speed: float = 1.0,
                          tick_interval: float = 0.01) -> None:
        """
        Replay historical price data as simulated price ticks.

        Parameters
        ----------
        start : str
            Start date for historical data.
        end : str
            End date for historical data.
        speed : float
            Replay speed multiplier (100 = 100x faster than real-time).
        tick_interval : float
            Minimum seconds between ticks (throttling).
        """
        import yfinance as yf

        print(f"[Publisher] Downloading historical data for {self.tickers}...")
        raw = yf.download(self.tickers, start=start, end=end, auto_adjust=True)
        close = raw["Close"].dropna()
        volume = raw["Volume"].reindex(close.index).fillna(0)

        print(f"[Publisher] Replaying {len(close)} trading days at {speed}x speed...")
        self._running = True

        for i, (date, row) in enumerate(close.iterrows()):
            if not self._running:
                print("[Publisher] Stopped.")
                break

            for ticker in self.tickers:
                if ticker in row.index and not np.isnan(row[ticker]):
                    tick = PriceTick(
                        ticker=ticker,
                        price=float(row[ticker]),
                        volume=float(volume.loc[date, ticker]) if ticker in volume.columns else 0,
                        timestamp=date.timestamp(),
                        bid=float(row[ticker]) * 0.9999,  # Simulated bid
                        ask=float(row[ticker]) * 1.0001,  # Simulated ask
                    )
                    self.broker.publish("PRICE", tick.to_json())

            if i % 100 == 0:
                print(f"[Publisher] Published day {i}/{len(close)} — {date.date()}")

            time.sleep(tick_interval / speed)

        # Send end-of-stream signal
        self.broker.publish("CTRL", "END_OF_STREAM")
        print(f"[Publisher] Replay complete. Published {len(close)} days of data.")

    def publish_tick(self, tick: PriceTick) -> None:
        """Publish a single price tick."""
        self.broker.publish("PRICE", tick.to_json())

    def stop(self) -> None:
        """Stop the publisher."""
        self._running = False

    def close(self) -> None:
        """Clean up resources."""
        self._running = False
        self.broker.close()
