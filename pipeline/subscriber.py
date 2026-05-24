# pipeline/subscriber.py
"""
Async price subscriber for the distributed pipeline.

Receives price ticks from the message broker, maintains a rolling
price buffer, and feeds the VECM math engine for live signal generation.

Architecture:
    Publisher → [Broker] → Subscriber → Math Engine → Signals
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Callable
from datetime import datetime

from pipeline.broker import MessageBroker, PriceTick, create_broker
from core.johansen import run_johansen
from core.vecm import fit_vecm
from core.spread import construct_spread, compute_zscore, generate_signals
import config


class PriceBuffer:
    """
    Rolling price buffer that maintains the last N observations.

    Used to feed the math engine with enough history for VECM estimation
    and z-score computation.
    """

    def __init__(self, tickers: list[str], max_size: int = 500):
        self.tickers = tickers
        self.max_size = max_size
        self.prices: dict[str, list[float]] = {t: [] for t in tickers}
        self.timestamps: list[float] = []
        self._count = 0

    def add_tick(self, tick: PriceTick) -> None:
        """Add a price tick to the buffer."""
        if tick.ticker not in self.prices:
            return

        # Only advance the buffer when we have a new timestamp
        if not self.timestamps or tick.timestamp > self.timestamps[-1]:
            self.timestamps.append(tick.timestamp)
            for t in self.tickers:
                if t != tick.ticker:
                    # Forward-fill missing tickers for this timestamp
                    last = self.prices[t][-1] if self.prices[t] else 0.0
                    self.prices[t].append(last)
            self.prices[tick.ticker].append(tick.price)
            self._count += 1

            # Trim to max size
            if self._count > self.max_size:
                self.timestamps = self.timestamps[-self.max_size:]
                for t in self.tickers:
                    self.prices[t] = self.prices[t][-self.max_size:]
                self._count = self.max_size
        else:
            # Update the current timestamp's price for this ticker
            if self.prices[tick.ticker]:
                self.prices[tick.ticker][-1] = tick.price

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to a DataFrame of log prices."""
        if self._count == 0:
            return pd.DataFrame()

        dates = pd.to_datetime(self.timestamps, unit="s")
        data = {t: self.prices[t] for t in self.tickers}
        df = pd.DataFrame(data, index=dates)
        return np.log(df.replace(0, np.nan).dropna())

    @property
    def is_ready(self) -> bool:
        """True if buffer has enough data for VECM estimation."""
        return self._count >= config.ZSCORE_WINDOW + 50  # Need extra for warmup

    @property
    def size(self) -> int:
        return self._count


class PriceSubscriber:
    """
    Subscribes to price ticks and feeds the VECM math engine.

    Maintains a rolling price buffer and recomputes VECM signals
    when enough new data has accumulated.

    Usage:
        sub = PriceSubscriber(tickers=ASSETS)
        sub.start(on_signal=my_signal_handler)
    """

    def __init__(
        self,
        tickers: list[str],
        broker: Optional[MessageBroker] = None,
        buffer_size: int = None,
        recompute_interval: int = 20,
    ):
        """
        Parameters
        ----------
        tickers : list[str]
            Assets to track.
        broker : MessageBroker, optional
            Pre-configured broker. Creates a new SUB broker if None.
        buffer_size : int, optional
            Rolling buffer size. Defaults to config.PRICE_BUFFER_SIZE.
        recompute_interval : int
            Recompute VECM every N new observations.
        """
        self.tickers = tickers
        self.broker = broker or create_broker(mode="sub")
        self.buffer = PriceBuffer(tickers, max_size=buffer_size or config.PRICE_BUFFER_SIZE)
        self.recompute_interval = recompute_interval

        self._ticks_since_recompute = 0
        self._current_signal = 0.0
        self._current_zscore = 0.0
        self._vecm_result = None
        self._johansen_result = None
        self._signal_callback = None

    def _on_message(self, topic: str, message: str) -> None:
        """Internal message handler."""
        if topic == "PRICE":
            tick = PriceTick.from_json(message)
            self.buffer.add_tick(tick)
            self._ticks_since_recompute += 1

            # Recompute VECM periodically if buffer is ready
            if (self.buffer.is_ready and
                    self._ticks_since_recompute >= self.recompute_interval):
                self._recompute()
                self._ticks_since_recompute = 0

        elif topic == "CTRL":
            if message == "END_OF_STREAM":
                print("[Subscriber] End of stream received.")
                if self.buffer.is_ready:
                    self._recompute()
                self.broker.stop()

    def _recompute(self) -> None:
        """Recompute VECM and generate signals from the current buffer."""
        prices = self.buffer.to_dataframe()
        if prices.empty or len(prices) < config.ZSCORE_WINDOW + 10:
            return

        try:
            # Run Johansen to get rank
            jr = run_johansen(prices, det_order=config.JOHANSEN_DET_ORDER,
                              k_ar_diff=config.JOHANSEN_K_AR_DIFF)
            self._johansen_result = jr

            if jr.rank == 0:
                self._current_signal = 0.0
                return

            # Fit VECM
            vr = fit_vecm(prices, rank=jr.rank,
                          k_ar_diff=config.JOHANSEN_K_AR_DIFF)
            self._vecm_result = vr

            # Compute spread and signal
            spread = construct_spread(prices, vr.beta, vec_idx=0)
            zscore = compute_zscore(spread, window=config.ZSCORE_WINDOW)
            signals = generate_signals(zscore, entry_z=config.ENTRY_ZSCORE,
                                       exit_z=config.EXIT_ZSCORE)

            self._current_zscore = zscore.iloc[-1] if not zscore.empty else 0.0
            new_signal = signals.iloc[-1] if not signals.empty else 0.0

            # Notify if signal changed
            if new_signal != self._current_signal:
                self._current_signal = new_signal
                if self._signal_callback:
                    self._signal_callback(new_signal, self._current_zscore, vr)

            print(f"[Subscriber] Recomputed — rank={jr.rank}, "
                  f"z={self._current_zscore:.2f}, signal={new_signal:.0f}, "
                  f"HL={vr.half_lives[0]:.1f}d, buffer={self.buffer.size}")

        except Exception as e:
            print(f"[Subscriber] Recompute error: {e}")

    def start(self, on_signal: Optional[Callable] = None) -> None:
        """
        Start listening for price ticks.

        Parameters
        ----------
        on_signal : callable, optional
            Called with (signal, zscore, vecm_result) when signal changes.
        """
        self._signal_callback = on_signal
        print(f"[Subscriber] Listening for ticks on {len(self.tickers)} assets...")
        self.broker.subscribe(["PRICE", "CTRL"], self._on_message)

    @property
    def current_signal(self) -> float:
        return self._current_signal

    @property
    def current_zscore(self) -> float:
        return self._current_zscore

    def close(self) -> None:
        """Clean up resources."""
        self.broker.close()
