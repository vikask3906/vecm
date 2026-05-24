# run_phase2.py
"""
Phase 2 Driver — Distributed Data Pipeline Demo

Demonstrates the ZeroMQ pub-sub pipeline by replaying historical
price data through the publisher → subscriber → signal pipeline.

Uses threading to run publisher and subscriber concurrently.
"""

import threading
import time
from core.data_loader import ASSETS
from pipeline.publisher import PricePublisher
from pipeline.subscriber import PriceSubscriber
from pipeline.broker import ZMQBroker
import config


def on_signal_change(signal: float, zscore: float, vecm_result) -> None:
    """Callback when the trading signal changes."""
    direction = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "FLAT")
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  SIGNAL CHANGE: {direction:>5}  |  z={zscore:+.2f}      ║")
    print(f"  ║  Half-life: {vecm_result.half_lives[0]:.1f} trading days          ║")
    print(f"  ╚══════════════════════════════════════════╝\n")


def run_subscriber(tickers: list[str]) -> None:
    """Run the subscriber in a separate thread."""
    time.sleep(1)  # Give publisher time to bind
    sub = PriceSubscriber(tickers=tickers, recompute_interval=50)
    sub.start(on_signal=on_signal_change)


def main():
    print("=" * 60)
    print("  VECM ARBITRAGE SYSTEM — PHASE 2: DISTRIBUTED PIPELINE")
    print("=" * 60)
    print(f"\n  Backend: ZeroMQ")
    print(f"  Address: {config.ZMQ_PUB_ADDRESS}")
    print(f"  Assets:  {ASSETS}")
    print(f"  Replaying: 2022-01-01 → 2024-01-01\n")

    # Start subscriber in a background thread
    sub_thread = threading.Thread(
        target=run_subscriber,
        args=(ASSETS,),
        daemon=True,
    )
    sub_thread.start()

    # Run publisher in the main thread (historical replay)
    pub = PricePublisher(tickers=ASSETS)
    try:
        pub.replay_historical(
            start="2022-01-01",
            end="2024-01-01",
            speed=100,
            tick_interval=0.01,
        )
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        pub.close()

    # Wait for subscriber to finish processing
    time.sleep(2)
    print(f"\n{'='*60}")
    print(f"  PHASE 2 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
