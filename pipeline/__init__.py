# pipeline/ — Phase 2: Distributed Data Pipeline
"""
Async pub-sub pipeline for real-time price feed ingestion.

Modules:
    broker      — Abstract message broker with ZMQ and Redis backends
    publisher   — Publishes price ticks to the message bus
    subscriber  — Async subscriber feeding the math engine
    orderbook   — Per-asset orderbook state management
"""
