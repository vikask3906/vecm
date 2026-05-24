# risk/ — Phase 3: Dynamic Hedging & Risk Engine
"""
Dynamic hedging and risk management for the VECM arbitrage system.

Modules:
    hedge_optimizer    — CVXPY convex optimizer for hedge weights
    liquidity_monitor  — Halt detection and hard-to-borrow flags
    position_manager   — Position tracking, delta, exposure
    rebalancer         — Reoptimization on liquidity events
"""
