# VECM Arbitrage System

A production-grade **Vector Error Correction Model (VECM)** statistical arbitrage system for correlated equity baskets. Built in 5 phases with modular architecture.

## Overview

This system identifies and trades mean-reverting spreads among cointegrated equity baskets using:
- **Johansen cointegration test** for rank detection and hedge ratio extraction
- **VECM estimation** for error-correction speed and half-life computation
- **Z-score based signal generation** with hysteresis and stop-loss
- **CVXPY convex optimization** for constrained hedge weight computation
- **Square-root market impact model** for realistic cost estimation
- **Cointegration breakdown protocol** with auto-liquidation on rank drop

## Architecture

```
Phase 1 (core/)     → Johansen test, VECM, spread construction, signals
Phase 2 (pipeline/) → ZeroMQ/Redis async pub-sub price feed
Phase 3 (risk/)     → CVXPY hedge optimization, liquidity monitoring
Phase 4 (backtest/) → Full backtester with fees, borrow costs, market impact
Phase 5 (core/)     → Rolling cointegration monitor, breakdown protocol
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Phase 1 — Core Math Engine

```bash
python run_phase1.py
```

Runs the full pipeline: data fetch → ADF stationarity test → Johansen cointegration → VECM estimation → spread construction → z-score signals.

### Phase 2 — Distributed Pipeline

```bash
python run_phase2.py
```

Demonstrates ZeroMQ pub-sub pipeline with historical price replay and live signal generation.

### Phase 3 — Risk Engine

```bash
python run_phase3_demo.py
```

Shows CVXPY hedge optimization with simulated liquidity events (halts, hard-to-borrow).

### Full Backtest (Phase 4 + 5)

```bash
python run_backtest.py
```

Runs the full backtest with:
- Transaction costs (maker/taker fees, bid-ask spread)
- Borrow costs for short positions
- Square-root market impact
- Cointegration breakdown monitoring and auto-liquidation
- Tearsheet output (Sharpe, drawdown, Calmar, capacity analysis)

## Asset Universe

Default: Energy sector ETF + majors
```
XLE, XOM, CVX, COP, SLB, HAL, MRO, DVN
```

Modify `config.py` → `ASSETS` to change the basket.

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `ENTRY_ZSCORE` | 2.0 | Z-score entry threshold |
| `EXIT_ZSCORE` | 0.5 | Z-score exit threshold |
| `STOP_LOSS_ZSCORE` | 4.0 | Hard stop-loss z-score |
| `ZSCORE_WINDOW` | 60 | Rolling z-score lookback (days) |
| `INITIAL_CAPITAL` | $1M | Starting capital |
| `MAKER_FEE_BPS` | 0.5 | Maker fee (basis points) |
| `BORROW_COST_ANNUAL_BPS` | 50 | Annual short borrow cost (bps) |
| `MONITOR_GRACE_PERIOD` | 5 | Days before breakdown liquidation |

## Tests

```bash
python -m pytest tests/ -v
```

## Key Mathematical References

- **Johansen (1991)**: "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models"
- **Engle & Granger (1987)**: "Co-Integration and Error Correction"
- **Almgren & Chriss (2001)**: "Optimal Execution of Portfolio Transactions"
