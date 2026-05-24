# config.py — Global constants and configuration
"""
Central configuration for the VECM Arbitrage System.
All tunable parameters live here to avoid magic numbers in module code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Asset Universe
# ─────────────────────────────────────────────
ASSETS = ["COP", "CVX", "DVN"]

# ─────────────────────────────────────────────
# Johansen / VECM Parameters
# ─────────────────────────────────────────────
JOHANSEN_DET_ORDER = 0          # 0 = constant in cointegrating relation
JOHANSEN_K_AR_DIFF = 2          # VAR lag order - 1 (use AIC/BIC to tune)
JOHANSEN_CONFIDENCE = 1         # Column index: 0=90%, 1=95%, 2=99%
ADF_ALPHA = 0.05                # Significance level for ADF stationarity test

# ─────────────────────────────────────────────
# Spread & Signal Parameters
# ─────────────────────────────────────────────
ZSCORE_WINDOW = 60              # Rolling z-score lookback (trading days)
ENTRY_ZSCORE = 2.0              # Enter position when |z| > this
EXIT_ZSCORE = 0.2               # Exit position when |z| < this (Optimized from 0.5)
STOP_LOSS_ZSCORE = 4.0          # Hard stop-loss threshold

# ─────────────────────────────────────────────
# Half-Life Bounds (trading days)
# ─────────────────────────────────────────────
MIN_HALF_LIFE = 5               # Below this = noise
MAX_HALF_LIFE = 60              # Above this = too slow to trade

# ─────────────────────────────────────────────
# Cointegration Monitor (Phase 5)
# ─────────────────────────────────────────────
MONITOR_WINDOW = 252            # Rolling window for Johansen retest (1 year)
MONITOR_RECHECK_FREQ = 21      # Recheck every N trading days (~monthly)
MONITOR_GRACE_PERIOD = 5        # Days of rank-drop before triggering liquidation
MONITOR_COOLDOWN = 15           # Days to wait after breakdown before re-entry (Optimized from 42)

# ─────────────────────────────────────────────
# Transaction Costs & Fees
# ─────────────────────────────────────────────
MAKER_FEE_BPS = 0.5             # 0.5 bps maker fee
TAKER_FEE_BPS = 1.0             # 1.0 bps taker fee
BORROW_COST_ANNUAL_BPS = 50.0   # 50 bps annualized borrow cost for shorts
BID_ASK_SPREAD_BPS = 2.0        # Average bid-ask spread in bps

# ─────────────────────────────────────────────
# Market Impact Model
# ─────────────────────────────────────────────
IMPACT_COEFFICIENT = 0.1        # Sigma coefficient in sqrt impact model
ADV_LOOKBACK = 20               # Days for average daily volume calculation

# ─────────────────────────────────────────────
# Risk Limits
# ─────────────────────────────────────────────
MAX_POSITION_SIZE_PCT = 0.20    # Max 20% of capital in any single asset
MAX_GROSS_EXPOSURE = 2.0        # Max 2x gross leverage
MAX_NET_EXPOSURE = 0.10         # Max 10% net exposure (near dollar-neutral)
INITIAL_CAPITAL = 1_000_000     # $1M starting capital for backtest

# ─────────────────────────────────────────────
# Liquidity Thresholds
# ─────────────────────────────────────────────
MIN_ADV_SHARES = 500_000        # Minimum average daily volume (shares)
LIQUIDITY_WARNING_PCT = 0.05    # Warn if trade > 5% of ADV
LIQUIDITY_HALT_PCT = 0.10       # Halt if trade > 10% of ADV

# ─────────────────────────────────────────────
# Pipeline (Phase 2)
# ─────────────────────────────────────────────
ZMQ_PUB_ADDRESS = "tcp://127.0.0.1:5555"
ZMQ_SUB_ADDRESS = "tcp://127.0.0.1:5555"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
PIPELINE_BACKEND = os.getenv("PIPELINE_BACKEND", "zmq")  # "zmq" or "redis"
PRICE_BUFFER_SIZE = 500         # Rolling price buffer for live recomputation

# ─────────────────────────────────────────────
# API Keys (from .env)
# ─────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
