# core/ — Phase 1: VECM Math Engine
"""
Core mathematical engine for cointegration-based statistical arbitrage.

Modules:
    data_loader     — Price data fetching and preprocessing
    johansen        — Johansen cointegration test and rank selection
    vecm            — VECM estimation (alpha, beta, half-life)
    spread          — Spread construction, z-score, and signal generation
    cointegration_monitor — Rolling cointegration monitoring (Phase 5)
"""

from core.data_loader import fetch_prices, ASSETS
from core.johansen import run_johansen, check_stationarity, JohansenResult
from core.vecm import fit_vecm, VECMResult
from core.spread import construct_spread, compute_zscore, generate_signals
