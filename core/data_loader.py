# core/data_loader.py
"""
Price data fetching and preprocessing for the VECM engine.
Downloads adjusted close prices via yfinance and returns log-transformed series.
"""

import yfinance as yf
import numpy as np
import pandas as pd

ASSETS = ["XLE", "XOM", "CVX", "COP", "SLB", "HAL", "DVN"]


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices and return log prices.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Log-transformed adjusted close prices, NaN rows dropped.
        Cointegration is defined on log prices for financial series
        (ensures returns are log-returns, and ratios become differences).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw = raw.dropna()
    return np.log(raw)


def fetch_prices_with_volume(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch both log prices and volume data.
    Volume is needed for market impact estimation in Phase 4.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (log_prices, volume) DataFrames with aligned indices.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close = raw["Close"].dropna()
    volume = raw["Volume"].reindex(close.index).ffill()
    return np.log(close), volume
