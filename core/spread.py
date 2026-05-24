# core/spread.py
"""
Spread construction, z-score normalization, and signal generation.

The cointegrating spread s_t = β'P_t is the key tradeable object:
    - When s_t deviates from its mean, we expect mean reversion
    - The z-score quantifies how extreme the deviation is
    - Entry/exit thresholds convert the z-score into discrete signals
"""

import numpy as np
import pandas as pd


def construct_spread(prices: pd.DataFrame, beta: np.ndarray,
                     vec_idx: int = 0) -> pd.Series:
    """
    Construct the cointegrating spread: s_t = β' @ P_t

    Parameters
    ----------
    prices : pd.DataFrame
        Log price series for k assets.
    beta : np.ndarray
        (k, r) cointegrating vector matrix from VECM estimation.
    vec_idx : int
        Which cointegrating vector to use (0 = strongest relationship).

    Returns
    -------
    pd.Series
        The cointegrating spread time series.
    """
    b = beta[:, vec_idx]
    spread = prices.values @ b          # (T, k) @ (k,) = (T,)
    return pd.Series(spread, index=prices.index, name="spread")


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score of the spread.

    Parameters
    ----------
    spread : pd.Series
        The cointegrating spread.
    window : int
        Lookback period in trading days (60 ≈ 3 months).

    Returns
    -------
    pd.Series
        Z-score series: (spread - rolling_mean) / rolling_std.
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    zscore = (spread - mean) / std
    zscore.name = "zscore"
    return zscore


def generate_signals(zscore: pd.Series,
                     entry_z: float = 2.0,
                     exit_z: float = 0.5,
                     stop_loss_z: float = 4.0) -> pd.Series:
    """
    Threshold-based signal generation with hysteresis and stop-loss.

    Signal values:
        +1.0 = long the spread  (zscore < -entry_z → spread is cheap)
        -1.0 = short the spread (zscore > +entry_z → spread is rich)
         0.0 = flat (|zscore| < exit_z, or stop-loss triggered)

    The hysteresis logic ensures we don't flip-flop on noise:
    once a position is entered, it stays until the exit threshold is
    crossed (mean reversion) or the stop-loss is hit (breakdown).

    Parameters
    ----------
    zscore : pd.Series
        Rolling z-score of the spread.
    entry_z : float
        Z-score threshold to enter a position.
    exit_z : float
        Z-score threshold to exit a position (closer to 0 = tighter).
    stop_loss_z : float
        Hard stop-loss z-score (exit immediately if exceeded).

    Returns
    -------
    pd.Series
        Signal series: +1 (long), -1 (short), 0 (flat).
    """
    in_long = False
    in_short = False
    signals = []

    for z in zscore:
        if np.isnan(z):
            signals.append(0.0)
            continue

        # Stop-loss check: if spread blows out, flatten immediately
        if abs(z) > stop_loss_z:
            in_long = False
            in_short = False
            signals.append(0.0)
            continue

        # Entry logic
        if z < -entry_z and not in_short:
            in_long = True
            in_short = False
        elif z > entry_z and not in_long:
            in_short = True
            in_long = False

        # Exit logic: mean reversion achieved
        if in_long and z > -exit_z:
            in_long = False
        if in_short and z < exit_z:
            in_short = False

        signals.append(1.0 if in_long else (-1.0 if in_short else 0.0))

    result = pd.Series(signals, index=zscore.index, name="signal")
    return result


def compute_signal_stats(signals: pd.Series, zscore: pd.Series) -> dict:
    """
    Compute summary statistics for the generated signals.

    Parameters
    ----------
    signals : pd.Series
        Signal series from generate_signals().
    zscore : pd.Series
        Z-score series used to generate the signals.

    Returns
    -------
    dict
        Statistics including trade counts, durations, and coverage.
    """
    total_days = len(signals)
    long_days = (signals == 1.0).sum()
    short_days = (signals == -1.0).sum()
    flat_days = (signals == 0.0).sum()

    # Count distinct trades (transitions from 0 to non-zero)
    trade_entries = ((signals != 0) & (signals.shift(1) == 0)).sum()

    # Average trade duration
    trades = []
    current_trade_len = 0
    for s in signals:
        if s != 0:
            current_trade_len += 1
        elif current_trade_len > 0:
            trades.append(current_trade_len)
            current_trade_len = 0
    if current_trade_len > 0:
        trades.append(current_trade_len)

    avg_duration = np.mean(trades) if trades else 0

    return {
        "total_days": total_days,
        "long_days": int(long_days),
        "short_days": int(short_days),
        "flat_days": int(flat_days),
        "active_pct": (long_days + short_days) / total_days * 100,
        "num_trades": len(trades),
        "avg_trade_duration": round(avg_duration, 1),
        "max_zscore": zscore.max(),
        "min_zscore": zscore.min(),
    }
