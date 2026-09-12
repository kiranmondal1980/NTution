"""
NSE MOMENTUM 5™ — Price Momentum & Velocity Feature Engineering
================================================================================
Calculates strictly backward-looking multi-period price momentum, velocity,
acceleration, linear regression slopes, and candle streak dynamics.

Features Computed:
1. Trailing Returns: 1D, 2D, 3D, 5D, 10D, 20D percent changes
2. Momentum Acceleration:
   - 1D vs 5D velocity differential: ret_1d - (ret_5d / 5.0)
   - 3D vs 10D velocity differential: (ret_3d / 3.0) - (ret_10d / 10.0)
   - Second derivative of price (ROC of ROC)
3. Normalized Linear Regression Slope (5-day rolling slope / close)
4. Consecutive Up Sessions: Streak counter of consecutive closes > prior close
5. Higher-High / Higher-Low Structure: Rolling 3D & 5D bullish swing persistence

LEAKAGE PROTECTION:
All operations are strictly backward-looking (historical lookbacks only).
No forward shifts or future data leaks.
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates comprehensive momentum and acceleration indicators from an OHLCV DataFrame.

    Args:
        df: Cleaned OHLCV DataFrame indexed by DatetimeIndex with 'close', 'high', 'low'.

    Returns:
        pd.DataFrame containing engineered momentum indicators aligned to df.index.
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)

    # 1. Multi-Period Trailing Returns (Velocity)
    # close_T / close_{T-N} - 1.0
    res["ret_1d"] = close.pct_change(1)
    res["ret_2d"] = close.pct_change(2)
    res["ret_3d"] = close.pct_change(3)
    res["ret_5d"] = close.pct_change(5)
    res["ret_10d"] = close.pct_change(10)
    res["ret_20d"] = close.pct_change(20)

    # 2. Momentum Acceleration Metrics
    # Compares instantaneous short-term velocity against medium-term trend rate
    res["mom_accel_1v5"] = res["ret_1d"] - (res["ret_5d"] / 5.0)
    res["mom_accel_3v10"] = (res["ret_3d"] / 3.0) - (res["ret_10d"] / 10.0)

    # Second derivative of price (Rate of Change of 3-day momentum)
    res["price_accel_3d"] = res["ret_3d"] - res["ret_3d"].shift(3)

    # 3. Normalized 5-Day Linear Regression Slope
    def compute_rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        def calc_slope(y_vals: np.ndarray) -> float:
            if np.isnan(y_vals).any():
                return np.nan
            y_mean = y_vals.mean()
            return float(((x - x_mean) * (y_vals - y_mean)).sum() / x_var)

        return series.rolling(window).apply(calc_slope, raw=True)

    slope_5d = compute_rolling_slope(close, window=5)
    res["mom_slope_5d"] = slope_5d / close

    # 4. Consecutive Up Sessions Streak Counter
    is_up = (close > close.shift(1)).astype(int)
    # Reset streak when day is not up
    streak = is_up.groupby((is_up != is_up.shift(1)).cumsum()).cumsum() * is_up
    res["consecutive_up_days"] = streak

    # 5. Higher-High and Higher-Low (HH/HL) Structure
    # 3-Day structure: Today High > High[T-1] and Today Low > Low[T-1]
    hh_3d = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
    hl_3d = (low > low.shift(1)) & (low.shift(1) > low.shift(2))
    res["hh_hl_structure_3d"] = (hh_3d & hl_3d).astype(int)

    # 5-Day swing structure: High > 5-day prior high reference
    res["higher_high_5d"] = (high >= high.shift(1).rolling(5).max()).astype(int)

    return res
