"""
NSE MOMENTUM 5™ — Volume Quality & Expansion Feature Engineering
================================================================================
Calculates institutional volume surges, liquidity acceleration, accumulation skew,
On-Balance Volume (OBV), and candlestick price-volume location metrics.

Features Computed:
1. Volume Baselines: 5-day SMA (`vol_sma_5`) and 20-day SMA (`vol_sma_20`)
2. Relative Volume Ratios (Surge Detection):
   - vol_ratio_5d: Volume / 5-day average volume
   - vol_ratio_20d: Volume / 20-day average volume
3. Volume Acceleration: 5-day volume SMA / 20-day volume SMA (`vol_accel`)
4. Up-Volume vs Down-Volume Skew:
   - 10-day & 20-day ratio of volume traded on positive vs negative sessions
5. Close Location Value (CLV):
   - Normalized position of Close within High-Low range (-1.0 to +1.0)
   - CLV = ((Close - Low) - (High - Close)) / (High - Low)
6. Candlestick Anatomy Ratios:
   - Body to Range ratio: |Close - Open| / (High - Low)
   - Upper Wick ratio: (High - max(Open, Close)) / (High - Low)
   - Lower Wick ratio: (min(Open, Close) - Low) / (High - Low)
7. On-Balance Volume (OBV) and 5-day OBV Slope
8. Institutional Accumulation Confirmation Flag
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes volume quality, surge metrics, and price-volume location features.

    Args:
        df: Cleaned OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pd.DataFrame containing engineered volume indicators aligned to df.index.
    """
    if df is None or df.empty or "volume" not in df.columns:
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    vol = df["volume"]
    close = df["close"]
    open_p = df.get("open", close)
    high = df.get("high", close)
    low = df.get("low", close)

    # 1. Volume Baseline Moving Averages
    res["vol_sma_5"] = vol.rolling(window=5).mean()
    res["vol_sma_20"] = vol.rolling(window=20).mean()

    # 2. Volume Surge Ratios (Avoid division by zero)
    safe_sma_5 = res["vol_sma_5"].replace(0, np.nan)
    safe_sma_20 = res["vol_sma_20"].replace(0, np.nan)
    res["vol_ratio_5d"] = (vol / safe_sma_5).fillna(1.0)
    res["vol_ratio_20d"] = (vol / safe_sma_20).fillna(1.0)

    # 3. Volume Acceleration (Short-term volume avg expanding over medium-term)
    res["vol_accel"] = (res["vol_sma_5"] / safe_sma_20).fillna(1.0)

    # 4. Up-Volume vs Down-Volume Accumulation Skew
    # Up-day: Close >= Open; Down-day: Close < Open
    is_up_session = close >= open_p
    up_volume = np.where(is_up_session, vol, 0.0)
    down_volume = np.where(~is_up_session, vol, 0.0)

    up_vol_sum_10 = pd.Series(up_volume, index=df.index).rolling(window=10).sum()
    down_vol_sum_10 = pd.Series(down_volume, index=df.index).rolling(window=10).sum()
    total_vol_10 = (up_vol_sum_10 + down_vol_sum_10).replace(0, np.nan)
    res["up_volume_ratio_10d"] = (up_vol_sum_10 / total_vol_10).fillna(0.5)

    up_vol_sum_20 = pd.Series(up_volume, index=df.index).rolling(window=20).sum()
    down_vol_sum_20 = pd.Series(down_volume, index=df.index).rolling(window=20).sum()
    total_vol_20 = (up_vol_sum_20 + down_vol_sum_20).replace(0, np.nan)
    res["up_volume_ratio_20d"] = (up_vol_sum_20 / total_vol_20).fillna(0.5)

    # 5. Candlestick Range Geometry & Close Location Value (CLV)
    candle_range = (high - low).replace(0, np.nan)
    candle_body = (close - open_p).abs()

    # Body to range ratio (high ratio = conviction candle; low ratio = doji/indecision)
    res["candle_body_range_ratio"] = (candle_body / candle_range).fillna(0.0)

    # Upper & Lower Wicks
    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    res["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0.0)
    res["lower_wick_ratio"] = (lower_wick / candle_range).fillna(0.0)

    # Close Location Value (CLV): Ranges from -1.0 (closed at low) to +1.0 (closed at high)
    # CLV = ((Close - Low) - (High - Close)) / (High - Low)
    clv_raw = ((close - low) - (high - close)) / candle_range
    res["clv"] = clv_raw.fillna(0.0)

    # 6. On-Balance Volume (OBV)
    price_direction = np.sign(close.diff()).fillna(0.0)
    obv_change = price_direction * vol
    res["obv"] = obv_change.cumsum()
    # 5-day OBV trend slope normalized by 20-day volume
    obv_diff_5d = res["obv"] - res["obv"].shift(5)
    res["obv_slope_5d"] = (obv_diff_5d / safe_sma_20).fillna(0.0)

    # 7. Institutional Accumulation Confirmation Flag
    # Defined as: Volume surge (>= 1.5x 20D avg) AND Strong positive close (CLV >= +0.3)
    res["institutional_volume_surge"] = (
        (res["vol_ratio_20d"] >= 1.50) & (res["clv"] >= 0.30)
    ).astype(int)

    return res
