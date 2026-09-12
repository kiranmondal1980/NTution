"""
NSE MOMENTUM 5™ — Trend Structure & Moving Average Feature Engineering
================================================================================
Calculates multi-timeframe moving averages, trend hierarchy stacks,
percentage distances, slopes, and crossover events.

Features Computed:
1. Simple Moving Averages: SMA 10, SMA 20, SMA 50, SMA 100, SMA 200
2. Exponential Moving Averages: EMA 5, EMA 10, EMA 20, EMA 50
3. Trend Hierarchy Stack (Institutional Bullish Alignment):
   Price > EMA 5 > EMA 10 > EMA 20 > EMA 50
4. Percentage Distances from Trend Support:
   - dist_ema_10_pct: (Close - EMA10) / EMA10
   - dist_ema_20_pct: (Close - EMA20) / EMA20
   - dist_ema_50_pct: (Close - EMA50) / EMA50
   - dist_sma_200_pct: (Close - SMA200) / SMA200
5. Moving Average Slopes (5-session rate of change of EMAs)
6. Dynamic Crossover Flags:
   - EMA 10 crossing above EMA 20
   - Price crossing above EMA 20
   - Golden Cross: SMA 50 > SMA 200
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes moving average alignments, slopes, and trend stack structures.

    Args:
        df: Cleaned OHLCV DataFrame with 'close' price column.

    Returns:
        pd.DataFrame containing engineered trend indicators aligned to df.index.
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    close = df["close"]

    # 1. Simple Moving Averages (SMA)
    res["sma_10"] = close.rolling(window=10).mean()
    res["sma_20"] = close.rolling(window=20).mean()
    res["sma_50"] = close.rolling(window=50).mean()
    res["sma_100"] = close.rolling(window=100).mean()
    res["sma_200"] = close.rolling(window=200).mean()

    # 2. Exponential Moving Averages (EMA)
    # adjust=False calculates pure recursive EMA (matches standard charting engines)
    res["ema_5"] = close.ewm(span=5, adjust=False).mean()
    res["ema_10"] = close.ewm(span=10, adjust=False).mean()
    res["ema_20"] = close.ewm(span=20, adjust=False).mean()
    res["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # 3. Trend Alignment Hierarchy: Price > EMA5 > EMA10 > EMA20 > EMA50
    res["bullish_ema_stack"] = (
        (close > res["ema_5"]) &
        (res["ema_5"] > res["ema_10"]) &
        (res["ema_10"] > res["ema_20"]) &
        (res["ema_20"] > res["ema_50"])
    ).astype(int)

    # Secondary Stack: Price > SMA20 > SMA50
    res["bullish_sma_stack"] = (
        (close > res["sma_20"]) &
        (res["sma_20"] > res["sma_50"])
    ).astype(int)

    # 4. Percentage Distances to Key Moving Averages
    res["dist_ema_10_pct"] = (close - res["ema_10"]) / res["ema_10"]
    res["dist_ema_20_pct"] = (close - res["ema_20"]) / res["ema_20"]
    res["dist_ema_50_pct"] = (close - res["ema_50"]) / res["ema_50"]
    res["dist_sma_200_pct"] = np.where(
        res["sma_200"] > 0,
        (close - res["sma_200"]) / res["sma_200"],
        np.nan
    )

    # 5. EMA Trailing Slopes (5-session percentage rate of change)
    res["ema_10_slope_5d"] = (res["ema_10"] - res["ema_10"].shift(5)) / res["ema_10"].shift(5)
    res["ema_20_slope_5d"] = (res["ema_20"] - res["ema_20"].shift(5)) / res["ema_20"].shift(5)
    res["ema_50_slope_5d"] = (res["ema_50"] - res["ema_50"].shift(5)) / res["ema_50"].shift(5)

    # 6. Trend Transition & Crossover Signals
    # Fast EMA crossing slow EMA (EMA 10 cross above EMA 20)
    ema10_above_ema20 = (res["ema_10"] > res["ema_20"]).astype(int)
    res["ema10_cross_ema20"] = (
        (ema10_above_ema20 == 1) & (ema10_above_ema20.shift(1) == 0)
    ).astype(int)

    # Price crossing above EMA 20
    price_above_ema20 = (close > res["ema_20"]).astype(int)
    res["price_cross_ema20"] = (
        (price_above_ema20 == 1) & (price_above_ema20.shift(1) == 0)
    ).astype(int)

    # Long-term Regime Health: Golden Cross (SMA 50 > SMA 200)
    res["golden_cross"] = (res["sma_50"] > res["sma_200"]).astype(int)

    return res
