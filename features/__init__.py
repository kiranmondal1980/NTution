"""
NSE MOMENTUM 5™ — Quantitative Feature Engineering Suite
================================================================================
Exposes all sub-feature extractors and provides a unified end-to-end
pipeline builder that transforms raw OHLCV bars into a quantitative matrix.

Sub-Modules:
- momentum: Velocity, acceleration, linear regression slopes, streak dynamics
- trend: Moving average hierarchies, distance to EMAs, trend stack checks
- volume: Surge ratios, volume acceleration, up/down volume skew, CLV
- volatility: True range, ATR 14, ATR%, Bollinger Bands, RSI 14, ADX 14, MACD
- breakout: Anti-leakage shifted resistance, penetration depth, persistence
- relative_strength: Multi-horizon alpha differentials vs NIFTY 50 benchmark
================================================================================
"""

from typing import Optional
import pandas as pd

from .momentum import calculate_momentum_features
from .trend import calculate_trend_features
from .volume import calculate_volume_features
from .volatility import calculate_volatility_features
from .breakout import calculate_breakout_features
from .relative_strength import calculate_relative_strength_features


def build_feature_pipeline(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Executes the entire feature extraction pipeline sequentially on an equity series.

    Guarantees:
    - Pure backward-looking indicators (no future data leakage)
    - Full alignment on the original DatetimeIndex
    - Column naming collisions resolved cleanly

    Args:
        df: Cleaned equity OHLCV DataFrame.
        benchmark_df: Cleaned NIFTY 50 OHLCV DataFrame (optional).

    Returns:
        pd.DataFrame containing original OHLCV columns + all quantitative indicators.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Calculate each feature component
    f_mom = calculate_momentum_features(df)
    f_trend = calculate_trend_features(df)
    f_vol = calculate_volume_features(df)
    f_volat = calculate_volatility_features(df)

    # Pass pre-calculated ATR to breakout engine for normalized penetration strength
    atr_series = f_volat["atr_14"] if "atr_14" in f_volat.columns else None
    f_bo = calculate_breakout_features(df, atr_series=atr_series)

    f_rs = calculate_relative_strength_features(df, benchmark_df=benchmark_df)

    # 2. Concatenate all features alongside the original OHLCV DataFrame
    combined = pd.concat([df, f_mom, f_trend, f_vol, f_volat, f_bo, f_rs], axis=1)

    # Remove any accidental duplicate columns while preserving order
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    return combined


__all__ = [
    "calculate_momentum_features",
    "calculate_trend_features",
    "calculate_volume_features",
    "calculate_volatility_features",
    "calculate_breakout_features",
    "calculate_relative_strength_features",
    "build_feature_pipeline"
]
