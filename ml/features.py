"""
NSE MOMENTUM 5™ — Machine Learning Feature Matrix Assembler
================================================================================
Curates, sanitizes, and normalizes the input feature matrix X for probabilistic
classification models.

CURATION PRINCIPLES:
1. Anti-Leakage Guarantee:
   - Includes exclusively backward-looking indicators (Price, Trend, Volume,
     ATR, RSI, Shifted Breakout, Relative Strength).
   - Excludes all forward-looking targets, raw unadjusted future bars, or IDs.
2. Stationarity & Normalization:
   - Prefers percentage returns, ratios, and bounded oscillators over nominal prices.
   - Enforces finite numbers (replaces infinite/NaN values with clean medians).
================================================================================
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


# Explicit list of validated quantitative features for ML models
ML_FEATURE_NAMES: List[str] = [
    # Momentum Velocity & Acceleration
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "mom_accel_1v5", "mom_accel_3v10", "mom_slope_5d", "consecutive_up_days",
    # Trend Alignment & Proximity
    "dist_ema_10_pct", "dist_ema_20_pct", "dist_ema_50_pct", "bullish_ema_stack",
    "ema_10_slope_5d", "ema_20_slope_5d",
    # Volume Dynamics & Candle Anatomy
    "vol_ratio_5d", "vol_ratio_20d", "vol_accel", "up_volume_ratio_10d",
    "clv", "candle_body_range_ratio",
    # Volatility & Oscillators
    "atr_pct", "volatility_20d", "bb_width", "bb_pos", "rsi_14", "rsi_slope_3d",
    "adx_14", "macd_hist",
    # Shifted Breakout Features
    "breakout_20d", "breakout_50d", "dist_breakout_20d_pct", "dist_52w_high_pct",
    "breakout_persisted_2d",
    # Benchmark Relative Strength vs NIFTY 50
    "rs_5d", "rs_20d", "rs_outperform_days_10d", "mansfield_rs"
]


def extract_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts and sanitizes the feature matrix X from an engineered dataset.

    Args:
        df: DataFrame containing all calculated technical indicators.

    Returns:
        Tuple of (X_matrix: pd.DataFrame, feature_names_used: List[str]).
    """
    if df is None or df.empty:
        return pd.DataFrame(), []

    available = [col for col in ML_FEATURE_NAMES if col in df.columns]
    if not available:
        return pd.DataFrame(), []

    X = df[available].copy()

    # 1. Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # 2. Forward-fill small indicator warmup gaps, fill remaining with 0.0
    X = X.ffill().fillna(0.0)

    return X, available


def prepare_training_dataset(
    df_combined: pd.DataFrame,
    target_col: str = "target_15pct_5d"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Aligns feature matrix X and target series y, dropping rows with missing targets
    (e.g., the trailing 5 rows where forward horizons have not yet concluded).

    Args:
        df_combined: DataFrame containing features and forward research targets.
        target_col: Binary target label column to predict.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series, feature_names: List[str]).
    """
    if df_combined is None or df_combined.empty or target_col not in df_combined.columns:
        return pd.DataFrame(), pd.Series(dtype=float), []

    X, features_used = extract_ml_features(df_combined)
    y_raw = df_combined[target_col]

    # Drop rows where target is NaN (e.g., trailing incomplete windows)
    valid_mask = y_raw.notna()
    X_clean = X.loc[valid_mask]
    y_clean = y_raw.loc[valid_mask].astype(int)

    return X_clean, y_clean, features_used
