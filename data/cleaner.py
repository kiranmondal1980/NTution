"""
NSE MOMENTUM 5™ — Data Cleaning & Corporate Action Sanitization
================================================================================
Cleans raw provider market data to prevent bad prints, negative spreads, and
split/bonus discontinuities from corrupting downstream feature engineering.

Core Capabilities:
1. Deduplication: Eliminates duplicate timestamps, preserving the latest bar.
2. Gap Management: Forward-fills small calendar gaps (<= 2 days) for prices.
3. Physical Boundaries: Enforces Open, High, Low, Close > 0 and High >= Low.
4. Bar Geometry: Ensures High >= max(Open, Close) and Low <= min(Open, Close).
5. Split & Bonus Adjustments: Computes adjusted OHLC price series using the
   adj_close / close factor so technical indicators (EMAs, Breakouts, ATR)
   operate on a mathematically continuous series.
================================================================================
"""

from typing import Optional
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("DATA_CLEANER")


def clean_ohlcv_dataframe(
    df: pd.DataFrame,
    fill_forward_limit: int = 2
) -> pd.DataFrame:
    """
    Cleans raw OHLCV bars thoroughly.

    Args:
        df: Raw OHLCV DataFrame from any provider.
        fill_forward_limit: Max consecutive missing trading days to forward-fill.

    Returns:
        pd.DataFrame: Sanitized DataFrame indexed by sorted unique DatetimeIndex.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cleaned = df.copy()

    # 1. Ensure DatetimeIndex and sort chronologically
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned[cleaned.index.notna()]

    cleaned = cleaned.sort_index()
    # Deduplicate timestamps, retaining the last bar
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    # 2. Normalize and enforce numeric types
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # 3. Handle small data gaps
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in cleaned.columns]
    if price_cols:
        cleaned[price_cols] = cleaned[price_cols].ffill(limit=fill_forward_limit)

    if "volume" in cleaned.columns:
        cleaned["volume"] = cleaned["volume"].fillna(0.0)

    # Drop any remaining unfillable rows with missing prices
    cleaned = cleaned.dropna(subset=price_cols)
    if cleaned.empty:
        return pd.DataFrame()

    # 4. Strict Positivity Constraint (prices must be > 0)
    valid_prices_mask = (cleaned["open"] > 0) & (cleaned["close"] > 0) & (cleaned["high"] > 0) & (cleaned["low"] > 0)
    cleaned = cleaned[valid_prices_mask]

    # 5. Enforce Bar Geometry
    cleaned["high"] = np.maximum(cleaned["high"], np.maximum(cleaned["open"], cleaned["close"]))
    cleaned["low"] = np.minimum(cleaned["low"], np.minimum(cleaned["open"], cleaned["close"]))
    cleaned["volume"] = np.maximum(cleaned["volume"], 0.0)

    return cleaned


def compute_split_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes split- and bonus-adjusted Open, High, and Low prices using the
    relationship between 'adj_close' and 'close'.

    Formula:
        adjustment_factor = adj_close / close
        adj_open = open * adjustment_factor
        adj_high = high * adjustment_factor
        adj_low = low * adjustment_factor

    Returns:
        pd.DataFrame with added columns: ['adj_open', 'adj_high', 'adj_low']
    """
    if df.empty or "adj_close" not in df.columns or "close" not in df.columns:
        return df

    res = df.copy()
    # Avoid zero division
    valid_close = res["close"].replace(0, np.nan)
    factor = res["adj_close"] / valid_close
    factor = factor.fillna(1.0)

    res["adj_open"] = res["open"] * factor
    res["adj_high"] = res["high"] * factor
    res["adj_low"] = res["low"] * factor

    return res


def remove_bad_prints(
    df: pd.DataFrame,
    return_threshold: float = 0.65
) -> pd.DataFrame:
    """
    Detects and filters out erroneous flash crashes or bad print spikes where
    single-session price returns jump by more than `return_threshold` (> 65%)
    and instantly revert on the next session.

    Args:
        df: Input cleaned DataFrame.
        return_threshold: Absolute return fraction threshold (0.65 = 65%).

    Returns:
        pd.DataFrame with bad print outliers removed.
    """
    if df.empty or len(df) < 3 or "close" not in df.columns:
        return df

    filtered = df.copy()
    pct = filtered["close"].pct_change()
    next_pct = filtered["close"].pct_change(periods=-1)

    # Condition: Jump up > 65% followed immediately by drop > -35%, or vice versa
    bad_spike_up = (pct > return_threshold) & (next_pct < -0.35)
    bad_spike_down = (pct < -return_threshold) & (next_pct > 0.35)

    bad_prints_mask = bad_spike_up | bad_spike_down
    if bad_prints_mask.sum() > 0:
        logger.warning(f"Removing {bad_prints_mask.sum()} bad print anomaly bars.")
        filtered = filtered[~bad_prints_mask]

    return filtered
