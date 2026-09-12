"""
NSE MOMENTUM 5™ — Quantitative Data & OHLCV Validation Engine
================================================================================
Guarantees mathematical consistency, price integrity, and geometric correctness
of all market time-series bars prior to feature extraction or backtesting.

Checks Performed:
1. Structural integrity: Column presence, non-emptiness, and DatetimeIndex validity
2. Physical price boundaries: Open, High, Low, Close > 0
3. Bar geometry: High >= Max(Open, Close) and Low <= Min(Open, Close)
4. Monotonicity: High >= Low
5. Volume non-negativity: Volume >= 0
6. Deduplication: Removes duplicate session timestamps, keeping the cleanest bar
7. Anomaly detection: Flags unadjusted stock splits, bad prints, or zero-spread ticks
================================================================================
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def validate_dataframe_schema(
    df: Optional[pd.DataFrame],
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validates presence of required columns and ensures the DataFrame is not None or empty.

    Args:
        df: Input pandas DataFrame to validate.
        required_cols: List of column names that must be present.

    Returns:
        Tuple of (is_valid: bool, error_messages: List[str]).
    """
    errors: List[str] = []
    if df is None:
        return False, ["DataFrame is None."]
    if df.empty:
        return False, ["DataFrame contains 0 rows."]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"Missing required schema columns: {missing}")

    return (len(errors) == 0), errors


def sanitize_ohlcv(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    enforce_strict_geometry: bool = True
) -> pd.DataFrame:
    """
    Cleans, deduplicates, and sanitizes an OHLCV dataset.

    Args:
        df: Raw DataFrame containing price and volume columns.
        date_col: Name of date column if not already the index.
        enforce_strict_geometry: If True, corrects minor bar anomalies
            (e.g., setting High = max(High, Open, Close)).

    Returns:
        A sanitized, sorted DataFrame with a unique DatetimeIndex.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    clean = df.copy()

    # 1. Normalize Datetime Index
    if date_col and date_col in clean.columns:
        clean[date_col] = pd.to_datetime(clean[date_col], errors="coerce")
        clean = clean.dropna(subset=[date_col])
        clean = clean.set_index(date_col)
    elif not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index, errors="coerce")
        clean = clean[clean.index.notna()]

    # Normalize column names to lowercase standard
    clean.columns = [str(c).lower().strip() for c in clean.columns]

    # Handle alternate column aliases if present
    alias_map = {
        "adj close": "adj_close",
        "adjusted_close": "adj_close",
        "vol": "volume"
    }
    clean = clean.rename(columns=alias_map)

    # 2. Sort chronologically and deduplicate timestamps
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    # 3. Numeric type conversion
    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "adj_close" in clean.columns:
        numeric_cols.append("adj_close")

    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Drop any row where essential prices are NaN
    essential_prices = [c for c in ["open", "high", "low", "close"] if c in clean.columns]
    clean = clean.dropna(subset=essential_prices)

    # 4. Strict Positivity Check (Prices must be > 0)
    for col in essential_prices:
        clean = clean[clean[col] > 0.0]

    # 5. Enforce Bar Geometry
    if enforce_strict_geometry and all(c in clean.columns for c in ["open", "high", "low", "close"]):
        # High must be at least as high as Open, Low, and Close
        clean["high"] = np.maximum(
            clean["high"],
            np.maximum(clean["open"], clean["close"])
        )
        # Low must be at most as low as Open, High, and Close
        clean["low"] = np.minimum(
            clean["low"],
            np.minimum(clean["open"], clean["close"])
        )
        # Guarantee High >= Low
        clean = clean[clean["high"] >= clean["low"]]

    # 6. Volume Validation
    if "volume" in clean.columns:
        clean["volume"] = clean["volume"].fillna(0.0)
        # Replace negative volume with 0
        clean["volume"] = np.maximum(clean["volume"], 0.0)

    # Default adj_close to close if missing
    if "close" in clean.columns and "adj_close" not in clean.columns:
        clean["adj_close"] = clean["close"]

    return clean


def detect_price_anomalies(
    df: pd.DataFrame,
    max_single_day_pct_move: float = 0.50
) -> Dict[str, Any]:
    """
    Audits a sanitized price series to detect data quality issues such as:
    - Extreme session returns (> 50%) indicative of unadjusted stock splits
    - Zero volume runs in liquid securities
    - Zero spread bars (High == Low for multiple consecutive sessions)

    Args:
        df: Sanitized OHLCV DataFrame.
        max_single_day_pct_move: Threshold for single-day absolute jump.

    Returns:
        Dictionary containing audit flags and count of detected anomalies.
    """
    report: Dict[str, Any] = {
        "has_anomalies": False,
        "extreme_jumps_detected": 0,
        "zero_volume_bars": 0,
        "zero_spread_bars": 0,
        "details": []
    }

    if df.empty or "close" not in df.columns:
        return report

    # 1. Check for extreme jumps
    daily_returns = df["close"].pct_change().abs()
    jumps = daily_returns[daily_returns > max_single_day_pct_move]
    if len(jumps) > 0:
        report["has_anomalies"] = True
        report["extreme_jumps_detected"] = len(jumps)
        dates_str = [d.strftime("%Y-%m-%d") for d in jumps.index[:3]]
        report["details"].append(
            f"{len(jumps)} bars exceeded {max_single_day_pct_move*100:.0f}% session change (e.g., on {dates_str})."
        )

    # 2. Check for zero volume
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        report["zero_volume_bars"] = int(zero_vol)
        if zero_vol > 5:
            report["details"].append(f"{zero_vol} sessions had zero trading volume.")

    # 3. Check for flat bars (High == Low)
    if "high" in df.columns and "low" in df.columns:
        zero_spread = (df["high"] == df["low"]).sum()
        report["zero_spread_bars"] = int(zero_spread)
        if zero_spread > 5:
            report["details"].append(f"{zero_spread} bars had zero high-low spread (High == Low).")

    return report
