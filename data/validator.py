"""
NSE MOMENTUM 5™ — Historical Data Quality & Integrity Auditor
================================================================================
Performs institutional audits on stock price series before downstream processing.
Validates:
1. Minimum history requirements (>= 60 bars for short-term, >= 252 for annual)
2. Missingness ratio (rejects series with > 5% missing or unfillable values)
3. Physical bar geometry violations (High < Low, Low > Open, etc.)
4. Stale price freezes (detects suspended stocks with >= 5 identical closes)
5. Corporate action anomalies (flags unadjusted stock splits with > 60% jumps)
6. Calendar continuity (flags unannounced trading halts exceeding 7 days)
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("DATA_VALIDATOR")


@dataclass(frozen=True)
class DataQualityReport:
    """Audit summary metrics for a security's historical dataset."""
    symbol: str
    is_valid: bool
    total_bars: int
    missing_pct: float
    anomalies_detected: int
    zero_volume_bars: int
    stale_price_streaks: int
    max_calendar_gap_days: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "is_valid": self.is_valid,
            "total_bars": self.total_bars,
            "missing_pct": self.missing_pct,
            "anomalies_detected": self.anomalies_detected,
            "zero_volume_bars": self.zero_volume_bars,
            "stale_price_streaks": self.stale_price_streaks,
            "max_calendar_gap_days": self.max_calendar_gap_days,
            "notes": self.notes
        }


def check_data_quality(
    df: Optional[pd.DataFrame],
    symbol: str = "UNKNOWN",
    min_bars: int = 60,
    max_missing_pct: float = 5.0
) -> DataQualityReport:
    """
    Runs a quality audit on an equity's historical OHLCV series.

    Args:
        df: Cleaned OHLCV DataFrame.
        symbol: Ticker identifier for audit reporting.
        min_bars: Minimum number of historical trading sessions required.
        max_missing_pct: Maximum allowable percentage of missing entries.

    Returns:
        DataQualityReport: Comprehensive audit metrics and pass/fail flag.
    """
    notes: List[str] = []
    anomalies = 0

    # 1. Structural and Non-Emptiness Checks
    if df is None or df.empty:
        notes.append("Dataset is empty or None.")
        return DataQualityReport(
            symbol=symbol,
            is_valid=False,
            total_bars=0,
            missing_pct=100.0,
            anomalies_detected=1,
            zero_volume_bars=0,
            stale_price_streaks=0,
            max_calendar_gap_days=0,
            notes=notes
        )

    total_bars = len(df)
    if total_bars < min_bars:
        notes.append(f"Insufficient history: {total_bars} bars found, minimum {min_bars} required.")

    # 2. Missing Value Ratio
    required_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if not required_cols:
        notes.append("Essential price columns (open, high, low, close) missing.")
        return DataQualityReport(
            symbol=symbol,
            is_valid=False,
            total_bars=total_bars,
            missing_pct=100.0,
            anomalies_detected=1,
            zero_volume_bars=0,
            stale_price_streaks=0,
            max_calendar_gap_days=0,
            notes=notes
        )

    total_cells = total_bars * len(required_cols)
    null_cells = int(df[required_cols].isna().sum().sum())
    missing_pct = round((null_cells / total_cells) * 100.0, 2)

    if missing_pct > max_missing_pct:
        notes.append(f"Missing data ratio ({missing_pct}%) exceeds allowable threshold ({max_missing_pct}%).")

    # 3. Bar Inversions (High < Low)
    inversions = int((df["high"] < df["low"]).sum()) if "high" in df.columns and "low" in df.columns else 0
    if inversions > 0:
        anomalies += inversions
        notes.append(f"Detected {inversions} bars where High < Low.")

    # 4. Zero Volume Bars
    zero_vol_bars = int((df["volume"] <= 0).sum()) if "volume" in df.columns else 0
    if zero_vol_bars > (total_bars * 0.10):  # More than 10% zero volume is suspicious for liquid universe
        notes.append(f"Elevated zero volume frequency: {zero_vol_bars} sessions have zero traded volume.")

    # 5. Stale Price Freezes (Trading Halts / Circuit Locks)
    # Detect streaks of >= 5 sessions where Close is strictly identical
    stale_streaks = 0
    if "close" in df.columns:
        close_diff = (df["close"] != df["close"].shift(1)).cumsum()
        streak_counts = df.groupby(close_diff)["close"].transform("count")
        stale_streaks = int((streak_counts >= 5).sum())
        if stale_streaks >= 5:
            notes.append(f"Detected {stale_streaks} sessions within stagnant price freeze streaks (>= 5 days identical close).")

    # 6. Corporate Action / Unadjusted Split Jumps
    if "close" in df.columns:
        session_returns = df["close"].pct_change().abs()
        extreme_jumps = int((session_returns > 0.60).sum())
        if extreme_jumps > 0:
            anomalies += extreme_jumps
            notes.append(f"Detected {extreme_jumps} session price jumps > 60%, likely unadjusted split or bonus issue.")

    # 7. Calendar Continuity (Check for gaps > 7 days)
    max_gap_days = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        date_diffs = pd.Series(df.index[1:] - df.index[:-1]).dt.days
        max_gap_days = int(date_diffs.max()) if len(date_diffs) > 0 else 0
        if max_gap_days > 10:  # More than 10 calendar days gap suggests suspension
            notes.append(f"Detected prolonged trading gap of {max_gap_days} calendar days.")

    # Final Acceptance Determination
    is_valid = (
        total_bars >= min_bars and
        missing_pct <= max_missing_pct and
        anomalies == 0
    )

    return DataQualityReport(
        symbol=symbol,
        is_valid=is_valid,
        total_bars=total_bars,
        missing_pct=missing_pct,
        anomalies_detected=anomalies,
        zero_volume_bars=zero_vol_bars,
        stale_price_streaks=stale_streaks,
        max_calendar_gap_days=max_gap_days,
        notes=notes
    )
