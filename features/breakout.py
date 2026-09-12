"""
NSE MOMENTUM 5™ — Breakout Level, Penetration & Persistence Feature Engineering
================================================================================
Calculates historical price resistance/support levels, breakout triggers,
ATR-normalized penetration strength, and multi-session persistence.

CRITICAL LEAKAGE PROTECTION (MANDATORY RULE):
When calculating breakout levels for today's bar T, NEVER include today's High
or Close in the historical breakout threshold.
All reference levels are computed using `shift(1).rolling(N)` so that today's
bar is strictly compared against the prior historical window [T-N to T-1].

Features Computed:
1. Shifted Historical Highs: 10D, 20D, 50D, and 252D (52-Week High)
2. Shifted Historical Lows: 10D, 20D support boundaries
3. Binary Breakout Triggers: Close > Prior High (10D, 20D, 50D, 52W)
4. Distance to Breakout Thresholds (%): (Close - Prior High) / Prior High
5. ATR-Normalized Breakout Strength: (Close - Prior High 20D) / ATR 14
6. Breakout Persistence: Verified close above breakout pivot for 2+ sessions
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_breakout_features(
    df: pd.DataFrame,
    atr_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Computes anti-leakage shifted resistance levels, breakout flags, and persistence metrics.

    Args:
        df: Cleaned OHLCV DataFrame with 'high', 'low', 'close'.
        atr_series: Optional pre-calculated ATR 14 series for penetration normalization.

    Returns:
        pd.DataFrame containing engineered breakout indicators aligned to df.index.
    """
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # 1. Shifted Historical Highs (Reference window strictly excludes today)
    # Reference for bar T is max(High[T-N] ... High[T-1])
    res["prior_high_10d"] = high.shift(1).rolling(window=10).max()
    res["prior_high_20d"] = high.shift(1).rolling(window=20).max()
    res["prior_high_50d"] = high.shift(1).rolling(window=50).max()
    res["prior_high_252d"] = high.shift(1).rolling(window=252).max()  # 52-Week High

    # Shifted Historical Lows (Support Boundaries)
    res["prior_low_10d"] = low.shift(1).rolling(window=10).min()
    res["prior_low_20d"] = low.shift(1).rolling(window=20).min()

    # 2. Binary Breakout Triggers (Close exceeds historical resistance)
    res["breakout_10d"] = (close > res["prior_high_10d"]).astype(int)
    res["breakout_20d"] = (close > res["prior_high_20d"]).astype(int)
    res["breakout_50d"] = (close > res["prior_high_50d"]).astype(int)
    res["breakout_52w"] = (close > res["prior_high_252d"]).astype(int)

    # Breakdown Triggers (Close breaches historical support)
    res["breakdown_20d"] = (close < res["prior_low_20d"]).astype(int)

    # 3. Percentage Distance to Breakout Thresholds
    # Positive = Above breakout level; Negative = Below resistance
    res["dist_breakout_10d_pct"] = (close - res["prior_high_10d"]) / res["prior_high_10d"].replace(0, np.nan)
    res["dist_breakout_20d_pct"] = (close - res["prior_high_20d"]) / res["prior_high_20d"].replace(0, np.nan)
    res["dist_breakout_50d_pct"] = (close - res["prior_high_50d"]) / res["prior_high_50d"].replace(0, np.nan)
    res["dist_52w_high_pct"] = (close - res["prior_high_252d"]) / res["prior_high_252d"].replace(0, np.nan)

    # Fill invalid historical warmup periods with 0.0
    for col in ["dist_breakout_10d_pct", "dist_breakout_20d_pct", "dist_breakout_50d_pct", "dist_52w_high_pct"]:
        res[col] = res[col].fillna(0.0)

    # 4. ATR-Normalized Breakout Strength
    # Measures penetration depth in units of volatility
    if atr_series is not None and not atr_series.empty:
        safe_atr = atr_series.replace(0, np.nan)
        res["breakout_strength_20d"] = ((close - res["prior_high_20d"]) / safe_atr).fillna(0.0)
    else:
        # Approximate ATR with typical 2.5% price proxy if ATR series not supplied
        res["breakout_strength_20d"] = ((close - res["prior_high_20d"]) / (close * 0.025)).fillna(0.0)

    # 5. Breakout Persistence (Sustained breakout confirmation)
    # Condition: Today's close > prior 20D high AND Yesterday's close was also > its prior 20D high
    prior_high_yesterday = high.shift(2).rolling(window=20).max()
    res["breakout_persisted_2d"] = (
        (close > res["prior_high_20d"]) &
        (close.shift(1) > prior_high_yesterday)
    ).astype(int)

    return res
