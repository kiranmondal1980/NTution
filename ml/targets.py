"""
NSE MOMENTUM 5™ — Machine Learning Research Target Construction
================================================================================
Constructs strictly formulated forward-looking binary classification labels
and continuous excursion targets for quantitative probability modeling.

EXECUTION ASSUMPTIONS & TIMING CONTRACT:
1. Signal Generation Point: Session T at the Market Close.
2. Hypothetical Execution Point: Session T+1 at the Market Open.
3. Forward High Horizons:
   - 3-Session Horizon: Max intraday High across sessions [T+1, T+2, T+3].
   - 5-Session Horizon: Max intraday High across sessions [T+1, T+2, T+3, T+4, T+5].
4. Target Formulation:
   - Gain is measured relative to Session T+1 Open:
     Forward_Gain = (Forward_Max_High - Open[T+1]) / Open[T+1]

BINARY TARGETS CONSTRUCTED:
- target_5pct_3d  : 1 if Forward_Gain_3D >= +0.05 (+5%), else 0
- target_10pct_5d : 1 if Forward_Gain_5D >= +0.10 (+10%), else 0
- target_15pct_5d : 1 if Forward_Gain_5D >= +0.15 (+15%), else 0
- target_20pct_5d : 1 if Forward_Gain_5D >= +0.20 (+20%), else 0

LEAKAGE TRUNCATION:
The trailing 5 rows of any dataset are explicitly marked NaN because their
forward realization horizons have not yet concluded.
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def create_research_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs forward-looking probability targets based on next-day open entry.

    Args:
        df: Cleaned OHLCV DataFrame with 'open', 'high', 'low', 'close'.

    Returns:
        pd.DataFrame containing binary event targets and continuous forward metrics,
        aligned to df.index with incomplete trailing rows masked as NaN.
    """
    if df is None or df.empty or len(df) < 6:
        return pd.DataFrame()

    targets = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    next_open = df["open"].shift(-1)  # Entry price at Session T+1 Open

    # --------------------------------------------------------------------------
    # 1. Forward Max High & Min Low Horizons
    # --------------------------------------------------------------------------
    # 3-session window: Sessions T+1, T+2, T+3
    # Shift high by -1, rolling 3, then shift by -2 to align back to Session T
    fwd_high_3d = high.shift(-1).rolling(window=3).max().shift(-2)

    # 5-session window: Sessions T+1, T+2, T+3, T+4, T+5
    fwd_high_5d = high.shift(-1).rolling(window=5).max().shift(-4)

    # Forward Minimum Low over 5 sessions (Maximum Adverse Excursion reference)
    fwd_low_5d = low.shift(-1).rolling(window=5).min().shift(-4)

    # --------------------------------------------------------------------------
    # 2. Continuous Percentage Return relative to Next Open
    # --------------------------------------------------------------------------
    safe_open = next_open.replace(0, np.nan)
    gain_3d = (fwd_high_3d - safe_open) / safe_open
    gain_5d = (fwd_high_5d - safe_open) / safe_open
    drawdown_5d = (fwd_low_5d - safe_open) / safe_open

    targets["fwd_max_gain_3d"] = gain_3d
    targets["fwd_max_gain_5d"] = gain_5d
    targets["fwd_max_drawdown_5d"] = drawdown_5d

    # --------------------------------------------------------------------------
    # 3. Binary Classification Targets
    # --------------------------------------------------------------------------
    targets["target_5pct_3d"] = np.where(gain_3d >= 0.05, 1, 0).astype(float)
    targets["target_10pct_5d"] = np.where(gain_5d >= 0.10, 1, 0).astype(float)
    targets["target_15pct_5d"] = np.where(gain_5d >= 0.15, 1, 0).astype(float)
    targets["target_20pct_5d"] = np.where(gain_5d >= 0.20, 1, 0).astype(float)

    # --------------------------------------------------------------------------
    # 4. Mandatory Incomplete-Window Masking
    # --------------------------------------------------------------------------
    # The last 5 rows cannot observe the full forward 5-session outcome
    targets.iloc[-5:] = np.nan

    return targets
