"""
NSE MOMENTUM 5™ — Benchmark-Relative Strength (RS) Feature Engineering
================================================================================
Calculates alpha differentials, Mansfield-style relative strength ratios,
and outperformance persistence against the NIFTY 50 benchmark.

Features Computed:
1. Multi-Horizon Excess Returns vs NIFTY 50:
   - rs_1d, rs_3d, rs_5d, rs_10d, rs_20d, rs_60d
   - Formula: Return_Stock(N) - Return_Benchmark(N)
2. Comparative Relative Strength Ratios:
   - Normalized ratio of stock price to benchmark index price
   - rs_ratio_5d, rs_ratio_20d
3. Mansfield Relative Strength:
   - RS_Ratio / SMA(RS_Ratio, 50) - 1.0
4. Outperformance Persistence:
   - Count of sessions over the last 10 trading days where stock beat NIFTY 50
5. Relative Strength Slope (5-session rate of change of alpha)

ALIGNMENT & INTEGRITY:
Stock and benchmark timestamps are strictly aligned by calendar date.
Missing benchmark dates are safely handled with neutral fallbacks (0.0 alpha).
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_relative_strength_features(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Computes alpha differentials and relative performance metrics against the benchmark.

    Args:
        stock_df: OHLCV DataFrame of the target equity.
        benchmark_df: OHLCV DataFrame of NIFTY 50 (or relevant benchmark).
        horizons: List of session lookback windows (defaults to [1, 3, 5, 10, 20, 60]).

    Returns:
        pd.DataFrame containing engineered relative strength features aligned to stock_df.index.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20, 60]

    res = pd.DataFrame(index=stock_df.index)

    # Neutral fallback if benchmark series is missing or insufficient
    if benchmark_df is None or benchmark_df.empty or "close" not in benchmark_df.columns:
        for w in horizons:
            res[f"rs_{w}d"] = 0.0
            res[f"rs_ratio_{w}d"] = 1.0
        res["mansfield_rs"] = 0.0
        res["rs_outperform_days_10d"] = 5
        res["rs_slope_5d"] = 0.0
        return res

    # 1. Align stock close and benchmark close on matching dates
    combined = pd.DataFrame({
        "stock_close": stock_df["close"],
        "bench_close": benchmark_df["close"]
    }).ffill().dropna()

    if combined.empty:
        for w in horizons:
            res[f"rs_{w}d"] = 0.0
            res[f"rs_ratio_{w}d"] = 1.0
        res["mansfield_rs"] = 0.0
        res["rs_outperform_days_10d"] = 5
        res["rs_slope_5d"] = 0.0
        return res

    stock_c = combined["stock_close"]
    bench_c = combined["bench_close"]

    # 2. Multi-Horizon Excess Returns (Stock Return - Benchmark Return)
    for w in horizons:
        stock_ret = stock_c.pct_change(w)
        bench_ret = bench_c.pct_change(w)
        rs_diff = stock_ret - bench_ret
        res[f"rs_{w}d"] = rs_diff.reindex(stock_df.index).fillna(0.0)

        # Comparative performance ratio: (Stock / Benchmark) / Prior(Stock / Benchmark)
        safe_bench_shift = bench_c.shift(w).replace(0, np.nan)
        safe_stock_shift = stock_c.shift(w).replace(0, np.nan)
        ratio = (stock_c / bench_c.replace(0, np.nan)) / (safe_stock_shift / safe_bench_shift)
        res[f"rs_ratio_{w}d"] = ratio.reindex(stock_df.index).fillna(1.0)

    # 3. Mansfield Relative Strength (50-session baseline)
    rs_base_ratio = stock_c / bench_c.replace(0, np.nan)
    rs_base_sma50 = rs_base_ratio.rolling(window=50).mean().replace(0, np.nan)
    mansfield = ((rs_base_ratio / rs_base_sma50) - 1.0) * 100.0
    res["mansfield_rs"] = mansfield.reindex(stock_df.index).fillna(0.0)

    # 4. Outperformance Persistence (Past 10 sessions)
    daily_stock_ret = stock_c.pct_change(1)
    daily_bench_ret = bench_c.pct_change(1)
    daily_beat = (daily_stock_ret > daily_bench_ret).astype(int)
    beat_streak_10d = daily_beat.rolling(window=10).sum()
    res["rs_outperform_days_10d"] = beat_streak_10d.reindex(stock_df.index).fillna(5).astype(int)

    # 5. 5-Session Relative Strength Momentum Slope
    if "rs_5d" in res.columns:
        res["rs_slope_5d"] = res["rs_5d"] - res["rs_5d"].shift(5).fillna(0.0)
    else:
        res["rs_slope_5d"] = 0.0

    return res
