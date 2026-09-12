"""
NSE MOMENTUM 5™ — Volatility, Oscillators & Range Dynamics Feature Engineering
================================================================================
Calculates True Range, Average True Range (ATR), ATR%, Historical Volatility,
Bollinger Bands, RSI, ADX, and MACD indicators.

Features Computed:
1. True Range (TR) & Average True Range (Wilder's Smoothing ATR 14)
2. ATR Percentage: (ATR 14 / Close) * 100.0 (Swing volatility sizing metric)
3. Annualized Historical Volatility (20-day rolling log-return standard deviation)
4. Volatility Expansion / Contraction Ratio (ATR 14 vs 20-session average ATR)
5. Bollinger Bands (20-period, 2.0 std):
   - Upper, Lower, Middle bands
   - Bollinger Band Width (volatility compression/squeeze indicator)
   - Bollinger Band %B Position (0.0 = Lower band, 1.0 = Upper band)
6. Relative Strength Index (RSI 14) & 3-Session RSI Slope:
   - Evaluates momentum regimes (55–75 is institutional momentum sweet spot)
   - Adheres strictly to the principle: Elevated RSI (>70) is NOT an automatic sell
7. Average Directional Index (ADX 14, +DI, -DI) for trend strength
8. MACD (12, 26, 9): MACD Line, Signal Line, and MACD Histogram

LEAKAGE PROTECTION:
All operations are strictly backward-looking.
================================================================================
"""

from typing import Optional
import numpy as np
import pandas as pd


def calculate_volatility_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    rsi_period: int = 14,
    adx_period: int = 14
) -> pd.DataFrame:
    """
    Computes volatility indicators, Bollinger Bands, RSI, ADX, and MACD.

    Args:
        df: Cleaned OHLCV DataFrame with 'high', 'low', 'close'.
        atr_period: Smoothing lookback for Average True Range.
        rsi_period: Lookback period for Relative Strength Index.
        adx_period: Lookback period for Average Directional Index.

    Returns:
        pd.DataFrame containing engineered volatility indicators aligned to df.index.
    """
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # 1. True Range (TR)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["tr"] = tr

    # 2. Average True Range (Wilder's Exponential Smoothing: alpha = 1 / period)
    res["atr_14"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
    # ATR Percentage (Volatility relative to price)
    safe_close = close.replace(0, np.nan)
    res["atr_pct"] = (res["atr_14"] / safe_close) * 100.0

    # Volatility Expansion Ratio (Current ATR vs 20-day average ATR)
    atr_avg_20 = res["atr_14"].rolling(window=20).mean().replace(0, np.nan)
    res["volatility_expansion_ratio"] = (res["atr_14"] / atr_avg_20).fillna(1.0)

    # 3. 20-Day Annualized Historical Volatility
    log_returns = np.log(close / prev_close)
    res["volatility_20d"] = log_returns.rolling(window=20).std() * np.sqrt(252)

    # 4. Bollinger Bands (20-period, 2-standard deviations)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    res["bb_upper"] = bb_mid + (2.0 * bb_std)
    res["bb_lower"] = bb_mid - (2.0 * bb_std)
    res["bb_mid"] = bb_mid

    safe_bb_mid = bb_mid.replace(0, np.nan)
    res["bb_width"] = ((res["bb_upper"] - res["bb_lower"]) / safe_bb_mid).fillna(0.0)

    # %B Position: (Close - Lower) / (Upper - Lower)
    band_range = (res["bb_upper"] - res["bb_lower"]).replace(0, np.nan)
    res["bb_pos"] = ((close - res["bb_lower"]) / band_range).fillna(0.5)

    # Bollinger Band Squeeze (Width in bottom 20th percentile over past 60 sessions)
    bb_width_p20 = res["bb_width"].rolling(window=60).quantile(0.20)
    res["bb_squeeze"] = (res["bb_width"] <= bb_width_p20).astype(int)

    # 5. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain, index=df.index).ewm(
        alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False
    ).mean()
    avg_loss = pd.Series(loss, index=df.index).ewm(
        alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False
    ).mean()

    safe_avg_loss = avg_loss.replace(0, np.nan)
    rs = (avg_gain / safe_avg_loss).fillna(1.0)
    res["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    res["rsi_14"] = res["rsi_14"].fillna(50.0)
    # 3-Session RSI Momentum Slope
    res["rsi_slope_3d"] = res["rsi_14"] - res["rsi_14"].shift(3)

    # 6. Average Directional Index (ADX 14)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smooth_tr = tr.ewm(alpha=1.0 / adx_period, min_periods=adx_period, adjust=False).mean().replace(0, np.nan)
    smooth_plus_dm = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / adx_period, min_periods=adx_period, adjust=False
    ).mean()
    smooth_minus_dm = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / adx_period, min_periods=adx_period, adjust=False
    ).mean()

    res["plus_di"] = (100.0 * smooth_plus_dm / smooth_tr).fillna(0.0)
    res["minus_di"] = (100.0 * smooth_minus_dm / smooth_tr).fillna(0.0)

    di_sum = (res["plus_di"] + res["minus_di"]).replace(0, np.nan)
    dx = (100.0 * (res["plus_di"] - res["minus_di"]).abs() / di_sum).fillna(0.0)
    res["adx_14"] = dx.ewm(alpha=1.0 / adx_period, min_periods=adx_period, adjust=False).mean().fillna(0.0)

    # 7. MACD (12, 26, 9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    res["macd_line"] = ema_fast - ema_slow
    res["macd_signal"] = res["macd_line"].ewm(span=9, adjust=False).mean()
    res["macd_hist"] = res["macd_line"] - res["macd_signal"]

    return res
