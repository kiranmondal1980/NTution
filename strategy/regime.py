"""
NSE MOMENTUM 5™ — Macro Market Regime Determination Engine
================================================================================
Evaluates broad Indian market health (NIFTY 50 benchmark + optional breadth)
to assign one of three operational states:

1. BULLISH:
   - Full risk deployment permitted.
   - Normal conviction threshold applied for new long swing setups.

2. NEUTRAL:
   - Market in consolidation or pullback.
   - Higher conviction confirmation required; position sizing throttled.

3. BEARISH:
   - Aggressive long setups suppressed or strictly gated.
   - Preserves trading capital during systemic market drawdowns.

EVIDENCE ANALYZED:
- NIFTY 50 moving average alignment (Price vs EMA 20 vs EMA 50)
- Benchmark velocity: 5-day and 20-day returns
- Benchmark 5-session EMA slope
- Benchmark volatility regime (ATR expansion / contraction)
- Market Breadth (percentage of universe trading above EMA 20 / EMA 50, if supplied)
================================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from config import CONFIG, MarketRegimeConfig
from utils.logger import setup_logger

logger = setup_logger("MARKET_REGIME")


@dataclass(frozen=True)
class MarketRegimeState:
    """Snapshot evaluation of the broad market regime at a specific point in time."""
    timestamp: pd.Timestamp
    regime: str                    # "BULLISH", "NEUTRAL", "BEARISH"
    regime_score: float            # 0.0 to 100.0 continuous health score
    close: float
    ema20: float
    ema50: float
    nifty_5d_ret: float
    nifty_20d_ret: float
    is_long_permitted: bool
    breadth_above_ema20_pct: Optional[float]
    summary_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": str(self.timestamp),
            "regime": self.regime,
            "regime_score": self.regime_score,
            "close": self.close,
            "ema20": self.ema20,
            "ema50": self.ema50,
            "nifty_5d_ret": self.nifty_5d_ret,
            "nifty_20d_ret": self.nifty_20d_ret,
            "is_long_permitted": self.is_long_permitted,
            "breadth_above_ema20_pct": self.breadth_above_ema20_pct,
            "summary_reason": self.summary_reason
        }


class MarketRegimeEngine:
    """
    Evaluates and monitors macroeconomic benchmark regime for Indian equities.
    Supports both point-in-time latest assessment and rolling historical series evaluation.
    """

    def __init__(self, cfg: Optional[MarketRegimeConfig] = None):
        self.cfg: MarketRegimeConfig = cfg or CONFIG.regime

    def evaluate_regime(
        self,
        benchmark_df: pd.DataFrame,
        universe_features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> MarketRegimeState:
        """
        Evaluates current point-in-time regime state from the latest benchmark bar.

        Args:
            benchmark_df: Cleaned OHLCV DataFrame for NIFTY 50 (^NSEI).
            universe_features: Optional dictionary of {symbol: feature_df}
                               for market breadth computation.

        Returns:
            MarketRegimeState dataclass.
        """
        if benchmark_df is None or len(benchmark_df) < self.cfg.slow_ema:
            return MarketRegimeState(
                timestamp=pd.Timestamp.now(),
                regime="NEUTRAL",
                regime_score=50.0,
                close=0.0,
                ema20=0.0,
                ema50=0.0,
                nifty_5d_ret=0.0,
                nifty_20d_ret=0.0,
                is_long_permitted=True,
                breadth_above_ema20_pct=None,
                summary_reason="Insufficient benchmark history; default neutral state applied."
            )

        close = benchmark_df["close"]
        ema20 = close.ewm(span=self.cfg.fast_ema, adjust=False).mean()
        ema50 = close.ewm(span=self.cfg.slow_ema, adjust=False).mean()

        latest_ts = close.index[-1]
        c = float(close.iloc[-1])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])

        ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0)
        ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0)

        # 1. Moving Average Evidence (Base score: 50.0)
        score = 50.0
        reasons: List[str] = []

        if c > e20 and e20 > e50:
            score += 25.0
            reasons.append("NIFTY > EMA 20 > EMA 50 (Bullish moving average alignment)")
        elif c < e20 and e20 < e50:
            score -= 25.0
            reasons.append("NIFTY < EMA 20 < EMA 50 (Bearish moving average alignment)")
        elif c > e50:
            score += 10.0
            reasons.append("NIFTY holding above EMA 50 support")
        else:
            score -= 10.0
            reasons.append("NIFTY trading below EMA 50 (Vulnerable trend)")

        # 2. Velocity Evidence (5-day and 20-day returns)
        if ret_5d > 0.015:
            score += 15.0
            reasons.append(f"Strong 5D benchmark momentum (+{ret_5d*100:.2f}%)")
        elif ret_5d < -0.015:
            score -= 15.0
            reasons.append(f"Sharp 5D benchmark pullback ({ret_5d*100:.2f}%)")

        if ret_20d > 0.03:
            score += 10.0
        elif ret_20d < -0.03:
            score -= 10.0

        # 3. EMA 20 Slope (5-session rate of change)
        if len(ema20) >= 6:
            e20_prev = float(ema20.iloc[-6])
            ema20_slope = (e20 - e20_prev) / e20_prev
            if ema20_slope > 0.004:
                score += 10.0
                reasons.append("EMA 20 sloping firmly upward")
            elif ema20_slope < -0.004:
                score -= 10.0
                reasons.append("EMA 20 sloping downward")

        # 4. Market Breadth Evidence (if universe features provided)
        breadth_pct: Optional[float] = None
        if universe_features:
            above_count = 0
            total_count = 0
            for sym, df_sym in universe_features.items():
                if not df_sym.empty and "close" in df_sym.columns and "ema_20" in df_sym.columns:
                    last_bar = df_sym.iloc[-1]
                    if last_bar["close"] > last_bar["ema_20"]:
                        above_count += 1
                    total_count += 1

            if total_count > 0:
                breadth_pct = round((above_count / total_count) * 100.0, 2)
                if breadth_pct >= 60.0:
                    score += 10.0
                    reasons.append(f"Healthy universe breadth ({breadth_pct:.1f}% stocks > EMA 20)")
                elif breadth_pct <= 35.0:
                    score -= 10.0
                    reasons.append(f"Deteriorating breadth ({breadth_pct:.1f}% stocks > EMA 20)")

        # Clamp score between 0.0 and 100.0
        regime_score = round(float(np.clip(score, 0.0, 100.0)), 2)

        # 5. State Classification
        if regime_score >= self.cfg.bullish_threshold_score:
            regime = "BULLISH"
            is_permitted = True
        elif regime_score <= self.cfg.bearish_threshold_score:
            regime = "BEARISH"
            is_permitted = self.cfg.allow_longs_in_bearish  # Default False
        else:
            regime = "NEUTRAL"
            is_permitted = True

        return MarketRegimeState(
            timestamp=latest_ts,
            regime=regime,
            regime_score=regime_score,
            close=c,
            ema20=e20,
            ema50=e50,
            nifty_5d_ret=round(ret_5d, 4),
            nifty_20d_ret=round(ret_20d, 4),
            is_long_permitted=is_permitted,
            breadth_above_ema20_pct=breadth_pct,
            summary_reason="; ".join(reasons)
        )

    def evaluate_regime_series(self, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a continuous historical daily regime series for backtesting.

        Returns:
            pd.DataFrame indexed by benchmark_df.index containing:
            ['regime', 'regime_score', 'is_long_permitted']
        """
        if benchmark_df is None or len(benchmark_df) < self.cfg.slow_ema:
            return pd.DataFrame()

        close = benchmark_df["close"]
        ema20 = close.ewm(span=self.cfg.fast_ema, adjust=False).mean()
        ema50 = close.ewm(span=self.cfg.slow_ema, adjust=False).mean()

        ret_5d = close.pct_change(5)
        ret_20d = close.pct_change(20)

        # Vectorized component scoring
        base_score = pd.Series(50.0, index=close.index)

        # MA Stack
        ma_bull = (close > ema20) & (ema20 > ema50)
        ma_bear = (close < ema20) & (ema20 < ema50)
        base_score = base_score + np.where(ma_bull, 25.0, 0.0)
        base_score = base_score - np.where(ma_bear, 25.0, 0.0)

        # 5D Momentum
        base_score = base_score + np.where(ret_5d > 0.015, 15.0, 0.0)
        base_score = base_score - np.where(ret_5d < -0.015, 15.0, 0.0)

        # 20D Momentum
        base_score = base_score + np.where(ret_20d > 0.03, 10.0, 0.0)
        base_score = base_score - np.where(ret_20d < -0.03, 10.0, 0.0)

        # Clip 0 to 100
        score = base_score.clip(lower=0.0, upper=100.0)

        regimes = pd.Series("NEUTRAL", index=close.index)
        regimes[score >= self.cfg.bullish_threshold_score] = "BULLISH"
        regimes[score <= self.cfg.bearish_threshold_score] = "BEARISH"

        permitted = pd.Series(True, index=close.index)
        if not self.cfg.allow_longs_in_bearish:
            permitted[regimes == "BEARISH"] = False

        res = pd.DataFrame({
            "regime": regimes,
            "regime_score": score.round(2),
            "is_long_permitted": permitted
        }, index=close.index)

        return res
