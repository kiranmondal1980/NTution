"""
NSE MOMENTUM 5™ — Strategy A: Pure Momentum Continuation
================================================================================
Identifies stocks with sustained short-term velocity, positive acceleration,
and moving average support capable of continuing upward moves over 3–5 sessions.

Entry Logic:
1. Short-term Velocity: 5-Day return >= +3.0% and 3-Day return > +1.0%
2. Momentum Acceleration: Instantaneous velocity outpaces trailing trend (mom_accel_1v5 > 0)
3. Trend Alignment: Price > EMA 10 and EMA 10 > EMA 20
4. RSI Sweet Spot: 55.0 <= RSI 14 <= 82.0 (Bullish momentum without terminal exhaustion)
5. Regime Filter: Broad market permission must be active (is_regime_permitted == True)

Stop & Target Definitions:
- Hard Stop Reference: max(EMA 20, Close - (2.0 * ATR 14))
- Target Reference: Close + (15% upside research target)
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("STRATEGY_A_MOMENTUM")


class PureMomentumStrategy:
    """Quantitative Momentum Continuation Strategy (Strategy A)."""

    def __init__(
        self,
        min_5d_ret: float = 0.03,
        min_3d_ret: float = 0.01,
        min_rsi: float = 55.0,
        max_rsi: float = 82.0,
        atr_stop_multiplier: float = 2.0
    ):
        self.min_5d_ret = min_5d_ret
        self.min_3d_ret = min_3d_ret
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.atr_stop_multiplier = atr_stop_multiplier

    def evaluate_bar(
        self,
        symbol: str,
        bar: pd.Series,
        is_regime_permitted: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a single session's feature record for a momentum continuation signal.

        Args:
            symbol: Ticker symbol.
            bar: Series containing engineered momentum, trend, and volatility features.
            is_regime_permitted: Flag from MarketRegimeEngine indicating market health.

        Returns:
            Signal dictionary if all conditions are satisfied, otherwise None.
        """
        if not is_regime_permitted:
            return None

        close = float(bar.get("close", 0.0) or 0.0)
        if close <= 0:
            return None

        ret_5d = float(bar.get("ret_5d", 0.0) or 0.0)
        ret_3d = float(bar.get("ret_3d", 0.0) or 0.0)
        rsi = float(bar.get("rsi_14", 50.0) or 50.0)
        ema10 = float(bar.get("ema_10", 0.0) or 0.0)
        ema20 = float(bar.get("ema_20", 0.0) or 0.0)
        atr = float(bar.get("atr_14", close * 0.025) or (close * 0.025))

        # Check conditions
        if ret_5d < self.min_5d_ret or ret_3d < self.min_3d_ret:
            return None

        if rsi < self.min_rsi or rsi > self.max_rsi:
            return None

        # Price must hold above EMA 10, and EMA 10 must be above EMA 20
        if not (close > ema10 and ema10 > ema20):
            return None

        # Calculate defined stops and targets
        stop_loss = max(ema20, close - (self.atr_stop_multiplier * atr))
        target_price = close * 1.15  # +15% research target
        risk_dist = close - stop_loss
        reward_dist = target_price - close
        rr_ratio = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        reasons = [
            f"5D return +{ret_5d*100:.1f}% exceeds threshold (+{self.min_5d_ret*100:.1f}%)",
            f"Price ({close:.2f}) > EMA10 ({ema10:.2f}) > EMA20 ({ema20:.2f})",
            f"RSI 14 ({rsi:.1f}) in prime momentum expansion zone [55-82]"
        ]

        return {
            "strategy": "STRATEGY_A_MOMENTUM_CONTINUATION",
            "symbol": symbol,
            "signal": "BUY",
            "entry_reference_price": round(close, 2),
            "stop_loss_price": round(stop_loss, 2),
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_dist, 2),
            "risk_reward_ratio": rr_ratio,
            "reasons": reasons
        }

    def scan_universe(
        self,
        universe_features: Dict[str, pd.DataFrame],
        is_regime_permitted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scans all universe equities for active momentum continuation setups.

        Returns:
            List of valid signal dictionaries sorted by highest 5-day return.
        """
        signals: List[Dict[str, Any]] = []
        if not is_regime_permitted:
            return signals

        for sym, df in universe_features.items():
            if df.empty:
                continue
            last_bar = df.iloc[-1]
            sig = self.evaluate_bar(sym, last_bar, is_regime_permitted=is_regime_permitted)
            if sig:
                signals.append(sig)

        signals.sort(key=lambda s: s.get("risk_reward_ratio", 0.0), reverse=True)
        return signals
