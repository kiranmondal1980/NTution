"""
NSE MOMENTUM 5™ — Strategy B: Breakout + Volume Confirmation
================================================================================
Detects genuine institutional price breakouts through historical multi-week
resistance pivots confirmed by significant volume surges and candle conviction.

Entry Conditions:
1. Pivot Breakout: Today's Close breaks above shifted 20-Day or 50-Day High
2. Volume Surge: 5-Day volume ratio >= 1.4x (or 20D volume ratio >= 1.5x)
3. Proximity Constraint: 0.0% <= Distance from Pivot <= 5.0%
   - Forbids chasing over-extended breakouts where risk/reward is compromised.
4. Candle Conviction: Close Location Value (CLV) >= +0.20 (closed in upper half)
5. Regime Filter: Broad market permission must be active (is_regime_permitted == True)

Stop & Target Definitions:
- Hard Stop Reference: max(Prior Breakout Level * 0.99, Close - (1.8 * ATR 14))
- Target Reference: Close + (15% upside research target)
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("STRATEGY_B_BREAKOUT")


class BreakoutVolumeStrategy:
    """Quantitative Breakout & Volume Confirmation Strategy (Strategy B)."""

    def __init__(
        self,
        min_vol_ratio: float = 1.40,
        max_pivot_distance_pct: float = 0.05,
        min_clv: float = 0.20,
        atr_stop_multiplier: float = 1.80
    ):
        self.min_vol_ratio = min_vol_ratio
        self.max_pivot_distance_pct = max_pivot_distance_pct
        self.min_clv = min_clv
        self.atr_stop_multiplier = atr_stop_multiplier

    def evaluate_bar(
        self,
        symbol: str,
        bar: pd.Series,
        is_regime_permitted: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a single session's feature record for a breakout volume signal.

        Args:
            symbol: Ticker symbol.
            bar: Series containing engineered breakout, volume, and volatility features.
            is_regime_permitted: Flag from MarketRegimeEngine indicating market health.

        Returns:
            Signal dictionary if all conditions are met, otherwise None.
        """
        if not is_regime_permitted:
            return None

        close = float(bar.get("close", 0.0) or 0.0)
        if close <= 0:
            return None

        bo_20 = int(bar.get("breakout_20d", 0) or 0)
        bo_50 = int(bar.get("breakout_50d", 0) or 0)
        prior_high_20d = float(bar.get("prior_high_20d", 0.0) or 0.0)
        dist_bo20 = float(bar.get("dist_breakout_20d_pct", -1.0) or -1.0)
        vol_r5 = float(bar.get("vol_ratio_5d", 1.0) or 1.0)
        vol_r20 = float(bar.get("vol_ratio_20d", 1.0) or 1.0)
        clv = float(bar.get("clv", 0.0) or 0.0)
        atr = float(bar.get("atr_14", close * 0.025) or (close * 0.025))

        # Must be a 20-day or 50-day breakout
        if bo_20 != 1 and bo_50 != 1:
            return None

        # Must not be over-extended beyond pivot tolerance (0% to +5%)
        if not (0.0 <= dist_bo20 <= self.max_pivot_distance_pct):
            return None

        # Must have volume expansion
        has_vol_surge = (vol_r5 >= self.min_vol_ratio) or (vol_r20 >= 1.50)
        if not has_vol_surge:
            return None

        # Candle conviction: Close Location Value check
        if clv < self.min_clv:
            return None

        # Define breakout reference level broken
        level_name = "50-Day High" if bo_50 == 1 else "20-Day High"
        pivot_price = prior_high_20d if prior_high_20d > 0 else (close * 0.98)

        # Stop loss: protect capital just below the breakout pivot or 1.8x ATR
        stop_loss = max(pivot_price * 0.99, close - (self.atr_stop_multiplier * atr))
        target_price = close * 1.15  # +15% research target
        risk_dist = close - stop_loss
        reward_dist = target_price - close
        rr_ratio = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        reasons = [
            f"Confirmed breakout above {level_name} (Pivot: {pivot_price:.2f})",
            f"Volume surge: {vol_r5:.1f}x 5D average volume",
            f"Priced near pivot (+{dist_bo20*100:.1f}% extension, within {self.max_pivot_distance_pct*100:.0f}% tolerance)",
            f"Bullish candle close (CLV: {clv:+.2f})"
        ]

        return {
            "strategy": "STRATEGY_B_BREAKOUT_VOLUME",
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
        Scans all universe equities for active breakout setups.

        Returns:
            List of valid signal dictionaries sorted by strongest volume ratio.
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

        signals.sort(key=lambda s: float(s.get("reasons", ["0"])[1].split()[2].replace("x", "")), reverse=True)
        return signals
