"""
NSE MOMENTUM 5™ — Strategy E: Multi-Factor Quantitative Momentum
================================================================================
Comprehensive quantitative strategy combining Price Momentum, Benchmark-Relative
Strength, Volume Expansion, Breakout Dynamics, and Volatility Banding.

Entry Logic:
1. Score Threshold: Multi-pillar Momentum Score >= 75.0 (MOMENTUM or STRONG MOMENTUM)
2. Relative Outperformance: Positive 5-day excess return vs NIFTY 50 (rs_5d > 0)
3. Volume Confirmation: 5-Day volume ratio >= 1.2x (or institutional surge flag)
4. Volatility Band: ATR% between 1.8% and 5.5% (sufficient swing range, bounded risk)
5. Regime Filter: Broad market permission must be active (is_regime_permitted == True)

Stop & Target Definitions:
- Hard Stop Reference: max(EMA 20, Close - (2.0 * ATR 14))
- Target Reference: Close + (15% upside research target)
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from strategy.scoring import MomentumScoringEngine
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("STRATEGY_E_MULTI_FACTOR")


class MultiFactorMomentumStrategy:
    """Multi-Factor Quantitative Momentum Strategy (Strategy E)."""

    def __init__(
        self,
        min_momentum_score: float = CONFIG.signal.momentum,
        require_positive_alpha: bool = True,
        min_vol_ratio: float = 1.15,
        atr_stop_multiplier: float = 2.0
    ):
        self.min_momentum_score = min_momentum_score
        self.require_positive_alpha = require_positive_alpha
        self.min_vol_ratio = min_vol_ratio
        self.atr_stop_multiplier = atr_stop_multiplier
        self.scorer = MomentumScoringEngine()

    def evaluate_bar(
        self,
        symbol: str,
        bar: pd.Series,
        regime_score: float = 50.0,
        is_regime_permitted: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a single session's feature record for a multi-factor momentum signal.

        Args:
            symbol: Ticker symbol.
            bar: Series containing engineered multi-pillar features.
            regime_score: Macro NIFTY 50 health score (0.0 to 100.0).
            is_regime_permitted: Regime permission state.

        Returns:
            Signal dictionary if all quantitative factors align, otherwise None.
        """
        if not is_regime_permitted:
            return None

        close = float(bar.get("close", 0.0) or 0.0)
        if close <= 0:
            return None

        # 1. Evaluate Multi-Pillar Score
        scoring_res = self.scorer.score_record(bar, regime_score=regime_score)
        score = scoring_res["momentum_score"]
        if score < self.min_momentum_score:
            return None

        # 2. Benchmark Relative Strength Confirmation
        rs_5d = float(bar.get("rs_5d", 0.0) or 0.0)
        if self.require_positive_alpha and rs_5d <= 0.0:
            return None

        # 3. Volume Surge Confirmation
        vol_r5 = float(bar.get("vol_ratio_5d", 1.0) or 1.0)
        vol_surge_flag = int(bar.get("institutional_volume_surge", 0) or 0)
        if vol_r5 < self.min_vol_ratio and vol_surge_flag == 0:
            return None

        # 4. Volatility Sizing Channel (ATR% between 1.8% and 6.0%)
        atr_pct = float(bar.get("atr_pct", 2.5) or 2.5)
        if not (1.8 <= atr_pct <= 6.0):
            return None

        atr = float(bar.get("atr_14", close * 0.025) or (close * 0.025))
        ema20 = float(bar.get("ema_20", close * 0.95) or (close * 0.95))

        # 5. Define Stops and Targets
        stop_loss = max(ema20, close - (self.atr_stop_multiplier * atr))
        target_price = close * 1.15  # +15% research target
        risk_dist = close - stop_loss
        reward_dist = target_price - close
        rr_ratio = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        reasons = [
            f"Multi-Factor Score: {score}/100 ({scoring_res['category']})",
            f"Alpha vs NIFTY 5D: {rs_5d*100:+.2f}%",
            f"Volume Ratio: {vol_r5:.2f}x (Surge flag: {vol_surge_flag})",
            f"ATR: {atr_pct:.1f}% within optimal swing risk channel"
        ] + scoring_res.get("reasons", [])[:2]

        return {
            "strategy": "STRATEGY_E_MULTI_FACTOR_MOMENTUM",
            "symbol": symbol,
            "signal": "BUY",
            "momentum_score": score,
            "category": scoring_res["category"],
            "pillar_breakdown": scoring_res["pillar_breakdown"],
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
        regime_score: float = 50.0,
        is_regime_permitted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scans all universe equities for active multi-factor setups.

        Returns:
            List of valid signal dictionaries sorted by highest multi-factor score.
        """
        signals: List[Dict[str, Any]] = []
        if not is_regime_permitted:
            return signals

        for sym, df in universe_features.items():
            if df.empty:
                continue
            last_bar = df.iloc[-1]
            sig = self.evaluate_bar(
                symbol=sym,
                bar=last_bar,
                regime_score=regime_score,
                is_regime_permitted=is_regime_permitted
            )
            if sig:
                signals.append(sig)

        signals.sort(key=lambda s: s.get("momentum_score", 0.0), reverse=True)
        return signals
