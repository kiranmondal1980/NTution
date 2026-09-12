"""
NSE MOMENTUM 5™ — Transparent 0–100 Momentum Scoring Engine
================================================================================
Calculates a multi-pillar quantitative momentum score (0 to 100) from engineered
indicators. Every point is transparently attributed to an underlying pillar.

INITIAL RESEARCH WEIGHTS (CONFIGURABLE):
1. Price Momentum:        20 points (1D, 3D, 5D velocity and regression slope)
2. Relative Strength:     15 points (Excess return vs NIFTY 50 and RS streak)
3. Volume Quality:        15 points (Surge ratio, up/down volume skew, CLV)
4. Breakout Strength:     15 points (20D/50D shifted pivot penetration & persistence)
5. Trend Alignment:       10 points (EMA 5 > 10 > 20 > 50 stack & distance to EMA20)
6. Volatility Suitability:10 points (Optimal 2.0% - 5.0% ATR% swing channel)
7. Market Regime:         10 points (NIFTY 50 macroeconomic health score)
8. Momentum Health:        5 points (RSI 55–75 momentum regime without exhaustion)
TOTAL = 100 POINTS

SIGNAL CLASSIFICATIONS:
- 85 – 100 : STRONG MOMENTUM
- 75 – 84  : MOMENTUM
- 65 – 74  : WATCHLIST
- 50 – 64  : WEAK
- < 50     : AVOID
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from config import CONFIG, ScoringWeightsConfig, SignalThresholdsConfig
from utils.logger import setup_logger

logger = setup_logger("MOMENTUM_SCORING")


class MomentumScoringEngine:
    """
    Modular quantitative scoring engine supporting configurable weighting schemes.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeightsConfig] = None,
        thresholds: Optional[SignalThresholdsConfig] = None
    ):
        self.weights: ScoringWeightsConfig = weights or CONFIG.scoring
        self.thresholds: SignalThresholdsConfig = thresholds or CONFIG.signal

    def score_record(
        self,
        features: Union[pd.Series, Dict[str, Any]],
        regime_score: float = 50.0
    ) -> Dict[str, Any]:
        """
        Calculates individual pillar sub-scores, aggregated score, and signal category
        for a single bar/session record.

        Args:
            features: Row series or dictionary of engineered technical features.
            regime_score: Current NIFTY 50 market regime score (0.0 to 100.0).

        Returns:
            Dictionary with 'momentum_score', 'category', 'pillar_breakdown', and 'reasons'.
        """
        reasons: List[str] = []

        # ----------------------------------------------------------------------
        # Pillar 1: Price Momentum (Weight: 20)
        # ----------------------------------------------------------------------
        ret_1d = float(features.get("ret_1d", 0.0) or 0.0)
        ret_3d = float(features.get("ret_3d", 0.0) or 0.0)
        ret_5d = float(features.get("ret_5d", 0.0) or 0.0)
        slope_5d = float(features.get("mom_slope_5d", 0.0) or 0.0)

        p_mom = 0.0
        if ret_5d >= 0.08:
            p_mom += 0.50
            reasons.append(f"Exceptional 5D return (+{ret_5d*100:.1f}%)")
        elif ret_5d >= 0.04:
            p_mom += 0.40
            reasons.append(f"Strong 5D return (+{ret_5d*100:.1f}%)")
        elif ret_5d >= 0.02:
            p_mom += 0.25
        elif ret_5d > 0.0:
            p_mom += 0.15

        if ret_3d >= 0.03:
            p_mom += 0.30
        elif ret_3d > 0.01:
            p_mom += 0.20

        if slope_5d > 0.005:
            p_mom += 0.20

        score_mom = min(1.0, p_mom) * self.weights.price_momentum

        # ----------------------------------------------------------------------
        # Pillar 2: Relative Strength vs NIFTY 50 (Weight: 15)
        # ----------------------------------------------------------------------
        rs_5d = float(features.get("rs_5d", 0.0) or 0.0)
        rs_20d = float(features.get("rs_20d", 0.0) or 0.0)
        rs_streak = int(features.get("rs_outperform_days_10d", 5) or 5)

        p_rs = 0.0
        if rs_5d >= 0.04:
            p_rs += 0.50
            reasons.append(f"High 5D alpha vs NIFTY (+{rs_5d*100:.1f}%)")
        elif rs_5d >= 0.02:
            p_rs += 0.35
        elif rs_5d > 0.0:
            p_rs += 0.20

        if rs_20d >= 0.05:
            p_rs += 0.30
        elif rs_20d > 0.0:
            p_rs += 0.15

        if rs_streak >= 7:
            p_rs += 0.20
            reasons.append(f"Persistent alpha ({rs_streak}/10 sessions beat NIFTY)")

        score_rs = min(1.0, p_rs) * self.weights.relative_strength

        # ----------------------------------------------------------------------
        # Pillar 3: Volume Quality & Expansion (Weight: 15)
        # ----------------------------------------------------------------------
        vol_r5 = float(features.get("vol_ratio_5d", 1.0) or 1.0)
        vol_r20 = float(features.get("vol_ratio_20d", 1.0) or 1.0)
        up_vol_ratio = float(features.get("up_volume_ratio_10d", 0.5) or 0.5)
        clv = float(features.get("clv", 0.0) or 0.0)

        p_vol = 0.0
        if vol_r5 >= 2.0:
            p_vol += 0.40
            reasons.append(f"Heavy volume surge ({vol_r5:.1f}x 5D avg)")
        elif vol_r5 >= 1.3:
            p_vol += 0.25

        if up_vol_ratio >= 0.65:
            p_vol += 0.30
            reasons.append("Strong institutional accumulation volume skew")
        elif up_vol_ratio >= 0.52:
            p_vol += 0.15

        if clv >= 0.50:
            p_vol += 0.30
            reasons.append("Bullish candle close near session highs (CLV >= 0.5)")
        elif clv > 0.10:
            p_vol += 0.15

        score_vol = min(1.0, p_vol) * self.weights.volume_quality

        # ----------------------------------------------------------------------
        # Pillar 4: Breakout Strength & Persistence (Weight: 15)
        # ----------------------------------------------------------------------
        bo_20 = int(features.get("breakout_20d", 0) or 0)
        bo_50 = int(features.get("breakout_50d", 0) or 0)
        dist_bo20 = float(features.get("dist_breakout_20d_pct", -0.05) or -0.05)
        persisted = int(features.get("breakout_persisted_2d", 0) or 0)

        p_bo = 0.0
        if bo_50 == 1 and 0.0 <= dist_bo20 <= 0.06:
            p_bo += 0.60
            reasons.append("Fresh 50-day high breakout near pivot level")
        elif bo_20 == 1 and 0.0 <= dist_bo20 <= 0.05:
            p_bo += 0.50
            reasons.append("Clean 20-day high breakout")
        elif bo_20 == 1 and dist_bo20 > 0.05:
            p_bo += 0.30  # Extended breakout, lower reward/risk
        elif dist_bo20 >= -0.02:
            p_bo += 0.20  # Coiling just below pivot

        if persisted == 1:
            p_bo += 0.40
            reasons.append("Breakout confirmed with multi-session persistence")

        score_bo = min(1.0, p_bo) * self.weights.breakout_strength

        # ----------------------------------------------------------------------
        # Pillar 5: Trend Alignment (Weight: 10)
        # ----------------------------------------------------------------------
        stack = int(features.get("bullish_ema_stack", 0) or 0)
        dist_ema20 = float(features.get("dist_ema_20_pct", 0.0) or 0.0)

        p_trend = 0.0
        if stack == 1:
            p_trend += 0.60
            reasons.append("Full bullish EMA stack (Price > EMA5 > EMA10 > EMA20 > EMA50)")
        elif dist_ema20 > 0.0:
            p_trend += 0.30

        # Optimal swing distance from EMA 20: 1% to 6% (not over-extended)
        if 0.01 <= dist_ema20 <= 0.06:
            p_trend += 0.40
        elif dist_ema20 > 0.06:
            p_trend += 0.15  # Extended from 20-day mean

        score_trend = min(1.0, p_trend) * self.weights.trend_alignment

        # ----------------------------------------------------------------------
        # Pillar 6: Volatility Suitability (Weight: 10)
        # ----------------------------------------------------------------------
        atr_pct = float(features.get("atr_pct", 2.5) or 2.5)

        p_volat = 0.0
        # Optimal swing volatility channel: 2.0% to 5.0%
        if 2.0 <= atr_pct <= 4.5:
            p_volat = 1.0
        elif 1.5 <= atr_pct < 2.0 or 4.5 < atr_pct <= 6.5:
            p_volat = 0.70
        elif atr_pct > 6.5:
            p_volat = 0.30  # Excessively erratic, wide stops needed
        else:
            p_volat = 0.20  # Excessively sluggish

        score_volat = p_volat * self.weights.volatility_suitability

        # ----------------------------------------------------------------------
        # Pillar 7: Market Regime Health (Weight: 10)
        # ----------------------------------------------------------------------
        score_regime = (float(np.clip(regime_score, 0.0, 100.0)) / 100.0) * self.weights.market_regime

        # ----------------------------------------------------------------------
        # Pillar 8: Momentum Health & RSI Regime (Weight: 5)
        # ----------------------------------------------------------------------
        rsi = float(features.get("rsi_14", 50.0) or 50.0)

        p_health = 0.0
        if 60.0 <= rsi <= 75.0:
            p_health = 1.0  # Prime institutional momentum zone
        elif 50.0 <= rsi < 60.0:
            p_health = 0.75
        elif 75.0 < rsi <= 82.0:
            p_health = 0.60  # Elevated but intact
        elif rsi > 82.0:
            p_health = 0.30  # Exhaustion/climax risk
        else:
            p_health = 0.15  # Sub-50 lack of momentum

        score_health = p_health * self.weights.momentum_health

        # ----------------------------------------------------------------------
        # Total Aggregated Score
        # ----------------------------------------------------------------------
        total = (
            score_mom + score_rs + score_vol + score_bo +
            score_trend + score_volat + score_regime + score_health
        )
        total_score = round(float(np.clip(total, 0.0, 100.0)), 2)

        # Categorization based on thresholds
        if total_score >= self.thresholds.strong_momentum:
            category = "STRONG MOMENTUM"
        elif total_score >= self.thresholds.momentum:
            category = "MOMENTUM"
        elif total_score >= self.thresholds.watchlist:
            category = "WATCHLIST"
        elif total_score >= self.thresholds.weak:
            category = "WEAK"
        else:
            category = "AVOID"

        return {
            "momentum_score": total_score,
            "category": category,
            "pillar_breakdown": {
                "price_momentum": round(score_mom, 2),
                "relative_strength": round(score_rs, 2),
                "volume_quality": round(score_vol, 2),
                "breakout_strength": round(score_bo, 2),
                "trend_alignment": round(score_trend, 2),
                "volatility_suitability": round(score_volat, 2),
                "market_regime": round(score_regime, 2),
                "momentum_health": round(score_health, 2)
            },
            "reasons": reasons
        }
