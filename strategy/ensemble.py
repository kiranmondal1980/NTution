"""
NSE MOMENTUM 5™ — Ensemble Quantitative Strategy Engine
================================================================================
Combines independent signals across:
- Strategy A: Pure Momentum Continuation
- Strategy B: Breakout + Volume Confirmation
- Strategy E: Multi-Factor Quantitative Momentum

CORE OBJECTIVE:
Eliminates idiosyncratic strategy false-positives by calculating cross-model
consensus conviction.

Conviction Levels:
- HIGH CONVICTION (3-Way Consensus): Confirmed by Strategy A, B, and E simultaneously.
- MODERATE CONVICTION (2-Way Consensus): Confirmed by 2 independent models.
- SINGLE MODEL CONVICTION: Triggered by 1 model with high individual score.

Harmonized Stop & Target Logic:
- Stop Loss: Most protective (highest) stop price among triggered models.
- Target: Defined +15% research upside target.
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from strategy.momentum import PureMomentumStrategy
from strategy.breakout import BreakoutVolumeStrategy
from strategy.multi_factor import MultiFactorMomentumStrategy
from utils.logger import setup_logger

logger = setup_logger("STRATEGY_ENSEMBLE")


class EnsembleStrategyEngine:
    """Consensus multi-model ensemble engine."""

    def __init__(self):
        self.strat_a = PureMomentumStrategy()
        self.strat_b = BreakoutVolumeStrategy()
        self.strat_e = MultiFactorMomentumStrategy()

    def evaluate_candidate(
        self,
        symbol: str,
        bar: pd.Series,
        regime_score: float = 50.0,
        is_regime_permitted: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates an individual stock's latest bar across all constituent strategies.

        Args:
            symbol: Ticker symbol.
            bar: Series containing all calculated features.
            regime_score: Macro NIFTY 50 health score (0.0 to 100.0).
            is_regime_permitted: Macro market permission state.

        Returns:
            Ensemble consensus dictionary if at least one strategy triggers, otherwise None.
        """
        if not is_regime_permitted:
            return None

        close = float(bar.get("close", 0.0) or 0.0)
        if close <= 0:
            return None

        # 1. Collect signals from independent strategies
        sig_a = self.strat_a.evaluate_bar(symbol, bar, is_regime_permitted=is_regime_permitted)
        sig_b = self.strat_b.evaluate_bar(symbol, bar, is_regime_permitted=is_regime_permitted)
        sig_e = self.strat_e.evaluate_bar(
            symbol, bar, regime_score=regime_score, is_regime_permitted=is_regime_permitted
        )

        triggered_strategies: List[str] = []
        collected_stops: List[float] = []
        all_reasons: List[str] = []

        if sig_a:
            triggered_strategies.append("STRATEGY_A_MOMENTUM")
            collected_stops.append(sig_a["stop_loss_price"])
            all_reasons.extend(sig_a.get("reasons", []))

        if sig_b:
            triggered_strategies.append("STRATEGY_B_BREAKOUT")
            collected_stops.append(sig_b["stop_loss_price"])
            all_reasons.extend(sig_b.get("reasons", []))

        if sig_e:
            triggered_strategies.append("STRATEGY_E_MULTI_FACTOR")
            collected_stops.append(sig_e["stop_loss_price"])
            all_reasons.extend(sig_e.get("reasons", []))

        # Must have at least one valid trigger
        if not triggered_strategies:
            return None

        consensus_count = len(triggered_strategies)

        # 2. Assign Ensemble Conviction Tier
        if consensus_count >= 3:
            conviction_tier = "HIGH CONVICTION (Triple Consensus)"
            confidence_score = 95.0
        elif consensus_count == 2:
            conviction_tier = "MODERATE CONVICTION (Dual Consensus)"
            confidence_score = 80.0
        else:
            conviction_tier = "SINGLE MODEL (Individual Setup)"
            confidence_score = 65.0

        # 3. Conservative Harmonized Stop Loss (highest stop price = lowest dollar risk)
        harmonized_stop = max(collected_stops) if collected_stops else (close * 0.95)
        target_price = close * 1.15  # +15% research target
        risk_dist = close - harmonized_stop
        reward_dist = target_price - close
        rr_ratio = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        momentum_score = float(sig_e.get("momentum_score", 70.0)) if sig_e else 70.0

        return {
            "strategy": "NSE_MOMENTUM_5_ENSEMBLE",
            "symbol": symbol,
            "signal": "BUY",
            "consensus_count": consensus_count,
            "conviction_tier": conviction_tier,
            "confidence_score": confidence_score,
            "triggered_models": triggered_strategies,
            "momentum_score": momentum_score,
            "entry_reference_price": round(close, 2),
            "stop_loss_price": round(harmonized_stop, 2),
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_dist, 2),
            "risk_reward_ratio": rr_ratio,
            "reasons": list(dict.fromkeys(all_reasons))[:4]  # Deduplicate top 4 reasons
        }

    def scan_ensemble(
        self,
        universe_features: Dict[str, pd.DataFrame],
        regime_score: float = 50.0,
        is_regime_permitted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scans all universe equities across the ensemble model suite.

        Returns:
            List of ensemble signals sorted by consensus count, then confidence score.
        """
        results: List[Dict[str, Any]] = []
        if not is_regime_permitted:
            return results

        for sym, df in universe_features.items():
            if df.empty:
                continue
            last_bar = df.iloc[-1]
            res = self.evaluate_candidate(
                symbol=sym,
                bar=last_bar,
                regime_score=regime_score,
                is_regime_permitted=is_regime_permitted
            )
            if res:
                results.append(res)

        # Sort: triple consensus first, then highest confidence score
        results.sort(key=lambda r: (r["consensus_count"], r["confidence_score"]), reverse=True)
        return results
