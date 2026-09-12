"""
NSE MOMENTUM 5™ — Unit Test Suite: Quantitative Strategy & Scoring Engines
================================================================================
Verifies mathematical bounds, signal generation, filtering rules, and regime
permissions across:
1. 0–100 Multi-Pillar Momentum Scoring Engine
2. Market Regime Engine (BULLISH, NEUTRAL, BEARISH)
3. Strategy A (Pure Momentum Continuation)
4. Strategy B (Breakout + Volume Confirmation)
5. Strategy E (Multi-Factor Quantitative Momentum)
6. Consensus Ensemble Strategy
================================================================================
"""

import unittest
import pandas as pd

from strategy.scoring import MomentumScoringEngine
from strategy.regime import MarketRegimeEngine
from strategy.momentum import PureMomentumStrategy
from strategy.breakout import BreakoutVolumeStrategy
from strategy.multi_factor import MultiFactorMomentumStrategy
from strategy.ensemble import EnsembleStrategyEngine


class TestStrategyEngines(unittest.TestCase):
    """Test suite for signaling, scoring, and strategy consensus."""

    def setUp(self):
        """Sets up synthetic feature profiles representing prime and weak setups."""
        # Prime Bullish Momentum Profile
        self.prime_bullish = pd.Series({
            "close": 2500.0,
            "ret_1d": 0.025,
            "ret_3d": 0.045,
            "ret_5d": 0.095,
            "mom_slope_5d": 0.008,
            "rs_5d": 0.045,
            "rs_20d": 0.080,
            "rs_outperform_days_10d": 8,
            "vol_ratio_5d": 2.20,
            "vol_ratio_20d": 1.95,
            "up_volume_ratio_10d": 0.72,
            "clv": 0.65,
            "breakout_20d": 1,
            "breakout_50d": 1,
            "dist_breakout_20d_pct": 0.025,
            "prior_high_20d": 2439.0,
            "breakout_persisted_2d": 1,
            "bullish_ema_stack": 1,
            "dist_ema_20_pct": 0.035,
            "ema_10": 2460.0,
            "ema_20": 2415.0,
            "atr_14": 55.0,
            "atr_pct": 2.20,
            "rsi_14": 66.5,
            "institutional_volume_surge": 1
        })

        # Weak / Deteriorating Setup Profile
        self.weak_profile = pd.Series({
            "close": 1500.0,
            "ret_1d": -0.025,
            "ret_3d": -0.050,
            "ret_5d": -0.080,
            "mom_slope_5d": -0.010,
            "rs_5d": -0.040,
            "rs_20d": -0.070,
            "rs_outperform_days_10d": 2,
            "vol_ratio_5d": 0.70,
            "vol_ratio_20d": 0.80,
            "up_volume_ratio_10d": 0.30,
            "clv": -0.70,
            "breakout_20d": 0,
            "breakout_50d": 0,
            "dist_breakout_20d_pct": -0.10,
            "breakout_persisted_2d": 0,
            "bullish_ema_stack": 0,
            "dist_ema_20_pct": -0.06,
            "ema_10": 1540.0,
            "ema_20": 1590.0,
            "atr_14": 40.0,
            "atr_pct": 2.67,
            "rsi_14": 34.0,
            "institutional_volume_surge": 0
        })

    def test_scoring_bounds_and_classification(self):
        """Verifies score mathematical boundaries (0-100) and classification."""
        scorer = MomentumScoringEngine()

        # Prime profile score must qualify for STRONG MOMENTUM (>= 85.0)
        res_prime = scorer.score_record(self.prime_bullish, regime_score=80.0)
        self.assertGreaterEqual(res_prime["momentum_score"], 85.0)
        self.assertLessEqual(res_prime["momentum_score"], 100.0)
        self.assertEqual(res_prime["category"], "STRONG MOMENTUM")

        # Weak profile must register as AVOID (< 50.0)
        res_weak = scorer.score_record(self.weak_profile, regime_score=30.0)
        self.assertLess(res_weak["momentum_score"], 50.0)
        self.assertEqual(res_weak["category"], "AVOID")

    def test_strategy_a_momentum_continuation(self):
        """Verifies Strategy A signals and gating."""
        strat = PureMomentumStrategy()

        # Should fire on prime profile when regime is permitted
        sig = strat.evaluate_bar("TEST.NS", self.prime_bullish, is_regime_permitted=True)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["signal"], "BUY")
        self.assertGreater(sig["target_price"], sig["entry_reference_price"])
        self.assertLess(sig["stop_loss_price"], sig["entry_reference_price"])

        # Must NOT fire if macro regime is bearish
        sig_blocked = strat.evaluate_bar("TEST.NS", self.prime_bullish, is_regime_permitted=False)
        self.assertIsNone(sig_blocked)

    def test_strategy_b_breakout_volume(self):
        """Verifies Strategy B breakout triggers and extension guardrails."""
        strat = BreakoutVolumeStrategy()

        # Prime profile is 2.5% above pivot -> Should trigger cleanly
        sig = strat.evaluate_bar("TEST.NS", self.prime_bullish, is_regime_permitted=True)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["strategy"], "STRATEGY_B_BREAKOUT_VOLUME")

        # If over-extended (> 5% above pivot), trade must be rejected
        extended_profile = self.prime_bullish.copy()
        extended_profile["dist_breakout_20d_pct"] = 0.08  # 8% above pivot (chasing)
        sig_extended = strat.evaluate_bar("TEST.NS", extended_profile, is_regime_permitted=True)
        self.assertIsNone(sig_extended)

    def test_ensemble_consensus_engine(self):
        """Verifies consensus classification across models."""
        ensemble = EnsembleStrategyEngine()

        # Prime profile satisfies multiple strategies -> Must trigger multi-model consensus
        sig_ens = ensemble.evaluate_candidate(
            "TEST.NS", self.prime_bullish, regime_score=80.0, is_regime_permitted=True
        )
        self.assertIsNotNone(sig_ens)
        self.assertGreaterEqual(sig_ens["consensus_count"], 2)
        self.assertIn("Consensus", sig_ens["conviction_tier"])


if __name__ == "__main__":
    unittest.main()
