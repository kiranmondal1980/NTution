"""
NSE MOMENTUM 5™ — Unit Test Suite: Exit Intelligence Engine (Engine B)
================================================================================
Verifies all position management and profit protection logic:
1. Emergency Exit on hard stop-loss violation
2. Dynamic ATR trailing stop breaches
3. Staged profit protection ratchets (Tier 1 Break-Even to Tier 4 Tight Trail)
4. Hold Score trend health calculation (0 to 100)
5. Distribution bar detection and terminal liquidation flags
================================================================================
"""

import unittest
import pandas as pd

from strategy.exit_engine import ExitIntelligenceEngine, ExitAction, ExitAssessment


class TestExitEngine(unittest.TestCase):
    """Test suite for Engine B Exit Intelligence and position governance."""

    def setUp(self):
        """Initializes the exit engine and baseline healthy technical bar."""
        self.engine = ExitIntelligenceEngine()

        self.healthy_features = pd.Series({
            "close": 1150.0,
            "ema_10": 1120.0,
            "ema_20": 1090.0,
            "atr_14": 25.0,
            "rsi_14": 68.0,
            "vol_ratio_5d": 1.20,
            "clv": 0.40
        })

    def test_emergency_exit_hard_stop_violation(self):
        """Emergency Exit must trigger when price breaches user's hard stop."""
        assessment = self.engine.evaluate_position(
            symbol="TEST.NS",
            entry_price=1000.0,
            current_price=940.0,
            highest_price_since_entry=1020.0,
            latest_features=self.healthy_features,
            user_hard_stop=950.0
        )

        self.assertEqual(assessment.action, ExitAction.EMERGENCY_EXIT)
        self.assertTrue(assessment.is_terminal)
        self.assertEqual(assessment.hold_score, 0.0)

    def test_staged_profit_protection_tier1_break_even(self):
        """At +8% gain (Tier 1), trailing stop must lock in at or above break-even."""
        assessment = self.engine.evaluate_position(
            symbol="TEST.NS",
            entry_price=1000.0,
            current_price=1080.0,
            highest_price_since_entry=1080.0,
            latest_features=self.healthy_features
        )

        # Trailing stop must be at least entry price (Break-even protected)
        self.assertGreaterEqual(assessment.active_trailing_stop, 1000.0)
        self.assertIn("Tier 1", assessment.profit_tier)
        self.assertIn(assessment.action, [ExitAction.HOLD, ExitAction.TRAIL])

    def test_staged_profit_protection_tier4_tight_trailing(self):
        """At +25% gain (Tier 4), 1.0x ATR tight trailing protection must be active."""
        assessment = self.engine.evaluate_position(
            symbol="TEST.NS",
            entry_price=1000.0,
            current_price=1250.0,
            highest_price_since_entry=1250.0,
            latest_features=self.healthy_features
        )

        # Peak 1250 - 1.0 * ATR(25) = 1225.0
        expected_trail = 1250.0 - (1.0 * 25.0)
        self.assertAlmostEqual(assessment.active_trailing_stop, expected_trail, delta=1.0)
        self.assertIn("Tier 4", assessment.profit_tier)

    def test_trailing_stop_breach(self):
        """Price dropping below ratcheted trailing stop must trigger immediate EXIT."""
        assessment = self.engine.evaluate_position(
            symbol="TEST.NS",
            entry_price=1000.0,
            current_price=1180.0,           # Pulled back from peak 1250
            highest_price_since_entry=1250.0, # Stop locked at ~1225
            latest_features=self.healthy_features
        )

        self.assertEqual(assessment.action, ExitAction.EXIT)
        self.assertTrue(assessment.is_terminal)

    def test_hold_score_trend_deterioration(self):
        """Price breaking below EMA20 with distribution volume must lower Hold Score."""
        deteriorating_features = pd.Series({
            "close": 1050.0,
            "ema_10": 1080.0,
            "ema_20": 1100.0,             # Close < EMA10 < EMA20 (Broken trend)
            "atr_14": 30.0,
            "rsi_14": 42.0,               # Soft RSI
            "vol_ratio_5d": 2.50,         # Heavy distribution volume
            "clv": -0.60                  # Closed near daily lows
        })

        assessment = self.engine.evaluate_position(
            symbol="TEST.NS",
            entry_price=1000.0,
            current_price=1050.0,
            highest_price_since_entry=1120.0,
            latest_features=deteriorating_features
        )

        # Broken trend and distribution should lower Hold Score into partial book or exit zone
        self.assertLess(assessment.hold_score, 50.0)
        self.assertIn(assessment.action, [ExitAction.PARTIAL_BOOK, ExitAction.EXIT])


if __name__ == "__main__":
    unittest.main()
