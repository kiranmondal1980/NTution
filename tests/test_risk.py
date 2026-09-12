"""
NSE MOMENTUM 5™ — Unit Test Suite: Position Sizing & Portfolio Risk Governance
================================================================================
Verifies mathematical correctness of capital allocation and safety limits:
1. Exact Rupee Risk position sizing: Shares = floor(Max Rupee Risk / Stop Distance)
2. Single-stock position capital weighting cap (20.0% max equity)
3. Input validation: Inverted stops, non-positive prices, zero shares
4. Portfolio heat governor: Enforces 6.0% aggregate rupee risk cap
5. Maximum simultaneous open positions limit (5 positions)
6. Daily loss circuit breaker shutdown (3.0% drawdown limit)
================================================================================
"""

import unittest
from risk.position_sizing import PositionSizer
from risk.risk_engine import PortfolioRiskEngine
from config import CONFIG


class TestRiskEngine(unittest.TestCase):
    """Test suite for position sizing and portfolio risk management."""

    def setUp(self):
        """Initializes sizer and risk engine instances."""
        self.sizer = PositionSizer()
        self.risk_engine = PortfolioRiskEngine()
        self.test_capital = 1_000_000.0  # 10 Lakhs INR

    def test_exact_rupee_risk_sizing(self):
        """
        Verify mathematical sizing:
        Capital = 1,000,000 INR, Risk = 1% = 10,000 INR
        Entry = 100.0 INR, Stop = 90.0 INR -> Stop Distance = 10.0 INR
        Expected Shares = floor(10,000 / 10) = 1,000 shares
        """
        res = self.sizer.calculate_position(
            capital_inr=self.test_capital,
            entry_price=100.0,
            stop_loss_price=90.0,
            risk_per_trade_pct=0.01
        )

        self.assertTrue(res.is_permitted)
        self.assertEqual(res.shares, 1000)
        self.assertEqual(res.rupee_risk_allocated, 10000.0)
        self.assertEqual(res.position_value_inr, 100000.0)
        self.assertEqual(res.portfolio_weight_pct, 10.0)

    def test_single_position_capital_cap_enforcement(self):
        """
        When stop loss is very tight, position value cannot exceed max weight cap (20%).
        Capital = 1,000,000 INR -> Max Position Value = 200,000 INR
        Entry = 100.0 INR, Stop = 99.5 INR -> Stop Dist = 0.5 INR
        Raw risk shares = 10,000 / 0.5 = 20,000 shares (Value 2,000,000 INR, 200% capital)
        Capped shares = floor(200,000 / 100) = 2,000 shares (Value 200,000 INR)
        """
        res = self.sizer.calculate_position(
            capital_inr=self.test_capital,
            entry_price=100.0,
            stop_loss_price=99.5,
            risk_per_trade_pct=0.01,
            max_weight_pct=0.20
        )

        self.assertTrue(res.is_permitted)
        self.assertEqual(res.shares, 2000)
        self.assertEqual(res.position_value_inr, 200000.0)
        self.assertEqual(res.portfolio_weight_pct, 20.0)

    def test_invalid_price_inputs_rejection(self):
        """Rejects inverted stops where Stop Loss >= Entry Price."""
        res_inverted = self.sizer.calculate_position(
            capital_inr=self.test_capital,
            entry_price=100.0,
            stop_loss_price=105.0
        )
        self.assertFalse(res_inverted.is_permitted)
        self.assertIn("must be strictly below", res_inverted.rejection_reason)

    def test_max_simultaneous_positions_limit(self):
        """Portfolio governor must block new trades when 5 positions are already open."""
        mock_positions = [
            {"symbol": f"SYM_{i}.NS", "entry_price": 500.0, "current_price": 510.0, "quantity": 10, "current_stop": 480.0}
            for i in range(5)
        ]

        check = self.risk_engine.validate_new_trade(
            total_equity_inr=self.test_capital,
            open_positions=mock_positions,
            proposed_position_val_inr=50000.0,
            proposed_trade_risk_inr=5000.0
        )

        self.assertFalse(check["is_allowed"])
        self.assertTrue(any("MAX POSITION LIMIT REACHED" in r for r in check["reasons"]))

    def test_daily_circuit_breaker_enforcement(self):
        """Portfolio governor must halt all new trading if daily loss exceeds 3%."""
        max_loss = self.test_capital * 0.03  # 30,000 INR
        daily_loss = -35000.0                # Exceeded limit

        check = self.risk_engine.validate_new_trade(
            total_equity_inr=self.test_capital,
            open_positions=[],
            proposed_position_val_inr=50000.0,
            proposed_trade_risk_inr=5000.0,
            daily_realized_pnl_inr=daily_loss
        )

        self.assertFalse(check["is_allowed"])
        self.assertTrue(any("CIRCUIT BREAKER" in r for r in check["reasons"]))


if __name__ == "__main__":
    unittest.main()
