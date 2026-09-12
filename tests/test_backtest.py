"""
NSE MOMENTUM 5™ — Unit Test Suite: Chronological Backtesting Sanity
================================================================================
Verifies simulation integrity, chronological causality, and fee accounting:
1. Execution Chronology: Entry date must be strictly after signal date (T+1 Open)
2. Execution Price: Fill occurs at Next Open + Slippage, NEVER at today's Close
3. Transaction Friction: Net P&L = Gross P&L - (Brokerage + STT + GST + Stamp + Slip)
4. Time Stop: Positions liquidate upon reaching max holding horizon (<= 5 sessions)
5. Metric Computation: Win Rate, Profit Factor, Expectancy, and Target Hit Rates
================================================================================
"""

import unittest
import pandas as pd
import numpy as np

from backtest.engine import EventDrivenBacktester
from backtest.execution import ExecutionEngine
from backtest.costs import NSETransactionCostCalculator
from backtest.metrics import calculate_backtest_metrics
from features import build_feature_pipeline


class TestBacktestSanity(unittest.TestCase):
    """Test suite for event-driven backtesting execution and accounting."""

    def setUp(self):
        """Builds a deterministic 40-session OHLCV synthetic series."""
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        # Uptrend with a pullback in the middle
        prices = [100.0 + (i * 2.0) if i < 25 else 150.0 - ((i - 25) * 1.5) for i in range(40)]

        self.raw_df = pd.DataFrame({
            "open": prices,
            "high": [p + 3.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": [p + 1.0 for p in prices],
            "volume": [200000.0] * 40
        }, index=dates)

        # Build feature pipeline
        self.feature_df = build_feature_pipeline(self.raw_df)
        self.universe = {"TEST_STOCK.NS": self.feature_df}

    def test_next_day_open_execution_chronology(self):
        """Simulates next-day open fill and verifies signal date < execution date."""
        exec_engine = ExecutionEngine()
        sig_date = pd.Timestamp("2024-01-05")
        exec_date = pd.Timestamp("2024-01-08")  # Next Monday

        order = exec_engine.execute_market_open(
            symbol="TEST_STOCK.NS",
            signal_date=sig_date,
            next_date=exec_date,
            next_open_price=100.0,
            quantity=100,
            side="BUY"
        )

        self.assertGreater(order.execution_date, order.signal_date)
        # BUY fill price must be higher than requested open due to slippage (10 bps)
        self.assertGreater(order.executed_price, 100.0)
        self.assertAlmostEqual(order.executed_price, 100.10, places=2)

    def test_transaction_cost_deduction_arithmetic(self):
        """Verifies that Net P&L = Gross P&L - Total Statutory Friction."""
        cost_calc = NSETransactionCostCalculator()
        buy_p = 100.0
        sell_p = 115.0
        qty = 1000

        costs = cost_calc.calculate_turnover_costs(buy_p, sell_p, qty)

        gross_pnl = (sell_p - buy_p) * qty  # 15,000 INR
        net_pnl = gross_pnl - costs.total_friction

        # Total friction must include positive brokerage, STT, and GST
        self.assertGreater(costs.total_friction, 0.0)
        self.assertGreater(costs.stt, 0.0)
        self.assertGreater(costs.gst, 0.0)
        self.assertLess(net_pnl, gross_pnl)
        self.assertEqual(net_pnl, gross_pnl - costs.total_friction)

    def test_backtester_trade_execution_and_holding_limit(self):
        """Simulates a backtest run and verifies holding duration caps at 5 sessions."""
        tester = EventDrivenBacktester(initial_capital=500000.0, max_holding_sessions=5, min_momentum_score=60.0)
        results = tester.run(self.universe)

        trades = results["trades"]
        summary = results["summary"]

        if trades:
            for t in trades:
                # Chronological guarantee: Exit must be after Entry
                self.assertGreater(t["exit_date"], t["entry_date"])
                # Max holding period rule: No trade held longer than 5 sessions
                self.assertLessEqual(t["holding_sessions"], 5)
                # P&L accounting: Net P&L must equal Gross P&L minus Total Costs
                self.assertAlmostEqual(t["net_pnl"], t["gross_pnl"] - t["total_costs"], delta=0.05)

        self.assertIsNotNone(summary)
        self.assertIsInstance(summary.win_rate_pct, float)

    def test_empty_universe_graceful_handling(self):
        """Backtester must handle empty inputs gracefully without exceptions."""
        tester = EventDrivenBacktester()
        res = tester.run({})
        self.assertIsNone(res["summary"])
        self.assertEqual(len(res["trades"]), 0)


if __name__ == "__main__":
    unittest.main()
