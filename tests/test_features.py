"""
NSE MOMENTUM 5™ — Unit Test Suite: Feature Engineering Engines
================================================================================
Verifies mathematical accuracy, boundary constraints, and indicator correctness
using synthetic deterministic OHLCV bars:
1. Multi-Period Returns & Momentum Acceleration
2. Moving Average Hierarchies & Trend Stack Verification
3. Volume Ratios, Skew, and Close Location Value (CLV)
4. ATR 14, ATR%, Bollinger Bands, and Bounded RSI 14
5. Shifted Resistance Pivot Breakouts
6. Benchmark Relative Strength Calculations
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from features.momentum import calculate_momentum_features
from features.trend import calculate_trend_features
from features.volume import calculate_volume_features
from features.volatility import calculate_volatility_features
from features.breakout import calculate_breakout_features
from features.relative_strength import calculate_relative_strength_features


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for all quantitative feature extraction modules."""

    def setUp(self):
        """Builds a deterministic 60-session synthetic OHLCV dataset."""
        dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
        # Uptrending price series with daily noise: 100.0 to 188.5
        base_prices = [100.0 + (i * 1.5) for i in range(60)]

        self.df = pd.DataFrame({
            "open": base_prices,
            "high": [p + 2.5 for p in base_prices],
            "low": [p - 1.5 for p in base_prices],
            "close": [p + 1.0 for p in base_prices],
            "volume": [100000.0 + (i * 1000.0) for i in range(60)]
        }, index=dates)

        # Benchmark index series (slower uptrend)
        bench_prices = [18000.0 + (i * 15.0) for i in range(60)]
        self.bench_df = pd.DataFrame({
            "open": bench_prices,
            "high": [p + 50.0 for p in bench_prices],
            "low": [p - 50.0 for p in bench_prices],
            "close": bench_prices,
            "volume": [5000000.0] * 60
        }, index=dates)

    def test_momentum_features(self):
        """Tests multi-period returns and acceleration metrics."""
        mom = calculate_momentum_features(self.df)

        self.assertIn("ret_1d", mom.columns)
        self.assertIn("ret_5d", mom.columns)
        self.assertIn("mom_accel_1v5", mom.columns)
        self.assertIn("consecutive_up_days", mom.columns)

        # In a steady uptrend, returns must be strictly positive
        valid_ret = mom["ret_5d"].dropna()
        self.assertTrue((valid_ret > 0).all())

        # Consecutive up days streak must increment
        self.assertGreater(float(mom["consecutive_up_days"].iloc[-1]), 5)

    def test_trend_features(self):
        """Tests moving average computations and bullish hierarchy stack."""
        trend = calculate_trend_features(self.df)

        self.assertIn("ema_5", trend.columns)
        self.assertIn("ema_10", trend.columns)
        self.assertIn("ema_20", trend.columns)
        self.assertIn("ema_50", trend.columns)
        self.assertIn("bullish_ema_stack", trend.columns)

        last_row = trend.iloc[-1]
        # In a steady linear uptrend: Price > EMA5 > EMA10 > EMA20 > EMA50
        self.assertGreater(last_row["ema_5"], last_row["ema_10"])
        self.assertGreater(last_row["ema_10"], last_row["ema_20"])
        self.assertEqual(int(last_row["bullish_ema_stack"]), 1)

    def test_volume_features(self):
        """Tests volume surge ratios and Close Location Value (CLV)."""
        vol = calculate_volume_features(self.df)

        self.assertIn("vol_ratio_5d", vol.columns)
        self.assertIn("clv", vol.columns)
        self.assertIn("candle_body_range_ratio", vol.columns)

        # CLV must be mathematically bounded between -1.0 and +1.0
        clv_series = vol["clv"].dropna()
        self.assertTrue(((clv_series >= -1.0) & (clv_series <= 1.0)).all())

        # Volume ratios must be strictly positive
        vol_ratios = vol["vol_ratio_5d"].dropna()
        self.assertTrue((vol_ratios > 0).all())

    def test_volatility_features(self):
        """Tests ATR 14, ATR%, Bollinger Bands, and bounded RSI 14."""
        volat = calculate_volatility_features(self.df)

        self.assertIn("atr_14", volat.columns)
        self.assertIn("atr_pct", volat.columns)
        self.assertIn("rsi_14", volat.columns)
        self.assertIn("bb_width", volat.columns)

        # ATR must be strictly positive
        atr_valid = volat["atr_14"].dropna()
        self.assertTrue((atr_valid > 0).all())

        # RSI must be strictly bounded between 0.0 and 100.0
        rsi_valid = volat["rsi_14"].dropna()
        self.assertTrue(((rsi_valid >= 0.0) & (rsi_valid <= 100.0)).all())

    def test_breakout_features(self):
        """Tests shifted historical resistance pivots."""
        bo = calculate_breakout_features(self.df)

        self.assertIn("prior_high_20d", bo.columns)
        self.assertIn("breakout_20d", bo.columns)

        # Verify bar 25's prior 20D high matches the maximum of bars 5 through 24
        expected_high = self.df["high"].iloc[5:25].max()
        actual_prior_high = bo["prior_high_20d"].iloc[25]
        self.assertAlmostEqual(expected_high, actual_prior_high, places=4)

    def test_relative_strength_features(self):
        """Tests benchmark alpha calculations."""
        rs = calculate_relative_strength_features(self.df, self.bench_df)

        self.assertIn("rs_5d", rs.columns)
        self.assertIn("rs_20d", rs.columns)
        self.assertIn("mansfield_rs", rs.columns)

        # The synthetic stock gained faster than benchmark, so alpha should be positive
        last_rs_20 = rs["rs_20d"].iloc[-1]
        self.assertGreater(last_rs_20, 0.0)


if __name__ == "__main__":
    unittest.main()
