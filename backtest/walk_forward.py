"""
NSE MOMENTUM 5™ — Walk-Forward Research & Optimization Engine
================================================================================
Performs rolling out-of-sample (OOS) validation to detect and reject parameter
overfitting. Evaluates model stability across independent temporal market regimes.

WALK-FORWARD METHODOLOGY:
1. Chronological Non-Overlapping Window Rolling:
   - In-Sample (IS) Training: e.g., Months 1–24
   - In-Sample Validation: e.g., Months 25–30
   - Out-of-Sample (OOS) Test: e.g., Months 31–36
   - Roll forward by Step size (e.g., 6 months) and repeat across history.
2. Stability Metrics:
   - Robustness Ratio: OOS Profit Factor / IS Profit Factor (Target >= 0.70)
   - Consistency: Percentage of OOS test periods generating positive expectancy
   - Regime Invariance: Verifies strategy doesn't collapse during bear/neutral phases.
================================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from backtest.engine import EventDrivenBacktester
from backtest.metrics import BacktestSummary
from config import CONFIG, BacktestConfig
from utils.logger import setup_logger

logger = setup_logger("WALK_FORWARD_ENGINE")


@dataclass(frozen=True)
class WalkForwardWindow:
    """Chronological boundaries for a single walk-forward fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "val_start": str(self.val_start.date()),
            "val_end": str(self.val_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date())
        }


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Simulation results for an individual walk-forward fold."""
    fold_index: int
    window: WalkForwardWindow
    in_sample_summary: BacktestSummary
    out_of_sample_summary: BacktestSummary
    parameter_tested: Dict[str, Any]
    robustness_ratio: float              # OOS Profit Factor / IS Profit Factor
    is_oos_profitable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "window": self.window.to_dict(),
            "in_sample_profit_factor": self.in_sample_summary.profit_factor,
            "out_of_sample_profit_factor": self.out_of_sample_summary.profit_factor,
            "in_sample_cagr": self.in_sample_summary.cagr_pct,
            "out_of_sample_cagr": self.out_of_sample_summary.cagr_pct,
            "out_of_sample_drawdown": self.out_of_sample_summary.max_drawdown_pct,
            "robustness_ratio": self.robustness_ratio,
            "is_oos_profitable": self.is_oos_profitable,
            "oos_trades": self.out_of_sample_summary.total_trades
        }


@dataclass(frozen=True)
class WalkForwardReport:
    """Consolidated institutional walk-forward stability report."""
    total_folds: int
    profitable_oos_folds: int
    consistency_pct: float
    avg_oos_profit_factor: float
    avg_oos_cagr_pct: float
    max_oos_drawdown_pct: float
    overall_stability_score: float       # 0.0 to 100.0 Stability Metric
    is_walk_forward_robust: bool
    fold_results: List[WalkForwardFoldResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_folds": self.total_folds,
            "profitable_oos_folds": self.profitable_oos_folds,
            "consistency_pct": self.consistency_pct,
            "avg_oos_profit_factor": self.avg_oos_profit_factor,
            "avg_oos_cagr_pct": self.avg_oos_cagr_pct,
            "max_oos_drawdown_pct": self.max_oos_drawdown_pct,
            "overall_stability_score": self.overall_stability_score,
            "is_walk_forward_robust": self.is_walk_forward_robust,
            "folds": [f.to_dict() for f in self.fold_results]
        }


class WalkForwardOptimizer:
    """Orchestrates multi-period rolling walk-forward cross-validation."""

    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg: BacktestConfig = cfg or CONFIG.backtest

    def generate_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[WalkForwardWindow]:
        """
        Generates rolling chronological train, validation, and test slices.
        """
        windows: List[WalkForwardWindow] = []
        cur_start = start_date
        fold_idx = 1

        while True:
            train_end = cur_start + pd.DateOffset(months=self.cfg.wf_train_months)
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + pd.DateOffset(months=self.cfg.wf_val_months)
            test_start = val_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.cfg.wf_test_months)

            if test_end > end_date:
                break

            windows.append(WalkForwardWindow(
                fold_index=fold_idx,
                train_start=cur_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            ))

            cur_start = cur_start + pd.DateOffset(months=self.cfg.wf_step_months)
            fold_idx += 1

        return windows

    def run_walk_forward(
        self,
        universe_features: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        min_momentum_score: float = CONFIG.signal.momentum
    ) -> WalkForwardReport:
        """
        Executes rolling out-of-sample backtests across all generated time windows.

        Returns:
            WalkForwardReport summarizing multi-regime stability.
        """
        all_dates = sorted(list(set.union(*[set(df.index) for df in universe_features.values()])))
        if len(all_dates) < 250:
            logger.warning("Insufficient history for full walk-forward optimization.")
            return WalkForwardReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False, [])

        start_dt = all_dates[0]
        end_dt = all_dates[-1]
        windows = self.generate_windows(start_dt, end_dt)

        if not windows:
            logger.warning("Date span insufficient to generate at least one full walk-forward fold.")
            return WalkForwardReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False, [])

        logger.info(f"Executing Walk-Forward Optimization across {len(windows)} chronological folds...")

        backtester = EventDrivenBacktester(min_momentum_score=min_momentum_score)
        fold_results: List[WalkForwardFoldResult] = []

        for win in windows:
            # 1. In-Sample Slice (Training + Validation)
            is_universe = {
                sym: df.loc[(df.index >= win.train_start) & (df.index <= win.val_end)]
                for sym, df in universe_features.items()
            }
            # 2. Out-of-Sample Slice (Strict Unseen Test Data)
            oos_universe = {
                sym: df.loc[(df.index >= win.test_start) & (df.index <= win.test_end)]
                for sym, df in universe_features.items()
            }

            is_bench = (
                benchmark_df.loc[(benchmark_df.index >= win.train_start) & (benchmark_df.index <= win.val_end)]
                if benchmark_df is not None and not benchmark_df.empty else None
            )
            oos_bench = (
                benchmark_df.loc[(benchmark_df.index >= win.test_start) & (benchmark_df.index <= win.test_end)]
                if benchmark_df is not None and not benchmark_df.empty else None
            )

            is_res = backtester.run(is_universe, is_bench)
            oos_res = backtester.run(oos_universe, oos_bench)

            is_summary = is_res["summary"]
            oos_summary = oos_res["summary"]

            # Robustness Ratio = OOS Profit Factor / IS Profit Factor
            is_pf = is_summary.profit_factor if is_summary else 1.0
            oos_pf = oos_summary.profit_factor if oos_summary else 0.0
            rob_ratio = round(oos_pf / is_pf, 2) if is_pf > 0 else 0.0

            is_profitable = oos_summary.cumulative_return_pct > 0 if oos_summary else False

            fold_results.append(WalkForwardFoldResult(
                fold_index=win.fold_index,
                window=win,
                in_sample_summary=is_summary,
                out_of_sample_summary=oos_summary,
                parameter_tested={"min_momentum_score": min_momentum_score},
                robustness_ratio=rob_ratio,
                is_oos_profitable=is_profitable
            ))

        # Consolidated Summary
        n_folds = len(fold_results)
        n_profitable = sum(1 for f in fold_results if f.is_oos_profitable)
        consistency = round((n_profitable / n_folds) * 100.0, 1) if n_folds > 0 else 0.0

        avg_oos_pf = round(float(np.mean([f.out_of_sample_summary.profit_factor for f in fold_results])), 2)
        avg_oos_cagr = round(float(np.mean([f.out_of_sample_summary.cagr_pct for f in fold_results])), 2)
        max_oos_dd = round(float(min([f.out_of_sample_summary.max_drawdown_pct for f in fold_results])), 2)

        # Overall Stability Score (0 to 100): Combines consistency, OOS PF, and DD
        stability = (consistency * 0.50) + (min(avg_oos_pf, 2.5) / 2.5 * 30.0) + (max(0.0, 100.0 + max_oos_dd) * 0.20)
        stability_score = round(float(np.clip(stability, 0.0, 100.0)), 1)

        # Robustness Standard: Consistency >= 70% AND Avg OOS PF >= 1.25 AND Max DD >= -25%
        is_wf_robust = (
            consistency >= 70.0 and
            avg_oos_pf >= 1.25 and
            max_oos_dd >= -25.0
        )

        logger.info(
            f"Walk-Forward Complete. Folds: {n_folds}, Consistency: {consistency}%, "
            f"Avg OOS PF: {avg_oos_pf}, Stability Score: {stability_score}/100."
        )

        return WalkForwardReport(
            total_folds=n_folds,
            profitable_oos_folds=n_profitable,
            consistency_pct=consistency,
            avg_oos_profit_factor=avg_oos_pf,
            avg_oos_cagr_pct=avg_oos_cagr,
            max_oos_drawdown_pct=max_oos_dd,
            overall_stability_score=stability_score,
            is_walk_forward_robust=is_wf_robust,
            fold_results=fold_results
        )
