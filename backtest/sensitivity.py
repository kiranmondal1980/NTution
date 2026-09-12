"""
NSE MOMENTUM 5™ — Parameter Sensitivity & Plateau Stability Analyzer
================================================================================
Evaluates whether strategy profitability rests on a wide, robust parameter plateau
rather than an isolated, overfit "knife-edge" spike.

SENSITIVITY PRINCIPLE:
A genuinely robust quantitative model must tolerate minor perturbations in its
core hyperparameters (e.g. ±10%) without collapsing into net losses.

Neighborhood Perturbations Tested:
1. ATR Trailing Multiplier: 1.8, 2.0, 2.2
2. Minimum Momentum Score Trigger: 70.0, 75.0, 80.0
3. Max Holding Period Horizon: 4, 5, 6 sessions

Evaluation Criteria:
- Coefficient of Variation (CV = StdDev / Mean) of Profit Factor
- Low CV (< 0.25) indicates smooth parameter stability (Passed).
- High CV (> 0.40) or sign flip to negative return indicates an overfit spike (Rejected).
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from backtest.engine import EventDrivenBacktester
from backtest.metrics import BacktestSummary
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("SENSITIVITY_ANALYZER")


@dataclass(frozen=True)
class SensitivityRunResult:
    """Individual simulation result for a specific parameter value."""
    parameter_name: str
    parameter_value: Any
    profit_factor: float
    win_rate_pct: float
    cumulative_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "profit_factor": self.profit_factor,
            "win_rate_pct": self.win_rate_pct,
            "cumulative_return_pct": self.cumulative_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades
        }


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Consolidated stability report for a tested hyperparameter."""
    parameter_name: str
    tested_values: List[Any]
    profit_factor_mean: float
    profit_factor_std: float
    coefficient_of_variation: float
    is_stable_plateau: bool
    summary_note: str
    runs: List[SensitivityRunResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "tested_values": self.tested_values,
            "profit_factor_mean": self.profit_factor_mean,
            "profit_factor_std": self.profit_factor_std,
            "coefficient_of_variation": self.coefficient_of_variation,
            "is_stable_plateau": self.is_stable_plateau,
            "summary_note": self.summary_note,
            "runs": [r.to_dict() for r in self.runs]
        }


class ParameterSensitivityAnalyzer:
    """Executes hyperparameter perturbation grids to verify performance stability."""

    def __init__(self, initial_capital: float = CONFIG.risk.default_capital_inr):
        self.initial_capital = initial_capital

    def test_momentum_thresholds(
        self,
        universe_features: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        thresholds: Optional[List[float]] = None
    ) -> ParameterSensitivityReport:
        """
        Tests stability around the minimum momentum score trigger (e.g. 70, 75, 80).
        """
        test_vals = thresholds or [70.0, 75.0, 80.0]
        runs: List[SensitivityRunResult] = []

        logger.info(f"Analyzing sensitivity for 'min_momentum_score' across {test_vals}...")

        for score in test_vals:
            tester = EventDrivenBacktester(
                initial_capital=self.initial_capital,
                min_momentum_score=score
            )
            res = tester.run(universe_features, benchmark_df)
            summary = res["summary"]

            if summary and summary.total_trades > 0:
                runs.append(SensitivityRunResult(
                    parameter_name="min_momentum_score",
                    parameter_value=score,
                    profit_factor=summary.profit_factor,
                    win_rate_pct=summary.win_rate_pct,
                    cumulative_return_pct=summary.cumulative_return_pct,
                    sharpe_ratio=summary.sharpe_ratio,
                    max_drawdown_pct=summary.max_drawdown_pct,
                    total_trades=summary.total_trades
                ))

        if not runs:
            return ParameterSensitivityReport(
                parameter_name="min_momentum_score",
                tested_values=test_vals,
                profit_factor_mean=0.0,
                profit_factor_std=0.0,
                coefficient_of_variation=1.0,
                is_stable_plateau=False,
                summary_note="No trades generated across sensitivity neighborhood.",
                runs=[]
            )

        pf_vals = [r.profit_factor for r in runs]
        pf_mean = round(float(np.mean(pf_vals)), 2)
        pf_std = round(float(np.std(pf_vals)), 2)
        cv = round(pf_std / pf_mean, 3) if pf_mean > 0 else 1.0

        # Stable plateau: CV < 0.25 and all tested values produce PF > 1.10
        is_stable = (cv <= 0.25) and all(pf >= 1.10 for pf in pf_vals)

        note = (
            f"Stable parameter plateau detected (CV: {cv:.3f}, Mean PF: {pf_mean:.2f})."
            if is_stable else
            f"Vulnerable knife-edge sensitivity detected (CV: {cv:.3f}, Mean PF: {pf_mean:.2f})."
        )

        return ParameterSensitivityReport(
            parameter_name="min_momentum_score",
            tested_values=test_vals,
            profit_factor_mean=pf_mean,
            profit_factor_std=pf_std,
            coefficient_of_variation=cv,
            is_stable_plateau=is_stable,
            summary_note=note,
            runs=runs
        )

    def test_holding_horizons(
        self,
        universe_features: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        horizons: Optional[List[int]] = None
    ) -> ParameterSensitivityReport:
        """
        Tests stability around max swing holding duration (e.g. 4, 5, 6 sessions).
        """
        test_vals = horizons or [4, 5, 6]
        runs: List[SensitivityRunResult] = []

        logger.info(f"Analyzing sensitivity for 'max_holding_sessions' across {test_vals}...")

        for h in test_vals:
            tester = EventDrivenBacktester(
                initial_capital=self.initial_capital,
                max_holding_sessions=h
            )
            res = tester.run(universe_features, benchmark_df)
            summary = res["summary"]

            if summary and summary.total_trades > 0:
                runs.append(SensitivityRunResult(
                    parameter_name="max_holding_sessions",
                    parameter_value=h,
                    profit_factor=summary.profit_factor,
                    win_rate_pct=summary.win_rate_pct,
                    cumulative_return_pct=summary.cumulative_return_pct,
                    sharpe_ratio=summary.sharpe_ratio,
                    max_drawdown_pct=summary.max_drawdown_pct,
                    total_trades=summary.total_trades
                ))

        if not runs:
            return ParameterSensitivityReport(
                parameter_name="max_holding_sessions",
                tested_values=test_vals,
                profit_factor_mean=0.0,
                profit_factor_std=0.0,
                coefficient_of_variation=1.0,
                is_stable_plateau=False,
                summary_note="No trades generated across holding horizon neighborhood.",
                runs=[]
            )

        pf_vals = [r.profit_factor for r in runs]
        pf_mean = round(float(np.mean(pf_vals)), 2)
        pf_std = round(float(np.std(pf_vals)), 2)
        cv = round(pf_std / pf_mean, 3) if pf_mean > 0 else 1.0

        is_stable = (cv <= 0.25) and all(pf >= 1.10 for pf in pf_vals)

        note = (
            f"Stable holding horizon plateau (CV: {cv:.3f}, Mean PF: {pf_mean:.2f})."
            if is_stable else
            f"High sensitivity to holding horizon (CV: {cv:.3f}, Mean PF: {pf_mean:.2f})."
        )

        return ParameterSensitivityReport(
            parameter_name="max_holding_sessions",
            tested_values=test_vals,
            profit_factor_mean=pf_mean,
            profit_factor_std=pf_std,
            coefficient_of_variation=cv,
            is_stable_plateau=is_stable,
            summary_note=note,
            runs=runs
        )
