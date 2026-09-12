"""
NSE MOMENTUM 5™ — Quantitative Backtesting & Research Simulation Engine
================================================================================
Exposes all components for event simulation, statutory friction calculation,
performance metrics, walk-forward cross-validation, and sensitivity analysis.

Modules Exported:
- costs: NSETransactionCostCalculator, CostBreakdown (Full statutory Indian delivery fees)
- execution: ExecutionEngine, SimulatedOrder (Strict next-day open execution + slippage)
- metrics: calculate_backtest_metrics, BacktestSummary (Institutional performance & hit rates)
- engine: EventDrivenBacktester (Chronological portfolio event simulator)
- walk_forward: WalkForwardOptimizer, WalkForwardReport (Out-of-sample stability validation)
- sensitivity: ParameterSensitivityAnalyzer (Hyperparameter neighborhood plateau testing)
================================================================================
"""

from .costs import NSETransactionCostCalculator, CostBreakdown
from .execution import ExecutionEngine, SimulatedOrder
from .metrics import calculate_backtest_metrics, BacktestSummary
from .engine import EventDrivenBacktester
from .walk_forward import (
    WalkForwardOptimizer,
    WalkForwardWindow,
    WalkForwardFoldResult,
    WalkForwardReport
)
from .sensitivity import (
    ParameterSensitivityAnalyzer,
    SensitivityRunResult,
    ParameterSensitivityReport
)

__all__ = [
    "NSETransactionCostCalculator",
    "CostBreakdown",
    "ExecutionEngine",
    "SimulatedOrder",
    "calculate_backtest_metrics",
    "BacktestSummary",
    "EventDrivenBacktester",
    "WalkForwardOptimizer",
    "WalkForwardWindow",
    "WalkForwardFoldResult",
    "WalkForwardReport",
    "ParameterSensitivityAnalyzer",
    "SensitivityRunResult",
    "ParameterSensitivityReport"
]
