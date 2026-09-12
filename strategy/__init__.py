"""
NSE MOMENTUM 5™ — Quantitative Strategy & Intelligence Engines
================================================================================
Exposes all quantitative signaling, market regime classification, multi-pillar
scoring, consensus ensemble, and position management engines.

Modules Exported:
- regime: Macro NIFTY 50 market regime determination (BULLISH, NEUTRAL, BEARISH)
- scoring: 0–100 Multi-Pillar Momentum Scoring & categorical classification
- momentum: Strategy A (Pure Momentum Continuation)
- breakout: Strategy B (Breakout + Volume Confirmation)
- multi_factor: Strategy E (Multi-Factor Quantitative Momentum)
- ensemble: Consensus multi-model ensemble (NSE MOMENTUM 5 ENSEMBLE)
- exit_engine: Engine B (Exit Intelligence, dynamic ATR trailing, Hold Scores)
================================================================================
"""

from .regime import MarketRegimeEngine, MarketRegimeState
from .scoring import MomentumScoringEngine
from .momentum import PureMomentumStrategy
from .breakout import BreakoutVolumeStrategy
from .multi_factor import MultiFactorMomentumStrategy
from .ensemble import EnsembleStrategyEngine
from .exit_engine import ExitIntelligenceEngine, ExitAction, ExitAssessment

__all__ = [
    "MarketRegimeEngine",
    "MarketRegimeState",
    "MomentumScoringEngine",
    "PureMomentumStrategy",
    "BreakoutVolumeStrategy",
    "MultiFactorMomentumStrategy",
    "EnsembleStrategyEngine",
    "ExitIntelligenceEngine",
    "ExitAction",
    "ExitAssessment"
]
