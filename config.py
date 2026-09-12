"""
NSE MOMENTUM 5™ — Master System Configuration
================================================================================
A 3–5 Day NSE India Quantitative Momentum, Probability, Risk & Exit Intelligence System.

This module centralizes ALL configuration parameters for:
- Universe definitions and liquidity/quality filters
- Feature engineering parameters (momentum, trend, volatility, volume, breakout)
- Transparent 0–100 momentum scoring pillar weights
- Market regime classification and long-permission rules
- Engine B Exit Intelligence and staged dynamic ATR profit-protection rules
- Position sizing, portfolio heat, and capital risk limits
- Exact NSE India cash delivery transaction fee and slippage modeling
- Chronological backtesting, walk-forward splits, and robustness criteria
- Machine learning research targets and probability estimators

NON-NEGOTIABLE DESIGN PRINCIPLES:
- No hard-coded magic numbers in downstream modules.
- Explicit distinction between research hypotheses and validated results.
- All transaction cost parameters are realistic and editable.
================================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# 1. FILE SYSTEM & PERSISTENCE PATHS
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data_store"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH: Path = DATA_DIR / "nse_momentum_5.db"
LOG_DIR: Path = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR: Path = BASE_DIR / "models_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR: Path = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DATABASE CONFIGURATION
# ==============================================================================
@dataclass(frozen=True)
class DatabaseConfig:
    """Relational database connection parameters."""
    db_uri: str = f"sqlite:///{DB_PATH}"
    echo_sql: bool = False
    pool_pre_ping: bool = True
    connect_args: Dict[str, bool] = field(
        default_factory=lambda: {"check_same_thread": False}
    )


# ==============================================================================
# 3. UNIVERSE & LIQUIDITY FILTERS
# ==============================================================================
@dataclass(frozen=True)
class UniverseConfig:
    """
    NSE India Equities Screening Universe and Quality Guardrails.
    Excludes illiquid micro-caps, penny stocks, and untradeable securities.
    """
    # Primary Benchmarks (Yahoo Finance Tickers for Indian Indices)
    benchmark_symbol: str = "^NSEI"        # NIFTY 50 (Systemic Regime Reference)
    nifty500_symbol: str = "^CRSLDX"       # NIFTY 500 Broad Market Reference

    # Default liquid watch universe (Top liquid liquid swing trading stocks on NSE)
    default_symbols: List[str] = field(
        default_factory=lambda: [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "TITAN.NS", "MARUTI.NS",
            "BAJFINANCE.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS",
            "POWERGRID.NS", "M&M.NS", "HCLTECH.NS", "TATAMOTORS.NS", "JSWSTEEL.NS",
            "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "ONGC.NS",
            "GRASIM.NS", "BPCL.NS", "TECHM.NS", "HINDALCO.NS", "BRITANNIA.NS",
            "INDUSINDBK.NS", "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
            "TATACONSUM.NS", "HEROMOTOCO.NS", "DRREDDY.NS", "WIPRO.NS", "BAJAJ-AUTO.NS",
            "SBILIFE.NS", "HDFCLIFE.NS", "LTIM.NS", "SHRIRAMFIN.NS", "BEL.NS",
            "TRENT.NS", "HAL.NS", "VEDL.NS", "CHOLAFIN.NS", "ZYDUSLIFE.NS",
            "PFC.NS", "RECLTD.NS", "PERSISTENT.NS", "POLYCAB.NS", "DIXON.NS",
            "BOSCHLTD.NS", "HAVELLS.NS", "ABB.NS", "SIEMENS.NS", "CUMMINSIND.NS",
            "VOLTAS.NS", "COLPAL.NS", "GODREJCP.NS", "PIDILITIND.NS", "DABUR.NS"
        ]
    )

    # Hard Liquidity & Safety Criteria
    min_price_inr: float = 50.0             # Exclude penny stocks
    max_price_inr: float = 65000.0          # Max tradeable price per share
    min_avg_daily_volume_20d: float = 50000.0  # Minimum 20-day average daily volume (shares)
    min_avg_daily_turnover_inr_20d: float = 25_000_000.0  # Min ADTV of 2.5 Crore INR
    max_bid_ask_spread_pct: float = 0.005   # Maximum acceptable spread proxy (0.5%)
    min_history_days_required: int = 252    # Minimum 1 trading year of history required


# ==============================================================================
# 4. FEATURE ENGINEERING PARAMETERS
# ==============================================================================
@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction rolling windows and mathematical lookbacks."""
    # Momentum Returns (session counts)
    momentum_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    mom_slope_window: int = 5

    # Moving Averages
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Volatility & Bands
    atr_period: int = 14
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Breakout Windows (Strictly shifted: excludes current bar from reference high)
    breakout_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 252])

    # Relative Strength comparison windows vs NIFTY 50
    relative_strength_windows: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])


# ==============================================================================
# 5. TRANSPARENT SCORING ENGINE (0–100)
# ==============================================================================
@dataclass(frozen=True)
class ScoringWeightsConfig:
    """
    Initial Research Weights summing to 100.0 points.
    Subject to walk-forward optimization and stability analysis.
    """
    price_momentum: float = 20.0          # 1D, 3D, 5D velocity and slope
    relative_strength: float = 15.0       # Outperformance vs NIFTY 50
    volume_quality: float = 15.0          # Volume surge, up/down volume skew, CLV
    breakout_strength: float = 15.0       # Proximity and persistence above 20D/50D high
    trend_alignment: float = 10.0         # Price > EMA5 > EMA10 > EMA20 > EMA50 stack
    volatility_suitability: float = 10.0  # ATR% in optimal swing channel (2%-5%)
    market_regime: float = 10.0           # NIFTY 50 macroeconomic health score
    momentum_health: float = 5.0          # RSI regime (bullish range without exhaustion)

    def total_weight(self) -> float:
        return (
            self.price_momentum + self.relative_strength + self.volume_quality +
            self.breakout_strength + self.trend_alignment + self.volatility_suitability +
            self.market_regime + self.momentum_health
        )


@dataclass(frozen=True)
class SignalThresholdsConfig:
    """Classification cutoffs for the 0–100 Momentum Score."""
    strong_momentum: float = 85.0         # Top conviction swing setups
    momentum: float = 75.0                # Actionable momentum setups
    watchlist: float = 65.0               # Developing setups near breakout
    weak: float = 50.0                    # Indifferent or mixed setups
    avoid: float = 0.0                    # Negative momentum or breakdown (< 50)


# ==============================================================================
# 6. MARKET REGIME ENGINE CONFIGURATION
# ==============================================================================
@dataclass(frozen=True)
class MarketRegimeConfig:
    """Rules for categorizing broad market permission (NIFTY 50)."""
    fast_ema: int = 20
    slow_ema: int = 50
    trend_lookback_days: int = 5

    # Thresholds for Regime assignment
    bullish_threshold_score: float = 65.0
    bearish_threshold_score: float = 35.0

    # Policy adjustments
    allow_longs_in_bearish: bool = False
    require_higher_conviction_in_neutral: bool = True
    neutral_score_penalty: float = 10.0   # Added bar requirement for neutral regime


# ==============================================================================
# 7. RESEARCH TARGET HORIZONS (PROBABILITY ENGINE)
# ==============================================================================
@dataclass(frozen=True)
class ExpectedMoveConfig:
    """
    Research target specifications.
    THESE ARE RESEARCH BENCHMARKS, NOT GUARANTEED PREDICTIONS.
    """
    target_horizons: List[Tuple[float, int]] = field(
        default_factory=lambda: [
            (0.05, 3),   # +5% within 3 trading sessions
            (0.10, 5),   # +10% within 5 trading sessions
            (0.15, 5),   # +15% within 5 trading sessions
            (0.20, 5)    # +20% within 5 trading sessions
        ]
    )
    # k-NN / Conditional historical lookup sample size
    similarity_min_samples: int = 30
    similarity_score_tolerance: float = 5.0


# ==============================================================================
# 8. ENGINE B — EXIT INTELLIGENCE & PROFIT PROTECTION
# ==============================================================================
@dataclass(frozen=True)
class ExitConfig:
    """
    Evidence-based position management, staged dynamic ATR stops,
    and Hold Score (0–100) thresholds.
    """
    # Base ATR trailing multiplier (when no profit tier is active)
    default_atr_multiplier: float = 2.0

    # Staged Profit Protection (Hypothesis to be backtested)
    tier1_gain_pct: float = 0.05          # +5% gain reached
    tier1_atr_multiplier: float = 2.5     # Move stop to break-even or loose trail

    tier2_gain_pct: float = 0.10          # +10% gain reached
    tier2_atr_multiplier: float = 2.0     # Lock in substantial portion

    tier3_gain_pct: float = 0.15          # +15% gain reached
    tier3_atr_multiplier: float = 1.5     # Tighten trailing protection

    tier4_gain_pct: float = 0.20          # +20% gain reached
    tier4_atr_multiplier: float = 1.0     # Ultra-aggressive trailing lock

    # Hold Score thresholds (Trend Health 0–100)
    hold_score_strong_hold: float = 80.0  # 80-100: STRONG HOLD
    hold_score_trail: float = 65.0        # 65-79: HOLD / TRAIL
    hold_score_watch: float = 50.0        # 50-64: WATCH / TRAIL
    hold_score_partial_book: float = 35.0 # 35-49: PARTIAL BOOK / EXIT WATCH
    # Below 35 = Mandatory Full Exit

    # Emergency Exit Hard Limits
    max_loss_from_entry_pct: float = 0.08 # Hard catastrophic stop (-8%)


# ==============================================================================
# 9. RISK MANAGEMENT & PORTFOLIO GOVERNANCE
# ==============================================================================
@dataclass(frozen=True)
class RiskConfig:
    """Portfolio capital preservation rules."""
    default_capital_inr: float = 1_000_000.0   # Default test capital: 10 Lakhs INR
    risk_per_trade_pct: float = 0.01          # Risk exactly 1% of total equity per setup
    max_portfolio_risk_pct: float = 0.06      # Max simultaneous portfolio heat (6%)
    max_position_weight_pct: float = 0.20     # Max 20% of capital in any single security
    max_simultaneous_positions: int = 5       # Max 5 concurrent holdings
    max_daily_portfolio_loss_pct: float = 0.03 # 3% daily circuit breaker shutdown


# ==============================================================================
# 10. REALISTIC NSE TRANSACTION COSTS & FRICTION
# ==============================================================================
@dataclass(frozen=True)
class CostConfig:
    """
    Standard NSE India Cash Equity Delivery Cost Structure.
    All rates are expressed as decimals of trade turnover.
    """
    brokerage_pct: float = 0.0003             # 0.03% or Rs 20 whichever is lower proxy
    stt_delivery_buy_pct: float = 0.0010      # 0.10% Securities Transaction Tax on Buy
    stt_delivery_sell_pct: float = 0.0010     # 0.10% Securities Transaction Tax on Sell
    exchange_charges_pct: float = 0.0000345   # NSE turnover charges ~0.00345%
    gst_pct: float = 0.18                     # 18% GST on (Brokerage + Exchange)
    sebi_turnover_pct: float = 0.000001       # Rs 10 per crore (0.0001%)
    stamp_duty_buy_pct: float = 0.00015       # 0.015% on Buy turnover only
    estimated_slippage_pct: float = 0.0010    # 10 bps estimated market slippage per side


# ==============================================================================
# 11. BACKTESTING & WALK-FORWARD STANDARDS
# ==============================================================================
@dataclass(frozen=True)
class BacktestConfig:
    """Chronological event backtester execution constraints."""
    default_holding_horizon_days: int = 5     # 3 to 5 sessions target
    execution_lag_sessions: int = 1          # Signal at T close -> Execute at T+1 Open
    annual_risk_free_rate: float = 0.065     # 6.50% Indian 10-Year Sovereign Yield

    # Minimum robustness thresholds for accepting a strategy
    min_out_of_sample_expectancy_pct: float = 0.005 # Min +0.5% net expectancy per trade
    min_profit_factor: float = 1.30
    max_allowable_drawdown_pct: float = -20.0
    min_closed_trades: int = 30
    min_sharpe_ratio: float = 0.80

    # Walk-forward optimization windows
    wf_train_months: int = 24
    wf_val_months: int = 6
    wf_test_months: int = 6
    wf_step_months: int = 6


# ==============================================================================
# 12. MACHINE LEARNING & REPRODUCIBILITY
# ==============================================================================
@dataclass(frozen=True)
class MLConfig:
    """Hyperparameters and cross-validation controls for ML modules."""
    random_state: int = 42
    n_splits_purged_cv: int = 4
    purge_embargo_bars: int = 5              # Embargo bars between train and test
    primary_target: str = "target_15pct_5d"
    all_targets: List[str] = field(
        default_factory=lambda: [
            "target_5pct_3d",
            "target_10pct_5d",
            "target_15pct_5d",
            "target_20pct_5d"
        ]
    )
    test_size_chronological: float = 0.20


# ==============================================================================
# 13. UNIFIED MASTER CONFIGURATION SINGLETON
# ==============================================================================
@dataclass(frozen=True)
class SystemConfig:
    """Master aggregated system configuration."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringWeightsConfig = field(default_factory=ScoringWeightsConfig)
    signal: SignalThresholdsConfig = field(default_factory=SignalThresholdsConfig)
    regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    expected_move: ExpectedMoveConfig = field(default_factory=ExpectedMoveConfig)
    exit_cfg: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ml: MLConfig = field(default_factory=MLConfig)


# Global instance
CONFIG: SystemConfig = SystemConfig()
