"""
NSE MOMENTUM 5™ — Comprehensive Relational Database Models
================================================================================
Defines all SQL schemas for the quantitative trading system:
1. SymbolMaster: Universe ticker metadata and sector classifications
2. DailyOHLCV: Normalized daily market bars with unique constraints & indices
3. MarketRegimeRecord: Historical macro benchmark regime assessments
4. SignalRecord: Engine A momentum candidates and probability calculations
5. PositionRecord: Engine B active holdings, trailing stops, and Hold Scores
6. TradeRecord: Persistent ledger of executed paper or live trades
7. BacktestRunRecord: Historical backtesting performance metadata and robustness
8. BacktestTradeRecord: Granular simulated trade log for backtest runs
9. ModelRunRecord: Machine learning model evaluation and feature importance audits
10. SystemSettingRecord: Dynamic runtime configuration overrides
11. SystemLogRecord: Persistent audit log for system events and warnings
================================================================================
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from database.db import Base


# ==============================================================================
# 1. UNIVERSE & SYMBOL MASTER
# ==============================================================================
class SymbolMaster(Base):
    """Registry of eligible NSE tickers, corporate identifiers, and activity status."""
    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), unique=True, nullable=False, index=True)
    company_name = Column(String(150), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    isin = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 1-to-Many relationship with daily OHLCV bars
    ohlcv_records = relationship(
        "DailyOHLCV",
        back_populates="symbol_rel",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


# ==============================================================================
# 2. NORMALIZED DAILY OHLCV BARS
# ==============================================================================
class DailyOHLCV(Base):
    """Historical daily price and volume records for universe equities and benchmarks."""
    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    symbol_rel = relationship("SymbolMaster", back_populates="ohlcv_records")

    __table_args__ = (
        UniqueConstraint("symbol_id", "timestamp", name="uq_symbol_timestamp"),
        Index("ix_symbol_timestamp", "symbol_id", "timestamp"),
    )


# ==============================================================================
# 3. MARKET REGIME AUDIT TRAIL
# ==============================================================================
class MarketRegimeRecord(Base):
    """Persistent daily records of broad market permission and benchmark trend health."""
    __tablename__ = "market_regime"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False, index=True)
    benchmark_symbol = Column(String(30), nullable=False)
    regime = Column(String(20), nullable=False)  # BULLISH, NEUTRAL, BEARISH
    close_price = Column(Float, nullable=False)
    ema20 = Column(Float, nullable=False)
    ema50 = Column(Float, nullable=False)
    trend_slope = Column(Float, nullable=False)
    regime_score = Column(Float, nullable=False)  # 0.0 to 100.0
    details = Column(Text, nullable=True)


# ==============================================================================
# 4. ENGINE A — MOMENTUM SCANNER SIGNALS
# ==============================================================================
class SignalRecord(Base):
    """Historical record of scanned momentum candidates and conditional probabilities."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(30), nullable=False, index=True)
    strategy_name = Column(String(50), nullable=False)
    signal_category = Column(String(30), nullable=False)  # STRONG MOMENTUM, MOMENTUM, etc.
    momentum_score = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    prob_5pct_3d = Column(Float, nullable=True)
    prob_10pct_5d = Column(Float, nullable=True)
    prob_15pct_5d = Column(Float, nullable=True)
    prob_20pct_5d = Column(Float, nullable=True)
    risk_reward = Column(Float, nullable=True)
    features_json = Column(Text, nullable=True)
    is_valid = Column(Boolean, default=True)


# ==============================================================================
# 5. ENGINE B — ACTIVE HOLDINGS & EXIT INTELLIGENCE
# ==============================================================================
class PositionRecord(Base):
    """Live and paper open positions managed by Engine B Exit Intelligence."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False, index=True)
    entry_date = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    current_stop = Column(Float, nullable=False)
    target_price = Column(Float, nullable=True)
    highest_price_since_entry = Column(Float, nullable=False)
    status = Column(String(20), default="OPEN", index=True)  # OPEN, CLOSED
    exit_action = Column(String(30), nullable=True)  # HOLD, TRAIL, PARTIAL BOOK, EXIT, EMERGENCY EXIT
    hold_score = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)


# ==============================================================================
# 6. TRADE JOURNAL LEDGER
# ==============================================================================
class TradeRecord(Base):
    """Persistent audit ledger of closed trades for analytics and tax/P&L exports."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False, index=True)
    entry_date = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_date = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    gross_pnl = Column(Float, nullable=False)
    total_costs = Column(Float, nullable=False)
    net_pnl = Column(Float, nullable=False)
    net_pnl_pct = Column(Float, nullable=False)
    holding_days = Column(Integer, nullable=False)
    exit_reason = Column(String(100), nullable=False)
    trade_mode = Column(String(20), default="PAPER")  # BACKTEST, PAPER, LIVE
    notes = Column(Text, nullable=True)


# ==============================================================================
# 7. BACKTEST AUDIT & PERFORMANCE SUMMARY
# ==============================================================================
class BacktestRunRecord(Base):
    """Aggregated quantitative performance summary of historical backtest runs."""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow)
    strategy_name = Column(String(50), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    expectancy = Column(Float, nullable=False)
    cumulative_return = Column(Float, nullable=False)
    cagr = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    is_robust = Column(Boolean, default=False)
    parameters_json = Column(Text, nullable=True)

    trades = relationship(
        "BacktestTradeRecord",
        back_populates="run_rel",
        cascade="all, delete-orphan"
    )


class BacktestTradeRecord(Base):
    """Individual trade execution simulated during a backtest run."""
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(30), nullable=False)
    signal_date = Column(DateTime, nullable=False)
    entry_date = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_date = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    net_pnl = Column(Float, nullable=False)
    net_pnl_pct = Column(Float, nullable=False)
    holding_days = Column(Integer, nullable=False)
    exit_reason = Column(String(100), nullable=False)

    run_rel = relationship("BacktestRunRecord", back_populates="trades")


# ==============================================================================
# 8. MACHINE LEARNING MODEL AUDIT
# ==============================================================================
class ModelRunRecord(Base):
    """Evaluation metrics and training provenance for target probability models."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(50), nullable=False)
    target_name = Column(String(50), nullable=False)
    train_start = Column(DateTime, nullable=False)
    train_end = Column(DateTime, nullable=False)
    test_start = Column(DateTime, nullable=False)
    test_end = Column(DateTime, nullable=False)
    roc_auc = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    feature_importance_json = Column(Text, nullable=True)


# ==============================================================================
# 9. DYNAMIC RUNTIME SETTINGS
# ==============================================================================
class SystemSettingRecord(Base):
    """User-editable system settings stored in the database."""
    __tablename__ = "settings"

    key = Column(String(60), primary_key=True)
    value = Column(Text, nullable=False)
    description = Column(String(200), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==============================================================================
# 10. SYSTEM EVENT & AUDIT LOGS
# ==============================================================================
class SystemLogRecord(Base):
    """Persistent audit log table for system events, warnings, and errors."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), nullable=False)
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
