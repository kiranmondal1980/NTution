"""
NSE MOMENTUM 5™ — Relational Database Initializer & Universe Seeder
================================================================================
Idempotent initialization script that:
1. Verifies database connectivity
2. Automatically creates all relational tables, unique constraints, and indices
3. Seeds the `SymbolMaster` table with the default liquid NSE universe and benchmarks
4. Seeds default system parameters into the `SystemSettingRecord` table
5. Can be safely executed multiple times without corrupting or duplicating records
================================================================================
"""

import sys
from datetime import datetime
from config import CONFIG
from database.db import engine, Base, get_db_session, check_db_connection
from database.models import SymbolMaster, SystemSettingRecord
from utils.logger import setup_logger

logger = setup_logger("DB_INITIALIZE")


def init_database() -> bool:
    """
    Creates database tables and populates default tickers and system settings.

    Returns:
        bool: True if initialization completed successfully, False otherwise.
    """
    logger.info("Verifying database connectivity...")
    if not check_db_connection():
        logger.critical("Database connection test failed. Halting initialization.")
        return False

    logger.info("Creating database tables from ORM metadata...")
    try:
        # Create all tables defined in database.models
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables and indices created successfully.")
    except Exception as exc:
        logger.critical(f"Failed to create database schema: {exc}")
        return False

    # Seed Universe and Benchmarks
    with get_db_session() as session:
        existing_symbols_count = session.query(SymbolMaster).count()
        if existing_symbols_count == 0:
            logger.info(
                f"Seeding default universe: {len(CONFIG.universe.default_symbols)} equities "
                f"+ 2 benchmark indices..."
            )

            # 1. Seed Benchmark Indices
            benchmarks = [
                (CONFIG.universe.benchmark_symbol, "NIFTY 50 Benchmark Index", "Benchmark", "Index"),
                (CONFIG.universe.nifty500_symbol, "NIFTY 500 Broad Market Index", "Benchmark", "Index")
            ]
            for sym, name, sec, ind in benchmarks:
                session.add(SymbolMaster(
                    symbol=sym,
                    company_name=name,
                    sector=sec,
                    industry=ind,
                    is_active=True,
                    created_at=datetime.utcnow()
                ))

            # 2. Seed Default Liquid Swing Universe
            for ticker in CONFIG.universe.default_symbols:
                readable_name = ticker.replace(".NS", "")
                session.add(SymbolMaster(
                    symbol=ticker,
                    company_name=readable_name,
                    sector="NSE Equity",
                    industry="Cash Segment",
                    is_active=True,
                    created_at=datetime.utcnow()
                ))

            session.commit()
            logger.info(
                f"Universe seeding complete. "
                f"Total entries inserted: {len(CONFIG.universe.default_symbols) + 2}"
            )
        else:
            logger.info(
                f"Universe already populated with {existing_symbols_count} symbols. "
                "Skipping ticker seeding."
            )

        # Seed Default System Settings
        existing_settings_count = session.query(SystemSettingRecord).count()
        if existing_settings_count == 0:
            logger.info("Populating default runtime system settings...")
            defaults = [
                ("capital_inr", str(CONFIG.risk.default_capital_inr), "Total available trading capital in INR"),
                ("risk_per_trade_pct", str(CONFIG.risk.risk_per_trade_pct), "Risk per trade as a fraction of equity"),
                ("max_portfolio_heat", str(CONFIG.risk.max_portfolio_risk_pct), "Max portfolio aggregate heat"),
                ("max_positions", str(CONFIG.risk.max_simultaneous_positions), "Maximum concurrent open positions"),
                ("default_atr_mult", str(CONFIG.exit_cfg.default_atr_multiplier), "Base ATR trailing multiplier"),
                ("min_momentum_score", str(CONFIG.signal.momentum), "Minimum momentum score to trigger long entry")
            ]
            for key, val, desc in defaults:
                session.add(SystemSettingRecord(
                    key=key,
                    value=val,
                    description=desc,
                    updated_at=datetime.utcnow()
                ))
            session.commit()
            logger.info(f"Populated {len(defaults)} default configuration settings.")

    logger.info("Database initialization verified and complete.")
    return True


if __name__ == "__main__":
    success = init_database()
    if not success:
        sys.exit(1)
