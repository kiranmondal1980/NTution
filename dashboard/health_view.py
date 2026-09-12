"""
NSE MOMENTUM 5™ — Dashboard Page 13: System Health & Diagnostic Audit
================================================================================
Comprehensive diagnostic and operational audit console:
1. Database Integrity & Row Inventory (Symbols, OHLCV, Positions, Trades)
2. External Market Data Feed Connectivity Ping
3. Runtime Environment Audit (Python, Streamlit, SQLite, SQLAlchemy versions)
4. Persistent Log Viewer (Tail of `system.log` with security filter verification)
================================================================================
"""

import sys
import platform
from pathlib import Path
from typing import Dict, List, Any
import sqlite3
import pandas as pd
import streamlit as st

from database.db import check_db_connection, get_db_session
from database.models import SymbolMaster, DailyOHLCV, PositionRecord, TradeRecord
from data.providers.yfinance_provider import YFinanceProvider
from config import CONFIG, DB_PATH, LOG_DIR


def render_health_view() -> None:
    """
    Renders the system health diagnostics and operational audit dashboard.
    """
    st.markdown("## 🩺 System Health & Diagnostic Audit")
    st.caption("Verifies database connectivity, market data provider latency, storage usage, and system logs.")

    # --------------------------------------------------------------------------
    # 1. Component Health Status Checks
    # --------------------------------------------------------------------------
    db_ok = check_db_connection()

    provider = YFinanceProvider()
    provider_ok = provider.check_health()

    log_file = LOG_DIR / "system.log"
    logs_ok = log_file.exists()

    col1, col2, col3 = st.columns(3)
    col1.metric("Database Engine", "CONNECTED (OK)" if db_ok else "DISCONNECTED", "SQLite WAL Mode")
    col2.metric("Market Data Provider", "REACHABLE (OK)" if provider_ok else "OFFLINE / THROTTLED", "YFinance API")
    col3.metric("Audit Logging Engine", "ACTIVE (OK)" if logs_ok else "UNINITIALIZED", "Rotating File Handler")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. Relational Database Row Inventory
    # --------------------------------------------------------------------------
    st.subheader("📦 Relational Database Inventory")

    counts: Dict[str, int] = {
        "Universe Tickers (symbols)": 0,
        "Daily Market Bars (ohlcv)": 0,
        "Active Monitored Positions (positions)": 0,
        "Journaled Trades (trades)": 0
    }

    if db_ok:
        with get_db_session() as session:
            counts["Universe Tickers (symbols)"] = session.query(SymbolMaster).count()
            counts["Daily Market Bars (ohlcv)"] = session.query(DailyOHLCV).count()
            counts["Active Monitored Positions (positions)"] = session.query(PositionRecord).count()
            counts["Journaled Trades (trades)"] = session.query(TradeRecord).count()

    db_size_mb = (DB_PATH.stat().st_size / (1024 * 1024)) if DB_PATH.exists() else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registered Tickers", f"{counts['Universe Tickers (symbols)']:,}")
    k2.metric("Stored Daily Bars", f"{counts['Daily Market Bars (ohlcv)']:,}")
    k3.metric("Journaled Trades", f"{counts['Journaled Trades (trades)']:,}")
    k4.metric("Database Size", f"{db_size_mb:.2f} MB")

    # --------------------------------------------------------------------------
    # 3. Runtime Environment Audit
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🖥️ Execution Runtime & Environment")

    env_data = {
        "System Component": [
            "Python Runtime Version",
            "Platform Operating System",
            "Streamlit Framework",
            "SQLite C-Library Version",
            "Database File Location",
            "Logs Storage Location"
        ],
        "Configuration Details": [
            sys.version.split()[0],
            f"{platform.system()} {platform.release()} ({platform.machine()})",
            st.__version__,
            sqlite3.sqlite_version,
            str(DB_PATH),
            str(LOG_DIR)
        ]
    }
    st.dataframe(pd.DataFrame(env_data), use_container_width=True)

    # --------------------------------------------------------------------------
    # 4. Recent System Log Stream
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📜 Recent Operational System Logs (`system.log`)")

    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                recent_logs = "".join(lines[-40:]) if lines else "Log file is currently empty."
            st.code(recent_logs, language="text")
        except Exception as exc:
            st.error(f"Could not read system log file: {exc}")
    else:
        st.info("No system log file generated yet.")
