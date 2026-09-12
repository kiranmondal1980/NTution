"""
NSE MOMENTUM 5™ — Dashboard Page 12: Settings & Data Synchronization
================================================================================
Central configuration and administrative control console:
1. Historical Market Data Sync Engine (Bulk ingest OHLCV bars into SQLite)
2. Universe Ticker Management (Add/remove NSE equities, toggle active state)
3. Dynamic Runtime Risk & Strategy Configuration Overrides
4. Database Maintenance & Streamlit Cache Invalidation
================================================================================
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st

from data.downloader import MarketDataDownloader
from database.db import get_db_session
from database.models import SymbolMaster, SystemSettingRecord
from database.initialize import init_database
from config import CONFIG


def render_settings_view() -> None:
    """
    Renders the central system administration and data ingestion console.
    """
    st.markdown("## ⚙️ System Settings & Data Synchronization")
    st.caption("Manage market data feeds, configure risk parameters, and maintain the universe registry.")

    # --------------------------------------------------------------------------
    # 1. Historical Market Data Sync Panel
    # --------------------------------------------------------------------------
    st.subheader("🔄 Market Data Synchronization Engine")
    st.info(
        "💡 **DATA PROVIDER STATUS: ACTIVE (YFinance Developmental Provider)**\n\n"
        "Downloads daily OHLCV bars from Yahoo Finance, executes geometry validation, "
        "computes split adjustments, and upserts bars into your local relational SQLite store."
    )

    c_sync1, c_sync2 = st.columns([2, 1])
    lookback = c_sync1.slider("Historical Lookback (Calendar Days)", 90, 730, 365, step=30)

    if c_sync2.button("📥 Sync Market Data Now", type="primary"):
        with st.spinner("Connecting to provider, cleaning bars, and persisting to SQLite..."):
            downloader = MarketDataDownloader()
            stats = downloader.sync_universe(lookback_days=lookback)

            st.success(
                f"✅ Data synchronization complete!\n\n"
                f"• **Symbols Processed:** {stats['synced_count']}\n"
                f"• **New Bars Inserted:** {stats['total_inserted']:,}\n"
                f"• **Existing Bars Updated:** {stats['total_updated']:,}\n"
                f"• **Failed / Unavailable:** {len(stats['failed_symbols'])}"
            )
            # Invalidate cached feature calculations
            st.cache_data.clear()

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. Dynamic Runtime Parameters
    # --------------------------------------------------------------------------
    st.subheader("📐 Runtime Strategy & Risk Overrides")

    with get_db_session() as session:
        settings_records = session.query(SystemSettingRecord).all()
        settings_dict = {s.key: s.value for s in settings_records}

    with st.form("runtime_settings_form"):
        col1, col2, col3 = st.columns(3)
        capital_val = col1.number_input(
            "Account Equity (INR)",
            value=float(settings_dict.get("capital_inr", CONFIG.risk.default_capital_inr)),
            step=50000.0
        )
        risk_pct_val = col2.slider(
            "Risk Per Trade %",
            0.5, 3.0,
            float(settings_dict.get("risk_per_trade_pct", CONFIG.risk.risk_per_trade_pct)) * 100.0,
            step=0.1
        ) / 100.0
        max_pos_val = col3.number_input(
            "Max Concurrent Positions",
            min_value=1, max_value=15,
            value=int(settings_dict.get("max_positions", CONFIG.risk.max_simultaneous_positions))
        )

        col4, col5 = st.columns(2)
        atr_mult_val = col4.slider(
            "Default ATR Trailing Multiplier",
            1.0, 3.5,
            float(settings_dict.get("default_atr_mult", CONFIG.exit_cfg.default_atr_multiplier)),
            step=0.1
        )
        min_score_val = col5.slider(
            "Min Momentum Score Trigger",
            50.0, 90.0,
            float(settings_dict.get("min_momentum_score", CONFIG.signal.momentum)),
            step=2.5
        )

        saved = st.form_submit_button("Save Configuration to Database")
        if saved:
            with get_db_session() as session:
                updates = {
                    "capital_inr": str(capital_val),
                    "risk_per_trade_pct": str(risk_pct_val),
                    "max_positions": str(max_pos_val),
                    "default_atr_mult": str(atr_mult_val),
                    "min_momentum_score": str(min_score_val)
                }
                for k, v in updates.items():
                    rec = session.query(SystemSettingRecord).filter(SystemSettingRecord.key == k).first()
                    if rec:
                        rec.value = v
                        rec.updated_at = datetime.utcnow()
                    else:
                        session.add(SystemSettingRecord(key=k, value=v, updated_at=datetime.utcnow()))
                session.commit()
            st.success("Configuration successfully saved to database!")
            st.cache_data.clear()

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Universe Ticker Management
    # --------------------------------------------------------------------------
    st.subheader("📋 Universe Ticker Management")

    with get_db_session() as session:
        all_symbols = session.query(SymbolMaster).order_by(SymbolMaster.symbol.asc()).all()
        sym_data = [{
            "ID": s.id,
            "Symbol": s.symbol,
            "Company": s.company_name or "",
            "Sector": s.sector or "",
            "Active": "YES" if s.is_active else "NO"
        } for s in all_symbols]

    st.dataframe(pd.DataFrame(sym_data), use_container_width=True, height=260)

    # Add custom ticker form
    with st.expander("➕ Add Custom Symbol to Universe"):
        with st.form("add_symbol_form"):
            new_ticker = st.text_input("New NSE Ticker (e.g. ZOMATO.NS)", "ZOMATO.NS")
            comp_name = st.text_input("Company Name", "Zomato Ltd")
            sector = st.text_input("Sector", "Technology")
            add_sub = st.form_submit_button("Add Symbol")

            if add_sub:
                clean_t = new_ticker.strip().upper()
                if not clean_t.endswith(".NS") and not clean_t.startswith("^"):
                    clean_t = f"{clean_t}.NS"

                with get_db_session() as session:
                    existing = session.query(SymbolMaster).filter(SymbolMaster.symbol == clean_t).first()
                    if not existing:
                        session.add(SymbolMaster(
                            symbol=clean_t,
                            company_name=comp_name,
                            sector=sector,
                            is_active=True
                        ))
                        session.commit()
                        st.success(f"Added {clean_t} to Symbol Registry!")
                        st.rerun()
                    else:
                        st.warning(f"{clean_t} already exists in Symbol Registry.")

    # --------------------------------------------------------------------------
    # 4. Database Maintenance
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🛠️ Database Maintenance")
    c_maint1, c_maint2 = st.columns(2)

    if c_maint1.button("🧹 Clear Streamlit Application Cache"):
        st.cache_data.clear()
        st.success("Application memory cache cleared successfully.")

    if c_maint2.button("🔧 Re-Verify Database Schema Integrity"):
        success = init_database()
        if success:
            st.success("Database schema verified and aligned.")
        else:
            st.error("Database schema verification failed.")
