"""
NSE MOMENTUM 5™ — Comprehensive Diagnostic & Function Audit Script
================================================================================
Inspects database connectivity, symbol registries, raw OHLCV retrieval,
symbol lookup matching, and feature pipeline execution.

Usage:
    python scripts/debug_diagnostics.py
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.initialize import init_database
from database.db import get_db_session, check_db_connection
from database.models import SymbolMaster, DailyOHLCV, PositionRecord
from data.downloader import MarketDataDownloader
from features import build_feature_pipeline
from strategy.exit_engine import ExitIntelligenceEngine
from utils.logger import setup_logger

logger = setup_logger("DEBUG_DIAGNOSTICS")


def run_diagnostics() -> None:
    print("\n" + "=" * 70)
    print("      NSE MOMENTUM 5™ — SYSTEM DIAGNOSTIC & FUNCTION AUDIT")
    print("=" * 70)

    # 1. Database Connection Check
    db_status = check_db_connection()
    print(f"[1] Database Connection Test : {'PASSED (OK)' if db_status else 'FAILED'}")
    if not db_status:
        print("    -> ERROR: Cannot connect to database. Run `python scripts/initialize_database.py`")
        return

    # 2. Inspect Symbol Master Registry
    with get_db_session() as session:
        symbols = session.query(SymbolMaster).all()
        print(f"[2] SymbolMaster Registry   : {len(symbols)} total tickers registered.")
        
        atgl_rec = session.query(SymbolMaster).filter(SymbolMaster.symbol.like("%ATGL%")).first()
        if atgl_rec:
            print(f"    -> Found ATGL record: ID={atgl_rec.id}, Symbol={atgl_rec.symbol}, Active={atgl_rec.is_active}")
            
            # Check OHLCV bars for ATGL
            bars = session.query(DailyOHLCV).filter(DailyOHLCV.symbol_id == atgl_rec.id).all()
            print(f"    -> Stored Daily OHLCV Bars : {len(bars)} bars found in SQLite.")
            if bars:
                latest_bar = bars[-1]
                print(f"       Latest Bar Date: {latest_bar.timestamp.date()} | Close: INR {latest_bar.close:.2f}")
            else:
                print("       WARNING: 0 bars found for ATGL.NS! Run `python scripts/download_data.py` to sync.")
        else:
            print("    -> WARNING: 'ATGL' / 'ATGL.NS' NOT found in SymbolMaster registry!")

    # 3. Test MarketDataDownloader OHLCV Loading
    print("\n[3] Testing MarketDataDownloader.load_ohlcv_from_db('ATGL.NS')...")
    downloader = MarketDataDownloader()
    df_atgl = downloader.load_ohlcv_from_db("ATGL.NS")
    if not df_atgl.empty:
        print(f"    -> SUCCESS: Loaded {len(df_atgl)} rows for ATGL.NS.")
        print(f"       Latest Close from DB: INR {df_atgl['close'].iloc[-1]:.2f}")
    else:
        print("    -> FAILED: load_ohlcv_from_db returned empty DataFrame for ATGL.NS.")
        # Try without suffix search
        df_alt = downloader.load_ohlcv_from_db("ATGL")
        if not df_alt.empty:
            print(f"       Found data under 'ATGL' (without .NS): {len(df_alt)} rows. Latest Close: INR {df_alt['close'].iloc[-1]:.2f}")

    # 4. Test Feature Pipeline & Exit Intelligence on ATGL
    if not df_atgl.empty:
        print("\n[4] Testing Feature Pipeline & Exit Engine on ATGL.NS...")
        bench_df = downloader.load_ohlcv_from_db("^NSEI")
        pipeline_df = build_feature_pipeline(df_atgl, benchmark_df=bench_df)
        print(f"    -> Feature Pipeline generated {len(pipeline_df.columns)} columns.")
        
        last_row = pipeline_df.iloc[-1]
        exit_engine = ExitIntelligenceEngine()
        assessment = exit_engine.evaluate_position(
            symbol="ATGL.NS",
            entry_price=float(last_row["close"]),
            current_price=float(last_row["close"]),
            highest_price_since_entry=float(last_row["high"]),
            latest_features=last_row
        )
        print(f"    -> Exit Engine Evaluation: Action={assessment.action.value}, Hold Score={assessment.hold_score}/100")
    else:
        print("\n[4] Skipping Feature Pipeline test because ATGL.NS data is empty.")

    # 5. Inspect Active Positions in DB
    with get_db_session() as session:
        positions = session.query(PositionRecord).all()
        print(f"\n[5] Active Positions in DB  : {len(positions)} positions registered.")
        for p in positions:
            print(f"    -> ID={p.id}, Symbol={p.symbol}, Entry={p.entry_price}, Qty={p.quantity}, Stop={p.current_stop}")

    print("\n" + "=" * 70)
    print("      DIAGNOSTIC AUDIT COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    init_database()
    run_diagnostics()
