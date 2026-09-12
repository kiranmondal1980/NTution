"""
NSE MOMENTUM 5™ — CLI Tool: Historical Market Data Synchronization
================================================================================
Command-line utility to download, validate, and synchronize historical daily OHLCV
market bars for the NSE liquid universe and benchmark index (^NSEI).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --days 365
    python scripts/download_data.py --symbols RELIANCE.NS,TCS.NS,INFY.NS --days 180
================================================================================
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.initialize import init_database
from data.downloader import MarketDataDownloader
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("CLI_DOWNLOAD_DATA")


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE MOMENTUM 5™ Historical Data Ingestion")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Historical calendar days lookback to download (default: 365 days)."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated list of symbols to download (defaults to full universe)."
    )
    args = parser.parse_args()

    # 1. Guarantee database schema is initialized
    init_database()

    # 2. Parse target symbols
    target_symbols = None
    if args.symbols.strip():
        target_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        logger.info(f"Targeting custom symbol list ({len(target_symbols)} symbols).")

    # 3. Execute historical synchronization
    downloader = MarketDataDownloader()
    logger.info(f"Starting universe data sync (Lookback: {args.days} days)...")

    stats = downloader.sync_universe(symbols=target_symbols, lookback_days=args.days)

    logger.info("=" * 60)
    logger.info("HISTORICAL DATA INGESTION SUMMARY:")
    logger.info(f"  • Tickers Processed     : {stats['synced_count']}")
    logger.info(f"  • New OHLCV Bars Inserted: {stats['total_inserted']:,}")
    logger.info(f"  • Existing Bars Updated : {stats['total_updated']:,}")
    logger.info(f"  • Failed Tickers Count  : {len(stats['failed_symbols'])}")
    if stats["failed_symbols"]:
        logger.warning(f"  • Failed Symbols        : {stats['failed_symbols']}")
    logger.info("=" * 60)

    # Return success code
    return 0


if __name__ == "__main__":
    sys.exit(main())
