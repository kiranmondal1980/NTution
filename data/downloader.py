"""
NSE MOMENTUM 5™ — Bulk Market Data Downloader & Database Synchronizer
================================================================================
Manages batch downloading, cleaning, validation, and upsertion of daily OHLCV
market bars from any `BaseDataProvider` into the relational SQLite database.

Key Capabilities:
1. Resilient Universe Sync: Iterates across symbols with error isolation.
2. End-to-End Pipeline: Download -> Clean -> Validate -> Upsert.
3. Upsert Logic: Seamlessly inserts new bars or updates existing records
   without violating unique constraints on (symbol_id, timestamp).
4. Fast In-Memory Loader: Fetches historical series from the database
   as standardized Pandas DataFrames for backtesting and scanning.
================================================================================
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from config import CONFIG
from database.db import get_db_session
from database.models import SymbolMaster, DailyOHLCV
from data.providers.base import BaseDataProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.cleaner import clean_ohlcv_dataframe, compute_split_adjusted_ohlc
from data.validator import check_data_quality
from utils.logger import setup_logger

logger = setup_logger("MARKET_DOWNLOADER")


class MarketDataDownloader:
    """Coordinates batch historical market data retrieval and database synchronization."""

    def __init__(self, provider: Optional[BaseDataProvider] = None):
        """
        Args:
            provider: Concrete implementation of BaseDataProvider.
                      Defaults to YFinanceProvider.
        """
        self.provider: BaseDataProvider = provider or YFinanceProvider()

    def sync_universe(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 365,
        min_bars_valid: int = 40
    ) -> Dict[str, Any]:
        """
        Downloads and stores daily bars for the specified universe.

        Args:
            symbols: List of tickers to sync. Defaults to CONFIG.universe.default_symbols
                     plus the benchmark index (^NSEI).
            lookback_days: Calendar days of history to fetch.
            min_bars_valid: Minimum bars required to pass data audit.

        Returns:
            Dictionary with execution statistics (total_inserted, total_updated, failed_symbols).
        """
        if not symbols:
            symbols = CONFIG.universe.default_symbols + [CONFIG.universe.benchmark_symbol]

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        stats: Dict[str, Any] = {
            "total_inserted": 0,
            "total_updated": 0,
            "failed_symbols": [],
            "synced_count": 0
        }

        logger.info(
            f"Initiating historical sync for {len(symbols)} tickers "
            f"from {start_date.date()} to {end_date.date()}..."
        )

        with get_db_session() as session:
            for symbol in symbols:
                try:
                    # 1. Ensure symbol exists in SymbolMaster
                    sym_obj = session.query(SymbolMaster).filter(SymbolMaster.symbol == symbol).first()
                    if not sym_obj:
                        sym_obj = SymbolMaster(
                            symbol=symbol,
                            company_name=symbol.replace(".NS", ""),
                            sector="NSE Equity",
                            is_active=True
                        )
                        session.add(sym_obj)
                        session.flush()

                    # 2. Fetch raw OHLCV from provider
                    raw_df = self.provider.get_historical_ohlcv(symbol, start_date, end_date)
                    if raw_df is None or raw_df.empty:
                        logger.warning(f"No bars returned for {symbol}.")
                        stats["failed_symbols"].append(symbol)
                        continue

                    # 3. Clean and compute split-adjusted prices
                    clean_df = clean_ohlcv_dataframe(raw_df)
                    adj_df = compute_split_adjusted_ohlc(clean_df)

                    # 4. Validate data quality
                    audit = check_data_quality(adj_df, symbol=symbol, min_bars=min_bars_valid)
                    if not audit.is_valid:
                        logger.warning(f"Data audit warning for {symbol}: {audit.notes}")

                    # 5. Upsert daily bars into database
                    existing_records = session.query(DailyOHLCV).filter(
                        DailyOHLCV.symbol_id == sym_obj.id,
                        DailyOHLCV.timestamp >= start_date
                    ).all()
                    existing_map = {rec.timestamp: rec for rec in existing_records}

                    for ts, row in adj_df.iterrows():
                        dt_val = pd.to_datetime(ts).to_pydatetime()
                        adj_c = float(row.get("adj_close", row["close"]))

                        if dt_val in existing_map:
                            # Update existing record
                            rec = existing_map[dt_val]
                            rec.open = float(row["open"])
                            rec.high = float(row["high"])
                            rec.low = float(row["low"])
                            rec.close = float(row["close"])
                            rec.adj_close = adj_c
                            rec.volume = float(row["volume"])
                            stats["total_updated"] += 1
                        else:
                            # Insert new bar
                            new_rec = DailyOHLCV(
                                symbol_id=sym_obj.id,
                                timestamp=dt_val,
                                open=float(row["open"]),
                                high=float(row["high"]),
                                low=float(row["low"]),
                                close=float(row["close"]),
                                adj_close=adj_c,
                                volume=float(row["volume"])
                            )
                            session.add(new_rec)
                            stats["total_inserted"] += 1

                    stats["synced_count"] += 1

                except Exception as exc:
                    logger.error(f"Error syncing data for {symbol}: {str(exc)}")
                    stats["failed_symbols"].append(symbol)

            session.commit()

        logger.info(
            f"Universe sync completed. Synced: {stats['synced_count']}, "
            f"Inserted: {stats['total_inserted']}, Updated: {stats['total_updated']}, "
            f"Failed: {len(stats['failed_symbols'])}"
        )
        return stats

    @staticmethod
    def load_ohlcv_from_db(
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Loads clean historical series for a given symbol from the database.

        Args:
            symbol: Ticker symbol.
            start_date: Optional start filter.
            end_date: Optional end filter.

        Returns:
            pd.DataFrame indexed by sorted DatetimeIndex.
        """
        with get_db_session() as session:
            sym_obj = session.query(SymbolMaster).filter(SymbolMaster.symbol == symbol).first()
            if not sym_obj:
                return pd.DataFrame()

            query = session.query(DailyOHLCV).filter(DailyOHLCV.symbol_id == sym_obj.id)
            if start_date:
                query = query.filter(DailyOHLCV.timestamp >= start_date)
            if end_date:
                query = query.filter(DailyOHLCV.timestamp <= end_date)

            records = query.order_by(DailyOHLCV.timestamp.asc()).all()
            if not records:
                return pd.DataFrame()

            data = [{
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "adj_close": r.adj_close,
                "volume": r.volume
            } for r in records]

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df

    def load_universe_from_db(
        self,
        symbols: Optional[List[str]] = None,
        min_bars: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Loads historical bars for multiple universe symbols into memory.

        Returns:
            Dict[symbol, pd.DataFrame]
        """
        sym_list = symbols or CONFIG.universe.default_symbols
        universe_data: Dict[str, pd.DataFrame] = {}

        for sym in sym_list:
            df = self.load_ohlcv_from_db(sym)
            if not df.empty and len(df) >= min_bars:
                universe_data[sym] = df

        return universe_data
