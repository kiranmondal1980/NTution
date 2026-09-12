"""
NSE MOMENTUM 5™ — Yahoo Finance Market Data Provider Implementation
================================================================================
Development & Research Provider implementing `BaseDataProvider`.
Provides resilient historical daily OHLCV bars and quotes for NSE equities.

Features & Resilience:
- Automatic flattening of yfinance MultiIndex column structures
- Strict normalization to standardized lowercase columns
- Naive datetime conversion (prevents timezone mismatch errors)
- Exponential backoff retry logic to gracefully respect rate limits
- Clear labeling: Intended for research, backtesting, and development
================================================================================
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import yfinance as yf
from .base import BaseDataProvider
from utils.logger import setup_logger

logger = setup_logger("YFINANCE_PROVIDER")


class YFinanceProvider(BaseDataProvider):
    """
    Robust Yahoo Finance implementation of BaseDataProvider.
    """

    def __init__(self, symbols: Optional[List[str]] = None, max_retries: int = 3):
        """
        Args:
            symbols: Optional list of universe symbols.
            max_retries: Maximum download retry attempts per request.
        """
        self.symbols: List[str] = symbols or []
        self.max_retries: int = max_retries

    def get_historical_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Downloads and normalizes daily OHLCV bars for an NSE symbol.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                # auto_adjust=False keeps unadjusted Close and calculates Adj Close
                df = ticker.history(
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    auto_adjust=False
                )

                if df is None or df.empty:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: No data returned for {symbol} "
                        f"between {start_str} and {end_str}"
                    )
                    time.sleep(1.0 * attempt)
                    continue

                # 1. Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                # 2. Normalize column names to lowercase standard
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume"
                })

                # If adj_close is absent, default to close
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]

                # 3. Filter down strictly to standard schema
                required_cols = ["open", "high", "low", "close", "adj_close", "volume"]
                available_cols = [col for col in required_cols if col in df.columns]
                df = df[available_cols]

                # 4. Standardize Index to naive UTC timestamp
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Ensure non-empty and valid rows
                df = df.dropna(subset=["open", "high", "low", "close"])
                if not df.empty:
                    return df

            except Exception as exc:
                logger.error(
                    f"Attempt {attempt}/{self.max_retries} failed for {symbol}: {str(exc)}"
                )
                time.sleep(1.5 * attempt)

        logger.error(f"Exhausted {self.max_retries} attempts fetching data for {symbol}.")
        return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves current snapshot quote using yfinance fast_info.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            # Retrieve price from fast_info attributes
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            volume = getattr(info, "last_volume", 0.0)

            if price is not None and price > 0:
                return {
                    "symbol": symbol,
                    "price": float(price),
                    "volume": float(volume),
                    "timestamp": datetime.utcnow()
                }
            return None
        except Exception as exc:
            logger.warning(f"Could not retrieve fast quote for {symbol}: {exc}")
            return None

    def get_universe_symbols(self) -> List[str]:
        """
        Returns list of symbols registered with this provider.
        """
        return self.symbols

    def check_health(self) -> bool:
        """
        Checks connectivity to Yahoo Finance via a benchmark ticker quote.
        """
        try:
            quote = self.get_latest_quote("^NSEI")
            return quote is not None and quote["price"] > 0
        except Exception:
            return False
