"""
NSE MOMENTUM 5™ — Abstract Market Data Provider Interface
================================================================================
Architectural abstraction layer decoupling the quantitative system from any
single commercial or free data feed vendor.

Core Design Rules:
1. All downstream analytics consume this abstract interface only.
2. Implementations (e.g., YFinance, TrueData, Interactive Brokers, Zerodha Kite)
   must strictly conform to the returned schema contract:
   - Output must be a pandas DataFrame with a DatetimeIndex
   - Lowercase columns: ['open', 'high', 'low', 'close', 'adj_close', 'volume']
   - Prices must be split- and bonus-adjusted to preserve mathematical continuity.
3. Does NOT scrape undocumented endpoints or violate third-party terms of service.
================================================================================
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class BaseDataProvider(ABC):
    """
    Abstract Base Class for Indian Equities and Index Market Data Retrieval.
    """

    @abstractmethod
    def get_historical_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Retrieves historical daily OHLCV bars for a specified symbol.

        Args:
            symbol: Ticker symbol (e.g. 'TRENT.NS', '^NSEI').
            start_date: Starting chronological boundary.
            end_date: Ending chronological boundary.
            interval: Bar aggregation frequency (default '1d').

        Returns:
            pd.DataFrame: DataFrame indexed by DatetimeIndex containing:
                - open: float
                - high: float
                - low: float
                - close: float
                - adj_close: float (split and bonus adjusted)
                - volume: float
            Returns an empty DataFrame if no data is found or on network failure.
        """
        pass

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest available session price snapshot for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Dictionary containing:
                {
                    'symbol': str,
                    'price': float,
                    'volume': float,
                    'timestamp': datetime
                }
            Returns None if unavailable.
        """
        pass

    @abstractmethod
    def get_universe_symbols(self) -> List[str]:
        """
        Returns the list of eligible ticker symbols supported by this provider.

        Returns:
            List[str]: List of valid tickers.
        """
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """
        Performs a ping or lightweight query to verify provider connectivity.

        Returns:
            bool: True if the provider is reachable, False otherwise.
        """
        pass
