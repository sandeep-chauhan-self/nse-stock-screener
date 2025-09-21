"""
Data source connectors with plugin architecture.
This module provides a unified interface for different data sources
(NSE, yfinance, etc.) with automatic retry, caching, and error handling.
"""
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict[str, Any], List[str], Optional, Union, Type, Any
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
logger = logging.getLogger(__name__)
@dataclass
class DataRequest:
    """Standard data request format across all connectors."""
    symbol: str
    start_date: date
    end_date: date
    data_type: str = "ohlcv"
  # ohlcv, corporate_actions, fundamentals
    frequency: str = "1d"
     # 1d, 1h, 5m, etc.
    adjusted: bool = True
     # Return adjusted prices
    metadata: Dict[str, Any] = field(default_factory=Dict[str, Any])
@dataclass
class DataResponse:
    """Standard response format from all connectors."""
    symbol: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    source: str
    timestamp: datetime
    checksum: str
    errors: List[str] = field(default_factory=List[str])
    warnings: List[str] = field(default_factory=List[str])
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if self.checksum == "":
            self.checksum = self._calculate_checksum()
    def _calculate_checksum(self) -> str:
        """Calculate MD5 checksum of the data."""
        if self.data.empty:
            return ""

        # Create string representation of data for hashing
        data_str = f"{self.symbol}_{self.source}_{len(self.data)}"
        if not self.data.empty:
            data_str += f"_{self.data.iloc[0].to_dict()}_{self.data.iloc[-1].to_dict()}"
        return hashlib.md5(data_str.encode()).hexdigest()
class DataConnector(ABC):
    """Abstract base class for all data connectors."""
    def __init__(self, name: str, config: Dict[str, Any] = None) -> None:
        self.name = name
        self.config = config or {}
        self.session = self._create_session()
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.get('rate_limit', 1.0)
        )
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set[str] user agent
        session.headers.update({
            'User-Agent': 'NSE-Stock-Screener/1.0'
        })
        return session
    @abstractmethod
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data for the given request."""
        pass
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get List[str] of available symbols from this connector."""
        pass
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for this connector."""
        pass
    def is_healthy(self) -> bool:
        """Check if the connector is healthy and responsive."""
        try:

            # Simple health check - try to get available symbols
            symbols = self.get_available_symbols()
            return len(symbols) > 0
        except Exception as e:
            logger.warning(f"Health check failed for {self.name}: {e}")
            return False
class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, requests_per_second: float = 1.0) -> None:
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
class NSEConnector(DataConnector):
    """Connector for NSE data sources (Bhavcopy, APIs)."""
    BASE_URL = "https://www.nseindia.com"
    BHAVCOPY_URL = "https://archives.nseindia.com/content/historical/EQUITIES"
    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__("nse", config)
        self._symbol_cache = None
        self._symbol_cache_time = None
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from NSE sources."""
        self.rate_limiter.wait_if_needed()
        try:
            if request.data_type == "ohlcv":
                data = self._fetch_historical_data(request)
            elif request.data_type == "corporate_actions":
                data = self._fetch_corporate_actions(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            return DataResponse(
                symbol=request.symbol,
                data=data,
                metadata={
                    "request": request,
                    "fetch_method": request.data_type
                },
                source="nse",
                timestamp=datetime.now(),
                checksum=""
  # Will be calculated in __post_init__
            )
        except Exception as e:
            logger.error(f"NSE fetch failed for {request.symbol}: {e}")
            return DataResponse(
                symbol=request.symbol,
                data=pd.DataFrame(),
                metadata={"request": request},
                source="nse",
                timestamp=datetime.now(),
                checksum="",
                errors=[str(e)]
            )
    def _fetch_historical_data(self, request: DataRequest) -> pd.DataFrame:
        """Fetch historical OHLCV data from NSE."""

        # For now, use a simplified approach
        # In production, this would fetch from NSE Bhavcopy archives

        # Placeholder implementation - would need to:
        # 1. Download bhavcopy files for date range

        # 2. Parse CSV data
        # 3. Filter for specific symbol

        # 4. Convert to standard OHLCV format
        logger.warning("NSE historical data fetch not fully implemented")
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    def _fetch_corporate_actions(self, request: DataRequest) -> pd.DataFrame:
        """Fetch corporate actions data from NSE."""

        # Would fetch from NSE corporate actions API/files
        logger.warning("NSE corporate actions fetch not fully implemented")
        return pd.DataFrame(columns=['Date', 'Action', 'Ratio', 'ExDate', 'RecordDate'])
    def get_available_symbols(self) -> List[str]:
        """Get List[str] of NSE symbols."""

        # Cache symbols for 1 hour
        if (self._symbol_cache is None or
            self._symbol_cache_time is None or
            time.time() - self._symbol_cache_time > 3600):
            try:

                # This would typically fetch from NSE symbol master
                # For now, use existing symbols file if available
                symbols_file = Path(__file__).parent.parent.parent / "data" / "nse_only_symbols.txt"
                if symbols_file.exists():
                    with open(symbols_file, 'r') as f:
                        symbols = [line.strip() for line in f if line.strip()]
                else:

                    # Fallback to common NSE symbols
                    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                              "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "HINDUNILVR"]
                self._symbol_cache = symbols
                self._symbol_cache_time = time.time()
            except Exception as e:
                logger.error(f"Failed to fetch NSE symbols: {e}")
                return []
        return self._symbol_cache or []
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid NSE symbol."""
        available_symbols = self.get_available_symbols()
        return symbol.upper() in [s.upper() for s in available_symbols]
class YfinanceConnector(DataConnector):
    """Connector for Yahoo Finance data."""
    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__("yfinance", config)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from Yahoo Finance."""
        self.rate_limiter.wait_if_needed()
        try:
            if request.data_type == "ohlcv":
                data = self._fetch_historical_data(request)
            elif request.data_type == "corporate_actions":
                data = self._fetch_corporate_actions(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            return DataResponse(
                symbol=request.symbol,
                data=data,
                metadata={
                    "request": request,
                    "yfinance_symbol": self._convert_to_yf_symbol(request.symbol)
                },
                source="yfinance",
                timestamp=datetime.now(),
                checksum=""
            )
        except Exception as e:
            logger.error(f"YFinance fetch failed for {request.symbol}: {e}")
            return DataResponse(
                symbol=request.symbol,
                data=pd.DataFrame(),
                metadata={"request": request},
                source="yfinance",
                timestamp=datetime.now(),
                checksum="",
                errors=[str(e)]
            )
    def _fetch_historical_data(self, request: DataRequest) -> pd.DataFrame:
        """Fetch historical data from yfinance."""
        yf_symbol = self._convert_to_yf_symbol(request.symbol)
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(
            start=request.start_date,
            end=request.end_date + timedelta(days=1),
  # yfinance end is exclusive
            auto_adjust=request.adjusted,
            prepost=False,
            threads=True
        )
        if data.empty:
            raise ValueError(f"No data returned for {yf_symbol}")

        # Standardize column names
        data = data.reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        # Keep only OHLCV columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])
        return data
    def _fetch_corporate_actions(self, request: DataRequest) -> pd.DataFrame:
        """Fetch corporate actions (splits, dividends) from yfinance."""
        yf_symbol = self._convert_to_yf_symbol(request.symbol)
        ticker = yf.Ticker(yf_symbol)

        # Get splits and dividends
        splits = ticker.splits
        dividends = ticker.dividends
        actions = []

        # Process splits
        for date, ratio in splits.items():
            if request.start_date <= date.date() <= request.end_date:
                actions.append({
                    'Date': date,
                    'Action': 'split',
                    'Ratio': ratio,
                    'Amount': None,
                    'ExDate': date,
                    'RecordDate': None
                })

        # Process dividends
        for date, amount in dividends.items():
            if request.start_date <= date.date() <= request.end_date:
                actions.append({
                    'Date': date,
                    'Action': 'dividend',
                    'Ratio': None,
                    'Amount': amount,
                    'ExDate': date,
                    'RecordDate': None
                })
        return pd.DataFrame(actions)
    def _convert_to_yf_symbol(self, nse_symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance symbol format."""

        # NSE symbols in Yahoo Finance typically have .NS suffix
        if not nse_symbol.endswith('.NS'):
            return f"{nse_symbol}.NS"
        return nse_symbol
    def get_available_symbols(self) -> List[str]:
        """Get available symbols - Yahoo Finance doesn't have a direct API for this."""

        # Return common NSE symbols that work with Yahoo Finance
        return [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "HINDUNILVR.NS",
            "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "HCLTECH.NS"
        ]
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol by attempting to fetch basic info."""
        try:
            yf_symbol = self._convert_to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            return 'symbol' in info and info['symbol'] is not None
        except:
            return False
class ConnectorRegistry:
    """Registry for managing data connectors."""
    _connectors: Dict[str, Type[DataConnector]] = {}
    _instances: Dict[str, DataConnector] = {}
    @classmethod
    def register(cls, name: str, connector_class: Type[DataConnector]):
        """Register a connector class."""
        cls._connectors[name] = connector_class
        logger.info(f"Registered connector: {name}")
    @classmethod
    def get_connector(cls, name: str, config: Dict[str, Any] = None) -> DataConnector:
        """Get a connector instance."""
        if name not in cls._instances:
            if name not in cls._connectors:
                raise ValueError(f"Unknown connector: {name}")
            connector_class = cls._connectors[name]
            cls._instances[name] = connector_class(config)
        return cls._instances[name]
    @classmethod
    def list_connectors(cls) -> List[str]:
        """List[str] all registered connectors."""
        return List[str](cls._connectors.keys())
    @classmethod
    def clear_instances(cls):
        """Clear all connector instances (useful for testing)."""
        cls._instances.clear()

# Register built-in connectors
ConnectorRegistry.register("nse", NSEConnector)
ConnectorRegistry.register("yfinance", YfinanceConnector)
def get_connector(name: str, config: Dict[str, Any] = None) -> DataConnector:
    """Convenience function to get a connector."""
    return ConnectorRegistry.get_connector(name, config)
def fetch_data_from_multiple_sources(
    request: DataRequest,
    sources: List[str],
    configs: Dict[str, Dict[str, Any]] = None
) -> Dict[str, DataResponse]:
    """Fetch data from multiple sources and return all responses."""
    configs = configs or {}
    responses = {}
    for source in sources:
        try:
            connector = get_connector(source, configs.get(source))
            response = connector.fetch_data(request)
            responses[source] = response
        except Exception as e:
            logger.error(f"Failed to fetch from {source}: {e}")
            responses[source] = DataResponse(
                symbol=request.symbol,
                data=pd.DataFrame(),
                metadata={"request": request},
                source=source,
                timestamp=datetime.now(),
                checksum="",
                errors=[str(e)]
            )
    return responses
def get_best_data_source(responses: Dict[str, DataResponse]) -> Optional[str]:
    """Determine the best data source from multiple responses."""

    # Simple heuristic: prefer source with most data and no errors
    best_source = None
    best_score = -1
    for source, response in responses.items():
        if response.errors:
            continue
        score = len(response.data)
        if score > best_score:
            best_score = score
            best_source = source
    return best_source
