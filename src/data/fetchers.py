"""
Robust data fetchers with retry logic, rate limiting, and error handling.

This module provides production-ready data fetching capabilities that can handle:
- Network failures with exponential backoff
- Rate limiting to avoid being blocked
- User-agent rotation for web scraping
- Corporate action handling for price data
- Data validation and quality checks
"""

from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import hashlib
import json
import logging
import random
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
import warnings
import yfinance as yf

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class FetchConfig:
    """Configuration for data fetching behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    timeout: int = 30  # Request timeout in seconds
    rate_limit_delay: float = 0.5  # Delay between requests
    cache_ttl_hours: int = 24  # Cache time-to-live in hours
    use_cache: bool = True
    validate_data: bool = True


def retry_with_backoff(config: FetchConfig):
    """
    Decorator for adding retry logic with exponential backoff to functions.

    Args:
        config: FetchConfig with retry parameters
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.base_delay

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(f"Function {func.__name__} failed after {config.max_retries} retries: {e}")
                        break

                    # Calculate delay with jitter
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = min(delay * jitter, config.max_delay)

                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time:.1f}s")
                    time.sleep(sleep_time)

                    # Exponential backoff
                    delay *= config.backoff_factor

            raise last_exception
        return wrapper
    return decorator


class UserAgentRotator:
    """Rotates through different user agents to avoid detection"""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]

    def __init__(self):
        self.current_index = 0

    def get_random(self) -> str:
        """Get a random user agent"""
        return random.choice(self.USER_AGENTS)

    def get_next(self) -> str:
        """Get the next user agent in rotation"""
        user_agent = self.USER_AGENTS[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.USER_AGENTS)
        return user_agent


class RateLimiter:
    """Rate limiter to avoid overwhelming APIs"""

    def __init__(self, min_interval: float = 0.5):
        self.min_interval = min_interval
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class DataFetcher(ABC):
    """Abstract base class for data fetchers"""

    def __init__(self, config: FetchConfig = None):
        self.config = config or FetchConfig()
        self.user_agent_rotator = UserAgentRotator()
        self.rate_limiter = RateLimiter(self.config.rate_limit_delay)
        self.session = requests.Session()

        # Set up session with retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @abstractmethod
    def fetch_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data for a given symbol"""
        pass

    def get_headers(self) -> Dict[str, str]:
        """Get headers with rotated user agent"""
        return {
            'User-Agent': self.user_agent_rotator.get_random(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }


class YahooFinanceFetcher(DataFetcher):
    """
    Robust Yahoo Finance data fetcher with caching and error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting to avoid being blocked
    - Corporate action handling (adjusted prices)
    - Data validation and quality checks
    - Local caching to reduce API calls
    """

    def __init__(self, config: FetchConfig = None, cache_dir: Path = None):
        super().__init__(config)
        self.cache_dir = cache_dir or Path("data/cache/yahoo")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key for given parameters"""
        key_data = f"{symbol}_{period}_{interval}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid based on TTL"""
        if not cache_path.exists():
            return False

        file_age = time.time() - cache_path.stat().st_mtime
        max_age = self.config.cache_ttl_hours * 3600

        return file_age < max_age

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self.config.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            try:
                logger.debug(f"Loading cached data for key: {cache_key}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)

        return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache"""
        if not self.config.use_cache:
            return

        try:
            cache_path = self._get_cache_path(cache_key)
            data.to_parquet(cache_path)
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @retry_with_backoff(FetchConfig())
    def _fetch_from_yahoo(self, symbol: str, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        """Internal method to fetch data from Yahoo Finance with retries"""
        self.rate_limiter.wait_if_needed()

        logger.debug(f"Fetching {symbol} from Yahoo Finance (period={period}, interval={interval})")

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=False,
                actions=True  # Include dividends and splits
            )

            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            # Reset index to make Date a column
            data.reset_index(inplace=True)

            return data

        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")
            raise

    def _validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate fetched data quality"""
        if not self.config.validate_data:
            return True

        try:
            # Basic checks
            if data.empty:
                logger.warning(f"Empty data for {symbol}")
                return False

            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                return False

            # Check for reasonable price ranges (basic sanity check)
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if (data[col] <= 0).any():
                    logger.warning(f"Invalid prices (<=0) found in {col} for {symbol}")
                    return False

            # Check that High >= Low for each row
            if (data['High'] < data['Low']).any():
                logger.warning(f"High < Low found for {symbol}")
                return False

            # Check volume is non-negative
            if (data['Volume'] < 0).any():
                logger.warning(f"Negative volume found for {symbol}")
                return False

            logger.debug(f"Data validation passed for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Data validation failed for {symbol}: {e}")
            return False

    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d",
                   auto_adjust: bool = True, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a symbol from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            auto_adjust: Whether to use adjusted prices (recommended for indicators)
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """

        # Generate cache key
        cache_key = self._get_cache_key(symbol, period, interval)

        # Try to load from cache first (unless force refresh)
        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached data for {symbol}")
                return cached_data

        try:
            # Fetch from Yahoo Finance
            data = self._fetch_from_yahoo(symbol, period, interval, auto_adjust)

            # Validate data quality
            if not self._validate_data(data, symbol):
                logger.error(f"Data validation failed for {symbol}")
                return None

            # Save to cache
            self._save_to_cache(data, cache_key)

            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic info about a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with symbol info or None if failed
        """
        try:
            self.rate_limiter.wait_if_needed()

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'symbol' not in info:
                return None

            # Extract key information
            return {
                'symbol': info.get('symbol'),
                'shortName': info.get('shortName'),
                'longName': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'marketCap': info.get('marketCap'),
                'currency': info.get('currency'),
                'exchange': info.get('exchange'),
                'country': info.get('country'),
                'timeZone': info.get('timeZone')
            }

        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            return None


class NSEFetcher(DataFetcher):
    """
    Robust NSE data fetcher for symbol lists and basic market data.

    Features:
    - Multiple fallback URLs for NSE data
    - Retry logic and error handling
    - Data validation and cleansing
    - Caching to reduce load on NSE servers
    """

    def __init__(self, config: FetchConfig = None, cache_dir: Path = None):
        super().__init__(config)
        self.cache_dir = cache_dir or Path("data/cache/nse")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # NSE data sources with fallbacks
        self.symbol_urls = [
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        ]

    def _get_cache_path(self, data_type: str) -> Path:
        """Get cache file path for NSE data"""
        return self.cache_dir / f"nse_{data_type}.csv"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if NSE cache is valid (NSE data updates daily)"""
        if not cache_path.exists():
            return False

        file_age = time.time() - cache_path.stat().st_mtime
        # NSE data is typically updated daily, so cache for 12 hours
        max_age = 12 * 3600

        return file_age < max_age

    @retry_with_backoff(FetchConfig())
    def _fetch_symbols_from_url(self, url: str) -> pd.DataFrame:
        """Fetch symbols from a specific NSE URL"""
        self.rate_limiter.wait_if_needed()

        logger.debug(f"Fetching NSE symbols from: {url}")

        headers = self.get_headers()
        # Add NSE-specific headers
        headers.update({
            'Referer': 'https://www.nseindia.com/',
            'Accept': 'text/csv,application/csv'
        })

        response = self.session.get(url, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response from NSE")

        # Parse CSV
        from io import StringIO
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        if df.empty:
            raise ValueError("Empty DataFrame from NSE CSV")

        return df

    def fetch_symbol_list(self, force_refresh: bool = False) -> Optional[List[str]]:
        """
        Fetch list of NSE equity symbols.

        Args:
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            List of symbol strings or None if all sources failed
        """
        cache_path = self._get_cache_path("symbols")

        # Try cache first
        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                logger.debug("Loading NSE symbols from cache")
                df = pd.read_csv(cache_path)
                return df['SYMBOL'].tolist()
            except Exception as e:
                logger.warning(f"Failed to load NSE symbols from cache: {e}")

        # Try each URL until one works
        for i, url in enumerate(self.symbol_urls):
            try:
                logger.info(f"Attempting to fetch NSE symbols from source {i+1}/{len(self.symbol_urls)}")
                df = self._fetch_symbols_from_url(url)

                # Different CSV formats, try to find symbol column
                symbol_col = None
                for col in ['SYMBOL', 'Symbol', 'symbol']:
                    if col in df.columns:
                        symbol_col = col
                        break

                if symbol_col is None:
                    logger.warning(f"No symbol column found in {url}. Columns: {list(df.columns)}")
                    continue

                # Extract and clean symbols
                symbols = df[symbol_col].dropna().astype(str).tolist()
                symbols = [symbol.strip() for symbol in symbols if symbol.strip()]

                if not symbols:
                    logger.warning(f"No symbols extracted from {url}")
                    continue

                # Save to cache
                if self.config.use_cache:
                    try:
                        df[[symbol_col]].rename(columns={symbol_col: 'SYMBOL'}).to_csv(cache_path, index=False)
                        logger.debug(f"Saved {len(symbols)} NSE symbols to cache")
                    except Exception as e:
                        logger.warning(f"Failed to save NSE symbols to cache: {e}")

                logger.info(f"Successfully fetched {len(symbols)} NSE symbols")
                return symbols

            except Exception as e:
                logger.warning(f"Failed to fetch NSE symbols from {url}: {e}")
                continue

        logger.error("All NSE symbol sources failed")
        return None

    def fetch_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        NSE doesn't provide direct OHLCV data APIs.
        This method is implemented for interface compatibility.
        Use YahooFinanceFetcher for historical price data.
        """
        logger.warning("NSEFetcher doesn't support OHLCV data. Use YahooFinanceFetcher instead.")
        return None


# Main data manager that coordinates different fetchers
class DataManager:
    """
    Main data manager that coordinates different data sources.

    This is the primary interface for the rest of the application.
    It automatically selects the appropriate fetcher and handles caching.
    """

    def __init__(self, config: FetchConfig = None, cache_dir: Path = None):
        self.config = config or FetchConfig()
        self.cache_dir = cache_dir or Path("data/cache")

        # Initialize fetchers
        self.yahoo_fetcher = YahooFinanceFetcher(self.config, self.cache_dir / "yahoo")
        self.nse_fetcher = NSEFetcher(self.config, self.cache_dir / "nse")

        logger.info("DataManager initialized with robust fetchers")

    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d",
                           auto_adjust: bool = True, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            auto_adjust: Use adjusted prices (recommended)
            force_refresh: Bypass cache

        Returns:
            DataFrame with historical data or None if failed
        """
        return self.yahoo_fetcher.fetch_data(
            symbol=symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            force_refresh=force_refresh
        )

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a symbol"""
        return self.yahoo_fetcher.get_info(symbol)

    def get_nse_symbols(self, force_refresh: bool = False) -> Optional[List[str]]:
        """Get list of NSE equity symbols"""
        return self.nse_fetcher.fetch_symbol_list(force_refresh=force_refresh)

    def is_symbol_valid(self, symbol: str) -> bool:
        """Check if a symbol is valid by attempting to fetch basic info"""
        info = self.get_symbol_info(symbol)
        return info is not None and 'symbol' in info


# Create default instance for easy importing
default_data_manager = DataManager()

# Convenience functions for backward compatibility
def get_stock_data(symbol: str, period: str = "1y", **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function to get stock data"""
    return default_data_manager.get_historical_data(symbol, period, **kwargs)

def get_stock_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get stock info"""
    return default_data_manager.get_symbol_info(symbol)

def get_nse_symbols(**kwargs) -> Optional[List[str]]:
    """Convenience function to get NSE symbols"""
    return default_data_manager.get_nse_symbols(**kwargs)