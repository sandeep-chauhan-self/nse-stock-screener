"""
Robust Data Fetching Module with Retry Logic and Error Handling
Provides resilient data fetching capabilities for the NSE Stock Screener
"""

from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
import random
import time

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import requests
import yfinance as yf

from .logging_config import get_logger, with_retry, operation_context, retry_manager
from .stock_analysis_monitor import monitor


class DataFetchError(Exception):
    """Base exception for data fetching errors"""
    pass


class RateLimitError(DataFetchError):
    """Raised when rate limit is exceeded"""
    pass


class NetworkError(DataFetchError):
    """Raised for network-related issues"""
    pass


class DataValidationError(DataFetchError):
    """Raised when fetched data fails validation"""
    pass


class RobustDataFetcher:
    """
    Robust data fetching with caching, retry logic, and comprehensive error handling
    """

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 cache_expiry_hours: int = 1,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 request_delay: float = 0.5):

        self.logger = get_logger(__name__)

        # Cache configuration
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours

        # Retry configuration
        self.retry_manager = retry_manager
        self.retry_manager.max_retries = max_retries
        self.retry_manager.base_delay = base_delay

        # Rate limiting
        self.request_delay = request_delay
        self.last_request_time = 0

        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]

        self.logger.info(
            "Initialized RobustDataFetcher",
            extra={
                'cache_dir': str(self.cache_dir),
                'cache_expiry_hours': cache_expiry_hours,
                'max_retries': max_retries,
                'request_delay': request_delay
            }
        )

    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate cache file path for given key"""
        # Create a safe filename from the cache key
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False

        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)

        return file_time > expiry_time

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            self.logger.debug(
                f"Loaded data from cache",
                extra={'cache_key': cache_key, 'cache_path': str(cache_path)}
            )

            return data

        except Exception as e:
            self.logger.warning(
                f"Failed to load from cache: {e}",
                extra={'cache_key': cache_key, 'error': str(e)}
            )
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.debug(
                f"Saved data to cache",
                extra={'cache_key': cache_key, 'cache_path': str(cache_path)}
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to save to cache: {e}",
                extra={'cache_key': cache_key, 'error': str(e)}
            )

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _validate_stock_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate fetched stock data"""
        if data is None or data.empty:
            raise DataValidationError(f"No data returned for {symbol}")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise DataValidationError(f"Missing columns for {symbol}: {missing_columns}")

        # Check for reasonable data ranges
        if len(data) < 5:
            raise DataValidationError(f"Insufficient data points for {symbol}: {len(data)}")

        # Check for data quality issues
        if data['Close'].isna().all():
            raise DataValidationError(f"All close prices are NaN for {symbol}")

        # Check for price sanity (assuming Indian stocks)
        max_price = data[['Open', 'High', 'Low', 'Close']].max().max()
        if max_price > 100000:  # Unreasonably high price
            self.logger.warning(
                f"Unusually high price detected for {symbol}: {max_price}",
                extra={'symbol': symbol, 'max_price': max_price}
            )

        return True

    def _handle_yfinance_error(self, e: Exception, symbol: str) -> Exception:
        """Convert yfinance errors to our custom exceptions"""
        error_str = str(e).lower()

        if any(phrase in error_str for phrase in ['rate limit', 'too many requests', '429']):
            monitor.record_network_error('rate_limit', {'symbol': symbol, 'original_error': str(e)})
            return RateLimitError(f"Rate limited while fetching {symbol}: {e}")

        elif any(phrase in error_str for phrase in ['connection', 'network', 'timeout', 'unreachable']):
            monitor.record_network_error('network_error', {'symbol': symbol, 'original_error': str(e)})
            return NetworkError(f"Network error while fetching {symbol}: {e}")

        else:
            return DataFetchError(f"Data fetch error for {symbol}: {e}")

    @with_retry("fetch_stock_data", max_retries=3)
    def _fetch_yfinance_data(self, symbol: str, period: str = "1y", **kwargs) -> pd.DataFrame:
        """Fetch data from yfinance with error handling"""

        # Rate limiting
        self._rate_limit()

        try:
            ticker = yf.Ticker(symbol)

            # Use auto_adjust=True for corporate action handling (as fixed in Requirement 3.9)
            data = ticker.history(period=period, auto_adjust=True, **kwargs)

            if data.empty:
                raise DataFetchError(f"No data returned by yfinance for {symbol}")

            # Validate the data
            self._validate_stock_data(data, symbol)

            return data

        except Exception as e:
            # Convert to our exception types
            raise self._handle_yfinance_error(e, symbol)

    def fetch_stock_data(self,
                        symbol: str,
                        period: str = "1y",
                        use_cache: bool = True,
                        **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with caching and comprehensive error handling

        Args:
            symbol: Stock symbol to fetch
            period: Time period for data (1y, 6mo, 3mo, etc.)
            use_cache: Whether to use cached data
            **kwargs: Additional arguments for yfinance

        Returns:
            DataFrame with stock data or None if failed
        """

        # Generate cache key
        cache_key = f"{symbol}_{period}_{hash(str(sorted(kwargs.items())))}"

        with operation_context("fetch_stock_data", symbol=symbol, period=period):
            start_time = time.time()

            try:
                # Try cache first
                if use_cache:
                    cached_data = self._load_from_cache(cache_key)
                    if cached_data is not None:
                        duration = time.time() - start_time
                        monitor.record_data_fetch(symbol, duration, True)

                        self.logger.info(
                            f"Using cached data for {symbol}",
                            extra={'symbol': symbol, 'duration': duration, 'source': 'cache'}
                        )

                        return cached_data

                # Fetch fresh data
                self.logger.debug(f"Fetching fresh data for {symbol}", extra={'symbol': symbol})

                data = self._fetch_yfinance_data(symbol, period, **kwargs)

                # Cache the data
                if use_cache:
                    self._save_to_cache(cache_key, data)

                duration = time.time() - start_time
                monitor.record_data_fetch(symbol, duration, True)

                self.logger.info(
                    f"Successfully fetched data for {symbol}",
                    extra={
                        'symbol': symbol,
                        'duration': duration,
                        'data_points': len(data),
                        'source': 'yfinance'
                    }
                )

                return data

            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__

                monitor.record_data_fetch(symbol, duration, False, error_type)

                self.logger.error(
                    f"Failed to fetch data for {symbol}",
                    extra={
                        'symbol': symbol,
                        'duration': duration,
                        'error_type': error_type,
                        'error_message': str(e)
                    },
                    exc_info=True
                )

                return None

    def fetch_multiple_stocks(self,
                             symbols: List[str],
                             period: str = "1y",
                             continue_on_error: bool = True,
                             **kwargs) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple stocks with error isolation

        Args:
            symbols: List of stock symbols
            period: Time period for data
            continue_on_error: Continue processing even if some symbols fail
            **kwargs: Additional arguments for yfinance

        Returns:
            Dictionary mapping symbols to their data (or None if failed)
        """

        results = {}
        failed_symbols = []

        with operation_context("fetch_multiple_stocks", symbol_count=len(symbols)):
            self.logger.info(
                f"Starting batch data fetch for {len(symbols)} symbols",
                extra={'symbol_count': len(symbols), 'period': period}
            )

            for i, symbol in enumerate(symbols, 1):
                try:
                    data = self.fetch_stock_data(symbol, period, **kwargs)
                    results[symbol] = data

                    if data is None:
                        failed_symbols.append(symbol)

                    # Log progress
                    if i % 10 == 0 or i == len(symbols):
                        success_count = sum(1 for v in results.values() if v is not None)
                        self.logger.info(
                            f"Batch progress: {i}/{len(symbols)} symbols processed",
                            extra={
                                'processed': i,
                                'total': len(symbols),
                                'successful': success_count,
                                'failed': len(failed_symbols)
                            }
                        )

                except Exception as e:
                    failed_symbols.append(symbol)

                    if continue_on_error:
                        self.logger.error(
                            f"Failed to fetch {symbol}, continuing with batch",
                            extra={'symbol': symbol, 'error': str(e)},
                            exc_info=True
                        )
                        results[symbol] = None
                    else:
                        self.logger.error(
                            f"Failed to fetch {symbol}, aborting batch",
                            extra={'symbol': symbol, 'error': str(e)},
                            exc_info=True
                        )
                        raise

            success_count = sum(1 for v in results.values() if v is not None)
            success_rate = success_count / len(symbols) if symbols else 0

            self.logger.info(
                f"Batch data fetch completed",
                extra={
                    'total_symbols': len(symbols),
                    'successful': success_count,
                    'failed': len(failed_symbols),
                    'success_rate': success_rate,
                    'failed_symbols': failed_symbols[:10]  # Log first 10 failed symbols
                }
            )

            return results

    @with_retry("fetch_nse_symbols", max_retries=2)
    def fetch_nse_symbol_list(self) -> List[str]:
        """Fetch NSE symbol list with retry logic"""

        with operation_context("fetch_nse_symbols"):
            try:
                # Use rotating user agent
                headers = {'User-Agent': random.choice(self.user_agents)}

                url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                # Parse CSV content
                lines = response.text.strip().split('\n')
                symbols = []

                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 1:
                        symbol = parts[0].strip().strip('"')
                        if symbol and symbol != 'SYMBOL':
                            symbols.append(symbol)

                self.logger.info(
                    f"Successfully fetched NSE symbol list",
                    extra={'symbol_count': len(symbols)}
                )

                return symbols

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch NSE symbol list: {e}",
                    exc_info=True
                )
                raise

    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cached data"""

        with operation_context("clear_cache"):
            if not self.cache_dir.exists():
                return

            files_removed = 0
            cutoff_time = None

            if older_than_hours:
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

            for cache_file in self.cache_dir.glob("*.pkl"):
                should_remove = True

                if cutoff_time:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    should_remove = file_time < cutoff_time

                if should_remove:
                    try:
                        cache_file.unlink()
                        files_removed += 1
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove cache file {cache_file}: {e}"
                        )

            self.logger.info(
                f"Cache cleanup completed",
                extra={
                    'files_removed': files_removed,
                    'older_than_hours': older_than_hours
                }
            )


# Global instance for easy access
data_fetcher = RobustDataFetcher()


# Convenience functions
def fetch_stock_data(symbol: str, period: str = "1y", **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function for fetching single stock data"""
    return data_fetcher.fetch_stock_data(symbol, period, **kwargs)


def fetch_multiple_stocks(symbols: List[str], period: str = "1y", **kwargs) -> Dict[str, Optional[pd.DataFrame]]:
    """Convenience function for fetching multiple stock data"""
    return data_fetcher.fetch_multiple_stocks(symbols, period, **kwargs)


def get_nse_symbols() -> List[str]:
    """Convenience function for fetching NSE symbol list"""
    return data_fetcher.fetch_nse_symbol_list()


if __name__ == "__main__":
    # Test the data fetcher
    from .logging_config import setup_logging

    setup_logging(level="INFO", console_output=True)

    # Test single stock fetch
    print("Testing single stock fetch...")
    data = fetch_stock_data("RELIANCE", period="3mo")
    if data is not None:
        print(f"Successfully fetched {len(data)} data points for RELIANCE")
    else:
        print("Failed to fetch RELIANCE data")

    # Test multiple stock fetch
    print("\nTesting multiple stock fetch...")
    symbols = ["TCS", "INFY", "HDFC", "INVALIDSTOCK"]
    results = fetch_multiple_stocks(symbols, period="1mo")

    print(f"Fetch results:")
    for symbol, data in results.items():
        status = "SUCCESS" if data is not None else "FAILED"
        data_points = len(data) if data is not None else 0
        print(f"  {symbol}: {status} ({data_points} points)")

    # Test NSE symbols (commented out to avoid rate limiting in demo)
    # print("\nTesting NSE symbol list fetch...")
    # symbols = get_nse_symbols()
    # print(f"Fetched {len(symbols)} NSE symbols")