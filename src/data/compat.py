"""
Enhanced data ingestion compatibility layer.

Provides drop-in replacements for yfinance and requests calls with robust error handling,
caching, and monitoring. Designed to be a seamless upgrade for existing code.

Usage:
    Instead of:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
    
    Use:
        from data.compat import enhanced_yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
"""

import logging
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

# Import our robust data infrastructure
from .fetchers import DataManager, FetchConfig, default_data_manager
from .validation import validate_stock_data, adjust_for_corporate_actions
from .cache import default_cache

logger = logging.getLogger(__name__)


class EnhancedTicker:
    """
    Enhanced wrapper around yfinance.Ticker with robust data fetching.
    
    Provides the same interface as yfinance.Ticker but with:
    - Automatic retry logic and exponential backoff
    - Local caching to reduce API calls
    - Data validation and quality checks
    - Corporate action handling
    - Better error handling and logging
    """
    
    def __init__(self, symbol: str, session=None):
        self.symbol = symbol
        self.session = session
        self._info = None
        self._data_manager = default_data_manager
        
        # Track when we last fetched info to avoid repeated calls
        self._info_last_fetched = None
    
    def history(self, period: str = "1y", interval: str = "1d", 
                start: str = None, end: str = None, prepost: bool = True,
                auto_adjust: bool = True, back_adjust: bool = False,
                proxy: str = None, rounding: bool = False,
                tz: str = None, timeout: int = None, **kwargs) -> pd.DataFrame:
        """
        Enhanced version of yfinance history() with robust fetching and caching.
        
        Args:
            period: Period to download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            auto_adjust: Automatically adjust OHLC prices for dividends and splits
            **kwargs: Additional arguments (mostly ignored for compatibility)
            
        Returns:
            DataFrame with historical data
        """
        
        try:
            # Use our robust data manager
            data = self._data_manager.get_historical_data(
                symbol=self.symbol,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                force_refresh=kwargs.get('force_refresh', False)
            )
            
            if data is None:
                logger.warning(f"Failed to fetch data for {self.symbol}, returning empty DataFrame")
                return pd.DataFrame()
            
            # Validate data quality
            validation_result = validate_stock_data(data, self.symbol)
            if not validation_result.is_valid:
                logger.warning(f"Data validation failed for {self.symbol}: {validation_result.errors}")
                # Continue with data but log warnings
                for warning in validation_result.warnings:
                    logger.debug(f"Data warning for {self.symbol}: {warning}")
            
            # Apply corporate action adjustments if requested
            if auto_adjust:
                data = adjust_for_corporate_actions(data)
            
            # Set index to Date if it's a column
            if 'Date' in data.columns:
                data.set_index('Date', inplace=True)
            
            logger.debug(f"Successfully fetched {len(data)} rows for {self.symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Enhanced history fetch failed for {self.symbol}: {e}")
            
            # Fallback to original yfinance as last resort
            try:
                logger.debug(f"Attempting fallback to original yfinance for {self.symbol}")
                original_ticker = yf.Ticker(self.symbol, session=self.session)
                return original_ticker.history(
                    period=period, interval=interval, start=start, end=end,
                    prepost=prepost, auto_adjust=auto_adjust, back_adjust=back_adjust,
                    proxy=proxy, rounding=rounding, tz=tz, timeout=timeout
                )
            except Exception as fallback_error:
                logger.error(f"Fallback yfinance also failed for {self.symbol}: {fallback_error}")
                return pd.DataFrame()
    
    @property
    def info(self) -> Dict[str, Any]:
        """
        Enhanced version of yfinance info property with caching.
        """
        
        # Check if we have cached info that's not too old
        if (self._info is not None and 
            self._info_last_fetched is not None and
            (datetime.now() - self._info_last_fetched).total_seconds() < 3600):  # 1 hour cache
            return self._info
        
        try:
            # Try to get from our cache first
            cached_info = default_cache.get_symbol_info(self.symbol)
            if cached_info is not None:
                self._info = cached_info
                self._info_last_fetched = datetime.now()
                return self._info
            
            # Fetch using our robust data manager
            info = self._data_manager.get_symbol_info(self.symbol)
            
            if info is not None:
                # Cache for future use
                default_cache.cache_symbol_info(self.symbol, info, ttl_hours=168)  # 1 week
                self._info = info
                self._info_last_fetched = datetime.now()
                return info
            else:
                # Fallback to original yfinance
                logger.debug(f"Attempting fallback to original yfinance info for {self.symbol}")
                original_ticker = yf.Ticker(self.symbol, session=self.session)
                info = original_ticker.info
                
                if info:
                    # Cache the fallback result too
                    default_cache.cache_symbol_info(self.symbol, info, ttl_hours=24)
                    self._info = info
                    self._info_last_fetched = datetime.now()
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get info for {self.symbol}: {e}")
            return {}
    
    def get_info(self) -> Dict[str, Any]:
        """Alternative method name for getting symbol info"""
        return self.info
    
    # Add other yfinance methods for compatibility
    def get_actions(self):
        """Get dividend and split history"""
        try:
            original_ticker = yf.Ticker(self.symbol, session=self.session)
            return original_ticker.actions
        except Exception as e:
            logger.error(f"Failed to get actions for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def get_dividends(self):
        """Get dividend history"""
        try:
            original_ticker = yf.Ticker(self.symbol, session=self.session)
            return original_ticker.dividends
        except Exception as e:
            logger.error(f"Failed to get dividends for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def get_splits(self):
        """Get stock split history"""
        try:
            original_ticker = yf.Ticker(self.symbol, session=self.session)
            return original_ticker.splits
        except Exception as e:
            logger.error(f"Failed to get splits for {self.symbol}: {e}")
            return pd.DataFrame()


class EnhancedYFinance:
    """
    Enhanced yfinance module replacement with robust data handling.
    
    Provides the same interface as the yfinance module but with enhanced reliability.
    """
    
    def __init__(self, config: FetchConfig = None):
        self.config = config or FetchConfig()
        self.data_manager = DataManager(self.config)
    
    def Ticker(self, symbol: str, session=None) -> EnhancedTicker:
        """Create an enhanced ticker object"""
        return EnhancedTicker(symbol, session)
    
    def download(self, tickers: Union[str, List[str]], period: str = "1y", 
                interval: str = "1d", group_by: str = 'column', 
                auto_adjust: bool = True, prepost: bool = True,
                threads: bool = True, proxy: str = None, **kwargs) -> pd.DataFrame:
        """
        Enhanced version of yfinance download() function.
        
        Note: This is a simplified implementation. For production use,
        consider implementing full multi-ticker download functionality.
        """
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        all_data = {}
        
        for ticker in tickers:
            try:
                ticker_obj = self.Ticker(ticker)
                data = ticker_obj.history(period=period, interval=interval, auto_adjust=auto_adjust)
                
                if not data.empty:
                    all_data[ticker] = data
                else:
                    logger.warning(f"No data downloaded for {ticker}")
                    
            except Exception as e:
                logger.error(f"Failed to download data for {ticker}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        if len(all_data) == 1:
            # Single ticker - return simple DataFrame
            return list(all_data.values())[0]
        else:
            # Multiple tickers - return multi-level DataFrame
            return pd.concat(all_data, axis=1, keys=all_data.keys())


class EnhancedRequests:
    """
    Enhanced requests wrapper with retry logic and rate limiting.
    
    Drop-in replacement for requests with improved reliability for financial data APIs.
    """
    
    def __init__(self, config: FetchConfig = None):
        self.config = config or FetchConfig()
        self.data_manager = DataManager(self.config)
        # Get the underlying session from our data manager
        self.session = self.data_manager.yahoo_fetcher.session
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Enhanced GET request with retry logic"""
        
        # Use the robust session from our fetchers
        headers = kwargs.get('headers', {})
        if not headers.get('User-Agent'):
            headers.update(self.data_manager.yahoo_fetcher.get_headers())
            kwargs['headers'] = headers
        
        # Apply rate limiting
        self.data_manager.yahoo_fetcher.rate_limiter.wait_if_needed()
        
        # Use the session with built-in retry logic
        try:
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Enhanced requests.get failed for {url}: {e}")
            raise
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Enhanced POST request with retry logic"""
        
        headers = kwargs.get('headers', {})
        if not headers.get('User-Agent'):
            headers.update(self.data_manager.yahoo_fetcher.get_headers())
            kwargs['headers'] = headers
        
        self.data_manager.yahoo_fetcher.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.post(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Enhanced requests.post failed for {url}: {e}")
            raise


# Create enhanced instances for easy importing
enhanced_yfinance = EnhancedYFinance()
enhanced_requests = EnhancedRequests()


# Convenience functions for backward compatibility
def get_stock_data(symbol: str, period: str = "1y", **kwargs) -> pd.DataFrame:
    """Convenience function to get stock data using enhanced fetcher"""
    ticker = enhanced_yfinance.Ticker(symbol)
    return ticker.history(period=period, **kwargs)


def get_nse_symbols(force_refresh: bool = False) -> List[str]:
    """Convenience function to get NSE symbols using enhanced fetcher"""
    return default_data_manager.get_nse_symbols(force_refresh=force_refresh) or []


def validate_symbol(symbol: str) -> bool:
    """Check if a symbol is valid using enhanced fetcher"""
    return default_data_manager.is_symbol_valid(symbol)


# Configuration helpers
def configure_data_fetching(max_retries: int = 3, cache_ttl_hours: int = 24,
                           rate_limit_delay: float = 0.5, use_cache: bool = True):
    """Configure the enhanced data fetching behavior"""
    
    config = FetchConfig(
        max_retries=max_retries,
        cache_ttl_hours=cache_ttl_hours,
        rate_limit_delay=rate_limit_delay,
        use_cache=use_cache
    )
    
    # Update the global data manager
    global enhanced_yfinance, enhanced_requests, default_data_manager
    default_data_manager = DataManager(config)
    enhanced_yfinance = EnhancedYFinance(config)
    enhanced_requests = EnhancedRequests(config)
    
    logger.info(f"Data fetching configured: retries={max_retries}, cache_ttl={cache_ttl_hours}h, rate_limit={rate_limit_delay}s")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the data cache"""
    return default_cache.get_cache_stats()


def clear_cache() -> int:
    """Clear all cached data"""
    return default_cache.clear_all()


def cleanup_cache() -> int:
    """Remove expired cache entries"""
    return default_cache.cleanup_expired()


# Initialize with default configuration
logger.info("Enhanced data ingestion compatibility layer loaded")