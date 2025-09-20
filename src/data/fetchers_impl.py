"""
Data fetchers implementing the IDataFetcher interface.

This module provides concrete implementations for fetching market data from
various sources while adhering to the stable interface contract.
"""

import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

from ..common.interfaces import IDataFetcher
from ..common.config import get_config


logger = logging.getLogger(__name__)


class YahooDataFetcher:
    """Yahoo Finance data fetcher implementation."""
    
    def __init__(self):
        """Initialize Yahoo Finance fetcher."""
        self.config = get_config().config.data
        self._last_request_time = 0
    
    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            start: Start date
            end: End date
            **kwargs: Additional parameters (period, interval, auto_adjust)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            # Ensure symbol has .NS suffix for NSE stocks
            if not symbol.endswith('.NS') and symbol != '^NSEI':
                symbol = f"{symbol}.NS"
            
            # Create ticker and fetch data
            ticker = yf.Ticker(symbol)
            
            # Use period if provided, otherwise use dates
            if 'period' in kwargs:
                data = ticker.history(
                    period=kwargs['period'],
                    interval=kwargs.get('interval', '1d'),
                    auto_adjust=kwargs.get('auto_adjust', True),
                    timeout=self.config.request_timeout_seconds
                )
            else:
                data = ticker.history(
                    start=start,
                    end=end,
                    interval=kwargs.get('interval', '1d'),
                    auto_adjust=kwargs.get('auto_adjust', True),
                    timeout=self.config.request_timeout_seconds
                )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Validate minimum data points
            if len(data) < self.config.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.config.min_data_points}")
                return None
            
            # Check for excessive missing data
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > self.config.max_missing_data_pct:
                logger.warning(f"Too much missing data for {symbol}: {missing_pct:.2%}")
                return None
            
            logger.debug(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_symbols(self, exchange: str = "NSE") -> List[str]:
        """
        Fetch list of symbols from NSE website.
        
        Args:
            exchange: Exchange identifier (only NSE supported)
            
        Returns:
            List of symbol strings
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            if exchange.upper() != "NSE":
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            # NSE equity symbols URL
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/'
            }
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.config.request_timeout_seconds
            )
            response.raise_for_status()
            
            data = response.json()
            symbols = [item['symbol'] for item in data.get('data', [])]
            
            logger.info(f"Fetched {len(symbols)} symbols from NSE")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols from {exchange}: {e}")
            # Fallback to local file if available
            return self._load_symbols_from_file()
    
    def _rate_limit(self) -> None:
        """Implement rate limiting based on configuration."""
        if self.config.requests_per_second <= 0:
            return
        
        min_interval = 1.0 / self.config.requests_per_second
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _load_symbols_from_file(self) -> List[str]:
        """Load symbols from local file as fallback."""
        try:
            from pathlib import Path
            symbol_file = Path("data/nse_only_symbols.txt")
            
            if symbol_file.exists():
                with open(symbol_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(symbols)} symbols from local file")
                return symbols
            else:
                logger.warning("No local symbol file found")
                return []
        except Exception as e:
            logger.error(f"Error loading symbols from file: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is valid by attempting to fetch its info.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            # Try to fetch basic info for the symbol
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            if not info or len(info) < 5:  # Basic threshold for valid response
                return False
            
            # Check for error indicators
            if 'symbol' not in info and 'shortName' not in info:
                return False
            
            logger.debug(f"Symbol {symbol} validated successfully")
            return True
            
        except Exception as e:
            logger.debug(f"Symbol {symbol} validation failed: {e}")
            return False


class AlphaVantageDataFetcher:
    """Alpha Vantage data fetcher (alternative implementation)."""
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage fetcher.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.config = get_config().config.data
        self._last_request_time = 0
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Rate limiting (Alpha Vantage has strict limits)
            self._rate_limit()
            
            # Alpha Vantage parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': f"{symbol}.BSE",  # Bombay Stock Exchange format
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.config.request_timeout_seconds
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit hit: {data['Note']}")
                return None
            
            # Parse time series data
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                logger.warning(f"No time series data for {symbol}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adj Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            df = df.loc[start:end]
            
            if len(df) < self.config.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.config.min_data_points}")
                return None
            
            logger.debug(f"Successfully fetched {len(df)} data points for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return None
    
    def fetch_symbols(self, exchange: str = "NSE") -> List[str]:
        """Alpha Vantage doesn't provide symbol lists, return empty list."""
        logger.warning("Alpha Vantage fetcher doesn't support symbol listing")
        return []
    
    def _rate_limit(self) -> None:
        """Alpha Vantage has strict rate limits (5 calls/minute for free tier)."""
        min_interval = 12.0  # 5 calls per minute = 12 seconds between calls
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Alpha Vantage rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()


class DataFetcherFactory:
    """Factory for creating data fetchers based on configuration."""
    
    @staticmethod
    def create_fetcher(source: str = None, **kwargs) -> IDataFetcher:
        """
        Create a data fetcher based on source type.
        
        Args:
            source: Data source type ("yfinance", "alphavantage")
            **kwargs: Additional parameters for the fetcher
            
        Returns:
            Data fetcher implementation
        """
        if source is None:
            source = get_config().config.data.default_data_source
        
        source = source.lower()
        
        if source == "yfinance":
            return YahooDataFetcher()
        elif source == "alphavantage":
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("Alpha Vantage API key required")
            return AlphaVantageDataFetcher(api_key)
        else:
            raise ValueError(f"Unsupported data source: {source}")


# Export public components
__all__ = [
    'YahooDataFetcher',
    'AlphaVantageDataFetcher', 
    'NSEDataFetcher',
    'NSEBhavcopyFetcher',
    'DataFetcherFactory'
]


class NSEDataFetcher:
    """NSE API data fetcher implementation."""
    
    def __init__(self):
        """Initialize NSE API fetcher."""
        self.config = get_config().config.data
        self._last_request_time = 0
        self.session = requests.Session()
        
        # Set headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
    
    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from NSE APIs.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            start: Start date  
            end: End date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            # NSE historical data API (if available)
            url = f"https://www.nseindia.com/api/historical/cm/equity"
            params = {
                'symbol': symbol.upper(),
                'series': '["EQ"]',
                'from': start.strftime('%d-%m-%Y'),
                'to': end.strftime('%d-%m-%Y')
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"NSE API returned status {response.status_code} for {symbol}")
                return None
            
            data = response.json()
            
            if not data.get('data'):
                logger.warning(f"No data returned from NSE API for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Standardize column names
            df = df.rename(columns={
                'CH_TIMESTAMP': 'Date',
                'CH_OPENING_PRICE': 'Open',
                'CH_TRADE_HIGH_PRICE': 'High', 
                'CH_TRADE_LOW_PRICE': 'Low',
                'CH_CLOSING_PRICE': 'Close',
                'CH_TOT_TRADED_QTY': 'Volume'
            })
            
            # Convert date and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Keep only OHLCV columns
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[ohlcv_columns]
            
            # Convert to numeric
            for col in ohlcv_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df.sort_index(inplace=True)
            
            logger.info(f"NSE API: Fetched {len(df)} records for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NSE API request failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing NSE API data for {symbol}: {e}")
            return None
    
    def fetch_symbols(self, exchange: str = "NSE") -> List[str]:
        """Fetch list of NSE symbols."""
        try:
            # Use NSE equity list API
            url = "https://www.nseindia.com/api/equity-stockIndices"
            params = {'index': 'SECURITIES%20IN%20F%26O'}
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data.get('data', [])]
                return symbols
            else:
                logger.warning(f"Failed to fetch NSE symbol list: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching NSE symbols: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists on NSE."""
        try:
            # Simple validation via NSE quote API
            url = f"https://www.nseindia.com/api/quote-equity"
            params = {'symbol': symbol.upper()}
            
            response = self.session.get(url, params=params, timeout=10)
            return response.status_code == 200 and 'data' in response.json()
            
        except Exception:
            return False
    
    def _rate_limit(self) -> None:
        """Implement rate limiting for NSE API."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 0.5  # 500ms between requests
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()


class NSEBhavcopyFetcher:
    """NSE Bhavcopy data fetcher implementation."""
    
    def __init__(self):
        """Initialize NSE Bhavcopy fetcher."""
        self.config = get_config().config.data
        self._last_request_time = 0
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from NSE Bhavcopy files.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            start: Start date
            end: End date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            all_data = []
            
            # Iterate through date range
            current_date = start
            while current_date <= end:
                # Skip weekends
                if current_date.weekday() < 5:  # Monday=0, Friday=4
                    day_data = self._fetch_bhavcopy_for_date(current_date, symbol)
                    if day_data is not None:
                        all_data.append(day_data)
                
                current_date += timedelta(days=1)
            
            if not all_data:
                logger.warning(f"No bhavcopy data found for {symbol}")
                return None
            
            # Combine all daily data
            df = pd.concat(all_data, ignore_index=True)
            
            # Set date as index
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"NSE Bhavcopy: Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching bhavcopy data for {symbol}: {e}")
            return None
    
    def _fetch_bhavcopy_for_date(self, target_date: date, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch bhavcopy data for a specific date."""
        try:
            # Rate limiting
            self._rate_limit()
            
            # Format date for bhavcopy filename
            date_str = target_date.strftime('%d%m%Y')
            
            # NSE bhavcopy URL pattern
            url = f"https://www.nseindia.com/content/historical/EQUITIES/{target_date.year}/cm{date_str}bhav.csv.zip"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                # Try alternative URL format
                month_str = target_date.strftime('%b').upper()
                alt_url = f"https://www.nseindia.com/content/historical/EQUITIES/{target_date.year}/{month_str}/cm{date_str}bhav.csv.zip"
                response = self.session.get(alt_url, timeout=30)
                
                if response.status_code != 200:
                    return None
            
            # Read CSV from zip
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_filename = f"cm{date_str}bhav.csv"
                
                if csv_filename not in zip_file.namelist():
                    return None
                
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
            
            # Filter for the specific symbol
            symbol_data = df[df['SYMBOL'] == symbol.upper()]
            
            if symbol_data.empty:
                return None
            
            # Take the first row (should be only one for equity)
            row = symbol_data.iloc[0]
            
            # Create standardized data
            result = pd.DataFrame({
                'Date': [target_date],
                'Open': [float(row['OPEN'])],
                'High': [float(row['HIGH'])],
                'Low': [float(row['LOW'])],
                'Close': [float(row['CLOSE'])],
                'Volume': [int(row['TOTTRDQTY'])]
            })
            
            result['Date'] = pd.to_datetime(result['Date'])
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to fetch bhavcopy for {symbol} on {target_date}: {e}")
            return None
    
    def fetch_symbols(self, exchange: str = "NSE") -> List[str]:
        """Fetch symbols from latest bhavcopy."""
        try:
            # Get latest trading day bhavcopy
            current_date = date.today()
            
            # Go back up to 5 days to find a trading day
            for i in range(5):
                check_date = current_date - timedelta(days=i)
                if check_date.weekday() < 5:  # Weekday
                    symbols = self._get_symbols_from_bhavcopy(check_date)
                    if symbols:
                        return symbols
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching symbols from bhavcopy: {e}")
            return []
    
    def _get_symbols_from_bhavcopy(self, target_date: date) -> List[str]:
        """Get all symbols from bhavcopy for a date."""
        try:
            date_str = target_date.strftime('%d%m%Y')
            url = f"https://www.nseindia.com/content/historical/EQUITIES/{target_date.year}/cm{date_str}bhav.csv.zip"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return []
            
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_filename = f"cm{date_str}bhav.csv"
                
                if csv_filename not in zip_file.namelist():
                    return []
                
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
            
            # Filter for equity series and return symbols
            equity_df = df[df['SERIES'] == 'EQ']
            return equity_df['SYMBOL'].unique().tolist()
            
        except Exception as e:
            logger.debug(f"Failed to get symbols from bhavcopy for {target_date}: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in NSE."""
        # Simple validation - check if symbol appears in recent bhavcopy
        symbols = self.fetch_symbols()
        return symbol.upper() in symbols
    
    def _rate_limit(self) -> None:
        """Implement rate limiting for NSE downloads."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0  # 1 second between requests
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()