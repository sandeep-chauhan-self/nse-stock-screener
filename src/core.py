"""
Core Functionality for NSE Stock Screener
Common functions and utilities used across the project.
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Import centralized constants with dual import strategy
try:
    # Try importing as module (when run from project root)
    from src.constants import (
        MarketRegime, TRADING_CONSTANTS, DATA_QUALITY_CONSTANTS,
        FILE_CONSTANTS, ERROR_MESSAGES, SUCCESS_MESSAGES, DISPLAY_CONSTANTS,
        PROJECT_ROOT_PATH
    )
except ImportError:
    # Fallback to direct imports (when run as script from src directory)
    from constants import (
        MarketRegime, TRADING_CONSTANTS, DATA_QUALITY_CONSTANTS,
        FILE_CONSTANTS, ERROR_MESSAGES, SUCCESS_MESSAGES, DISPLAY_CONSTANTS,
        PROJECT_ROOT_PATH
    )

class DataFetcher:
    """Core data fetching functionality"""
    
    @staticmethod
    def fetch_stock_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Data period ('1y', '6mo', '3mo', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"⚠️ No data available for {symbol}")
                return None
                
            # Basic data quality check
            if len(data) < DATA_QUALITY_CONSTANTS['MIN_DATA_POINTS_GENERAL']:
                print(f"⚠️ Insufficient data for {symbol}: {len(data)} points")
                return None
                
            # Check for excessive missing data
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > DATA_QUALITY_CONSTANTS['MAX_MISSING_DATA_PCT']:
                print(f"⚠️ Too much missing data for {symbol}: {missing_pct:.1%}")
                return None
                
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    @staticmethod
    def fetch_nifty_data(period: str = "3mo") -> Optional[pd.DataFrame]:
        """Fetch NIFTY index data for market regime detection"""
        return DataFetcher.fetch_stock_data(TRADING_CONSTANTS['NIFTY_SYMBOL'], period)

class StockLoader:
    """Core stock symbol loading functionality"""
    
    @staticmethod
    def ensure_ns_suffix(symbol: str) -> str:
        """Ensure stock symbol has .NS suffix"""
        if not symbol.endswith(TRADING_CONSTANTS['NSE_SUFFIX']):
            return f"{symbol}{TRADING_CONSTANTS['NSE_SUFFIX']}"
        return symbol
    
    @staticmethod
    def load_from_file(file_path: str) -> List[str]:
        """Load stock symbols from a text file"""
        try:
            if not os.path.exists(file_path):
                print(f"⚠️ {ERROR_MESSAGES['FILE_NOT_FOUND']}: {file_path}")
                return []
                
            with open(file_path, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
                
            # Ensure all stocks have .NS suffix
            stocks = [StockLoader.ensure_ns_suffix(s) for s in stocks]
            
            print(f"📂 Loaded {len(stocks)} stocks from {file_path}")
            return stocks
            
        except Exception as e:
            print(f"❌ Error loading stocks from file {file_path}: {e}")
            return []
    
    @staticmethod
    def load_custom_stocks(custom_stocks: List[str]) -> List[str]:
        """Process custom stock list"""
        if not custom_stocks:
            return []
            
        # Ensure all custom stocks have .NS suffix
        stocks = [StockLoader.ensure_ns_suffix(s) for s in custom_stocks]
        print(f"📝 Using custom stock list: {len(stocks)} stocks")
        return stocks
    
    @staticmethod
    def load_default_stocks() -> List[str]:
        """Load default NSE stock symbols"""
        # Use centralized path
        default_file = FILE_CONSTANTS['DEFAULT_STOCK_FILE_PATH']
        
        if default_file.exists():
            stocks = StockLoader.load_from_file(str(default_file))
            if stocks:
                return stocks[:FILE_CONSTANTS['DEFAULT_STOCK_LIMIT']]  # Limit for testing
        
        # Fallback to hardcoded list
        print("📋 Using fallback sample stocks")
        return FILE_CONSTANTS['FALLBACK_STOCKS']
    
    @staticmethod
    def load_stocks(custom_stocks: Optional[List[str]] = None, 
                   input_file: Optional[str] = None) -> List[str]:
        """
        Main stock loading function - handles all sources
        
        Args:
            custom_stocks: Custom list of stock symbols
            input_file: Path to file containing stock symbols
            
        Returns:
            List of stock symbols with .NS suffix
        """
        # Priority: custom_stocks > input_file > default
        if custom_stocks:
            return StockLoader.load_custom_stocks(custom_stocks)
            
        if input_file:
            stocks = StockLoader.load_from_file(input_file)
            if stocks:
                return stocks
                
        return StockLoader.load_default_stocks()

class MarketRegimeDetector:
    """Market regime detection functionality"""
    
    @staticmethod
    def detect_regime() -> MarketRegime:
        """
        Detect current market regime using NIFTY data
        
        Returns:
            MarketRegime enum value
        """
        try:
            print("🌍 Detecting market regime...")
            
            # Fetch NIFTY data
            data = DataFetcher.fetch_nifty_data()
            
            if data is None or data.empty:
                print("⚠️ Could not fetch NIFTY data, using SIDEWAYS regime")
                return MarketRegime.SIDEWAYS
            
            # Calculate regime indicators
            close = data['Close']
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trend (20-day vs 50-day MA)
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # Recent momentum (last 10 days)
            recent_return = close.pct_change(10).iloc[-1]
            
            # Determine regime based on multiple factors
            if volatility > DATA_QUALITY_CONSTANTS['VOLATILITY_THRESHOLD']:
                regime = MarketRegime.HIGH_VOLATILITY
            elif current_price > ma20 > ma50 and recent_return > 0.03:  # 3% gain in 10 days
                regime = MarketRegime.BULLISH
            elif current_price < ma20 < ma50 and recent_return < -0.03:  # 3% loss in 10 days
                regime = MarketRegime.BEARISH
            else:
                regime = MarketRegime.SIDEWAYS
            
            print(f"📈 Market regime detected: {regime.value.upper()}")
            print(f"📊 NIFTY volatility: {volatility:.1%}")
            print(f"📊 Recent return (10d): {recent_return:.1%}")
            
            return regime
            
        except Exception as e:
            print(f"⚠️ Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame, min_points: Optional[int] = None) -> bool:
        """Validate OHLCV data quality"""
        if data is None or data.empty:
            return False
            
        # Check minimum data points
        if min_points and len(data) < min_points:
            return False
            
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # Check for reasonable price relationships
        if not (data['High'] >= data['Close']).all():
            return False
        if not (data['Low'] <= data['Close']).all():
            return False
        if not (data['High'] >= data['Low']).all():
            return False
            
        # Check for positive volume
        if not (data['Volume'] >= 0).all():
            return False
            
        return True
    
    @staticmethod
    def validate_symbol_format(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
            
        # Basic format check
        if len(symbol) < 3:
            return False
            
        # Should contain only alphanumeric characters, dots, and hyphens
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if not set(symbol.upper()).issubset(allowed_chars):
            return False
            
        return True

class PathManager:
    """Path and directory management utilities"""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory"""
        return PROJECT_ROOT_PATH
    
    @staticmethod
    def setup_output_directories() -> Dict[str, str]:
        """Setup output directories for reports, charts, and backtests"""        
        output_dirs = {}
        for dir_name in FILE_CONSTANTS['OUTPUT_DIRS']:
            dir_path = FILE_CONSTANTS['OUTPUT_DIR'] / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            output_dirs[dir_name] = str(dir_path)
        
        return output_dirs
    
    @staticmethod
    def get_data_file_path(filename: str) -> Path:
        """Get path to data file"""
        return FILE_CONSTANTS['DATA_DIR'] / filename

class RateLimiter:
    """Rate limiting utilities for API calls"""
    
    def __init__(self, delay: float = TRADING_CONSTANTS['RATE_LIMIT_DELAY']):
        self.delay = delay
        self.last_call = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_call = time.time()

class PerformanceUtils:
    """Performance optimization utilities"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
        """Safe division that handles zero denominator"""
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        return numerator / denominator
    
    @staticmethod
    def safe_log(value: float, default: float = np.nan) -> float:
        """Safe logarithm that handles zero and negative values"""
        if value <= 0 or np.isnan(value):
            return default
        return np.log(value)
    
    @staticmethod
    def handle_nan_values(data: pd.Series, method: str = 'forward') -> pd.Series:
        """Handle NaN values in pandas Series"""
        if method == 'forward':
            return data.ffill()
        elif method == 'backward':
            return data.bfill()
        elif method == 'drop':
            return data.dropna()
        elif method == 'zero':
            return data.fillna(0)
        else:
            return data
    
    @staticmethod
    def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score for a series"""
        try:
            rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
            rolling_std = series.rolling(window=window, min_periods=window//2).std()
            
            # Avoid division by zero
            z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            return z_score
        except Exception:
            return pd.Series(np.nan, index=series.index)

class DisplayUtils:
    """Display and formatting utilities"""
    
    @staticmethod
    def format_currency(amount: float) -> str:
        """Format amount as currency"""
        if np.isnan(amount):
            return "N/A"
        return f"{DISPLAY_CONSTANTS['CURRENCY_SYMBOL']}{amount:.1f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format value as percentage"""
        if np.isnan(value):
            return "N/A"
        return DISPLAY_CONSTANTS['PERCENTAGE_FORMAT'].format(value)
    
    @staticmethod
    def format_ratio(value: float) -> str:
        """Format value as ratio"""
        if np.isnan(value):
            return "N/A"
        return DISPLAY_CONSTANTS['RATIO_FORMAT'].format(value)
    
    @staticmethod
    def get_timestamp(format_type: str = 'display') -> str:
        """Get formatted timestamp"""
        now = datetime.now()
        if format_type == 'file':
            return now.strftime(DISPLAY_CONSTANTS['FILE_DATE_FORMAT'])
        else:
            return now.strftime(DISPLAY_CONSTANTS['DATE_FORMAT'])
    
    @staticmethod
    def print_progress(current: int, total: int, symbol: str = "") -> None:
        """Print progress information"""
        progress = (current / total) * 100
        print(f"[{progress:.1f}%] Processing {symbol}...")

def get_weekly_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch weekly data for a symbol
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Weekly OHLCV data or None if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch more data and resample to weekly
        data = ticker.history(period="2y")
        if data.empty:
            return None
            
        # Resample to weekly data (Friday close)
        weekly_data = data.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return weekly_data
        
    except Exception as e:
        print(f"Error fetching weekly data for {symbol}: {e}")
        return None

def calculate_relative_strength(symbol: str, benchmark_symbol: Optional[str] = None) -> float:
    """
    Calculate relative strength vs benchmark (default: NIFTY)
    
    Args:
        symbol: Stock symbol
        benchmark_symbol: Benchmark symbol (default: NIFTY)
        
    Returns:
        Relative strength ratio or NaN if calculation fails
    """
    try:
        benchmark_sym = benchmark_symbol if benchmark_symbol is not None else TRADING_CONSTANTS['NIFTY_SYMBOL']
            
        # Fetch data for both symbol and benchmark
        stock_data = DataFetcher.fetch_stock_data(symbol, period="3mo")
        benchmark_data = DataFetcher.fetch_stock_data(benchmark_sym, period="3mo")
        
        if stock_data is None or benchmark_data is None:
            return np.nan
            
        # Calculate 20-day returns
        stock_return = stock_data['Close'].pct_change(20).iloc[-1]
        benchmark_return = benchmark_data['Close'].pct_change(20).iloc[-1]
        
        if np.isnan(stock_return) or np.isnan(benchmark_return):
            return np.nan
            
        # Relative strength = (1 + stock_return) / (1 + benchmark_return)
        relative_strength = (1 + stock_return) / (1 + benchmark_return)
        
        return relative_strength
        
    except Exception as e:
        print(f"Error calculating relative strength for {symbol}: {e}")
        return np.nan