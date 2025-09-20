"""
Enhanced test fixtures for NSE Stock Screener testing.
This module provides comprehensive fixtures and utilities for testing.
"""

import json
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import tempfile
import shutil


class TestDataManager:
    """Manages test data and fixtures for consistent testing."""
    
    def __init__(self, test_data_dir: Optional[str] = None):
        """Initialize test data manager."""
        self.test_data_dir = Path(test_data_dir or "data/test")
        self.temp_dirs = []
    
    def get_test_stock_data(self, symbol: str, period: str = "full") -> pd.DataFrame:
        """Get test stock data for a symbol."""
        if period == "quick":
            file_path = self.test_data_dir / f"{symbol}_quick_test.csv"
        else:
            file_path = self.test_data_dir / f"{symbol}_test_data.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test data not found: {file_path}")
        
        return pd.read_csv(file_path, index_col="Date", parse_dates=True)
    
    def get_test_symbols(self, count: Optional[int] = None) -> List[str]:
        """Get list of test symbols."""
        symbols_file = self.test_data_dir / "test_symbols.txt"
        
        if not symbols_file.exists():
            # Fallback to default test symbols
            symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        else:
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
        
        if count:
            return symbols[:count]
        return symbols
    
    def get_test_config(self, config_name: str = "quick_test") -> Dict[str, Any]:
        """Get test configuration."""
        config_file = self.test_data_dir / "configs" / f"{config_name}.json"
        
        if not config_file.exists():
            # Return default test config
            return {
                "risk_tolerance": "medium",
                "max_position_size": 0.02,
                "stop_loss_percent": 0.03,
                "take_profit_percent": 0.08,
                "technical_indicators": {
                    "rsi_period": 7,
                    "rsi_oversold": 35,
                    "rsi_overbought": 65,
                    "sma_short": 5,
                    "sma_long": 15,
                    "volume_threshold": 1.2
                }
            }
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def get_test_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Get test scenario configuration."""
        scenario_file = self.test_data_dir / "scenarios" / f"{scenario_name}.json"
        
        if not scenario_file.exists():
            raise FileNotFoundError(f"Test scenario not found: {scenario_file}")
        
        with open(scenario_file, 'r') as f:
            return json.load(f)
    
    def create_temp_output_dir(self) -> Path:
        """Create a temporary output directory for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nse_test_"))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()


@pytest.fixture(scope="session")
def test_data_manager():
    """Session-scoped test data manager."""
    manager = TestDataManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Generate realistic test data
    rng = pd.np.random.default_rng(42)
    base_price = 1000
    returns = rng.normal(0.001, 0.02, len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        volatility = 0.015
        high = close * (1 + rng.uniform(0, volatility))
        low = close * (1 - rng.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else close
        volume = int(1000000 * (1 + abs(returns[i]) * 10))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(max(high, open_price, close), 2),
            'Low': round(min(low, open_price, close), 2),
            'Close': round(close, 2),
            'Volume': volume,
            'Adj Close': round(close, 2)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


@pytest.fixture
def mock_yfinance_data(sample_stock_data):
    """Mock yfinance data for testing."""
    def mock_download(*args, **kwargs):
        # Return sample data for any symbol requested
        return sample_stock_data.copy()
    
    with patch('yfinance.download', side_effect=mock_download):
        yield mock_download


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "risk_tolerance": "medium",
        "max_position_size": 0.05,
        "stop_loss_percent": 0.03,
        "take_profit_percent": 0.08,
        "technical_indicators": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_short": 10,
            "sma_long": 20,
            "volume_threshold": 1.5
        },
        "filters": {
            "min_price": 10,
            "max_price": 10000,
            "min_volume": 100000
        }
    }


@pytest.fixture
def test_symbols():
    """Provide test symbols list."""
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]


@pytest.fixture
def temp_output_dir():
    """Provide temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="nse_test_output_"))
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_file_system(temp_output_dir):
    """Mock file system operations."""
    original_paths = {}
    
    def setup_mock_paths():
        # Mock common paths used in the application
        original_paths['output'] = Path("output")
        original_paths['data'] = Path("data")
        original_paths['logs'] = Path("logs")
        
        # Create mock directory structure
        (temp_output_dir / "reports").mkdir(parents=True)
        (temp_output_dir / "charts").mkdir(parents=True)
        (temp_output_dir / "backtests").mkdir(parents=True)
        (temp_output_dir / "data").mkdir(parents=True)
        (temp_output_dir / "logs").mkdir(parents=True)
    
    setup_mock_paths()
    
    with patch.dict('os.environ', {
        'NSE_SCREENER_OUTPUT_PATH': str(temp_output_dir),
        'NSE_SCREENER_DATA_PATH': str(temp_output_dir / "data"),
        'NSE_SCREENER_LOG_PATH': str(temp_output_dir / "logs")
    }):
        yield temp_output_dir


@pytest.fixture
def mock_redis():
    """Mock Redis for caching tests."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
            return True
        
        def delete(self, key):
            return self.data.pop(key, None) is not None
        
        def exists(self, key):
            return key in self.data
        
        def flushdb(self):
            self.data.clear()
            return True
    
    mock_redis = MockRedis()
    
    with patch('redis.Redis', return_value=mock_redis):
        yield mock_redis


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for tests."""
    import time
    import psutil
    import tracemalloc
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            tracemalloc.start()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
        
        def stop(self):
            if self.start_time is None:
                return {}
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            current, peak = tracemalloc.get_traced_memory()
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_used': end_memory - self.start_memory,
                'peak_memory': peak,
                'current_memory': current
            }
        
        def assert_performance(self, max_time=30, max_memory_mb=500):
            """Assert performance constraints."""
            metrics = self.stop()
            
            if metrics.get('execution_time', 0) > max_time:
                pytest.fail(f"Test took too long: {metrics['execution_time']:.2f}s > {max_time}s")
            
            memory_mb = metrics.get('memory_used', 0) / (1024 * 1024)
            if memory_mb > max_memory_mb:
                pytest.fail(f"Test used too much memory: {memory_mb:.2f}MB > {max_memory_mb}MB")
    
    monitor = PerformanceMonitor()
    yield monitor
    tracemalloc.stop()


@pytest.fixture
def market_data_mocker():
    """Mock market data for different scenarios."""
    class MarketDataMocker:
        def __init__(self):
            self.scenarios = {
                'bullish': {'trend': 'up', 'volatility': 0.01, 'volume_multiplier': 1.5},
                'bearish': {'trend': 'down', 'volatility': 0.02, 'volume_multiplier': 1.2},
                'sideways': {'trend': 'flat', 'volatility': 0.015, 'volume_multiplier': 1.0},
                'volatile': {'trend': 'up', 'volatility': 0.05, 'volume_multiplier': 2.0}
            }
        
        def generate_scenario_data(self, scenario: str, days: int = 30) -> pd.DataFrame:
            """Generate data for specific market scenario."""
            if scenario not in self.scenarios:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            config = self.scenarios[scenario]
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            dates = dates[dates.weekday < 5]
            
            rng = pd.np.random.default_rng(42)
            base_price = 1000
            
            # Generate prices based on scenario
            if config['trend'] == 'up':
                trend_return = 0.01
            elif config['trend'] == 'down':
                trend_return = -0.01
            else:
                trend_return = 0.0
            
            prices = [base_price]
            for i in range(1, len(dates)):
                daily_return = trend_return + rng.normal(0, config['volatility'])
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 0.01))
            
            # Generate OHLCV data
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = prices[i-1] if i > 0 else close
                high = close * (1 + rng.uniform(0, config['volatility']))
                low = close * (1 - rng.uniform(0, config['volatility']))
                volume = int(1000000 * config['volume_multiplier'] * (1 + rng.uniform(0, 0.5)))
                
                data.append({
                    'Date': date,
                    'Open': round(open_price, 2),
                    'High': round(max(high, open_price, close), 2),
                    'Low': round(min(low, open_price, close), 2),
                    'Close': round(close, 2),
                    'Volume': volume,
                    'Adj Close': round(close, 2)
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            return df
    
    return MarketDataMocker()


# Integration test utilities
def assert_analysis_results(results: Dict[str, Any], min_stocks: int = 1):
    """Assert analysis results have expected structure."""
    assert 'analysis_timestamp' in results
    assert 'market_overview' in results
    assert 'top_picks' in results
    assert isinstance(results['top_picks'], list)
    assert len(results['top_picks']) >= min_stocks
    
    for pick in results['top_picks']:
        assert 'symbol' in pick
        assert 'score' in pick
        assert 'signal' in pick
        assert pick['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= pick['score'] <= 100


def assert_backtest_results(results: Dict[str, Any]):
    """Assert backtest results have expected structure."""
    required_fields = [
        'total_return', 'annualized_return', 'volatility',
        'sharpe_ratio', 'max_drawdown', 'total_trades'
    ]
    
    for field in required_fields:
        assert field in results, f"Missing required field: {field}"
        assert isinstance(results[field], (int, float)), f"Field {field} should be numeric"


def create_mock_environment():
    """Create a complete mock environment for testing."""
    return {
        'NSE_SCREENER_ENV': 'test',
        'NSE_SCREENER_CONFIG_PATH': 'data/test/configs',
        'NSE_SCREENER_DATA_PATH': 'data/test',
        'NSE_SCREENER_OUTPUT_PATH': 'test-output',
        'LOG_LEVEL': 'DEBUG',
        'ENABLE_CACHE': 'false',
        'MOCK_API_CALLS': 'true'
    }