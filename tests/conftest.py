"""
Conftest.py - Pytest configuration and shared fixtures

This file provides common fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Add src to Python path for testing
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Import test data generators
from tests.fixtures.test_data import (
    create_test_ohlcv_data,
    create_expected_indicator_results,
    create_regime_test_data,
    create_backtest_test_data,
    create_risk_management_test_scenarios
)

@pytest.fixture(scope="session")
def test_ohlcv_data():
    """Fixture providing deterministic OHLCV test data."""
    return create_test_ohlcv_data()

@pytest.fixture(scope="session")
def expected_results():
    """Fixture providing expected indicator calculation results."""
    return create_expected_indicator_results()

@pytest.fixture(scope="session")
def regime_test_data():
    """Fixture providing test data for different market regimes."""
    return create_regime_test_data()

@pytest.fixture(scope="session")
def backtest_test_data():
    """Fixture providing test data for backtesting scenarios."""
    return create_backtest_test_data()

@pytest.fixture(scope="session")
def risk_scenarios():
    """Fixture providing risk management test scenarios."""
    return create_risk_management_test_scenarios()

@pytest.fixture
def mock_yfinance():
    """Mock yfinance to return deterministic test data."""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = create_test_ohlcv_data()
        yield mock_instance

@pytest.fixture
def mock_nifty_data():
    """Mock NIFTY data for market regime testing."""
    nifty_data = create_test_ohlcv_data().copy()
    # Modify to represent index behavior
    nifty_data['Close'] = nifty_data['Close'] * 150  # Scale up like an index
    nifty_data['Volume'] = nifty_data['Volume'] * 10  # Higher index volume
    return nifty_data

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        'portfolio_capital': 100000.0,
        'risk_per_trade': 0.01,
        'transaction_cost': 0.0005,
        'slippage': 0.0005,
        'max_positions': 10,
        'max_position_size': 0.15,
        'max_portfolio_risk': 0.02,
        'min_risk_reward_ratio': 2.0,
        'indicators': {
            'rsi_period': 14,
            'atr_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14
        },
        'scoring': {
            'volume_weight': 25,
            'momentum_weight': 25,
            'trend_weight': 15,
            'volatility_weight': 10,
            'relative_strength_weight': 10,
            'volume_profile_weight': 10,
            'weekly_confirmation_bonus': 10
        },
        'regime_adjustments': {
            'bullish': {
                'rsi_min': 58,
                'volume_threshold': 1.8,
                'extreme_multiplier': 4.0
            },
            'bearish': {
                'rsi_min': 62,
                'volume_threshold': 2.2,
                'extreme_multiplier': 6.0
            },
            'sideways': {
                'rsi_min': 60,
                'volume_threshold': 2.0,
                'extreme_multiplier': 5.0
            },
            'high_volatility': {
                'rsi_min': 65,
                'volume_threshold': 2.5,
                'extreme_multiplier': 7.0
            }
        }
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "reports").mkdir()
    (output_dir / "charts").mkdir()
    (output_dir / "backtests").mkdir()
    return output_dir

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    import random
    random.seed(42)

@pytest.fixture
def mock_file_operations():
    """Mock file I/O operations to avoid actual file system access during tests."""
    with patch('pandas.DataFrame.to_csv'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('builtins.open', create=True), \
         patch('pathlib.Path.write_text'), \
         patch('pathlib.Path.mkdir'):
        yield

@pytest.fixture
def indicator_test_cases():
    """Provide comprehensive test cases for indicator validation."""
    return {
        'rsi_test_cases': [
            {
                'name': 'oversold_condition',
                'close_prices': [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72],
                'expected_rsi_range': (0, 35),  # Should be oversold
                'description': 'Continuous decline should result in low RSI'
            },
            {
                'name': 'overbought_condition', 
                'close_prices': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128],
                'expected_rsi_range': (65, 100),  # Should be overbought
                'description': 'Continuous rise should result in high RSI'
            },
            {
                'name': 'neutral_oscillation',
                'close_prices': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93],
                'expected_rsi_range': (40, 60),  # Should be neutral
                'description': 'Oscillating prices should result in neutral RSI'
            }
        ],
        'atr_test_cases': [
            {
                'name': 'high_volatility',
                'ohlc_data': [
                    {'Open': 100, 'High': 110, 'Low': 90, 'Close': 105},
                    {'Open': 105, 'High': 115, 'Low': 95, 'Close': 100},
                    {'Open': 100, 'High': 120, 'Low': 80, 'Close': 110},
                ],
                'expected_atr_min': 15.0,  # High volatility should produce high ATR
                'description': 'Wide trading ranges should produce high ATR'
            },
            {
                'name': 'low_volatility',
                'ohlc_data': [
                    {'Open': 100, 'High': 101, 'Low': 99, 'Close': 100.5},
                    {'Open': 100.5, 'High': 101.5, 'Low': 99.5, 'Close': 100},
                    {'Open': 100, 'High': 102, 'Low': 98, 'Close': 101},
                ],
                'expected_atr_max': 3.0,  # Low volatility should produce low ATR
                'description': 'Narrow trading ranges should produce low ATR'
            }
        ]
    }

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment before running any tests."""
    # Ensure test directories exist
    test_dir = Path(__file__).parent
    (test_dir / "output").mkdir(exist_ok=True)
    (test_dir / "temp").mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ['NSE_SCREENER_ENV'] = 'test'
    os.environ['PYTHONPATH'] = str(SRC_DIR)
    
    yield
    
    # Cleanup after all tests
    import shutil
    temp_dirs = [test_dir / "output", test_dir / "temp"]
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

# Custom markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take time")
    config.addinivalue_line("markers", "requires_network: Tests requiring internet")
    config.addinivalue_line("markers", "requires_talib: Tests requiring TA-Lib")

# Test data validation
def pytest_runtest_setup(item):
    """Run setup for each test item."""
    # Skip network tests if no connection available
    if item.get_closest_marker("requires_network"):
        pytest.importorskip("requests")
        # Could add actual network connectivity check here
    
    # Skip TA-Lib tests if not installed
    if item.get_closest_marker("requires_talib"):
        pytest.importorskip("talib")

# Performance monitoring for tests
@pytest.fixture(autouse=True)
def measure_test_time(request):
    """Measure and report test execution time."""
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    
    # Log slow tests
    if duration > 5.0:  # Tests taking more than 5 seconds
        print(f"\n⚠️  Slow test detected: {request.node.name} took {duration:.2f}s")