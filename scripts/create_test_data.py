# Test Data Generation for NSE Stock Screener

# This script creates sample datasets for reproducible testing
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
def create_sample_stock_symbols():
    """Create a sample List[str] of stock symbols for testing."""
    test_symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "ITC.NS",
        "BHARTIARTL.NS",
        "SBIN.NS",
        "LT.NS",
        "KOTAKBANK.NS",
        "HINDUNILVR.NS",
        "AXISBANK.NS",
        "ASIANPAINT.NS",
        "MARUTI.NS",
        "NTPC.NS",
        "TITAN.NS",
        "SUNPHARMA.NS",
        "ULTRACEMCO.NS",
        "NESTLEIND.NS",
        "POWERGRID.NS"
    ]
    return test_symbols
def generate_synthetic_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Generate synthetic stock data for testing."""
    rng = np.random.default_rng(hash(symbol) % 2**32)
  # Deterministic but different per symbol
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = dates[dates.weekday < 5]
  # Only weekdays
    # Base price varies by symbol
    base_prices = {
        'RELIANCE.NS': 2500,
        'TCS.NS': 3500,
        'INFY.NS': 1600,
        'HDFCBANK.NS': 1500,
        'ICICIBANK.NS': 900,
        'ITC.NS': 450,
        'BHARTIARTL.NS': 900,
        'SBIN.NS': 550,
        'LT.NS': 2000,
        'KOTAKBANK.NS': 1800
    }
    base_price = base_prices.get(symbol, 1000)

    # Generate realistic price movements
    returns = rng.normal(0.001, 0.02, len(dates))
  # Daily returns
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))
  # Ensure positive prices
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):

        # Intraday volatility
        volatility = 0.015
        high = close * (1 + rng.uniform(0, volatility))
        low = close * (1 - rng.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else close

        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume based on price movement
        volume_base = 1000000
        volume = int(volume_base * (1 + abs(returns[i]) * 10))
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume,
            'Adj Close': round(close, 2)
        })
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df
def create_test_stock_data():
    """Create comprehensive test stock data."""
    test_symbols = create_sample_stock_symbols()
    data_dir = Path("data/test")
    data_dir.mkdir(exist_ok=True)
    print("📊 Generating test stock data...")
    for symbol in test_symbols:
        print(f"  Generating data for {symbol}")
        df = generate_synthetic_stock_data(symbol)

        # Save as CSV
        csv_path = data_dir / f"{symbol.replace('.NS', '')}_test_data.csv"
        df.to_csv(csv_path)

        # Also create a smaller dataset for quick tests
        quick_df = df.tail(30)
  # Last 30 days
        quick_path = data_dir / f"{symbol.replace('.NS', '')}_quick_test.csv"
        quick_df.to_csv(quick_path)

    # Create test symbols List[str]
    symbols_path = data_dir / "test_symbols.txt"
    with open(symbols_path, 'w') as f:
        for symbol in test_symbols:
            f.write(f"{symbol.replace('.NS', '')}\n")
    print(f"✅ Test data created in {data_dir}")
def create_test_configurations():
    """Create test configuration files."""
    config_dir = Path("data/test/configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Test configuration for different scenarios
    configs = {
        "conservative_test.json": {
            "risk_tolerance": "low",
            "max_position_size": 0.05,
            "stop_loss_percent": 0.02,
            "take_profit_percent": 0.06,
            "technical_indicators": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "sma_short": 20,
                "sma_long": 50,
                "volume_threshold": 1.5
            },
            "filters": {
                "min_price": 10,
                "max_price": 10000,
                "min_volume": 100000,
                "min_market_cap": 1000000000
            }
        },
        "aggressive_test.json": {
            "risk_tolerance": "high",
            "max_position_size": 0.10,
            "stop_loss_percent": 0.05,
            "take_profit_percent": 0.15,
            "technical_indicators": {
                "rsi_period": 14,
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "sma_short": 10,
                "sma_long": 30,
                "volume_threshold": 2.0
            },
            "filters": {
                "min_price": 1,
                "max_price": 50000,
                "min_volume": 50000,
                "min_market_cap": 100000000
            }
        },
        "quick_test.json": {
            "risk_tolerance": "medium",
            "max_position_size": 0.02,
            "stop_loss_percent": 0.03,
            "take_profit_percent": 0.08,
            "technical_indicators": {
                "rsi_period": 7,
  # Shorter for quick tests
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "sma_short": 5,
                "sma_long": 15,
                "volume_threshold": 1.2
            },
            "filters": {
                "min_price": 50,
                "max_price": 5000,
                "min_volume": 200000,
                "min_market_cap": 5000000000
            }
        }
    }
    print("⚙️  Creating test configurations...")
    for filename, config in configs.items():
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Created {filename}")
    print(f"✅ Test configurations created in {config_dir}")
def create_test_scenarios():
    """Create test scenarios for different market conditions."""
    scenarios_dir = Path("data/test/scenarios")
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "bullish_market.json": {
            "description": "Bullish market conditions with upward trends",
            "market_sentiment": "bullish",
            "volatility": "low",
            "test_symbols": ["RELIANCE", "TCS", "INFY", "HDFCBANK"],
            "expected_signals": "buy",
            "duration_days": 30,
            "trend_direction": "up",
            "avg_daily_return": 0.015
        },
        "bearish_market.json": {
            "description": "Bearish market conditions with downward trends",
            "market_sentiment": "bearish",
            "volatility": "high",
            "test_symbols": ["ITC", "BHARTIARTL", "SBIN"],
            "expected_signals": "sell",
            "duration_days": 30,
            "trend_direction": "down",
            "avg_daily_return": -0.012
        },
        "sideways_market.json": {
            "description": "Sideways market with range-bound movement",
            "market_sentiment": "neutral",
            "volatility": "medium",
            "test_symbols": ["LT", "KOTAKBANK", "HINDUNILVR"],
            "expected_signals": "hold",
            "duration_days": 30,
            "trend_direction": "sideways",
            "avg_daily_return": 0.002
        },
        "high_volatility.json": {
            "description": "High volatility market conditions",
            "market_sentiment": "uncertain",
            "volatility": "very_high",
            "test_symbols": ["AXISBANK", "ASIANPAINT", "MARUTI"],
            "expected_signals": "mixed",
            "duration_days": 15,
            "trend_direction": "volatile",
            "avg_daily_return": 0.005
        }
    }
    print("📈 Creating test scenarios...")
    for filename, scenario in scenarios.items():
        scenario_path = scenarios_dir / filename
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        print(f"  Created {filename}")
    print(f"✅ Test scenarios created in {scenarios_dir}")
def create_sample_results():
    """Create sample analysis results for testing."""
    results_dir = Path("data/test/sample_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sample analysis results
    sample_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "market_overview": {
            "total_stocks_analyzed": 20,
            "bullish_signals": 8,
            "bearish_signals": 3,
            "neutral_signals": 9,
            "market_sentiment": "moderately_bullish"
        },
        "top_picks": [
            {
                "symbol": "RELIANCE",
                "score": 85.5,
                "signal": "BUY",
                "confidence": 0.78,
                "key_indicators": {
                    "rsi": 35.2,
                    "price_trend": "upward",
                    "volume_surge": True,
                    "support_level": 2450.0,
                    "resistance_level": 2650.0
                }
            },
            {
                "symbol": "TCS",
                "score": 82.1,
                "signal": "BUY",
                "confidence": 0.73,
                "key_indicators": {
                    "rsi": 42.8,
                    "price_trend": "upward",
                    "volume_surge": False,
                    "support_level": 3400.0,
                    "resistance_level": 3700.0
                }
            }
        ],
        "risk_warnings": [
            {
                "symbol": "ITC",
                "warning": "Downward trend with high volume",
                "risk_level": "high"
            }
        ]
    }

    # Save sample results
    results_path = results_dir / "sample_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(sample_results, f, indent=2)

    # Create sample CSV report
    csv_data = pd.DataFrame([
        {"Symbol": "RELIANCE", "Score": 85.5, "Signal": "BUY", "Price": 2550.0, "Change": 2.5},
        {"Symbol": "TCS", "Score": 82.1, "Signal": "BUY", "Price": 3580.0, "Change": 1.8},
        {"Symbol": "INFY", "Score": 78.3, "Signal": "HOLD", "Price": 1625.0, "Change": 0.5},
        {"Symbol": "HDFCBANK", "Score": 75.9, "Signal": "HOLD", "Price": 1485.0, "Change": -0.2},
        {"Symbol": "ITC", "Score": 45.2, "Signal": "SELL", "Price": 445.0, "Change": -3.1},
    ])
    csv_path = results_dir / "sample_analysis.csv"
    csv_data.to_csv(csv_path, index=False)
    print(f"✅ Sample results created in {results_dir}")
def create_test_environment_file():
    """Create a test environment configuration file."""
    env_content = """# Test Environment Configuration for NSE Stock Screener

# Copy this to .env for local testing
# Data Sources
NSE_DATA_SOURCE=test
YAHOO_FINANCE_ENABLED=false
USE_CACHE=true
CACHE_EXPIRY_HOURS=24

# Testing Configuration
TEST_MODE=true
TEST_DATA_PATH=data/test
MOCK_API_CALLS=true
ENABLE_LIVE_DATA=false

# Logging
LOG_LEVEL=DEBUG
LOG_TO_FILE=true
LOG_FILE_PATH=logs/test.log

# Performance
MAX_CONCURRENT_REQUESTS=5
REQUEST_DELAY_SECONDS=0.1
ENABLE_RATE_LIMITING=true

# Output
OUTPUT_FORMAT=json,csv
SAVE_CHARTS=true
CHART_FORMAT=png
ENABLE_REPORTS=true

# Alerts (disabled for testing)
ENABLE_EMAIL_ALERTS=false
ENABLE_SLACK_ALERTS=false
ENABLE_WEBHOOK_ALERTS=false

# Database (use SQLite for testing)
DATABASE_TYPE=sqlite
DATABASE_PATH=data/test/test_database.db

# Security (test keys - not for production)
API_KEY_ENCRYPTION=false
ENABLE_SSL_VERIFICATION=false

# Development
DEBUG_MODE=true
PROFILING_ENABLED=true
MEMORY_MONITORING=true
"""
    env_path = Path("data/test/.env.test")
    with open(env_path, 'w') as f:
        f.write(env_content)
    print(f"✅ Test environment file created at {env_path}")
def main():
    """Main function to create all test data and fixtures."""
    print("🧪 Creating Test Data and Fixtures for NSE Stock Screener")
    print("=" * 60)
    try:

        # Create all test components
        create_test_stock_data()
        create_test_configurations()
        create_test_scenarios()
        create_sample_results()
        create_test_environment_file()
        print("\n" + "=" * 60)
        print("✅ All test data and fixtures created successfully!")
        print("=" * 60)
        print("\n📁 Created test data structure:")
        print("  data/test/")
        print("  ├── *.csv (stock data for 20 symbols)")
        print("  ├── *_quick_test.csv (30-day datasets)")
        print("  ├── test_symbols.txt")
        print("  ├── configs/ (test configurations)")
        print("  ├── scenarios/ (market scenarios)")
        print("  ├── sample_results/ (expected outputs)")
        print("  └── .env.test (test environment)")
        print("\n🧪 Usage in tests:")
        print("  - Load test data: pd.read_csv('data/test/RELIANCE_test_data.csv')")
        print("  - Use test config: json.load(open('data/test/configs/quick_test.json'))")
        print("  - Run scenarios: pytest tests/ -m 'integration'")
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        return 1
    return 0
if __name__ == "__main__":
    import sys
    sys.exit(main())
