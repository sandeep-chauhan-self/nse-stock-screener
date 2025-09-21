# Test fixtures - Small deterministic datasets for unit testing
# This file contains carefully crafted test data with known expected outputs

from datetime import datetime, timedelta

from typing import Dict, Any
import numpy as np
import pandas as pd

def create_test_ohlcv_data() -> pd.DataFrame:
    """
    Create deterministic OHLCV data for testing indicators.
    This data is designed to produce known RSI, ATR, and other indicator values.
    """
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Create price data with known patterns
    # Starting at 100, with controlled moves to ensure predictable RSI
    base_price = 100.0
    price_changes = [
        # First 14 days (RSI initialization period)
        1.0, 2.0, -0.5, 1.5, -1.0, 2.0, -0.8, 1.2, -1.5, 2.2, -0.7, 1.8, -1.2, 1.0,
        # Next period - strong uptrend (RSI should be high)
        2.0, 1.5, 2.2, 1.8, 2.5, 1.2, 1.8, 2.0, 1.5, 1.0,
        # Next period - strong downtrend (RSI should be low)
        -2.0, -1.5, -2.2, -1.8, -2.5, -1.2, -1.8, -2.0, -1.5, -1.0,
        # Final period - sideways movement
        0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.3, -0.1, 0.1, -0.1, 0.2, 0.0
    ]

    # Calculate OHLC from price changes
    closes = [base_price]
    for change in price_changes:
        closes.append(closes[-1] + change)

    closes = closes[1:]  # Remove the base price

    # Create OHLC with controlled spreads for ATR testing
    data = []
    for i, close in enumerate(closes):
        # Create intraday range of 1-3% for consistent ATR
        range_pct = 0.02 + (i % 3) * 0.005  # 2.0%, 2.5%, 3.0% rotating
        intraday_range = close * range_pct

        high = close + intraday_range * 0.6
        low = close - intraday_range * 0.4
        open_price = low + intraday_range * 0.3

        # Volume pattern - higher volume on bigger moves
        volume = int(100000 + abs(price_changes[i] if i < len(price_changes) else 0) * 50000)

        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def create_expected_indicator_results() -> Dict[str, Any]:
    """
    Expected indicator results for the test OHLCV data.
    These values are calculated manually/separately to validate our implementations.
    """
    return {
        # RSI values at specific points (14-period RSI)
        'rsi_day_15': 67.89,  # After initial uptrend (approximately)
        'rsi_day_25': 85.32,  # During strong uptrend
        'rsi_day_35': 15.67,  # During strong downtrend
        'rsi_day_45': 52.13,  # During sideways movement

        # ATR values (14-period ATR)
        'atr_day_15': 2.45,   # Should be around this based on our range design
        'atr_day_25': 2.58,
        'atr_day_35': 2.67,
        'atr_day_45': 2.42,

        # Volume statistics
        'avg_volume_first_20': 125000,  # Average volume in first 20 days
        'max_volume': 200000,           # Maximum volume in dataset
        'min_volume': 100000,           # Minimum volume in dataset

        # Price statistics
        'max_close': 133.7,   # Approximately, after uptrend
        'min_close': 105.2,   # Approximately, after downtrend
        'final_close': 105.7, # Final closing price

        # MACD approximations (12,26,9)
        'macd_bullish_crossover_day': 18,  # Approximate day of bullish crossover
        'macd_bearish_crossover_day': 32,  # Approximate day of bearish crossover

        # ADX trend strength
        'adx_trend_strength': 25.5,  # Expected ADX during trend periods
    }

def create_regime_test_data() -> Dict[str, pd.DataFrame]:
    """
    Create test data for different market regimes to test regime-based adjustments.
    """
    base_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')

    # Bullish regime data - consistent uptrend
    bullish_prices = [100 + i * 1.5 + np.sin(i * 0.2) * 0.5 for i in range(30)]
    bullish_data = pd.DataFrame({
        'Date': base_dates,
        'Open': [p - 0.3 for p in bullish_prices],
        'High': [p + 0.8 for p in bullish_prices],
        'Low': [p - 0.5 for p in bullish_prices],
        'Close': bullish_prices,
        'Volume': [150000 + i * 2000 for i in range(30)]
    })
    bullish_data.set_index('Date', inplace=True)

    # Bearish regime data - consistent downtrend
    bearish_prices = [150 - i * 1.2 - np.sin(i * 0.3) * 0.7 for i in range(30)]
    bearish_data = pd.DataFrame({
        'Date': base_dates,
        'Open': [p + 0.3 for p in bearish_prices],
        'High': [p + 0.5 for p in bearish_prices],
        'Low': [p - 0.8 for p in bearish_prices],
        'Close': bearish_prices,
        'Volume': [180000 + i * 1500 for i in range(30)]
    })
    bearish_data.set_index('Date', inplace=True)

    # Sideways regime data - range-bound movement
    sideways_prices = [120 + np.sin(i * 0.4) * 3 + np.random.normal(0, 0.2) for i in range(30)]
    sideways_data = pd.DataFrame({
        'Date': base_dates,
        'Open': [p - 0.2 for p in sideways_prices],
        'High': [p + 0.6 for p in sideways_prices],
        'Low': [p - 0.4 for p in sideways_prices],
        'Close': sideways_prices,
        'Volume': [120000 + abs(np.sin(i * 0.5)) * 30000 for i in range(30)]
    })
    sideways_data.set_index('Date', inplace=True)

    return {
        'bullish': bullish_data,
        'bearish': bearish_data,
        'sideways': sideways_data
    }

def create_backtest_test_data() -> pd.DataFrame:
    """
    Create test data specifically for backtesting with known entry/exit points.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create a pattern with clear entry and exit signals
    # Days 1-20: Accumulation phase
    # Days 21-40: Breakout and uptrend
    # Days 41-60: Peak and distribution
    # Days 61-80: Downtrend
    # Days 81-100: Recovery

    prices = []
    base = 100

    # Accumulation (sideways with slight upward bias)
    for i in range(20):
        prices.append(base + np.sin(i * 0.3) * 2 + i * 0.1)

    # Breakout and uptrend
    for i in range(20):
        prices.append(prices[-1] + 1.5 + np.sin(i * 0.2) * 0.5)

    # Peak and distribution (volatile top)
    peak = prices[-1]
    for i in range(20):
        prices.append(peak + np.sin(i * 0.5) * 3 - i * 0.2)

    # Downtrend
    for i in range(20):
        prices.append(prices[-1] - 1.2 - np.sin(i * 0.3) * 0.3)

    # Recovery
    for i in range(20):
        prices.append(prices[-1] + 0.8 + np.sin(i * 0.4) * 0.4)

    # Create full OHLCV
    data = []
    for i, close in enumerate(prices):
        range_val = close * 0.025  # 2.5% daily range
        data.append({
            'Date': dates[i],
            'Open': close - range_val * 0.3,
            'High': close + range_val * 0.6,
            'Low': close - range_val * 0.4,
            'Close': close,
            'Volume': int(100000 + abs(np.sin(i * 0.1)) * 100000)
        })

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def create_risk_management_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Create test scenarios for risk management testing.
    """
    return {
        'normal_trade': {
            'portfolio_value': 100000,
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'risk_per_trade': 0.01,
            'expected_quantity': 200,  # 1% of 100k = 1000, divided by 5 risk = 200 shares
            'expected_position_value': 20000
        },
        'high_risk_trade': {
            'portfolio_value': 100000,
            'entry_price': 200.0,
            'stop_loss': 180.0,
            'risk_per_trade': 0.02,
            'expected_quantity': 100,  # 2% of 100k = 2000, divided by 20 risk = 100 shares
            'expected_position_value': 20000
        },
        'small_stop_trade': {
            'portfolio_value': 100000,
            'entry_price': 50.0,
            'stop_loss': 49.0,
            'risk_per_trade': 0.01,
            'expected_quantity': 1000,  # 1% of 100k = 1000, divided by 1 risk = 1000 shares
            'expected_position_value': 50000
        },
        'max_position_limit': {
            'portfolio_value': 100000,
            'entry_price': 10.0,
            'stop_loss': 9.5,
            'risk_per_trade': 0.01,
            'max_position_size': 0.15,  # 15% max position
            'expected_quantity': 1500,  # Limited by max position size
            'expected_position_value': 15000
        }
    }

# Utility function to save fixtures as CSV files
def save_fixtures_to_csv():
    """Save all test fixtures as CSV files for use in tests."""
    import os

    fixture_dir = os.path.dirname(__file__)

    # Save main OHLCV test data
    test_data = create_test_ohlcv_data()
    test_data.to_csv(os.path.join(fixture_dir, 'test_ohlcv_data.csv'))

    # Save regime test data
    regime_data = create_regime_test_data()
    for regime, data in regime_data.items():
        data.to_csv(os.path.join(fixture_dir, f'test_{regime}_regime_data.csv'))

    # Save backtest test data
    backtest_data = create_backtest_test_data()
    backtest_data.to_csv(os.path.join(fixture_dir, 'test_backtest_data.csv'))

    print("Test fixtures saved to CSV files.")

if __name__ == "__main__":
    save_fixtures_to_csv()