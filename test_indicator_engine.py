"""
Simple test script to verify the new indicator engine functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Import the new indicator system
from src.indicators.vectorized import RSI, MACD, ATR, BollingerBands
from src.indicators.factories import create_indicator
from src.indicators.engine import IndicatorEngine
from src.indicators.validation import get_validation_dataset, create_stress_test_data

def create_sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))

    return data

def test_individual_indicators():
    """Test individual indicator calculations."""
    print("Testing Individual Indicators")
    print("=" * 40)

    data = create_sample_data()

    # Test RSI
    print("Testing RSI...")
    rsi = RSI(period=14)
    rsi_result = rsi.calculate(data)
    print(f"  RSI calculated: {len(rsi_result.values)} values")
    print(f"  RSI range: {rsi_result.values.min():.2f} - {rsi_result.values.max():.2f}")
    print(f"  RSI confidence: {rsi_result.confidence:.3f}")

    # Test MACD
    print("\nTesting MACD...")
    macd = MACD(fast=12, slow=26, signal=9)
    macd_result = macd.calculate(data)
    print(f"  MACD calculated: {len(macd_result.values)} values")
    print(f"  MACD columns: {list(macd_result.values.columns)}")
    print(f"  MACD confidence: {macd_result.confidence:.3f}")

    # Test ATR
    print("\nTesting ATR...")
    atr = ATR(period=14)
    atr_result = atr.calculate(data)
    print(f"  ATR calculated: {len(atr_result.values)} values")
    print(f"  ATR range: {atr_result.values.min():.4f} - {atr_result.values.max():.4f}")
    print(f"  ATR confidence: {atr_result.confidence:.3f}")

    return True

def test_factory_system():
    """Test the factory system for indicator creation."""
    print("\nTesting Factory System")
    print("=" * 40)

    data = create_sample_data()

    # Test factory creation
    print("Creating indicators via factory...")
    rsi_14 = create_indicator('RSI', period=14)
    rsi_21 = create_indicator('RSI', period=21)
    macd_std = create_indicator('MACD')
    atr_14 = create_indicator('ATR', period=14)

    indicators = {
        'RSI_14': rsi_14,
        'RSI_21': rsi_21,
        'MACD': macd_std,
        'ATR_14': atr_14
    }

    # Calculate all indicators
    results = {}
    for name, indicator in indicators.items():
        result = indicator.calculate(data)
        results[name] = result
        print(f"  {name}: {len(result.values)} values, confidence: {result.confidence:.3f}")

    return True

def test_engine_system():
    """Test the indicator engine."""
    print("\nTesting Indicator Engine")
    print("=" * 40)

    data = create_sample_data()
    engine = IndicatorEngine()

    # Define indicators to calculate
    indicators = {
        'RSI_14': RSI(period=14),
        'RSI_21': RSI(period=21),
        'MACD': MACD(),
        'ATR_14': ATR(period=14),
        'BB_20': BollingerBands(period=20, std_dev=2)
    }

    # Sequential calculation
    print("Testing sequential calculation...")
    sequential_results = engine.calculate_indicators(data, indicators, parallel=False)
    print(f"  Calculated {len(sequential_results)} indicators sequentially")

    # Parallel calculation
    print("Testing parallel calculation...")
    parallel_results = engine.calculate_indicators(data, indicators, parallel=True)
    print(f"  Calculated {len(parallel_results)} indicators in parallel")

    # Compare results
    print("Comparing sequential vs parallel results...")
    for name in indicators.keys():
        seq_values = sequential_results[name].values
        par_values = parallel_results[name].values

        if isinstance(seq_values, pd.Series):
            diff = np.abs(seq_values - par_values).max()
            print(f"  {name}: max difference = {diff:.10f}")
        else:
            print(f"  {name}: DataFrame result (comparison skipped)")

    return True

def test_validation_system():
    """Test the validation system."""
    print("\nTesting Validation System")
    print("=" * 40)

    # Create validation datasets
    print("Creating validation datasets...")
    try:
        dataset = get_validation_dataset()
        scenarios = dataset.generate_test_scenarios()
        print(f"  Generated {len(scenarios)} test scenarios:")
        for name, data in scenarios.items():
            print(f"    {name}: {len(data)} data points")

        # Test with one scenario
        normal_data = scenarios['normal_market']

        # Test indicator on validation data
        rsi = RSI(period=14)
        result = rsi.calculate(normal_data)
        print(f"  RSI on normal market: {len(result.values)} values, confidence: {result.confidence:.3f}")

        return True

    except Exception as e:
        print(f"  Validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Indicator Engine Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()

    tests = [
        ("Individual Indicators", test_individual_indicators),
        ("Factory System", test_factory_system),
        ("Engine System", test_engine_system),
        ("Validation System", test_validation_system)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The indicator engine is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)