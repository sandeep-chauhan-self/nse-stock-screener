"""
Comprehensive test framework for indicator validation.

This module provides test suites for validating indicator implementations
across different market scenarios, stress events, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import logging
import time
import json

from ..base import BaseIndicator, IndicatorResult, VectorizedIndicator
from ..vectorized import RSI, MACD, ATR, BollingerBands, ADX, VolumeProfile
from ..engine import IndicatorEngine
from ..config import IndicatorEngineConfig, ValidationConfig
from .stress_data import ValidationDataset, StressEvent, get_validation_dataset

logger = logging.getLogger(__name__)


class IndicatorTestCase(unittest.TestCase):
    """Base test case for indicator testing with common utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.validation_dataset = get_validation_dataset()
        self.engine = IndicatorEngine()
        self.tolerance = 1e-6  # Numerical tolerance for comparisons
        
    def assertIndicatorResult(self, 
                            result: IndicatorResult,
                            expected_length: int,
                            min_confidence: float = 0.8):
        """Assert that an indicator result meets basic quality criteria."""
        self.assertIsInstance(result, IndicatorResult)
        self.assertIsNotNone(result.values)
        self.assertGreaterEqual(len(result.values), expected_length)
        self.assertGreaterEqual(result.confidence, min_confidence)
        self.assertIsInstance(result.metadata, dict)
        
        # Check for NaN values in final period
        if len(result.values) > 0:
            final_values = result.values.iloc[-10:]  # Last 10 values
            nan_count = final_values.isna().sum()
            self.assertLess(nan_count, len(final_values) * 0.3,  # Max 30% NaN
                           "Too many NaN values in final period")
    
    def assertIndicatorRange(self, 
                           result: IndicatorResult,
                           min_value: float,
                           max_value: float,
                           column: str = None):
        """Assert that indicator values fall within expected range."""
        values = result.values
        if column and isinstance(values, pd.DataFrame):
            values = values[column]
        
        # Remove NaN values for range checking
        clean_values = values.dropna()
        if len(clean_values) > 0:
            self.assertGreaterEqual(clean_values.min(), min_value)
            self.assertLessEqual(clean_values.max(), max_value)
    
    def assertIndicatorStability(self, 
                                result: IndicatorResult,
                                max_consecutive_nan: int = 5):
        """Assert that indicator doesn't have excessive consecutive NaN values."""
        values = result.values
        if isinstance(values, pd.DataFrame):
            # Check each column
            for col in values.columns:
                self._check_consecutive_nan(values[col], max_consecutive_nan)
        else:
            self._check_consecutive_nan(values, max_consecutive_nan)
    
    def _check_consecutive_nan(self, series: pd.Series, max_consecutive: int):
        """Check for consecutive NaN values in a series."""
        is_nan = series.isna()
        consecutive_count = 0
        max_found = 0
        
        for is_na in is_nan:
            if is_na:
                consecutive_count += 1
                max_found = max(max_found, consecutive_count)
            else:
                consecutive_count = 0
        
        self.assertLessEqual(max_found, max_consecutive,
                           f"Found {max_found} consecutive NaN values")


class RSITests(IndicatorTestCase):
    """Test suite for RSI indicator."""
    
    def test_rsi_basic_functionality(self):
        """Test basic RSI calculation."""
        data = self.validation_dataset.load_dataset('normal_market')
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        self.assertIndicatorResult(result, len(data) - 14)
        self.assertIndicatorRange(result, 0, 100)
        self.assertIndicatorStability(result)
    
    def test_rsi_known_values(self):
        """Test RSI with known expected values."""
        known_datasets = self.validation_dataset.get_known_values_dataset()
        constant_data, expected = known_datasets['constant_price']
        
        rsi = RSI(period=14)
        result = rsi.calculate(constant_data)
        
        # For constant prices, RSI should converge to 50
        final_rsi = result.values.iloc[-1]
        self.assertAlmostEqual(final_rsi, expected['RSI_14'], delta=1.0)
    
    def test_rsi_stress_scenarios(self):
        """Test RSI behavior during stress scenarios."""
        stress_scenarios = ['market_crash', 'high_volatility', 'bull_market']
        
        for scenario in stress_scenarios:
            with self.subTest(scenario=scenario):
                data = self.validation_dataset.load_dataset(scenario)
                
                rsi = RSI(period=14)
                result = rsi.calculate(data)
                
                self.assertIndicatorResult(result, len(data) - 14, min_confidence=0.7)
                self.assertIndicatorRange(result, 0, 100)
                
                # During crashes, should see extreme RSI values
                if scenario == 'market_crash':
                    min_rsi = result.values.min()
                    self.assertLess(min_rsi, 30, "Expected oversold conditions during crash")
    
    def test_rsi_different_periods(self):
        """Test RSI with different period parameters."""
        data = self.validation_dataset.load_dataset('normal_market')
        periods = [9, 14, 21, 50]
        
        results = {}
        for period in periods:
            rsi = RSI(period=period)
            result = rsi.calculate(data)
            results[period] = result
            
            self.assertIndicatorResult(result, len(data) - period)
            self.assertIndicatorRange(result, 0, 100)
        
        # Shorter periods should be more volatile (higher standard deviation)
        std_9 = results[9].values.std()
        std_50 = results[50].values.std()
        self.assertGreater(std_9, std_50, "Shorter period RSI should be more volatile")


class MACDTests(IndicatorTestCase):
    """Test suite for MACD indicator."""
    
    def test_macd_basic_functionality(self):
        """Test basic MACD calculation."""
        data = self.validation_dataset.load_dataset('normal_market')
        
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd.calculate(data)
        
        self.assertIndicatorResult(result, len(data) - 26)
        self.assertIsInstance(result.values, pd.DataFrame)
        self.assertIn('MACD', result.values.columns)
        self.assertIn('Signal', result.values.columns) 
        self.assertIn('Histogram', result.values.columns)
    
    def test_macd_trending_market(self):
        """Test MACD in trending market conditions."""
        data = self.validation_dataset.load_dataset('bull_market')
        
        macd = MACD()
        result = macd.calculate(data)
        
        self.assertIndicatorResult(result, len(data) - 26)
        
        # In bull market, MACD should eventually turn positive
        final_macd = result.values['MACD'].iloc[-10:].mean()
        self.assertGreater(final_macd, 0, "Expected positive MACD in bull market")
    
    def test_macd_crossover_signals(self):
        """Test MACD crossover signal generation."""
        data = self.validation_dataset.load_dataset('high_volatility')
        
        macd = MACD()
        result = macd.calculate(data)
        
        macd_line = result.values['MACD']
        signal_line = result.values['Signal']
        
        # Check for crossovers
        crossovers = (macd_line > signal_line) != (macd_line.shift(1) > signal_line.shift(1))
        crossover_count = crossovers.sum()
        
        # High volatility market should have multiple crossovers
        self.assertGreater(crossover_count, 5, "Expected multiple MACD crossovers")


class ATRTests(IndicatorTestCase):
    """Test suite for ATR indicator."""
    
    def test_atr_basic_functionality(self):
        """Test basic ATR calculation."""
        data = self.validation_dataset.load_dataset('normal_market')
        
        atr = ATR(period=14)
        result = atr.calculate(data)
        
        self.assertIndicatorResult(result, len(data) - 14)
        self.assertGreater(result.values.min(), 0, "ATR should always be positive")
    
    def test_atr_volatility_scaling(self):
        """Test ATR scaling with market volatility."""
        low_vol_data = self.validation_dataset.load_dataset('low_volatility')
        high_vol_data = self.validation_dataset.load_dataset('high_volatility')
        
        atr = ATR(period=14)
        low_vol_result = atr.calculate(low_vol_data)
        high_vol_result = atr.calculate(high_vol_data)
        
        low_vol_atr = low_vol_result.values.mean()
        high_vol_atr = high_vol_result.values.mean()
        
        self.assertGreater(high_vol_atr, low_vol_atr * 1.5,
                          "High volatility ATR should be significantly higher")
    
    def test_atr_known_values(self):
        """Test ATR with known expected values."""
        known_datasets = self.validation_dataset.get_known_values_dataset()
        constant_data, expected = known_datasets['constant_price']
        
        atr = ATR(period=14)
        result = atr.calculate(constant_data)
        
        # For constant prices, ATR should be very close to 0
        final_atr = result.values.iloc[-1]
        self.assertAlmostEqual(final_atr, expected['ATR_14'], delta=0.1)


class PerformanceTests(IndicatorTestCase):
    """Test suite for performance validation."""
    
    def test_indicator_calculation_speed(self):
        """Test that indicators calculate within reasonable time."""
        data = self.validation_dataset.load_dataset('normal_market')
        
        # Test individual indicators
        indicators = [
            RSI(period=14),
            MACD(fast_period=12, slow_period=26, signal_period=9),
            ATR(period=14),
            BollingerBands(period=20, std_dev=2),
            ADX(period=14)
        ]
        
        for indicator in indicators:
            with self.subTest(indicator=type(indicator).__name__):
                start_time = time.time()
                result = indicator.calculate(data)
                calculation_time = time.time() - start_time
                
                # Should calculate within 1 second for normal dataset
                self.assertLess(calculation_time, 1.0,
                              f"{type(indicator).__name__} took too long: {calculation_time:.3f}s")
                self.assertIndicatorResult(result, 0)
    
    def test_engine_parallel_performance(self):
        """Test parallel calculation performance."""
        data = self.validation_dataset.load_dataset('normal_market')
        
        indicators = {
            'RSI_14': RSI(period=14),
            'RSI_21': RSI(period=21),
            'MACD': MACD(),
            'ATR_14': ATR(period=14),
            'BB_20': BollingerBands(period=20)
        }
        
        # Sequential calculation
        start_time = time.time()
        sequential_results = {}
        for name, indicator in indicators.items():
            sequential_results[name] = indicator.calculate(data)
        sequential_time = time.time() - start_time
        
        # Parallel calculation
        start_time = time.time()
        parallel_results = self.engine.calculate_indicators(data, indicators, parallel=True)
        parallel_time = time.time() - start_time
        
        # Parallel should be faster for multiple indicators
        if len(indicators) > 2:
            self.assertLess(parallel_time, sequential_time * 0.8,
                          "Parallel calculation should be faster")
        
        # Results should be equivalent
        for name in indicators.keys():
            seq_values = sequential_results[name].values
            par_values = parallel_results[name].values
            
            if isinstance(seq_values, pd.DataFrame):
                for col in seq_values.columns:
                    np.testing.assert_array_almost_equal(
                        seq_values[col].dropna().values,
                        par_values[col].dropna().values,
                        decimal=6
                    )
            else:
                np.testing.assert_array_almost_equal(
                    seq_values.dropna().values,
                    par_values.dropna().values,
                    decimal=6
                )


class StressTests(IndicatorTestCase):
    """Test suite for stress scenario validation."""
    
    def test_crash_scenario_behavior(self):
        """Test indicator behavior during market crash."""
        crash_data = self.validation_dataset.load_dataset('market_crash')
        
        indicators = {
            'RSI': RSI(period=14),
            'ATR': ATR(period=14),
            'MACD': MACD()
        }
        
        results = self.engine.calculate_indicators(crash_data, indicators)
        
        # During crash, expect:
        # - RSI to reach oversold levels
        # - ATR to spike (high volatility)
        # - MACD to turn negative
        
        rsi_min = results['RSI'].values.min()
        atr_max = results['ATR'].values.max()
        macd_final = results['MACD'].values['MACD'].iloc[-5:].mean()
        
        self.assertLess(rsi_min, 30, "Expected oversold RSI during crash")
        self.assertGreater(atr_max, crash_data['Close'].iloc[0] * 0.05,
                          "Expected ATR spike during crash")
        self.assertLess(macd_final, 0, "Expected negative MACD during crash")
    
    def test_high_volatility_stability(self):
        """Test indicator stability during high volatility."""
        volatile_data = self.validation_dataset.load_dataset('high_volatility')
        
        indicators = {
            'RSI_14': RSI(period=14),
            'RSI_50': RSI(period=50),  # Longer period for stability
            'ATR': ATR(period=14),
            'BB': BollingerBands(period=20)
        }
        
        results = self.engine.calculate_indicators(volatile_data, indicators)
        
        # All indicators should maintain stability (no excessive NaN values)
        for name, result in results.items():
            with self.subTest(indicator=name):
                self.assertIndicatorStability(result, max_consecutive_nan=3)
                self.assertGreater(result.confidence, 0.6,
                                 f"{name} confidence too low: {result.confidence}")
    
    def test_edge_case_data(self):
        """Test indicators with edge case data."""
        # Create edge case scenarios
        edge_cases = {
            'single_day': pd.DataFrame({
                'Open': [100], 'High': [101], 'Low': [99], 'Close': [100.5], 'Volume': [1000]
            }, index=[pd.Timestamp('2023-01-01')]),
            
            'extreme_gap': self._create_gap_data(),
            'zero_volume': self._create_zero_volume_data()
        }
        
        for case_name, data in edge_cases.items():
            with self.subTest(case=case_name):
                try:
                    rsi = RSI(period=14)
                    result = rsi.calculate(data)
                    
                    # Should handle gracefully without errors
                    self.assertIsNotNone(result)
                    
                except Exception as e:
                    self.fail(f"Indicator failed on {case_name}: {e}")
    
    def _create_gap_data(self) -> pd.DataFrame:
        """Create data with extreme price gaps."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = [100] * 25 + [200] * 25  # 100% gap in middle
        
        return pd.DataFrame({
            'Open': prices,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)
    
    def _create_zero_volume_data(self) -> pd.DataFrame:
        """Create data with zero volume periods."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        return pd.DataFrame({
            'Open': [100] * 30,
            'High': [101] * 30,
            'Low': [99] * 30,
            'Close': [100] * 30,
            'Volume': [0] * 30  # Zero volume
        }, index=dates)


class ValidationTestSuite:
    """Complete validation test suite for indicator engine."""
    
    def __init__(self):
        self.test_classes = [
            RSITests,
            MACDTests, 
            ATRTests,
            PerformanceTests,
            StressTests
        ]
    
    def run_all_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run all validation tests."""
        suite = unittest.TestSuite()
        
        for test_class in self.test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return runner.run(suite)
    
    def run_specific_tests(self, test_names: List[str], verbosity: int = 2) -> unittest.TestResult:
        """Run specific tests by name."""
        suite = unittest.TestSuite()
        
        for test_class in self.test_classes:
            for test_name in test_names:
                if hasattr(test_class, test_name):
                    suite.addTest(test_class(test_name))
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return runner.run(suite)
    
    def generate_test_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'timestamp': time.time(),
            'test_results': {},
            'performance_metrics': {},
            'stress_test_results': {}
        }
        
        # Run tests and collect results
        result = self.run_all_tests(verbosity=0)
        
        report['test_results'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


def run_validation_tests(test_names: Optional[List[str]] = None, 
                        verbosity: int = 2) -> unittest.TestResult:
    """
    Run validation tests for the indicator engine.
    
    Args:
        test_names: Specific test names to run (optional)
        verbosity: Test output verbosity level
        
    Returns:
        Test results
    """
    suite = ValidationTestSuite()
    
    if test_names:
        return suite.run_specific_tests(test_names, verbosity)
    else:
        return suite.run_all_tests(verbosity)


if __name__ == "__main__":
    # Run all validation tests when executed as script
    print("Running indicator validation tests...")
    result = run_validation_tests()
    
    if result.wasSuccessful():
        print("\n✅ All validation tests passed!")
    else:
        print(f"\n❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        
        for test, error in result.failures + result.errors:
            print(f"  - {test}: {error}")