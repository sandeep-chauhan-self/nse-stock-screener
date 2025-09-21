"""
Unit tests for RSI and ATR functions in advanced_indicators.py
Tests use deterministic CSV fixtures to ensure reproducible results.
Tests verify that RSI and ATR calculations follow expected formulas and ranges.
"""
import unittest
import pandas as pd
import numpy as np
import math
from pathlib import Path
import sys
import os
# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Mock the data validation imports to avoid import errors
class MockDataContract:
    indicator_dict = dict
sys.modules['data.validation'] = type(sys)('data.validation')
sys.modules['data.validation'].DataContract = MockDataContract
sys.modules['data.validation'].safe_float = lambda x, default, name: x if not math.isnan(x) else default
sys.modules['data.validation'].safe_bool = lambda x, default, name: x if x is not None else default
sys.modules['data.validation'].is_valid_numeric = lambda x: not math.isnan(x) if isinstance(x, (int, float)) else False
sys.modules['data.validation'].replace_invalid_with_nan = lambda x: x
from advanced_indicators import AdvancedIndicator
class TestRSIATRFunctions(unittest.TestCase):
    """Test suite for RSI and ATR indicator calculations."""
    @classmethod
    def setUpClass(cls):
        """Set up test data and indicators calculator."""
        # Load deterministic test data
        fixtures_path = Path(__file__).parent / 'fixtures' / 'test_ohlc_data.csv'
        cls.test_data = pd.read_csv(fixtures_path)
        cls.test_data['Date'] = pd.to_datetime(cls.test_data['Date'])
        cls.test_data.set_index('Date', inplace=True)
        # Initialize indicators calculator
        cls.indicators = AdvancedIndicator()
    def test_rsi_calculation_basic(self):
        """Test RSI calculation with known data returns expected range."""
        result = self.indicators.compute_momentum_signals(self.test_data)
        # RSI should be between 0 and 100
        self.assertGreaterEqual(result['rsi'], 0.0, "RSI should be >= 0")
        self.assertLessEqual(result['rsi'], 100.0, "RSI should be <= 100")
        # RSI should not be NaN with sufficient data
        self.assertFalse(math.isnan(result['rsi']), "RSI should not be NaN with sufficient data")
    def test_rsi_calculation_deterministic(self):
        """Test RSI calculation produces consistent results with deterministic data."""
        # Run calculation twice to ensure consistency
        result1 = self.indicators.compute_momentum_signals(self.test_data)
        result2 = self.indicators.compute_momentum_signals(self.test_data)
        # Results should be identical
        self.assertEqual(result1['rsi'], result2['rsi'], "RSI calculation should be deterministic")
    def test_rsi_with_trending_data(self):
        """Test RSI behavior with known trending pattern."""
        # Use full dataset for trending comparisons since RSI needs sufficient data
        full_result = self.indicators.compute_momentum_signals(self.test_data)
        # RSI should be valid with full dataset
        self.assertFalse(math.isnan(full_result['rsi']), "RSI should be valid with full dataset")
        # Create clearly trending data by extending our pattern
        trending_up_data = self.test_data.copy()
        for i in range(1, len(trending_up_data)):
            trending_up_data.iloc[i, trending_up_data.columns.get_loc('Close')] = trending_up_data.iloc[i-1]['Close'] * 1.01
        uptrend_result = self.indicators.compute_momentum_signals(trending_up_data)
        trending_down_data = self.test_data.copy()
        for i in range(1, len(trending_down_data)):
            trending_down_data.iloc[i, trending_down_data.columns.get_loc('Close')] = trending_down_data.iloc[i-1]['Close'] * 0.99
        downtrend_result = self.indicators.compute_momentum_signals(trending_down_data)
        # RSI should be higher during strong uptrend than strong downtrend
        self.assertGreater(
            uptrend_result['rsi'],
            downtrend_result['rsi'],
            "RSI should be higher during uptrend than downtrend"
        )
        # Uptrend RSI should indicate overbought conditions (> 60)
        self.assertGreater(uptrend_result['rsi'], 60.0, "Strong uptrend RSI should be > 60")
        # Downtrend RSI should indicate oversold conditions (< 40)
        self.assertLess(downtrend_result['rsi'], 40.0, "Strong downtrend RSI should be < 40")
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data returns NaN."""
        # Use only first 10 rows (insufficient for RSI calculation)
        insufficient_data = self.test_data.iloc[:10]
        result = self.indicators.compute_momentum_signals(insufficient_data)
        # Should return NaN for insufficient data
        self.assertTrue(math.isnan(result['rsi']), "RSI should be NaN with insufficient data")
    def test_atr_calculation_basic(self):
        """Test ATR calculation with known data returns expected range."""
        result = self.indicators.compute_volatility_signals(self.test_data)
        # ATR should be positive
        self.assertGreater(result['atr'], 0.0, "ATR should be positive")
        # ATR should not be NaN with sufficient data
        self.assertFalse(math.isnan(result['atr']), "ATR should not be NaN with sufficient data")
        # ATR percentage should be reasonable (0-20% typically)
        self.assertGreater(result['atr_pct'], 0.0, "ATR percentage should be positive")
        self.assertLess(result['atr_pct'], 20.0, "ATR percentage should be reasonable (< 20%)")
    def test_atr_calculation_deterministic(self):
        """Test ATR calculation produces consistent results with deterministic data."""
        # Run calculation twice to ensure consistency
        result1 = self.indicators.compute_volatility_signals(self.test_data)
        result2 = self.indicators.compute_volatility_signals(self.test_data)
        # Results should be identical
        self.assertEqual(result1['atr'], result2['atr'], "ATR calculation should be deterministic")
        self.assertEqual(result1['atr_pct'], result2['atr_pct'], "ATR percentage should be deterministic")
        self.assertEqual(result1['atr_trend'], result2['atr_trend'], "ATR trend should be deterministic")
    def test_atr_with_volatility_changes(self):
        """Test ATR responds appropriately to volatility changes."""
        # Early data has smaller ranges (less volatile)
        low_vol_data = self.test_data.iloc[:20]
        low_vol_result = self.indicators.compute_volatility_signals(low_vol_data)
        # Later data has larger ranges (more volatile)
        # Create high volatility data by using the original but with wider ranges
        high_vol_data = self.test_data.copy()
        high_vol_data['High'] = high_vol_data['High'] * 1.05  # Increase highs by 5%
        high_vol_data['Low'] = high_vol_data['Low'] * 0.95   # Decrease lows by 5%
        high_vol_result = self.indicators.compute_volatility_signals(high_vol_data)
        # High volatility data should have higher ATR
        self.assertGreater(
            high_vol_result['atr'],
            low_vol_result['atr'],
            "High volatility data should have higher ATR"
        )
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data returns NaN."""
        # Use only first 10 rows (insufficient for ATR calculation)
        insufficient_data = self.test_data.iloc[:10]
        result = self.indicators.compute_volatility_signals(insufficient_data)
        # Should return NaN for insufficient data
        self.assertTrue(math.isnan(result['atr']), "ATR should be NaN with insufficient data")
    def test_atr_true_range_calculation(self):
        """Test that ATR calculation includes all True Range components."""
        # Create data with specific True Range scenarios
        tr_test_data = pd.DataFrame({
            'High': [100, 105, 98, 102],
            'Low': [95, 102, 95, 99],
            'Close': [98, 103, 96, 101]
        })
        tr_test_data.index = pd.date_range('2024-01-01', periods=4, freq='D')
        # Need more data for ATR calculation
        extended_data = pd.concat([self.test_data.iloc[:16], tr_test_data])
        result = self.indicators.compute_volatility_signals(extended_data)
        # ATR should reflect the volatility in the data
        self.assertGreater(result['atr'], 0.0, "ATR should capture True Range volatility")
    def test_rsi_extreme_values(self):
        """Test RSI calculation with extreme market conditions."""
        # Create extremely bullish data (all gains)
        bullish_data = self.test_data.copy()
        for i in range(1, len(bullish_data)):
            bullish_data.loc[bullish_data.index[i], 'Close'] = bullish_data.iloc[i-1]['Close'] * 1.02
        bullish_result = self.indicators.compute_momentum_signals(bullish_data)
        # RSI should be very high (near 100) for extreme bullish scenario
        self.assertGreater(bullish_result['rsi'], 80.0, "Extreme bullish RSI should be > 80")
        # Create extremely bearish data (all losses)
        bearish_data = self.test_data.copy()
        for i in range(1, len(bearish_data)):
            bearish_data.loc[bearish_data.index[i], 'Close'] = bearish_data.iloc[i-1]['Close'] * 0.98
        bearish_result = self.indicators.compute_momentum_signals(bearish_data)
        # RSI should be very low (near 0) for extreme bearish scenario
        self.assertLess(bearish_result['rsi'], 20.0, "Extreme bearish RSI should be < 20")
    def test_macd_included_in_momentum_signals(self):
        """Test that MACD components are included in momentum signals."""
        result = self.indicators.compute_momentum_signals(self.test_data)
        # Check all momentum components are present
        required_keys = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'macd_strength', 'rsi_bullish', 'macd_bullish']
        for key in required_keys:
            self.assertIn(key, result, f"Momentum signals should include {key}")
        # Check boolean flags are actually boolean (handle numpy.bool_)
        self.assertTrue(isinstance(result['rsi_bullish'], (bool, np.bool_)), "RSI bullish should be boolean type")
        self.assertTrue(isinstance(result['macd_bullish'], (bool, np.bool_)), "MACD bullish should be boolean type")
    def test_indicators_integration(self):
        """Test that RSI and ATR work together in the full indicators system."""
        # Test that both functions can be called together without conflicts
        momentum_result = self.indicators.compute_momentum_signals(self.test_data)
        volatility_result = self.indicators.compute_volatility_signals(self.test_data)
        # Both should return valid results
        self.assertFalse(math.isnan(momentum_result['rsi']), "RSI should be valid")
        self.assertFalse(math.isnan(volatility_result['atr']), "ATR should be valid")
        # Results should be independent (RSI not affected by ATR call)
        momentum_result_2 = self.indicators.compute_momentum_signals(self.test_data)
        self.assertEqual(momentum_result['rsi'], momentum_result_2['rsi'], "RSI should be independent")
if __name__ == '__main__':
    # Set up logging to avoid noise during tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    # Run the tests
    unittest.main(verbosity=2)
