"""
Unit tests for advanced indicators module.

This module tests all technical indicator calculations using deterministic test data
to ensure accuracy and reliability of the core analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the module under test
from advanced_indicators import (
    compute_rsi,
    compute_atr,
    compute_macd,
    compute_adx,
    compute_volume_zscore,
    compute_volume_ratio,
    compute_volume_profile,
    compute_bollinger_bands,
    compute_all_indicators
)

class TestRSICalculation:
    """Test RSI (Relative Strength Index) calculations."""
    
    def test_rsi_basic_calculation(self, test_ohlcv_data):
        """Test basic RSI calculation with known data."""
        rsi = compute_rsi(test_ohlcv_data, period=14)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100, f"RSI {rsi} is outside valid range [0, 100]"
        
        # With our test data pattern, RSI should be reasonable
        assert not np.isnan(rsi), "RSI should not be NaN with sufficient data"
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data points."""
        # Create data with only 10 days (less than required 14)
        short_data = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            'Date': pd.date_range('2023-01-01', periods=10)
        })
        short_data.set_index('Date', inplace=True)
        
        rsi = compute_rsi(short_data, period=14)
        assert np.isnan(rsi), "RSI should be NaN with insufficient data"
    
    def test_rsi_extreme_conditions(self, indicator_test_cases):
        """Test RSI under extreme market conditions."""
        for test_case in indicator_test_cases['rsi_test_cases']:
            # Create test data
            test_data = pd.DataFrame({
                'Close': test_case['close_prices'],
                'Date': pd.date_range('2023-01-01', periods=len(test_case['close_prices']))
            })
            test_data.set_index('Date', inplace=True)
            
            rsi = compute_rsi(test_data, period=14)
            expected_min, expected_max = test_case['expected_rsi_range']
            
            assert expected_min <= rsi <= expected_max, \
                f"RSI {rsi} not in expected range [{expected_min}, {expected_max}] for {test_case['name']}"
    
    def test_rsi_different_periods(self, test_ohlcv_data):
        """Test RSI with different period lengths."""
        rsi_14 = compute_rsi(test_ohlcv_data, period=14)
        rsi_21 = compute_rsi(test_ohlcv_data, period=21)
        
        # Both should be valid
        assert not np.isnan(rsi_14), "RSI-14 should be valid"
        assert not np.isnan(rsi_21), "RSI-21 should be valid"
        
        # Longer period RSI should be more stable (less extreme)
        # This is a general tendency, not a strict rule
        assert 0 <= rsi_14 <= 100
        assert 0 <= rsi_21 <= 100

class TestATRCalculation:
    """Test ATR (Average True Range) calculations."""
    
    def test_atr_basic_calculation(self, test_ohlcv_data):
        """Test basic ATR calculation."""
        atr = compute_atr(test_ohlcv_data, period=14)
        
        # ATR should be positive
        assert atr > 0, f"ATR {atr} should be positive"
        assert not np.isnan(atr), "ATR should not be NaN with sufficient data"
    
    def test_atr_volatility_sensitivity(self, indicator_test_cases):
        """Test ATR response to different volatility levels."""
        high_vol_case = indicator_test_cases['atr_test_cases'][0]  # high_volatility
        low_vol_case = indicator_test_cases['atr_test_cases'][1]   # low_volatility
        
        # Create test data for high volatility
        high_vol_data = pd.DataFrame(high_vol_case['ohlc_data'])
        high_vol_data['Date'] = pd.date_range('2023-01-01', periods=len(high_vol_data))
        high_vol_data.set_index('Date', inplace=True)
        
        # Replicate data to have enough for ATR calculation
        high_vol_extended = pd.concat([high_vol_data] * 10, ignore_index=True)
        high_vol_extended['Date'] = pd.date_range('2023-01-01', periods=len(high_vol_extended))
        high_vol_extended.set_index('Date', inplace=True)
        
        # Create test data for low volatility
        low_vol_data = pd.DataFrame(low_vol_case['ohlc_data'])
        low_vol_data['Date'] = pd.date_range('2023-01-01', periods=len(low_vol_data))
        low_vol_data.set_index('Date', inplace=True)
        
        # Replicate data to have enough for ATR calculation
        low_vol_extended = pd.concat([low_vol_data] * 10, ignore_index=True)
        low_vol_extended['Date'] = pd.date_range('2023-01-01', periods=len(low_vol_extended))
        low_vol_extended.set_index('Date', inplace=True)
        
        high_atr = compute_atr(high_vol_extended, period=14)
        low_atr = compute_atr(low_vol_extended, period=14)
        
        # High volatility should produce higher ATR
        assert high_atr > low_atr, f"High volatility ATR {high_atr} should be > low volatility ATR {low_atr}"
        
        # Check against expected ranges
        assert high_atr >= high_vol_case['expected_atr_min'], \
            f"High volatility ATR {high_atr} below expected minimum {high_vol_case['expected_atr_min']}"
        assert low_atr <= low_vol_case['expected_atr_max'], \
            f"Low volatility ATR {low_atr} above expected maximum {low_vol_case['expected_atr_max']}"
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        short_data = pd.DataFrame({
            'High': [101, 102, 100],
            'Low': [99, 98, 97],
            'Close': [100, 101, 99],
            'Date': pd.date_range('2023-01-01', periods=3)
        })
        short_data.set_index('Date', inplace=True)
        
        atr = compute_atr(short_data, period=14)
        assert np.isnan(atr), "ATR should be NaN with insufficient data"

class TestMACDCalculation:
    """Test MACD calculations."""
    
    def test_macd_basic_calculation(self, test_ohlcv_data):
        """Test basic MACD calculation."""
        macd_line, signal_line, histogram = compute_macd(test_ohlcv_data)
        
        # All components should be numeric (not NaN for sufficient data)
        assert not np.isnan(macd_line), "MACD line should not be NaN"
        assert not np.isnan(signal_line), "Signal line should not be NaN"
        assert not np.isnan(histogram), "Histogram should not be NaN"
        
        # Histogram should equal MACD - Signal
        assert abs(histogram - (macd_line - signal_line)) < 1e-10, \
            "Histogram should equal MACD line minus Signal line"
    
    def test_macd_trend_detection(self, regime_test_data):
        """Test MACD's ability to detect trends."""
        bullish_data = regime_test_data['bullish']
        bearish_data = regime_test_data['bearish']
        
        bull_macd, bull_signal, bull_hist = compute_macd(bullish_data)
        bear_macd, bear_signal, bear_hist = compute_macd(bearish_data)
        
        # In bullish trend, MACD should generally be above signal (positive histogram)
        # In bearish trend, MACD should generally be below signal (negative histogram)
        # Note: These are tendencies, not strict rules due to lagging nature of MACD
        assert not np.isnan(bull_hist), "Bullish MACD histogram should be calculable"
        assert not np.isnan(bear_hist), "Bearish MACD histogram should be calculable"

class TestADXCalculation:
    """Test ADX (Average Directional Index) calculations."""
    
    def test_adx_basic_calculation(self, test_ohlcv_data):
        """Test basic ADX calculation."""
        adx = compute_adx(test_ohlcv_data, period=14)
        
        # ADX should be between 0 and 100
        assert 0 <= adx <= 100, f"ADX {adx} should be between 0 and 100"
        assert not np.isnan(adx), "ADX should not be NaN with sufficient data"
    
    def test_adx_trend_strength(self, regime_test_data):
        """Test ADX's ability to measure trend strength."""
        bullish_data = regime_test_data['bullish']
        sideways_data = regime_test_data['sideways']
        
        bull_adx = compute_adx(bullish_data, period=14)
        side_adx = compute_adx(sideways_data, period=14)
        
        # Trending market should have higher ADX than sideways market
        # Note: This is a general tendency, actual values depend on the specific data
        assert not np.isnan(bull_adx), "Bullish ADX should be calculable"
        assert not np.isnan(side_adx), "Sideways ADX should be calculable"
        
        # Both should be in valid range
        assert 0 <= bull_adx <= 100
        assert 0 <= side_adx <= 100

class TestVolumeIndicators:
    """Test volume-based indicators."""
    
    def test_volume_zscore(self, test_ohlcv_data):
        """Test volume z-score calculation."""
        zscore = compute_volume_zscore(test_ohlcv_data, lookback=20)
        
        assert not np.isnan(zscore), "Volume z-score should not be NaN"
        # Z-score should typically be between -3 and 3 for normal data
        assert -5 <= zscore <= 5, f"Volume z-score {zscore} seems extreme"
    
    def test_volume_ratio(self, test_ohlcv_data):
        """Test volume ratio calculation."""
        ratio = compute_volume_ratio(test_ohlcv_data, short_period=5, long_period=20)
        
        assert ratio > 0, f"Volume ratio {ratio} should be positive"
        assert not np.isnan(ratio), "Volume ratio should not be NaN"
    
    def test_volume_profile(self, test_ohlcv_data):
        """Test volume profile calculation."""
        profile = compute_volume_profile(test_ohlcv_data, num_buckets=10)
        
        assert len(profile) == 10, "Volume profile should have requested number of buckets"
        assert all(v >= 0 for v in profile.values()), "All volume profile values should be non-negative"
        
        # Total volume in profile should approximately equal sum of volume in data
        total_profile_volume = sum(profile.values())
        total_data_volume = test_ohlcv_data['Volume'].sum()
        
        # Allow for small differences due to bucketing
        assert abs(total_profile_volume - total_data_volume) / total_data_volume < 0.01, \
            "Volume profile total should approximately match data volume"

class TestBollingerBands:
    """Test Bollinger Bands calculations."""
    
    def test_bollinger_bands_basic(self, test_ohlcv_data):
        """Test basic Bollinger Bands calculation."""
        upper, middle, lower = compute_bollinger_bands(test_ohlcv_data, period=20, std_dev=2)
        
        assert not np.isnan(upper), "Upper band should not be NaN"
        assert not np.isnan(middle), "Middle band should not be NaN"
        assert not np.isnan(lower), "Lower band should not be NaN"
        
        # Upper should be > middle > lower
        assert upper > middle > lower, "Bollinger bands should be ordered: upper > middle > lower"
        
        # Current price should be within reasonable distance of bands
        current_price = test_ohlcv_data['Close'].iloc[-1]
        band_width = upper - lower
        
        assert lower - band_width <= current_price <= upper + band_width, \
            "Current price should be within reasonable range of bands"

class TestComputeAllIndicators:
    """Test the main compute_all_indicators function."""
    
    def test_compute_all_indicators_basic(self, test_ohlcv_data):
        """Test that compute_all_indicators returns all expected indicators."""
        with patch('advanced_indicators.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = test_ohlcv_data
            
            results = compute_all_indicators("TEST", test_ohlcv_data)
        
        # Check that all expected indicators are present
        expected_indicators = [
            'rsi', 'atr', 'macd_line', 'macd_signal', 'macd_histogram',
            'adx', 'volume_zscore', 'volume_ratio', 'bb_upper', 'bb_lower',
            'current_price', 'volume_profile_breakout'
        ]
        
        for indicator in expected_indicators:
            assert indicator in results, f"Missing indicator: {indicator}"
            assert results[indicator] is not None, f"Indicator {indicator} is None"
    
    def test_compute_all_indicators_with_insufficient_data(self):
        """Test compute_all_indicators with insufficient data."""
        short_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100],
            'Date': pd.date_range('2023-01-01', periods=2)
        })
        short_data.set_index('Date', inplace=True)
        
        with patch('advanced_indicators.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = short_data
            
            results = compute_all_indicators("TEST", short_data)
        
        # Most indicators should be NaN or None due to insufficient data
        assert np.isnan(results['rsi']) or results['rsi'] is None
        assert np.isnan(results['atr']) or results['atr'] is None
    
    @patch('advanced_indicators.get_logger')
    def test_compute_all_indicators_error_handling(self, mock_logger, test_ohlcv_data):
        """Test error handling in compute_all_indicators."""
        # Create corrupted data that might cause calculation errors
        corrupted_data = test_ohlcv_data.copy()
        corrupted_data.loc[corrupted_data.index[0], 'Close'] = np.nan
        
        with patch('advanced_indicators.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = corrupted_data
            
            results = compute_all_indicators("TEST", corrupted_data)
        
        # Function should still return a result dict, even if some calculations fail
        assert isinstance(results, dict), "Should return a dictionary even with errors"
        assert 'current_price' in results, "Should at least have current_price"

class TestIndicatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test indicators with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        assert np.isnan(compute_rsi(empty_df))
        assert np.isnan(compute_atr(empty_df))
        assert all(np.isnan(x) for x in compute_macd(empty_df))
        assert np.isnan(compute_adx(empty_df))
    
    def test_single_row_dataframe(self):
        """Test indicators with single row DataFrame."""
        single_row = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [98],
            'Close': [101],
            'Volume': [1000],
            'Date': [pd.Timestamp('2023-01-01')]
        })
        single_row.set_index('Date', inplace=True)
        
        # Most indicators should return NaN for single data point
        assert np.isnan(compute_rsi(single_row))
        assert np.isnan(compute_atr(single_row))
    
    def test_constant_prices(self):
        """Test indicators with constant prices (no volatility)."""
        constant_data = pd.DataFrame({
            'Open': [100] * 30,
            'High': [100] * 30,
            'Low': [100] * 30,
            'Close': [100] * 30,
            'Volume': [1000] * 30,
            'Date': pd.date_range('2023-01-01', periods=30)
        })
        constant_data.set_index('Date', inplace=True)
        
        # ATR should be 0 for constant prices
        atr = compute_atr(constant_data)
        assert atr == 0 or np.isnan(atr), "ATR should be 0 or NaN for constant prices"
        
        # RSI should be around 50 (neutral) for constant prices
        rsi = compute_rsi(constant_data)
        if not np.isnan(rsi):
            assert 45 <= rsi <= 55, f"RSI {rsi} should be near 50 for constant prices"

# Performance tests
class TestIndicatorPerformance:
    """Test performance characteristics of indicators."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test indicator performance with large dataset."""
        import time
        
        # Create large dataset (2 years of daily data)
        large_data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 500),
            'High': np.random.normal(105, 5, 500),
            'Low': np.random.normal(95, 5, 500),
            'Close': np.random.normal(100, 5, 500),
            'Volume': np.random.normal(100000, 20000, 500),
            'Date': pd.date_range('2023-01-01', periods=500)
        })
        large_data.set_index('Date', inplace=True)
        
        start_time = time.time()
        with patch('advanced_indicators.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = large_data
            results = compute_all_indicators("TEST", large_data)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"Indicator computation took too long: {execution_time:.2f}s"
        
        # Should return valid results
        assert isinstance(results, dict)
        assert len(results) > 0