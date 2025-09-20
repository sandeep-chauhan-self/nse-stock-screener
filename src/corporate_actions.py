"""
Corporate Action Handling Module for NSE Stock Screener

This module provides comprehensive corporate action handling to ensure all price data
is properly adjusted for splits, dividends, and other corporate actions.

Requirements 3.9 Implementation: Missing corporate actions & adjusted price handling
- Ensures all return/volatility-based indicators use adjusted prices
- Provides utilities for detecting and handling corporate actions
- Standardizes data fetching across all modules
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CorporateActionHandler:
    """
    Handles corporate actions and provides standardized data fetching
    with proper price adjustments for technical analysis
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes cache for repeated requests
        
    def fetch_adjusted_data(self, symbol: str, period: str = "1y", 
                           interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch price data with corporate action adjustments
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            period: Data period ("1y", "6mo", etc.)
            interval: Data interval ("1d", "1wk", etc.)
            **kwargs: Additional arguments for yf.history()
            
        Returns:
            DataFrame with adjusted OHLCV data, or None if fetch fails
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if (current_time - timestamp).seconds < self._cache_timeout:
                    logger.debug(f"Using cached data for {symbol}")
                    return cached_data
            
            # Fetch fresh data with corporate action adjustments
            logger.debug(f"Fetching adjusted data for {symbol} (period={period}, interval={interval})")
            ticker = yf.Ticker(symbol)
            
            # CRITICAL: Always use auto_adjust=True for corporate action handling
            data = ticker.history(period=period, interval=interval, auto_adjust=True, **kwargs)
            
            if data is None or data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Validate data quality
            if not self._validate_data_quality(data, symbol):
                return None
            
            # Cache the result
            self._cache[cache_key] = (data, current_time)
            
            logger.debug(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_raw_and_adjusted_data(self, symbol: str, period: str = "1y", 
                                   interval: str = "1d") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch both raw and adjusted data for comparison and analysis
        
        Returns:
            Tuple of (raw_data, adjusted_data) DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch raw data
            raw_data = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            # Fetch adjusted data
            adjusted_data = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            return raw_data, adjusted_data
            
        except Exception as e:
            logger.error(f"Error fetching raw/adjusted data for {symbol}: {e}")
            return None, None
    
    def detect_corporate_actions(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """
        Detect corporate actions by comparing raw and adjusted prices
        
        Returns:
            Dictionary with corporate action information
        """
        try:
            raw_data, adjusted_data = self.fetch_raw_and_adjusted_data(symbol, period)
            
            if raw_data is None or adjusted_data is None:
                return {'splits_detected': False, 'dividends_detected': False, 'events': []}
            
            # Calculate adjustment ratios
            raw_close = raw_data['Close']
            adj_close = adjusted_data['Close']
            
            # Find significant discrepancies indicating corporate actions
            adjustment_ratio = raw_close / adj_close
            
            # Detect stock splits (ratio changes significantly)
            ratio_changes = adjustment_ratio.diff().abs()
            split_threshold = 0.1  # 10% change in ratio indicates possible split
            
            splits = ratio_changes > split_threshold
            split_dates = splits[splits].index.tolist()
            
            # Detect dividends (gradual ratio increases)
            dividend_threshold = 0.01  # 1% gradual increase
            dividends = (adjustment_ratio.diff() > dividend_threshold) & (adjustment_ratio.diff() < split_threshold)
            dividend_dates = dividends[dividends].index.tolist()
            
            events = []
            for date in split_dates:
                ratio = adjustment_ratio.loc[date]
                events.append({
                    'date': date,
                    'type': 'split',
                    'ratio': ratio,
                    'description': f"Possible stock split detected (ratio: {ratio:.4f})"
                })
            
            for date in dividend_dates:
                ratio_change = adjustment_ratio.diff().loc[date]
                events.append({
                    'date': date,
                    'type': 'dividend', 
                    'ratio_change': ratio_change,
                    'description': f"Possible dividend detected (ratio change: {ratio_change:.4f})"
                })
            
            return {
                'splits_detected': len(split_dates) > 0,
                'dividends_detected': len(dividend_dates) > 0,
                'events': events,
                'total_events': len(events)
            }
            
        except Exception as e:
            logger.error(f"Error detecting corporate actions for {symbol}: {e}")
            return {'splits_detected': False, 'dividends_detected': False, 'events': []}
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality for corporate action handling
        
        Args:
            data: Price data DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            True if data quality is acceptable
        """
        if data is None or data.empty:
            logger.warning(f"Empty data for {symbol}")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return False
        
        # Check for excessive NaN values
        nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_ratio > 0.1:  # More than 10% NaN values
            logger.warning(f"Excessive NaN values for {symbol}: {nan_ratio:.2%}")
            return False
        
        # Check for price consistency (High >= Low, etc.)
        price_inconsistencies = (data['High'] < data['Low']).sum()
        if price_inconsistencies > 0:
            logger.warning(f"Price inconsistencies for {symbol}: {price_inconsistencies} rows")
            return False
        
        # Check for negative prices
        negative_prices = (data[['Open', 'High', 'Low', 'Close']] <= 0).sum().sum()
        if negative_prices > 0:
            logger.warning(f"Negative prices detected for {symbol}: {negative_prices} values")
            return False
        
        return True
    
    def adjust_price_series(self, raw_prices: pd.Series, adjustment_factor: float) -> pd.Series:
        """
        Manually adjust price series using adjustment factor
        
        Args:
            raw_prices: Raw price series
            adjustment_factor: Adjustment factor for corporate actions
            
        Returns:
            Adjusted price series
        """
        return raw_prices * adjustment_factor
    
    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()
        logger.debug("Corporate action handler cache cleared")

# Decorator for corporate action aware data fetching
def corporate_action_aware(func):
    """
    Decorator to ensure functions use corporate action adjusted data
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Inject corporate action handler if not present
        if 'ca_handler' not in kwargs:
            kwargs['ca_handler'] = CorporateActionHandler()
        return func(*args, **kwargs)
    return wrapper

# Utility functions for common corporate action scenarios
def get_split_adjusted_quantity(original_quantity: int, split_ratio: float) -> int:
    """
    Calculate new quantity after stock split
    
    Args:
        original_quantity: Original number of shares
        split_ratio: Split ratio (e.g., 2.0 for 2:1 split)
        
    Returns:
        New quantity after split
    """
    return int(original_quantity * split_ratio)

def get_dividend_impact(price: float, dividend_per_share: float) -> float:
    """
    Calculate price impact of dividend payment
    
    Args:
        price: Stock price before ex-dividend
        dividend_per_share: Dividend amount per share
        
    Returns:
        Expected price after ex-dividend adjustment
    """
    return price - dividend_per_share

def validate_corporate_action_consistency(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that corporate action adjustments are consistent
    
    Args:
        data: Price data DataFrame
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check for sudden price jumps that might indicate unadjusted splits
        price_changes = data['Close'].pct_change().abs()
        large_jumps = price_changes > 0.5  # 50% single-day changes
        
        if large_jumps.any():
            jump_dates = large_jumps[large_jumps].index.tolist()
            validation_results['warnings'].append({
                'type': 'large_price_jumps',
                'dates': jump_dates,
                'message': f"Large price jumps detected on {len(jump_dates)} dates - possible unadjusted corporate actions"
            })
        
        # Check for volume spikes that might correlate with corporate actions
        if 'Volume' in data.columns:
            volume_changes = data['Volume'].pct_change().abs()
            volume_spikes = volume_changes > 2.0  # 200% volume increases
            
            if volume_spikes.any():
                spike_dates = volume_spikes[volume_spikes].index.tolist()
                validation_results['warnings'].append({
                    'type': 'volume_spikes',
                    'dates': spike_dates,
                    'message': f"Volume spikes detected on {len(spike_dates)} dates - may correlate with corporate actions"
                })
        
        # Overall validation
        if len(validation_results['issues']) > 0:
            validation_results['is_valid'] = False
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['issues'].append({
            'type': 'validation_error',
            'message': f"Error during validation: {e}"
        })
    
    return validation_results

# Global instance for easy access
corporate_action_handler = CorporateActionHandler()

# Example usage and testing
if __name__ == "__main__":
    # Test corporate action handling
    ca_handler = CorporateActionHandler()
    
    # Test with a known stock that has corporate actions
    test_symbol = "RELIANCE.NS"
    print(f"Testing corporate action handling for {test_symbol}...")
    
    # Fetch adjusted data
    adjusted_data = ca_handler.fetch_adjusted_data(test_symbol, period="2y")
    if adjusted_data is not None:
        print(f"Fetched {len(adjusted_data)} adjusted data points")
        
        # Detect corporate actions
        ca_events = ca_handler.detect_corporate_actions(test_symbol, period="2y")
        print(f"Corporate action analysis:")
        print(f"  Splits detected: {ca_events['splits_detected']}")
        print(f"  Dividends detected: {ca_events['dividends_detected']}")
        print(f"  Total events: {ca_events['total_events']}")
        
        if ca_events['events']:
            print(f"  Recent events:")
            for event in ca_events['events'][-3:]:  # Show last 3 events
                print(f"    {event['date'].strftime('%Y-%m-%d')}: {event['description']}")
    
    print("Corporate action handling test completed.")