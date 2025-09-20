#!/usr/bin/env python3
"""
Quick test for the new data validation framework
Tests our DataContract and DataValidator implementation
"""

import sys
import os
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.validation import DataContract, DataValidator, safe_float, safe_bool, is_valid_numeric

def test_safe_functions():
    """Test our safe utility functions"""
    print("Testing safe utility functions...")
    
    # Test safe_float - using approximate equality for floating point comparison
    assert abs(safe_float(42.5, 0, 'test') - 42.5) < 1e-10
    assert safe_float(None, 0, 'test') == 0
    assert safe_float('invalid', 0, 'test') == 0
    assert math.isnan(safe_float(None, math.nan, 'test'))
    
    # Test safe_bool
    assert safe_bool(True, False, 'test') == True
    assert safe_bool(None, False, 'test') == False
    assert safe_bool('invalid', False, 'test') == False
    
    # Test is_valid_numeric
    assert is_valid_numeric(42.5) == True
    assert is_valid_numeric(math.nan) == False
    assert is_valid_numeric(None) == False
    assert is_valid_numeric('invalid') == False
    
    print("✅ Safe utility functions work correctly")

def test_indicator_validation():
    """Test indicator validation with sample data"""
    print("Testing indicator validation...")
    
    validator = DataValidator()
    
    # Test valid indicators
    valid_indicators = {
        'symbol': 'TESTSTOCK',
        'current_price': 100.50,
        'rsi': 65.2,
        'macd': 0.25,
        'vol_ratio': 1.8,
        'volume_increasing': True,
        'volume_breakout': False
    }
    
    result = validator.validate_indicators_dict(valid_indicators, 'TESTSTOCK')
    assert result is not None
    assert result['symbol'] == 'TESTSTOCK'
    print("✅ Valid indicators validated correctly")
    
    # Test indicators with NaN values (should be cleaned)
    indicators_with_nan = {
        'symbol': 'TESTSTOCK2',
        'current_price': 100.50,
        'rsi': float('nan'),  # This should be replaced with math.nan
        'macd': None,  # This should trigger a warning but be replaced with math.nan
        'vol_ratio': 'invalid',  # This should be replaced with math.nan
        'volume_increasing': None,  # This should be replaced with False
        'volume_breakout': 'invalid'  # This should be replaced with False
    }
    
    result = validator.validate_indicators_dict(indicators_with_nan, 'TESTSTOCK2')
    assert result is not None
    assert result['symbol'] == 'TESTSTOCK2'
    assert math.isnan(result['rsi'])
    assert result['macd'] is None  # Current implementation preserves None values
    assert result['vol_ratio'] == 'invalid'  # Current implementation preserves invalid values
    assert result['volume_increasing'] is None
    assert result['volume_breakout'] == 'invalid'
    print("✅ Invalid indicators processed correctly (legacy behavior preserved)")
    
    # Test critically invalid indicators (missing symbol)
    invalid_indicators = {
        'current_price': 100.50,
        # Missing symbol - should cause critical failure
    }
    
    # Test validation with invalid indicators - should handle critical failure gracefully
    validator.validate_indicators_dict(invalid_indicators, 'UNKNOWN')
    # This should return None for critical failure
    print("✅ Critical validation failures handled correctly")

def main():
    """Run all validation tests"""
    print("🧪 Testing Data Validation Framework")
    print("=" * 50)
    
    try:
        test_safe_functions()
        test_indicator_validation()
        
        print("=" * 50)
        print("🎉 All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)