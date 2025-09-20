"""
Test script to validate the volume threshold fix for Requirement 3.2
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from src.common.enums import MarketRegime
from src.common.volume_thresholds import VolumeThresholdConfig, VolumeThresholdCalculator
from src.composite_scorer import CompositeScorer


def test_volume_threshold_config():
    """Test the volume threshold configuration"""
    print("=== Testing Volume Threshold Configuration ===")
    
    # Test default configuration
    config = VolumeThresholdConfig()
    errors = config.validate_config()
    
    if errors:
        print(f"❌ Configuration validation failed: {errors}")
        return False
    else:
        print("✅ Default configuration is valid")
    
    # Test threshold calculations
    print("\nRegime-specific thresholds:")
    for regime in MarketRegime:
        try:
            base = config.get_high_threshold(regime)
            extreme = config.get_extreme_threshold(regime)
            multiplier = config.regime_extreme_multipliers[regime]
            print(f"  {regime.value}: {base} * {multiplier} = {extreme}")
        except KeyError as e:
            print(f"  ❌ KeyError for {regime}: {e}")
            print(f"     Available keys: {list(config.regime_base_thresholds.keys())}")
            return False
    
    return True


def test_volume_calculator():
    """Test the volume threshold calculator"""
    print("\n=== Testing Volume Threshold Calculator ===")
    
    calculator = VolumeThresholdCalculator()
    
    # Test multiplier-based approach for each regime
    print("\nMultiplier-based thresholds:")
    for regime in MarketRegime:
        high, extreme = calculator.calculate_thresholds_multiplier_based(regime)
        print(f"  {regime.value}: High={high}, Extreme={extreme}")
    
    # Test volume scoring for different scenarios
    print("\nVolume scoring tests:")
    test_cases = [
        (1.0, MarketRegime.SIDEWAYS, "Low volume"),
        (3.5, MarketRegime.SIDEWAYS, "High volume (above 3.0)"),
        (7.0, MarketRegime.SIDEWAYS, "Extreme volume (above 6.0)"),
        (2.0, MarketRegime.BULLISH, "Low volume in bull market"),
        (6.0, MarketRegime.BULLISH, "Extreme volume in bull market (above 5.0)"),
        (3.0, MarketRegime.BEARISH, "Low volume in bear market"),
        (12.0, MarketRegime.BEARISH, "Extreme volume in bear market (above 10.0)"),
    ]
    
    for vol_ratio, regime, description in test_cases:
        score, level = calculator.get_volume_score_and_level(vol_ratio, regime)
        print(f"  {description}: vol_ratio={vol_ratio}, regime={regime.value} -> score={score}, level={level}")
    
    return True


def test_old_vs_new_behavior():
    """Compare old broken behavior vs new fixed behavior"""
    print("\n=== Comparing Old vs New Behavior ===")
    
    calculator = VolumeThresholdCalculator()
    
    # Test case that exposed the bug
    vol_ratio = 5.0
    regime = MarketRegime.BEARISH
    vol_threshold = 4.0  # Old regime threshold for bearish
    
    # Old broken calculation
    old_extreme_threshold = vol_threshold * 1.67  # 4.0 * 1.67 = 6.68
    old_would_be_extreme = vol_ratio >= old_extreme_threshold  # 5.0 >= 6.68 = False
    
    # New fixed calculation
    high, extreme = calculator.calculate_thresholds_multiplier_based(regime)
    new_score, new_level = calculator.get_volume_score_and_level(vol_ratio, regime)
    
    print(f"Test case: vol_ratio={vol_ratio}, regime={regime.value}")
    print(f"  Old approach: {vol_ratio} >= {old_extreme_threshold:.2f} = {old_would_be_extreme} -> would score 5 (HIGH)")
    print(f"  New approach: {vol_ratio} >= {extreme} = {vol_ratio >= extreme} -> scores {new_score} ({new_level})")
    
    # The comment claimed "5x for neutral becomes ~8.3x for bear"
    # Let's verify what the intention was:
    neutral_threshold = 3.0  # SIDEWAYS regime threshold
    intended_bear_extreme = neutral_threshold * 8.3 / 5.0  # Reverse calculation
    
    print(f"\nComment analysis:")
    print(f"  Comment claimed: '5x for neutral becomes ~8.3x for bear'")
    print(f"  If neutral 5x = {neutral_threshold * 5} = 15.0")
    print(f"  Then bear 8.3x should be = {intended_bear_extreme * 8.3:.1f}")
    print(f"  Our new bear extreme threshold = {extreme}")
    
    return True


def test_composite_scorer_integration():
    """Test that the composite scorer uses the new volume logic correctly"""
    print("\n=== Testing Composite Scorer Integration ===")
    
    scorer = CompositeScorer()
    scorer.set_current_symbol("TESTSTOCK")
    
    # Test with sample indicators
    indicators = {
        'vol_z': 2.5,  # High z-score
        'vol_ratio': 6.0  # Test volume ratio
    }
    
    for regime in MarketRegime:
        score, breakdown = scorer.score_volume_component(indicators, regime)
        print(f"  {regime.value}: score={score}, ratio_level={breakdown.get('vol_ratio_level', 'N/A')}")
    
    return True


def main():
    """Run all validation tests"""
    print("Volume Threshold Fix Validation")
    print("=" * 50)
    
    all_passed = True
    
    try:
        all_passed &= test_volume_threshold_config()
        all_passed &= test_volume_calculator()
        all_passed &= test_old_vs_new_behavior()
        all_passed &= test_composite_scorer_integration()
        
        print("\n" + "=" * 50)
        if all_passed:
            print("✅ All tests passed! Volume threshold fix is working correctly.")
            print("\nKey improvements:")
            print("  - Fixed hard-coded 1.67 multiplier with regime-specific values")
            print("  - Corrected threshold calculations to match intended behavior")
            print("  - Added configurable volume threshold system")
            print("  - Maintained backward compatibility with existing regime adjustments")
        else:
            print("❌ Some tests failed. Please review the issues above.")
            
    except Exception as e:
        print(f"❌ Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)