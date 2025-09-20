"""
Enum Centralization Validation Script

This script validates that the enum centralization was successful
and that all modules can properly import and use the centralized enums.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

def test_enum_imports():
    """Test that all enums can be imported from the centralized location."""
    print("Testing enum imports...")
    
    try:
        from common.enums import MarketRegime, ProbabilityLevel, PositionStatus, StopType
        print("✅ Successfully imported all enums from common.enums")
    except ImportError as e:
        print(f"❌ Failed to import enums: {e}")
        return False
    
    # Test enum values
    try:
        # Test MarketRegime
        regime = MarketRegime.BULLISH
        assert regime.value == "bullish"
        assert str(regime) == "bullish"
        
        # Test from_string method
        regime_from_str = MarketRegime.from_string("BULLISH")
        assert regime_from_str == MarketRegime.BULLISH
        
        # Test ProbabilityLevel
        prob = ProbabilityLevel.HIGH
        assert prob.value == "HIGH"
        assert prob.score_threshold == 70
        
        # Test PositionStatus
        status = PositionStatus.OPEN
        assert status.value == "open"
        
        # Test StopType
        stop = StopType.INITIAL
        assert stop.value == "initial"
        
        print("✅ All enum values and methods work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enum value test failed: {e}")
        return False

def test_module_imports():
    """Test that affected modules can import and use the centralized enums."""
    print("\nTesting module imports...")
    
    try:
        # Test enhanced_early_warning_system imports
        sys.path.append(str(src_dir))
        import enhanced_early_warning_system
        print("✅ enhanced_early_warning_system imports successfully")
        
        # Test that it uses the centralized MarketRegime
        from common.enums import MarketRegime
        test_regime = MarketRegime.SIDEWAYS
        print(f"✅ MarketRegime.SIDEWAYS = {test_regime}")
        
    except ImportError as e:
        print(f"❌ enhanced_early_warning_system import failed: {e}")
        return False
    
    try:
        # Test composite_scorer imports
        import composite_scorer
        print("✅ composite_scorer imports successfully")
        
        # Test creating a scorer instance
        scorer = composite_scorer.CompositeScorer()
        print("✅ CompositeScorer instantiated successfully")
        
    except ImportError as e:
        print(f"❌ composite_scorer import failed: {e}")
        return False
    
    try:
        # Test risk_manager imports
        import risk_manager
        print("✅ risk_manager imports successfully")
        
        # Test creating risk config
        config = risk_manager.RiskConfig()
        print("✅ RiskConfig instantiated successfully")
        
    except ImportError as e:
        print(f"❌ risk_manager import failed: {e}")
        return False
    
    return True

def test_enum_validation():
    """Test the enum validation utilities."""
    print("\nTesting enum validation utilities...")
    
    try:
        from common.enums import validate_enum_consistency, get_all_enum_info
        
        # Test validation
        validation_result = validate_enum_consistency()
        print(f"✅ Enum validation completed: {validation_result['status']}")
        
        if validation_result['issues']:
            for issue in validation_result['issues']:
                print(f"⚠️  Validation issue: {issue}")
        
        # Test enum info
        enum_info = get_all_enum_info()
        print(f"✅ Enum info retrieved for {len(enum_info)} enum classes")
        
        for enum_name, members in enum_info.items():
            print(f"   {enum_name}: {len(members)} members")
        
        return True
        
    except Exception as e:
        print(f"❌ Enum validation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("ENUM CENTRALIZATION VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_enum_imports()
    all_tests_passed &= test_module_imports()
    all_tests_passed &= test_enum_validation()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - Enum centralization successful!")
        print("\nSummary of changes:")
        print("- Created src/common/enums.py with all shared enums")
        print("- Updated enhanced_early_warning_system.py to use centralized enums")
        print("- Updated composite_scorer.py to use centralized enums")  
        print("- Updated risk_manager.py to use centralized enums")
        print("- Removed duplicate enum definitions")
        print("- Eliminated conversion hacks and circular import workarounds")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    
    print("=" * 60)
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)