#!/usr/bin/env python3
"""
Architecture Validation Test Script
Tests the new modular architecture for circular imports and interface compatibility.
"""
import sys
import traceback
from pathlib import Path
def test_imports():
    """Test that all new modular imports work correctly"""
    test_results = {
        "passed": [],
        "failed": [],
        "total_tests": 0
    }
    print("🔍 Testing New Modular Architecture")
    print("=" * 50)
    # Test 1: Core interfaces import
    test_results["total_tests"] += 1
    try:
        from src.common.interfaces import (
            IDataFetcher, IIndicator, IScorer, IRiskManager,
            IBacktester, StockData, IndicatorResult
        )
        print("✅ Core interfaces import successful")
        test_results["passed"].append("Core interfaces")
    except Exception as e:
        print(f"❌ Core interfaces import failed: {e}")
        test_results["failed"].append(f"Core interfaces: {e}")
    # Test 2: Configuration system import
    test_results["total_tests"] += 1
    try:
        from src.common.config import SystemConfig, ConfigManager
        print("✅ Configuration system import successful")
        test_results["passed"].append("Configuration system")
    except Exception as e:
        print(f"❌ Configuration system import failed: {e}")
        test_results["failed"].append(f"Configuration system: {e}")
    # Test 3: Enums import
    test_results["total_tests"] += 1
    try:
        from src.common.enums import MarketRegime, ProbabilityLevel
        print("✅ Enums import successful")
        test_results["passed"].append("Enums")
    except Exception as e:
        print(f"❌ Enums import failed: {e}")
        test_results["failed"].append(f"Enums: {e}")
    # Test 4: Data package import
    test_results["total_tests"] += 1
    try:
        from src.data import YahooDataFetcher, DataFetcherFactory
        print("✅ Data package import successful")
        test_results["passed"].append("Data package")
    except Exception as e:
        print(f"❌ Data package import failed: {e}")
        test_results["failed"].append(f"Data package: {e}")
    # Test 5: Indicators package import
    test_results["total_tests"] += 1
    try:
        from src.indicators import RSIIndicator, IndicatorEngine
        print("✅ Indicators package import successful")
        test_results["passed"].append("Indicators package")
    except Exception as e:
        print(f"❌ Indicators package import failed: {e}")
        test_results["failed"].append(f"Indicators package: {e}")
    # Test 6: Main package import
    test_results["total_tests"] += 1
    try:
        import src
        print("✅ Main package import successful")
        test_results["passed"].append("Main package")
    except Exception as e:
        print(f"❌ Main package import failed: {e}")
        test_results["failed"].append(f"Main package: {e}")
    return test_results
def test_interface_compliance():
    """Test that implementations properly follow interface contracts"""
    test_results = {
        "passed": [],
        "failed": [],
        "total_tests": 0
    }
    print("\n🔧 Testing Interface Compliance")
    print("=" * 50)
    # Test 1: YahooDataFetcher implements IDataFetcher
    test_results["total_tests"] += 1
    try:
        from src.data.fetchers_impl import YahooDataFetcher
        from src.common.interfaces import IDataFetcher
        # Check if YahooDataFetcher has required methods
        required_methods = ['fetch', 'fetch_symbols', 'validate_symbol']
        fetcher = YahooDataFetcher()
        for method in required_methods:
            if not hasattr(fetcher, method):
                raise AttributeError(f"Missing required method: {method}")
        print("✅ YahooDataFetcher implements IDataFetcher correctly")
        test_results["passed"].append("YahooDataFetcher interface compliance")
    except Exception as e:
        print(f"❌ YahooDataFetcher interface compliance failed: {e}")
        test_results["failed"].append(f"YahooDataFetcher interface: {e}")
    # Test 2: RSIIndicator implements IIndicator
    test_results["total_tests"] += 1
    try:
        from src.indicators.technical import RSIIndicator
        from src.common.interfaces import IIndicator
        # Check if RSIIndicator has required methods
        required_methods = ['compute', 'name']
        indicator = RSIIndicator()
        for method in required_methods:
            if method == 'name':
                # Check if it's a property
                if not hasattr(indicator, method):
                    raise AttributeError(f"Missing required property: {method}")
            else:
                # Check if it's a method
                if not hasattr(indicator, method):
                    raise AttributeError(f"Missing required method: {method}")
        print("✅ RSIIndicator implements IIndicator correctly")
        test_results["passed"].append("RSIIndicator interface compliance")
    except Exception as e:
        print(f"❌ RSIIndicator interface compliance failed: {e}")
        test_results["failed"].append(f"RSIIndicator interface: {e}")
    return test_results
def test_configuration_system():
    """Test the centralized configuration system"""
    test_results = {
        "passed": [],
        "failed": [],
        "total_tests": 0
    }
    print("\n⚙️  Testing Configuration System")
    print("=" * 50)
    # Test 1: Configuration creation and validation
    test_results["total_tests"] += 1
    try:
        from src.common.config import SystemConfig, ConfigManager
        # Create default configuration
        config = SystemConfig()
        assert config.data.cache_enabled == True
        assert config.indicators.rsi_period == 14
        print("✅ Configuration creation and validation successful")
        test_results["passed"].append("Configuration creation")
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        test_results["failed"].append(f"Configuration creation: {e}")
    # Test 2: ConfigManager functionality
    test_results["total_tests"] += 1
    try:
        from src.common.config import ConfigManager, get_config
        # Test the global config function
        config = get_config()
        assert config is not None
        # Test ConfigManager creation
        manager = ConfigManager()
        assert manager is not None
        print("✅ ConfigManager functionality successful")
        test_results["passed"].append("ConfigManager functionality")
    except Exception as e:
        print(f"❌ ConfigManager functionality failed: {e}")
        test_results["failed"].append(f"ConfigManager functionality: {e}")
    return test_results
def main():
    """Run all architecture validation tests"""
    print("🏗️  NSE Stock Screener - Architecture Validation")
    print("=" * 60)
    print("Testing new modular architecture implementation...")
    print()
    # Add src to path for testing
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    all_results = {
        "passed": [],
        "failed": [],
        "total_tests": 0
    }
    # Run all test suites
    test_suites = [
        ("Import Tests", test_imports),
        ("Interface Compliance Tests", test_interface_compliance),
        ("Configuration System Tests", test_configuration_system)
    ]
    for suite_name, test_func in test_suites:
        try:
            results = test_func()
            all_results["passed"].extend(results["passed"])
            all_results["failed"].extend(results["failed"])
            all_results["total_tests"] += results["total_tests"]
        except Exception as e:
            print(f"❌ Test suite '{suite_name}' crashed: {e}")
            traceback.print_exc()
            all_results["failed"].append(f"{suite_name}: Suite crashed - {e}")
            all_results["total_tests"] += 1
    # Print summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {all_results['total_tests']}")
    print(f"Passed: {len(all_results['passed'])}")
    print(f"Failed: {len(all_results['failed'])}")
    if all_results["failed"]:
        print(f"\n❌ Failed Tests:")
        for failure in all_results["failed"]:
            print(f"   - {failure}")
    if len(all_results["passed"]) == all_results["total_tests"]:
        print(f"\n🎉 ALL TESTS PASSED! Architecture validation successful.")
        return True
    else:
        success_rate = len(all_results["passed"]) / all_results["total_tests"] * 100
        print(f"\n⚠️  Some tests failed. Success rate: {success_rate:.1f}%")
        return False
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
