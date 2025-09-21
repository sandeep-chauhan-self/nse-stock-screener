"""
Comprehensive test suite for the enhanced data ingestion system.
Tests all components of the robust data fetching infrastructure:
- Data fetchers with retry logic
- Caching system
- Data validation
- Corporate action handling
- Integration with existing modules
"""
from pathlib import Path
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)
class DataIngestionTester(object):
    """Comprehensive test suite for data ingestion system"""
    def __init__(self) -> None:
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'AAPL', 'MSFT']
        self.test_results = {}
        self.cache_dir = Path("data/cache/test")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    def run_all_tests(self) -> tuple[int, int]:
        """Run all data ingestion tests"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE DATA INGESTION TESTS")
        logger.info("=" * 60)
        test_methods = [
            ('test_enhanced_yfinance_fetcher', self.test_enhanced_yfinance_fetcher),
            ('test_nse_symbol_fetching', self.test_nse_symbol_fetching),
            ('test_caching_system', self.test_caching_system),
            ('test_data_validation', self.test_data_validation),
            ('test_corporate_action_handling', self.test_corporate_action_handling),
            ('test_retry_logic', self.test_retry_logic),
            ('test_rate_limiting', self.test_rate_limiting),
            ('test_integration_with_indicators', self.test_integration_with_indicators)
        ]
        return self._execute_test_methods(test_methods)
    def _execute_test_methods(self, test_methods: list[tuple[str, callable]]) -> tuple[int, int]:
        """Execute test methods and track results."""
        passed = 0
        failed = 0
        for test_name, test_method in test_methods:
            try:
                logger.info(f"\n🧪 Running {test_name}...")
                success = test_method()
                if success:
                    logger.info(f"✅ {test_name} PASSED")
                    self.test_results[test_name] = "PASSED"
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    self.test_results[test_name] = "FAILED"
                    failed += 1
            except Exception as e:
                logger.error(f"💥 {test_name} CRASHED: {e}")
                self.test_results[test_name] = f"CRASHED: {e}"
                failed += 1
        self._print_test_summary(passed, failed)
        return passed, failed
    def _print_test_summary(self, passed: int, failed: int) -> None:
        """Print test execution summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_emoji} {test_name}: {result}")
        total_tests = passed + failed
        if total_tests > 0:
            success_rate = passed / total_tests * 100
            logger.info(f"\nTotal: {total_tests} tests")
            logger.info(f"Passed: {passed}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
    @staticmethod
    def test_enhanced_yfinance_fetcher() -> bool:
        """Test enhanced Yahoo Finance fetcher"""
        success = False
        try:
            from src.data.compat import enhanced_yfinance as yf
            # Test single symbol fetch
            ticker = yf.Ticker('RELIANCE.NS')
            data = ticker.history(period="1mo", auto_adjust=True)
            if not data.empty:
                logger.info(f"✓ Fetched {len(data)} rows for RELIANCE.NS")
                # Test that data has required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if not missing_cols:
                    logger.info("✓ All required columns present")
                    # Test symbol info
                    info = ticker.info
                    if info and 'symbol' in info:
                        logger.info(f"✓ Symbol info retrieved: {info.get('shortName', 'Unknown')}")
                        success = True
                    else:
                        logger.error("Failed to get symbol info")
                else:
                    logger.error(f"Missing required columns: {missing_cols}")
            else:
                logger.error("Enhanced yfinance returned empty data")
        except Exception as e:
            logger.error(f"Enhanced yfinance test failed: {e}")
        return success
    @staticmethod
    def test_nse_symbol_fetching() -> bool:
        """Test NSE symbol fetching with enhanced data layer"""
        success = False
        try:
            from src.data.compat import get_nse_symbols
            # Test fetching NSE symbols
            symbols = get_nse_symbols(force_refresh=False)
            if symbols:
                logger.info(f"✓ Fetched {len(symbols)} NSE symbols")
                # Validate some symbols
                valid_symbols = [s for s in symbols[:10] if s and len(s) > 0]
                if len(valid_symbols) >= 5:
                    logger.info(f"✓ Sample symbols: {valid_symbols[:5]}")
                    success = True
                else:
                    logger.error("Too few valid symbols in result")
            else:
                logger.error("No NSE symbols fetched")
        except Exception as e:
            logger.error(f"NSE symbol fetching test failed: {e}")
        return success
    def test_caching_system(self) -> bool:
        """Test caching system functionality"""
        success = False
        try:
            from src.data.cache import DataCache
            # Create test cache
            cache = DataCache(self.cache_dir / "test_cache")
            # Test storing and retrieving data
            test_data = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })
            # Store data
            store_success = cache.set("test_key", test_data, ttl_hours=1)
            if store_success:
                logger.info("✓ Data stored in cache")
                # Retrieve data
                retrieved_data = cache.get("test_key")
                if retrieved_data is not None:
                    logger.info("✓ Data retrieved from cache")
                    # Verify data integrity
                    if test_data.equals(retrieved_data):
                        logger.info("✓ Data integrity verified")
                        # Test cache stats
                        stats = cache.get_cache_stats()
                        if stats['total_entries'] >= 1:
                            logger.info(f"✓ Cache stats: {stats['total_entries']} entries, {stats['total_size_mb']} MB")
                            success = True
                        else:
                            logger.error("Cache stats show no entries")
                    else:
                        logger.error("Retrieved data doesn't match stored data")
                else:
                    logger.error("Failed to retrieve data from cache")
            else:
                logger.error("Failed to store data in cache")
            # Cleanup
            try:
                cache.delete("test_key")
            except Exception:
                # Ignore cleanup errors
                pass
        except Exception as e:
            logger.error(f"Caching system test failed: {e}")
        return success
    @staticmethod
    def test_data_validation() -> bool:
        """Test data validation system"""
        success = False
        try:
            from src.data.validation import validate_stock_data
            from src.data.compat import enhanced_yfinance as yf
            # Get some real data for validation
            ticker = yf.Ticker('TCS.NS')
            data = ticker.history(period="1mo")
            if not data.empty:
                # Validate the data
                result = validate_stock_data(data, 'TCS.NS')
                logger.info(f"✓ Validation result: Valid={result.is_valid}")
                logger.info(f"✓ Errors: {len(result.errors)}")
                logger.info(f"✓ Warnings: {len(result.warnings)}")
                logger.info(f"✓ Quality score: {result.metadata.get('quality_score', 'N/A')}")
                # Should have some basic metadata
                if 'rows' in result.metadata:
                    logger.info(f"✓ Validated {result.metadata['rows']} rows of data")
                    success = True
                else:
                    logger.error("Missing rows metadata")
            else:
                logger.error("No data for validation test")
        except Exception as e:
            logger.error(f"Data validation test failed: {e}")
        return success
    @staticmethod
    def test_corporate_action_handling() -> bool:
        """Test corporate action detection and adjustment"""
        success = False
        try:
            from src.data.validation import adjust_for_corporate_actions
            from src.data.compat import enhanced_yfinance as yf
            # Get data for a stock that might have corporate actions
            # Apple often has splits
            ticker = yf.Ticker('AAPL')
            # Get raw data
            data = ticker.history(period="2y", auto_adjust=False)
            if not data.empty:
                # Apply corporate action adjustments
                adjusted_data = adjust_for_corporate_actions(data)
                if not adjusted_data.empty:
                    logger.info(f"✓ Applied corporate action adjustments to {len(adjusted_data)} rows")
                    # Check if adjustments were applied (if there's Adj Close)
                    if 'Adj Close' in data.columns and 'Close' in data.columns:
                        # There should be some difference if adjustments were applied
                        close_diff = (data['Close'] - adjusted_data['Close']).abs().max()
                        logger.info(f"✓ Max price adjustment: {close_diff:.2f}")
                    success = True
                else:
                    logger.error("Corporate action adjustment returned empty data")
            else:
                logger.error("No data for corporate action test")
        except Exception as e:
            logger.error(f"Corporate action handling test failed: {e}")
        return success
    @staticmethod
    def test_retry_logic() -> bool:
        """Test retry logic by simulating failures"""
        success = False
        try:
            from src.data.fetchers import retry_with_backoff, FetchConfig
            # Create a function that fails a few times then succeeds
            attempt_count = 0
            @retry_with_backoff(FetchConfig(max_retries=3, base_delay=0.1))
            def flaky_function() -> str:
                """Simulates a flaky function that fails first few times."""
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise RuntimeError(f"Simulated failure #{attempt_count}")
                return "Success!"
            # Test that it eventually succeeds
            start_time = time.time()
            result = flaky_function()
            end_time = time.time()
            if result == "Success!" and attempt_count == 3:
                duration = end_time - start_time
                logger.info(f"✓ Retry logic worked: {attempt_count} attempts in {duration:.2f}s")
                success = True
            elif result != "Success!":
                logger.error("Retry logic didn't return expected result")
            else:
                logger.error(f"Expected 3 attempts, got {attempt_count}")
        except Exception as e:
            logger.error(f"Retry logic test failed: {e}")
        return success
    @staticmethod
    def test_rate_limiting() -> bool:
        """Test rate limiting functionality"""
        success = False
        try:
            from src.data.fetchers import RateLimiter
            # Create rate limiter with short interval
            limiter = RateLimiter(min_interval=0.1)
            # Make rapid requests and measure timing
            times = []
            for _ in range(3):
                start_time = time.time()
                limiter.wait_if_needed()
                end_time = time.time()
                times.append(end_time - start_time)
            # First should be quick
            if times[0] > 0.05:
                logger.error("First request was delayed unexpectedly")
            # Subsequent should be delayed
            elif times[1] < 0.05 or times[2] < 0.05:
                logger.error("Rate limiting not working properly")
            else:
                logger.info(f"✓ Rate limiting working: delays {times[1:]}")
                success = True
        except Exception as e:
            logger.error(f"Rate limiting test failed: {e}")
        return success
    @staticmethod
    def test_integration_with_indicators() -> bool:
        """Test integration with existing indicator system"""
        success = False
        try:
            from src.advanced_indicators import AdvancedIndicator
            # Create indicator engine
            engine = AdvancedIndicator()
            # Test computing indicators (this should use our enhanced data layer)
            indicators = engine.compute_all_indicators('RELIANCE.NS', period="3mo")
            if indicators is None:
                logger.error("Indicator computation returned None")
            else:
                # Check for key indicators
                expected_indicators = ['rsi', 'macd', 'vol_ratio', 'adx']
                missing_indicators = [ind for ind in expected_indicators if ind not in indicators]
                if missing_indicators:
                    logger.error(f"Missing indicators: {missing_indicators}")
                else:
                    logger.info(f"✓ Computed {len(indicators)} indicators")
                    logger.info(f"✓ Sample indicators: RSI={indicators.get('rsi')}, MACD={indicators.get('macd')}")
                    success = True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
        return success
def main() -> int:
    """Run the comprehensive test suite"""
    print("🚀 Starting Enhanced Data Ingestion Test Suite")
    print("=" * 60)
    tester = DataIngestionTester()
    passed, failed = tester.run_all_tests()
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced data ingestion system is working perfectly.")
    elif passed > failed:
        print(f"⚠️  MOSTLY SUCCESSFUL: {passed} passed, {failed} failed")
        print("The enhanced data ingestion system is mostly working but needs attention.")
    else:
        print(f"❌ SIGNIFICANT ISSUES: {passed} passed, {failed} failed")
        logging.debug("The enhanced data ingestion system needs debugging.")
    return 0 if failed == 0 else 1
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
