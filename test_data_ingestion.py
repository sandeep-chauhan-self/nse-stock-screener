"""
Comprehensive test suite for the enhanced data ingestion system.

Tests all components of the robust data fetching infrastructure:
- Data fetchers with retry logic
- Caching system 
- Data validation
- Corporate action handling
- Integration with existing modules
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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


class DataIngestionTester:
    """Comprehensive test suite for data ingestion system"""
    
    def __init__(self):
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'AAPL', 'MSFT']
        self.test_results = {}
        self.cache_dir = Path("data/cache/test")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_tests(self):
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
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_emoji} {test_name}: {result}")
        
        logger.info(f"\nTotal: {passed + failed} tests")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        return passed, failed
    
    def test_enhanced_yfinance_fetcher(self):
        """Test enhanced Yahoo Finance fetcher"""
        try:
            from src.data.compat import enhanced_yfinance as yf
            
            # Test single symbol fetch
            ticker = yf.Ticker('RELIANCE.NS')
            data = ticker.history(period="1mo", auto_adjust=True)
            
            if data.empty:
                logger.error("Enhanced yfinance returned empty data")
                return False
            
            logger.info(f"✓ Fetched {len(data)} rows for RELIANCE.NS")
            
            # Test that data has required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            logger.info("✓ All required columns present")
            
            # Test symbol info
            info = ticker.info
            if not info or 'symbol' not in info:
                logger.error("Failed to get symbol info")
                return False
            
            logger.info(f"✓ Symbol info retrieved: {info.get('shortName', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced yfinance test failed: {e}")
            return False
    
    def test_nse_symbol_fetching(self):
        """Test NSE symbol fetching with enhanced data layer"""
        try:
            from src.data.compat import get_nse_symbols
            
            # Test fetching NSE symbols
            symbols = get_nse_symbols(force_refresh=False)
            
            if not symbols:
                logger.error("No NSE symbols fetched")
                return False
            
            logger.info(f"✓ Fetched {len(symbols)} NSE symbols")
            
            # Validate some symbols
            valid_symbols = [s for s in symbols[:10] if s and len(s) > 0]
            
            if len(valid_symbols) < 5:
                logger.error("Too few valid symbols in result")
                return False
            
            logger.info(f"✓ Sample symbols: {valid_symbols[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f"NSE symbol fetching test failed: {e}")
            return False
    
    def test_caching_system(self):
        """Test caching system functionality"""
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
            success = cache.set("test_key", test_data, ttl_hours=1)
            if not success:
                logger.error("Failed to store data in cache")
                return False
            
            logger.info("✓ Data stored in cache")
            
            # Retrieve data
            retrieved_data = cache.get("test_key")
            if retrieved_data is None:
                logger.error("Failed to retrieve data from cache")
                return False
            
            logger.info("✓ Data retrieved from cache")
            
            # Verify data integrity
            if not test_data.equals(retrieved_data):
                logger.error("Retrieved data doesn't match stored data")
                return False
            
            logger.info("✓ Data integrity verified")
            
            # Test cache stats
            stats = cache.get_cache_stats()
            if stats['total_entries'] < 1:
                logger.error("Cache stats show no entries")
                return False
            
            logger.info(f"✓ Cache stats: {stats['total_entries']} entries, {stats['total_size_mb']} MB")
            
            # Cleanup
            cache.delete("test_key")
            
            return True
            
        except Exception as e:
            logger.error(f"Caching system test failed: {e}")
            return False
    
    def test_data_validation(self):
        """Test data validation system"""
        try:
            from src.data.validation import validate_stock_data
            from src.data.compat import enhanced_yfinance as yf
            
            # Get some real data for validation
            ticker = yf.Ticker('TCS.NS')
            data = ticker.history(period="1mo")
            
            if data.empty:
                logger.error("No data for validation test")
                return False
            
            # Validate the data
            result = validate_stock_data(data, 'TCS.NS')
            
            logger.info(f"✓ Validation result: Valid={result.is_valid}")
            logger.info(f"✓ Errors: {len(result.errors)}")
            logger.info(f"✓ Warnings: {len(result.warnings)}")
            logger.info(f"✓ Quality score: {result.metadata.get('quality_score', 'N/A')}")
            
            # Should have some basic metadata
            if 'rows' not in result.metadata:
                logger.error("Missing rows metadata")
                return False
            
            logger.info(f"✓ Validated {result.metadata['rows']} rows of data")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation test failed: {e}")
            return False
    
    def test_corporate_action_handling(self):
        """Test corporate action detection and adjustment"""
        try:
            from src.data.validation import adjust_for_corporate_actions
            from src.data.compat import enhanced_yfinance as yf
            
            # Get data for a stock that might have corporate actions
            ticker = yf.Ticker('AAPL')  # Apple often has splits
            data = ticker.history(period="2y", auto_adjust=False)  # Get raw data
            
            if data.empty:
                logger.error("No data for corporate action test")
                return False
            
            # Apply corporate action adjustments
            adjusted_data = adjust_for_corporate_actions(data)
            
            if adjusted_data.empty:
                logger.error("Corporate action adjustment returned empty data")
                return False
            
            logger.info(f"✓ Applied corporate action adjustments to {len(adjusted_data)} rows")
            
            # Check if adjustments were applied (if there's Adj Close)
            if 'Adj Close' in data.columns and 'Close' in data.columns:
                # There should be some difference if adjustments were applied
                close_diff = (data['Close'] - adjusted_data['Close']).abs().max()
                logger.info(f"✓ Max price adjustment: {close_diff:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Corporate action handling test failed: {e}")
            return False
    
    def test_retry_logic(self):
        """Test retry logic by simulating failures"""
        try:
            from src.data.fetchers import retry_with_backoff, FetchConfig
            
            # Create a function that fails a few times then succeeds
            attempt_count = 0
            
            @retry_with_backoff(FetchConfig(max_retries=3, base_delay=0.1))
            def flaky_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise Exception(f"Simulated failure #{attempt_count}")
                return "Success!"
            
            # Test that it eventually succeeds
            start_time = time.time()
            result = flaky_function()
            end_time = time.time()
            
            if result != "Success!":
                logger.error("Retry logic didn't return expected result")
                return False
            
            if attempt_count != 3:
                logger.error(f"Expected 3 attempts, got {attempt_count}")
                return False
            
            duration = end_time - start_time
            logger.info(f"✓ Retry logic worked: {attempt_count} attempts in {duration:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Retry logic test failed: {e}")
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        try:
            from src.data.fetchers import RateLimiter
            
            # Create rate limiter with short interval
            limiter = RateLimiter(min_interval=0.1)
            
            # Make rapid requests and measure timing
            times = []
            
            for i in range(3):
                start_time = time.time()
                limiter.wait_if_needed()
                end_time = time.time()
                times.append(end_time - start_time)
            
            # First request should be immediate, subsequent ones should be delayed
            if times[0] > 0.05:  # First should be quick
                logger.error("First request was delayed unexpectedly")
                return False
            
            if times[1] < 0.05 or times[2] < 0.05:  # Subsequent should be delayed
                logger.error("Rate limiting not working properly")
                return False
            
            logger.info(f"✓ Rate limiting working: delays {times[1:]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting test failed: {e}")
            return False
    
    def test_integration_with_indicators(self):
        """Test integration with existing indicator system"""
        try:
            from src.advanced_indicators import AdvancedIndicator
            
            # Create indicator engine
            engine = AdvancedIndicator()
            
            # Test computing indicators (this should use our enhanced data layer)
            indicators = engine.compute_all_indicators('RELIANCE.NS', period="3mo")
            
            if indicators is None:
                logger.error("Indicator computation returned None")
                return False
            
            # Check for key indicators
            expected_indicators = ['rsi', 'macd', 'vol_ratio', 'adx']
            missing_indicators = [ind for ind in expected_indicators if ind not in indicators]
            
            if missing_indicators:
                logger.error(f"Missing indicators: {missing_indicators}")
                return False
            
            logger.info(f"✓ Computed {len(indicators)} indicators")
            logger.info(f"✓ Sample indicators: RSI={indicators.get('rsi')}, MACD={indicators.get('macd')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False


def main():
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
        print("The enhanced data ingestion system needs debugging.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)