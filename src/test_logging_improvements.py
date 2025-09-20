"""
Comprehensive Test Suite for Logging and Error Handling Improvements
Tests Requirement 3.10 implementation: Logging, observability, and error handling
"""

import pytest
import tempfile
import time
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

# Import modules to test
try:
    from src.logging_config import (
        setup_logging, get_logger, metrics, retry_manager, 
        operation_context, with_retry, timed_operation,
        CorrelationIdManager, MetricsCollector, RetryManager
    )
    from src.stock_analysis_monitor import (
        monitor, StockAnalysisMonitor, start_batch_analysis, 
        end_batch_analysis, track_symbol_analysis
    )
    from src.robust_data_fetcher import (
        RobustDataFetcher, DataFetchError, RateLimitError, 
        NetworkError, fetch_stock_data
    )
    from src.error_isolation import (
        BatchProcessor, CircuitBreaker, process_symbols_with_isolation,
        safe_stock_analysis
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Running tests from test directory...")
    import sys
    sys.path.append('..')
    from logging_config import *
    from stock_analysis_monitor import *
    from robust_data_fetcher import *
    from error_isolation import *


class TestLoggingConfiguration:
    """Test logging system configuration and functionality"""
    
    def test_setup_logging_console_only(self):
        """Test basic console logging setup"""
        setup_logging(level="DEBUG", console_output=True, json_format=False)
        
        logger = get_logger("test_logger")
        assert logger.level == logging.DEBUG
        
        # Test log message
        logger.info("Test message")
        assert True  # If no exception, test passes
    
    def test_setup_logging_json_format(self):
        """Test JSON format logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            setup_logging(
                level="INFO", 
                json_format=True, 
                log_file=str(log_file),
                console_output=False
            )
            
            logger = get_logger("test_json")
            logger.info("Test JSON message", extra={'test_field': 'test_value'})
            
            # Verify log file was created and contains JSON
            assert log_file.exists()
            content = log_file.read_text()
            
            # Parse JSON to verify format
            log_entry = json.loads(content.strip())
            assert 'timestamp' in log_entry
            assert 'level' in log_entry
            assert 'message' in log_entry
            assert 'correlation_id' in log_entry
    
    def test_correlation_id_management(self):
        """Test correlation ID functionality"""
        # Test automatic ID generation
        id1 = CorrelationIdManager.get_correlation_id()
        id2 = CorrelationIdManager.get_correlation_id()
        assert id1 == id2  # Same thread should return same ID
        
        # Test manual ID setting
        custom_id = "CUSTOM123"
        CorrelationIdManager.set_correlation_id(custom_id)
        assert CorrelationIdManager.get_correlation_id() == custom_id
        
        # Test ID clearing
        CorrelationIdManager.clear_correlation_id()
        new_id = CorrelationIdManager.get_correlation_id()
        assert new_id != custom_id


class TestMetricsCollection:
    """Test metrics collection and monitoring functionality"""
    
    def setup_method(self):
        """Reset metrics for each test"""
        global metrics
        metrics = MetricsCollector()
    
    def test_counter_metrics(self):
        """Test counter increment functionality"""
        metrics.increment_counter("test_counter", 5)
        metrics.increment_counter("test_counter", 3)
        
        summary = metrics.get_metrics_summary()
        assert summary['metrics']['test_counter']['count'] == 8
    
    def test_duration_metrics(self):
        """Test duration recording"""
        metrics.record_duration("test_operation", 1.5)
        metrics.record_duration("test_operation", 2.5)
        
        summary = metrics.get_metrics_summary()
        test_metric = summary['metrics']['test_operation']
        
        assert test_metric['count'] == 2
        assert test_metric['total_time'] == 4.0
        assert test_metric['avg_time'] == 2.0
    
    def test_error_recording(self):
        """Test error metrics collection"""
        error_details = {
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'symbol': 'TEST'
        }
        
        metrics.record_error("test_errors", error_details)
        
        summary = metrics.get_metrics_summary()
        assert summary['metrics']['test_errors']['errors'] == 1
        assert len(summary['recent_errors']) == 1
        
        recent_error = summary['recent_errors'][0]
        assert recent_error['metric'] == 'test_errors'
        assert recent_error['details'] == error_details
    
    def test_operation_context(self):
        """Test operation context manager"""
        with operation_context("test_operation", test_param="value"):
            time.sleep(0.1)
        
        summary = metrics.get_metrics_summary()
        assert 'test_operation_duration' in summary['metrics']
        assert 'test_operation_completed' in summary['metrics']
    
    def test_operation_context_with_error(self):
        """Test operation context with errors"""
        with pytest.raises(ValueError):
            with operation_context("test_error_operation"):
                raise ValueError("Test error")
        
        summary = metrics.get_metrics_summary()
        assert 'test_error_operation_failures' in summary['metrics']


class TestRetryMechanism:
    """Test retry logic and exponential backoff"""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful operation"""
        call_count = 0
        
        @with_retry("test_operation", max_retries=2)
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_operation()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_decorator_with_retries(self):
        """Test retry decorator with failures then success"""
        call_count = 0
        
        @with_retry("test_retry_operation", max_retries=3)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_operation()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_permanent_failure(self):
        """Test retry decorator with permanent failure"""
        call_count = 0
        
        @with_retry("test_permanent_failure", max_retries=2)
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")  # Non-retryable error
        
        with pytest.raises(ValueError):
            failing_operation()
        
        assert call_count == 1  # Should not retry ValueError
    
    def test_retry_manager_backoff_calculation(self):
        """Test exponential backoff calculation"""
        retry_mgr = RetryManager(base_delay=1.0, max_delay=10.0)
        
        assert retry_mgr.calculate_delay(0) == 1.0
        assert retry_mgr.calculate_delay(1) == 2.0
        assert retry_mgr.calculate_delay(2) == 4.0
        assert retry_mgr.calculate_delay(10) == 10.0  # Capped at max_delay
    
    def test_retryable_error_detection(self):
        """Test retryable error classification"""
        retry_mgr = RetryManager()
        
        # Retryable errors
        assert retry_mgr.is_retryable_error(ConnectionError("Network issue"))
        assert retry_mgr.is_retryable_error(TimeoutError("Request timeout"))
        assert retry_mgr.is_retryable_error(Exception("Rate limit exceeded"))
        
        # Non-retryable errors
        assert not retry_mgr.is_retryable_error(ValueError("Invalid input"))
        assert not retry_mgr.is_retryable_error(KeyError("Missing key"))


class TestStockAnalysisMonitor:
    """Test stock analysis monitoring functionality"""
    
    def setup_method(self):
        """Reset monitor for each test"""
        global monitor
        monitor = StockAnalysisMonitor()
    
    def test_batch_session_tracking(self):
        """Test batch session lifecycle"""
        symbols = ["RELIANCE", "TCS", "INFY"]
        session_id = start_batch_analysis(symbols, "test_session")
        
        assert monitor.current_session is not None
        assert monitor.current_session.session_id == session_id
        assert monitor.current_session.symbols_requested == symbols
        
        # Simulate some processing
        time.sleep(0.1)
        
        summary = end_batch_analysis()
        assert summary is not None
        assert summary['session_id'] == session_id
        assert summary['total_duration'] > 0
    
    def test_symbol_analysis_tracking(self):
        """Test individual symbol analysis tracking"""
        symbol = "RELIANCE"
        
        # Start tracking
        symbol_metrics = monitor.start_symbol_analysis(symbol)
        assert symbol_metrics.symbol == symbol
        
        # Record data fetch
        monitor.record_data_fetch(symbol, 0.5, True)
        assert symbol_metrics.data_fetch_time == 0.5
        
        # Record indicator computation
        monitor.record_indicator_computation(symbol, 1.2, True)
        assert symbol_metrics.indicator_compute_time == 1.2
        assert symbol_metrics.indicators_computed == True
        
        # Record scoring
        monitor.record_scoring(symbol, 0.3, True, 75.5)
        assert symbol_metrics.scoring_time == 0.3
        assert symbol_metrics.score_computed == True
        
        # Complete analysis
        monitor.complete_symbol_analysis(symbol, True)
        assert symbol_metrics.was_successful == True
    
    def test_error_tracking(self):
        """Test error recording and tracking"""
        symbol = "TESTFAIL"
        
        monitor.start_symbol_analysis(symbol)
        
        # Record data fetch failure
        monitor.record_data_fetch(symbol, 2.0, False, "NetworkError")
        
        # Record indicator failure
        monitor.record_indicator_computation(symbol, 0.5, False, "InsufficientData")
        
        # Complete as failed
        monitor.complete_symbol_analysis(symbol, False)
        
        # Verify error tracking
        assert monitor.data_fetch_failures["NetworkError"] == 1
        assert monitor.indicator_failures["InsufficientData"] == 1
    
    def test_health_status(self):
        """Test operational health status"""
        health = monitor.get_operational_health()
        
        assert 'timestamp' in health
        assert 'current_session' in health
        assert 'recent_performance' in health
        assert 'error_summary' in health
        assert 'health_indicators' in health
        
        # Health indicators should have boolean values
        indicators = health['health_indicators']
        assert isinstance(indicators['success_rate_healthy'], bool)
        assert isinstance(indicators['performance_healthy'], bool)
        assert isinstance(indicators['error_rate_acceptable'], bool)
        assert isinstance(indicators['overall_healthy'], bool)


class TestRobustDataFetcher:
    """Test robust data fetching with retries and caching"""
    
    def setup_method(self):
        """Setup test data fetcher"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_fetcher = RobustDataFetcher(
            cache_dir=self.temp_dir.name,
            cache_expiry_hours=1,
            max_retries=2,
            request_delay=0.1
        )
    
    def teardown_method(self):
        """Cleanup temporary directory"""
        self.temp_dir.cleanup()
    
    @patch('yfinance.Ticker')
    def test_successful_data_fetch(self, mock_ticker):
        """Test successful data fetching"""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test fetch
        result = self.data_fetcher.fetch_stock_data("RELIANCE", "1mo")
        
        assert result is not None
        assert len(result) == 3
        assert 'Close' in result.columns
    
    @patch('yfinance.Ticker')
    def test_data_validation_failure(self, mock_ticker):
        """Test data validation failure handling"""
        # Mock empty data
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.data_fetcher.fetch_stock_data("INVALID", "1mo")
        assert result is None
    
    def test_cache_functionality(self):
        """Test caching mechanism"""
        cache_key = "TEST_SYMBOL_1mo"
        test_data = pd.DataFrame({'Close': [100, 101, 102]})
        
        # Test cache save and load
        self.data_fetcher._save_to_cache(cache_key, test_data)
        cached_data = self.data_fetcher._load_from_cache(cache_key)
        
        assert cached_data is not None
        assert len(cached_data) == 3
    
    def test_multiple_stocks_error_isolation(self):
        """Test multiple stock fetching with error isolation"""
        symbols = ["VALID1", "INVALID", "VALID2"]
        
        with patch('yfinance.Ticker') as mock_ticker:
            def mock_ticker_side_effect(symbol):
                mock_instance = MagicMock()
                if "VALID" in symbol:
                    mock_instance.history.return_value = pd.DataFrame({
                        'Open': [100], 'High': [105], 'Low': [99], 
                        'Close': [104], 'Volume': [1000]
                    })
                else:
                    mock_instance.history.side_effect = Exception("Mock error")
                return mock_instance
            
            mock_ticker.side_effect = mock_ticker_side_effect
            
            results = self.data_fetcher.fetch_multiple_stocks(symbols, continue_on_error=True)
            
            assert "VALID1" in results
            assert "VALID2" in results
            assert "INVALID" in results
            assert results["VALID1"] is not None
            assert results["INVALID"] is None


class TestErrorIsolation:
    """Test error isolation and batch processing"""
    
    def test_batch_processor_sequential_success(self):
        """Test sequential batch processing with all successes"""
        def mock_processor(item):
            return {'item': item, 'result': f"processed_{item}"}
        
        processor = BatchProcessor(continue_on_error=True)
        items = ["A", "B", "C"]
        
        results = processor.process_items_sequentially(items, mock_processor)
        
        assert len(results['successful']) == 3
        assert len(results['failed']) == 0
        assert results['metadata']['success_rate'] == 1.0
    
    def test_batch_processor_with_failures(self):
        """Test batch processing with some failures"""
        def mock_processor(item):
            if item == "FAIL":
                raise ValueError("Mock failure")
            return {'item': item, 'result': f"processed_{item}"}
        
        processor = BatchProcessor(continue_on_error=True)
        items = ["A", "FAIL", "B"]
        
        results = processor.process_items_sequentially(items, mock_processor)
        
        assert len(results['successful']) == 2
        assert len(results['failed']) == 1
        assert "FAIL" in results['failed']
        assert results['metadata']['success_rate'] == 2/3
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern"""
        circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
        
        def failing_function():
            raise ConnectionError("Always fails")
        
        # Cause failures to trip circuit breaker
        for i in range(3):
            with pytest.raises(ConnectionError):
                circuit_breaker.call(failing_function)
        
        # Should be open now
        assert circuit_breaker.state == 'OPEN'
        
        # Next call should fail with circuit breaker exception
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            circuit_breaker.call(failing_function)
    
    def test_safe_stock_analysis(self):
        """Test safe stock analysis wrapper"""
        def mock_analysis(symbol):
            if symbol == "FAIL":
                raise ConnectionError("Network error")
            return {'symbol': symbol, 'score': 75}
        
        # Successful analysis
        result = safe_stock_analysis(mock_analysis, "RELIANCE")
        assert result is not None
        assert result['symbol'] == "RELIANCE"
        
        # Failed analysis should return None
        result = safe_stock_analysis(mock_analysis, "FAIL")
        assert result is None


class TestIntegrationScenarios:
    """Integration tests for complete logging and error handling workflows"""
    
    def test_complete_stock_analysis_workflow(self):
        """Test complete workflow with logging, monitoring, and error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logging
            log_file = Path(temp_dir) / "integration_test.log"
            setup_logging(
                level="INFO",
                json_format=True,
                log_file=str(log_file),
                console_output=False
            )
            
            # Mock analysis function
            def mock_stock_analysis(symbol):
                logger = get_logger("integration_test")
                
                with operation_context("stock_analysis", symbol=symbol):
                    # Simulate data fetch
                    time.sleep(0.01)
                    logger.info(f"Fetched data for {symbol}")
                    
                    # Simulate indicator computation
                    time.sleep(0.01)
                    logger.info(f"Computed indicators for {symbol}")
                    
                    # Return result
                    return {
                        'symbol': symbol,
                        'score': 75,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Process multiple symbols
            symbols = ["RELIANCE", "TCS", "INFY"]
            session_id = start_batch_analysis(symbols, "integration_test")
            
            results = process_symbols_with_isolation(
                symbols, 
                mock_stock_analysis,
                session_name="integration_test"
            )
            
            session_summary = end_batch_analysis()
            
            # Verify results
            assert len(results['successful']) == 3
            assert len(results['failed']) == 0
            assert session_summary is not None
            
            # Verify logging
            assert log_file.exists()
            log_content = log_file.read_text()
            
            # Should contain multiple JSON log entries
            log_lines = [line for line in log_content.strip().split('\n') if line]
            assert len(log_lines) > 0
            
            # Parse first log entry to verify JSON format
            first_entry = json.loads(log_lines[0])
            assert 'timestamp' in first_entry
            assert 'correlation_id' in first_entry
            assert 'level' in first_entry


def run_comprehensive_test_suite():
    """Run all tests with detailed reporting"""
    
    print("🧪 Running Comprehensive Logging and Error Handling Test Suite")
    print("=" * 70)
    
    test_classes = [
        TestLoggingConfiguration,
        TestMetricsCollection,
        TestRetryMechanism,
        TestStockAnalysisMonitor,
        TestRobustDataFetcher,
        TestErrorIsolation,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}:")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Setup method if exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                getattr(test_instance, test_method)()
                
                # Teardown method if exists
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {len(failed_tests)}")
    print(f"   Success Rate: {passed_tests/total_tests:.1%}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for failure in failed_tests:
            print(f"   - {failure}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)