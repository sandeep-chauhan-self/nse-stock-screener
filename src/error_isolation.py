"""
Graceful Error Handling and Batch Processing for NSE Stock Screener
Implements per-symbol error isolation and robust batch processing
"""

import time
from typing import List, Dict, Any, Optional, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import traceback

from .logging_config import get_logger, operation_context
from .stock_analysis_monitor import monitor, start_batch_analysis, end_batch_analysis


class BatchProcessor:
    """
    Robust batch processing with error isolation and comprehensive monitoring
    """
    
    def __init__(self, 
                 max_workers: int = 3,
                 continue_on_error: bool = True,
                 timeout_per_item: float = 30.0,
                 progress_callback: Optional[Callable] = None):
        
        self.logger = get_logger(__name__)
        self.max_workers = max_workers
        self.continue_on_error = continue_on_error
        self.timeout_per_item = timeout_per_item
        self.progress_callback = progress_callback
        
        self.logger.info("Initialized BatchProcessor", extra={
            'max_workers': max_workers,
            'continue_on_error': continue_on_error,
            'timeout_per_item': timeout_per_item
        })
    
    def process_items_sequentially(self, 
                                 items: List[str], 
                                 process_func: Callable[[str], Any],
                                 session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process items sequentially with comprehensive error isolation
        
        Args:
            items: List of items to process (e.g., stock symbols)
            process_func: Function to process each item
            session_name: Optional session name for monitoring
            
        Returns:
            Dictionary containing results and processing metadata
        """
        
        session_id = start_batch_analysis(items, session_name)
        
        with operation_context("batch_process_sequential", 
                             item_count=len(items), 
                             session_id=session_id):
            
            results = {
                'successful': {},
                'failed': {},
                'metadata': {
                    'total_items': len(items),
                    'start_time': datetime.now().isoformat(),
                    'session_id': session_id
                }
            }
            
            self.logger.info(f"Starting sequential processing of {len(items)} items", extra={
                'item_count': len(items),
                'session_id': session_id
            })
            
            for i, item in enumerate(items, 1):
                item_start_time = time.time()
                
                try:
                    # Process individual item with timeout and error isolation
                    result = self._process_single_item_safely(item, process_func)
                    
                    if result is not None:
                        results['successful'][item] = result
                        self.logger.debug(f"Successfully processed {item}", extra={
                            'item': item,
                            'duration': time.time() - item_start_time,
                            'progress': f"{i}/{len(items)}"
                        })
                    else:
                        results['failed'][item] = {
                            'error': 'Processing returned None',
                            'timestamp': datetime.now().isoformat(),
                            'duration': time.time() - item_start_time
                        }
                        
                        if not self.continue_on_error:
                            self.logger.error(f"Processing failed for {item}, aborting batch", extra={
                                'item': item,
                                'continue_on_error': False
                            })
                            break
                
                except Exception as e:
                    error_info = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now().isoformat(),
                        'duration': time.time() - item_start_time,
                        'traceback': traceback.format_exc()
                    }
                    results['failed'][item] = error_info
                    
                    self.logger.error(f"Error processing {item}: {e}", extra={
                        'item': item,
                        'error_type': type(e).__name__,
                        'duration': time.time() - item_start_time
                    }, exc_info=True)
                    
                    if not self.continue_on_error:
                        self.logger.error(f"Error processing {item}, aborting batch", extra={
                            'item': item,
                            'continue_on_error': False
                        })
                        break
                
                # Progress reporting
                if self.progress_callback and i % 5 == 0:
                    self.progress_callback(i, len(items), results)
                
                # Log progress periodically
                if i % 10 == 0 or i == len(items):
                    success_rate = len(results['successful']) / i
                    self.logger.info(f"Batch progress: {i}/{len(items)} processed", extra={
                        'processed': i,
                        'total': len(items),
                        'successful': len(results['successful']),
                        'failed': len(results['failed']),
                        'success_rate': success_rate,
                        'session_id': session_id
                    })
            
            # Finalize results
            results['metadata'].update({
                'end_time': datetime.now().isoformat(),
                'total_successful': len(results['successful']),
                'total_failed': len(results['failed']),
                'success_rate': len(results['successful']) / len(items) if items else 0,
                'processing_complete': True
            })
            
            # End monitoring session
            session_summary = end_batch_analysis()
            if session_summary:
                results['metadata']['session_summary'] = session_summary
            
            self.logger.info(f"Completed sequential processing", extra={
                'total_items': len(items),
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'success_rate': results['metadata']['success_rate'],
                'session_id': session_id
            })
            
            return results
    
    def process_items_parallel(self, 
                             items: List[str], 
                             process_func: Callable[[str], Any],
                             session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process items in parallel with error isolation
        
        Note: Use with caution due to rate limiting concerns with external APIs
        """
        
        session_id = start_batch_analysis(items, session_name)
        
        with operation_context("batch_process_parallel", 
                             item_count=len(items), 
                             session_id=session_id,
                             max_workers=self.max_workers):
            
            results = {
                'successful': {},
                'failed': {},
                'metadata': {
                    'total_items': len(items),
                    'start_time': datetime.now().isoformat(),
                    'session_id': session_id,
                    'max_workers': self.max_workers
                }
            }
            
            self.logger.info(f"Starting parallel processing of {len(items)} items", extra={
                'item_count': len(items),
                'max_workers': self.max_workers,
                'session_id': session_id
            })
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(self._process_single_item_safely, item, process_func): item 
                    for item in items
                }
                
                # Process completed tasks
                for future in as_completed(future_to_item, timeout=len(items) * self.timeout_per_item):
                    item = future_to_item[future]
                    
                    try:
                        result = future.result(timeout=self.timeout_per_item)
                        
                        if result is not None:
                            results['successful'][item] = result
                        else:
                            results['failed'][item] = {
                                'error': 'Processing returned None',
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    except Exception as e:
                        error_info = {
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'timestamp': datetime.now().isoformat(),
                            'traceback': traceback.format_exc()
                        }
                        results['failed'][item] = error_info
                        
                        self.logger.error(f"Parallel processing error for {item}: {e}", extra={
                            'item': item,
                            'error_type': type(e).__name__
                        }, exc_info=True)
            
            # Finalize results
            results['metadata'].update({
                'end_time': datetime.now().isoformat(),
                'total_successful': len(results['successful']),
                'total_failed': len(results['failed']),
                'success_rate': len(results['successful']) / len(items) if items else 0,
                'processing_complete': True
            })
            
            # End monitoring session
            session_summary = end_batch_analysis()
            if session_summary:
                results['metadata']['session_summary'] = session_summary
            
            self.logger.info(f"Completed parallel processing", extra={
                'total_items': len(items),
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'success_rate': results['metadata']['success_rate'],
                'session_id': session_id
            })
            
            return results
    
    def _process_single_item_safely(self, item: str, process_func: Callable[[str], Any]) -> Any:
        """
        Process a single item with comprehensive error handling and timeout
        """
        
        start_time = time.time()
        
        try:
            with operation_context("process_single_item", item=item):
                result = process_func(item)
                
                duration = time.time() - start_time
                self.logger.debug(f"Processed {item} successfully", extra={
                    'item': item,
                    'duration': duration,
                    'result_type': type(result).__name__ if result else 'None'
                })
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error processing {item}: {e}", extra={
                'item': item,
                'duration': duration,
                'error_type': type(e).__name__
            }, exc_info=True)
            
            raise  # Re-raise to be handled by caller


class CircuitBreaker:
    """
    Circuit breaker pattern for handling cascading failures
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 reset_timeout: float = 300.0,  # 5 minutes
                 expected_exception: type = Exception):
        
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - too many recent failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.reset_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# Global instances for easy access
default_batch_processor = BatchProcessor()
data_fetch_circuit_breaker = CircuitBreaker(failure_threshold=10, reset_timeout=600)  # 10 minutes


# Convenience functions
def process_symbols_with_isolation(symbols: List[str], 
                                 process_func: Callable[[str], Any],
                                 parallel: bool = False,
                                 session_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for processing stock symbols with error isolation
    """
    processor = BatchProcessor(max_workers=3 if parallel else 1)
    
    if parallel:
        return processor.process_items_parallel(symbols, process_func, session_name)
    else:
        return processor.process_items_sequentially(symbols, process_func, session_name)


def safe_stock_analysis(analysis_func: Callable[[str], Any], 
                       symbol: str) -> Optional[Any]:
    """
    Wrapper for individual stock analysis with circuit breaker protection
    """
    try:
        return data_fetch_circuit_breaker.call(analysis_func, symbol)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Circuit breaker blocked or analysis failed for {symbol}: {e}", 
                    extra={'symbol': symbol, 'error_type': type(e).__name__})
        return None


if __name__ == "__main__":
    # Demo the error isolation system
    from .logging_config import setup_logging
    import random
    
    setup_logging(level="INFO", console_output=True)
    logger = get_logger(__name__)
    
    def mock_analysis_function(symbol: str) -> Dict[str, Any]:
        """Mock analysis function that sometimes fails"""
        time.sleep(0.1)  # Simulate processing time
        
        # Simulate random failures
        if random.random() < 0.3:  # 30% failure rate
            raise ValueError(f"Simulated failure for {symbol}")
        
        return {
            'symbol': symbol,
            'score': random.uniform(40, 90),
            'analysis_time': datetime.now().isoformat()
        }
    
    # Test sequential processing
    test_symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "FAIL1", "FAIL2"]
    
    logger.info("Testing sequential processing with error isolation")
    results = process_symbols_with_isolation(
        test_symbols, 
        mock_analysis_function, 
        parallel=False,
        session_name="demo_sequential"
    )
    
    print(f"\nResults Summary:")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Success Rate: {results['metadata']['success_rate']:.1%}")
    
    # Test circuit breaker
    logger.info("\nTesting circuit breaker")
    def always_fail(symbol):
        raise ConnectionError("Network unavailable")
    
    for i in range(12):  # Should trigger circuit breaker
        try:
            safe_stock_analysis(always_fail, f"TEST{i}")
        except Exception as e:
            logger.info(f"Attempt {i+1}: {e}")