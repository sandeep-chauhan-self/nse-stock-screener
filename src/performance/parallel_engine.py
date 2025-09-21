"""
Parallel Processing Engine for NSE Stock Screener

This module provides high-performance parallel processing capabilities for scaling
to the full NSE universe with production reliability.

Key Features:
- Concurrent symbol processing using joblib and multiprocessing
- Async I/O for data fetching
- Intelligent batching and load balancing
- Resource management and memory optimization
- Error handling and fault tolerance
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import multiprocessing as mp
import queue
import threading
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

# Import monitoring and caching
try:
    from ..monitoring.metrics import performance_monitor, PerformanceMetrics
    from ..caching.cache_manager import CacheManager
    from ..logging_config import get_logger, operation_context
except ImportError:
    # Fallback imports
    def get_logger(name): return logging.getLogger(name)
    def operation_context(_name, **_kwargs):
        from contextlib import nullcontext
        return nullcontext()
    class PerformanceMetrics:
        def record_processing_time(self, *_args):
            # Fallback implementation - no-op when monitoring not available
            pass
        def record_memory_usage(self, *_args):
            # Fallback implementation - no-op when monitoring not available
            pass
        def record_error(self, *_args):
            # Fallback implementation - no-op when monitoring not available
            pass
    performance_monitor = PerformanceMetrics()
    class CacheManager:
        def get(self, _key):
            # Fallback implementation - always cache miss
            return None
        def set(self, _key, _value):
            # Fallback implementation - no-op when caching not available
            pass
    cache_manager = CacheManager()


logger = get_logger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None  # Auto-detect based on CPU count
    batch_size: int = 50  # Symbols per batch
    chunk_size: int = 10  # Symbols per chunk within batch
    memory_limit_gb: float = 8.0  # Memory limit per worker
    timeout_seconds: int = 300  # Timeout per symbol processing
    enable_async_io: bool = True
    use_process_pool: bool = True  # Use ProcessPool vs ThreadPool
    prefetch_data: bool = True  # Prefetch data for next batch
    adaptive_batching: bool = True  # Adjust batch size based on performance

    def __post_init__(self):
        if self.max_workers is None:
            # Conservative CPU usage: leave 1-2 cores for system
            cpu_count = mp.cpu_count()
            self.max_workers = max(1, min(cpu_count - 1, 8))

        # Adjust batch size based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:
            self.batch_size = min(self.batch_size, 20)
            self.chunk_size = min(self.chunk_size, 5)


class ResourceMonitor:
    """Monitor system resources during parallel processing."""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory
        self.warnings_issued = set()

    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        current_memory = psutil.virtual_memory()
        used_gb = current_memory.used / (1024**3)

        if used_gb > self.memory_limit_gb:
            if "memory_limit" not in self.warnings_issued:
                logger.warning(f"Memory usage {used_gb:.1f}GB exceeds limit {self.memory_limit_gb}GB")
                self.warnings_issued.add("memory_limit")
            return False

        self.peak_memory = max(self.peak_memory, current_memory.used)
        return True

    def get_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        current_memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()

        return {
            "memory_used_gb": current_memory.used / (1024**3),
            "memory_available_gb": current_memory.available / (1024**3),
            "memory_percent": current_memory.percent,
            "peak_memory_gb": self.peak_memory / (1024**3),
            "cpu_percent": cpu_percent,
            "memory_growth_gb": (current_memory.used - self.start_memory) / (1024**3)
        }


class AsyncDataFetcher:
    """Async data fetching for improved I/O performance."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_symbol_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol asynchronously."""
        async with self.semaphore:
            try:
                # Use thread pool for blocking I/O operations
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(
                        executor,
                        self._sync_fetch_data,
                        symbol,
                        kwargs
                    )
                    data = await asyncio.wait_for(future, timeout=30.0)
                    return data
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching data for {symbol}")
                performance_monitor.record_error("data_fetch_timeout", symbol)
                return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                performance_monitor.record_error("data_fetch_error", symbol)
                return None

    def _sync_fetch_data(self, symbol: str, kwargs: Dict) -> pd.DataFrame:
        """Synchronous data fetching (to be run in thread pool)."""
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        return ticker.history(**kwargs)

    async def fetch_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently."""
        tasks = [self.fetch_symbol_data(symbol, **kwargs) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        symbol_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
                performance_monitor.record_error("async_fetch_error", symbol)
            elif result is not None and not result.empty:
                symbol_data[symbol] = result

        return symbol_data


class BatchProcessor:
    """Process symbols in optimized batches."""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor(config.memory_limit_gb)
        self.processing_times = []
        self.adaptive_batch_size = config.batch_size

    def process_symbol_batch(
        self,
        symbols: List[str],
        processing_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a batch of symbols."""
        start_time = time.time()
        batch_results = {}

        logger.info(f"Processing batch of {len(symbols)} symbols")

        try:
            if self.config.enable_async_io:
                # Use async I/O for data fetching
                batch_results = asyncio.run(
                    self._process_batch_async(symbols, processing_func, **kwargs)
                )
            else:
                # Use traditional parallel processing
                batch_results = self._process_batch_sync(symbols, processing_func, **kwargs)

            # Record performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            performance_monitor.record_processing_time(
                "batch_processing",
                processing_time,
                {"batch_size": len(symbols), "success_count": len(batch_results)}
            )

            # Update adaptive batch size
            if self.config.adaptive_batching:
                self._update_adaptive_batch_size(processing_time, len(symbols))

            logger.info(f"Completed batch in {processing_time:.2f}s, {len(batch_results)} successful")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            performance_monitor.record_error("batch_processing_error", str(e))

        return batch_results

    async def _process_batch_async(
        self,
        symbols: List[str],
        processing_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Process batch using async I/O."""
        results = {}

        # First, fetch all data asynchronously
        async with AsyncDataFetcher(max_concurrent=self.config.max_workers) as fetcher:
            symbol_data = await fetcher.fetch_multiple_symbols(
                symbols,
                period="1y",
                interval="1d"
            )

        # Then process the data in parallel
        if symbol_data:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(processing_func, symbol, data, **kwargs): symbol
                    for symbol, data in symbol_data.items()
                }

                for future in as_completed(future_to_symbol, timeout=self.config.timeout_seconds):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[symbol] = result
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        performance_monitor.record_error("symbol_processing_error", symbol)

        return results

    def _process_batch_sync(
        self,
        symbols: List[str],
        processing_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Process batch using traditional parallel processing."""

        def process_single_symbol(symbol: str) -> Tuple[str, Any]:
            try:
                # Check memory before processing
                if not self.resource_monitor.check_memory():
                    return symbol, None

                result = processing_func(symbol, **kwargs)
                return symbol, result
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                performance_monitor.record_error("symbol_processing_error", symbol)
                return symbol, None

        # Use joblib for parallel processing with memory management
        with joblib.parallel_backend('threading', n_jobs=self.config.max_workers):
            results = Parallel()(
                delayed(process_single_symbol)(symbol)
                for symbol in symbols
            )

        # Filter successful results
        return {symbol: result for symbol, result in results if result is not None}

    def _update_adaptive_batch_size(self, _processing_time: float, _batch_size: int):
        """Update batch size based on performance feedback."""
        if len(self.processing_times) < 3:
            return

        # Calculate recent average processing time
        recent_avg = np.mean(self.processing_times[-3:])

        # Adjust batch size based on performance
        if recent_avg < 30:  # Fast processing, can increase batch size
            self.adaptive_batch_size = min(self.adaptive_batch_size + 5, 100)
        elif recent_avg > 60:  # Slow processing, decrease batch size
            self.adaptive_batch_size = max(self.adaptive_batch_size - 5, 10)

        logger.debug(f"Adaptive batch size updated to {self.adaptive_batch_size}")


class ParallelStockProcessor:
    """Main parallel processing engine for stock analysis."""

    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self.batch_processor = BatchProcessor(self.config)
        self.cache_manager = cache_manager
        self.results_cache = {}

        logger.info(f"Initialized parallel processor with {self.config.max_workers} workers")

    def process_symbol_universe(
        self,
        symbols: List[str],
        processing_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Process the entire symbol universe efficiently."""

        total_symbols = len(symbols)
        logger.info(f"Starting parallel processing of {total_symbols} symbols")

        # Record start of batch processing
        session_id = f"batch_{int(time.time())}"
        performance_monitor.record_processing_time("batch_start", 0, {"session_id": session_id})

        try:
            all_results = {}
            processed_count = 0

            # Process in batches
            batch_size = self.batch_processor.adaptive_batch_size
            for i in range(0, total_symbols, batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_symbols + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches}")

                # Process batch
                batch_results = self.batch_processor.process_symbol_batch(
                    batch_symbols, processing_func, **kwargs
                )

                all_results.update(batch_results)
                processed_count += len(batch_results)

                # Log progress
                progress = (processed_count / total_symbols) * 100
                logger.info(f"Progress: {progress:.1f}% ({processed_count}/{total_symbols})")

                # Memory cleanup between batches
                if batch_num % 5 == 0:  # Every 5 batches
                    import gc
                    gc.collect()

                # Resource monitoring
                resource_stats = self.batch_processor.resource_monitor.get_stats()
                performance_monitor.record_memory_usage(
                    "batch_processing",
                    resource_stats["memory_used_gb"]
                )

                # Adaptive delay for system stability
                if resource_stats["memory_percent"] > 85:
                    logger.warning("High memory usage, adding delay between batches")
                    time.sleep(2)

            # Final statistics
            success_rate = (len(all_results) / total_symbols) * 100
            logger.info(f"Parallel processing completed: {len(all_results)}/{total_symbols} symbols ({success_rate:.1f}%)")

            return all_results

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            performance_monitor.record_error("parallel_processing_error", str(e))
            raise

    def process_with_caching(
        self,
        symbols: List[str],
        processing_func: Callable,
        cache_key_func: Optional[Callable] = None,
        cache_ttl: int = 3600,
        **kwargs
    ) -> Dict[str, Any]:
        """Process symbols with intelligent caching."""

        if cache_key_func is None:
            cache_key_func = lambda symbol: f"analysis_{symbol}_{int(time.time() // cache_ttl)}"

        # Check cache for existing results
        cached_results = {}
        symbols_to_process = []

        for symbol in symbols:
            cache_key = cache_key_func(symbol)
            cached_result = self.cache_manager.get(cache_key)

            if cached_result is not None:
                cached_results[symbol] = cached_result
            else:
                symbols_to_process.append(symbol)

        logger.info(f"Cache hit: {len(cached_results)}, Processing: {len(symbols_to_process)}")

        # Process remaining symbols
        new_results = {}
        if symbols_to_process:
            new_results = self.process_symbol_universe(
                symbols_to_process, processing_func, **kwargs
            )

            # Cache new results
            for symbol, result in new_results.items():
                cache_key = cache_key_func(symbol)
                self.cache_manager.set(cache_key, result, ttl=cache_ttl)

        # Combine cached and new results
        all_results = {**cached_results, **new_results}
        return all_results

    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            processing_time = end_time - start_time
            memory_used = (end_memory - start_memory) / (1024**3)  # GB

            performance_monitor.record_processing_time(operation_name, processing_time)
            performance_monitor.record_memory_usage(operation_name, memory_used)

            logger.info(f"{operation_name} completed in {processing_time:.2f}s, memory: {memory_used:.2f}GB")


# Utility decorators for performance optimization
def parallel_cached(cache_ttl: int = 3600):
    """Decorator to add caching to parallel processing functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            # Check cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.set(cache_key, result, ttl=cache_ttl)

            return result
        return wrapper
    return decorator


def performance_tracked(operation_name: str = None):
    """Decorator to track performance metrics."""
    def decorator(func):
        op_name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used

            try:
                result = func(*args, **kwargs)

                # Record success metrics
                end_time = time.time()
                end_memory = psutil.virtual_memory().used

                processing_time = end_time - start_time
                memory_used = (end_memory - start_memory) / (1024**3)

                performance_monitor.record_processing_time(op_name, processing_time)
                performance_monitor.record_memory_usage(op_name, memory_used)

                return result

            except Exception as e:
                performance_monitor.record_error(op_name, str(e))
                raise

        return wrapper
    return decorator


# Example usage function
def create_parallel_processor(
    max_workers: Optional[int] = None,
    batch_size: int = 50,
    enable_async_io: bool = True
) -> ParallelStockProcessor:
    """Factory function to create optimally configured parallel processor."""

    config = ParallelConfig(
        max_workers=max_workers,
        batch_size=batch_size,
        enable_async_io=enable_async_io,
        adaptive_batching=True,
        prefetch_data=True
    )

    return ParallelStockProcessor(config)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))

    # Demo function
    def demo_processing_function(symbol: str, **kwargs) -> Dict[str, float]:
        """Demo function for testing parallel processing."""
        import random
        import time

        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))

        return {
            "symbol": symbol,
            "score": random.uniform(0, 100),
            "confidence": random.uniform(0.5, 1.0)
        }

    # Test symbols
    test_symbols = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "ITC", "BHARTIARTL", "SBIN", "LT", "KOTAKBANK"
    ]

    # Create processor and run test
    processor = create_parallel_processor(max_workers=4, batch_size=5)

    with processor.performance_context("demo_processing"):
        results = processor.process_symbol_universe(
            test_symbols,
            demo_processing_function
        )

    print(f"Processed {len(results)} symbols successfully")
    for symbol, result in results.items():
        print(f"  {symbol}: score={result['score']:.1f}, confidence={result['confidence']:.2f}")