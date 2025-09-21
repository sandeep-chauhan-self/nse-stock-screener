"""
Performance and Scaling Module for NSE Stock Screener

This module provides comprehensive performance optimization and scaling capabilities
to handle the full NSE universe with production-level reliability.

Components:
- parallel_engine: Concurrent processing with async I/O
- profiling_engine: Performance profiling and optimization
- caching_system: Intelligent caching with multiple backends
- integration: Complete performance stack integration

Usage:
    from src.performance import initialize_performance_stack

    # Initialize complete performance stack
    stack = initialize_performance_stack()

    # Process symbols at scale
    results = stack.process_symbols_sync(symbol_list, analysis_function)
"""

from .integration import initialize_performance_stack, PerformanceStackConfig
from .parallel_engine import ParallelStockProcessor, ParallelConfig
from .profiling_engine import PerformanceProfiler, ProfilingConfig
from .caching_system import initialize_cache, CacheConfig, get_cache, cached

__all__ = [
    'initialize_performance_stack',
    'PerformanceStackConfig',
    'ParallelStockProcessor',
    'ParallelConfig',
    'PerformanceProfiler',
    'ProfilingConfig',
    'initialize_cache',
    'CacheConfig',
    'get_cache',
    'cached'
]