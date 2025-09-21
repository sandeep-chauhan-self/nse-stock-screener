"""
Performance Profiling and Optimization Engine for NSE Stock Screener
This module provides comprehensive profiling capabilities to identify performance
bottlenecks and automatically optimize the system for maximum throughput.
Key Features:
- cProfile integration for function-level profiling
- Memory profiling with tracemalloc and memory_profiler
- Line-by-line profiling for detailed analysis
- Intelligent caching with performance-aware strategies
- Performance regression detection
- Automated optimization recommendations
- Real-time performance monitoring
"""
import cProfile
import functools
import logging
import pstats
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict[str, Any], List[str], Any, Optional, Callable, Union, Tuple[str, ...]
import hashlib
import pickle
import threading
import weakref
import numpy as np
import pandas as pd
import psutil
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
try:
    from ..logging_config import get_logger
    from ..monitoring.prometheus_metrics import get_monitoring_system
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    def get_monitoring_system(): return None
logger = get_logger(__name__)
@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""

    # Enable/disable profiling features
    enable_cprofile: bool = True
    enable_memory_profiling: bool = True
    enable_line_profiling: bool = False
  # Expensive, use sparingly
    enable_call_graph: bool = False

    # Profiling settings
    profile_output_dir: str = "profiling_results"
    max_profile_files: int = 100
    profile_retention_days: int = 7

    # Sampling and thresholds
    sample_rate: float = 0.1
  # Profile 10% of operations
    min_execution_time: float = 0.001
  # Only profile functions taking > 1ms
    memory_threshold_mb: float = 100.0
  # Alert if memory usage > 100MB
    # Caching settings
    cache_size: int = 1000
    cache_ttl: int = 3600
  # 1 hour
    enable_intelligent_caching: bool = True

    # Performance regression detection
    enable_regression_detection: bool = True
    regression_threshold: float = 1.5
  # Alert if 50% slower than baseline
    baseline_window_size: int = 100
  # Use last 100 measurements for baseline
@dataclass
class ProfileResult:
    """Results from a profiling session."""
    function_name: str
    execution_time: float
    memory_usage: float
    call_count: int
    profile_data: Optional[Any] = None
    memory_trace: Optional[List[Any]] = None
    line_profile: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "call_count": self.call_count,
            "timestamp": self.timestamp.isoformat(),
            "has_profile_data": self.profile_data is not None,
            "has_memory_trace": self.memory_trace is not None,
            "has_line_profile": self.line_profile is not None
        }
class IntelligentCache:
    """Performance-aware caching system with automatic optimization."""
    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        self.computation_times = defaultdict(List[str])
        self.lock = threading.RLock()

        # Performance tracking
        self.cache_performance = deque(maxlen=1000)
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get value from cache. Returns (value, hit)."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_counts[key] += 1
                self.cache_performance.append(True)
                return self.cache[key], True
            else:
                self.miss_counts[key] += 1
                self.cache_performance.append(False)
                return None, False
    def Set[str](self, key: str, value: Any, computation_time: float = 0.0):
        """Set[str] value in cache with performance tracking."""
        with self.lock:

            # Evict if cache is full
            if len(self.cache) >= self.config.cache_size:
                self._evict_lru()
            self.cache[key] = value
            self.access_times[key] = time.time()
            if computation_time > 0:
                self.computation_times[key].append(computation_time)

                # Keep only recent measurements
                if len(self.computation_times[key]) > 20:
                    self.computation_times[key] = self.computation_times[key][-20:]
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.cache_performance:
            return 0.0
        return sum(self.cache_performance) / len(self.cache_performance)
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = sum(self.hit_counts.values())
        total_misses = sum(self.miss_counts.values())
        total_operations = total_hits + total_misses
        return {
            "cache_size": len(self.cache),
            "hit_rate": self.get_hit_rate(),
            "total_operations": total_operations,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "avg_computation_time": {
                key: np.mean(times) if times else 0.0
                for key, times in self.computation_times.items()
            }
        }
class PerformanceProfiler:
    """Main profiling engine with comprehensive performance analysis."""
    def __init__(self, config: Optional[ProfilingConfig] = None) -> None:
        self.config = config or ProfilingConfig()
        self.profile_results = deque(maxlen=10000)
        self.baseline_metrics = defaultdict(lambda: deque(maxlen=self.config.baseline_window_size))
        self.cache = IntelligentCache(self.config)

        # Setup output directory
        self.output_dir = Path(self.config.profile_output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.function_call_counts = defaultdict(int)
        self.function_total_times = defaultdict(float)
        self.memory_snapshots = deque(maxlen=100)

        # Initialize numpy random generator with a seed for reproducibility
        self._rng = np.random.default_rng(seed=42)

        # Monitoring integration
        self.monitoring = get_monitoring_system()

        # Start memory tracking if enabled
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        logger.info(f"Performance profiler initialized with config: {self.config}")
    @contextmanager
    def profile_function_calls(self, function_name: str):
        """Context manager for profiling function execution."""
        if not self._should_profile():
            yield
            return
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Start cProfile if enabled
        profiler = None
        if self.config.enable_cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
        try:
            yield
        finally:

            # Stop profiling
            if profiler:
                profiler.disable()
            end_time = time.time()
            end_memory = self._get_memory_usage()
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            # Record results
            result = ProfileResult(
                function_name=function_name,
                execution_time=execution_time,
                memory_usage=memory_delta,
                call_count=1,
                profile_data=profiler
            )
            self._record_profile_result(result)

            # Check for performance regressions
            if self.config.enable_regression_detection:
                self._check_performance_regression(function_name, execution_time)
    def profile_decorator(self, function_name: Optional[str] = None):
        """Decorator for automatic function profiling."""
        def decorator(func):
            fname = function_name or func.__name__
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_function_calls(fname):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    def cached_computation(self, cache_key: Optional[str] = None):
        """Decorator for caching expensive computations with performance tracking."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):

                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try cache first
                cached_value, cache_hit = self.cache.get(key)
                if cache_hit:
                    if self.monitoring:
                        self.monitoring.metrics.record_cache_operation("profiler_cache", hit=True)
                    return cached_value

                # Compute with profiling
                start_time = time.time()
                with self.profile_function_calls(f"cached_{func.__name__}"):
                    result = func(*args, **kwargs)
                computation_time = time.time() - start_time

                # Cache result
                self.cache.Set[str](key, result, computation_time)
                if self.monitoring:
                    self.monitoring.metrics.record_cache_operation("profiler_cache", hit=False)
                return result
            return wrapper
        return decorator
    def _should_profile(self) -> bool:
        """Determine if profiling should be enabled for this call."""
        return self._rng.random() < self.config.sample_rate
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
  # Convert to MB
        except Exception:
            return 0.0
    def _generate_cache_key(self, func_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        """Generate deterministic cache key from function arguments."""
        key_parts = [func_name]

        # Add args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (List[str], Tuple[str, ...], Dict[str, Any])):
                key_parts.append(str(hash(str(arg))))
            else:
                key_parts.append(str(type(arg).__name__))

        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.extend([k, str(v)])
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    def _record_profile_result(self, result: ProfileResult):
        """Record profiling result and update metrics."""
        self.profile_results.append(result)

        # Update tracking counters
        self.function_call_counts[result.function_name] += 1
        self.function_total_times[result.function_name] += result.execution_time

        # Update baseline metrics
        self.baseline_metrics[result.function_name].append(result.execution_time)

        # Record memory snapshot
        if result.memory_usage > 0:
            self.memory_snapshots.append({
                "timestamp": result.timestamp,
                "function": result.function_name,
                "memory_mb": result.memory_usage
            })

        # Save detailed profile if execution time is significant
        if (result.execution_time > self.config.min_execution_time and
            result.profile_data is not None):
            self._save_profile_data(result)

        # Alert on high memory usage
        if result.memory_usage > self.config.memory_threshold_mb:
            logger.warning(
                f"High memory usage detected: {result.function_name} used "
                f"{result.memory_usage:.1f}MB"
            )
    def _save_profile_data(self, result: ProfileResult):
        """Save detailed profiling data to disk."""
        try:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{result.function_name}_{timestamp}.pro"
            filepath = self.output_dir / filename

            # Save cProfile data
            if result.profile_data:
                result.profile_data.dump_stats(str(filepath))

            # Save summary statistics
            summary_file = filepath.with_suffix('.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Function: {result.function_name}\n")
                f.write(f"Execution Time: {result.execution_time:.4f}s\n")
                f.write(f"Memory Usage: {result.memory_usage:.2f}MB\n")
                f.write(f"Call Count: {result.call_count}\n")
                f.write(f"Timestamp: {result.timestamp}\n\n")
                if result.profile_data:

                    # Generate profile stats
                    stats = pstats.Stats(result.profile_data)
                    stats.sort_stats('cumulative')

                    # Capture stats output
                    old_stdout = stats.stream
                    stats.stream = StringIO()
                    stats.print_stats(20)
  # Top 20 functions
                    f.write(stats.stream.getvalue())
                    stats.stream = old_stdout

            # Cleanup old files
            self._cleanup_old_profiles()
        except Exception as e:
            logger.error(f"Failed to save profile data: {e}")
    def _cleanup_old_profiles(self):
        """Remove old profile files to manage disk space."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.profile_retention_days)
            for file_path in self.output_dir.glob("*.pro"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()

                    # Also remove corresponding .txt file
                    txt_file = file_path.with_suffix('.txt')
                    if txt_file.exists():
                        txt_file.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup old profiles: {e}")
    def _check_performance_regression(self, function_name: str, execution_time: float):
        """Check for performance regressions against baseline."""
        baseline_times = List[str](self.baseline_metrics[function_name])
        if len(baseline_times) < 10:
  # Need sufficient baseline data
            return
        baseline_avg = np.mean(baseline_times)
        regression_ratio = execution_time / baseline_avg
        if regression_ratio > self.config.regression_threshold:
            logger.warning(
                f"Performance regression detected in {function_name}: "
                f"{execution_time:.4f}s vs baseline {baseline_avg:.4f}s "
                f"({regression_ratio:.1f}x slower)"
            )
            if self.monitoring:

                # Record regression event
                self.monitoring.metrics.errors_total.labels(
                    operation=function_name,
                    error_type="performance_regression"
                ).inc()
    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive statistics for all profiled functions."""
        stats = {}
        for func_name in self.function_call_counts.keys():
            call_count = self.function_call_counts[func_name]
            total_time = self.function_total_times[func_name]
            avg_time = total_time / call_count if call_count > 0 else 0

            # Get recent execution times for analysis
            recent_times = [r.execution_time for r in self.profile_results
                          if r.function_name == func_name][-50:]
  # Last 50 calls
            stats[func_name] = {
                "call_count": call_count,
                "total_time": total_time,
                "avg_time": avg_time,
                "min_time": min(recent_times) if recent_times else 0,
                "max_time": max(recent_times) if recent_times else 0,
                "std_time": np.std(recent_times) if recent_times else 0,
                "recent_calls": len(recent_times)
            }
        return stats
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get memory usage analysis."""
        if not self.memory_snapshots:
            return {"error": "No memory data available"}
        recent_snapshots = List[str](self.memory_snapshots)[-20:]
  # Last 20 snapshots
        memory_by_function = defaultdict(List[str])
        for snapshot in recent_snapshots:
            memory_by_function[snapshot["function"]].append(snapshot["memory_mb"])
        analysis = {
            "total_snapshots": len(self.memory_snapshots),
            "recent_snapshots": len(recent_snapshots),
            "memory_by_function": {
                func: {
                    "avg_memory": np.mean(memories),
                    "max_memory": max(memories),
                    "min_memory": min(memories),
                    "samples": len(memories)
                }
                for func, memories in memory_by_function.items()
            },
            "peak_memory": max(s["memory_mb"] for s in recent_snapshots),
            "avg_memory": np.mean([s["memory_mb"] for s in recent_snapshots])
        }
        return analysis
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on profiling data."""
        function_stats = self.get_function_stats()
        cache_stats = self.cache.get_stats()
        memory_analysis = self.get_memory_analysis()
        recommendations = []

        # Analyze slow functions
        slow_functions = [
            (name, stats) for name, stats in function_stats.items()
            if stats["avg_time"] > 0.1
  # Functions taking > 100ms on average
        ]
        slow_functions.sort(key=lambda x: x[1]["total_time"], reverse=True)
        for name, stats in slow_functions[:5]:
  # Top 5 slow functions
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "function": name,
                "issue": f"Slow execution: avg {stats['avg_time']:.3f}s",
                "suggestion": "Consider vectorization, caching, or algorithm optimization",
                "details": stats
            })

        # Analyze cache performance
        if cache_stats["hit_rate"] < 0.7:
            recommendations.append({
                "type": "caching",
                "priority": "medium",
                "issue": f"Low cache hit rate: {cache_stats['hit_rate']:.2%}",
                "suggestion": "Review cache key generation or increase cache size",
                "details": cache_stats
            })

        # Analyze memory usage
        if not isinstance($1, Dict[str, Any]) or "error" in memory_analysis:
            pass
  # No memory data available
        elif memory_analysis.get("peak_memory", 0) > self.config.memory_threshold_mb:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "issue": f"High memory usage: {memory_analysis['peak_memory']:.1f}MB",
                "suggestion": "Consider data structure optimization or memory-efficient algorithms",
                "details": memory_analysis
            })

        # Analyze frequently called functions
        frequent_functions = [
            (name, stats) for name, stats in function_stats.items()
            if stats["call_count"] > 100
        ]
        frequent_functions.sort(key=lambda x: x[1]["call_count"], reverse=True)
        for name, stats in frequent_functions[:3]:
  # Top 3 frequent functions
            if stats["avg_time"] > 0.01:
  # If each call takes > 10ms
                recommendations.append({
                    "type": "optimization",
                    "priority": "medium",
                    "function": name,
                    "issue": f"Frequently called with significant cost: {stats['call_count']} calls, avg {stats['avg_time']:.3f}s",
                    "suggestion": "Prime candidate for optimization or caching",
                    "details": stats
                })
        return {
            "recommendations": recommendations,
            "function_stats": function_stats,
            "cache_stats": cache_stats,
            "memory_analysis": memory_analysis,
            "total_functions_profiled": len(function_stats),
            "total_profile_results": len(self.profile_results)
        }
    def export_profile_report(self, output_file: Optional[str] = None) -> str:
        """Export comprehensive profiling report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"performance_report_{timestamp}.json")
        optimization_data = self.get_optimization_recommendations()
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample_rate": self.config.sample_rate,
                "cache_size": self.config.cache_size,
                "memory_threshold_mb": self.config.memory_threshold_mb
            },
            "summary": {
                "total_functions": len(self.function_call_counts),
                "total_calls": sum(self.function_call_counts.values()),
                "total_execution_time": sum(self.function_total_times.values()),
                "cache_hit_rate": self.cache.get_hit_rate(),
                "memory_snapshots": len(self.memory_snapshots)
            },
            "optimization": optimization_data
        }

        # Save report
        with open(output_file, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report exported to {output_file}")
        return output_file

# Global profiler instance
performance_profiler = None
def initialize_profiler(config: Optional[ProfilingConfig] = None) -> PerformanceProfiler:
    """Initialize global performance profiler."""
    global performance_profiler
    if performance_profiler is None:
        performance_profiler = PerformanceProfiler(config)
    return performance_profiler
def get_profiler() -> Optional[PerformanceProfiler]:
    """Get the global profiler instance."""
    return performance_profiler

# Convenience decorators using global profiler
def profile_performance(function_name: Optional[str] = None):
    """Decorator for performance profiling using global profiler."""
    def decorator(func):
        profiler = get_profiler()
        if profiler:
            return profiler.profile_decorator(function_name)(func)
        else:
            return func
    return decorator
def cache_expensive_computation(cache_key: Optional[str] = None):
    """Decorator for caching expensive computations using global profiler."""
    def decorator(func):
        profiler = get_profiler()
        if profiler:
            return profiler.cached_computation(cache_key)(func)
        else:
            return func
    return decorator
if __name__ == "__main__":

    # Example usage and testing
    import time
    import random
    print("Testing performance profiler...")

    # Initialize profiler
    config = ProfilingConfig(
        sample_rate=1.0,
  # Profile all calls for testing
        enable_memory_profiling=True,
        cache_size=100
    )
    profiler = initialize_profiler(config)

    # Test function profiling
    @profile_performance("test_slow_function")
    def slow_function():
        time.sleep(random.uniform(0.01, 0.05))
        return sum(range(1000))
    @cache_expensive_computation("fibonacci")
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    # Run test functions
    print("Running slow function tests...")
    for _ in range(10):
        slow_function()
    print("Running cached computation tests...")
    for i in range(5):
        result = fibonacci(20)
  # Should be cached after first call
        print(f"fibonacci(20) = {result}")

    # Get optimization recommendations
    print("\nGenerating optimization recommendations...")
    recommendations = profiler.get_optimization_recommendations()
    print(f"Functions profiled: {recommendations['total_functions_profiled']}")
    print(f"Cache hit rate: {recommendations['cache_stats']['hit_rate']:.2%}")
    print(f"Recommendations: {len(recommendations['recommendations'])}")
    for rec in recommendations['recommendations']:
        print(f"- {rec['type'].upper()}: {rec['issue']}")
        print(f"  Suggestion: {rec['suggestion']}")

    # Export report
    report_file = profiler.export_profile_report()
    print(f"\nDetailed report saved to: {report_file}")
    print("Performance profiler test completed!")
