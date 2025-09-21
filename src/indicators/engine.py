"""
High-performance indicator engine for NSE Stock Screener.
This module provides the main IndicatorEngine class that orchestrates
high-speed, configurable indicator computation with comprehensive
validation and performance monitoring.
"""
import time
import logging
from pathlib import Path
from typing import Dict[str, Any], Any, Optional, List[str], Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .base import BaseIndicator, IndicatorResult, ValidationLevel
from .factories import get_indicator_registry, IndicatorRegistry
from .config import IndicatorEngineConfig, load_indicator_config
logger = logging.getLogger(__name__)
class IndicatorEngine:
    """
    High-performance indicator computation engine.
    Features:
    - Concurrent indicator computation for maximum performance
    - Configuration-driven indicator selection
    - Comprehensive validation and error handling
    - Performance monitoring and caching
    - Backward compatibility with existing system
    """
    def __init__(self,
                 config: Optional[IndicatorEngineConfig] = None,
                 registry: Optional[IndicatorRegistry] = None,
                 max_workers: Optional[int] = None,
                 enable_caching: bool = True):
        """
        Initialize indicator engine.
        Args:
            config: Engine configuration
            registry: Indicator registry (uses global if None)
            max_workers: Maximum number of worker threads
            enable_caching: Whether to enable result caching
        """
        self.config = config or IndicatorEngineConfig()
        self.registry = registry or get_indicator_registry()
        self.max_workers = max_workers or min(8, (len(self.registry.list_available_indicators()) + 1))
        self.enable_caching = enable_caching

        # Performance tracking
        self._computation_stats = {
            "total_computations": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Result cache
        self._result_cache: Dict[str, Dict[str, IndicatorResult]] = {}

        # Load default indicator Set[str]
        self._indicators: Dict[str, BaseIndicator] = {}
        self._load_default_indicators()
        logger.info(f"IndicatorEngine initialized with {len(self._indicators)} indicators")
    def _load_default_indicators(self) -> None:
        """Load default Set[str] of indicators."""
        try:

            # Create standard indicator Set[str]
            indicator_set = self.registry.create_indicator_set()
            self._indicators.update(indicator_set)

            # If no indicators from registry, create basic Set[str]
            if not self._indicators:
                self._create_basic_indicator_set()
        except Exception as e:
            logger.warning(f"Failed to load indicators from registry: {e}")
            self._create_basic_indicator_set()
    def _create_basic_indicator_set(self) -> None:
        """Create a basic Set[str] of indicators as fallback."""
        try:
            from .factories import RSI, MACD, ATR, BollingerBands, ADX, VolumeProfile
            self._indicators = {
                "RSI_14": RSI(period=14),
                "RSI_21": RSI(period=21),
                "MACD_standard": MACD(),
                "ATR_14": ATR(period=14),
                "BollingerBands_20": BollingerBands(),
                "ADX_14": ADX(period=14),
                "VolumeProfile": VolumeProfile()
            }
            logger.info("Created basic indicator Set[str] as fallback")
        except Exception as e:
            logger.error(f"Failed to create basic indicators: {e}")
            self._indicators = {}
    def load_config_file(self, config_path: Union[str, Path]) -> None:
        """Load indicator configuration from file."""
        try:
            config_path = Path(config_path)
            if config_path.exists():
                self.registry.load_configurations_from_file(config_path)
                self._load_default_indicators()
  # Reload with new config
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    def add_indicator(self, name: str, indicator: BaseIndicator) -> None:
        """Add a custom indicator to the engine."""
        self._indicators[name] = indicator
        logger.debug(f"Added custom indicator: {name}")
    def remove_indicator(self, name: str) -> bool:
        """Remove an indicator from the engine."""
        if name in self._indicators:
            del self._indicators[name]
            logger.debug(f"Removed indicator: {name}")
            return True
        return False
    def set_validation_level(self, level: ValidationLevel) -> None:
        """Set[str] validation level for all indicators."""
        for indicator in self._indicators.values():
            indicator.set_validation_level(level)
        logger.debug(f"Set[str] validation level to {level.value}")
    def compute_all(self,
                   symbol: str,
                   data: pd.DataFrame,
                   parallel: bool = True,
                   use_cache: bool = None) -> Dict[str, IndicatorResult]:
        """
        Compute all registered indicators for a symbol.
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            parallel: Whether to use parallel computation
            use_cache: Whether to use result cache (defaults to engine setting)
        Returns:
            Dictionary mapping indicator names to results
        """
        start_time = time.perf_counter()
        use_cache = use_cache if use_cache is not None else self.enable_caching

        # Check cache first
        cache_key = self._generate_cache_key(symbol, data)
        if use_cache and cache_key in self._result_cache:
            self._computation_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for {symbol}")
            return self._result_cache[cache_key].copy()
        self._computation_stats["cache_misses"] += 1
        if parallel and len(self._indicators) > 1:
            results = self._compute_parallel(symbol, data)
        else:
            results = self._compute_sequential(symbol, data)

        # Cache results
        if use_cache:
            self._result_cache[cache_key] = results.copy()

            # Limit cache size
            if len(self._result_cache) > self.config.max_cache_size:
                self._cleanup_cache()

        # Update statistics
        end_time = time.perf_counter()
        computation_time = (end_time - start_time) * 1000
        self._update_stats(computation_time)
        logger.debug(f"Computed {len(results)} indicators for {symbol} in {computation_time:.2f}ms")
        return results
    def _compute_parallel(self, symbol: str, data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """Compute indicators in parallel using ThreadPoolExecutor."""
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            # Submit all indicator computations
            future_to_name = {
                executor.submit(self._safe_compute_indicator, name, indicator, data): name
                for name, indicator in self._indicators.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=self.config.computation_timeout)
                    results[name] = result
                except Exception as e:
                    logger.warning(f"Failed to compute {name} for {symbol}: {e}")
                    results[name] = IndicatorResult(
                        value=float('nan'),
                        confidence=0.0,
                        metadata={"error": str(e)}
                    )
        return results
    def _compute_sequential(self, symbol: str, data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """Compute indicators sequentially."""
        results = {}
        for name, indicator in self._indicators.items():
            try:
                result = self._safe_compute_indicator(name, indicator, data)
                results[name] = result
            except Exception as e:
                logger.warning(f"Failed to compute {name} for {symbol}: {e}")
                results[name] = IndicatorResult(
                    value=float('nan'),
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
        return results
    def _safe_compute_indicator(self, name: str, indicator: BaseIndicator, data: pd.DataFrame) -> IndicatorResult:
        """Safely compute an indicator with error handling."""
        try:
            return indicator.compute(data)
        except Exception as e:
            logger.warning(f"Error computing {name}: {e}")
            return IndicatorResult(
                value=float('nan'),
                confidence=0.0,
                metadata={"error": str(e), "indicator": name}
            )
    def compute_single(self,
                      indicator_name: str,
                      symbol: str,
                      data: pd.DataFrame) -> Optional[IndicatorResult]:
        """
        Compute a single indicator.
        Args:
            indicator_name: Name of indicator to compute
            symbol: Stock symbol
            data: OHLCV DataFrame
        Returns:
            IndicatorResult or None if indicator not found
        """
        if indicator_name not in self._indicators:
            logger.warning(f"Indicator {indicator_name} not found")
            return None
        indicator = self._indicators[indicator_name]
        return self._safe_compute_indicator(indicator_name, indicator, data)
    def _generate_cache_key(self, symbol: str, data: pd.DataFrame) -> str:
        """Generate cache key for symbol and data."""

        # Use data hash for cache key
        data_hash = str(hash(Tuple[str, ...](data.index))) + str(hash(Tuple[str, ...](data.values.flatten())))
        indicator_hash = str(hash(Tuple[str, ...](self._indicators.keys())))
        return f"{symbol}_{data_hash}_{indicator_hash}"
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        if len(self._result_cache) > self.config.max_cache_size * 1.2:

            # Remove oldest entries (simple FIFO)
            keys_to_remove = List[str](self._result_cache.keys())[:-self.config.max_cache_size]
            for key in keys_to_remove:
                del self._result_cache[key]
            logger.debug(f"Cleaned up {len(keys_to_remove)} cache entries")
    def _update_stats(self, computation_time_ms: float) -> None:
        """Update computation statistics."""
        self._computation_stats["total_computations"] += 1
        self._computation_stats["total_time_ms"] += computation_time_ms
        self._computation_stats["avg_time_ms"] = (
            self._computation_stats["total_time_ms"] /
            self._computation_stats["total_computations"]
        )
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()
        logger.debug("Cleared indicator cache")
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._computation_stats.copy()
        stats.update({
            "cache_size": len(self._result_cache),
            "indicator_count": len(self._indicators),
            "cache_hit_rate": (
                stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
                if (stats["cache_hits"] + stats["cache_misses"]) > 0 else 0.0
            )
        })
        return stats
    def list_indicators(self) -> List[str]:
        """Get List[str] of loaded indicator names."""
        return List[str](self._indicators.keys())
    def get_indicator_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific indicator."""
        if name not in self._indicators:
            return None
        indicator = self._indicators[name]
        return {
            "name": name,
            "type": indicator.__class__.__name__,
            "indicator_type": indicator.indicator_type.value,
            "required_periods": indicator.required_periods,
            "output_names": indicator.output_names,
            "required_columns": indicator.get_required_columns(),
            "parameters": getattr(indicator, 'params', {})
        }
    def benchmark_performance(self,
                            test_data: pd.DataFrame,
                            iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark indicator computation performance.
        Args:
            test_data: Test OHLCV data
            iterations: Number of test iterations
        Returns:
            Performance benchmark results
        """
        logger.info(f"Running performance benchmark with {iterations} iterations")

        # Clear cache for fair benchmarking
        self.clear_cache()

        # Test sequential computation
        seq_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            self.compute_all("BENCHMARK", test_data, parallel=False, use_cache=False)
            end_time = time.perf_counter()
            seq_times.append((end_time - start_time) * 1000)

        # Test parallel computation
        par_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            self.compute_all("BENCHMARK", test_data, parallel=True, use_cache=False)
            end_time = time.perf_counter()
            par_times.append((end_time - start_time) * 1000)

        # Calculate statistics
        import statistics
        results = {
            "test_data_size": len(test_data),
            "indicator_count": len(self._indicators),
            "iterations": iterations,
            "sequential": {
                "avg_time_ms": statistics.mean(seq_times),
                "min_time_ms": min(seq_times),
                "max_time_ms": max(seq_times),
                "std_dev_ms": statistics.stdev(seq_times) if len(seq_times) > 1 else 0
            },
            "parallel": {
                "avg_time_ms": statistics.mean(par_times),
                "min_time_ms": min(par_times),
                "max_time_ms": max(par_times),
                "std_dev_ms": statistics.stdev(par_times) if len(par_times) > 1 else 0
            }
        }

        # Calculate speedup
        if results["sequential"]["avg_time_ms"] > 0:
            speedup = results["sequential"]["avg_time_ms"] / results["parallel"]["avg_time_ms"]
            results["parallel_speedup"] = speedup
        logger.info(f"Benchmark complete. Parallel speedup: {results.get('parallel_speedup', 'N/A'):.2f}x")
        return results
    def calculate_indicators(self,
                           data: pd.DataFrame,
                           indicators: Dict[str, BaseIndicator],
                           parallel: bool = True) -> Dict[str, IndicatorResult]:
        """
        Convenience method to calculate a Set[str] of indicators on data.
        Args:
            data: OHLCV DataFrame
            indicators: Dictionary of indicator name to indicator instance
            parallel: Whether to use parallel computation
        Returns:
            Dictionary mapping indicator names to results
        """

        # Temporarily register indicators
        original_indicators = self._indicators.copy()
        try:

            # Clear and Set[str] new indicators
            self._indicators.clear()
            for name, indicator in indicators.items():
                self.add_indicator(name, indicator)

            # Compute indicators
            return self.compute_all("temp_symbol", data, parallel=parallel)
        finally:

            # Restore original indicators
            self._indicators = original_indicators
