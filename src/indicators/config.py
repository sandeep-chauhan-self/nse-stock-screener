"""
Configuration module for the indicator engine.
This module provides configuration classes and utilities for
managing indicator engine settings, validation parameters,
and performance tuning options.
"""
from dataclasses import dataclass, field
from typing import Dict[str, Any], Any, Optional, Union, List[str]
from pathlib import Path
import json
from enum import Enum
class PerformanceMode(Enum):
    """Performance optimization modes."""
    ACCURACY = "accuracy"
      # Prioritize accuracy over speed
    BALANCED = "balanced"
      # Balance accuracy and speed
    SPEED = "speed"
           # Prioritize speed over accuracy
@dataclass
class IndicatorEngineConfig:
    """Configuration for the indicator engine."""

    # Performance settings
    max_workers: int = 4
    computation_timeout: float = 30.0
  # seconds
    enable_numba: bool = True
    performance_mode: PerformanceMode = PerformanceMode.BALANCED

    # Caching settings
    enable_caching: bool = True
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
  # 1 hour
    # Validation settings
    strict_validation: bool = True
    min_data_quality_threshold: float = 0.8
    max_missing_data_ratio: float = 0.1

    # Computation settings
    default_periods: Dict[str, int] = field(default_factory=lambda: {
        "rsi": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr": 14,
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "adx": 14,
        "volume_profile_lookback": 90
    })

    # Error handling
    continue_on_error: bool = True
    log_computation_errors: bool = True
    return_partial_results: bool = True
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IndicatorEngineConfig':
        """Create configuration from dictionary."""

        # Handle enum conversion
        if 'performance_mode' in config_dict:
            if isinstance(config_dict['performance_mode'], str):
                config_dict['performance_mode'] = PerformanceMode(config_dict['performance_mode'])
        return cls(**config_dict)
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'IndicatorEngineConfig':
        """Load configuration from JSON file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    def get_numba_settings(self) -> Dict[str, bool]:
        """Get Numba optimization settings based on performance mode."""
        if self.performance_mode == PerformanceMode.SPEED:
            return {"use_numba": True, "parallel": True}
        elif self.performance_mode == PerformanceMode.BALANCED:
            return {"use_numba": self.enable_numba, "parallel": False}
        else:
  # ACCURACY
            return {"use_numba": False, "parallel": False}
@dataclass
class ValidationConfig:
    """Configuration for data validation."""

    # Data quality thresholds
    min_data_points: int = 100
    max_missing_ratio: float = 0.05
    max_outlier_ratio: float = 0.02

    # Price validation
    min_price: float = 0.01
    max_price_change_ratio: float = 0.50
  # 50% single-day change limit
    # Volume validation
    min_volume: int = 1
    max_volume_spike_ratio: float = 20.0
  # 20x normal volume
    # OHLC consistency checks
    allow_equal_ohlc: bool = True
  # Allow O=H=L=C (common for illiquid stocks)
    ohlc_tolerance: float = 0.001
  # 0.1% tolerance for OHLC relationships
    @classmethod
    def strict(cls) -> 'ValidationConfig':
        """Create strict validation configuration."""
        return cls(
            min_data_points=200,
            max_missing_ratio=0.02,
            max_outlier_ratio=0.01,
            max_price_change_ratio=0.25,
            max_volume_spike_ratio=10.0,
            allow_equal_ohlc=False,
            ohlc_tolerance=0.0001
        )
    @classmethod
    def permissive(cls) -> 'ValidationConfig':
        """Create permissive validation configuration."""
        return cls(
            min_data_points=50,
            max_missing_ratio=0.15,
            max_outlier_ratio=0.05,
            max_price_change_ratio=1.0,
            max_volume_spike_ratio=50.0,
            allow_equal_ohlc=True,
            ohlc_tolerance=0.01
        )
@dataclass
class StressTestConfig:
    """Configuration for stress testing indicators."""

    # Test scenarios
    test_2008_crash: bool = True
    test_2020_covid: bool = True
    test_2022_volatility: bool = True
    test_custom_scenarios: List[str] = field(default_factory=List[str])

    # Test parameters
    min_data_points: int = 252
  # 1 year of trading days
    stress_event_window: int = 60
  # Days around stress event to analyze
    # Validation thresholds
    max_indicator_nan_ratio: float = 0.1
  # 10% NaN values allowed during stress
    min_indicator_stability: float = 0.7
  # Stability score threshold
    max_computation_time_ms: float = 1000.0
  # Per indicator timeout
    # Expected indicator ranges during stress events
    expected_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RSI": {"min": 0, "max": 100},
        "ATR": {"min": 0, "max": 50},
  # % of price
        "ADX": {"min": 0, "max": 100},
        "MACD": {"min": -10, "max": 10}
  # Relative to price
    })
def load_indicator_config(config_path: Optional[Union[str, Path]] = None) -> IndicatorEngineConfig:
    """
    Load indicator configuration from file or return default.
    Args:
        config_path: Path to configuration file (optional)
    Returns:
        IndicatorEngineConfig instance
    """
    if config_path is None:
        return IndicatorEngineConfig()
    try:
        return IndicatorEngineConfig.from_file(config_path)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        print("Using default configuration")
        return IndicatorEngineConfig()
def create_default_config_file(file_path: Union[str, Path]) -> None:
    """Create a default configuration file."""
    config = IndicatorEngineConfig()
    config.save_to_file(file_path)
    print(f"Created default configuration file: {file_path}")

# Default configurations for different use cases
def get_production_config() -> IndicatorEngineConfig:
    """Get production-optimized configuration."""
    return IndicatorEngineConfig(
        max_workers=8,
        computation_timeout=60.0,
        enable_numba=True,
        performance_mode=PerformanceMode.BALANCED,
        enable_caching=True,
        max_cache_size=5000,
        strict_validation=True,
        continue_on_error=True,
        log_computation_errors=True
    )
def get_development_config() -> IndicatorEngineConfig:
    """Get development-friendly configuration."""
    return IndicatorEngineConfig(
        max_workers=2,
        computation_timeout=30.0,
        enable_numba=False,
  # Easier debugging
        performance_mode=PerformanceMode.ACCURACY,
        enable_caching=False,
  # Fresh results for testing
        strict_validation=True,
        continue_on_error=False,
  # Fail fast for debugging
        log_computation_errors=True
    )
def get_performance_config() -> IndicatorEngineConfig:
    """Get maximum performance configuration."""
    return IndicatorEngineConfig(
        max_workers=16,
        computation_timeout=10.0,
        enable_numba=True,
        performance_mode=PerformanceMode.SPEED,
        enable_caching=True,
        max_cache_size=10000,
        strict_validation=False,
        continue_on_error=True,
        log_computation_errors=False,
  # Reduce logging overhead
        min_data_quality_threshold=0.6,
  # More permissive
        max_missing_data_ratio=0.2
    )

# Example configuration files
DEFAULT_CONFIG_TEMPLATE = {
    "max_workers": 4,
    "computation_timeout": 30.0,
    "enable_numba": True,
    "performance_mode": "balanced",
    "enable_caching": True,
    "max_cache_size": 1000,
    "cache_ttl_seconds": 3600,
    "strict_validation": True,
    "min_data_quality_threshold": 0.8,
    "max_missing_data_ratio": 0.1,
    "default_periods": {
        "rsi": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr": 14,
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "adx": 14,
        "volume_profile_lookback": 90
    },
    "continue_on_error": True,
    "log_computation_errors": True,
    "return_partial_results": True
}
INDICATOR_CONFIG_TEMPLATE = {
    "indicators": [
        {
            "name": "rsi_14",
            "type": "RSI",
            "parameters": {"period": 14, "use_numba": True},
            "enabled": True,
            "priority": 10
        },
        {
            "name": "rsi_21",
            "type": "RSI",
            "parameters": {"period": 21, "use_numba": True},
            "enabled": True,
            "priority": 20
        },
        {
            "name": "macd_standard",
            "type": "MACD",
            "parameters": {"fast": 12, "slow": 26, "signal": 9},
            "enabled": True,
            "priority": 30
        },
        {
            "name": "atr_14",
            "type": "ATR",
            "parameters": {"period": 14, "use_numba": True},
            "enabled": True,
            "priority": 40
        },
        {
            "name": "bollinger_bands",
            "type": "BollingerBands",
            "parameters": {"period": 20, "std_dev": 2.0, "use_numba": True},
            "enabled": True,
            "priority": 50
        },
        {
            "name": "adx_14",
            "type": "ADX",
            "parameters": {"period": 14, "use_numba": True},
            "enabled": True,
            "priority": 60
        },
        {
            "name": "volume_profile",
            "type": "VolumeProfile",
            "parameters": {"lookback": 90, "num_buckets": 20},
            "enabled": True,
            "priority": 70
        }
    ]
}
