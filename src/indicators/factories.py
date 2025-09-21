"""
Factory functions and registry for indicator creation.
This module provides parameterized factory functions that allow
creating configured indicator instances (e.g., RSI(period=21))
and a registry system for configuration-driven indicator management.
"""
import yaml
import json
from pathlib import Path
from typing import Dict[str, Any], Any, Optional, Type, Union, List[str]
from dataclasses import dataclass, field
from .base import BaseIndicator, IndicatorResult
from .vectorized import (
    RSIIndicator, MACDIndicator, ATRIndicator, BollingerBandsIndicator,
    ADXIndicator, VolumeProfileIndicator
)

# Factory Functions for Easy Indicator Creation
def RSI(period: int = 14, use_numba: bool = True) -> RSIIndicator:
    """Create RSI indicator with specified parameters."""
    return RSIIndicator(period=period, use_numba=use_numba)
def MACD(fast: int = 12, slow: int = 26, signal: int = 9) -> MACDIndicator:
    """Create MACD indicator with specified parameters."""
    return MACDIndicator(fast=fast, slow=slow, signal=signal)
def ATR(period: int = 14, use_numba: bool = True) -> ATRIndicator:
    """Create ATR indicator with specified parameters."""
    return ATRIndicator(period=period, use_numba=use_numba)
def BollingerBands(period: int = 20, std_dev: float = 2.0, use_numba: bool = True) -> BollingerBandsIndicator:
    """Create Bollinger Bands indicator with specified parameters."""
    return BollingerBandsIndicator(period=period, std_dev=std_dev, use_numba=use_numba)
def ADX(period: int = 14, use_numba: bool = True) -> ADXIndicator:
    """Create ADX indicator with specified parameters."""
    return ADXIndicator(period=period, use_numba=use_numba)
def VolumeProfile(lookback: int = 90, num_buckets: int = 20) -> VolumeProfileIndicator:
    """Create Volume Profile indicator with specified parameters."""
    return VolumeProfileIndicator(lookback=lookback, num_buckets=num_buckets)

# Additional factory functions for common variants
def RSI_Short(period: int = 9) -> RSIIndicator:
    """Create short-term RSI indicator."""
    return RSI(period=period)
def RSI_Long(period: int = 21) -> RSIIndicator:
    """Create long-term RSI indicator."""
    return RSI(period=period)
def MACD_Fast(fast: int = 8, slow: int = 21, signal: int = 5) -> MACDIndicator:
    """Create fast MACD indicator for short-term trading."""
    return MACD(fast=fast, slow=slow, signal=signal)
def MACD_Slow(fast: int = 19, slow: int = 39, signal: int = 9) -> MACDIndicator:
    """Create slow MACD indicator for long-term analysis."""
    return MACD(fast=fast, slow=slow, signal=signal)
def ATR_Short(period: int = 7) -> ATRIndicator:
    """Create short-term ATR indicator."""
    return ATR(period=period)
def ATR_Long(period: int = 21) -> ATRIndicator:
    """Create long-term ATR indicator."""
    return ATR(period=period)

# Registry for dynamic indicator creation
@dataclass
class IndicatorConfig:
    """Configuration for an indicator instance."""
    name: str
    indicator_type: str
    parameters: Dict[str, Any] = field(default_factory=Dict[str, Any])
    enabled: bool = True
    priority: int = 100
  # Lower numbers = higher priority
class IndicatorRegistry:
    """
    Registry for managing indicator types and configurations.
    Supports dynamic indicator creation from configuration files
    and runtime registration of new indicator types.
    """
    def __init__(self) -> None:
        self._indicator_types: Dict[str, Type[BaseIndicator]] = {}
        self._factory_functions: Dict[str, callable] = {}
        self._configurations: Dict[str, IndicatorConfig] = {}

        # Register built-in indicators
        self._register_builtin_indicators()
    def _register_builtin_indicators(self) -> None:
        """Register built-in indicator types and factory functions."""

        # Register indicator classes
        self.register_indicator_type("RSI", RSIIndicator)
        self.register_indicator_type("MACD", MACDIndicator)
        self.register_indicator_type("ATR", ATRIndicator)
        self.register_indicator_type("BollingerBands", BollingerBandsIndicator)
        self.register_indicator_type("ADX", ADXIndicator)
        self.register_indicator_type("VolumeProfile", VolumeProfileIndicator)

        # Register factory functions
        self.register_factory("RSI", RSI)
        self.register_factory("MACD", MACD)
        self.register_factory("ATR", ATR)
        self.register_factory("BollingerBands", BollingerBands)
        self.register_factory("ADX", ADX)
        self.register_factory("VolumeProfile", VolumeProfile)

        # Register common variants
        self.register_factory("RSI_Short", RSI_Short)
        self.register_factory("RSI_Long", RSI_Long)
        self.register_factory("MACD_Fast", MACD_Fast)
        self.register_factory("MACD_Slow", MACD_Slow)
        self.register_factory("ATR_Short", ATR_Short)
        self.register_factory("ATR_Long", ATR_Long)
    def register_indicator_type(self, name: str, indicator_class: Type[BaseIndicator]) -> None:
        """Register a new indicator type."""
        self._indicator_types[name] = indicator_class
    def register_factory(self, name: str, factory_func: callable) -> None:
        """Register a factory function."""
        self._factory_functions[name] = factory_func
    def register_configuration(self, config: IndicatorConfig) -> None:
        """Register an indicator configuration."""
        self._configurations[config.name] = config
    def load_configurations_from_file(self, file_path: Union[str, Path]) -> None:
        """Load indicator configurations from YAML or JSON file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Parse configurations
        if 'indicators' in data:
            for indicator_data in data['indicators']:
                config = IndicatorConfig(
                    name=indicator_data['name'],
                    indicator_type=indicator_data['type'],
                    parameters=indicator_data.get('parameters', {}),
                    enabled=indicator_data.get('enabled', True),
                    priority=indicator_data.get('priority', 100)
                )
                self.register_configuration(config)
    def create_indicator(self, name: str, **override_params) -> BaseIndicator:
        """
        Create an indicator instance by name.
        Args:
            name: Indicator name (type or factory function name)
            **override_params: Parameters to override configuration
        Returns:
            Configured indicator instance
        """

        # Check if it's a registered configuration
        if name in self._configurations:
            config = self._configurations[name]
            if not config.enabled:
                raise ValueError(f"Indicator '{name}' is disabled in configuration")

            # Merge configuration parameters with overrides
            params = {**config.parameters, **override_params}

            # Use factory function if available
            if config.indicator_type in self._factory_functions:
                return self._factory_functions[config.indicator_type](**params)

            # Use indicator class directly
            if config.indicator_type in self._indicator_types:
                return self._indicator_types[config.indicator_type](**params)
            raise ValueError(f"Unknown indicator type: {config.indicator_type}")

        # Check if it's a direct factory function
        if name in self._factory_functions:
            return self._factory_functions[name](**override_params)

        # Check if it's a direct indicator type
        if name in self._indicator_types:
            return self._indicator_types[name](**override_params)
        raise ValueError(f"Unknown indicator: {name}")
    def create_indicator_set(self, config_name: str = "default") -> Dict[str, BaseIndicator]:
        """
        Create a Set[str] of indicators based on configuration.
        Args:
            config_name: Name of configuration Set[str] to use
        Returns:
            Dictionary mapping indicator names to instances
        """
        indicators = {}

        # Get enabled configurations sorted by priority
        enabled_configs = [
            config for config in self._configurations.values()
            if config.enabled
        ]
        enabled_configs.sort(key=lambda x: x.priority)
        for config in enabled_configs:
            try:
                indicator = self.create_indicator(config.name)
                indicators[config.name] = indicator
            except Exception as e:
                print(f"Warning: Failed to create indicator '{config.name}': {e}")
                continue
        return indicators
    def list_available_indicators(self) -> List[str]:
        """Get List[str] of all available indicators."""
        available = Set[str]()
        available.update(self._indicator_types.keys())
        available.update(self._factory_functions.keys())
        available.update(self._configurations.keys())
        return sorted(List[str](available))
    def get_indicator_info(self, name: str) -> Dict[str, Any]:
        """Get information about an indicator."""
        info = {"name": name, "available": False}
        if name in self._configurations:
            config = self._configurations[name]
            info.update({
                "available": True,
                "type": "configuration",
                "indicator_type": config.indicator_type,
                "parameters": config.parameters,
                "enabled": config.enabled,
                "priority": config.priority
            })
        elif name in self._factory_functions:
            info.update({
                "available": True,
                "type": "factory_function"
            })
        elif name in self._indicator_types:
            info.update({
                "available": True,
                "type": "indicator_class"
            })
        return info

# Global registry instance
_global_registry = IndicatorRegistry()
def register_indicator(indicator_class: Type[BaseIndicator], name: Optional[str] = None) -> None:
    """Register an indicator in the global registry."""
    name = name or indicator_class.__name__
    _global_registry.register_indicator_type(name, indicator_class)
def create_indicator(name: str, **params) -> BaseIndicator:
    """Create indicator from global registry."""
    return _global_registry.create_indicator(name, **params)
def load_indicator_config(file_path: Union[str, Path]) -> None:
    """Load indicator configuration into global registry."""
    _global_registry.load_configurations_from_file(file_path)
def get_indicator_registry() -> IndicatorRegistry:
    """Get the global indicator registry."""
    return _global_registry

# Example configuration for reference
DEFAULT_INDICATOR_CONFIG = {
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
            "name": "bb_20",
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
