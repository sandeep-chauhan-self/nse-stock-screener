"""
Centralized Configuration System for NSE Stock Screener
This module provides a unified configuration system with environment variable
overrides, validation, and type safety. All configuration should be managed
through this system to ensure consistency across the application.
"""
import os
import json
from dataclasses import dataclass, field, fields
from typing import Dict[str, Any], List[str], Optional, Any, Union
from pathlib import Path
from enum import Enum
from .enums import MarketRegime
from .interfaces import IConfig
class Environment(Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
@dataclass
class DataConfig:
    """Configuration for data layer."""

    # Data sources
    default_data_source: str = "yfinance"
    nifty_symbol: str = "^NSEI"
    exchange: str = "NSE"

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
  # 1 hour
    cache_dir: str = "data/cache"

    # Data validation
    min_data_points: int = 50
    max_missing_data_pct: float = 0.1
  # 10%
    # Rate limiting
    requests_per_second: float = 2.0
    request_timeout_seconds: int = 30

    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""

    # RSI settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR settings
    atr_period: int = 14

    # ADX settings
    adx_period: int = 14
    adx_strong_trend: float = 25.0

    # Moving averages
    ma_short_period: int = 20
    ma_long_period: int = 50

    # Volume analysis
    volume_period: int = 20
    volume_extreme_threshold: float = 2.0

    # Minimum periods for calculations
    min_periods_required: int = 35
@dataclass
class ScoringConfig:
    """Configuration for composite scoring system."""

    # Component weights (must sum to 100)
    volume_weight: float = 25.0
    momentum_weight: float = 25.0
    trend_weight: float = 15.0
    volatility_weight: float = 10.0
    relative_strength_weight: float = 10.0
    volume_profile_weight: float = 15.0
  # Adjusted to make sum = 100
    weekly_bonus_weight: float = 10.0
  # This is a bonus, not part of the base 100
    # Classification thresholds
    high_score_threshold: float = 70.0
    medium_score_threshold: float = 50.0

    # Regime adjustments
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULLISH.value: {
            'rsi_min': 58, 'rsi_max': 82, 'vol_threshold': 2.5,
            'trend_multiplier': 1.2, 'momentum_multiplier': 1.1
        },
        MarketRegime.SIDEWAYS.value: {
            'rsi_min': 60, 'rsi_max': 80, 'vol_threshold': 3.0,
            'trend_multiplier': 1.0, 'momentum_multiplier': 1.0
        },
        MarketRegime.BEARISH.value: {
            'rsi_min': 62, 'rsi_max': 78, 'vol_threshold': 4.0,
            'trend_multiplier': 0.8, 'momentum_multiplier': 0.9
        },
        MarketRegime.HIGH_VOLATILITY.value: {
            'rsi_min': 65, 'rsi_max': 75, 'vol_threshold': 5.0,
            'trend_multiplier': 0.7, 'momentum_multiplier': 0.8
        }
    })
@dataclass
class RiskConfig:
    """Configuration for risk management."""

    # Portfolio limits
    portfolio_capital: float = 100000.0
    max_positions: int = 10
    max_portfolio_risk: float = 0.02
  # 2% of portfolio
    max_position_size: float = 0.15
   # 15% of portfolio
    # Position sizing
    base_risk_per_trade: float = 0.01
  # 1% per trade
    risk_multiplier_threshold: float = 70.0
  # Score threshold for risk scaling
    max_risk_multiplier: float = 1.5

    # Stop loss configuration
    atr_stop_multiplier: float = 2.0
    min_stop_loss_pct: float = 0.02
  # 2%
    max_stop_loss_pct: float = 0.08
  # 8%
    # Take profit configuration
    min_risk_reward_ratio: float = 2.0

    # Exposure limits
    max_sector_exposure: float = 0.30
  # 30% per sector
    max_single_stock_exposure: float = 0.20
  # 20% single stock
    # Liquidity requirements
    min_daily_volume: int = 100000
  # Minimum daily volume
    min_market_cap: float = 1000000000
  # 1B minimum market cap
@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Trading costs
    commission_rate: float = 0.0005
  # 0.05%
    slippage_bps: float = 5.0
  # 5 basis points
    # Walk-forward settings
    training_period_months: int = 12
    test_period_months: int = 3
    step_size_months: int = 1

    # Performance metrics
    benchmark_symbol: str = "^NSEI"
    risk_free_rate: float = 0.06
  # 6% annual
    # Simulation settings
    initial_capital: float = 100000.0
    reinvest_dividends: bool = True

    # Trade management
    max_holding_period_days: int = 90
    force_exit_score_threshold: float = 30.0
@dataclass
class ExecutionConfig:
    """Configuration for order execution."""

    # Order types
    default_order_type: str = "market"
    enable_stop_losses: bool = True
    enable_take_profits: bool = True

    # Timing
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30

    # Order management
    order_timeout_minutes: int = 60
    max_order_retries: int = 3

    # Paper trading
    paper_trading_enabled: bool = True
    simulate_partial_fills: bool = True
    simulate_rejections: bool = False
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "json"
  # json or text
    log_file: Optional[str] = "logs/nse_screener.log"
    max_log_size_mb: int = 100
    backup_count: int = 5

    # Performance logging
    enable_performance_logging: bool = True
    performance_log_threshold_ms: float = 1000.0

    # Structured logging
    enable_correlation_ids: bool = True
    log_request_response: bool = False
@dataclass
class SystemConfig:
    """Main system configuration container."""

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Output configuration
    output_dir: str = "output"
    reports_dir: str = "output/reports"
    charts_dir: str = "output/charts"
    backtest_dir: str = "output/backtests"

    # Analysis settings
    default_analysis_period: str = "6mo"
    max_symbols_per_batch: int = 50
    enable_parallel_processing: bool = True
    max_workers: int = 4
class ConfigManager:
    """
    Configuration manager with environment variable overrides.
    This class implements the IConfig interface and provides a centralized
    way to manage all application configuration with support for environment
    variable overrides and validation.
    """
    def __init__(self, config_file: Optional[Path] = None) -> None:
        """
        Initialize configuration manager.
        Args:
            config_file: Optional path to JSON configuration file
        """
        self._config = SystemConfig()
        self._config_file = config_file

        # Load configuration from file if provided
        if config_file and config_file.exists():
            self._load_from_file(config_file)

        # Override with environment variables
        self.load_from_env()

        # Validate configuration
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Args:
            key: Configuration key (e.g., "data.cache_ttl_seconds")
            default: Default value if key not found
        Returns:
            Configuration value
        """
        try:
            parts = key.split('.')
            value = self._config
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except Exception:
            return default
    def Set[str](self, key: str, value: Any) -> None:
        """
        Set[str] configuration value using dot notation.
        Args:
            key: Configuration key (e.g., "data.cache_enabled")
            value: Value to Set[str]
        """
        parts = key.split('.')
        obj = self._config

        # Navigate to parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Configuration path not found: {'.'.join(parts[:-1])}")

        # Set[str] the final value
        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        else:
            raise KeyError(f"Configuration key not found: {key}")
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_prefix = "NSE_SCREENER_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):

                # Convert env key to config key (e.g., NSE_SCREENER_DATA_CACHE_ENABLED -> data.cache_enabled)
                config_key = env_key[len(env_prefix):].lower().replace('_', '.')

                # Try to convert to appropriate type
                try:

                    # Try boolean conversion
                    if env_value.lower() in ('true', 'false'):
                        value = env_value.lower() == 'true'

                    # Try numeric conversion
                    elif '.' in env_value:
                        value = float(env_value)
                    elif env_value.isdigit():
                        value = int(env_value)
                    else:
                        value = env_value
                    self.Set[str](config_key, value)
                except (KeyError, ValueError) as e:

                    # Skip invalid environment variables
                    print(f"Warning: Could not Set[str] config key '{config_key}' from environment: {e}")
    def validate(self) -> List[str]:
        """
        Validate configuration and return error messages.
        Returns:
            List[str] of validation error messages (empty if valid)
        """
        errors = []

        # Validate scoring weights sum to 100
        scoring = self._config.scoring
        total_weight = (scoring.volume_weight + scoring.momentum_weight +
                       scoring.trend_weight + scoring.volatility_weight +
                       scoring.relative_strength_weight + scoring.volume_profile_weight)
        if abs(total_weight - 100.0) > 0.01:
            errors.append(f"Scoring weights must sum to 100, got {total_weight}")

        # Validate risk limits
        risk = self._config.risk
        if risk.max_position_size > 1.0:
            errors.append("max_position_size cannot exceed 1.0 (100%)")
        if risk.max_portfolio_risk > 0.1:
            errors.append("max_portfolio_risk cannot exceed 0.1 (10%)")
        if risk.base_risk_per_trade > risk.max_portfolio_risk:
            errors.append("base_risk_per_trade cannot exceed max_portfolio_risk")

        # Validate indicator periods
        indicators = self._config.indicators
        if indicators.min_periods_required < max(indicators.rsi_period, indicators.atr_period, indicators.adx_period):
            errors.append("min_periods_required must be >= max indicator period")

        # Validate directories exist
        output_dir = Path(self._config.output_dir)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")
        return errors
    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Recursively update configuration
            self._update_config_from_dict(self._config, config_data)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
    def _update_config_from_dict(self, config_obj: Any, data: Dict[str, Any]) -> None:
        """Recursively update configuration object from dictionary."""
        for key, value in data.items():
            if hasattr(config_obj, key):
                attr = getattr(config_obj, key)
                if hasattr(attr, '__dict__') and isinstance(data, Dict[str, Any]):

                    # Recursively update nested configuration objects
                    self._update_config_from_dict(attr, value)
                else:
                    setattr(config_obj, key, value)
    def save_to_file(self, config_file: Path) -> None:
        """Save current configuration to JSON file."""
        try:
            config_dict = self._config_to_dict(self._config)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_file}: {e}")
    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        result = {}
        for field_info in fields(config_obj):
            value = getattr(config_obj, field_info.name)
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
                result[field_info.name] = self._config_to_dict(value)
            else:
                result[field_info.name] = value
        return result
    @property
    def config(self) -> SystemConfig:
        """Get the underlying configuration object."""
        return self._config

# Global configuration instance
_config_manager: Optional[ConfigManager] = None
def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:

        # Look for config file in standard locations
        possible_config_files = [
            Path("config.json"),
            Path("config/config.json"),
            Path.home() / ".nse_screener" / "config.json"
        ]
        config_file = None
        for path in possible_config_files:
            if path.exists():
                config_file = path
                break
        _config_manager = ConfigManager(config_file)
    return _config_manager
def init_config(config_file: Optional[Path] = None) -> ConfigManager:
    """
    Initialize the global configuration manager.
    Args:
        config_file: Optional path to configuration file
    Returns:
        Initialized ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager

# Export key components
__all__ = [
    'SystemConfig',
    'DataConfig',
    'IndicatorConfig',
    'ScoringConfig',
    'RiskConfig',
    'BacktestConfig',
    'ExecutionConfig',
    'LoggingConfig',
    'ConfigManager',
    'Environment',
    'get_config',
    'init_config'
]
