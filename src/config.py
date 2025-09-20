"""
Centralized Configuration Management
Provides a single source of truth for all system configuration with validation,
environment variable support, and type safety.
"""

from pathlib import Path
import json
import logging
import os

from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """
    Centralized system configuration with canonical field names and validation.
    
    This class consolidates all configuration across modules to eliminate
    inconsistent naming and provide a single source of truth.
    """
    
    # Portfolio & Capital Management
    portfolio_capital: float = 1000000.0  # Initial capital (10 Lakh)
    currency: str = "INR"
    
    # Risk Management
    risk_per_trade: float = 0.01          # 1% risk per trade
    max_portfolio_risk: float = 0.20      # 20% max portfolio at risk
    max_position_size: float = 0.10       # 10% max per position (as fraction of capital)
    max_daily_loss: float = 0.02          # 2% max daily loss
    max_monthly_loss: float = 0.08        # 8% max monthly loss
    max_sector_exposure: float = 0.30     # 30% max per sector
    max_positions: int = 10               # Maximum concurrent positions (canonical name)
    min_risk_reward_ratio: float = 2.0    # Minimum risk-reward ratio
    
    # Stop Loss & Position Management
    stop_loss_atr_multiplier: float = 2.0     # ATR multiplier for initial stop
    breakeven_trigger_ratio: float = 1.5      # Move to breakeven after 1.5x risk
    trailing_stop_atr_multiplier: float = 1.0 # Trail at 1x ATR
    take_profit_multiplier: float = 2.5       # Take profit at 2.5x risk
    
    # Transaction Costs & Execution (Enhanced Model)
    # Commission Structure (per side)
    brokerage_rate: float = 0.0003       # 0.03% brokerage per side
    stt_rate: float = 0.00025           # 0.025% STT (Securities Transaction Tax)
    exchange_charges: float = 0.0000345  # 0.00345% exchange charges
    gst_rate: float = 0.18              # 18% GST on brokerage + exchange charges
    stamp_duty_rate: float = 0.00015    # 0.015% stamp duty (buy side only)
    
    # Legacy combined rate (for backward compatibility)
    transaction_cost: float = 0.0005    # Calculated: ~0.05% total per side
    
    # Slippage Model (Price Impact)
    slippage_model: str = "adaptive"     # "fixed", "adaptive", "liquidity_based"
    base_slippage_bps: float = 2.0      # Base slippage in basis points (0.02%)
    slippage_per_share: float = 0.05    # Fixed slippage in Rs per share
    market_impact_factor: float = 0.1   # Market impact coefficient
    
    # Execution Model Parameters
    partial_fill_enabled: bool = False  # Enable partial fill simulation
    min_fill_ratio: float = 0.5        # Minimum % of order that gets filled
    liquidity_impact_threshold: float = 0.01  # Order size % of volume for impact
    max_spread_bps: float = 50.0        # Max bid-ask spread in basis points
    
    # Backtesting Parameters
    max_holding_days: int = 30           # Maximum days to hold position
    min_score_threshold: int = 45        # Minimum composite score for entry
    walk_forward_window: int = 252       # 1 year training window (trading days)
    test_window: int = 63               # 3 months test window (trading days)
    
    # Data & Analysis
    batch_size: int = 50                # Stocks per batch in analysis
    request_timeout: int = 10           # Seconds between API requests
    data_cache_days: int = 1            # Days to cache data
    min_trading_volume: int = 100000    # Minimum daily volume for analysis
    
    # Scoring & Indicators
    regime_detection_window: int = 50   # Days for market regime detection
    volatility_window: int = 20         # Days for volatility calculations
    volume_threshold_multiplier: float = 1.5  # Volume threshold for extremes
    
    # Correlation & Diversification
    correlation_limit: float = 0.7      # Max correlation between positions
    max_sector_positions: int = 3       # Max positions per sector
    
    # Advanced Risk Controls & Position Sizing
    max_risk_multiplier: float = 2.0    # Maximum risk multiplier cap
    min_risk_multiplier: float = 0.25   # Minimum risk multiplier floor
    risk_multiplier_high_score: float = 1.8   # Multiplier for scores >= 70
    risk_multiplier_medium_score: float = 1.0 # Multiplier for scores >= 50
    risk_multiplier_low_score: float = 0.5    # Multiplier for scores < 50
    min_stop_atr_ratio: float = 0.5     # Minimum stop distance as ratio of ATR
    max_stop_atr_ratio: float = 4.0     # Maximum stop distance as ratio of ATR
    kelly_fraction_conservative: float = 0.25 # Conservative Kelly fraction cap
    volatility_parity_enabled: bool = False  # Enable volatility parity sizing
    lot_size_enforcement: bool = True   # Enforce NSE lot size constraints
    min_lot_size: int = 1              # Minimum lot size (shares)
    default_lot_size: int = 1          # Default lot size for unlisted symbols
    max_position_concentration: float = 0.05  # Max 5% in single position for high vol
    liquidity_check_enabled: bool = True     # Enable liquidity-based sizing
    min_avg_volume_multiple: float = 0.01    # Position size as % of avg volume
    
    # Performance & Monitoring
    benchmark_symbol: str = "^NSEI"     # NIFTY 50 as benchmark
    performance_lookback_days: int = 252 # 1 year for performance metrics
    
    # File Paths & Output
    data_directory: str = "data"
    output_directory: str = "output"
    symbols_file: str = "nse_only_symbols.txt"
    
    # Environment-specific overrides
    environment: str = "development"     # development, testing, production
    debug_mode: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration values and ensure consistency
        """
        errors = []
        
        # Portfolio validation
        if self.portfolio_capital <= 0:
            errors.append("portfolio_capital must be positive")
        
        # Risk validation
        if not 0 < self.risk_per_trade <= 0.1:
            errors.append("risk_per_trade must be between 0 and 0.1 (10%)")
        
        if not 0 < self.max_portfolio_risk <= 1.0:
            errors.append("max_portfolio_risk must be between 0 and 1.0")
        
        if not 0 < self.max_position_size <= 0.5:
            errors.append("max_position_size must be between 0 and 0.5 (50%)")
        
        if self.max_positions <= 0:
            errors.append("max_positions must be positive")
        
        if self.min_risk_reward_ratio < 1.0:
            errors.append("min_risk_reward_ratio should be >= 1.0")
        
        # Enhanced cost validation
        if not 0 <= self.brokerage_rate <= 0.01:
            errors.append("brokerage_rate should be between 0 and 1%")
        
        if not 0 <= self.stt_rate <= 0.01:
            errors.append("stt_rate should be between 0 and 1%")
        
        if not 0 <= self.transaction_cost <= 0.02:
            errors.append("transaction_cost should be between 0 and 2%")
        
        if self.slippage_model not in ["fixed", "adaptive", "liquidity_based"]:
            errors.append("slippage_model must be 'fixed', 'adaptive', or 'liquidity_based'")
        
        if not 0 <= self.base_slippage_bps <= 100:
            errors.append("base_slippage_bps should be between 0 and 100 basis points")
        
        if self.slippage_per_share < 0:
            errors.append("slippage_per_share must be non-negative")
        
        # Timing validation
        if self.max_holding_days <= 0:
            errors.append("max_holding_days must be positive")
        
        if not 0 <= self.min_score_threshold <= 100:
            errors.append("min_score_threshold must be between 0 and 100")
        
        # Logical consistency checks
        max_theoretical_exposure = self.max_position_size * self.max_positions
        if max_theoretical_exposure > 2.0:  # Allow up to 200% theoretical exposure (with margin)
            errors.append(
                f"max_position_size ({self.max_position_size}) * "
                f"max_positions ({self.max_positions}) = {max_theoretical_exposure:.1f} "
                f"exceeds reasonable exposure limit (200%)"
            )
        
        if self.risk_per_trade > self.max_position_size:
            errors.append("risk_per_trade should not exceed max_position_size")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def to_json(self, file_path: Optional[str] = None) -> str:
        """Export configuration to JSON"""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if file_path:
            Path(file_path).write_text(json_str)
            logger.info(f"Configuration exported to {file_path}")
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create configuration from dictionary with validation"""
        # Filter only known fields
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        config_dict = json.loads(Path(file_path).read_text())
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_environment(cls, prefix: str = "NSE_") -> 'SystemConfig':
        """
        Create configuration with environment variable overrides
        
        Environment variables should be prefixed (default: NSE_) and in UPPER_CASE.
        Example: NSE_PORTFOLIO_CAPITAL=2000000
        """
        config = cls()  # Start with defaults
        
        # Override with environment variables
        for field_info in fields(cls):
            env_var = f"{prefix}{field_info.name.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                try:
                    # Type conversion based on field type
                    if field_info.type == int:
                        value = int(env_value)
                    elif field_info.type == float:
                        value = float(env_value)
                    elif field_info.type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = env_value
                    
                    setattr(config, field_info.name, value)
                    logger.info(f"Override from environment: {field_info.name} = {value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment value for {env_var}: {env_value} ({e})")
        
        return config

class ConfigManager:
    """
    Configuration manager providing convenient access patterns and environment handling
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self._config = config or SystemConfig()
    
    @property
    def config(self) -> SystemConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise ValueError(f"Unknown configuration field: {key}")
        
        # Re-validate after updates
        self._config.validate()
    
    def get_risk_settings(self) -> Dict[str, Any]:
        """Get risk management specific settings"""
        return {
            'risk_per_trade': self._config.risk_per_trade,
            'max_portfolio_risk': self._config.max_portfolio_risk,
            'max_position_size': self._config.max_position_size,
            'max_positions': self._config.max_positions,
            'min_risk_reward_ratio': self._config.min_risk_reward_ratio,
            'stop_loss_atr_multiplier': self._config.stop_loss_atr_multiplier,
            'trailing_stop_atr_multiplier': self._config.trailing_stop_atr_multiplier,
            'correlation_limit': self._config.correlation_limit
        }
    
    def get_backtest_settings(self) -> Dict[str, Any]:
        """Get backtesting specific settings"""
        return {
            'initial_capital': self._config.portfolio_capital,
            'risk_per_trade': self._config.risk_per_trade,
            # Enhanced transaction cost settings
            'brokerage_rate': self._config.brokerage_rate,
            'stt_rate': self._config.stt_rate,
            'exchange_charges': self._config.exchange_charges,
            'gst_rate': self._config.gst_rate,
            'stamp_duty_rate': self._config.stamp_duty_rate,
            'transaction_cost': self._config.transaction_cost,  # Legacy
            'slippage_model': self._config.slippage_model,
            'base_slippage_bps': self._config.base_slippage_bps,
            'slippage_per_share': self._config.slippage_per_share,
            'market_impact_factor': self._config.market_impact_factor,
            'partial_fill_enabled': self._config.partial_fill_enabled,
            'min_fill_ratio': self._config.min_fill_ratio,
            'liquidity_impact_threshold': self._config.liquidity_impact_threshold,
            # Other backtest settings
            'stop_loss_atr_multiplier': self._config.stop_loss_atr_multiplier,
            'take_profit_multiplier': self._config.take_profit_multiplier,
            'max_holding_days': self._config.max_holding_days,
            'min_score_threshold': self._config.min_score_threshold,
            'max_positions': self._config.max_positions,
            'walk_forward_window': self._config.walk_forward_window,
            'test_window': self._config.test_window
        }
    
    def get_data_settings(self) -> Dict[str, Any]:
        """Get data processing specific settings"""
        return {
            'batch_size': self._config.batch_size,
            'request_timeout': self._config.request_timeout,
            'data_cache_days': self._config.data_cache_days,
            'min_trading_volume': self._config.min_trading_volume,
            'benchmark_symbol': self._config.benchmark_symbol
        }

# Global configuration instance
_global_config: Optional[ConfigManager] = None

def get_config() -> SystemConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config.config

def set_config(config: SystemConfig) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = ConfigManager(config)

def load_config_from_file(file_path: str) -> SystemConfig:
    """Load configuration from file and set as global"""
    config = SystemConfig.from_json(file_path)
    set_config(config)
    return config

def load_config_from_environment(prefix: str = "NSE_") -> SystemConfig:
    """Load configuration from environment and set as global"""
    config = SystemConfig.from_environment(prefix)
    set_config(config)
    return config

# Development utilities
def create_sample_config_file(file_path: str = "config_sample.json") -> None:
    """Create a sample configuration file for reference"""
    config = SystemConfig()
    config.to_json(file_path)
    print(f"Sample configuration created at: {file_path}")

def validate_current_config() -> bool:
    """Validate current global configuration"""
    try:
        get_config().validate()
        return True
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    print("🔧 NSE Stock Screener - Configuration Management")
    print("=" * 50)
    
    # Test default configuration
    print("\n1. Testing default configuration...")
    config = SystemConfig()
    print(f"✅ Default config created successfully")
    print(f"   Portfolio Capital: ₹{config.portfolio_capital:,.2f}")
    print(f"   Max Positions: {config.max_positions}")
    print(f"   Risk per Trade: {config.risk_per_trade*100:.1f}%")
    
    # Test validation
    print("\n2. Testing validation...")
    try:
        config.validate()
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test environment loading
    print("\n3. Testing environment override...")
    os.environ["NSE_PORTFOLIO_CAPITAL"] = "2000000"
    os.environ["NSE_MAX_POSITIONS"] = "15"
    env_config = SystemConfig.from_environment()
    print(f"✅ Environment config created")
    print(f"   Portfolio Capital: ₹{env_config.portfolio_capital:,.2f}")
    print(f"   Max Positions: {env_config.max_positions}")
    
    # Test configuration manager
    print("\n4. Testing configuration manager...")
    manager = ConfigManager(env_config)
    risk_settings = manager.get_risk_settings()
    print(f"✅ Risk settings extracted: {len(risk_settings)} parameters")
    
    # Create sample config file
    print("\n5. Creating sample configuration file...")
    create_sample_config_file("config_sample.json")
    
    print("\n✅ Configuration system ready for integration!")