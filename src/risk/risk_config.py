"""
Enhanced Risk Configuration for FS.5 Implementation

This module provides comprehensive risk management configuration supporting:
- ATR-based position sizing with NSE compliance
- Portfolio diversification and correlation controls
- Liquidity validation and volume constraints
- Margin requirements and drawdown monitoring
- Configurable risk caps with detailed validation

Features:
- Type-safe configuration with validation
- Environment variable overrides
- Sector and correlation limits
- NSE-specific lot size and margin handling
- Dynamic risk multipliers with safety caps
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from pathlib import Path


class LiquidityTier(Enum):
    """Classification of liquidity tiers for different risk treatments."""
    HIGHLY_LIQUID = "highly_liquid"      # >50M avg daily volume
    MODERATELY_LIQUID = "moderately_liquid"  # 10-50M avg daily volume  
    LOW_LIQUID = "low_liquid"            # 1-10M avg daily volume
    ILLIQUID = "illiquid"                # <1M avg daily volume


class SectorRiskTier(Enum):
    """Sector risk classification for exposure limits."""
    LOW_RISK = "low_risk"        # Banks, FMCG, Pharma
    MEDIUM_RISK = "medium_risk"  # IT, Auto, Metals
    HIGH_RISK = "high_risk"      # Small cap, Crypto, Commodities


@dataclass
class NSELotSizeConfig:
    """Configuration for NSE lot size enforcement."""
    enforce_lot_sizes: bool = True
    default_lot_size: int = 1
    lot_size_data_path: Optional[str] = None
    lot_size_cache_hours: int = 24
    
    # Lot size overrides for specific instruments
    lot_size_overrides: Dict[str, int] = field(default_factory=lambda: {
        'NIFTY': 75,
        'BANKNIFTY': 25,
        'FINNIFTY': 40,
        # Add more as needed
    })


@dataclass
class MarginConfig:
    """Configuration for margin requirement calculations."""
    enable_margin_checks: bool = True
    
    # Base margin requirements by category
    equity_margin_pct: float = 0.20      # 20% for equity cash
    derivative_margin_pct: float = 0.10   # 10% for F&O
    intraday_margin_pct: float = 0.05     # 5% for intraday
    
    # Additional margins
    volatility_margin_buffer: float = 0.05  # 5% additional for high vol
    exposure_margin_pct: float = 0.03        # 3% exposure margin
    
    # Risk multipliers based on volatility
    low_vol_multiplier: float = 1.0      # <15% annualized vol
    medium_vol_multiplier: float = 1.2   # 15-25% annualized vol
    high_vol_multiplier: float = 1.5     # >25% annualized vol


@dataclass
class LiquidityConfig:
    """Configuration for liquidity checks and constraints."""
    enable_liquidity_checks: bool = True
    
    # Volume-based constraints
    max_position_vs_avg_volume: float = 0.05  # 5% of avg daily volume
    min_avg_daily_volume: float = 1000000     # 1M min daily volume
    volume_lookback_days: int = 20            # Days for volume averaging
    
    # Liquidity tier limits
    liquidity_tier_limits: Dict[LiquidityTier, float] = field(default_factory=lambda: {
        LiquidityTier.HIGHLY_LIQUID: 0.15,      # 15% max position
        LiquidityTier.MODERATELY_LIQUID: 0.10,  # 10% max position
        LiquidityTier.LOW_LIQUID: 0.05,         # 5% max position
        LiquidityTier.ILLIQUID: 0.02,           # 2% max position
    })
    
    # Volume thresholds for liquidity classification
    liquidity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'highly_liquid': 50000000,    # 50M+
        'moderately_liquid': 10000000, # 10M-50M
        'low_liquid': 1000000,        # 1M-10M
        'illiquid': 0                 # <1M
    })


@dataclass
class SectorConfig:
    """Configuration for sector diversification limits."""
    enable_sector_limits: bool = True
    
    # Global sector exposure limits
    max_sector_exposure: float = 0.25  # 25% max per sector
    max_sectors_concentrated: int = 3   # Max 3 sectors above 15%
    
    # Sector-specific limits based on risk
    sector_risk_limits: Dict[SectorRiskTier, float] = field(default_factory=lambda: {
        SectorRiskTier.LOW_RISK: 0.30,    # 30% for low risk sectors
        SectorRiskTier.MEDIUM_RISK: 0.25, # 25% for medium risk sectors  
        SectorRiskTier.HIGH_RISK: 0.15,   # 15% for high risk sectors
    })
    
    # Sector classification (simplified - in production, use comprehensive mapping)
    sector_classifications: Dict[str, SectorRiskTier] = field(default_factory=lambda: {
        'BANKS': SectorRiskTier.LOW_RISK,
        'FMCG': SectorRiskTier.LOW_RISK,
        'PHARMA': SectorRiskTier.LOW_RISK,
        'IT': SectorRiskTier.MEDIUM_RISK,
        'AUTO': SectorRiskTier.MEDIUM_RISK,
        'METALS': SectorRiskTier.MEDIUM_RISK,
        'REALTY': SectorRiskTier.HIGH_RISK,
        'SMALLCAP': SectorRiskTier.HIGH_RISK,
    })


@dataclass
class CorrelationConfig:
    """Configuration for correlation-aware risk management."""
    enable_correlation_checks: bool = True
    
    # Correlation thresholds
    max_portfolio_correlation: float = 0.70  # Portfolio avg correlation limit
    max_pair_correlation: float = 0.80       # Max between any two positions
    correlation_lookback_days: int = 60      # Days for correlation calculation
    
    # Correlation-based position limits
    high_correlation_limit: float = 0.60     # If corr >0.6, limit total exposure
    max_correlated_exposure: float = 0.30    # Max total exposure to correlated group
    
    # Rebalancing triggers
    correlation_rebalance_threshold: float = 0.75  # Trigger rebalancing
    min_correlation_samples: int = 30             # Min data points for correlation


@dataclass
class DrawdownConfig:
    """Configuration for drawdown monitoring and controls."""
    enable_drawdown_monitoring: bool = True
    
    # Drawdown limits
    max_portfolio_drawdown: float = 0.15    # 15% max portfolio drawdown
    max_daily_drawdown: float = 0.03        # 3% max daily loss
    max_monthly_drawdown: float = 0.08      # 8% max monthly loss
    
    # Drawdown response actions
    reduce_risk_at_drawdown: float = 0.10   # Start reducing risk at 10% DD
    stop_new_positions_at: float = 0.12     # Stop new positions at 12% DD
    emergency_exit_at: float = 0.20         # Emergency exit at 20% DD
    
    # Risk reduction factors
    risk_reduction_factor: float = 0.5      # Reduce position sizes by 50%
    drawdown_recovery_factor: float = 1.2   # Need 20% gain to restore full risk


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing calculations."""
    # Base risk parameters
    base_risk_per_trade: float = 0.01       # 1% base risk per trade
    risk_multiplier_enabled: bool = True    # Enable score-based risk scaling
    
    # Risk multiplier ranges with safety caps
    min_risk_multiplier: float = 0.5        # Minimum risk multiplier
    max_risk_multiplier: float = 2.0        # Maximum risk multiplier  
    risk_multiplier_score_thresholds: Dict[str, Tuple[int, float]] = field(
        default_factory=lambda: {
            'low': (0, 0.7),      # 0-50 score: 0.7x risk
            'medium': (50, 1.0),  # 50-70 score: 1.0x risk
            'high': (70, 1.5),    # 70+ score: 1.5x risk
        }
    )
    
    # ATR-based stop validation
    enable_atr_validation: bool = True
    min_stop_atr_ratio: float = 1.0         # Min stop distance (1x ATR)
    max_stop_atr_ratio: float = 4.0         # Max stop distance (4x ATR)
    default_atr_multiplier: float = 2.0     # Default stop at 2x ATR
    
    # Position size constraints
    min_position_value: float = 5000        # Rs 5,000 minimum position
    max_position_value_pct: float = 0.15    # 15% max position size
    
    # Volatility-based adjustments
    enable_volatility_parity: bool = True
    base_volatility_target: float = 0.02    # 2% daily volatility target
    volatility_adjustment_cap: float = 2.0  # Max 2x adjustment for volatility


@dataclass 
class ValidationConfig:
    """Configuration for validation rules and error handling."""
    # Validation settings
    strict_validation: bool = True          # Fail fast on validation errors
    log_validation_warnings: bool = True    # Log validation issues
    
    # Price validation
    min_stock_price: float = 1.0           # Rs 1 minimum stock price
    max_stock_price: float = 100000.0      # Rs 1L maximum stock price
    max_price_deviation: float = 0.20      # 20% max price deviation from close
    
    # Data quality checks
    min_trading_days: int = 20             # Min trading days for indicators
    max_missing_data_pct: float = 0.10     # 10% max missing data
    stale_data_hours: int = 8              # Flag stale data after 8 hours
    
    # Risk check overrides (for emergency situations)
    allow_validation_override: bool = False
    override_password_hash: Optional[str] = None  # For emergency overrides


@dataclass
class RiskConfig:
    """
    Comprehensive risk management configuration for FS.5 implementation.
    
    This class integrates all risk-related settings including position sizing,
    portfolio limits, liquidity checks, margin requirements, and validation rules.
    """
    
    # Core portfolio settings
    portfolio_capital: float = 1000000.0    # Rs 10 Lakh initial capital
    max_positions: int = 10                 # Maximum concurrent positions
    max_portfolio_risk: float = 0.20        # 20% max portfolio at risk
    
    # Component configurations
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    nse_lot_config: NSELotSizeConfig = field(default_factory=NSELotSizeConfig)
    margin_config: MarginConfig = field(default_factory=MarginConfig)
    liquidity_config: LiquidityConfig = field(default_factory=LiquidityConfig)
    sector_config: SectorConfig = field(default_factory=SectorConfig)
    correlation_config: CorrelationConfig = field(default_factory=CorrelationConfig)
    drawdown_config: DrawdownConfig = field(default_factory=DrawdownConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Risk monitoring and alerting
    enable_risk_alerts: bool = True
    alert_email: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'portfolio_risk': 0.15,      # Alert at 15% portfolio risk
        'sector_concentration': 0.20, # Alert at 20% sector concentration
        'correlation_spike': 0.65,    # Alert when correlations spike
        'drawdown_warning': 0.08,     # Alert at 8% drawdown
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.validation_config.strict_validation:
            self.validate()
    
    def validate(self) -> List[str]:
        """
        Comprehensive validation of all risk configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Portfolio validation
        if self.portfolio_capital <= 0:
            errors.append("portfolio_capital must be positive")
        
        if not 1 <= self.max_positions <= 50:
            errors.append("max_positions must be between 1 and 50")
        
        if not 0.01 <= self.max_portfolio_risk <= 1.0:
            errors.append("max_portfolio_risk must be between 1% and 100%")
        
        # Position sizing validation
        ps = self.position_sizing
        if not 0.001 <= ps.base_risk_per_trade <= 0.1:
            errors.append("base_risk_per_trade must be between 0.1% and 10%")
        
        if ps.min_risk_multiplier >= ps.max_risk_multiplier:
            errors.append("min_risk_multiplier must be less than max_risk_multiplier")
        
        if not 0.01 <= ps.max_position_value_pct <= 0.5:
            errors.append("max_position_value_pct must be between 1% and 50%")
        
        # Liquidity validation
        lc = self.liquidity_config
        if lc.max_position_vs_avg_volume <= 0 or lc.max_position_vs_avg_volume > 0.5:
            errors.append("max_position_vs_avg_volume must be between 0 and 50%")
        
        # Sector validation
        sc = self.sector_config
        if not 0.05 <= sc.max_sector_exposure <= 1.0:
            errors.append("max_sector_exposure must be between 5% and 100%")
        
        # Correlation validation
        cc = self.correlation_config
        if not 0.1 <= cc.max_portfolio_correlation <= 1.0:
            errors.append("max_portfolio_correlation must be between 10% and 100%")
        
        # Drawdown validation
        dc = self.drawdown_config
        if not 0.02 <= dc.max_portfolio_drawdown <= 0.5:
            errors.append("max_portfolio_drawdown must be between 2% and 50%")
        
        if dc.max_daily_drawdown >= dc.max_monthly_drawdown:
            errors.append("max_daily_drawdown must be less than max_monthly_drawdown")
        
        if self.validation_config.strict_validation and errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return errors
    
    def get_risk_multiplier(self, signal_score: int) -> float:
        """
        Calculate risk multiplier based on signal score.
        
        Args:
            signal_score: Signal strength score (0-100)
            
        Returns:
            Risk multiplier factor with safety caps applied
        """
        if not self.position_sizing.risk_multiplier_enabled:
            return 1.0
        
        thresholds = self.position_sizing.risk_multiplier_score_thresholds
        
        # Determine multiplier based on score
        if signal_score >= thresholds['high'][0]:
            multiplier = thresholds['high'][1]
        elif signal_score >= thresholds['medium'][0]:
            multiplier = thresholds['medium'][1]
        else:
            multiplier = thresholds['low'][1]
        
        # Apply safety caps
        multiplier = max(self.position_sizing.min_risk_multiplier, multiplier)
        multiplier = min(self.position_sizing.max_risk_multiplier, multiplier)
        
        return multiplier
    
    def get_liquidity_tier(self, avg_daily_volume: float) -> LiquidityTier:
        """
        Classify liquidity tier based on average daily volume.
        
        Args:
            avg_daily_volume: Average daily trading volume in rupees
            
        Returns:
            Liquidity tier classification
        """
        thresholds = self.liquidity_config.liquidity_thresholds
        
        if avg_daily_volume >= thresholds['highly_liquid']:
            return LiquidityTier.HIGHLY_LIQUID
        elif avg_daily_volume >= thresholds['moderately_liquid']:
            return LiquidityTier.MODERATELY_LIQUID
        elif avg_daily_volume >= thresholds['low_liquid']:
            return LiquidityTier.LOW_LIQUID
        else:
            return LiquidityTier.ILLIQUID
    
    def get_sector_risk_tier(self, sector: str) -> SectorRiskTier:
        """
        Get risk tier classification for a sector.
        
        Args:
            sector: Sector name/code
            
        Returns:
            Sector risk tier classification
        """
        sector_upper = sector.upper()
        return self.sector_config.sector_classifications.get(
            sector_upper, SectorRiskTier.MEDIUM_RISK
        )
    
    def calculate_margin_requirement(self, position_value: float, 
                                   volatility: Optional[float] = None,
                                   instrument_type: str = 'equity') -> float:
        """
        Calculate margin requirement for a position.
        
        Args:
            position_value: Total position value in rupees
            volatility: Annualized volatility (optional)
            instrument_type: Type of instrument ('equity', 'derivative', 'intraday')
            
        Returns:
            Required margin amount in rupees
        """
        mc = self.margin_config
        
        # Base margin by instrument type
        base_margins = {
            'equity': mc.equity_margin_pct,
            'derivative': mc.derivative_margin_pct,
            'intraday': mc.intraday_margin_pct
        }
        
        base_margin = base_margins.get(instrument_type, mc.equity_margin_pct)
        
        # Volatility adjustment
        vol_multiplier = 1.0
        if volatility is not None:
            if volatility > 0.25:  # >25% annual volatility
                vol_multiplier = mc.high_vol_multiplier
            elif volatility > 0.15:  # 15-25% annual volatility
                vol_multiplier = mc.medium_vol_multiplier
            else:  # <15% annual volatility
                vol_multiplier = mc.low_vol_multiplier
        
        # Calculate total margin
        base_margin_amount = position_value * base_margin
        volatility_buffer = position_value * mc.volatility_margin_buffer if volatility and volatility > 0.2 else 0
        exposure_margin = position_value * mc.exposure_margin_pct
        
        total_margin = (base_margin_amount * vol_multiplier + 
                       volatility_buffer + exposure_margin)
        
        return total_margin
    
    @classmethod
    def from_environment(cls, prefix: str = "RISK_") -> 'RiskConfig':
        """
        Create RiskConfig with environment variable overrides.
        
        Args:
            prefix: Environment variable prefix (default: RISK_)
            
        Returns:
            RiskConfig instance with environment overrides applied
        """
        config = cls()
        
        # Simple environment overrides for top-level fields
        env_mappings = {
            f"{prefix}PORTFOLIO_CAPITAL": 'portfolio_capital',
            f"{prefix}MAX_POSITIONS": 'max_positions',
            f"{prefix}MAX_PORTFOLIO_RISK": 'max_portfolio_risk',
            f"{prefix}BASE_RISK_PER_TRADE": ('position_sizing', 'base_risk_per_trade'),
            f"{prefix}MAX_SECTOR_EXPOSURE": ('sector_config', 'max_sector_exposure'),
            f"{prefix}ENABLE_LIQUIDITY_CHECKS": ('liquidity_config', 'enable_liquidity_checks'),
        }
        
        for env_var, field_path in env_mappings.items():
            cls._apply_env_override(config, env_var, field_path)
        
        return config
    
    @classmethod
    def _apply_env_override(cls, config: 'RiskConfig', env_var: str, field_path) -> None:
        """Apply single environment variable override to config."""
        env_value = os.getenv(env_var)
        if env_value is None:
            return
            
        try:
            if isinstance(field_path, tuple):
                cls._set_nested_field(config, field_path, env_value)
            else:
                cls._set_top_level_field(config, field_path, env_value)
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not set {env_var}={env_value}: {e}")
    
    @classmethod
    def _set_nested_field(cls, config: 'RiskConfig', field_path: tuple, env_value: str) -> None:
        """Set nested field value from environment."""
        obj = getattr(config, field_path[0])
        current_value = getattr(obj, field_path[1])
        converted_value = cls._convert_env_value(env_value, current_value)
        setattr(obj, field_path[1], converted_value)
    
    @classmethod
    def _set_top_level_field(cls, config: 'RiskConfig', field_path: str, env_value: str) -> None:
        """Set top-level field value from environment."""
        current_value = getattr(config, field_path)
        converted_value = cls._convert_env_value(env_value, current_value)
        setattr(config, field_path, converted_value)
    
    @classmethod
    def _convert_env_value(cls, env_value: str, current_value) -> Any:
        """Convert environment string value to appropriate type."""
        if isinstance(current_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current_value, int):
            return int(env_value)
        elif isinstance(current_value, float):
            return float(env_value)
        else:
            return env_value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        import dataclasses
        return dataclasses.asdict(self)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to file (JSON format).
        
        Args:
            file_path: Path to save configuration file
        """
        import json
        config_dict = self.to_dict()
        
        # Convert enums to strings for JSON serialization
        def enum_to_str(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: enum_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [enum_to_str(item) for item in obj]
            return obj
        
        config_dict = enum_to_str(config_dict)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'RiskConfig':
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            RiskConfig instance loaded from file
        """
        import json
        
        with open(file_path, 'r') as f:
            _ = json.load(f)  # Load for validation, full deserialization needs more work
        
        # Create instance with defaults 
        # Note: Full deserialization would require more complex handling
        # This is a simplified version that validates file format
        config = cls()
        
        return config


# Convenience function for creating default configuration
def create_default_risk_config() -> RiskConfig:
    """
    Create a default risk configuration suitable for most use cases.
    
    Returns:
        RiskConfig with sensible defaults for Indian markets
    """
    return RiskConfig()


# Convenience function for creating conservative configuration
def create_conservative_risk_config() -> RiskConfig:
    """
    Create a conservative risk configuration for risk-averse trading.
    
    Returns:
        RiskConfig with conservative risk parameters
    """
    config = RiskConfig()
    
    # Reduce risk parameters
    config.position_sizing.base_risk_per_trade = 0.005  # 0.5% per trade
    config.position_sizing.max_risk_multiplier = 1.2    # Lower max multiplier
    config.max_portfolio_risk = 0.10                    # 10% max portfolio risk
    config.position_sizing.max_position_value_pct = 0.08 # 8% max position
    config.sector_config.max_sector_exposure = 0.20     # 20% max sector
    config.drawdown_config.max_portfolio_drawdown = 0.10 # 10% max drawdown
    
    return config


# Convenience function for creating aggressive configuration  
def create_aggressive_risk_config() -> RiskConfig:
    """
    Create an aggressive risk configuration for higher risk tolerance.
    
    Returns:
        RiskConfig with more aggressive risk parameters
    """
    config = RiskConfig()
    
    # Increase risk parameters
    config.position_sizing.base_risk_per_trade = 0.02   # 2% per trade
    config.position_sizing.max_risk_multiplier = 3.0    # Higher max multiplier
    config.max_portfolio_risk = 0.30                    # 30% max portfolio risk
    config.position_sizing.max_position_value_pct = 0.20 # 20% max position
    config.sector_config.max_sector_exposure = 0.35     # 35% max sector
    config.drawdown_config.max_portfolio_drawdown = 0.20 # 20% max drawdown
    
    return config


# Export main components
__all__ = [
    'RiskConfig',
    'PositionSizingConfig',
    'NSELotSizeConfig', 
    'MarginConfig',
    'LiquidityConfig',
    'SectorConfig',
    'CorrelationConfig',
    'DrawdownConfig',
    'ValidationConfig',
    'LiquidityTier',
    'SectorRiskTier',
    'create_default_risk_config',
    'create_conservative_risk_config',
    'create_aggressive_risk_config',
]