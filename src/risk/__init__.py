"""
Enhanced Risk Management Package for FS.5 Implementation

This package provides comprehensive risk management capabilities including:
- Position sizing with ATR-based calculations and NSE compliance
- Liquidity validation and market impact analysis
- Portfolio-level risk controls and diversification
- Real-time risk monitoring and alerting
- Margin requirements and drawdown management

Main Components:
- EnhancedRiskManager: Main interface for all risk management
- RiskConfig: Comprehensive configuration management
- PositionSizer: ATR-based position sizing with NSE compliance
- LiquidityValidator: Volume analysis and liquidity constraints
- PortfolioController: Portfolio-level risk controls

Quick Start:
    from src.risk import create_enhanced_risk_manager
    
    # Create risk manager with default settings
    risk_manager = create_enhanced_risk_manager(
        portfolio_capital=1000000,
        max_positions=10,
        risk_per_trade=0.01
    )
    
    # Evaluate a position
    decision = risk_manager.evaluate_position(
        symbol="RELIANCE",
        signal_score=75,
        stock_data={
            'current_price': 2500.0,
            'atr': 50.0,
            'avg_daily_volume': 50000000,
            'sector': 'ENERGY'
        }
    )
    
    if decision.approved:
        print(f"Position approved: {decision.position_size} shares")
    else:
        print(f"Position rejected: {decision.warnings}")
"""

# Import main components for easy access
from .enhanced_risk_manager import (
    EnhancedRiskManager,
    RiskDecision,
    create_enhanced_risk_manager,
    calculate_position_size_simple
)

from .risk_config import (
    RiskConfig,
    PositionSizingConfig,
    NSELotSizeConfig,
    MarginConfig,
    LiquidityConfig,
    SectorConfig,
    CorrelationConfig,
    DrawdownConfig,
    ValidationConfig,
    LiquidityTier,
    SectorRiskTier,
    create_default_risk_config,
    create_conservative_risk_config,
    create_aggressive_risk_config
)

from .position_sizer import (
    EnhancedPositionSizer,
    PositionSizeResult,
    PositionSizeError,
    StockData,
    PortfolioState as PositionPortfolioState,
    create_stock_data_from_dict,
    create_portfolio_state_from_positions,
    calculate_position_risk_metrics
)

from .liquidity_validator import (
    LiquidityValidator,
    LiquidityValidationResult,
    LiquidityAlert,
    VolumeData,
    LiquidityMetrics,
    create_volume_data_from_dataframe,
    calculate_portfolio_liquidity_score
)

from .portfolio_controller import (
    PortfolioRiskController,
    PortfolioRiskResult,
    RiskAlert,
    PortfolioAction,
    Position,
    PortfolioState
)

# Version information
__version__ = "1.0.0"
__author__ = "Stock Analysis Team"
__description__ = "Enhanced Risk Management System with FS.5 Compliance"

# Package-level configuration
DEFAULT_CONFIG = None

def set_default_config(config: RiskConfig) -> None:
    """Set package-level default configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

def get_default_config() -> RiskConfig:
    """Get package-level default configuration."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = create_default_risk_config()
    return DEFAULT_CONFIG

# Convenience functions for quick setup

def create_risk_manager_for_intraday(capital: float = 500000) -> EnhancedRiskManager:
    """
    Create risk manager optimized for intraday trading.
    
    Args:
        capital: Trading capital
        
    Returns:
        Configured risk manager for intraday trading
    """
    config = create_aggressive_risk_config()
    config.portfolio_capital = capital
    config.position_sizing.base_risk_per_trade = 0.005  # 0.5% per trade
    config.margin_config.intraday_margin_pct = 0.05     # 5% margin
    config.drawdown_config.max_daily_drawdown = 0.02    # 2% daily limit
    
    return EnhancedRiskManager(config)

def create_risk_manager_for_swing(capital: float = 1000000) -> EnhancedRiskManager:
    """
    Create risk manager optimized for swing trading.
    
    Args:
        capital: Trading capital
        
    Returns:
        Configured risk manager for swing trading
    """
    config = create_default_risk_config()
    config.portfolio_capital = capital
    config.position_sizing.base_risk_per_trade = 0.01   # 1% per trade
    config.max_positions = 15                           # More positions
    config.sector_config.max_sector_exposure = 0.30     # 30% sector limit
    
    return EnhancedRiskManager(config)

def create_risk_manager_for_conservative(capital: float = 2000000) -> EnhancedRiskManager:
    """
    Create risk manager for conservative long-term investing.
    
    Args:
        capital: Investment capital
        
    Returns:
        Configured risk manager for conservative investing
    """
    config = create_conservative_risk_config()
    config.portfolio_capital = capital
    config.position_sizing.base_risk_per_trade = 0.005  # 0.5% per trade
    config.max_positions = 20                           # Diversified portfolio
    config.sector_config.max_sector_exposure = 0.25     # 25% sector limit
    config.liquidity_config.enable_liquidity_checks = True
    
    return EnhancedRiskManager(config)

# Export all components
__all__ = [
    # Main interface
    'EnhancedRiskManager',
    'RiskDecision',
    'create_enhanced_risk_manager',
    'calculate_position_size_simple',
    
    # Configuration
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
    
    # Position sizing
    'EnhancedPositionSizer',
    'PositionSizeResult',
    'PositionSizeError',
    'StockData',
    'PositionPortfolioState',
    'create_stock_data_from_dict',
    'create_portfolio_state_from_positions',
    'calculate_position_risk_metrics',
    
    # Liquidity validation
    'LiquidityValidator',
    'LiquidityValidationResult',
    'LiquidityAlert',
    'VolumeData',
    'LiquidityMetrics',
    'create_volume_data_from_dataframe',
    'calculate_portfolio_liquidity_score',
    
    # Portfolio risk control
    'PortfolioRiskController',
    'PortfolioRiskResult',
    'RiskAlert',
    'PortfolioAction',
    'Position',
    'PortfolioState',
    
    # Convenience functions
    'create_risk_manager_for_intraday',
    'create_risk_manager_for_swing',
    'create_risk_manager_for_conservative',
    'set_default_config',
    'get_default_config',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]