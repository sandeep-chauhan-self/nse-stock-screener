"""
Common package for shared components across the NSE Stock Screener system.
This package contains shared enums, constants, utilities, interfaces,
and configuration management used across multiple modules.
"""
from .interfaces import (
    IDataFetcher, IIndicator, IScorer, IRiskManager, IBacktester,
    IOrderExecutor, IAnalysisAPI, IReportGenerator, IConfig,
    IDataCache, IDataValidator, IIndicatorEngine,
    IndicatorResult, ScoreBreakdown, PositionSizeResult, RiskLimits,
    Trade, BacktestResult, OrderType, OrderStatus, Order
)
from .config import (
    SystemConfig, DataConfig, IndicatorConfig, ScoringConfig,
    RiskConfig, BacktestConfig, ConfigManager
)
__version__ = "1.0.0"
__all__ = [

    # Interface protocols
    'IDataFetcher', 'IIndicator', 'IScorer', 'IRiskManager', 'IBacktester',
    'IOrderExecutor', 'IAnalysisAPI', 'IReportGenerator', 'IConfig',
    'IDataCache', 'IDataValidator', 'IIndicatorEngine',

    # Data classes
    'IndicatorResult', 'ScoreBreakdown', 'PositionSizeResult', 'RiskLimits',
    'Trade', 'BacktestResult', 'OrderType', 'OrderStatus', 'Order',

    # Configuration classes
    'SystemConfig', 'DataConfig', 'IndicatorConfig', 'ScoringConfig',
    'RiskConfig', 'BacktestConfig', 'ConfigManager'
]
