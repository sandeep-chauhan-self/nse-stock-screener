"""
NSE Stock Screener package

A modular, pluggable stock screening system with clean architecture
and stable interfaces for production use.
"""

# Export core interfaces for external use
from .common.interfaces import (
    IDataFetcher, IIndicator, IScorer, IRiskManager, IBacktester,
    IOrderExecutor, IAnalysisAPI, IReportGenerator, IConfig,
    IndicatorResult, ScoreBreakdown, PositionSizeResult, RiskLimits,
    Trade, BacktestResult, OrderType, OrderStatus, Order
)

# Export configuration management
from .common.config import SystemConfig, ConfigManager

# Export shared enums
from .common.enums import MarketRegime, ProbabilityLevel, PositionStatus, StopType

# Export modular implementations
from .data import YahooDataFetcher, AlphaVantageDataFetcher, DataFetcherFactory
from .indicators import RSIIndicator, ATRIndicator, MACDIndicator, IndicatorEngine

__version__ = "2.0.0"
__all__ = [
    # Core interfaces
    'IDataFetcher', 'IIndicator', 'IScorer', 'IRiskManager', 'IBacktester',
    'IOrderExecutor', 'IAnalysisAPI', 'IReportGenerator', 'IConfig',

    # Data classes
    'IndicatorResult', 'ScoreBreakdown', 'PositionSizeResult', 'RiskLimits',
    'Trade', 'BacktestResult', 'OrderType', 'OrderStatus', 'Order',

    # Configuration
    'SystemConfig', 'ConfigManager',

    # Enums
    'MarketRegime', 'ProbabilityLevel', 'PositionStatus', 'StopType',

    # Implementations
    'YahooDataFetcher', 'AlphaVantageDataFetcher', 'DataFetcherFactory',
    'RSIIndicator', 'ATRIndicator', 'MACDIndicator', 'IndicatorEngine'
]