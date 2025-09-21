"""
Core interfaces for the NSE Stock Screener system.
This module defines the stable interfaces (protocols and abstract base classes) that
each layer must implement. These interfaces ensure pluggability, testability, and
clear separation of concerns.
"""
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any, List, Union
from datetime import datetime, date
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Import enums directly to avoid circular imports
try:
    from .enums import MarketRegime
except ImportError:
    from src.common.enums import MarketRegime

# ============================================================================
# Data Layer Interfaces

# ============================================================================
class IDataFetcher(Protocol):
    """Interface for fetching market data from various sources."""
    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol between start and end dates.
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            start: Start date for data
            end: End date for data
            **kwargs: Additional provider-specific parameters
        Returns:
            DataFrame with OHLCV columns, or None if data unavailable
        """
        ...
    def fetch_symbols(self, exchange: str = "NSE") -> List[str]:
        """
        Fetch List[str] of available symbols for an exchange.
        Args:
            exchange: Exchange identifier (default: "NSE")
        Returns:
            List[str] of symbol strings
        """
        ...
class IDataCache(Protocol):
    """Interface for caching market data."""
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data by key."""
        ...
    def Set[str](self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> None:
        """Cache data with optional TTL."""
        ...
    def invalidate(self, pattern: str) -> None:
        """Invalidate cached data matching pattern."""
        ...
class IDataValidator(Protocol):
    """Interface for validating market data quality."""
    def validate(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate data quality and return validation report.
        Args:
            data: OHLCV DataFrame to validate
            symbol: Symbol being validated
        Returns:
            Validation report with errors, warnings, and quality metrics
        """
        ...
@dataclass
class StockData:
    """Standard container for stock market data."""
    symbol: str
    data: pd.DataFrame
  # OHLCV data
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None

# ============================================================================
# Indicators Layer Interfaces

# ============================================================================
@dataclass
class IndicatorResult:
    """Standard result container for indicator calculations."""
    value: Union[float, int, bool]
    confidence: float
  # 0.0 to 1.0
    metadata: Dict[str, Any]
class IIndicator(Protocol):
    """Interface for technical indicators."""
    def compute(self, data: pd.DataFrame, **params) -> IndicatorResult:
        """
        Compute indicator value from OHLCV data.
        Args:
            data: OHLCV DataFrame
            **params: Indicator-specific parameters
        Returns:
            IndicatorResult with value, confidence, and metadata
        """
        ...
    @property
    def name(self) -> str:
        """Indicator name."""
        ...
    @property
    def required_periods(self) -> int:
        """Minimum periods required for calculation."""
        ...
class IIndicatorEngine(Protocol):
    """Interface for computing multiple indicators."""
    def compute_all(self, symbol: str, data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Compute all configured indicators for a symbol.
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
        Returns:
            Dictionary mapping indicator names to results
        """
        ...
    def register_indicator(self, indicator: IIndicator) -> None:
        """Register a new indicator."""
        ...

# ============================================================================
# Scoring Layer Interfaces

# ============================================================================
@dataclass
class ScoreBreakdown:
    """Detailed breakdown of scoring components."""
    volume_score: float
    momentum_score: float
    trend_score: float
    volatility_score: float
    relative_strength_score: float
    volume_profile_score: float
    weekly_bonus: float
    total_score: float
    classification: str
  # HIGH, MEDIUM, LOW
    regime_adjustments: Dict[str, Any]
class IScorer(Protocol):
    """Interface for composite scoring systems."""
    def score(self,
             symbol: str,
             indicators: Dict[str, IndicatorResult],
             regime: MarketRegime) -> ScoreBreakdown:
        """
        Compute composite score from indicators.
        Args:
            symbol: Stock symbol
            indicators: Dictionary of indicator results
            regime: Current market regime
        Returns:
            Detailed score breakdown
        """
        ...
    def classify(self, score: float) -> str:
        """Classify score into HIGH/MEDIUM/LOW."""
        ...

# ============================================================================
# Risk Management Layer Interfaces

# ============================================================================
@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    can_enter: bool
    quantity: int
    position_value: float
    risk_amount: float
    reason: str
    stop_loss_price: float
    take_profit_price: Optional[float]
@dataclass
class RiskLimits:
    """Risk management constraints."""
    max_portfolio_risk: float
    max_position_size: float
    max_positions: int
    max_sector_exposure: float
    min_liquidity: float
class IRiskManager(Protocol):
    """Interface for risk management systems."""
    def can_enter_position(self,
                          symbol: str,
                          entry_price: float,
                          score: ScoreBreakdown,
                          atr: float) -> PositionSizeResult:
        """
        Determine if position can be entered and calculate size.
        Args:
            symbol: Stock symbol
            entry_price: Proposed entry price
            score: Composite score breakdown
            atr: Average True Range for stop loss calculation
        Returns:
            Position sizing result with all details
        """
        ...
    def update_portfolio(self, symbol: str, quantity: int, price: float) -> None:
        """Update portfolio state after position entry/exit."""
        ...
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status and exposures."""
        ...

# ============================================================================
# Backtest Layer Interfaces

# ============================================================================
@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    quantity: int
    stop_loss: float
    take_profit: Optional[float]
    pnl: Optional[float]
    commission: float
    slippage: float
@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[Trade]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
class IBacktester(Protocol):
    """Interface for backtesting systems."""
    def run(self,
           start_date: date,
           end_date: date,
           symbols: List[str],
           initial_capital: float) -> BacktestResult:
        """
        Run backtest over specified period.
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List[str] of symbols to test
            initial_capital: Starting capital
        Returns:
            Comprehensive backtest results
        """
        ...

# ============================================================================
# Execution Layer Interfaces

# ============================================================================
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    order_type: OrderType
    quantity: int
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    order_id: Optional[str] = None
class IOrderExecutor(Protocol):
    """Interface for order execution systems."""
    def place_order(self, order: Order) -> str:
        """
        Place a trading order.
        Args:
            order: Order to place
        Returns:
            Order ID
        """
        ...
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order."""
        ...

# ============================================================================
# API Layer Interfaces

# ============================================================================
class IAnalysisAPI(Protocol):
    """Interface for analysis API endpoints."""
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a single symbol and return results."""
        ...
    def screen_stocks(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Screen stocks based on criteria."""
        ...
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and regime."""
        ...

# ============================================================================
# UI Layer Interfaces

# ============================================================================
class IReportGenerator(Protocol):
    """Interface for generating analysis reports."""
    def generate_analysis_report(self,
                               results: List[ScoreBreakdown],
                               format: str = "csv") -> str:
        """
        Generate analysis report in specified format.
        Args:
            results: List[str] of analysis results
            format: Output format (csv, pdf, html)
        Returns:
            Path to generated report
        """
        ...
    def generate_chart(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Generate technical analysis chart."""
        ...

# ============================================================================
# Configuration Interface

# ============================================================================
class IConfig(Protocol):
    """Interface for configuration management."""
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...
    def Set[str](self, key: str, value: Any) -> None:
        """Set[str] configuration value."""
        ...
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        ...
    def validate(self) -> List[str]:
        """Validate configuration and return error messages."""
        ...
