"""
Comprehensive Performance Metrics Module
========================================
This module implements sophisticated performance measurement and analytics:
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Information ratios)
- Drawdown analysis (maximum, rolling, duration)
- Trade-level analytics (win rate, payoff ratio, expectancy)
- Attribution analysis (sector, time-based, factor decomposition)
- Statistical tests and confidence intervals
- Benchmark comparison and tracking error
Designed to provide institutional-quality performance measurement for FS.6 compliance.
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from scipy import stats
import math
logger = logging.getLogger(__name__)

# =====================================================================================
# DATA STRUCTURES AND ENUMS

# =====================================================================================
class MetricCategory(Enum):
    """Categories of performance metrics"""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    TRADE_LEVEL = "trade_level"
    ATTRIBUTION = "attribution"
    STATISTICAL = "statistical"
class BenchmarkType(Enum):
    """Types of benchmarks for comparison"""
    ABSOLUTE = "absolute"
  # Compare to zero return
    INDEX = "index"
       # Compare to market index
    RISK_FREE = "risk_free"
  # Compare to risk-free rate
    CUSTOM = "custom"
     # Custom benchmark series
@dataclass
class TradeAnalysis:
    """Individual trade analysis results"""
    trade_id: str
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    gross_return: float
    net_return: float
    commission: float
    holding_period_days: int
    return_pct: float
    annualized_return: float
    exit_reason: str
    sector: Optional[str] = None
    market_cap: Optional[str] = None
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""

    # Basic return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    drawdown_frequency: float = 0.0
    recovery_factor: float = 0.0

    # Trade-level metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Statistical metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
  # Value at Risk
    cvar_95: float = 0.0
  # Conditional Value at Risk
    hit_ratio: float = 0.0

    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    correlation: float = 0.0

    # Attribution
    sector_attribution: dict[str, float] = field(default_factory=dict[str, Any])
    time_attribution: dict[str, float] = field(default_factory=dict[str, Any])

# =====================================================================================
# PERFORMANCE CALCULATOR ENGINE

# =====================================================================================
class PerformanceCalculator:
    """
    Main engine for calculating comprehensive performance metrics
    """
    def __init__(self, risk_free_rate: float = 0.06,
                 trading_days_per_year: int = 252) -> None:
        """
        Initialize performance calculator
        Args:
            risk_free_rate: Annual risk-free rate (default 6% for India)
            trading_days_per_year: Trading days per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.daily_risk_free = (1 + risk_free_rate) ** (1/trading_days_per_year) - 1
        logger.info("Initialized PerformanceCalculator with %.2f%% risk-free rate, %d trading days/year",
                   risk_free_rate*100, trading_days_per_year)
    def calculate_returns(self, prices: Union[pd.Series, list[float]]) -> pd.Series:
        """
        Calculate returns from price series
        Args:
            prices: Price series or list[str]
        Returns:
            Returns series
        """
        if isinstance(prices, List):
            prices = pd.Series(prices)
        if len(prices) < 2:
            return pd.Series(dtype=float)
        returns = prices.pct_change().dropna()
        return returns
    def calculate_basic_metrics(self, returns: pd.Series) -> dict[str, float]:
        """
        Calculate basic return and risk metrics
        Args:
            returns: Daily returns series
        Returns:
            Dictionary of basic metrics
        """
        if len(returns) < 2:
            return {}

        # Clean returns
        returns = returns.dropna()

        # Basic calculations
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.trading_days)

        # Downside volatility (Sortino denominator)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0.0
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'trading_days': len(returns)
        }
    def calculate_risk_adjusted_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> dict[str, float]:
        """
        Calculate risk-adjusted performance metrics
        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns for relative metrics
        Returns:
            Dictionary of risk-adjusted metrics
        """
        if len(returns) < 2:
            return {}
        returns = returns.dropna()
        basic_metrics = self.calculate_basic_metrics(returns)
        metrics = {}

        # Sharpe Ratio
        excess_returns = returns - self.daily_risk_free
        if excess_returns.std() > 0:
            metrics['sharpe_ratio'] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days)
        else:
            metrics['sharpe_ratio'] = 0.0

        # Sortino Ratio
        downside_returns = returns[returns < self.daily_risk_free]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics['sortino_ratio'] = (excess_returns.mean() / downside_returns.std()) * np.sqrt(self.trading_days)
        else:
            metrics['sortino_ratio'] = 0.0

        # Calmar Ratio (need drawdown calculation)
        drawdown_metrics = self.calculate_drawdown_metrics(returns)
        max_drawdown = abs(drawdown_metrics.get('max_drawdown', 0.001))
  # Avoid division by zero
        if max_drawdown > 0:
            metrics['calmar_ratio'] = basic_metrics['annualized_return'] / max_drawdown
        else:
            metrics['calmar_ratio'] = 0.0

        # Benchmark-relative metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:

            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_returns) > 10:
  # Need sufficient data
                # Information Ratio
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(self.trading_days)
                if tracking_error > 0:
                    metrics['information_ratio'] = (active_returns.mean() * self.trading_days) / tracking_error
                    metrics['tracking_error'] = tracking_error
                else:
                    metrics['information_ratio'] = 0.0
                    metrics['tracking_error'] = 0.0

                # Alpha and Beta (CAPM)
                if aligned_benchmark.var() > 0:
                    beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
                    alpha = (aligned_returns.mean() - self.daily_risk_free) - beta * (aligned_benchmark.mean() - self.daily_risk_free)
                    alpha_annualized = alpha * self.trading_days
                    metrics['alpha'] = alpha_annualized
                    metrics['beta'] = beta
                    metrics['correlation'] = aligned_returns.corr(aligned_benchmark)
                else:
                    metrics['alpha'] = 0.0
                    metrics['beta'] = 0.0
                    metrics['correlation'] = 0.0

                # Treynor Ratio
                if metrics.get('beta', 0) != 0:
                    metrics['treynor_ratio'] = (basic_metrics['annualized_return'] - self.risk_free_rate) / metrics['beta']
                else:
                    metrics['treynor_ratio'] = 0.0
        return metrics
    def calculate_drawdown_metrics(self, returns: pd.Series) -> dict[str, float]:
        """
        Calculate comprehensive drawdown metrics
        Args:
            returns: Daily returns series
        Returns:
            Dictionary of drawdown metrics
        """
        if len(returns) < 2:
            return {}
        returns = returns.dropna()

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Calculate running maximum (peak)
        running_max = cumulative_returns.expanding().max()

        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdowns.min()

        # Drawdown duration analysis
        is_in_drawdown = drawdowns < 0
        drawdown_periods = []
        if is_in_drawdown.any():

            # Find drawdown periods
            in_drawdown = False
            start_idx = None
            for i, in_dd in enumerate(is_in_drawdown):
                if in_dd and not in_drawdown:

                    # Start of drawdown
                    in_drawdown = True
                    start_idx = i
                elif not in_dd and in_drawdown:

                    # End of drawdown
                    in_drawdown = False
                    if start_idx is not None:
                        drawdown_periods.append(i - start_idx)

            # Handle case where we end in drawdown
            if in_drawdown and start_idx is not None:
                drawdown_periods.append(len(is_in_drawdown) - start_idx)

        # Calculate metrics
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
        drawdown_frequency = len(drawdown_periods) / len(returns) * self.trading_days if len(returns) > 0 else 0.0

        # Recovery factor
        total_return = (cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0.0
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'drawdown_frequency': drawdown_frequency,
            'recovery_factor': recovery_factor,
            'drawdowns_series': drawdowns
        }
    def calculate_trade_metrics(self, trades: list[TradeAnalysis]) -> dict[str, float]:
        """
        Calculate trade-level performance metrics
        Args:
            trades: list[str] of trade analysis objects
        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return {}

        # Extract returns (keep for potential future use)
        net_returns = [trade.net_return for trade in trades]

        # Basic counts
        total_trades = len(trades)
        winning_trades = len([r for r in net_returns if r > 0])
        losing_trades = len([r for r in net_returns if r < 0])

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Average win/loss
        wins = [r for r in net_returns if r > 0]
        losses = [r for r in net_returns if r < 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Largest win/loss
        largest_win = max(net_returns) if net_returns else 0.0
        largest_loss = min(net_returns) if net_returns else 0.0

        # Profit factor
        gross_profit = sum([r for r in net_returns if r > 0])
        gross_loss = abs(sum([r for r in net_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Average holding period
        holding_periods = [trade.holding_period_days for trade in trades]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_holding_period': avg_holding_period
        }
    def calculate_statistical_metrics(self, returns: pd.Series) -> dict[str, float]:
        """
        Calculate statistical metrics and risk measures
        Args:
            returns: Daily returns series
        Returns:
            Dictionary of statistical metrics
        """
        if len(returns) < 4:
  # Need minimum data for statistical measures
            return {}
        returns = returns.dropna()

        # Moments
        skewness = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns)
  # Excess kurtosis
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(returns, 5)
  # 95% VaR (5th percentile)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        # Hit ratio (percentage of positive returns)
        hit_ratio = (returns > 0).sum() / len(returns)

        # Tail ratio
        tail_ratio = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0.0

        # Maximum consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(returns > 0)
        consecutive_losses = self._calculate_max_consecutive(returns < 0)
        return {
            'skewness': skewness,
            'kurtosis': kurtosis_val,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'hit_ratio': hit_ratio,
            'tail_ratio': tail_ratio,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    def _calculate_max_consecutive(self, bool_series: pd.Series) -> int:
        """Calculate maximum consecutive True values in boolean series"""
        if len(bool_series) == 0:
            return 0
        max_consecutive = 0
        current_consecutive = 0
        for value in bool_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive
    def calculate_rolling_metrics(self, returns: pd.Series,
                                window_days: int = 63) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        Args:
            returns: Daily returns series
            window_days: Rolling window size in days
        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window_days:
            return pd.DataFrame()
        returns = returns.dropna()

        # Calculate rolling metrics
        rolling_return = returns.rolling(window_days).apply(lambda x: (1 + x).prod() - 1)
        rolling_vol = returns.rolling(window_days).std() * np.sqrt(self.trading_days)

        # Rolling Sharpe (approximation)
        rolling_excess = returns - self.daily_risk_free
        rolling_sharpe = (rolling_excess.rolling(window_days).mean() /
                         rolling_excess.rolling(window_days).std()) * np.sqrt(self.trading_days)

        # Rolling drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window_days).max()
        rolling_drawdown = (cumulative - rolling_max) / rolling_max

        # Combine into DataFrame
        rolling_metrics = pd.DataFrame({
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_drawdown
        }, index=returns.index)
        return rolling_metrics.dropna()
    def calculate_comprehensive_metrics(self, returns: pd.Series,
                                      trades: Optional[list[TradeAnalysis]] = None,
                                      benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        Args:
            returns: Daily returns series
            trades: Optional list[str] of individual trades
            benchmark_returns: Optional benchmark returns
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        metrics = PerformanceMetrics()
        if len(returns) < 2:
            logger.warning("Insufficient data for performance calculation")
            return metrics
        try:

            # Basic metrics
            basic = self.calculate_basic_metrics(returns)
            metrics.total_return = basic.get('total_return', 0.0)
            metrics.annualized_return = basic.get('annualized_return', 0.0)
            metrics.volatility = basic.get('volatility', 0.0)
            metrics.downside_volatility = basic.get('downside_volatility', 0.0)

            # Risk-adjusted metrics
            risk_adj = self.calculate_risk_adjusted_metrics(returns, benchmark_returns)
            metrics.sharpe_ratio = risk_adj.get('sharpe_ratio', 0.0)
            metrics.sortino_ratio = risk_adj.get('sortino_ratio', 0.0)
            metrics.calmar_ratio = risk_adj.get('calmar_ratio', 0.0)
            metrics.information_ratio = risk_adj.get('information_ratio', 0.0)
            metrics.treynor_ratio = risk_adj.get('treynor_ratio', 0.0)
            metrics.alpha = risk_adj.get('alpha', 0.0)
            metrics.beta = risk_adj.get('beta', 0.0)
            metrics.tracking_error = risk_adj.get('tracking_error', 0.0)
            metrics.correlation = risk_adj.get('correlation', 0.0)

            # Drawdown metrics
            drawdown = self.calculate_drawdown_metrics(returns)
            metrics.max_drawdown = drawdown.get('max_drawdown', 0.0)
            metrics.max_drawdown_duration = drawdown.get('max_drawdown_duration', 0)
            metrics.avg_drawdown = drawdown.get('avg_drawdown', 0.0)
            metrics.drawdown_frequency = drawdown.get('drawdown_frequency', 0.0)
            metrics.recovery_factor = drawdown.get('recovery_factor', 0.0)

            # Trade-level metrics
            if trades:
                trade_metrics = self.calculate_trade_metrics(trades)
                metrics.total_trades = trade_metrics.get('total_trades', 0)
                metrics.winning_trades = trade_metrics.get('winning_trades', 0)
                metrics.losing_trades = trade_metrics.get('losing_trades', 0)
                metrics.win_rate = trade_metrics.get('win_rate', 0.0)
                metrics.avg_win = trade_metrics.get('avg_win', 0.0)
                metrics.avg_loss = trade_metrics.get('avg_loss', 0.0)
                metrics.largest_win = trade_metrics.get('largest_win', 0.0)
                metrics.largest_loss = trade_metrics.get('largest_loss', 0.0)
                metrics.profit_factor = trade_metrics.get('profit_factor', 0.0)
                metrics.expectancy = trade_metrics.get('expectancy', 0.0)

            # Statistical metrics
            statistical = self.calculate_statistical_metrics(returns)
            metrics.skewness = statistical.get('skewness', 0.0)
            metrics.kurtosis = statistical.get('kurtosis', 0.0)
            metrics.var_95 = statistical.get('var_95', 0.0)
            metrics.cvar_95 = statistical.get('cvar_95', 0.0)
            metrics.hit_ratio = statistical.get('hit_ratio', 0.0)
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        return metrics

# =====================================================================================
# ATTRIBUTION ANALYSIS

# =====================================================================================
class AttributionAnalyzer:
    """
    Analyze performance attribution by various factors
    """
    def __init__(self, performance_calculator: PerformanceCalculator) -> None:
        self.calc = performance_calculator
    def sector_attribution(self, trades: list[TradeAnalysis]) -> dict[str, dict[str, float]]:
        """
        Calculate performance attribution by sector
        Args:
            trades: list[str] of trades with sector information
        Returns:
            Dictionary of sector -> metrics
        """
        if not trades:
            return {}

        # Group trades by sector
        sector_trades = {}
        for trade in trades:
            sector = trade.sector or 'Unknown'
            if sector not in sector_trades:
                sector_trades[sector] = []
            sector_trades[sector].append(trade)

        # Calculate metrics for each sector
        sector_metrics = {}
        for sector, sector_trade_list in sector_trades.items():
            trade_metrics = self.calc.calculate_trade_metrics(sector_trade_list)

            # Calculate sector returns
            sector_returns = [trade.return_pct / 100 for trade in sector_trade_list]
  # Convert to decimal
            if sector_returns:
                sector_total_return = sum(sector_returns)
                sector_avg_return = np.mean(sector_returns)
                sector_vol = np.std(sector_returns) if len(sector_returns) > 1 else 0.0
            else:
                sector_total_return = 0.0
                sector_avg_return = 0.0
                sector_vol = 0.0
            sector_metrics[sector] = {
                'total_return': sector_total_return,
                'avg_return': sector_avg_return,
                'volatility': sector_vol,
                'trade_count': len(sector_trade_list),
                'win_rate': trade_metrics.get('win_rate', 0.0),
                'profit_factor': trade_metrics.get('profit_factor', 0.0)
            }
        return sector_metrics
    def time_attribution(self, returns: pd.Series,
                        frequency: str = 'monthly') -> dict[str, float]:
        """
        Calculate performance attribution by time periods
        Args:
            returns: Daily returns series with datetime index
            frequency: 'monthly', 'quarterly', or 'yearly'
        Returns:
            Dictionary of time period -> return
        """
        if len(returns) < 2:
            return {}
        returns = returns.dropna()

        # Group by time period
        if frequency == 'monthly':
            period_returns = returns.groupby(returns.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'quarterly':
            period_returns = returns.groupby(returns.index.to_period('Q')).apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'yearly':
            period_returns = returns.groupby(returns.index.to_period('Y')).apply(lambda x: (1 + x).prod() - 1)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Convert to dictionary with string keys
        time_attribution = {str(period): ret for period, ret in period_returns.items()}
        return time_attribution

# =====================================================================================
# UTILITY FUNCTIONS

# =====================================================================================
def create_trade_analysis_from_backtest(backtest_results: dict[str, Any]) -> list[TradeAnalysis]:
    """
    Convert backtest results to TradeAnalysis objects
    Args:
        backtest_results: Backtest results dictionary
    Returns:
        list[str] of TradeAnalysis objects
    """
    trades = []

    # Extract trades from backtest results (format may vary)
    if 'trades' in backtest_results:
        for i, trade_data in enumerate(backtest_results['trades']):
            trade = TradeAnalysis(
                trade_id=str(i),
                symbol=trade_data.get('symbol', ''),
                entry_date=trade_data.get('entry_date', datetime.now()),
                exit_date=trade_data.get('exit_date', datetime.now()),
                entry_price=trade_data.get('entry_price', 0.0),
                exit_price=trade_data.get('exit_price', 0.0),
                quantity=trade_data.get('quantity', 0),
                gross_return=trade_data.get('gross_return', 0.0),
                net_return=trade_data.get('net_return', 0.0),
                commission=trade_data.get('commission', 0.0),
                holding_period_days=trade_data.get('holding_period_days', 0),
                return_pct=trade_data.get('return_pct', 0.0),
                annualized_return=trade_data.get('annualized_return', 0.0),
                exit_reason=trade_data.get('exit_reason', 'unknown'),
                sector=trade_data.get('sector'),
                market_cap=trade_data.get('market_cap')
            )
            trades.append(trade)
    return trades
def calculate_performance_summary(returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                trades: Optional[list[TradeAnalysis]] = None,
                                risk_free_rate: float = 0.06) -> PerformanceMetrics:
    """
    Convenience function to calculate comprehensive performance summary
    Args:
        returns: Daily returns series
        benchmark_returns: Optional benchmark returns
        trades: Optional trade list[str]
        risk_free_rate: Risk-free rate for calculations
    Returns:
        PerformanceMetrics object
    """
    calculator = PerformanceCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate_comprehensive_metrics(returns, trades, benchmark_returns)

# =====================================================================================
# EXAMPLE USAGE

# =====================================================================================
if __name__ == "__main__":

    # Example usage of performance metrics
    # Create sample data
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')

    # Generate sample returns with some autocorrelation
    returns = []
    current_return = 0.0
    for _ in range(len(dates)):
        shock = rng.normal(0, 0.02)
        current_return = 0.1 * current_return + shock
  # AR(1) process
        returns.append(current_return)
    returns_series = pd.Series(returns, index=dates)

    # Create benchmark (slightly lower returns, lower volatility)
    benchmark_returns = returns_series * 0.8 + rng.normal(0, 0.005, len(returns_series))

    # Calculate performance metrics
    calculator = PerformanceCalculator(risk_free_rate=0.06)
    metrics = calculator.calculate_comprehensive_metrics(
        returns_series,
        benchmark_returns=benchmark_returns
    )

    # Print results
    print("📊 Performance Metrics Summary")
    print("=" * 40)
    print(f"Total Return: {metrics.total_return*100:.2f}%")
    print(f"Annualized Return: {metrics.annualized_return*100:.2f}%")
    print(f"Volatility: {metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
    print(f"Alpha: {metrics.alpha*100:.2f}%")
    print(f"Beta: {metrics.beta:.3f}")
    print(f"Information Ratio: {metrics.information_ratio:.3f}")

    # Example rolling metrics
    rolling_metrics = calculator.calculate_rolling_metrics(returns_series, window_days=30)
    print("\nRolling Metrics (30-day):")
    print(f"Latest Rolling Sharpe: {rolling_metrics['rolling_sharpe'].iloc[-1]:.3f}")
    print(f"Latest Rolling Drawdown: {rolling_metrics['rolling_drawdown'].iloc[-1]*100:.2f}%")
