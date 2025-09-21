"""
Portfolio Risk Controller for FS.5 Implementation
This module provides comprehensive portfolio-level risk management including:
- Sector diversification limits and monitoring
- Correlation-aware risk budgeting and position sizing
- Maximum drawdown guardrails and circuit breakers
- Portfolio exposure tracking across multiple dimensions
- Dynamic risk adjustments based on market conditions
Features:
- Real-time portfolio risk monitoring and alerting
- Correlation matrix calculation and concentration limits
- Sector exposure tracking with risk-based limits
- Drawdown calculation and automatic risk reduction
- Dynamic position sizing based on portfolio state
- Integration with position sizing and liquidity validation
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict[str, Any], List[str], Optional, Tuple[str, ...], NamedTuple, Set[str]
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import warnings
from .risk_config import RiskConfig, SectorRiskTier, LiquidityTier
class RiskAlert(Enum):
    """Types of portfolio risk alerts."""
    HIGH_CORRELATION = "high_correlation"
    SECTOR_CONCENTRATION = "sector_concentration"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    LIQUIDITY_CONCENTRATION = "liquidity_concentration"
    MARGIN_UTILIZATION = "margin_utilization"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
class PortfolioAction(Enum):
    """Actions to take based on portfolio risk."""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_EXIT = "emergency_exit"
class PortfolioRiskResult(NamedTuple):
    """Result of portfolio risk assessment."""
    action: PortfolioAction
    risk_score: float
    sector_exposures: Dict[str, float]
    correlation_risk: float
    drawdown_risk: float
    alerts: List[RiskAlert]
    recommendations: List[str]
    risk_details: Dict[str, float]
@dataclass
class Position:
    """Portfolio position with risk attributes."""
    symbol: str
    shares: int
    entry_price: float
    current_price: float
    sector: Optional[str] = None
    entry_date: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    liquidity_tier: Optional[LiquidityTier] = None
    @property
    def position_value(self) -> float:
        """Current position value."""
        return self.shares * self.current_price
    @property
    def cost_basis(self) -> float:
        """Original cost basis."""
        return self.shares * self.entry_price
    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis
@dataclass
class PortfolioState:
    """Comprehensive portfolio state for risk assessment."""
    positions: List[Position]
    cash_available: float
    total_capital: float
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    margin_used: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    @property
    def total_position_value(self) -> float:
        """Total value of all positions."""
        return sum(pos.position_value for pos in self.positions)
    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (positions + cash)."""
        return self.total_position_value + self.cash_available
    @property
    def cash_utilization(self) -> float:
        """Percentage of capital deployed in positions."""
        if self.total_capital == 0:
            return 0.0
        return self.total_position_value / self.total_capital
    @property
    def sector_exposures(self) -> Dict[str, float]:
        """Calculate sector exposures as percentage of total capital."""
        exposures = {}
        for pos in self.positions:
            sector = pos.sector or 'UNKNOWN'
            exposures[sector] = exposures.get(sector, 0) + pos.position_value

        # Convert to percentages
        return {sector: value / self.total_capital for sector, value in exposures.items()}
class PortfolioRiskController:
    """
    Comprehensive portfolio risk controller implementing FS.5 requirements.
    Provides sector diversification, correlation management, drawdown monitoring,
    and portfolio-level risk controls for position sizing and trading decisions.
    """
    def __init__(self, risk_config: RiskConfig) -> None:
        """
        Initialize portfolio risk controller.
        Args:
            risk_config: Risk management configuration
        """
        self.config = risk_config
        self.logger = logging.getLogger(__name__)

        # Portfolio state
        self._current_portfolio: Optional[PortfolioState] = None

        # Risk monitoring
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_last_updated: Optional[datetime] = None

        # Alert tracking
        self._active_alerts: Dict[RiskAlert, List[str]] = defaultdict(List[str])
        self._alert_history: List[Dict[str, Any]] = []

        # Performance tracking
        self._risk_assessments = 0
        self._rejections = 0
        self._emergency_exits = 0

        # Historical data for drawdown calculations
        self._portfolio_history: List[Tuple[datetime, float]] = []
    def assess_portfolio_risk(
        self,
        portfolio_state: PortfolioState,
        proposed_position: Optional[Position] = None
    ) -> PortfolioRiskResult:
        """
        Assess comprehensive portfolio risk with optional new position.
        Args:
            portfolio_state: Current portfolio state
            proposed_position: Optional new position to evaluate
        Returns:
            PortfolioRiskResult with action and risk analysis
        """
        self._risk_assessments += 1
        self._current_portfolio = portfolio_state

        # Create test portfolio with proposed position
        test_portfolio = self._create_test_portfolio(portfolio_state, proposed_position)

        # Calculate risk components
        risk_details = {}
        alerts = []
        recommendations = []

        # 1. Sector diversification risk
        sector_risk, sector_alerts = self._assess_sector_risk(test_portfolio, risk_details)
        alerts.extend(sector_alerts)

        # 2. Correlation risk
        correlation_risk, corr_alerts = self._assess_correlation_risk(test_portfolio, risk_details)
        alerts.extend(corr_alerts)

        # 3. Drawdown risk
        drawdown_risk, dd_alerts = self._assess_drawdown_risk(test_portfolio, risk_details)
        alerts.extend(dd_alerts)

        # 4. Liquidity concentration risk
        liquidity_risk, liq_alerts = self._assess_liquidity_concentration(test_portfolio, risk_details)
        alerts.extend(liq_alerts)

        # 5. Portfolio limits
        limit_risk, limit_alerts = self._assess_portfolio_limits(test_portfolio, risk_details)
        alerts.extend(limit_alerts)

        # Calculate overall risk score and determine action
        overall_risk_score = self._calculate_overall_risk_score(
            sector_risk, correlation_risk, drawdown_risk, liquidity_risk, limit_risk
        )
        action = self._determine_portfolio_action(overall_risk_score, alerts)

        # Generate recommendations
        recommendations = self._generate_recommendations(alerts)

        # Track rejections and emergency exits
        if action in [PortfolioAction.REJECT, PortfolioAction.CLOSE_POSITIONS]:
            self._rejections += 1
        if action == PortfolioAction.EMERGENCY_EXIT:
            self._emergency_exits += 1

        # Update alert tracking
        self._update_alert_tracking(alerts)
        return PortfolioRiskResult(
            action=action,
            risk_score=overall_risk_score,
            sector_exposures=test_portfolio.sector_exposures,
            correlation_risk=correlation_risk,
            drawdown_risk=drawdown_risk,
            alerts=alerts,
            recommendations=recommendations,
            risk_details=risk_details
        )
    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, List[float]],
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio positions.
        Args:
            price_data: Dict[str, Any] of symbol -> List[str] of prices
            lookback_days: Optional lookback period (uses config default)
        Returns:
            Correlation matrix as pandas DataFrame
        """
        if lookback_days is None:
            lookback_days = self.config.correlation_config.correlation_lookback_days

        # Convert to returns
        returns_data = {}
        for symbol, prices in price_data.items():
            if len(prices) < 2:
                continue
            prices_array = np.array(prices[-lookback_days:])
            returns = np.diff(np.log(prices_array))
            returns_data[symbol] = returns

        # Create DataFrame and calculate correlation
        if returns_data:
            df = pd.DataFrame(returns_data)
            correlation_matrix = df.corr()

            # Cache the result
            self._correlation_matrix = correlation_matrix
            self._correlation_last_updated = datetime.now()
            return correlation_matrix
        return pd.DataFrame()
    def check_position_correlation(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[float, List[str]]:
        """
        Check correlation of new position with existing positions.
        Args:
            new_symbol: Symbol for new position
            existing_symbols: List[str] of existing position symbols
            correlation_matrix: Optional correlation matrix
        Returns:
            Tuple[str, ...] of (max_correlation, warnings)
        """
        warnings_list = []
        if correlation_matrix is None:
            correlation_matrix = self._correlation_matrix
        if correlation_matrix is None or new_symbol not in correlation_matrix.index:
            warnings_list.append(f"No correlation data for {new_symbol}")
            return 0.0, warnings_list
        cc = self.config.correlation_config
        max_correlation = 0.0
        for symbol in existing_symbols:
            if symbol in correlation_matrix.index:
                corr = abs(correlation_matrix.loc[new_symbol, symbol])
                max_correlation = max(max_correlation, corr)
                if corr > cc.max_pair_correlation:
                    warnings_list.append(
                        f"High correlation with {symbol}: {corr:.2f} > {cc.max_pair_correlation:.2f}"
                    )
        return max_correlation, warnings_list
    def calculate_risk_adjusted_position_size(
        self,
        base_position_size: float,
        portfolio_state: PortfolioState,
        risk_factors: Dict[str, float]
    ) -> float:
        """
        Adjust position size based on portfolio risk factors.
        Args:
            base_position_size: Base position size from position sizer
            portfolio_state: Current portfolio state
            risk_factors: Dict[str, Any] of risk factor names and values
        Returns:
            Risk-adjusted position size
        """
        dc = self.config.drawdown_config

        # Start with base size
        adjusted_size = base_position_size

        # Drawdown adjustment
        current_drawdown = abs(portfolio_state.max_drawdown)
        if current_drawdown > dc.reduce_risk_at_drawdown:
            reduction_factor = dc.risk_reduction_factor
            adjusted_size *= (1 - reduction_factor *
                            (current_drawdown - dc.reduce_risk_at_drawdown) /
                            (dc.max_portfolio_drawdown - dc.reduce_risk_at_drawdown))

        # Portfolio concentration adjustment
        cash_utilization = portfolio_state.cash_utilization
        if cash_utilization > 0.8:
  # 80% deployed
            concentration_factor = 1 - (cash_utilization - 0.8) / 0.2 * 0.5
            adjusted_size *= concentration_factor

        # Risk factor adjustments
        for factor_name, factor_value in risk_factors.items():
            if factor_name == 'correlation_risk' and factor_value > 0.6:
                adjusted_size *= (1 - (factor_value - 0.6) / 0.4 * 0.3)
            elif factor_name == 'sector_concentration' and factor_value > 0.2:
                adjusted_size *= (1 - (factor_value - 0.2) / 0.3 * 0.4)

        # Ensure minimum and maximum bounds
        min_size = base_position_size * 0.1
  # At least 10% of base
        max_size = base_position_size * 1.0
   # No increase above base
        return max(min_size, min(adjusted_size, max_size))
    def update_portfolio_history(self, portfolio_value: float) -> None:
        """
        Update portfolio value history for drawdown calculations.
        Args:
            portfolio_value: Current portfolio value
        """
        timestamp = datetime.now()
        self._portfolio_history.append((timestamp, portfolio_value))

        # Keep only recent history (last 252 trading days)
        if len(self._portfolio_history) > 252:
            self._portfolio_history = self._portfolio_history[-252:]

        # Update peak and drawdown
        if self._current_portfolio:
            values = [value for _, value in self._portfolio_history]
            peak_value = max(values)
            current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            self._current_portfolio.peak_value = peak_value
            self._current_portfolio.max_drawdown = current_drawdown
    def get_risk_alerts(self, active_only: bool = True) -> Dict[RiskAlert, List[str]]:
        """
        Get current risk alerts.
        Args:
            active_only: If True, return only active alerts
        Returns:
            Dictionary of alert type -> List[str] of messages
        """
        if active_only:
            return dict(self._active_alerts)

        # Return recent alert history
        recent_alerts = defaultdict(List[str])
        cutoff_time = datetime.now() - timedelta(hours=24)
        for alert_record in self._alert_history:
            if alert_record['timestamp'] > cutoff_time:
                for alert_type, messages in alert_record['alerts'].items():
                    recent_alerts[RiskAlert(alert_type)].extend(messages)
        return dict(recent_alerts)
    def clear_alerts(self, alert_type: Optional[RiskAlert] = None) -> None:
        """Clear risk alerts."""
        if alert_type:
            self._active_alerts.pop(alert_type, None)
        else:
            self._active_alerts.clear()
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio risk statistics."""
        return {
            'risk_assessments': self._risk_assessments,
            'rejections': self._rejections,
            'emergency_exits': self._emergency_exits,
            'rejection_rate': self._rejections / self._risk_assessments if self._risk_assessments > 0 else 0,
            'active_alerts': sum(len(alerts) for alerts in self._active_alerts.values()),
            'correlation_age_hours': (
                (datetime.now() - self._correlation_last_updated).total_seconds() / 3600
                if self._correlation_last_updated else float('inf')
            )
        }

    # Private methods
    def _create_test_portfolio(
        self,
        portfolio_state: PortfolioState,
        proposed_position: Optional[Position]
    ) -> PortfolioState:
        """Create test portfolio including proposed position."""
        test_positions = portfolio_state.positions.copy()
        if proposed_position:

            # Check if we already have this symbol
            existing_pos = None
            for i, pos in enumerate(test_positions):
                if pos.symbol == proposed_position.symbol:
                    existing_pos = i
                    break
            if existing_pos is not None:

                # Update existing position
                old_pos = test_positions[existing_pos]
                total_shares = old_pos.shares + proposed_position.shares
                if total_shares > 0:

                    # Weighted average entry price
                    weighted_price = (
                        (old_pos.shares * old_pos.entry_price +
                         proposed_position.shares * proposed_position.entry_price) / total_shares
                    )
                    test_positions[existing_pos] = Position(
                        symbol=old_pos.symbol,
                        shares=total_shares,
                        entry_price=weighted_price,
                        current_price=proposed_position.current_price,
                        sector=old_pos.sector or proposed_position.sector,
                        entry_date=old_pos.entry_date,
                        liquidity_tier=old_pos.liquidity_tier or proposed_position.liquidity_tier
                    )
                else:

                    # Remove position if shares become zero or negative
                    test_positions.pop(existing_pos)
            else:

                # Add new position
                test_positions.append(proposed_position)
        return PortfolioState(
            positions=test_positions,
            cash_available=portfolio_state.cash_available,
            total_capital=portfolio_state.total_capital,
            daily_pnl=portfolio_state.daily_pnl,
            max_drawdown=portfolio_state.max_drawdown,
            peak_value=portfolio_state.peak_value,
            margin_used=portfolio_state.margin_used
        )
    def _assess_sector_risk(
        self,
        portfolio_state: PortfolioState,
        risk_details: Dict[str, float]
    ) -> Tuple[float, List[RiskAlert]]:
        """Assess sector diversification risk."""
        sc = self.config.sector_config
        alerts = []
        if not sc.enable_sector_limits:
            risk_details['sector_risk'] = 0.0
            return 0.0, alerts
        sector_exposures = portfolio_state.sector_exposures
        max_exposure = max(sector_exposures.values()) if sector_exposures else 0.0

        # Check maximum sector exposure
        if max_exposure > sc.max_sector_exposure:
            alerts.append(RiskAlert.SECTOR_CONCENTRATION)

        # Check concentrated sectors
        concentrated_sectors = sum(
            1 for exposure in sector_exposures.values()
            if exposure > 0.15
  # 15% threshold
        )
        if concentrated_sectors > sc.max_sectors_concentrated:
            alerts.append(RiskAlert.SECTOR_CONCENTRATION)

        # Calculate sector risk score (0-1)
        sector_risk = min(max_exposure / sc.max_sector_exposure, 1.0)
        risk_details.update({
            'sector_risk': sector_risk,
            'max_sector_exposure': max_exposure,
            'concentrated_sectors': concentrated_sectors,
            'sector_exposures': sector_exposures
        })
        return sector_risk, alerts
    def _assess_correlation_risk(
        self,
        portfolio_state: PortfolioState,
        risk_details: Dict[str, float]
    ) -> Tuple[float, List[RiskAlert]]:
        """Assess correlation risk."""
        cc = self.config.correlation_config
        alerts = []
        if not cc.enable_correlation_checks:
            risk_details['correlation_risk'] = 0.0
            return 0.0, alerts
        if self._correlation_matrix is None:
            risk_details['correlation_risk'] = 0.5
  # Unknown risk
            return 0.5, alerts
        symbols = [pos.symbol for pos in portfolio_state.positions]
        available_symbols = [s for s in symbols if s in self._correlation_matrix.index]
        if len(available_symbols) < 2:
            risk_details['correlation_risk'] = 0.0
            return 0.0, alerts

        # Calculate portfolio correlation metrics
        correlations = []
        for i, sym1 in enumerate(available_symbols):
            for sym2 in available_symbols[i+1:]:
                corr = abs(self._correlation_matrix.loc[sym1, sym2])
                correlations.append(corr)
                if corr > cc.max_pair_correlation:
                    alerts.append(RiskAlert.HIGH_CORRELATION)
        avg_correlation = np.mean(correlations) if correlations else 0.0
        max_correlation = max(correlations) if correlations else 0.0
        if avg_correlation > cc.max_portfolio_correlation:
            alerts.append(RiskAlert.HIGH_CORRELATION)

        # Risk score based on average correlation
        correlation_risk = min(avg_correlation / cc.max_portfolio_correlation, 1.0)
        risk_details.update({
            'correlation_risk': correlation_risk,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'correlation_pairs': len(correlations)
        })
        return correlation_risk, alerts
    def _assess_drawdown_risk(
        self,
        portfolio_state: PortfolioState,
        risk_details: Dict[str, float]
    ) -> Tuple[float, List[RiskAlert]]:
        """Assess drawdown risk."""
        dc = self.config.drawdown_config
        alerts = []
        if not dc.enable_drawdown_monitoring:
            risk_details['drawdown_risk'] = 0.0
            return 0.0, alerts
        current_drawdown = abs(portfolio_state.max_drawdown)
        daily_loss_pct = abs(portfolio_state.daily_pnl / portfolio_state.total_capital) if portfolio_state.total_capital > 0 else 0

        # Check drawdown thresholds
        if current_drawdown > dc.stop_new_positions_at:
            alerts.append(RiskAlert.PORTFOLIO_DRAWDOWN)
        if current_drawdown > dc.emergency_exit_at:
            alerts.append(RiskAlert.PORTFOLIO_DRAWDOWN)
        if daily_loss_pct > dc.max_daily_drawdown:
            alerts.append(RiskAlert.DAILY_LOSS_LIMIT)

        # Risk score based on drawdown level
        drawdown_risk = min(current_drawdown / dc.max_portfolio_drawdown, 1.0)
        risk_details.update({
            'drawdown_risk': drawdown_risk,
            'current_drawdown': current_drawdown,
            'daily_loss_pct': daily_loss_pct,
            'peak_value': portfolio_state.peak_value
        })
        return drawdown_risk, alerts
    def _assess_liquidity_concentration(
        self,
        portfolio_state: PortfolioState,
        risk_details: Dict[str, float]
    ) -> Tuple[float, List[RiskAlert]]:
        """Assess liquidity concentration risk."""
        alerts = []

        # Count positions by liquidity tier
        tier_counts = defaultdict(int)
        tier_values = defaultdict(float)
        for pos in portfolio_state.positions:
            tier = pos.liquidity_tier or LiquidityTier.ILLIQUID
            tier_counts[tier] += 1
            tier_values[tier] += pos.position_value
        total_value = portfolio_state.total_position_value

        # Check for too many illiquid positions
        illiquid_pct = tier_values[LiquidityTier.ILLIQUID] / total_value if total_value > 0 else 0
        if illiquid_pct > 0.2:
  # 20% threshold
            alerts.append(RiskAlert.LIQUIDITY_CONCENTRATION)

        # Risk score based on liquidity distribution
        liquidity_score = (
            tier_values.get(LiquidityTier.HIGHLY_LIQUID, 0) * 1.0 +
            tier_values.get(LiquidityTier.MODERATELY_LIQUID, 0) * 0.7 +
            tier_values.get(LiquidityTier.LOW_LIQUID, 0) * 0.4 +
            tier_values.get(LiquidityTier.ILLIQUID, 0) * 0.1
        ) / total_value if total_value > 0 else 1.0
        liquidity_risk = 1.0 - liquidity_score
        risk_details.update({
            'liquidity_risk': liquidity_risk,
            'illiquid_percentage': illiquid_pct,
            'liquidity_score': liquidity_score,
            'tier_distribution': Dict[str, Any](tier_values)
        })
        return liquidity_risk, alerts
    def _assess_portfolio_limits(
        self,
        portfolio_state: PortfolioState,
        risk_details: Dict[str, float]
    ) -> Tuple[float, List[RiskAlert]]:
        """Assess portfolio limit violations."""
        alerts = []

        # Position count limit
        if len(portfolio_state.positions) > self.config.max_positions:
            alerts.append(RiskAlert.POSITION_LIMIT_BREACH)

        # Portfolio risk limit
        total_risk = portfolio_state.total_position_value / portfolio_state.total_capital if portfolio_state.total_capital > 0 else 0
        if total_risk > self.config.max_portfolio_risk:
            alerts.append(RiskAlert.RISK_BUDGET_EXCEEDED)

        # Calculate limit risk score
        position_risk = min(len(portfolio_state.positions) / self.config.max_positions, 1.0)
        capital_risk = min(total_risk / self.config.max_portfolio_risk, 1.0)
        limit_risk = max(position_risk, capital_risk)
        risk_details.update({
            'limit_risk': limit_risk,
            'position_count': len(portfolio_state.positions),
            'capital_utilization': total_risk,
            'margin_utilization': portfolio_state.margin_used / portfolio_state.total_capital if portfolio_state.total_capital > 0 else 0
        })
        return limit_risk, alerts
    def _calculate_overall_risk_score(
        self,
        sector_risk: float,
        correlation_risk: float,
        drawdown_risk: float,
        liquidity_risk: float,
        limit_risk: float
    ) -> float:
        """Calculate weighted overall risk score."""

        # Weights for different risk components
        weights = {
            'sector': 0.15,
            'correlation': 0.20,
            'drawdown': 0.30,
            'liquidity': 0.15,
            'limit': 0.20
        }
        overall_score = (
            sector_risk * weights['sector'] +
            correlation_risk * weights['correlation'] +
            drawdown_risk * weights['drawdown'] +
            liquidity_risk * weights['liquidity'] +
            limit_risk * weights['limit']
        )
        return min(overall_score, 1.0)
    def _determine_portfolio_action(
        self,
        risk_score: float,
        alerts: List[RiskAlert]
    ) -> PortfolioAction:
        """Determine action based on risk score and alerts."""

        # Emergency conditions
        if RiskAlert.PORTFOLIO_DRAWDOWN in alerts:
            dc = self.config.drawdown_config
            current_dd = self._current_portfolio.max_drawdown if self._current_portfolio else 0
            if abs(current_dd) > dc.emergency_exit_at:
                return PortfolioAction.EMERGENCY_EXIT
            elif abs(current_dd) > dc.stop_new_positions_at:
                return PortfolioAction.CLOSE_POSITIONS

        # High risk conditions
        if risk_score > 0.8:
            return PortfolioAction.REJECT
        elif risk_score > 0.6:
            return PortfolioAction.REDUCE_SIZE
        else:
            return PortfolioAction.ALLOW
    def _generate_recommendations(
        self,
        alerts: List[RiskAlert]
    ) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        if RiskAlert.SECTOR_CONCENTRATION in alerts:
            recommendations.append("Consider diversifying across more sectors")
        if RiskAlert.HIGH_CORRELATION in alerts:
            recommendations.append("Reduce position sizes in highly correlated stocks")
        if RiskAlert.PORTFOLIO_DRAWDOWN in alerts:
            recommendations.append("Consider reducing position sizes or closing losing positions")
        if RiskAlert.LIQUIDITY_CONCENTRATION in alerts:
            recommendations.append("Increase allocation to more liquid stocks")
        if RiskAlert.POSITION_LIMIT_BREACH in alerts:
            recommendations.append("Close some positions before adding new ones")
        return recommendations
    def _update_alert_tracking(self, alerts: List[RiskAlert]) -> None:
        """Update alert tracking and history."""

        # Clear existing alerts
        self._active_alerts.clear()

        # Set[str] new alerts
        for alert in alerts:
            self._active_alerts[alert].append(f"Alert triggered at {datetime.now()}")

        # Add to history
        if alerts:
            self._alert_history.append({
                'timestamp': datetime.now(),
                'alerts': {alert.value: [f"Alert at {datetime.now()}"] for alert in alerts}
            })

            # Keep history manageable
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]

# Export main components
__all__ = [
    'PortfolioRiskController',
    'PortfolioRiskResult',
    'RiskAlert',
    'PortfolioAction',
    'Position',
    'PortfolioState',
]
