"""
Enhanced Risk Manager for FS.5 Implementation

This module integrates all FS.5 risk management components into a unified system:
- Enhanced position sizing with ATR-based calculations
- Comprehensive liquidity validation and constraints
- Portfolio-level risk controls and diversification limits
- Real-time risk monitoring and alerting
- NSE compliance and margin requirements

This replaces and enhances the existing risk_manager.py with full FS.5 compliance
while maintaining backward compatibility with existing interfaces.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

from .risk_config import RiskConfig, create_default_risk_config
from .position_sizer import (
    EnhancedPositionSizer, StockData, PositionSizeResult, 
    create_stock_data_from_dict, create_portfolio_state_from_positions
)
from .liquidity_validator import LiquidityValidator, VolumeData, LiquidityValidationResult
from .portfolio_controller import (
    PortfolioRiskController, Position, PortfolioState, 
    PortfolioRiskResult, RiskAlert, PortfolioAction
)


class RiskDecision(NamedTuple):
    """Comprehensive risk decision result."""
    approved: bool
    action: PortfolioAction
    position_size: int
    position_value: float
    risk_amount: float
    margin_required: float
    warnings: List[str]
    alerts: List[RiskAlert]
    risk_score: float
    details: Dict[str, Any]


class EnhancedRiskManager:
    """
    Enhanced Risk Manager implementing FS.5 requirements.
    
    Integrates position sizing, liquidity validation, and portfolio risk controls
    into a unified risk management system with comprehensive monitoring and alerting.
    """
    
    def __init__(self, risk_config: Optional[RiskConfig] = None):
        """
        Initialize enhanced risk manager.
        
        Args:
            risk_config: Optional risk configuration (uses default if None)
        """
        self.config = risk_config or create_default_risk_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize component modules
        self.position_sizer = EnhancedPositionSizer(self.config)
        self.liquidity_validator = LiquidityValidator(self.config)
        self.portfolio_controller = PortfolioRiskController(self.config)
        
        # Current portfolio state
        self._current_portfolio: Optional[PortfolioState] = None
        
        # Performance tracking
        self._total_decisions = 0
        self._approved_decisions = 0
        self._rejected_decisions = 0
        
        self.logger.info("Enhanced Risk Manager initialized with FS.5 compliance")
    
    def evaluate_position(
        self,
        symbol: str,
        signal_score: int,
        stock_data: Dict[str, Any],
        portfolio_data: Optional[Dict[str, Any]] = None,
        volume_data: Optional[VolumeData] = None
    ) -> RiskDecision:
        """
        Comprehensive position evaluation with all FS.5 risk controls.
        
        Args:
            symbol: Stock symbol
            signal_score: Signal strength score (0-100)
            stock_data: Dictionary containing stock data
            portfolio_data: Optional current portfolio data
            volume_data: Optional historical volume data
            
        Returns:
            RiskDecision with approval status and detailed analysis
        """
        self._total_decisions += 1
        
        try:
            # 1. Prepare input data
            stock_input = self._prepare_stock_data(symbol, stock_data)
            portfolio_state = self._prepare_portfolio_state(portfolio_data)
            
            # 2. Calculate position size
            position_result = self.position_sizer.calculate_position_size(
                stock_data=stock_input,
                portfolio_state=portfolio_state,
                signal_score=signal_score
            )
            
            # 3. Validate liquidity
            liquidity_result = self.liquidity_validator.validate_position_liquidity(
                symbol=symbol,
                position_value=position_result.position_value,
                volume_data=volume_data
            )
            
            # 4. Check portfolio risk
            proposed_position = Position(
                symbol=symbol,
                shares=position_result.shares,
                entry_price=stock_input.current_price,
                current_price=stock_input.current_price,
                sector=stock_input.sector,
                entry_date=datetime.now(),
                liquidity_tier=liquidity_result.liquidity_tier
            )
            
            portfolio_result = self.portfolio_controller.assess_portfolio_risk(
                portfolio_state=portfolio_state,
                proposed_position=proposed_position
            )
            
            # 5. Make final decision
            decision = self._make_risk_decision(
                position_result, liquidity_result, portfolio_result, stock_input
            )
            
            # 6. Update tracking
            if decision.approved:
                self._approved_decisions += 1
            else:
                self._rejected_decisions += 1
            
            # 7. Log decision
            self._log_decision(symbol, decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Position evaluation failed for {symbol}: {e}")
            return self._create_error_decision(symbol, str(e))
    
    def update_portfolio_state(
        self,
        positions: List[Dict[str, Any]],
        cash_available: float,
        total_capital: float,
        price_data: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Update current portfolio state for risk calculations.
        
        Args:
            positions: List of current position dictionaries
            cash_available: Available cash
            total_capital: Total portfolio capital
            price_data: Optional price history for correlation calculation
        """
        # Convert position data
        portfolio_positions = []
        for pos_data in positions:
            position = Position(
                symbol=pos_data['symbol'],
                shares=pos_data.get('shares', 0),
                entry_price=pos_data.get('entry_price', 0.0),
                current_price=pos_data.get('current_price', 0.0),
                sector=pos_data.get('sector'),
                entry_date=pos_data.get('entry_date'),
                unrealized_pnl=pos_data.get('unrealized_pnl', 0.0),
                stop_loss=pos_data.get('stop_loss'),
                liquidity_tier=pos_data.get('liquidity_tier')
            )
            portfolio_positions.append(position)
        
        # Calculate daily P&L and drawdown
        current_value = sum(pos.position_value for pos in portfolio_positions) + cash_available
        daily_pnl = sum(pos.unrealized_pnl for pos in portfolio_positions)
        
        # Update portfolio state
        self._current_portfolio = PortfolioState(
            positions=portfolio_positions,
            cash_available=cash_available,
            total_capital=total_capital,
            daily_pnl=daily_pnl,
            max_drawdown=0.0,  # Will be calculated by controller
            peak_value=0.0     # Will be calculated by controller
        )
        
        # Update portfolio history in controller
        self.portfolio_controller.update_portfolio_history(current_value)
        
        # Update correlation matrix if price data provided
        if price_data:
            self.portfolio_controller.calculate_correlation_matrix(price_data)
        
        self.logger.info(f"Portfolio state updated: {len(portfolio_positions)} positions, "
                        f"${current_value:,.0f} total value")
    
    def update_volume_data(self, symbol: str, volume_data: VolumeData) -> None:
        """Update volume data for liquidity analysis."""
        self.liquidity_validator.update_volume_data(symbol, volume_data)
    
    def get_risk_alerts(self) -> Dict[str, List[str]]:
        """Get all active risk alerts from all components."""
        alerts = {}
        
        # Portfolio alerts
        portfolio_alerts = self.portfolio_controller.get_risk_alerts()
        for alert_type, messages in portfolio_alerts.items():
            alerts[f"portfolio_{alert_type.value}"] = messages
        
        # Liquidity alerts
        liquidity_alerts = self.liquidity_validator.get_liquidity_alerts()
        for symbol, alert_list in liquidity_alerts.items():
            alert_key = f"liquidity_{symbol}"
            alerts[alert_key] = [alert.value for alert in alert_list]
        
        return alerts
    
    def clear_alerts(self, alert_type: Optional[str] = None) -> None:
        """Clear risk alerts from all components."""
        self.portfolio_controller.clear_alerts()
        self.liquidity_validator.clear_alerts()
        self.logger.info(f"Cleared {alert_type or 'all'} risk alerts")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        if not self._current_portfolio:
            return {"error": "No portfolio data available"}
        
        portfolio = self._current_portfolio
        
        # Basic portfolio metrics
        summary = {
            "total_positions": len(portfolio.positions),
            "total_capital": portfolio.total_capital,
            "cash_available": portfolio.cash_available,
            "portfolio_value": portfolio.portfolio_value,
            "cash_utilization": portfolio.cash_utilization,
            "daily_pnl": portfolio.daily_pnl,
            "max_drawdown": portfolio.max_drawdown,
            "sector_exposures": portfolio.sector_exposures
        }
        
        # Risk component statistics
        summary.update({
            "position_sizer_stats": self.position_sizer.get_position_sizing_stats(),
            "liquidity_stats": self.liquidity_validator.get_validation_stats(),
            "portfolio_stats": self.portfolio_controller.get_portfolio_stats(),
            "risk_manager_stats": self._get_risk_manager_stats()
        })
        
        return summary
    
    def validate_configuration(self) -> List[str]:
        """Validate risk configuration and return any errors."""
        return self.config.validate()
    
    def reconfigure(self, new_config: RiskConfig) -> None:
        """Update risk configuration and reinitialize components."""
        self.config = new_config
        
        # Reinitialize components with new config
        self.position_sizer = EnhancedPositionSizer(self.config)
        self.liquidity_validator = LiquidityValidator(self.config)
        self.portfolio_controller = PortfolioRiskController(self.config)
        
        self.logger.info("Risk manager reconfigured with new settings")
    
    def export_risk_report(self) -> Dict[str, Any]:
        """Export comprehensive risk report for analysis."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_summary": {
                "portfolio_capital": self.config.portfolio_capital,
                "max_positions": self.config.max_positions,
                "max_portfolio_risk": self.config.max_portfolio_risk,
                "base_risk_per_trade": self.config.position_sizing.base_risk_per_trade,
                "max_sector_exposure": self.config.sector_config.max_sector_exposure,
                "liquidity_checks_enabled": self.config.liquidity_config.enable_liquidity_checks,
                "correlation_checks_enabled": self.config.correlation_config.enable_correlation_checks,
                "drawdown_monitoring": self.config.drawdown_config.enable_drawdown_monitoring
            },
            "portfolio_summary": self.get_portfolio_summary(),
            "active_alerts": self.get_risk_alerts(),
            "performance_metrics": {
                "total_decisions": self._total_decisions,
                "approval_rate": self._approved_decisions / self._total_decisions if self._total_decisions > 0 else 0,
                "rejection_rate": self._rejected_decisions / self._total_decisions if self._total_decisions > 0 else 0
            }
        }
        
        return report
    
    # Private methods
    
    def _prepare_stock_data(self, symbol: str, stock_data: Dict[str, Any]) -> StockData:
        """Prepare StockData from input dictionary."""
        return StockData(
            symbol=symbol,
            current_price=float(stock_data['current_price']),
            atr=float(stock_data.get('atr', stock_data['current_price'] * 0.02)),  # Default 2%
            avg_daily_volume=float(stock_data.get('avg_daily_volume', 1000000)),  # Default 1M
            sector=stock_data.get('sector'),
            lot_size=stock_data.get('lot_size'),
            volatility=stock_data.get('volatility'),
            margin_requirement=stock_data.get('margin_requirement')
        )
    
    def _prepare_portfolio_state(self, portfolio_data: Optional[Dict[str, Any]]) -> PortfolioState:
        """Prepare PortfolioState from input data or use current state."""
        if portfolio_data is not None:
            positions = [
                Position(
                    symbol=pos['symbol'],
                    shares=pos.get('shares', 0),
                    entry_price=pos.get('entry_price', 0.0),
                    current_price=pos.get('current_price', 0.0),
                    sector=pos.get('sector'),
                    liquidity_tier=pos.get('liquidity_tier')
                )
                for pos in portfolio_data.get('positions', [])
            ]
            
            return PortfolioState(
                positions=positions,
                cash_available=portfolio_data.get('cash_available', 0.0),
                total_capital=portfolio_data.get('total_capital', self.config.portfolio_capital)
            )
        
        # Use current portfolio state or create default
        return self._current_portfolio or PortfolioState(
            positions=[],
            cash_available=self.config.portfolio_capital,
            total_capital=self.config.portfolio_capital
        )
    
    def _make_risk_decision(
        self,
        position_result: PositionSizeResult,
        liquidity_result: LiquidityValidationResult,
        portfolio_result: PortfolioRiskResult,
        stock_data: StockData
    ) -> RiskDecision:
        """Make final risk decision based on all validation results."""
        
        # Collect all warnings and alerts
        all_warnings = position_result.warnings.copy()
        all_warnings.extend(liquidity_result.recommendations)
        all_warnings.extend(portfolio_result.recommendations)
        
        all_alerts = portfolio_result.alerts.copy()
        all_alerts.extend(liquidity_result.alerts)
        
        # Determine approval
        approved = (
            position_result.shares > 0 and
            position_result.lot_size_compliant and
            position_result.liquidity_approved and
            liquidity_result.approved and
            portfolio_result.action in [PortfolioAction.ALLOW, PortfolioAction.REDUCE_SIZE]
        )
        
        # Adjust position size if needed
        final_shares = position_result.shares
        final_value = position_result.position_value
        
        if portfolio_result.action == PortfolioAction.REDUCE_SIZE:
            # Reduce position size based on portfolio risk
            reduction_factor = max(0.5, 1.0 - (portfolio_result.risk_score - 0.6) / 0.4)
            final_shares = int(final_shares * reduction_factor)
            final_value = final_shares * stock_data.current_price
            all_warnings.append(f"Position reduced by {(1-reduction_factor)*100:.0f}% due to portfolio risk")
        
        # Collect detailed analysis
        details = {
            "position_calculation": position_result.calculation_details,
            "liquidity_analysis": liquidity_result.analysis_details,
            "portfolio_risk": portfolio_result.risk_details,
            "signal_score": position_result.calculation_details.get('signal_score', 0),
            "risk_multiplier": position_result.calculation_details.get('risk_multiplier', 1.0),
            "liquidity_tier": liquidity_result.liquidity_tier.value,
            "market_impact": liquidity_result.estimated_market_impact,
            "portfolio_action": portfolio_result.action.value
        }
        
        return RiskDecision(
            approved=approved,
            action=portfolio_result.action,
            position_size=final_shares,
            position_value=final_value,
            risk_amount=position_result.risk_amount,
            margin_required=position_result.margin_required,
            warnings=all_warnings,
            alerts=all_alerts,
            risk_score=portfolio_result.risk_score,
            details=details
        )
    
    def _create_error_decision(self, symbol: str, error_message: str) -> RiskDecision:
        """Create error decision when evaluation fails."""
        return RiskDecision(
            approved=False,
            action=PortfolioAction.REJECT,
            position_size=0,
            position_value=0.0,
            risk_amount=0.0,
            margin_required=0.0,
            warnings=[f"Evaluation failed: {error_message}"],
            alerts=[],
            risk_score=1.0,  # Maximum risk
            details={"error": error_message, "symbol": symbol}
        )
    
    def _log_decision(self, symbol: str, decision: RiskDecision) -> None:
        """Log risk decision for monitoring."""
        status = "APPROVED" if decision.approved else "REJECTED"
        
        self.logger.info(
            f"Risk Decision [{status}]: {symbol} - "
            f"Shares: {decision.position_size}, "
            f"Value: ${decision.position_value:,.0f}, "
            f"Risk Score: {decision.risk_score:.2f}, "
            f"Action: {decision.action.value}"
        )
        
        if decision.warnings:
            self.logger.warning(f"Warnings for {symbol}: {'; '.join(decision.warnings)}")
        
        if decision.alerts:
            alert_names = [alert.value for alert in decision.alerts]
            self.logger.warning(f"Alerts for {symbol}: {'; '.join(alert_names)}")
    
    def _get_risk_manager_stats(self) -> Dict[str, Any]:
        """Get risk manager performance statistics."""
        return {
            "total_decisions": self._total_decisions,
            "approved_decisions": self._approved_decisions,
            "rejected_decisions": self._rejected_decisions,
            "approval_rate": (
                self._approved_decisions / self._total_decisions 
                if self._total_decisions > 0 else 0
            ),
            "rejection_rate": (
                self._rejected_decisions / self._total_decisions 
                if self._total_decisions > 0 else 0
            )
        }


# Backward compatibility functions for existing code

def create_enhanced_risk_manager(
    portfolio_capital: float = 1000000.0,
    max_positions: int = 10,
    risk_per_trade: float = 0.01
) -> EnhancedRiskManager:
    """
    Create enhanced risk manager with simplified parameters for backward compatibility.
    
    Args:
        portfolio_capital: Total portfolio capital
        max_positions: Maximum number of positions
        risk_per_trade: Risk per trade as percentage
        
    Returns:
        Configured EnhancedRiskManager instance
    """
    config = create_default_risk_config()
    config.portfolio_capital = portfolio_capital
    config.max_positions = max_positions
    config.position_sizing.base_risk_per_trade = risk_per_trade
    
    return EnhancedRiskManager(config)


def calculate_position_size_simple(
    symbol: str,
    current_price: float,
    atr: float,
    signal_score: int,
    risk_manager: EnhancedRiskManager
) -> Tuple[int, float, List[str]]:
    """
    Simple position size calculation for backward compatibility.
    
    Args:
        symbol: Stock symbol
        current_price: Current stock price
        atr: Average True Range
        signal_score: Signal strength (0-100)
        risk_manager: Enhanced risk manager instance
        
    Returns:
        Tuple of (shares, position_value, warnings)
    """
    stock_data = {
        'current_price': current_price,
        'atr': atr,
        'avg_daily_volume': 10000000  # Default 10M volume
    }
    
    decision = risk_manager.evaluate_position(
        symbol=symbol,
        signal_score=signal_score,
        stock_data=stock_data
    )
    
    return decision.position_size, decision.position_value, decision.warnings


# Export main components
__all__ = [
    'EnhancedRiskManager',
    'RiskDecision',
    'create_enhanced_risk_manager',
    'calculate_position_size_simple',
]