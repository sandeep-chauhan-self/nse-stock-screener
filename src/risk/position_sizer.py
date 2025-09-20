"""
Enhanced Position Sizing Module for FS.5 Implementation

This module provides comprehensive position sizing calculations including:
- ATR-based risk calculation with dynamic adjustments
- NSE lot size enforcement and validation
- Margin requirement calculations
- Volatility-based position sizing adjustments
- Signal score-based risk multipliers
- Liquidity tier adjustments

Features:
- Type-safe calculations with comprehensive validation
- NSE-specific lot size and margin handling
- Dynamic volatility parity adjustments
- Risk multiplier scaling based on signal strength
- Integration with enhanced risk configuration
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from .risk_config import RiskConfig, LiquidityTier, SectorRiskTier


class PositionSizeResult(NamedTuple):
    """Result of position sizing calculation with detailed breakdown."""
    shares: int
    position_value: float
    risk_amount: float
    margin_required: float
    lot_size_compliant: bool
    liquidity_approved: bool
    warnings: List[str]
    calculation_details: Dict[str, float]


class PositionSizeError(Exception):
    """Custom exception for position sizing errors."""
    pass


@dataclass
class StockData:
    """Input data required for position sizing calculations."""
    symbol: str
    current_price: float
    atr: float  # Average True Range
    avg_daily_volume: float  # In rupees
    sector: Optional[str] = None
    lot_size: Optional[int] = None
    volatility: Optional[float] = None  # Annualized volatility
    margin_requirement: Optional[float] = None


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    total_capital: float
    cash_available: float
    current_positions: List[Dict]  # List of current position data
    sector_exposures: Dict[str, float]  # Sector -> exposure amount
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0


class EnhancedPositionSizer:
    """
    Enhanced position sizing calculator implementing FS.5 requirements.
    
    Provides ATR-based risk calculations, NSE compliance, margin validation,
    and comprehensive risk controls for position sizing decisions.
    """
    
    def __init__(self, risk_config: RiskConfig):
        """
        Initialize position sizer with risk configuration.
        
        Args:
            risk_config: Comprehensive risk management configuration
        """
        self.config = risk_config
        self.logger = logging.getLogger(__name__)
        
        # Cache for lot sizes and other data
        self._lot_size_cache: Dict[str, int] = {}
        self._volume_cache: Dict[str, float] = {}
        
        # Validation counters
        self._position_count = 0
        self._rejected_count = 0
        
    def calculate_position_size(
        self,
        stock_data: StockData,
        portfolio_state: PortfolioState,
        signal_score: int
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on comprehensive risk analysis.
        
        Args:
            stock_data: Stock data including price, ATR, volume, etc.
            portfolio_state: Current portfolio state and exposures
            signal_score: Signal strength score (0-100) for risk multiplier
            
        Returns:
            PositionSizeResult with shares, value, risk, and validation details
            
        Raises:
            PositionSizeError: If calculation fails due to validation errors
        """
        warnings_list = []
        calculation_details = {}
        
        try:
            # Step 1: Basic validation
            self._validate_inputs(stock_data, portfolio_state, signal_score)
            
            # Step 2: Calculate base risk amount
            base_risk = self._calculate_base_risk(
                portfolio_state, signal_score, calculation_details
            )
            
            # Step 3: ATR-based stop distance calculation
            stop_distance = self._calculate_atr_stop_distance(
                stock_data, calculation_details
            )
            
            # Step 4: Volatility-adjusted position size
            raw_position_size = self._calculate_volatility_adjusted_size(
                base_risk, stock_data, stop_distance, calculation_details
            )
            
            # Step 5: Apply liquidity constraints
            liquidity_adjusted_size = self._apply_liquidity_constraints(
                raw_position_size, stock_data, portfolio_state, 
                warnings_list, calculation_details
            )
            
            # Step 6: NSE lot size compliance
            lot_compliant_size = self._enforce_nse_lot_sizes(
                liquidity_adjusted_size, stock_data, warnings_list, calculation_details
            )
            
            # Step 7: Portfolio constraints
            final_position_size = self._apply_portfolio_constraints(
                lot_compliant_size, stock_data, portfolio_state,
                warnings_list, calculation_details
            )
            
            # Step 8: Margin validation
            margin_required = self._calculate_margin_requirement(
                final_position_size, stock_data, calculation_details
            )
            
            # Step 9: Final validation and result construction
            return self._build_result(
                final_position_size, stock_data, margin_required,
                warnings_list, calculation_details
            )
            
        except Exception as e:
            self.logger.error(f"Position sizing failed for {stock_data.symbol}: {e}")
            raise PositionSizeError(f"Position sizing calculation failed: {e}")
    
    def _validate_inputs(
        self, 
        stock_data: StockData, 
        portfolio_state: PortfolioState, 
        signal_score: int
    ) -> None:
        """Validate all input parameters."""
        vc = self.config.validation_config
        
        # Stock data validation
        if stock_data.current_price < vc.min_stock_price:
            raise PositionSizeError(f"Stock price {stock_data.current_price} below minimum {vc.min_stock_price}")
        
        if stock_data.current_price > vc.max_stock_price:
            raise PositionSizeError(f"Stock price {stock_data.current_price} above maximum {vc.max_stock_price}")
        
        if stock_data.atr <= 0:
            raise PositionSizeError("ATR must be positive")
        
        if stock_data.avg_daily_volume <= 0:
            raise PositionSizeError("Average daily volume must be positive")
        
        # Portfolio validation
        if portfolio_state.cash_available < 0:
            raise PositionSizeError("Insufficient cash available")
        
        # Signal score validation
        if not 0 <= signal_score <= 100:
            raise PositionSizeError("Signal score must be between 0 and 100")
    
    def _calculate_base_risk(
        self,
        portfolio_state: PortfolioState,
        signal_score: int,
        calculation_details: Dict[str, float]
    ) -> float:
        """Calculate base risk amount with signal score multiplier."""
        ps_config = self.config.position_sizing
        
        # Base risk calculation
        base_risk_pct = ps_config.base_risk_per_trade
        risk_multiplier = self.config.get_risk_multiplier(signal_score)
        
        # Calculate risk amount
        base_risk = portfolio_state.total_capital * base_risk_pct * risk_multiplier
        
        # Store calculation details
        calculation_details.update({
            'base_risk_pct': base_risk_pct,
            'risk_multiplier': risk_multiplier,
            'signal_score': signal_score,
            'base_risk_amount': base_risk
        })
        
        return base_risk
    
    def _calculate_atr_stop_distance(
        self,
        stock_data: StockData,
        calculation_details: Dict[str, float]
    ) -> float:
        """Calculate stop distance based on ATR with validation."""
        ps_config = self.config.position_sizing
        
        # Default ATR multiplier
        atr_multiplier = ps_config.default_atr_multiplier
        
        # Validate ATR multiplier bounds
        if atr_multiplier < ps_config.min_stop_atr_ratio:
            atr_multiplier = ps_config.min_stop_atr_ratio
        elif atr_multiplier > ps_config.max_stop_atr_ratio:
            atr_multiplier = ps_config.max_stop_atr_ratio
        
        stop_distance = stock_data.atr * atr_multiplier
        
        # Store calculation details
        calculation_details.update({
            'atr': stock_data.atr,
            'atr_multiplier': atr_multiplier,
            'stop_distance': stop_distance,
            'stop_distance_pct': stop_distance / stock_data.current_price
        })
        
        return stop_distance
    
    def _calculate_volatility_adjusted_size(
        self,
        base_risk: float,
        stock_data: StockData,
        stop_distance: float,
        calculation_details: Dict[str, float]
    ) -> float:
        """Calculate position size with volatility parity adjustment."""
        ps_config = self.config.position_sizing
        
        # Basic position size calculation
        shares_base = base_risk / stop_distance
        position_value_base = shares_base * stock_data.current_price
        
        # Volatility parity adjustment if enabled
        if ps_config.enable_volatility_parity and stock_data.volatility:
            target_vol = ps_config.base_volatility_target
            current_vol = stock_data.volatility / np.sqrt(252)  # Convert to daily
            
            vol_adjustment = min(
                target_vol / current_vol if current_vol > 0 else 1.0,
                ps_config.volatility_adjustment_cap
            )
            
            position_value_adjusted = position_value_base * vol_adjustment
            
            calculation_details.update({
                'volatility_adjustment': vol_adjustment,
                'target_volatility': target_vol,
                'current_volatility': current_vol,
                'position_value_vol_adjusted': position_value_adjusted
            })
            
            return position_value_adjusted
        
        calculation_details['position_value_base'] = position_value_base
        return position_value_base
    
    def _apply_liquidity_constraints(
        self,
        position_value: float,
        stock_data: StockData,
        portfolio_state: PortfolioState,
        warnings_list: List[str],
        calculation_details: Dict[str, float]
    ) -> float:
        """Apply liquidity-based position size constraints."""
        lc = self.config.liquidity_config
        
        if not lc.enable_liquidity_checks:
            calculation_details['liquidity_adjustment'] = 1.0
            return position_value
        
        # Get liquidity tier and limits
        liquidity_tier = self.config.get_liquidity_tier(stock_data.avg_daily_volume)
        max_position_pct = lc.liquidity_tier_limits[liquidity_tier]
        
        # Volume-based constraint
        max_by_volume = stock_data.avg_daily_volume * lc.max_position_vs_avg_volume
        
        # Portfolio-based constraint
        max_by_portfolio = portfolio_state.total_capital * max_position_pct
        
        # Take the most restrictive constraint
        max_allowed = min(max_by_volume, max_by_portfolio)
        
        if position_value > max_allowed:
            adjustment_factor = max_allowed / position_value
            adjusted_value = max_allowed
            
            warnings_list.append(
                f"Position reduced by {(1-adjustment_factor)*100:.1f}% due to "
                f"liquidity constraints (tier: {liquidity_tier.value})"
            )
        else:
            adjustment_factor = 1.0
            adjusted_value = position_value
        
        calculation_details.update({
            'liquidity_tier': liquidity_tier.value,
            'max_by_volume': max_by_volume,
            'max_by_portfolio': max_by_portfolio,
            'liquidity_adjustment': adjustment_factor,
            'position_value_liquidity_adjusted': adjusted_value
        })
        
        return adjusted_value
    
    def _enforce_nse_lot_sizes(
        self,
        position_value: float,
        stock_data: StockData,
        warnings_list: List[str],
        calculation_details: Dict[str, float]
    ) -> float:
        """Enforce NSE lot size requirements."""
        nse_config = self.config.nse_lot_config
        
        if not nse_config.enforce_lot_sizes:
            calculation_details['lot_size_adjustment'] = 1.0
            return position_value
        
        # Get lot size
        lot_size = self._get_lot_size(stock_data.symbol, stock_data.lot_size)
        
        # Calculate shares and round to lot size
        target_shares = position_value / stock_data.current_price
        lot_compliant_shares = self._round_to_lot_size(target_shares, lot_size)
        
        # Calculate adjusted position value
        adjusted_position_value = lot_compliant_shares * stock_data.current_price
        
        # Check if adjustment was significant
        if abs(adjusted_position_value - position_value) / position_value > 0.1:
            warnings_list.append(
                f"Position adjusted by "
                f"{abs(adjusted_position_value - position_value)/position_value*100:.1f}% "
                f"for lot size compliance (lot size: {lot_size})"
            )
        
        calculation_details.update({
            'lot_size': lot_size,
            'target_shares': target_shares,
            'lot_compliant_shares': lot_compliant_shares,
            'lot_size_adjustment': adjusted_position_value / position_value if position_value > 0 else 1.0,
            'position_value_lot_adjusted': adjusted_position_value
        })
        
        return adjusted_position_value
    
    def _apply_portfolio_constraints(
        self,
        position_value: float,
        stock_data: StockData,
        portfolio_state: PortfolioState,
        warnings_list: List[str],
        calculation_details: Dict[str, float]
    ) -> float:
        """Apply portfolio-level position size constraints."""
        ps_config = self.config.position_sizing
        
        # Maximum position size constraint
        max_position_value = portfolio_state.total_capital * ps_config.max_position_value_pct
        
        # Sector exposure constraint (if sector provided)
        sector_limit = None
        if stock_data.sector:
            sector_config = self.config.sector_config
            if sector_config.enable_sector_limits:
                sector_risk_tier = self.config.get_sector_risk_tier(stock_data.sector)
                sector_limit_pct = sector_config.sector_risk_limits[sector_risk_tier]
                
                current_sector_exposure = portfolio_state.sector_exposures.get(stock_data.sector, 0)
                available_sector_capacity = (portfolio_state.total_capital * sector_limit_pct) - current_sector_exposure
                sector_limit = max(0, available_sector_capacity)
        
        # Apply most restrictive constraint
        constraints = [max_position_value]
        if sector_limit is not None:
            constraints.append(sector_limit)
        
        final_limit = min(constraints)
        
        if position_value > final_limit:
            adjustment_factor = final_limit / position_value
            adjusted_value = final_limit
            
            if final_limit == sector_limit:
                warnings_list.append(
                    f"Position reduced by {(1-adjustment_factor)*100:.1f}% due to "
                    f"sector exposure limits ({stock_data.sector})"
                )
            else:
                warnings_list.append(
                    f"Position reduced by {(1-adjustment_factor)*100:.1f}% due to "
                    f"maximum position size limits"
                )
        else:
            adjustment_factor = 1.0
            adjusted_value = position_value
        
        calculation_details.update({
            'max_position_value': max_position_value,
            'sector_limit': sector_limit,
            'portfolio_adjustment': adjustment_factor,
            'position_value_final': adjusted_value
        })
        
        return adjusted_value
    
    def _calculate_margin_requirement(
        self,
        position_value: float,
        stock_data: StockData,
        calculation_details: Dict[str, float]
    ) -> float:
        """Calculate margin requirement for the position."""
        margin_required = self.config.calculate_margin_requirement(
            position_value=position_value,
            volatility=stock_data.volatility,
            instrument_type='equity'
        )
        
        calculation_details['margin_required'] = margin_required
        calculation_details['margin_pct'] = margin_required / position_value if position_value > 0 else 0
        
        return margin_required
    
    def _build_result(
        self,
        position_value: float,
        stock_data: StockData,
        margin_required: float,
        warnings_list: List[str],
        calculation_details: Dict[str, float]
    ) -> PositionSizeResult:
        """Build final position size result."""
        # Calculate final shares
        shares = int(position_value / stock_data.current_price)
        actual_position_value = shares * stock_data.current_price
        
        # Calculate actual risk amount
        stop_distance = calculation_details.get('stop_distance', stock_data.atr * 2.0)
        risk_amount = shares * stop_distance
        
        # Validation flags
        ps_config = self.config.position_sizing
        lot_size_compliant = (
            not self.config.nse_lot_config.enforce_lot_sizes or
            shares % self._get_lot_size(stock_data.symbol, stock_data.lot_size) == 0
        )
        
        lc = self.config.liquidity_config
        liquidity_approved = (
            not lc.enable_liquidity_checks or
            actual_position_value <= stock_data.avg_daily_volume * lc.max_position_vs_avg_volume
        )
        
        # Add minimum position size check
        if actual_position_value < ps_config.min_position_value:
            warnings_list.append(
                f"Position value {actual_position_value:.0f} below minimum "
                f"{ps_config.min_position_value:.0f}"
            )
        
        return PositionSizeResult(
            shares=shares,
            position_value=actual_position_value,
            risk_amount=risk_amount,
            margin_required=margin_required,
            lot_size_compliant=lot_size_compliant,
            liquidity_approved=liquidity_approved,
            warnings=warnings_list,
            calculation_details=calculation_details
        )
    
    def _get_lot_size(self, symbol: str, provided_lot_size: Optional[int] = None) -> int:
        """Get lot size for symbol with caching."""
        if provided_lot_size is not None:
            return provided_lot_size
        
        # Check cache
        if symbol in self._lot_size_cache:
            return self._lot_size_cache[symbol]
        
        # Check overrides
        nse_config = self.config.nse_lot_config
        if symbol in nse_config.lot_size_overrides:
            lot_size = nse_config.lot_size_overrides[symbol]
            self._lot_size_cache[symbol] = lot_size
            return lot_size
        
        # Default lot size
        lot_size = nse_config.default_lot_size
        self._lot_size_cache[symbol] = lot_size
        return lot_size
    
    def _round_to_lot_size(self, shares: float, lot_size: int) -> int:
        """Round shares to nearest lot size."""
        if lot_size <= 1:
            return int(shares)
        
        # Round to nearest lot size
        lots = round(shares / lot_size)
        return max(1, lots) * lot_size
    
    def validate_position_size(self, result: PositionSizeResult) -> Tuple[bool, List[str]]:
        """
        Validate position size result against all risk rules.
        
        Args:
            result: Position size calculation result
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []
        
        # Check basic constraints
        if result.shares <= 0:
            errors.append("Shares must be positive")
        
        if result.position_value <= 0:
            errors.append("Position value must be positive")
        
        # Check compliance flags
        if not result.lot_size_compliant:
            errors.append("Position not compliant with NSE lot sizes")
        
        if not result.liquidity_approved:
            errors.append("Position exceeds liquidity constraints")
        
        # Check minimum position size
        ps_config = self.config.position_sizing
        if result.position_value < ps_config.min_position_value:
            errors.append(f"Position value below minimum {ps_config.min_position_value}")
        
        return len(errors) == 0, errors
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        """Get statistics about position sizing operations."""
        return {
            'total_calculations': self._position_count,
            'rejected_positions': self._rejected_count,
            'acceptance_rate': (
                (self._position_count - self._rejected_count) / self._position_count 
                if self._position_count > 0 else 0
            ),
            'cache_size': len(self._lot_size_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._lot_size_cache.clear()
        self._volume_cache.clear()
        self.logger.info("Position sizer caches cleared")


# Utility functions for position sizing

def create_stock_data_from_dict(data: Dict) -> StockData:
    """
    Create StockData from dictionary (useful for API integration).
    
    Args:
        data: Dictionary containing stock data
        
    Returns:
        StockData instance
    """
    return StockData(
        symbol=data['symbol'],
        current_price=float(data['current_price']),
        atr=float(data['atr']),
        avg_daily_volume=float(data['avg_daily_volume']),
        sector=data.get('sector'),
        lot_size=data.get('lot_size'),
        volatility=data.get('volatility'),
        margin_requirement=data.get('margin_requirement')
    )


def create_portfolio_state_from_positions(
    total_capital: float,
    cash_available: float,
    positions: List[Dict]
) -> PortfolioState:
    """
    Create PortfolioState from position list.
    
    Args:
        total_capital: Total portfolio capital
        cash_available: Available cash for new positions
        positions: List of current position dictionaries
        
    Returns:
        PortfolioState instance
    """
    # Calculate sector exposures
    sector_exposures = {}
    for pos in positions:
        sector = pos.get('sector', 'UNKNOWN')
        value = pos.get('position_value', 0)
        sector_exposures[sector] = sector_exposures.get(sector, 0) + value
    
    return PortfolioState(
        total_capital=total_capital,
        cash_available=cash_available,
        current_positions=positions,
        sector_exposures=sector_exposures,
        daily_pnl=sum(pos.get('unrealized_pnl', 0) for pos in positions),
        max_drawdown=0.0  # Would be calculated from historical data
    )


def calculate_position_risk_metrics(result: PositionSizeResult) -> Dict[str, float]:
    """
    Calculate additional risk metrics for a position.
    
    Args:
        result: Position size calculation result
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {}
    
    if result.position_value > 0:
        metrics['risk_percentage'] = (result.risk_amount / result.position_value) * 100
        metrics['margin_percentage'] = (result.margin_required / result.position_value) * 100
        
        # Calculate leverage implied by margin
        if result.margin_required > 0:
            metrics['implied_leverage'] = result.position_value / result.margin_required
        else:
            metrics['implied_leverage'] = 1.0
    
    return metrics


# Export main components
__all__ = [
    'EnhancedPositionSizer',
    'PositionSizeResult',
    'PositionSizeError',
    'StockData',
    'PortfolioState',
    'create_stock_data_from_dict',
    'create_portfolio_state_from_positions',
    'calculate_position_risk_metrics',
]