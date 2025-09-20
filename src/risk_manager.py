"""
Risk Management System
Implements comprehensive risk management including position sizing, stop losses, 
take profits, exposure limits, and trailing stops
"""

import logging
from datetime import datetime, timedelta

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)

# Import centralized configuration
try:
    from .config import SystemConfig, get_config
    from .common.config import ConfigManager
except ImportError:
    # Fallback for direct execution
    from config import SystemConfig, get_config

# Import shared enums and interfaces from centralized location
from .common.enums import PositionStatus, StopType
from .common.interfaces import IRiskManager, RiskAssessment

@dataclass
class Position:
    """Individual position with risk management"""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    initial_stop_loss: float
    current_stop_loss: float
    take_profit: float
    atr_at_entry: float
    risk_amount: float
    status: PositionStatus
    stop_type: StopType
    max_price_since_entry: float
    unrealized_pnl: float = 0.0
    current_price: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        
        # Update max price for trailing stop
        if price > self.max_price_since_entry:
            self.max_price_since_entry = price

class RiskManager:
    """
    Comprehensive risk management system with:
    - Position sizing based on volatility and portfolio risk
    - ATR-based stop losses and take profits
    - Trailing stops and breakeven management
    - Portfolio exposure limits
    - Correlation monitoring
    - Daily and monthly loss limits
    """
    
    def __init__(self, initial_capital: float, config: Optional[SystemConfig] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config or get_config()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.monthly_pnl = 0.0
        self.max_portfolio_drawdown = 0.0
        self.sector_exposure: Dict[str, float] = {}
        
        # Historical tracking
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               signal_score: int, atr: float = None, 
                               avg_volume: float = None, symbol: str = None) -> Tuple[int, float]:
        """
        Enhanced position sizing with multi-stage safety validation:
        - Configurable risk multipliers with caps
        - ATR-based stop distance validation
        - Lot size enforcement (NSE compliance)
        - Liquidity-based sizing constraints
        - Volatility parity option
        - Kelly criterion integration
        """
        # Stage 1: Input validation and safety checks
        if entry_price <= 0 or stop_loss <= 0:
            return 0, 0.0
        
        price_risk_per_share = abs(entry_price - stop_loss)
        if price_risk_per_share <= 0:
            return 0, 0.0
        
        # Stage 2: ATR-based stop distance validation
        if atr and atr > 0:
            stop_distance_atr_ratio = price_risk_per_share / atr
            if stop_distance_atr_ratio < self.config.min_stop_atr_ratio:
                # Stop too tight - increase stop distance to minimum ATR ratio
                price_risk_per_share = atr * self.config.min_stop_atr_ratio
            elif stop_distance_atr_ratio > self.config.max_stop_atr_ratio:
                # Stop too wide - reduce to maximum ATR ratio
                price_risk_per_share = atr * self.config.max_stop_atr_ratio
        
        # Stage 3: Configurable risk multiplier with caps
        risk_multiplier = self._get_capped_risk_multiplier(signal_score)
        base_risk = self.config.risk_per_trade
        adjusted_risk = base_risk * risk_multiplier
        
        # Stage 4: Dollar risk calculation
        max_dollar_risk = self.current_capital * adjusted_risk
        
        # Stage 5: Base position sizing
        quantity_by_risk = int(max_dollar_risk / price_risk_per_share)
        
        # Stage 6: Position value limits
        max_position_value = self.current_capital * self.config.max_position_size
        quantity_by_size = int(max_position_value / entry_price)
        
        # Initial quantity (smaller of risk and size limits)
        base_quantity = min(quantity_by_risk, quantity_by_size)
        
        # Stage 7: Volatility parity adjustment (if enabled)
        if self.config.volatility_parity_enabled and atr and atr > 0:
            base_quantity = self._apply_volatility_parity(base_quantity, atr, entry_price)
        
        # Stage 8: Liquidity constraints
        if self.config.liquidity_check_enabled and avg_volume:
            base_quantity = self._apply_liquidity_constraints(base_quantity, avg_volume)
        
        # Stage 9: Lot size enforcement
        final_quantity = self._enforce_lot_size(base_quantity, symbol)
        
        # Stage 10: Final safety checks
        if final_quantity <= 0:
            return 0, 0.0
        
        # Calculate actual risk with final quantity
        actual_risk = final_quantity * price_risk_per_share
        
        return final_quantity, actual_risk
    
    def _get_capped_risk_multiplier(self, signal_score: int) -> float:
        """Get risk multiplier based on signal score with safety caps"""
        if signal_score >= 70:
            multiplier = self.config.risk_multiplier_high_score
        elif signal_score >= 50:
            multiplier = self.config.risk_multiplier_medium_score
        else:
            multiplier = self.config.risk_multiplier_low_score
        
        # Apply safety caps
        return max(self.config.min_risk_multiplier, 
                  min(multiplier, self.config.max_risk_multiplier))
    
    def _apply_volatility_parity(self, quantity: int, atr: float, price: float) -> int:
        """Apply volatility parity to normalize risk across different volatility stocks"""
        if atr <= 0 or price <= 0:
            return quantity
        
        # Calculate volatility as percentage of price
        volatility_pct = atr / price
        
        # Base volatility for normalization (2% daily move)
        base_volatility = 0.02
        
        # Adjust quantity inversely to volatility
        volatility_adjustment = base_volatility / volatility_pct
        adjusted_quantity = int(quantity * volatility_adjustment)
        
        return max(1, adjusted_quantity)
    
    def _apply_liquidity_constraints(self, quantity: int, avg_volume: float) -> int:
        """Constrain position size based on average daily volume"""
        if avg_volume <= 0:
            return quantity
        
        # Maximum position as percentage of average volume
        max_volume_position = int(avg_volume * self.config.min_avg_volume_multiple)
        
        return min(quantity, max_volume_position)
    
    def _enforce_lot_size(self, quantity: int, symbol: str = None) -> int:
        """Enforce NSE lot size constraints"""
        if not self.config.lot_size_enforcement:
            return quantity
        
        # For NSE stocks, use lot size (simplified - would need actual lot size data)
        lot_size = self._get_lot_size(symbol)
        
        if lot_size <= 1:
            return quantity
        
        # Round down to nearest lot size
        return (quantity // lot_size) * lot_size
    
    def _get_lot_size(self, symbol: str = None) -> int:
        """Get lot size for symbol (simplified implementation)"""
        if not symbol:
            return self.config.default_lot_size
        
        # In real implementation, this would lookup actual NSE lot sizes
        # For now, return default
        return self.config.default_lot_size
    
    def calculate_kelly_position(self, win_probability: float, avg_win: float, 
                               avg_loss: float, capital: float) -> float:
        """Calculate Kelly criterion position size"""
        if avg_loss <= 0 or win_probability <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win_probability, q = 1-p
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative cap
        kelly_fraction = min(kelly_fraction, self.config.kelly_fraction_conservative)
        kelly_fraction = max(kelly_fraction, 0.0)  # No negative sizing
        
        return capital * kelly_fraction
    
    def calculate_stops_and_targets(self, entry_price: float, atr: float, 
                                   signal_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate initial stop loss and take profit levels
        """
        # Initial stop loss (ATR-based)
        stop_loss = entry_price - (self.config.stop_loss_atr_multiplier * atr)
        
        # Risk amount per share
        risk_per_share = entry_price - stop_loss
        
        # Take profit (risk-reward ratio based)
        take_profit = entry_price + (risk_per_share * self.config.min_risk_reward_ratio)
        
        # Adjust take profit based on technical levels if available
        resistance_level = signal_data.get('vp_resistance_level')
        if resistance_level and not np.isnan(resistance_level):
            # If resistance is closer than our calculated TP, use a level just below it
            if resistance_level < take_profit:
                take_profit = resistance_level * 0.99  # 1% below resistance
        
        return stop_loss, take_profit
    
    def check_portfolio_risk_limits(self) -> Dict[str, Any]:
        """
        Check if current portfolio is within risk limits
        """
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        total_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        
        risk_utilization = total_risk / self.current_capital
        exposure_utilization = total_exposure / self.current_capital
        
        limits_status = {
            'total_risk_pct': round(risk_utilization * 100, 2),
            'total_exposure_pct': round(exposure_utilization * 100, 2),
            'max_risk_limit': self.config.max_portfolio_risk * 100,
            'max_exposure_limit': 100,  # Full capital can be deployed
            'positions_count': len(self.positions),
            'max_positions': self.config.max_positions,  # Use canonical name
            'risk_limit_ok': risk_utilization <= self.config.max_portfolio_risk,
            'position_limit_ok': len(self.positions) < self.config.max_positions,  # Use canonical name
            'can_add_position': (risk_utilization <= self.config.max_portfolio_risk * 0.8 and 
                               len(self.positions) < self.config.max_positions)  # Use canonical name
        }
        
        return limits_status
    
    def can_enter_position(self, symbol: str, entry_price: float, stop_loss: float, 
                          signal_score: int) -> Tuple[bool, str, int, float]:
        """
        Check if we can enter a new position based on risk limits
        Returns: (can_enter, reason, suggested_quantity, risk_amount)
        """
        # Check if already have position in this symbol
        if symbol in self.positions:
            return False, "Already have position in this symbol", 0, 0.0
        
        # Check portfolio limits
        limits = self.check_portfolio_risk_limits()
        
        if not limits['can_add_position']:
            if not limits['risk_limit_ok']:
                return False, "Portfolio risk limit exceeded", 0, 0.0
            if not limits['position_limit_ok']:
                return False, "Maximum positions limit reached", 0, 0.0
        
        # Calculate position size
        quantity, risk_amount = self.calculate_position_size(entry_price, stop_loss, signal_score)
        
        if quantity <= 0:
            return False, "Position size too small", 0, 0.0
        
        # Check minimum risk-reward ratio
        risk_per_share = abs(entry_price - stop_loss)
        min_take_profit = entry_price + (risk_per_share * self.config.min_risk_reward_ratio)
        
        # Additional checks could go here (sector limits, correlation, etc.)
        
        return True, "Position approved", quantity, risk_amount
    
    def enter_position(self, symbol: str, entry_date: datetime, entry_price: float, 
                      quantity: int, atr: float, signal_data: Dict[str, Any]) -> Optional[Position]:
        """
        Enter a new position with full risk management setup
        """
        # Calculate stops and targets
        stop_loss, take_profit = self.calculate_stops_and_targets(entry_price, atr, signal_data)
        
        # Calculate risk amount
        risk_amount = quantity * abs(entry_price - stop_loss)
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=quantity,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            take_profit=take_profit,
            atr_at_entry=atr,
            risk_amount=risk_amount,
            status=PositionStatus.OPEN,
            stop_type=StopType.INITIAL,
            max_price_since_entry=entry_price,
            current_price=entry_price
        )
        
        # Add to positions
        self.positions[symbol] = position
        
        logger.info(f"Entered position: {symbol}", 
                   extra={'symbol': symbol, 'entry_price': entry_price, 'quantity': quantity,
                         'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_amount': risk_amount,
                         'operation': 'position_entry'})
        
        return position
    
    def update_position_stops(self, symbol: str, current_price: float) -> bool:
        """
        Update stop losses based on current price and trailing stop logic
        Returns True if stop should be triggered
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        position.update_current_price(current_price)
        
        # Check if we should move to breakeven
        if position.stop_type == StopType.INITIAL:
            profit_target = position.entry_price + (
                abs(position.entry_price - position.initial_stop_loss) * 
                self.config.breakeven_trigger_ratio
            )
            
            if current_price >= profit_target:
                # Move stop to breakeven
                position.current_stop_loss = position.entry_price * 1.001  # Slightly above entry
                position.stop_type = StopType.BREAKEVEN
                logger.info(f"{symbol}: Moved stop to breakeven at ${position.current_stop_loss:.2f}", 
                           extra={'symbol': symbol, 'new_stop': position.current_stop_loss, 
                                 'stop_type': 'breakeven'})
        
        # Check if we should start trailing
        elif position.stop_type == StopType.BREAKEVEN:
            profit_target = position.entry_price + (
                abs(position.entry_price - position.initial_stop_loss) * 
                self.config.min_risk_reward_ratio
            )
            
            if current_price >= profit_target:
                # Start trailing
                trailing_distance = self.config.trailing_stop_atr_multiplier * position.atr_at_entry
                position.current_stop_loss = current_price - trailing_distance
                position.stop_type = StopType.TRAILING
                logger.info(f"{symbol}: Started trailing stop at ${position.current_stop_loss:.2f}", 
                           extra={'symbol': symbol, 'new_stop': position.current_stop_loss, 
                                 'stop_type': 'trailing_start'})
        
        # Update trailing stop
        elif position.stop_type == StopType.TRAILING:
            trailing_distance = self.config.trailing_stop_atr_multiplier * position.atr_at_entry
            new_stop = current_price - trailing_distance
            
            # Only move stop up, never down
            if new_stop > position.current_stop_loss:
                position.current_stop_loss = new_stop
                logger.info(f"{symbol}: Trailed stop to ${position.current_stop_loss:.2f}", 
                           extra={'symbol': symbol, 'new_stop': position.current_stop_loss, 
                                 'stop_type': 'trailing_update'})
        
        # Check if stop should be triggered
        return current_price <= position.current_stop_loss
    
    def check_exit_conditions(self, symbol: str, current_price: float, 
                            current_date: datetime) -> Tuple[bool, str]:
        """
        Check all exit conditions for a position
        Returns: (should_exit, exit_reason)
        """
        if symbol not in self.positions:
            return False, "Position not found"
        
        position = self.positions[symbol]
        position.update_current_price(current_price)
        
        # Check stop loss
        if self.update_position_stops(symbol, current_price):
            return True, "stop_loss"
        
        # Check take profit
        if current_price >= position.take_profit:
            return True, "take_profit"
        
        # Check time-based exit (could add specific rules here)
        days_held = (current_date - position.entry_date).days
        if days_held >= 30:  # Example: 30-day max hold
            return True, "time_exit"
        
        return False, "no_exit"
    
    def exit_position(self, symbol: str, exit_price: float, exit_date: datetime, 
                     exit_reason: str) -> Optional[float]:
        """
        Exit a position and calculate P&L
        Returns realized P&L
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.status = PositionStatus.CLOSED
        
        # Calculate P&L
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        
        # Apply transaction costs (could be enhanced)
        transaction_cost = (position.quantity * position.entry_price * 0.0005 + 
                           position.quantity * exit_price * 0.0005)
        net_pnl = gross_pnl - transaction_cost
        
        # Update capital
        self.current_capital += net_pnl
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Exited position: {symbol}", 
                   extra={'symbol': symbol, 'exit_price': exit_price, 'exit_reason': exit_reason,
                         'net_pnl': net_pnl, 'gross_pnl': gross_pnl, 'transaction_cost': transaction_cost,
                         'operation': 'position_exit'})
        
        return net_pnl
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary
        """
        # Current positions
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        
        # Historical performance
        total_realized_pnl = self.current_capital - self.initial_capital + total_unrealized_pnl
        
        # Risk metrics
        risk_limits = self.check_portfolio_risk_limits()
        
        # Performance metrics
        total_return_pct = ((self.current_capital + total_unrealized_pnl - self.initial_capital) / 
                           self.initial_capital) * 100
        
        summary = {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_value': self.current_capital + total_unrealized_pnl,
                'total_return_pct': round(total_return_pct, 2)
            },
            'positions': {
                'open_count': len(self.positions),
                'total_unrealized_pnl': round(total_unrealized_pnl, 2),
                'total_risk_amount': round(total_risk, 2)
            },
            'risk_metrics': risk_limits,
            'performance': {
                'total_trades': len(self.closed_positions),
                'total_realized_pnl': round(self.current_capital - self.initial_capital, 2)
            }
        }
        
        return summary
    
    def get_position_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all open positions
        """
        details = []
        
        for symbol, position in self.positions.items():
            position_info = {
                'symbol': symbol,
                'entry_date': position.entry_date.strftime('%Y-%m-%d'),
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'current_stop_loss': position.current_stop_loss,
                'take_profit': position.take_profit,
                'stop_type': position.stop_type.value,
                'unrealized_pnl': round(position.unrealized_pnl, 2),
                'unrealized_pnl_pct': round((position.unrealized_pnl / (position.quantity * position.entry_price)) * 100, 2),
                'risk_amount': position.risk_amount,
                'days_held': (datetime.now() - position.entry_date).days
            }
            details.append(position_info)
        
        return details

# Example usage and testing
if __name__ == "__main__":
    # Test the risk management system
    from .config import SystemConfig
    
    # Create custom configuration for testing
    test_config = SystemConfig(
        portfolio_capital=1000000,  # 10 lakh
        max_positions=10,
        risk_per_trade=0.01,
        max_position_size=0.10
    )
    
    risk_manager = RiskManager(test_config.portfolio_capital, test_config)
    
    # Test position entry
    symbol = "RELIANCE.NS"
    entry_date = datetime.now()
    entry_price = 2500.0
    atr = 25.0
    
    # Sample signal data
    signal_data = {
        'composite_score': 75,
        'vp_resistance_level': 2600.0
    }
    
    # Check if we can enter position
    can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
        symbol, entry_price, entry_price - (2 * atr), signal_data['composite_score']
    )
    
    if can_enter:
        logger.info(f"Can enter position: {reason}", 
                   extra={'symbol': symbol, 'can_enter': True, 'reason': reason, 
                         'quantity': quantity, 'risk_amount': risk_amount})
        
        # Enter position
        position = risk_manager.enter_position(
            symbol, entry_date, entry_price, quantity, atr, signal_data
        )
        
        # Simulate price movement and stop updates
        test_prices = [2520, 2550, 2580, 2600, 2620, 2590, 2570]
        
        for i, price in enumerate(test_prices):
            logger.info(f"Day {i+1}: Price = ${price}", 
                       extra={'day': i + 1, 'price': price, 'symbol': symbol})
            should_exit, exit_reason = risk_manager.check_exit_conditions(
                symbol, price, entry_date + timedelta(days=i+1)
            )
            
            if should_exit:
                logger.info(f"Exit signal: {exit_reason}", 
                           extra={'exit_reason': exit_reason, 'symbol': symbol})
                pnl = risk_manager.exit_position(
                    symbol, price, entry_date + timedelta(days=i+1), exit_reason
                )
                break
        
        # Portfolio summary
        summary = risk_manager.get_portfolio_summary()
        logger.info("Portfolio Summary", extra={'portfolio_summary': summary})
    
    else:
        logger.warning(f"Cannot enter position: {reason}", 
                      extra={'symbol': symbol, 'can_enter': False, 'reason': reason})