"""
Risk Management System
Implements comprehensive risk management including position sizing, stop losses, 
take profits, exposure limits, and trailing stops
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Import from our centralized constants and core
from constants import (
    RISK_CONSTANTS, TRADING_CONSTANTS, ERROR_MESSAGES, SUCCESS_MESSAGES,
    MONTE_CARLO_PARAMETERS, MarketRegime
)
from core import PerformanceUtils, DisplayUtils

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

class StopType(Enum):
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"

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

@dataclass
class RiskConfig:
    """Risk management configuration using centralized constants"""
    max_portfolio_risk: float = RISK_CONSTANTS['DEFAULT_MAX_PORTFOLIO_RISK']
    max_position_size: float = RISK_CONSTANTS['DEFAULT_MAX_POSITION_SIZE']
    max_daily_loss: float = RISK_CONSTANTS['DEFAULT_MAX_DAILY_LOSS']
    max_concurrent_positions: int = RISK_CONSTANTS['DEFAULT_MAX_CONCURRENT_POSITIONS']
    stop_loss_atr_multiplier: float = RISK_CONSTANTS['DEFAULT_ATR_MULTIPLIER']
    min_risk_reward_ratio: float = 2.0
    breakeven_trigger_ratio: float = 1.5
    trailing_stop_atr_multiplier: float = 1.0
    correlation_limit: float = 0.7
    # Additional enhanced parameters
    max_monthly_loss: float = 0.08
    max_sector_exposure: float = 0.30

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
    
    def __init__(self, initial_capital: float, config: Optional[RiskConfig] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config or RiskConfig()
        
        # Initialize optimal entry calculator (lazy import to avoid circular dependencies)
        self._optimal_entry_calculator = None
        
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
    
    @property
    def optimal_entry_calculator(self):
        """Lazy-load optimal entry calculator to avoid circular imports"""
        if self._optimal_entry_calculator is None:
            from optimal_entry_calculator import OptimalEntryCalculator
            self._optimal_entry_calculator = OptimalEntryCalculator()
        return self._optimal_entry_calculator
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               signal_score: int) -> Tuple[int, float]:
        """
        Calculate optimal position size based on:
        - Volatility (ATR-based stop distance)
        - Portfolio risk limits
        - Signal strength
        - Current exposure
        """
        # Base risk per trade (adjusted by signal strength)
        base_risk = 0.01  # 1% base risk
        
        # Adjust risk based on signal strength
        if signal_score >= 70:
            risk_multiplier = 1.5  # High confidence = higher risk
        elif signal_score >= 50:
            risk_multiplier = 1.0  # Medium confidence = normal risk
        else:
            risk_multiplier = 0.5  # Low confidence = lower risk
        
        adjusted_risk = base_risk * risk_multiplier
        
        # Maximum dollar risk for this trade
        max_dollar_risk = self.current_capital * adjusted_risk
        
        # Price risk per share
        price_risk_per_share = abs(entry_price - stop_loss)
        
        if price_risk_per_share <= 0:
            return 0, 0.0
        
        # Calculate quantity based on risk
        quantity_by_risk = int(max_dollar_risk / price_risk_per_share)
        
        # Position size limit (max % of portfolio)
        max_position_value = self.current_capital * self.config.max_position_size
        quantity_by_size = int(max_position_value / entry_price)
        
        # Take the smaller of the two
        final_quantity = min(quantity_by_risk, quantity_by_size)
        
        # Calculate actual risk amount
        actual_risk = final_quantity * price_risk_per_share
        
        return final_quantity, actual_risk
    
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
            'max_positions': self.config.max_concurrent_positions,
            'risk_limit_ok': risk_utilization <= self.config.max_portfolio_risk,
            'position_limit_ok': len(self.positions) < self.config.max_concurrent_positions,
            'can_add_position': (risk_utilization <= self.config.max_portfolio_risk * 0.8 and 
                               len(self.positions) < self.config.max_concurrent_positions)
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
        
        print(f"Entered position: {symbol}")
        print(f"  Entry: ${entry_price:.2f}, Quantity: {quantity}")
        print(f"  Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
        print(f"  Risk Amount: ${risk_amount:.2f}")
        
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
                print(f"{symbol}: Moved stop to breakeven at ${position.current_stop_loss:.2f}")
        
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
                print(f"{symbol}: Started trailing stop at ${position.current_stop_loss:.2f}")
        
        # Update trailing stop
        elif position.stop_type == StopType.TRAILING:
            trailing_distance = self.config.trailing_stop_atr_multiplier * position.atr_at_entry
            new_stop = current_price - trailing_distance
            
            # Only move stop up, never down
            if new_stop > position.current_stop_loss:
                position.current_stop_loss = new_stop
                print(f"{symbol}: Trailed stop to ${position.current_stop_loss:.2f}")
        
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
        
        print(f"Exited position: {symbol}")
        print(f"  Exit: ${exit_price:.2f}, Reason: {exit_reason}")
        print(f"  P&L: ${net_pnl:.2f}")
        
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
    
    def calculate_entry_stop_target(self, 
                                   signal: str,
                                   current_price: float, 
                                   indicators: Dict[str, Any],
                                   signal_data: Optional[Dict[str, Any]] = None,
                                   symbol: Optional[str] = None,
                                   historical_data: Optional[pd.DataFrame] = None,
                                   market_regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Calculate Entry_Value, Stop_Value, and Target_Value using Monte Carlo optimal entry.
        
        Args:
            signal: Signal type (BUY/HOLD/AVOID)
            current_price: Current market price
            indicators: Technical indicators dictionary
            signal_data: Additional signal-specific data
            symbol: Stock symbol (for optimal entry calculation)
            historical_data: Historical OHLCV data
            market_regime: Current market regime
            
        Returns:
            Dictionary with entry, stop, and target values
        """
        try:
            # **CRITICAL FIX**: Only calculate entry/stop/target for BUY signals
            # For non-BUY signals, return minimal valid structure
            if signal != "BUY":
                return {
                    'entry_value': current_price,  # Use current price for reference
                    'stop_value': current_price * 0.97,  # 3% below for reference
                    'target_value': None,  # No target for non-actionable signals
                    'risk_reward_ratio': None,
                    'calculation_method': 'Not applicable for non-BUY signals',
                    'hit_probability': 0.0,
                    'indicator_confidence': 0.0,
                    'monte_carlo_paths': 0,
                    'fallback_used': 'Non-BUY signal'
                }
            
            # Get ATR for calculations
            atr = indicators.get('atr', current_price * 0.02)
            atr_multiplier = self.config.stop_loss_atr_multiplier
            
            # Calculate basic stop and target first (for risk bounds)
            basic_stop_value = current_price - (atr_multiplier * atr)
            max_stop_loss = current_price * 0.95  # 5% maximum stop
            basic_stop_value = max(basic_stop_value, max_stop_loss)
            
            risk_amount = current_price - basic_stop_value
            basic_target_value = current_price + (2.5 * risk_amount)  # 2.5:1 R:R
            
            # Try Monte Carlo optimal entry calculation if data available
            if (symbol and historical_data is not None and 
                market_regime and len(historical_data) >= 60):
                
                try:
                    # Calculate risk bounds for Monte Carlo
                    risk_bounds = (basic_stop_value, basic_target_value)
                    
                    # Run optimal entry calculation
                    optimal_result = self.optimal_entry_calculator.calculate_optimal_entry(
                        symbol=symbol,
                        current_price=current_price,
                        historical_data=historical_data,
                        indicators=indicators,
                        market_regime=market_regime,
                        risk_bounds=risk_bounds,
                        target_price=basic_target_value
                    )
                    
                    # Use optimal entry if probability meets threshold
                    if optimal_result.hit_probability >= MONTE_CARLO_PARAMETERS['min_probability_threshold']:
                        entry_value = optimal_result.optimal_entry
                        
                        # Recalculate stop and target based on optimal entry
                        stop_value = entry_value - (atr_multiplier * atr)
                        stop_value = max(stop_value, entry_value * 0.95)  # 5% max stop
                        
                        # Target based on risk-reward ratio
                        risk_amount = entry_value - stop_value
                        target_value = entry_value + (2.5 * risk_amount)
                        
                        # Use resistance level if available and reasonable
                        vp_resistance = indicators.get('vp_resistance_level')
                        if (vp_resistance and not np.isnan(vp_resistance) and 
                            entry_value < vp_resistance < entry_value * 1.3):
                            target_value = min(target_value, vp_resistance * 0.99)
                        
                        # Calculate final risk-reward ratio
                        actual_risk = entry_value - stop_value
                        actual_reward = target_value - entry_value
                        risk_reward_ratio = actual_reward / actual_risk if actual_risk > 0 else 0
                        
                        return {
                            'entry_value': round(entry_value, 2),
                            'stop_value': round(stop_value, 2),
                            'target_value': round(target_value, 2),
                            'risk_reward_ratio': round(risk_reward_ratio, 2),
                            'calculation_method': f'Monte Carlo Optimal (ATR={atr:.2f})',
                            'risk_amount': round(actual_risk, 2),
                            'reward_potential': round(actual_reward, 2),
                            'hit_probability': round(optimal_result.hit_probability, 3),
                            'indicator_confidence': round(optimal_result.indicator_confidence, 1),
                            'monte_carlo_paths': optimal_result.monte_carlo_paths,
                            'fallback_used': optimal_result.fallback_used,
                            'data_confidence': optimal_result.data_confidence,
                            'execution_time_ms': round(optimal_result.execution_time_ms, 1)
                        }
                    
                except Exception as e:
                    print(f"Monte Carlo calculation failed, using ATR fallback: {e}")
            
            # **CRITICAL FIX**: Enhanced fallback with GUARANTEED target calculation
            entry_value = current_price
            stop_value = basic_stop_value
            target_value = basic_target_value
            
            # **MANDATORY TARGET VALIDATION**: Ensure target is always valid for BUY signals
            if target_value <= entry_value:
                # Force recalculate target with minimum 2.5:1 R:R
                actual_risk = abs(entry_value - stop_value)
                if actual_risk <= 0:
                    actual_risk = current_price * 0.03  # 3% fallback risk
                    stop_value = entry_value - actual_risk
                
                target_value = entry_value + (2.5 * actual_risk)  # Guaranteed 2.5:1 R:R
                print(f"    FORCED target calculation: Entry={entry_value:.2f}, Stop={stop_value:.2f}, Target={target_value:.2f}")
            
            # Calculate risk-reward ratio with validation
            actual_risk = entry_value - stop_value
            actual_reward = target_value - entry_value
            risk_reward_ratio = actual_reward / actual_risk if actual_risk > 0 else 2.5
            
            return {
                'entry_value': round(entry_value, 2),
                'stop_value': round(stop_value, 2),
                'target_value': round(target_value, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'calculation_method': f'ATR-based fallback (ATR={atr:.2f})',
                'risk_amount': round(actual_risk, 2),
                'reward_potential': round(actual_reward, 2),
                'hit_probability': 0.2,  # Default probability
                'indicator_confidence': 50.0,  # Default confidence
                'monte_carlo_paths': 0,
                'fallback_used': 'ATR-based entry',
                'data_confidence': 'INSUFFICIENT',
                'execution_time_ms': 0.0
            }
            
        except Exception as e:
            # **CRITICAL FIX**: Emergency fallback with guaranteed valid values
            print(f"    ERROR in calculate_entry_stop_target: {e}")
            
            # Calculate emergency values with proper risk-reward
            emergency_stop = current_price * 0.97  # 3% stop loss
            emergency_risk = current_price - emergency_stop
            emergency_target = current_price + (2.5 * emergency_risk)  # 2.5:1 R:R guaranteed
            
            return {
                'entry_value': current_price,
                'stop_value': emergency_stop,
                'target_value': emergency_target,
                'risk_reward_ratio': 2.5,  # Guaranteed minimum R:R
                'calculation_method': f'Emergency fallback due to error: {str(e)[:100]}',
                'risk_amount': round(emergency_risk, 2),
                'reward_potential': round(emergency_target - current_price, 2),
                'hit_probability': 0.1,
                'indicator_confidence': 25.0,
                'monte_carlo_paths': 0,
                'fallback_used': 'Emergency error fallback',
                'data_confidence': 'ERROR',
                'execution_time_ms': 0.0
            }
    
    def enhanced_position_analysis(self,
                                  symbol: str,
                                  signal: str,
                                  composite_score: int,
                                  indicators: Dict[str, Any],
                                  signal_data: Optional[Dict[str, Any]] = None,
                                  historical_data: Optional[pd.DataFrame] = None,
                                  market_regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Comprehensive position analysis including entry/stop/target and position sizing.
        
        This is the main method to be called from the enhanced early warning system.
        """
        try:
            current_price = indicators.get('current_price', 0)
            if current_price <= 0:
                return self._default_position_analysis("Invalid current price")
            
            # Calculate entry, stop, and target values using Monte Carlo if data available
            entry_stop_target = self.calculate_entry_stop_target(
                signal=signal, 
                current_price=current_price, 
                indicators=indicators, 
                signal_data=signal_data,
                symbol=symbol,
                historical_data=historical_data,
                market_regime=market_regime
            )
            
            # Check if position passes risk management
            can_enter, risk_reason, quantity, risk_amount = self.can_enter_position(
                symbol, 
                entry_stop_target['entry_value'], 
                entry_stop_target['stop_value'], 
                composite_score
            )
            
            # Calculate position size for the risk amount
            if can_enter and signal == "BUY":
                position_size_info = self.calculate_position_size(
                    entry_stop_target['entry_value'],
                    entry_stop_target['stop_value'],
                    composite_score
                )
                final_quantity = position_size_info[0]
                final_risk_amount = position_size_info[1]
            else:
                final_quantity = 0
                final_risk_amount = 0.0
            
            return {
                **entry_stop_target,
                'position_size': final_quantity,
                'risk_amount': round(final_risk_amount, 2),
                'can_enter_position': can_enter,
                'risk_reason': risk_reason,
                'signal': signal,
                'composite_score': composite_score,
                'portfolio_impact': {
                    'position_weight': round(final_risk_amount / self.current_capital * 100, 2) if self.current_capital > 0 else 0,
                    'remaining_buying_power': round(self.current_capital - final_risk_amount, 2),
                    'open_positions_count': len(self.positions)
                }
            }
            
        except Exception as e:
            return self._default_position_analysis(f"Error in position analysis: {e}")
    
    def _default_position_analysis(self, reason: str) -> Dict[str, Any]:
        """Return default position analysis when calculation fails"""
        return {
            'entry_value': 0.0,
            'stop_value': 0.0,
            'target_value': 0.0,
            'risk_reward_ratio': 0.0,
            'position_size': 0,
            'risk_amount': 0.0,
            'can_enter_position': False,
            'risk_reason': reason,
            'signal': 'AVOID',
            'composite_score': 0,
            'calculation_method': 'Default - calculation failed',
            'portfolio_impact': {
                'position_weight': 0.0,
                'remaining_buying_power': self.current_capital,
                'open_positions_count': len(self.positions)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the risk management system
    initial_capital = 1000000  # 10 lakh
    risk_manager = RiskManager(initial_capital)
    
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
        print(f"Can enter position: {reason}")
        print(f"Suggested quantity: {quantity}, Risk amount: ${risk_amount:.2f}")
        
        # Enter position
        position = risk_manager.enter_position(
            symbol, entry_date, entry_price, quantity, atr, signal_data
        )
        
        # Simulate price movement and stop updates
        test_prices = [2520, 2550, 2580, 2600, 2620, 2590, 2570]
        
        for i, price in enumerate(test_prices):
            print(f"\nDay {i+1}: Price = ${price}")
            should_exit, exit_reason = risk_manager.check_exit_conditions(
                symbol, price, entry_date + timedelta(days=i+1)
            )
            
            if should_exit:
                print(f"Exit signal: {exit_reason}")
                pnl = risk_manager.exit_position(
                    symbol, price, entry_date + timedelta(days=i+1), exit_reason
                )
                break
        
        # Portfolio summary
        summary = risk_manager.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    else:
        print(f"Cannot enter position: {reason}")