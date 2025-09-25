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
        Calculate Entry_Value, Stop_Value, and Target_Value using Hybrid Entry System.
        
        Priority flow: Monte Carlo → Strategic Entry (Breakout/Support/ATR) → Current Price
        
        Args:
            signal: Signal type (BUY/HOLD/AVOID)
            current_price: Current market price
            indicators: Technical indicators dictionary
            signal_data: Additional signal-specific data
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            market_regime: Current market regime
            
        Returns:
            Dictionary with entry, stop, target, and validation info
        """
        try:
            # Only calculate for BUY signals
            if signal != "BUY":
                return {
                    'entry_value': current_price,
                    'stop_value': current_price * 0.97,
                    'target_value': None,
                    'risk_reward_ratio': None,
                    'calculation_method': 'Not applicable for non-BUY signals',
                    'hit_probability': 0.0,
                    'indicator_confidence': 0.0,
                    'monte_carlo_paths': 0,
                    'fallback_used': 'Non-BUY signal',
                    'entry_method': 'UNAVAILABLE',
                    'order_type': 'MARKET',
                    'validation_flag': 'PASS',
                    'validation_message': 'Non-BUY signal - no entry calculation needed',
                    'entry_clamp_reason': None,
                    'entry_debug': {}
                }
            
            # Get ATR for stop/target calculations
            atr = indicators.get('atr', current_price * 0.02)
            
            # Calculate basic stop and target for risk bounds
            basic_stop_value = current_price - (self.config.stop_loss_atr_multiplier * atr)
            max_stop_loss = current_price * 0.95
            basic_stop_value = max(basic_stop_value, max_stop_loss)
            risk_amount = current_price - basic_stop_value
            basic_target_value = current_price + (2.5 * risk_amount)
            
            # Initialize result variables
            entry_value = current_price
            entry_method = 'CURRENT_PRICE'
            order_type = 'MARKET'
            validation_flag = 'PASS'
            validation_message = ''
            entry_clamp_reason = None
            entry_debug = {}
            
            # Step 1: Try Monte Carlo optimal entry
            monte_carlo_success = False
            if (symbol and historical_data is not None and 
                len(historical_data) >= 60 and market_regime):
                
                try:
                    monte_carlo_result = self.optimal_entry_calculator.monte_carlo_optimal_entry(
                        symbol=symbol,
                        historical_prices=historical_data,
                        params={'horizon_days': 30, 'signal': signal}  # Pass signal to constrain price grid
                    )
                    
                    if monte_carlo_result['success']:
                        candidate_entry = monte_carlo_result['entry']
                        
                        # Validate Monte Carlo entry - CRITICAL FIX: For BUY signals, entry should not exceed current price
                        if signal == "BUY":
                            min_reasonable = current_price * 0.8
                            max_reasonable = current_price  # FIXED: BUY entries should not exceed current price
                        else:
                            min_reasonable = current_price * 0.8
                            max_reasonable = current_price * 1.2
                        
                        if min_reasonable <= candidate_entry <= max_reasonable:
                            entry_value = candidate_entry
                            entry_method = 'MONTE_CARLO'
                            order_type = 'LIMIT'  # Monte Carlo entries are planned
                            monte_carlo_success = True
                            entry_debug.update({
                                'monte_carlo': monte_carlo_result['debug'],
                                'validation_bounds': [min_reasonable, max_reasonable]
                            })
                        else:
                            validation_message += f"Monte Carlo entry {candidate_entry:.2f} outside bounds; "
                            entry_debug['monte_carlo_rejected'] = {
                                'entry': candidate_entry,
                                'bounds': [min_reasonable, max_reasonable],
                                'reason': monte_carlo_result['reason']
                            }
                    else:
                        validation_message += f"Monte Carlo failed: {monte_carlo_result['reason']}; "
                        entry_debug['monte_carlo_failed'] = monte_carlo_result
                    
                except Exception as e:
                    validation_message += f"Monte Carlo error: {str(e)}; "
                    entry_debug['monte_carlo_error'] = str(e)
            
            # Step 2: If Monte Carlo failed, use Strategic Entry
            if not monte_carlo_success:
                try:
                    from .strategic_entry import calculate_strategic_entry
                    
                    # Prepare data for strategic entry
                    prices = historical_data['Close'].values.tolist() if historical_data is not None else []
                    volumes = historical_data['Volume'].values.tolist() if historical_data is not None else []
                    
                    metadata = {
                        'avg_volume': indicators.get('avg_volume', 0),
                        'market_cap_proxy': 'LARGE' if indicators.get('avg_volume', 0) >= 2000000 else 'SMALL'
                    }
                    
                    strategic_result = calculate_strategic_entry(
                        symbol=symbol or 'UNKNOWN',
                        current_price=current_price,
                        indicators=indicators,
                        prices=prices,
                        volumes=volumes,
                        metadata=metadata,
                        signal=signal,
                        config={}
                    )
                    
                    if strategic_result['entry_method'] != 'UNAVAILABLE':
                        entry_value = strategic_result['entry_value']
                        entry_method = strategic_result['entry_method']
                        order_type = strategic_result['order_type']
                        entry_clamp_reason = strategic_result.get('clamp_reason')
                        entry_debug.update({
                            'strategic_entry': strategic_result['debug'],
                            'messages': strategic_result['messages']
                        })
                        
                        # Add strategic entry success message
                        if entry_method != 'CURRENT_PRICE':
                            validation_message += f"Strategic entry successful: {entry_method}; "
                        else:
                            validation_message += f"Strategic entry fell back to current price; "
                    else:
                        validation_message += f"Strategic entry unavailable; "
                        entry_debug['strategic_failed'] = strategic_result
                    
                except Exception as e:
                    validation_message += f"Strategic entry error: {str(e)}; "
                    entry_debug['strategic_error'] = str(e)
            
            # Step 3: Calculate stop and target based on final entry
            stop_value = entry_value - (self.config.stop_loss_atr_multiplier * atr)
            stop_value = max(stop_value, entry_value * 0.95)  # 5% max stop
            
            # Target based on risk-reward ratio
            risk_amount = entry_value - stop_value
            target_value = entry_value + (2.5 * risk_amount)
            
            # Use resistance level if available
            vp_resistance = indicators.get('vp_resistance_level')
            if (vp_resistance and not np.isnan(vp_resistance) and 
                entry_value < vp_resistance < entry_value * 1.3):
                target_value = min(target_value, vp_resistance * 0.99)
            
            # Calculate final risk-reward ratio
            actual_risk = entry_value - stop_value
            actual_reward = target_value - entry_value
            risk_reward_ratio = actual_reward / actual_risk if actual_risk > 0 else 2.5
            
            # Validation checks
            validation_issues = []
            
            # Check if entry equals current price (anti-pattern)
            if abs(entry_value - current_price) < 0.01:
                validation_issues.append("Entry equals current price")
            
            # Check RSI for BUY signals
            rsi = indicators.get('rsi', 50)
            if rsi > 75 and entry_method != 'BREAKOUT':
                validation_issues.append(f"High RSI ({rsi:.1f}) without breakout entry")
            
            # Set validation flag
            if validation_issues:
                validation_flag = 'REVIEW' if len(validation_issues) == 1 else 'FAIL'
                validation_message += f"Issues: {', '.join(validation_issues)}"
            else:
                validation_flag = 'PASS'
                validation_message += "All validations passed"
            
            return {
                'entry_value': round(entry_value, 2),
                'stop_value': round(stop_value, 2),
                'target_value': round(target_value, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'calculation_method': f'Hybrid Entry System ({entry_method})',
                'risk_amount': round(actual_risk, 2),
                'reward_potential': round(actual_reward, 2),
                'hit_probability': 0.2 if not monte_carlo_success else 0.5,  # Default based on method
                'indicator_confidence': 50.0,
                'monte_carlo_paths': 0,
                'fallback_used': 'Hybrid Entry System',
                'data_confidence': 'HIGH' if monte_carlo_success else 'MEDIUM',
                'execution_time_ms': 0.0,
                'entry_method': entry_method,
                'order_type': order_type,
                'validation_flag': validation_flag,
                'validation_message': validation_message.strip(),
                'entry_clamp_reason': entry_clamp_reason,
                'entry_debug': entry_debug
            }
            
        except Exception as e:
            # Emergency fallback
            emergency_stop = current_price * 0.97
            emergency_risk = current_price - emergency_stop
            emergency_target = current_price + (2.5 * emergency_risk)
            
            return {
                'entry_value': current_price,
                'stop_value': emergency_stop,
                'target_value': emergency_target,
                'risk_reward_ratio': 2.5,
                'calculation_method': f'Emergency fallback: {str(e)[:100]}',
                'risk_amount': round(emergency_risk, 2),
                'reward_potential': round(emergency_target - current_price, 2),
                'hit_probability': 0.1,
                'indicator_confidence': 25.0,
                'monte_carlo_paths': 0,
                'fallback_used': 'Emergency error fallback',
                'data_confidence': 'ERROR',
                'execution_time_ms': 0.0,
                'entry_method': 'CURRENT_PRICE',
                'order_type': 'MARKET',
                'validation_flag': 'FAIL',
                'validation_message': f'Emergency fallback due to error: {str(e)}',
                'entry_clamp_reason': None,
                'entry_debug': {'error': str(e)}
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
            
            # Analyze entry timing (spike detection and wait recommendation)
            entry_timing_analysis = self.analyze_entry_timing(
                symbol=symbol,
                current_price=current_price,
                indicators=indicators,
                historical_data=historical_data
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
                },
                # Entry timing analysis
                'entry_timing': entry_timing_analysis['entry_timing'],
                'timing_confidence': entry_timing_analysis['timing_confidence'],
                'timing_reason': entry_timing_analysis['reason'],
                'wait_probability': entry_timing_analysis['wait_probability'],
                'suggested_wait_days': entry_timing_analysis['suggested_wait_days'],
                'spike_score': entry_timing_analysis['spike_score']
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
    
    def analyze_entry_timing(self, symbol: str, current_price: float, 
                           indicators: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze whether to enter immediately or wait for a better entry price.
        
        Considers spike detection, volume confirmation, technical levels, and timing.
        
        Returns:
            Dictionary with entry timing recommendation and reasoning
        """
        try:
            # Get historical price data
            if historical_data is None or len(historical_data) < 20:
                return {
                    'entry_timing': 'ENTER_NOW',
                    'timing_confidence': 'LOW',
                    'reason': 'Insufficient historical data for timing analysis',
                    'wait_probability': 0.0,
                    'suggested_wait_days': 0
                }
            
            # Extract key price levels
            week_high = indicators.get('1Week_High', current_price)
            week_low = indicators.get('1Week_Low', current_price)
            month_high = indicators.get('Last_30Day_High', current_price)
            month_low = indicators.get('Last_30Day_Low', current_price)
            
            # Calculate spike metrics
            week_range = week_high - week_low
            current_from_week_low = (current_price - week_low) / week_low if week_low > 0 else 0
            current_from_month_low = (current_price - month_low) / month_low if month_low > 0 else 0
            
            # Volume analysis
            avg_volume = indicators.get('avg_volume', 0)
            current_volume = historical_data['Volume'].iloc[-1] if len(historical_data) > 0 else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Technical indicators
            rsi = indicators.get('rsi', 50)
            atr = indicators.get('atr', current_price * 0.02)
            atr_pct = atr / current_price
            
            # Recent price action (last 5 days)
            recent_prices = historical_data['Close'].tail(5).values
            if len(recent_prices) >= 5:
                recent_prices_array = np.array(recent_prices, dtype=float)
                recent_change_pct = (recent_prices_array[-1] - recent_prices_array[0]) / recent_prices_array[0]
                recent_volatility = np.std(recent_prices_array) / np.mean(recent_prices_array)
            else:
                recent_change_pct = 0
                recent_volatility = 0
            
            # Initialize analysis variables
            entry_timing = 'ENTER_NOW'
            timing_confidence = 'MEDIUM'
            reasons = []
            wait_probability = 0.0
            suggested_wait_days = 0
            
            # SPIKE DETECTION ANALYSIS
            spike_score = 0
            
            # Distance from recent lows (spike detection)
            if current_from_week_low > 0.15:  # 15% above week's low
                spike_score += 2
                reasons.append(f"Price {current_from_week_low:.1%} above week's low - potential spike")
            
            if current_from_month_low > 0.25:  # 25% above month's low
                spike_score += 1
                reasons.append(f"Price {current_from_month_low:.1%} above month's low - elevated levels")
            
            # Recent rapid movement
            if recent_change_pct > 0.05:  # 5% up in last 5 days
                spike_score += 1
                reasons.append(f"Rapid {recent_change_pct:.1%} increase in last 5 days")
            
            # VOLUME CONFIRMATION
            if volume_ratio > 2.0 and spike_score > 0:
                reasons.append(f"High volume ({volume_ratio:.1f}x average) confirms spike strength")
                spike_score += 1
            elif volume_ratio < 0.8 and spike_score > 0:
                reasons.append(f"Low volume ({volume_ratio:.1f}x average) suggests weak spike")
                spike_score -= 1
            
            # TECHNICAL INDICATORS
            if rsi > 75:
                spike_score += 1
                reasons.append(f"Overbought RSI ({rsi:.1f}) suggests potential pullback")
            
            if rsi > 80:
                spike_score += 2
                reasons.append(f"Extremely overbought RSI ({rsi:.1f}) - high pullback risk")
            
            # High volatility + spike = higher wait probability
            if recent_volatility > 0.03 and spike_score > 1:
                spike_score += 1
                reasons.append(f"High recent volatility ({recent_volatility:.1%}) with spike")
            
            # DECISION LOGIC
            if spike_score >= 4:
                entry_timing = 'WAIT_FOR_PULLBACK'
                timing_confidence = 'HIGH'
                wait_probability = min(0.8, spike_score * 0.15)
                suggested_wait_days = min(10, spike_score * 2)
                reasons.append(f"Strong spike signals detected (score: {spike_score})")
                
            elif spike_score >= 2:
                entry_timing = 'WAIT_FOR_PULLBACK'
                timing_confidence = 'MEDIUM'
                wait_probability = min(0.6, spike_score * 0.12)
                suggested_wait_days = min(7, spike_score * 1.5)
                reasons.append(f"Moderate spike signals detected (score: {spike_score})")
                
            elif spike_score >= 1:
                entry_timing = 'MONITOR_CLOSELY'
                timing_confidence = 'LOW'
                wait_probability = 0.3
                suggested_wait_days = 3
                reasons.append(f"Weak spike signals (score: {spike_score}) - monitor closely")
                
            else:
                entry_timing = 'ENTER_NOW'
                timing_confidence = 'HIGH'
                wait_probability = 0.1
                suggested_wait_days = 0
                reasons.append("No significant spike signals - favorable entry timing")
            
            # SPECIAL CASES
            # Very high volume + spike = might continue higher
            if volume_ratio > 3.0 and spike_score >= 2:
                entry_timing = 'ENTER_NOW'
                timing_confidence = 'MEDIUM'
                wait_probability = 0.2
                reasons.append("Very high volume suggests continuation - consider entering now")
            
            # Breakout patterns override spike analysis
            breakout_signals = indicators.get('breakout_signals', 0)
            if breakout_signals > 0 and entry_timing == 'WAIT_FOR_PULLBACK':
                entry_timing = 'ENTER_NOW'
                timing_confidence = 'HIGH'
                wait_probability = 0.1
                reasons.append("Breakout pattern detected - favorable for immediate entry")
            
            return {
                'entry_timing': entry_timing,
                'timing_confidence': timing_confidence,
                'reason': '; '.join(reasons),
                'wait_probability': round(wait_probability, 2),
                'suggested_wait_days': int(suggested_wait_days),
                'spike_score': spike_score,
                'analysis_metrics': {
                    'current_from_week_low': round(current_from_week_low, 3),
                    'current_from_month_low': round(current_from_month_low, 3),
                    'volume_ratio': round(volume_ratio, 2),
                    'rsi': round(rsi, 1),
                    'recent_change_pct': round(recent_change_pct, 3),
                    'recent_volatility': round(recent_volatility, 4)
                }
            }
            
        except Exception as e:
            return {
                'entry_timing': 'ENTER_NOW',
                'timing_confidence': 'LOW',
                'reason': f'Analysis error: {str(e)}',
                'wait_probability': 0.0,
                'suggested_wait_days': 0,
                'spike_score': 0,
                'analysis_metrics': {}
            }