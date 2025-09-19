"""
Robust Backtesting Framework
Implements walk-forward backtesting with transaction costs, slippage, and comprehensive performance metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeResult:
    """Individual trade result"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    signal_score: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'time_exit', 'manual'
    days_held: int
    gross_return_pct: float
    net_return_pct: float  # After transaction costs
    dollar_profit: float
    risk_amount: float

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 1000000  # 10 lakh
    risk_per_trade: float = 0.01  # 1% of portfolio per trade
    transaction_cost: float = 0.0005  # 0.05% per side (0.1% total)
    slippage: float = 0.0005  # 0.05% slippage
    stop_loss_atr_multiplier: float = 2.0  # 2x ATR for stop loss
    take_profit_multiplier: float = 2.5  # 2.5x risk for take profit
    max_holding_days: int = 30  # Maximum days to hold a position
    min_score_threshold: int = 45  # Minimum composite score to enter trade
    max_positions: int = 10  # Maximum concurrent positions
    walk_forward_window: int = 252  # 1 year training window
    test_window: int = 63  # 3 months test window

class AdvancedBacktester:
    """
    Advanced backtesting system with:
    - Walk-forward analysis
    - Transaction costs and slippage
    - Position sizing based on volatility
    - Multiple exit strategies
    - Comprehensive performance metrics
    - Risk management
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[TradeResult] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              portfolio_value: float) -> int:
        """
        Calculate position size based on volatility and risk management
        """
        # Dollar risk per trade
        dollar_risk = portfolio_value * self.config.risk_per_trade
        
        # Price risk per share
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk <= 0:
            return 0
        
        # Quantity calculation
        quantity = int(dollar_risk / price_risk)
        
        # Ensure we don't risk more than intended
        max_position_value = portfolio_value * 0.1  # Max 10% per position
        max_quantity = int(max_position_value / entry_price)
        
        return min(quantity, max_quantity)
    
    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """Calculate stop loss based on ATR"""
        return entry_price - (self.config.stop_loss_atr_multiplier * atr)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        """Calculate take profit based on risk-reward ratio"""
        risk = entry_price - stop_loss
        return entry_price + (risk * self.config.take_profit_multiplier)
    
    def apply_transaction_costs(self, trade_value: float) -> float:
        """Apply transaction costs and slippage"""
        costs = trade_value * (self.config.transaction_cost + self.config.slippage)
        return costs
    
    def simulate_single_trade(self, symbol: str, entry_date: datetime, 
                            signal_data: Dict[str, Any], portfolio_value: float) -> Optional[TradeResult]:
        """
        Simulate a single trade from entry to exit
        """
        try:
            # Fetch data for the trade period
            start_date = entry_date - timedelta(days=5)  # Buffer for stop loss calculation
            end_date = entry_date + timedelta(days=self.config.max_holding_days + 10)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty or len(data) < 10:
                return None
            
            # Find entry date in data
            entry_idx = None
            for i, date in enumerate(data.index):
                if date.date() >= entry_date.date():
                    entry_idx = i
                    break
            
            if entry_idx is None or entry_idx >= len(data) - 1:
                return None
            
            # Entry details
            entry_price = data['Open'].iloc[entry_idx + 1]  # Next day open
            atr = signal_data.get('atr', entry_price * 0.02)  # Default 2% if no ATR
            
            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(entry_price, atr)
            take_profit = self.calculate_take_profit(entry_price, stop_loss)
            
            # Position sizing
            quantity = self.calculate_position_size(entry_price, stop_loss, portfolio_value)
            
            if quantity <= 0:
                return None
            
            # Track trade through time
            exit_date = None
            exit_price = None
            exit_reason = 'time_exit'
            
            for i in range(entry_idx + 1, min(entry_idx + self.config.max_holding_days + 1, len(data))):
                current_date = data.index[i]
                low = data['Low'].iloc[i]
                high = data['High'].iloc[i]
                close = data['Close'].iloc[i]
                
                # Check stop loss
                if low <= stop_loss:
                    exit_date = current_date
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                
                # Check take profit
                if high >= take_profit:
                    exit_date = current_date
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
                
                # Last day - exit at close
                if i == min(entry_idx + self.config.max_holding_days, len(data) - 1):
                    exit_date = current_date
                    exit_price = close
                    exit_reason = 'time_exit'
            
            if exit_date is None:
                # Exit at last available price
                exit_date = data.index[-1]
                exit_price = data['Close'].iloc[-1]
                exit_reason = 'time_exit'
            
            # Calculate returns
            gross_return_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Apply transaction costs
            entry_value = quantity * entry_price
            exit_value = quantity * exit_price
            entry_costs = self.apply_transaction_costs(entry_value)
            exit_costs = self.apply_transaction_costs(exit_value)
            total_costs = entry_costs + exit_costs
            
            net_profit = (exit_value - entry_value) - total_costs
            net_return_pct = (net_profit / entry_value) * 100
            
            # Create trade result
            trade_result = TradeResult(
                symbol=symbol,
                entry_date=data.index[entry_idx + 1],
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                signal_score=signal_data.get('composite_score', 0),
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_reason=exit_reason,
                days_held=(exit_date - data.index[entry_idx + 1]).days,
                gross_return_pct=gross_return_pct,
                net_return_pct=net_return_pct,
                dollar_profit=net_profit,
                risk_amount=entry_value * self.config.risk_per_trade
            )
            
            return trade_result
            
        except Exception as e:
            print(f"Error simulating trade for {symbol}: {e}")
            return None
    
    def walk_forward_backtest(self, signals_data: Dict[str, List[Dict]], 
                            start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting
        signals_data: {symbol: [{'date': datetime, 'signal_data': dict}, ...]}
        """
        print("Starting walk-forward backtesting...")
        
        # Initialize
        current_date = start_date
        portfolio_value = self.config.initial_capital
        active_positions = {}  # {symbol: TradeResult}
        all_trades = []
        portfolio_history = [(current_date, portfolio_value)]
        
        # Walk forward through time
        while current_date <= end_date:
            print(f"Processing date: {current_date.strftime('%Y-%m-%d')}, Portfolio: ${portfolio_value:,.0f}")
            
            # Check exit conditions for active positions
            positions_to_close = []
            for symbol, trade in active_positions.items():
                # Simulate the trade continuation
                try:
                    ticker = yf.Ticker(symbol)
                    recent_data = ticker.history(start=trade.entry_date, end=current_date + timedelta(days=1))
                    
                    if not recent_data.empty and len(recent_data) > 0:
                        current_price = recent_data['Close'].iloc[-1]
                        
                        # Check if we should exit
                        days_held = (current_date - trade.entry_date).days
                        
                        if (current_price <= trade.stop_loss or 
                            current_price >= trade.take_profit or 
                            days_held >= self.config.max_holding_days):
                            
                            # Update trade with exit info
                            trade.exit_date = current_date
                            trade.exit_price = current_price
                            trade.days_held = days_held
                            
                            if current_price <= trade.stop_loss:
                                trade.exit_reason = 'stop_loss'
                            elif current_price >= trade.take_profit:
                                trade.exit_reason = 'take_profit'
                            else:
                                trade.exit_reason = 'time_exit'
                            
                            # Recalculate returns
                            gross_return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                            entry_value = trade.quantity * trade.entry_price
                            exit_value = trade.quantity * trade.exit_price
                            entry_costs = self.apply_transaction_costs(entry_value)
                            exit_costs = self.apply_transaction_costs(exit_value)
                            total_costs = entry_costs + exit_costs
                            net_profit = (exit_value - entry_value) - total_costs
                            net_return_pct = (net_profit / entry_value) * 100
                            
                            trade.gross_return_pct = gross_return_pct
                            trade.net_return_pct = net_return_pct
                            trade.dollar_profit = net_profit
                            
                            # Update portfolio
                            portfolio_value += net_profit
                            
                            positions_to_close.append(symbol)
                            all_trades.append(trade)
                            
                except Exception as e:
                    print(f"Error checking position {symbol}: {e}")
            
            # Close positions
            for symbol in positions_to_close:
                del active_positions[symbol]
            
            # Look for new signals if we have capacity
            if len(active_positions) < self.config.max_positions:
                # Check for signals on current date
                for symbol, signal_list in signals_data.items():
                    if symbol in active_positions:
                        continue  # Already have position
                    
                    # Find signals for current date
                    for signal in signal_list:
                        signal_date = signal.get('date')
                        if (signal_date and 
                            abs((signal_date - current_date).days) <= 1 and
                            signal.get('signal_data', {}).get('composite_score', 0) >= self.config.min_score_threshold):
                            
                            # Try to enter trade
                            trade_result = self.simulate_single_trade(
                                symbol, current_date, signal['signal_data'], portfolio_value
                            )
                            
                            if trade_result:
                                active_positions[symbol] = trade_result
                                print(f"Entered position: {symbol} at ${trade_result.entry_price:.2f}")
                                break  # Only one signal per symbol per day
            
            # Update portfolio history
            portfolio_history.append((current_date, portfolio_value))
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close any remaining positions at end date
        for symbol, trade in active_positions.items():
            try:
                ticker = yf.Ticker(symbol)
                final_data = ticker.history(start=trade.entry_date, end=end_date + timedelta(days=1))
                if not final_data.empty:
                    final_price = final_data['Close'].iloc[-1]
                    
                    trade.exit_date = end_date
                    trade.exit_price = final_price
                    trade.exit_reason = 'end_of_backtest'
                    trade.days_held = (end_date - trade.entry_date).days
                    
                    # Recalculate returns
                    gross_return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    entry_value = trade.quantity * trade.entry_price
                    exit_value = trade.quantity * trade.exit_price
                    entry_costs = self.apply_transaction_costs(entry_value)
                    exit_costs = self.apply_transaction_costs(exit_value)
                    total_costs = entry_costs + exit_costs
                    net_profit = (exit_value - entry_value) - total_costs
                    net_return_pct = (net_profit / entry_value) * 100
                    
                    trade.gross_return_pct = gross_return_pct
                    trade.net_return_pct = net_return_pct
                    trade.dollar_profit = net_profit
                    
                    portfolio_value += net_profit
                    all_trades.append(trade)
            except Exception as e:
                print(f"Error closing final position {symbol}: {e}")
        
        # Store results
        self.trades = all_trades
        self.portfolio_value_history = portfolio_history
        
        print(f"Backtesting completed. Total trades: {len(all_trades)}")
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        # Convert trades to DataFrame for easier analysis
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'symbol': trade.symbol,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'days_held': trade.days_held,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'signal_score': trade.signal_score,
                'gross_return_pct': trade.gross_return_pct,
                'net_return_pct': trade.net_return_pct,
                'dollar_profit': trade.dollar_profit,
                'exit_reason': trade.exit_reason,
                'risk_amount': trade.risk_amount
            })
        
        df = pd.DataFrame(trade_data)
        
        # Basic statistics
        total_trades = len(df)
        winning_trades = len(df[df['net_return_pct'] > 0])
        losing_trades = len(df[df['net_return_pct'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Return statistics
        avg_return = df['net_return_pct'].mean()
        avg_win = df[df['net_return_pct'] > 0]['net_return_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['net_return_pct'] < 0]['net_return_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        largest_win = df['net_return_pct'].max()
        largest_loss = df['net_return_pct'].min()
        std_returns = df['net_return_pct'].std()
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
        
        # Profit factor
        gross_profit = df[df['dollar_profit'] > 0]['dollar_profit'].sum()
        gross_loss = abs(df[df['dollar_profit'] < 0]['dollar_profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Portfolio metrics
        if self.portfolio_value_history:
            initial_capital = self.portfolio_value_history[0][1]
            final_capital = self.portfolio_value_history[-1][1]
            total_return = ((final_capital - initial_capital) / initial_capital) * 100
            
            # Calculate daily returns for Sharpe ratio
            portfolio_values = [pv[1] for pv in self.portfolio_value_history]
            daily_rets = []
            for i in range(1, len(portfolio_values)):
                daily_ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                daily_rets.append(daily_ret)
            
            if daily_rets:
                sharpe_ratio = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
                
                # Maximum drawdown
                peak = portfolio_values[0]
                max_dd = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if drawdown > max_dd:
                        max_dd = drawdown
                max_drawdown = max_dd * 100
            else:
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            total_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            initial_capital = self.config.initial_capital
            final_capital = self.config.initial_capital
        
        # Exit reason analysis
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        # Performance by signal score
        score_analysis = {}
        if 'signal_score' in df.columns:
            high_score_trades = df[df['signal_score'] >= 70]
            medium_score_trades = df[(df['signal_score'] >= 45) & (df['signal_score'] < 70)]
            low_score_trades = df[df['signal_score'] < 45]
            
            score_analysis = {
                'high_score': {
                    'count': len(high_score_trades),
                    'win_rate': (len(high_score_trades[high_score_trades['net_return_pct'] > 0]) / len(high_score_trades) * 100) if len(high_score_trades) > 0 else 0,
                    'avg_return': high_score_trades['net_return_pct'].mean() if len(high_score_trades) > 0 else 0
                },
                'medium_score': {
                    'count': len(medium_score_trades),
                    'win_rate': (len(medium_score_trades[medium_score_trades['net_return_pct'] > 0]) / len(medium_score_trades) * 100) if len(medium_score_trades) > 0 else 0,
                    'avg_return': medium_score_trades['net_return_pct'].mean() if len(medium_score_trades) > 0 else 0
                },
                'low_score': {
                    'count': len(low_score_trades),
                    'win_rate': (len(low_score_trades[low_score_trades['net_return_pct'] > 0]) / len(low_score_trades) * 100) if len(low_score_trades) > 0 else 0,
                    'avg_return': low_score_trades['net_return_pct'].mean() if len(low_score_trades) > 0 else 0
                }
            }
        
        # Compile results
        metrics = {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': round(win_rate, 2),
                'total_return_pct': round(total_return, 2),
                'initial_capital': initial_capital,
                'final_capital': round(final_capital, 2),
                'net_profit': round(final_capital - initial_capital, 2)
            },
            'returns': {
                'avg_return_pct': round(avg_return, 2),
                'avg_win_pct': round(avg_win, 2),
                'avg_loss_pct': round(avg_loss, 2),
                'largest_win_pct': round(largest_win, 2),
                'largest_loss_pct': round(largest_loss, 2),
                'std_returns_pct': round(std_returns, 2),
                'expectancy_pct': round(expectancy, 2)
            },
            'risk_metrics': {
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf'
            },
            'exit_analysis': exit_reasons,
            'score_analysis': score_analysis,
            'trade_details': df.to_dict('records')
        }
        
        return metrics
    
    def generate_backtest_report(self, metrics: Dict[str, Any], save_to_file: bool = True) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
ADVANCED BACKTESTING REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'-'*30}
Total Trades: {metrics['summary']['total_trades']}
Winning Trades: {metrics['summary']['winning_trades']}
Losing Trades: {metrics['summary']['losing_trades']}
Win Rate: {metrics['summary']['win_rate_pct']}%

PORTFOLIO PERFORMANCE
{'-'*30}
Initial Capital: ${metrics['summary']['initial_capital']:,.0f}
Final Capital: ${metrics['summary']['final_capital']:,.0f}
Net Profit: ${metrics['summary']['net_profit']:,.0f}
Total Return: {metrics['summary']['total_return_pct']}%

RETURN ANALYSIS
{'-'*30}
Average Return per Trade: {metrics['returns']['avg_return_pct']}%
Average Winning Trade: {metrics['returns']['avg_win_pct']}%
Average Losing Trade: {metrics['returns']['avg_loss_pct']}%
Largest Win: {metrics['returns']['largest_win_pct']}%
Largest Loss: {metrics['returns']['largest_loss_pct']}%
Expectancy: {metrics['returns']['expectancy_pct']}%

RISK METRICS
{'-'*30}
Sharpe Ratio: {metrics['risk_metrics']['sharpe_ratio']}
Maximum Drawdown: {metrics['risk_metrics']['max_drawdown_pct']}%
Profit Factor: {metrics['risk_metrics']['profit_factor']}

EXIT REASON ANALYSIS
{'-'*30}"""
        
        for reason, count in metrics['exit_analysis'].items():
            pct = (count / metrics['summary']['total_trades']) * 100
            report += f"\n{reason}: {count} ({pct:.1f}%)"
        
        report += f"""

SIGNAL SCORE ANALYSIS
{'-'*30}"""
        
        for score_level, data in metrics['score_analysis'].items():
            report += f"""
{score_level.upper()} SCORES:
  Trades: {data['count']}
  Win Rate: {data['win_rate']:.1f}%
  Avg Return: {data['avg_return']:.2f}%"""
        
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_report_{timestamp}.txt'
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to {filename}")
        
        return report

# Example usage
if __name__ == "__main__":
    # Test the backtesting framework
    config = BacktestConfig(
        initial_capital=1000000,
        risk_per_trade=0.01,
        min_score_threshold=50
    )
    
    backtester = AdvancedBacktester(config)
    
    # Sample signals data (you would get this from your screening system)
    sample_signals = {
        'RELIANCE.NS': [
            {
                'date': datetime(2024, 1, 15),
                'signal_data': {
                    'composite_score': 75,
                    'atr': 25.5,
                    'current_price': 2500
                }
            }
        ]
    }
    
    # Run backtest
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    print("Running sample backtest...")
    metrics = backtester.walk_forward_backtest(sample_signals, start_date, end_date)
    
    # Generate report
    report = backtester.generate_backtest_report(metrics)
    print(report)