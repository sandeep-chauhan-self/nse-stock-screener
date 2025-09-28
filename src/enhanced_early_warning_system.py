"""
Enhanced Early Warning System for Stock Analysis
Advanced version with comprehensive technical            # Add debugging info
            print(f"Market regime: {self.market_regime}, type: {type(self.market_regime)}")
            print(f"Market regime from composite_scorer module: {type(self.scorer.regime_adjustments).__name__}")
            print(f"cs_regime: {cs_regime}, type: {type(cs_regime)}")
            
            # Ensure the market regime is of the correct type for CompositeScorer
            # This handles the case when types don't match despite having the same values
            from composite_scorer import MarketRegime as CSMarketRegime
            regime_name = self.market_regime.name
            cs_regime = CSMarketRegime[regime_name]
            
            print(f"cs_regime after conversion: {cs_regime}, type: {type(cs_regime)}")
            
            # Compute composite score, composite scoring,
risk management, and backtesting capabilities.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import our centralized constants and core functionality
from constants import (
    MarketRegime, TRADING_CONSTANTS, RISK_CONSTANTS, DISPLAY_CONSTANTS,
    ERROR_MESSAGES, SUCCESS_MESSAGES, FILE_CONSTANTS, PROJECT_ROOT_PATH
)
from core import (
    StockLoader, DataFetcher, MarketRegimeDetector, PathManager,
    DisplayUtils, RateLimiter
)

# Import our enhanced modules
from advanced_indicators import AdvancedIndicator
from composite_scorer import CompositeScorer
from advanced_backtester import AdvancedBacktester, BacktestConfig
from risk_manager import RiskManager, RiskConfig
from signal_generator import SignalGenerator
from forecast_engine import ForecastEngine

class EnhancedEarlyWarningSystem:
    """Enhanced Early Warning System with advanced technical analysis"""
    
    def __init__(self, custom_stocks: Optional[List[str]] = None, 
                 input_file: Optional[str] = None,
                 batch_size: Optional[int] = None, timeout: Optional[int] = None):
        """
        Initialize Enhanced Early Warning System
        
        Args:
            custom_stocks: List of custom stock symbols
            input_file: Path to file containing stock symbols
            batch_size: Number of stocks to process in each batch
            timeout: Seconds to wait between batches
        """
        self.batch_size = batch_size or TRADING_CONSTANTS['DEFAULT_BATCH_SIZE']
        self.timeout = timeout or TRADING_CONSTANTS['DEFAULT_TIMEOUT']
        self.market_regime = MarketRegime.SIDEWAYS
        self.analysis_results = []
        
        # Initialize output directories using PathManager
        self.output_dirs = PathManager.setup_output_directories()
        
        # Load stock symbols using StockLoader
        self.nse_stocks = StockLoader.load_stocks(custom_stocks, input_file)
        
        # Initialize enhanced engines
        self.indicators_engine = AdvancedIndicator()
        self.scorer = CompositeScorer()
        self.signal_generator = SignalGenerator()
        self.forecast_engine = ForecastEngine()
        
        # Create risk config with defaults
        risk_config = RiskConfig(
            max_portfolio_risk=RISK_CONSTANTS['DEFAULT_MAX_PORTFOLIO_RISK'],
            max_position_size=RISK_CONSTANTS['DEFAULT_MAX_POSITION_SIZE'],
            max_daily_loss=RISK_CONSTANTS['DEFAULT_MAX_DAILY_LOSS'],
            max_concurrent_positions=RISK_CONSTANTS['DEFAULT_MAX_CONCURRENT_POSITIONS']
        )
        
        # Initialize risk manager with config
        self.risk_manager = RiskManager(
            initial_capital=TRADING_CONSTANTS['DEFAULT_INITIAL_CAPITAL'],
            config=risk_config
        )
        
        print(f"{SUCCESS_MESSAGES['SYSTEM_INITIALIZED']}")
        print(f"📊 Loaded {len(self.nse_stocks)} stocks for analysis")
        print(f"🔧 Batch size: {self.batch_size}, Timeout: {self.timeout}s")
    
    def analyze_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock with full indicator suite and optimal entry calculation"""
        try:
            print(f"Analyzing {symbol}...")
            
            # Get historical data for optimal entry calculation
            from core import DataFetcher
            historical_data = DataFetcher.fetch_stock_data(symbol, period="1y")
            
            # Calculate historical price levels
            historical_prices = {}
            if historical_data is not None and not historical_data.empty:
                historical_prices = {
                    'All_time_high': round(historical_data['High'].max(), 2),
                    'All_time_low': round(historical_data['Low'].min(), 2),
                    'Last_30Day_High': round(historical_data['High'].tail(30).max(), 2) if len(historical_data) >= 30 else round(historical_data['High'].max(), 2),
                    'Last_30Day_Low': round(historical_data['Low'].tail(30).min(), 2) if len(historical_data) >= 30 else round(historical_data['Low'].min(), 2),
                    '1Week_High': round(historical_data['High'].tail(7).max(), 2) if len(historical_data) >= 7 else round(historical_data['High'].max(), 2),
                    '1Week_Low': round(historical_data['Low'].tail(7).min(), 2) if len(historical_data) >= 7 else round(historical_data['Low'].min(), 2)
                }
            else:
                historical_prices = {
                    'All_time_high': 'N/A',
                    'All_time_low': 'N/A',
                    'Last_30Day_High': 'N/A',
                    'Last_30Day_Low': 'N/A',
                    '1Week_High': 'N/A',
                    '1Week_Low': 'N/A'
                }
            
            # Get all technical indicators
            indicators = self.indicators_engine.compute_all_indicators(symbol)
            
            if indicators is None:
                print(f"Failed to compute indicators for {symbol}")
                return None
            
            # Add debugging info
            #print(f"Market regime: {self.market_regime}, type: {type(self.market_regime)}")
            #print(f"Market regime from composite_scorer module: {type(self.scorer.regime_adjustments).__name__}")
            
            # Ensure the market regime is of the correct type for CompositeScorer
            # This handles the case when types don't match despite having the same values
            from composite_scorer import MarketRegime as CSMarketRegime
            regime_name = self.market_regime.name
            cs_regime = CSMarketRegime[regime_name]
            
            # Compute composite score
            scoring_result = self.scorer.compute_composite_score(indicators, cs_regime)
            
            if scoring_result is None:
                print(f"Failed to compute score for {symbol}")
                return None
            
            # Generate trading signal
            # Use the original MarketRegime from constants for signal_generator
            signal_result = self.signal_generator.generate_signal(
                scoring_result['composite_score'], 
                indicators, 
                self.market_regime  # Use original regime instead of cs_regime
            )
            
            # Enhanced risk management with entry/stop/target calculations
            entry_price = indicators.get('current_price', 0)
            
            # Debug output for signal generation
            print(f"Debug - {symbol}: Score={scoring_result['composite_score']}, Signal={signal_result['signal']}, Confidence={signal_result.get('confidence', 'N/A')}, Entry=${entry_price:.1f}")
            
            # For BUY signals, calculate detailed entry/stop/target with Monte Carlo
            if signal_result['signal'] == 'BUY':
                # Enhanced position analysis with Monte Carlo optimal entry
                position_analysis = self.risk_manager.enhanced_position_analysis(
                    symbol=symbol, 
                    signal=signal_result['signal'], 
                    composite_score=scoring_result['composite_score'], 
                    indicators=indicators, 
                    signal_data=signal_result,
                    historical_data=historical_data,
                    market_regime=self.market_regime
                )
                
                # **CRITICAL FIX**: Validate and ensure target values for BUY signals
                optimal_entry = position_analysis.get('entry_value', entry_price)
                target_value = position_analysis.get('target_value')
                
                # **MANDATORY**: BUY signals MUST have valid targets
                if signal_result['signal'] == 'BUY' and (target_value is None or target_value <= optimal_entry):
                    # Force calculate target if missing or invalid
                    atr = indicators.get('atr', optimal_entry * 0.02)
                    stop_value = position_analysis.get('stop_value', optimal_entry * 0.97)
                    risk_per_share = abs(optimal_entry - stop_value)
                    
                    if risk_per_share <= 0:
                        risk_per_share = optimal_entry * 0.03  # 3% minimum risk
                    
                    target_value = optimal_entry + (2.5 * risk_per_share)  # Force 2.5:1 R:R
                    print(f"    FORCED target for {symbol}: Entry={optimal_entry:.2f}, Target={target_value:.2f}")
                
                # Estimate duration using optimal entry and target (only if valid target exists)
                if signal_result['signal'] == 'BUY' and target_value and target_value > optimal_entry:
                    duration_estimate = self.forecast_engine.estimate_duration(
                        symbol, optimal_entry, target_value, indicators
                    )
                else:
                    duration_estimate = None
                
                # Extract values from position_analysis (includes Monte Carlo results)
                risk_info = {
                    'can_enter_position': position_analysis['can_enter_position'],
                    'risk_reason': position_analysis['risk_reason'],
                    'suggested_quantity': position_analysis['position_size'],
                    'risk_amount': position_analysis['risk_amount'],
                    'entry_value': position_analysis['entry_value'],
                    'stop_value': position_analysis['stop_value'],
                    'target_value': position_analysis['target_value'],
                    'risk_reward_ratio': position_analysis['risk_reward_ratio'],
                    'position_size_pct': position_analysis['portfolio_impact']['position_weight'],
                    'max_loss_amount': position_analysis['risk_amount'],
                    # Monte Carlo specific fields
                    'hit_probability': position_analysis.get('hit_probability', 0.0),
                    'indicator_confidence': position_analysis.get('indicator_confidence', 0.0),
                    'monte_carlo_paths': position_analysis.get('monte_carlo_paths', 0),
                    'fallback_used': position_analysis.get('fallback_used', 'Unknown'),
                    'data_confidence': position_analysis.get('data_confidence', 'UNKNOWN'),
                    'calculation_method': position_analysis.get('calculation_method', 'ATR-based'),
                    'execution_time_ms': position_analysis.get('execution_time_ms', 0.0)
                }
            else:
                # For HOLD/AVOID signals, basic risk info but still include timing analysis
                atr = indicators.get('atr', entry_price * 0.02)
                stop_loss = entry_price - (2.0 * atr)
                
                can_enter, risk_reason, quantity, risk_amount = self.risk_manager.can_enter_position(
                    symbol, entry_price, stop_loss, scoring_result['composite_score']
                )
                
                # Still perform entry timing analysis even for AVOID signals
                entry_timing_analysis = self.risk_manager.analyze_entry_timing(
                    symbol=symbol,
                    current_price=entry_price,
                    indicators=indicators,
                    historical_data=historical_data
                )
                
                risk_info = {
                    'can_enter_position': can_enter,
                    'risk_reason': risk_reason,
                    'suggested_quantity': quantity,
                    'risk_amount': round(risk_amount, 2),
                    'entry_value': entry_price,
                    'stop_value': round(stop_loss, 2),
                    'target_value': None,
                    'risk_reward_ratio': None,
                    'position_size_pct': None,
                    'max_loss_amount': None,
                    # Include timing analysis even for AVOID signals
                    'entry_timing': entry_timing_analysis['entry_timing'],
                    'timing_confidence': entry_timing_analysis['timing_confidence'],
                    'timing_reason': entry_timing_analysis['reason'],
                    'wait_probability': entry_timing_analysis['wait_probability'],
                    'suggested_wait_days': entry_timing_analysis['suggested_wait_days'],
                    'spike_score': entry_timing_analysis['spike_score']
                }
                
                # **FIXED**: Estimate duration for ALL signals where target exists
                # For non-BUY signals, we may still want to estimate duration for theoretical targets
                # or provide fallback duration based on volatility
                duration_estimate = None
                if risk_info.get('target_value') and risk_info['target_value'] > 0:
                    # If target exists, estimate duration regardless of signal type
                    duration_estimate = self.forecast_engine.estimate_duration(
                        symbol, entry_price, risk_info['target_value'], indicators
                    )
                else:
                    # Fallback: Estimate duration based on historical volatility even without target
                    duration_estimate = self._estimate_fallback_duration(symbol, entry_price, indicators, historical_data)
            
            # **CRITICAL FIX**: Validate data integrity before returning
            risk_info = self._validate_and_fix_trade_data(risk_info, signal_result['signal'], symbol)
            
            # Add new components to result
            scoring_result['signal_info'] = signal_result
            scoring_result['duration_estimate'] = duration_estimate
            scoring_result['risk_management'] = risk_info
            scoring_result['historical_prices'] = historical_prices
            
            return scoring_result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_and_fix_trade_data(self, risk_info: Dict[str, Any], signal: str, symbol: str) -> Dict[str, Any]:
        """
        **CRITICAL FIX**: Validate and fix trade data to ensure mathematical consistency
        
        Enhanced validation for hybrid entry system:
        - Entry/stop/target validation for BUY/SELL signals
        - Risk-reward ratio recomputation and validation
        - Entry method validation and current_price fallback detection
        - Validation flags and messages for CI gates
        """
        try:
            # Initialize validation fields
            validation_flag = 'PASS'
            validation_message = ''
            entry_method = risk_info.get('entry_method', 'UNAVAILABLE')
            current_price = risk_info.get('entry_value', 0)  # Approximation for validation
            
            if signal in {'BUY', 'SELL'}:
                entry_value = risk_info.get('entry_value', 0)
                stop_value = risk_info.get('stop_value', 0)
                target_value = risk_info.get('target_value')
                
                # **MANDATORY VALIDATION**: Check critical values exist and are valid
                if entry_value <= 0:
                    validation_flag = 'FAIL'
                    validation_message += 'Missing or invalid entry_value for actionable signal; '
                
                if stop_value <= 0:
                    validation_flag = 'FAIL'
                    validation_message += 'Missing or invalid stop_value for actionable signal; '
                
                if target_value is None or target_value <= 0:
                    validation_flag = 'FAIL'
                    validation_message += 'Missing target_value for actionable signal; '
                
                # **MANDATORY VALIDATION**: Target must be profitable for signal direction
                if signal == 'BUY' and target_value and target_value <= entry_value:
                    validation_flag = 'FAIL'
                    validation_message += f'BUY signal target {target_value:.2f} <= entry {entry_value:.2f}; '
                
                if signal == 'SELL' and target_value and target_value >= entry_value:
                    validation_flag = 'FAIL'
                    validation_message += f'SELL signal target {target_value:.2f} >= entry {entry_value:.2f}; '
                
                # **MANDATORY FIX**: Recompute risk-reward ratio
                if entry_value > 0 and stop_value > 0 and target_value and target_value > 0:
                    if signal == 'BUY':
                        risk = entry_value - stop_value
                        reward = target_value - entry_value
                    else:  # SELL
                        risk = stop_value - entry_value
                        reward = entry_value - target_value
                    
                    if risk > 0:
                        computed_rrr = reward / risk
                        
                        # Check against stored RRR
                        stored_rrr = risk_info.get('risk_reward_ratio')
                        if stored_rrr and abs(computed_rrr - stored_rrr) > 0.05:
                            validation_message += f'R:R mismatch: stored {stored_rrr:.2f}, computed {computed_rrr:.2f}; '
                            risk_info['risk_reward_ratio'] = round(computed_rrr, 2)
                        
                        # Minimum R:R validation
                        if computed_rrr < 1.5:
                            validation_flag = 'REVIEW'
                            validation_message += f'Low R:R ratio {computed_rrr:.2f} < 1.5; '
                    else:
                        validation_flag = 'FAIL'
                        validation_message += 'Invalid risk calculation (risk <= 0); '
                
                # **ENTRY METHOD VALIDATION**: Check for current_price fallback anti-pattern
                if abs(entry_value - current_price) < 0.01 and entry_method == 'CURRENT_PRICE':
                    validation_message += 'Entry equals current price (anti-pattern); '
                    if validation_flag == 'PASS':
                        validation_flag = 'REVIEW'
                
                # **RSI VALIDATION**: High RSI for BUY signals
                rsi = risk_info.get('rsi', 50)  # Need to get RSI from somewhere
                if signal == 'BUY' and rsi > 75 and entry_method != 'BREAKOUT':
                    validation_message += f'High RSI {rsi:.1f} for BUY signal without breakout; '
                    if validation_flag == 'PASS':
                        validation_flag = 'REVIEW'
            
            # Set validation fields in risk_info
            risk_info['validation_flag'] = validation_flag
            risk_info['validation_message'] = validation_message.strip()
            risk_info['entry_method'] = entry_method
            risk_info['order_type'] = risk_info.get('order_type', 'MARKET')
            risk_info['entry_clamp_reason'] = risk_info.get('entry_clamp_reason')
            risk_info['entry_debug'] = risk_info.get('entry_debug', {})
            
            # Log validation results
            if validation_flag == 'FAIL':
                print(f"    ❌ VALIDATION FAILED {symbol}: {validation_message}")
            elif validation_flag == 'REVIEW':
                print(f"    ⚠️  VALIDATION REVIEW {symbol}: {validation_message}")
            else:
                print(f"    ✅ VALIDATION PASSED {symbol}")
            
            return risk_info
            
        except Exception as e:
            print(f"    ERROR validating {symbol}: {e}")
            risk_info['validation_flag'] = 'FAIL'
            risk_info['validation_message'] = f'Validation error: {str(e)}'
            return risk_info
    
    def filter_and_rank_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Filter and rank results by probability levels"""
        high_probability = []
        medium_probability = []
        low_probability = []
        
        for result in results:
            if result['probability_level'] == 'HIGH':
                high_probability.append(result)
            elif result['probability_level'] == 'MEDIUM':
                medium_probability.append(result)
            else:
                low_probability.append(result)
        
        # Sort by composite score (descending)
        high_probability.sort(key=lambda x: x['composite_score'], reverse=True)
        medium_probability.sort(key=lambda x: x['composite_score'], reverse=True)
        low_probability.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return {
            'HIGH': high_probability,
            'MEDIUM': medium_probability,
            'LOW': low_probability
        }
    
    def generate_enhanced_chart(self, symbol: str, indicators: Dict[str, Any]) -> Optional[str]:
        """Generate enhanced technical analysis chart"""
        try:
            print(f"Generating enhanced chart for {symbol}...")
            
            # Fetch data for charting
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            
            if data.empty or len(data) < 50:
                print(f"Insufficient data for {symbol} chart")
                return None
            
            # Create comprehensive chart
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], hspace=0.3)
            
            # Main price chart with volume
            ax1 = fig.add_subplot(gs[0, :])
            
            # Price and moving averages
            ax1.plot(data.index, data['Close'], 'b-', label='Close Price', linewidth=2)
            
            # Add moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            ax1.plot(data.index, data['MA20'], 'r-', label='20-day MA', alpha=0.7)
            ax1.plot(data.index, data['MA50'], 'g-', label='50-day MA', alpha=0.7)
            
            # Volume on secondary axis
            ax1_vol = ax1.twinx()
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                     for i in range(len(data))]
            ax1_vol.bar(data.index, data['Volume'], alpha=0.3, color=colors)
            ax1_vol.set_ylabel('Volume', fontsize=10)
            ax1_vol.set_ylim(0, data['Volume'].max() * 4)
            
            # Chart title with key metrics
            title = f"{symbol.replace('.NS', '')} - Enhanced Analysis\n"
            title += f"Score: {indicators.get('composite_score', 'N/A')}, "
            title += f"Probability: {indicators.get('probability_level', 'N/A')}, "
            title += f"Regime: {indicators.get('market_regime', 'N/A')}"
            ax1.set_title(title, fontsize=14, fontweight='bold')
            
            ax1.set_ylabel('Price (₹)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # RSI subplot
            ax2 = fig.add_subplot(gs[1, :])
            
            # Calculate RSI for chart
            close = data['Close']
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.ewm(alpha=1/14, adjust=False).mean()
            roll_down = down.ewm(alpha=1/14, adjust=False).mean()
            rs = roll_up / roll_down
            rsi = 100 - (100 / (1 + rs))
            
            ax2.plot(data.index, rsi, 'purple', label='RSI(14)', linewidth=2)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            ax2.fill_between(data.index, 30, 70, alpha=0.1, color='yellow')
            ax2.set_ylabel('RSI', fontsize=10)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # MACD subplot
            ax3 = fig.add_subplot(gs[2, :])
            
            # Calculate MACD for chart
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            ax3.plot(data.index, macd_line, 'b-', label='MACD', linewidth=2)
            ax3.plot(data.index, signal_line, 'r-', label='Signal', linewidth=2)
            
            # Histogram with colors
            colors = ['green' if h >= 0 else 'red' for h in histogram]
            ax3.bar(data.index, histogram, alpha=0.6, color=colors, label='Histogram')
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_ylabel('MACD', fontsize=10)
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Volume analysis subplot
            ax4 = fig.add_subplot(gs[3, :])
            
            # Volume ratio and z-score
            volume = data['Volume']
            vol_sma20 = volume.rolling(20).mean()
            vol_ratio = volume / vol_sma20
            
            ax4.plot(data.index, vol_ratio, 'orange', label='Volume Ratio (20d)', linewidth=2)
            ax4.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='High Volume (3x)')
            ax4.axhline(y=1, color='gray', linestyle='-', alpha=0.5, label='Average')
            ax4.fill_between(data.index, 0, 3, alpha=0.1, color='blue')
            ax4.set_ylabel('Vol Ratio', fontsize=10)
            ax4.set_xlabel('Date', fontsize=10)
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = FILE_CONSTANTS['CHARTS_DIR'] / f"{symbol.replace('.NS', '')}_enhanced_chart.png"
            plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced chart saved: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            print(f"Error generating enhanced chart for {symbol}: {e}")
            return None
    
    def save_enhanced_reports(self, categorized_results: Dict[str, List[Dict]], 
                            analysis_summary: Dict[str, Any]) -> List[str]:
        """Save comprehensive reports to files"""
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. High probability stocks report
            if categorized_results['HIGH']:
                high_prob_df = pd.DataFrame([
                    {
                        # Column order matching comprehensive report
                        'Symbol': result['symbol'].replace('.NS', ''),
                        'Signal': result['signal_info']['signal'],
                        'Risk_Approved': result['risk_management']['can_enter_position'],
                        'Current_Price': result['key_indicators']['current_price'],
                        'Optimal_Entry': result['risk_management']['entry_value'],
                        
                        'price_difference': int(round(result['risk_management']['entry_value'] - result['key_indicators']['current_price'])) 
                                           if result['key_indicators']['current_price'] and result['risk_management']['entry_value'] 
                                           else 'N/A',
                        'Target_Value': result['risk_management']['target_value'] or 'N/A',
                        'Profit_Value': int(round(result['risk_management']['target_value'] - result['risk_management']['entry_value'])) 
                                       if result['risk_management']['target_value'] and result['risk_management']['target_value'] != 'N/A' 
                                       else 'N/A',
                        'All_time_high': result['historical_prices']['All_time_high'],
                        'All_time_low': result['historical_prices']['All_time_low'],
                        'Last_30Day_High': result['historical_prices']['Last_30Day_High'],
                        'Last_30Day_Low': result['historical_prices']['Last_30Day_Low'],
                        '1Week_High': result['historical_prices']['1Week_High'],
                        '1Week_Low': result['historical_prices']['1Week_Low'],
                       
                        'Stop_Value': result['risk_management']['stop_value'],
                        'Price_Change_%': result['key_indicators']['price_change_pct'],
                        # Entry timing analysis columns
                        'Entry_Timing': result['risk_management'].get('entry_timing', 'UNKNOWN'),
                        'Timing_Confidence': result['risk_management'].get('timing_confidence', 'UNKNOWN'),
                        'Timing_Reason': result['risk_management'].get('timing_reason', ''),
                        'Wait_Probability': result['risk_management'].get('wait_probability', 0.0),
                        'Suggested_Wait_Days': result['risk_management'].get('suggested_wait_days', 0),
                        'Spike_Score': result['risk_management'].get('spike_score', 0),
                        'Duration_Days': (result['duration_estimate']['estimated_duration_days'] 
                                        if result['duration_estimate'] 
                                        else 'N/A'),
                        'RSI': result['key_indicators']['rsi'],
                        'ADX': result['key_indicators']['adx'],
                        'Market_Regime': result.get('market_regime', 'Unknown'),
                        'Composite_Score': result['composite_score'],
                        'Indicator_Confidence': f"{result['risk_management'].get('indicator_confidence', 0):.1f}",
                        'Probability_Level': result['probability_level'],
                        'Hit_Probability': f"{result['risk_management'].get('hit_probability', 0):.1%}",
                        'Data_Confidence': result['risk_management'].get('data_confidence', 'UNKNOWN'),
                        'Monte_Carlo_Paths': result['risk_management'].get('monte_carlo_paths', 0),
                        'Position_Size': result['risk_management']['suggested_quantity'],
                        'Risk_Amount': result['risk_management']['risk_amount'],
                        'Risk_Reward_Ratio': result['risk_management']['risk_reward_ratio'] or 'N/A',
                        'Position_Size_Pct': result['risk_management'].get('position_size_pct', 'N/A'),
                        'Calculation_Method': result['risk_management'].get('calculation_method', 'ATR-based'),
                        'Fallback_Used': result['risk_management'].get('fallback_used', 'Unknown'),
                        'Volume_Score': result.get('component_scores', {}).get('volume', 0),
                        'Momentum_Score': result.get('component_scores', {}).get('momentum', 0),
                        'Trend_Score': result.get('component_scores', {}).get('trend', 0),
                        'Volatility_Score': result.get('component_scores', {}).get('volatility', 0),
                        'RelStrength_Score': result.get('component_scores', {}).get('relative_strength', 0),
                        'VolumeProfile_Score': result.get('component_scores', {}).get('volume_profile', 0),
                        'Weekly_Confirmation': result.get('component_scores', {}).get('weekly_confirmation', 0),
                        'Volume_Ratio': result['key_indicators']['volume_ratio'],
                        'Volume_Z_Score': result['key_indicators']['volume_z_score'],
                        'MACD_Signal': result['key_indicators']['macd_signal'],
                        'ATR_%': result['key_indicators']['atr_pct'],
                        'Rel_Strength_20d': result['key_indicators']['relative_strength_20d'],
                        'Risk_Reason': result['risk_management']['risk_reason'],
                        'Execution_Time_ms': f"{result['risk_management'].get('execution_time_ms', 0):.1f}",
                        # New hybrid entry system columns
                        'entry_method': result['risk_management'].get('entry_method', 'UNAVAILABLE'),
                        'order_type': result['risk_management'].get('order_type', 'MARKET'),
                        'validation_flag': result['risk_management'].get('validation_flag', 'PASS'),
                        'validation_message': result['risk_management'].get('validation_message', ''),
                        'entry_clamp_reason': result['risk_management'].get('entry_clamp_reason'),
                        'entry_debug': str(result['risk_management'].get('entry_debug', {})),
                    }
                    for result in categorized_results['HIGH']
                ])
                
                high_file = FILE_CONSTANTS['REPORTS_DIR'] / f'high_probability_enhanced_{timestamp}.csv'
                high_prob_df.to_csv(str(high_file), index=False)
                saved_files.append(str(high_file))
                print(f"High probability report saved: {high_file}")
            
            # 2. All results comprehensive report
            all_results = []
            for level, results in categorized_results.items():
                all_results.extend(results)
            
            if all_results:
                comprehensive_df = pd.DataFrame([
                    {
                        # Column order as requested by user
                        'Symbol': result['symbol'].replace('.NS', ''),
                        'Signal': result['signal_info']['signal'],
                        'Risk_Approved': result['risk_management']['can_enter_position'],
                        'Current_Price': result['key_indicators']['current_price'],
                        'Optimal_Entry': result['risk_management']['entry_value'],
                        'price_difference': int(round(result['risk_management']['entry_value'] - result['key_indicators']['current_price'])) 
                                           if result['key_indicators']['current_price'] and result['risk_management']['entry_value'] 
                                           else 'N/A',
                        'Target_Value': result['risk_management']['target_value'] or 'N/A',
                        'Profit_Value': int(round(result['risk_management']['target_value'] - result['risk_management']['entry_value'])) 
                                       if result['risk_management']['target_value'] and result['risk_management']['target_value'] != 'N/A' 
                                       else 'N/A',
                        'All_time_high': result['historical_prices']['All_time_high'],
                        'All_time_low': result['historical_prices']['All_time_low'],
                        'Last_30Day_High': result['historical_prices']['Last_30Day_High'],
                        'Last_30Day_Low': result['historical_prices']['Last_30Day_Low'],
                        '1Week_High': result['historical_prices']['1Week_High'],
                        '1Week_Low': result['historical_prices']['1Week_Low'],
                        
                        'Stop_Value': result['risk_management']['stop_value'],
                        'Price_Change_%': result['key_indicators']['price_change_pct'],
                        # Entry timing analysis columns
                        'Entry_Timing': result['risk_management'].get('entry_timing', 'UNKNOWN'),
                        'Timing_Confidence': result['risk_management'].get('timing_confidence', 'UNKNOWN'),
                        'Timing_Reason': result['risk_management'].get('timing_reason', ''),
                        'Wait_Probability': result['risk_management'].get('wait_probability', 0.0),
                        'Suggested_Wait_Days': result['risk_management'].get('suggested_wait_days', 0),
                        'Spike_Score': result['risk_management'].get('spike_score', 0),
                        'Duration_Days': (result['duration_estimate']['estimated_duration_days'] 
                                        if result['duration_estimate'] 
                                        else 'N/A'),
                        'RSI': result['key_indicators']['rsi'],
                        'ADX': result['key_indicators']['adx'],
                        'Market_Regime': result['market_regime'],
                        'Composite_Score': result['composite_score'],
                        'Indicator_Confidence': f"{result['risk_management'].get('indicator_confidence', 0):.1f}",
                        'Probability_Level': result['probability_level'],
                        'Hit_Probability': f"{result['risk_management'].get('hit_probability', 0):.1%}",
                        'Data_Confidence': result['risk_management'].get('data_confidence', 'UNKNOWN'),
                        'Monte_Carlo_Paths': result['risk_management'].get('monte_carlo_paths', 0),
                        'Position_Size': result['risk_management']['suggested_quantity'],
                        'Risk_Amount': result['risk_management']['risk_amount'],
                        'Risk_Reward_Ratio': result['risk_management']['risk_reward_ratio'] or 'N/A',
                        'Position_Size_Pct': result['risk_management']['position_size_pct'] or 'N/A',
                        'Calculation_Method': result['risk_management'].get('calculation_method', 'ATR-based'),
                        'Fallback_Used': result['risk_management'].get('fallback_used', 'Unknown'),
                        'Volume_Score': result['component_scores']['volume'],
                        'Momentum_Score': result['component_scores']['momentum'],
                        'Trend_Score': result['component_scores']['trend'],
                        'Volatility_Score': result['component_scores']['volatility'],
                        'RelStrength_Score': result['component_scores']['relative_strength'],
                        'VolumeProfile_Score': result['component_scores']['volume_profile'],
                        'Weekly_Confirmation': result['component_scores']['weekly_confirmation'],
                        'Volume_Ratio': result['key_indicators']['volume_ratio'],
                        'Volume_Z_Score': result['key_indicators']['volume_z_score'],
                        'MACD_Signal': result['key_indicators']['macd_signal'],
                        'ATR_%': result['key_indicators']['atr_pct'],
                        'Rel_Strength_20d': result['key_indicators']['relative_strength_20d'],
                        'Risk_Reason': result['risk_management']['risk_reason'],
                        'Execution_Time_ms': f"{result['risk_management'].get('execution_time_ms', 0):.1f}",
                        # New hybrid entry system columns
                        'entry_method': result['risk_management'].get('entry_method', 'UNAVAILABLE'),
                        'order_type': result['risk_management'].get('order_type', 'MARKET'),
                        'validation_flag': result['risk_management'].get('validation_flag', 'PASS'),
                        'validation_message': result['risk_management'].get('validation_message', ''),
                        'entry_clamp_reason': result['risk_management'].get('entry_clamp_reason'),
                        'entry_debug': str(result['risk_management'].get('entry_debug', {})),
                    }
                    for result in all_results
                ])
                
                comprehensive_file = FILE_CONSTANTS['REPORTS_DIR'] / f'comprehensive_analysis_{timestamp}.csv'
                comprehensive_df.to_csv(str(comprehensive_file), index=False)
                saved_files.append(str(comprehensive_file))
                print(f"Comprehensive report saved: {comprehensive_file}")
            
            # 3. Analysis summary report
            summary_file = FILE_CONSTANTS['REPORTS_DIR'] / f'analysis_summary_{timestamp}.txt'
            
            with open(str(summary_file), 'w') as f:
                f.write("ENHANCED EARLY WARNING SYSTEM ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Market Regime: {analysis_summary['market_regime']}\n")
                f.write(f"Total Stocks Analyzed: {analysis_summary['total_analyzed']}\n")
                f.write(f"Successful Analysis: {analysis_summary['successful_analysis']}\n")
                f.write(f"Analysis Success Rate: {analysis_summary['success_rate']:.1f}%\n\n")
                
                f.write("PROBABILITY DISTRIBUTION:\n")
                f.write("-" * 30 + "\n")
                for level, count in analysis_summary['probability_distribution'].items():
                    pct = (count / analysis_summary['successful_analysis']) * 100 if analysis_summary['successful_analysis'] > 0 else 0
                    f.write(f"{level}: {count} stocks ({pct:.1f}%)\n")
                
                f.write("\nRISK MANAGEMENT SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Risk Approved Positions: {analysis_summary['risk_approved']}\n")
                f.write(f"Risk Approval Rate: {analysis_summary['risk_approval_rate']:.1f}%\n")
                
                if analysis_summary['top_scores']:
                    f.write("\nTOP 10 SCORES:\n")
                    f.write("-" * 30 + "\n")
                    for i, result in enumerate(analysis_summary['top_scores'][:10], 1):
                        f.write(f"{i}. {result['symbol'].replace('.NS', '')} - "
                               f"Score: {result['composite_score']}, "
                               f"Level: {result['probability_level']}\n")
            
            saved_files.append(summary_file)
            print(f"Summary report saved: {summary_file}")
            
        except Exception as e:
            print(f"Error saving reports: {e}")
        
        return saved_files
    
    def run_enhanced_analysis(self) -> Dict[str, Any]:
        """Run the complete enhanced analysis pipeline"""
        print("🚀 ENHANCED EARLY WARNING SYSTEM")
        print("=" * 60)
        print(f"Analyzing {len(self.nse_stocks)} stocks with advanced indicators...\n")
        
        # Detect market regime
        self.market_regime = MarketRegimeDetector.detect_regime()
        
        # Analysis tracking
        all_results = []
        failed_analysis = []
        
        # Process stocks in batches
        total_stocks = len(self.nse_stocks)
        total_batches = (total_stocks + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_stocks)
            batch = self.nse_stocks[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_num+1}/{total_batches} ({len(batch)} stocks)...")
            print(f"Stocks {start_idx+1}-{end_idx} of {total_stocks}")
            
            for i, symbol in enumerate(batch):
                progress = (i + 1) / len(batch) * 100
                print(f"[{progress:.1f}%] Analyzing {symbol}...")
                
                # Analyze stock
                result = self.analyze_single_stock(symbol)
                
                if result:
                    all_results.append(result)
                else:
                    failed_analysis.append(symbol)
                
                # Brief pause to avoid rate limits
                time.sleep(0.5)
            
            # Pause between batches
            if batch_num < total_batches - 1:
                print(f"\nPausing for {self.timeout} seconds...")
                time.sleep(self.timeout)
        
        # Filter and categorize results
        categorized_results = self.filter_and_rank_results(all_results)
        
        # Generate analysis summary
        analysis_summary = {
            'market_regime': self.market_regime.value,
            'total_analyzed': len(self.nse_stocks),
            'successful_analysis': len(all_results),
            'failed_analysis': len(failed_analysis),
            'success_rate': (len(all_results) / len(self.nse_stocks)) * 100,
            'probability_distribution': {
                'HIGH': len(categorized_results['HIGH']),
                'MEDIUM': len(categorized_results['MEDIUM']),
                'LOW': len(categorized_results['LOW'])
            },
            'risk_approved': len([r for r in all_results if r['risk_management']['can_enter_position']]),
            'risk_approval_rate': 0,
            'top_scores': sorted(all_results, key=lambda x: x['composite_score'], reverse=True)
        }
        
        if len(all_results) > 0:
            analysis_summary['risk_approval_rate'] = (analysis_summary['risk_approved'] / len(all_results)) * 100
        
        # Display results
        self.display_analysis_results(categorized_results, analysis_summary)
        
        # Generate charts for top stocks
        self.generate_charts_for_top_stocks(categorized_results)
        
        # Save reports
        saved_files = self.save_enhanced_reports(categorized_results, analysis_summary)
        
        # Store results for potential backtesting
        self.analysis_results = all_results
        
        print(f"\n📊 ANALYSIS COMPLETE")
        print(f"Total successful analysis: {len(all_results)}/{len(self.nse_stocks)}")
        print(f"Reports saved: {len(saved_files)} files")
        
        return {
            'categorized_results': categorized_results,
            'analysis_summary': analysis_summary,
            'saved_files': saved_files
        }
    
    def display_analysis_results(self, categorized_results: Dict[str, List[Dict]], 
                               analysis_summary: Dict[str, Any]):
        """Display formatted analysis results"""
        print("\n" + "=" * 80)
        print("📈 ENHANCED ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\n🌍 Market Regime: {analysis_summary['market_regime'].upper()}")
        print(f"📊 Analysis Success Rate: {analysis_summary['success_rate']:.1f}%")
        print(f"✅ Risk Approved Positions: {analysis_summary['risk_approved']} "
              f"({analysis_summary['risk_approval_rate']:.1f}%)")
        
        # High probability stocks
        if categorized_results['HIGH']:
            print("\n🎯 HIGH PROBABILITY STOCKS (Score ≥ 70)")
            print("-" * 80)
            
            high_df = pd.DataFrame([
                {
                    'Symbol': r['symbol'].replace('.NS', ''),
                    'Signal': r['signal_info']['signal'],
                    'Score': r['composite_score'],
                    'Price': f"₹{r['key_indicators']['current_price']:.1f}",
                    'Entry': f"₹{r['risk_management']['entry_value']:.1f}",
                    'Stop': f"₹{r['risk_management']['stop_value']:.1f}",
                    'Target': (f"₹{r['risk_management']['target_value']:.1f}" 
                             if r['risk_management']['target_value'] else 'N/A'),
                    'Duration': (f"{r['duration_estimate']['estimated_duration_days']:.0f}d" 
                               if r['duration_estimate'] 
                               else 'N/A'),
                    'Vol_Ratio': f"{r['key_indicators']['volume_ratio']:.1f}x",
                    'RSI': f"{r['key_indicators']['rsi']:.1f}",
                    'Risk_OK': '✅' if r['risk_management']['can_enter_position'] else '❌',
                    'Qty': r['risk_management']['suggested_quantity'],
                    'Entry_Timing': r['risk_management'].get('entry_timing', 'UNKNOWN'),
                    'Wait_Prob': f"{r['risk_management'].get('wait_probability', 0):.0%}"
                }
                for r in categorized_results['HIGH'][:15]  # Top 15
            ])
            
            print(high_df.to_string(index=False))
        else:
            print("\n❌ No HIGH probability stocks found today.")
        
        # Medium probability stocks (top 10)
        if categorized_results['MEDIUM']:
            print("\n📊 TOP MEDIUM PROBABILITY STOCKS (Score 45-69)")
            print("-" * 80)
            
            medium_df = pd.DataFrame([
                {
                    'Symbol': r['symbol'].replace('.NS', ''),
                    'Signal': r['signal_info']['signal'],
                    'Score': r['composite_score'],
                    'Price': f"₹{r['key_indicators']['current_price']:.1f}",
                    'Change%': f"{r['key_indicators']['price_change_pct']:.1f}%",
                    'Vol_Ratio': f"{r['key_indicators']['volume_ratio']:.1f}x",
                    'RSI': f"{r['key_indicators']['rsi']:.1f}",
                    'Risk_OK': '✅' if r['risk_management']['can_enter_position'] else '❌'
                }
                for r in categorized_results['MEDIUM'][:10]  # Top 10
            ])
            
            print(medium_df.to_string(index=False))
        
        print(f"\n📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⚠️  This is probability-based analysis. Always do your own research!")
    
    def generate_charts_for_top_stocks(self, categorized_results: Dict[str, List[Dict]]):
        """Generate enhanced charts for top-scoring stocks"""
        print("\n📈 Generating enhanced charts...")
        
        # Generate charts for top high probability stocks
        chart_count = 0
        max_charts = 10  # Limit to avoid too many API calls
        
        for result in categorized_results['HIGH'][:5]:  # Top 5 high prob
            if chart_count >= max_charts:
                break
            
            symbol = result['symbol']
            self.generate_enhanced_chart(symbol, result)
            chart_count += 1
            time.sleep(1)  # Rate limiting
        
        # Generate charts for top medium probability stocks
        for result in categorized_results['MEDIUM'][:5]:  # Top 5 medium prob
            if chart_count >= max_charts:
                break
            
            symbol = result['symbol']
            self.generate_enhanced_chart(symbol, result)
            chart_count += 1
            time.sleep(1)  # Rate limiting
        
        print(f"Generated {chart_count} enhanced charts")
    
    def run_backtest_analysis(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Run backtesting analysis on historical signals"""
        if not self.analysis_results:
            print("No analysis results available for backtesting")
            return None
        
        print("\n🔄 RUNNING BACKTEST ANALYSIS")
        print("=" * 50)
        
        # Setup dates
        if end_date is None:
            end_date_val = datetime.now()
        else:
            end_date_val = end_date
            
        if start_date is None:
            start_date_val = end_date_val - timedelta(days=365)  # 1 year back
        else:
            start_date_val = start_date
        
        # Convert analysis results to backtest format
        signals_data = {}
        for result in self.analysis_results:
            if result['composite_score'] >= 45:  # Only test medium+ signals
                symbol = result['symbol']
                if symbol not in signals_data:
                    signals_data[symbol] = []
                
                signals_data[symbol].append({
                    'date': datetime.now().date(),  # In real implementation, use signal date
                    'signal_data': result
                })
        
        # Setup backtester
        config = BacktestConfig(
            initial_capital=50000,  # 50K
            risk_per_trade=0.01,
            min_score_threshold=45
        )
        backtester = AdvancedBacktester(config)
        
        # Run backtest
        try:
            metrics = backtester.walk_forward_backtest(signals_data, start_date_val, end_date_val)
            
            # Generate and save backtest report
            report = backtester.generate_backtest_report(metrics, save_to_file=True)
            
            # Save backtest results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_file = FILE_CONSTANTS['BACKTESTS_DIR'] / f'backtest_results_{timestamp}.csv'
            
            if 'trade_details' in metrics:
                trade_df = pd.DataFrame(metrics['trade_details'])
                trade_df.to_csv(str(backtest_file), index=False)
                print(f"Backtest results saved: {backtest_file}")
            
            print("\n" + report)
            
            return metrics
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return None
        
    def _estimate_fallback_duration(self, symbol: str, current_price: float, 
                                   indicators: Dict[str, Any], historical_data: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        **FIXED**: Provide fallback duration estimation based on historical volatility
        when no specific target exists. This ensures duration estimates are available
        for all stocks, providing better context for trading decisions.
        """
        try:
            if historical_data is None or len(historical_data) < 20:
                return {
                    'estimated_duration_days': 21,  # Default 3 weeks
                    'confidence': 0.1,
                    'method_breakdown': {'Fallback': {'days': 21, 'confidence': 0.1}},
                    'consensus_level': 'Low',
                    'estimation_method': 'Default - insufficient data'
                }
            
            # Calculate historical volatility
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Estimate typical price move based on volatility
            # Higher volatility = faster moves, so shorter duration estimates
            if volatility > 0.4:  # Very high volatility
                estimated_days = 7  # 1 week
                confidence = 0.3
            elif volatility > 0.3:  # High volatility
                estimated_days = 14  # 2 weeks
                confidence = 0.4
            elif volatility > 0.2:  # Medium volatility
                estimated_days = 21  # 3 weeks
                confidence = 0.5
            elif volatility > 0.15:  # Low-medium volatility
                estimated_days = 30  # 1 month
                confidence = 0.4
            else:  # Low volatility
                estimated_days = 45  # 1.5 months
                confidence = 0.3
            
            # Adjust based on ATR for more precision
            atr = indicators.get('atr', current_price * 0.02)
            atr_pct = atr / current_price if current_price > 0 else 0
            
            if atr_pct > 0.03:  # High ATR (volatile stock)
                estimated_days = max(5, int(estimated_days * 0.8))  # Reduce by 20%
            elif atr_pct < 0.01:  # Low ATR (stable stock)
                estimated_days = int(estimated_days * 1.2)  # Increase by 20%
            
            # Cap at reasonable limits
            estimated_days = min(90, max(3, estimated_days))
            
            return {
                'estimated_duration_days': estimated_days,
                'confidence': round(confidence, 2),
                'method_breakdown': {
                    'Volatility-based': {
                        'days': estimated_days, 
                        'confidence': round(confidence, 2)
                    }
                },
                'consensus_level': 'Medium' if confidence >= 0.4 else 'Low',
                'estimation_method': 'Fallback volatility-based'
            }
            
        except Exception as e:
            return {
                'estimated_duration_days': 21,
                'confidence': 0.1,
                'method_breakdown': {'Error': {'days': 21, 'confidence': 0.1}},
                'consensus_level': 'Low',
                'estimation_method': f'Error: {str(e)}'
            }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Early Warning System for Stock Analysis')
    
    # Input options
    parser.add_argument('-f', '--file', type=str, help='Path to file containing stock symbols')
    parser.add_argument('-s', '--stocks', type=str, help='Comma-separated list of stock symbols')
    
    # Processing options
    parser.add_argument('-b', '--batch-size', type=int, default=50, 
                        help='Batch size for processing (default: 50)')
    parser.add_argument('-t', '--timeout', type=int, default=10, 
                        help='Timeout between batches in seconds (default: 10)')
    
    # Analysis options
    parser.add_argument('--backtest', action='store_true', 
                        help='Run backtesting analysis')
    parser.add_argument('--min-score', type=int, default=45, 
                        help='Minimum score threshold for reporting (default: 45)')
    
    # Output options
    parser.add_argument('-o', '--output-dir', type=str, default='', 
                        help='Directory to save output files')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set output directory if specified
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            os.chdir(args.output_dir)
            print(f"Output directory: {os.path.abspath(args.output_dir)}")
        except Exception as e:
            print(f"Error setting output directory: {e}")
    
    # Create custom stock list if provided
    custom_stocks = None
    if args.stocks:
        custom_stocks = [s.strip() for s in args.stocks.split(',') if s.strip()]
        # No need to add .NS suffix here as it's handled in _load_stock_symbols
    
    try:
        # Initialize enhanced system
        ews = EnhancedEarlyWarningSystem(
            custom_stocks=custom_stocks,
            input_file=args.file,
            batch_size=args.batch_size,
            timeout=args.timeout
        )
        
        # Run analysis
        results = ews.run_enhanced_analysis()
        
        # Run backtesting if requested
        if args.backtest:
            ews.run_backtest_analysis()
        
        print("\n✅ Enhanced Early Warning System analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
