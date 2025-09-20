import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import logging

# Use enhanced data ingestion layer for robust data fetching
from .data.compat import enhanced_yfinance as yf

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import path utilities for cross-platform compatibility
from .common.paths import PathManager, get_data_path, get_output_path, ensure_dir

# Import shared enums from centralized location
from common.enums import MarketRegime

# Import our enhanced modules
from advanced_indicators import AdvancedIndicator
# We'll import CompositeScorer at runtime to avoid circular import issues
# from composite_scorer import CompositeScorer
from advanced_backtester import AdvancedBacktester, BacktestConfig
from risk_manager import RiskManager, RiskConfig

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedEarlyWarningSystem:
    """Enhanced Early Warning System with advanced technical analysis"""
    
    def __init__(self, custom_stocks: Optional[List[str]] = None, 
                 input_file: Optional[str] = None,
                 batch_size: int = 50, timeout: int = 10):
        """
        Initialize Enhanced Early Warning System
        
        Args:
            custom_stocks: List of custom stock symbols
            input_file: Path to file containing stock symbols
            batch_size: Number of stocks to process in each batch
            timeout: Seconds to wait between batches
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.market_regime = MarketRegime.SIDEWAYS
        self.analysis_results = []
        
        # Initialize output directories
        self.output_dirs = self._setup_output_directories()
        
        # Load stock symbols
        self.nse_stocks = self._load_stock_symbols(custom_stocks, input_file)
        
        # Initialize enhanced engines
        self.indicators_engine = AdvancedIndicator()
        
        # Import CompositeScorer here to avoid circular imports
        from composite_scorer import CompositeScorer
        self.scorer = CompositeScorer()
        
        # Create risk config
        risk_config = RiskConfig(
            max_portfolio_risk=0.02,  # 2% portfolio risk
            max_position_size=0.005,  # 0.5% per position
            max_daily_loss=0.01,  # 1% max daily loss
            max_concurrent_positions=10
        )
        
        # Initialize risk manager with config
        self.risk_manager = RiskManager(
            initial_capital=1000000,  # 10 Lakh initial capital
            config=risk_config
        )
        
        print(f"✅ Enhanced Early Warning System initialized")
        print(f"📊 Loaded {len(self.nse_stocks)} stocks for analysis")
        print(f"🔧 Batch size: {self.batch_size}, Timeout: {self.timeout}s")
    
    def _setup_output_directories(self) -> Dict[str, str]:
        """Setup output directories for reports and charts"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        
        output_dirs = {
            'reports': os.path.join(parent_dir, 'output', 'reports'),
            'charts': os.path.join(parent_dir, 'output', 'charts'),
            'backtests': os.path.join(parent_dir, 'output', 'backtests')
        }
        
        # Create directories if they don't exist
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return output_dirs
    
    def _load_stock_symbols(self, custom_stocks: Optional[List[str]], 
                           input_file: Optional[str]) -> List[str]:
        """Load stock symbols from various sources"""
        if custom_stocks:
            print(f"Using custom stock list: {len(custom_stocks)} stocks")
            # Ensure all custom stocks have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in custom_stocks]
            return stocks
        
        if input_file and os.path.exists(input_file):
            print(f"Loading stocks from file: {input_file}")
            with open(input_file, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
            # Ensure all stocks from file have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in stocks]
            return stocks
        
        # Default: load from NSE symbols file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        default_file = os.path.join(parent_dir, 'data', 'nse_only_symbols.txt')
        
        if os.path.exists(default_file):
            print(f"Loading stocks from default file: {default_file}")
            with open(default_file, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
            # Ensure all stocks from default file have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in stocks]
            return stocks[:35]  # Limit to first 35 for testing
        
        # Fallback: sample stocks
        print("Using fallback sample stocks")
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime using NIFTY data"""
        try:
            logger.info("Detecting market regime using NIFTY data")
            print("🌍 Detecting market regime...")
            
            # Fetch NIFTY data using enhanced fetcher
            nifty = yf.Ticker("^NSEI")
            data = nifty.history(period="3mo", auto_adjust=True)
            
            if data.empty:
                logger.warning("Could not fetch NIFTY data, using SIDEWAYS regime")
                print("⚠️ Could not fetch NIFTY data, using SIDEWAYS regime")
                return MarketRegime.SIDEWAYS
            
            logger.debug(f"Fetched {len(data)} NIFTY data points for regime detection")
            
            # Calculate regime indicators
            close = data['Close']
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trend (20-day vs 50-day MA)
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # Determine regime
            if volatility > 0.25:  # 25% annual volatility threshold
                regime = MarketRegime.HIGH_VOLATILITY
            elif current_price > ma20 > ma50:
                regime = MarketRegime.BULLISH
            elif current_price < ma20 < ma50:
                regime = MarketRegime.BEARISH
            else:
                regime = MarketRegime.SIDEWAYS
            
            print(f"📈 Market regime detected: {regime.value.upper()}")
            print(f"📊 NIFTY volatility: {volatility:.1%}")
            return regime
            
        except Exception as e:
            print(f"⚠️ Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS
            
    def analyze_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock with full indicator suite"""
        try:
            print(f"Analyzing {symbol}...")
            
            # Get all technical indicators
            indicators = self.indicators_engine.compute_all_indicators(symbol)
            
            if indicators is None:
                print(f"Failed to compute indicators for {symbol}")
                return None
            
            # Add debugging info
            print(f"Market regime: {self.market_regime}, type: {type(self.market_regime)}")
            
            # Compute composite score using the centralized enum
            scoring_result = self.scorer.compute_composite_score(indicators, self.market_regime)
            
            if scoring_result is None:
                print(f"Failed to compute score for {symbol}")
                return None
            
            # Check risk management constraints
            entry_price = indicators.get('current_price', 0)
            atr = indicators.get('atr', entry_price * 0.02)  # Default 2% if no ATR
            stop_loss = entry_price - (2.0 * atr)  # 2x ATR stop
            
            can_enter, risk_reason, quantity, risk_amount = self.risk_manager.can_enter_position(
                symbol, entry_price, stop_loss, scoring_result['composite_score']
            )
            
            # Add risk management info to result
            scoring_result['risk_management'] = {
                'can_enter_position': can_enter,
                'risk_reason': risk_reason,
                'suggested_quantity': quantity,
                'risk_amount': round(risk_amount, 2),
                'suggested_stop_loss': round(stop_loss, 2),
                'risk_reward_ratio': round(abs(entry_price - stop_loss) / (entry_price * 0.025), 2)  # Assume 2.5% TP
            }
            
            return scoring_result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
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
            logger.debug(f"Generating enhanced chart for {symbol}")
            print(f"Generating enhanced chart for {symbol}...")
            
            # Fetch data for charting using enhanced fetcher
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo", auto_adjust=True)
            
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol} chart: {len(data)} rows")
                print(f"Insufficient data for {symbol} chart")
                return None
            
            logger.debug(f"Fetched {len(data)} data points for {symbol} chart")
            
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
            chart_path = os.path.join(self.output_dirs['charts'], 
                                    f"{symbol.replace('.NS', '')}_enhanced_chart.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced chart saved: {chart_path}")
            return chart_path
            
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
                        'Symbol': result['symbol'].replace('.NS', ''),
                        'Composite_Score': result['composite_score'],
                        'Probability': result['probability_level'],
                        'Current_Price': result['key_indicators']['current_price'],
                        'Price_Change_%': result['key_indicators']['price_change_pct'],
                        'Volume_Ratio': result['key_indicators']['volume_ratio'],
                        'Volume_Z_Score': result['key_indicators']['volume_z_score'],
                        'RSI': result['key_indicators']['rsi'],
                        'MACD_Signal': result['key_indicators']['macd_signal'],
                        'ADX': result['key_indicators']['adx'],
                        'ATR_%': result['key_indicators']['atr_pct'],
                        'Rel_Strength_20d': result['key_indicators']['relative_strength_20d'],
                        'Can_Enter': result['risk_management']['can_enter_position'],
                        'Suggested_Qty': result['risk_management']['suggested_quantity'],
                        'Risk_Amount': result['risk_management']['risk_amount'],
                        'Stop_Loss': result['risk_management']['suggested_stop_loss']
                    }
                    for result in categorized_results['HIGH']
                ])
                
                high_file = os.path.join(self.output_dirs['reports'], 
                                       f'high_probability_enhanced_{timestamp}.csv')
                high_prob_df.to_csv(high_file, index=False)
                saved_files.append(high_file)
                print(f"High probability report saved: {high_file}")
            
            # 2. All results comprehensive report
            all_results = []
            for level, results in categorized_results.items():
                all_results.extend(results)
            
            if all_results:
                comprehensive_df = pd.DataFrame([
                    {
                        'Symbol': result['symbol'].replace('.NS', ''),
                        'Composite_Score': result['composite_score'],
                        'Probability_Level': result['probability_level'],
                        'Market_Regime': result['market_regime'],
                        'Volume_Score': result['component_scores']['volume'],
                        'Momentum_Score': result['component_scores']['momentum'],
                        'Trend_Score': result['component_scores']['trend'],
                        'Volatility_Score': result['component_scores']['volatility'],
                        'RelStrength_Score': result['component_scores']['relative_strength'],
                        'VolumeProfile_Score': result['component_scores']['volume_profile'],
                        'Weekly_Confirmation': result['component_scores']['weekly_confirmation'],
                        'Current_Price': result['key_indicators']['current_price'],
                        'Price_Change_%': result['key_indicators']['price_change_pct'],
                        'Volume_Ratio': result['key_indicators']['volume_ratio'],
                        'Volume_Z_Score': result['key_indicators']['volume_z_score'],
                        'RSI': result['key_indicators']['rsi'],
                        'MACD_Signal': result['key_indicators']['macd_signal'],
                        'ADX': result['key_indicators']['adx'],
                        'ATR_%': result['key_indicators']['atr_pct'],
                        'Rel_Strength_20d': result['key_indicators']['relative_strength_20d'],
                        'Risk_Approved': result['risk_management']['can_enter_position'],
                        'Risk_Reason': result['risk_management']['risk_reason']
                    }
                    for result in all_results
                ])
                
                comprehensive_file = os.path.join(self.output_dirs['reports'], 
                                                f'comprehensive_analysis_{timestamp}.csv')
                comprehensive_df.to_csv(comprehensive_file, index=False)
                saved_files.append(comprehensive_file)
                print(f"Comprehensive report saved: {comprehensive_file}")
            
            # 3. Analysis summary report
            summary_file = os.path.join(self.output_dirs['reports'], 
                                      f'analysis_summary_{timestamp}.txt')
            
            with open(summary_file, 'w') as f:
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
        self.market_regime = self.detect_market_regime()
        
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
                    'Score': r['composite_score'],
                    'Price': f"₹{r['key_indicators']['current_price']:.1f}",
                    'Change%': f"{r['key_indicators']['price_change_pct']:.1f}%",
                    'Vol_Ratio': f"{r['key_indicators']['volume_ratio']:.1f}x",
                    'RSI': f"{r['key_indicators']['rsi']:.1f}",
                    'Risk_OK': '✅' if r['risk_management']['can_enter_position'] else '❌',
                    'Qty': r['risk_management']['suggested_quantity']
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
    
    def run_backtest_analysis(self, start_date: datetime = None, 
                            end_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Run backtesting analysis on historical signals"""
        if not self.analysis_results:
            print("No analysis results available for backtesting")
            return None
        
        print("\n🔄 RUNNING BACKTEST ANALYSIS")
        print("=" * 50)
        
        # Setup dates
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # 1 year back
        
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
            metrics = backtester.walk_forward_backtest(signals_data, start_date, end_date)
            
            # Generate and save backtest report
            report = backtester.generate_backtest_report(metrics, save_to_file=True)
            
            # Save backtest results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_file = os.path.join(self.output_dirs['backtests'], 
                                       f'backtest_results_{timestamp}.csv')
            
            if 'trade_details' in metrics:
                trade_df = pd.DataFrame(metrics['trade_details'])
                trade_df.to_csv(backtest_file, index=False)
                print(f"Backtest results saved: {backtest_file}")
            
            print("\n" + report)
            
            return metrics
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return None

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