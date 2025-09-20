from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
import os
import sys
import time

from typing import Dict, List, Any, Optional, Tuple
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants for chart formatting
LEGEND_LOCATION_UPPER_RIGHT = 'upper right'

# Import our logging and monitoring infrastructure first
try:
    from .logging_config import setup_logging, get_logger, operation_context
    from .stock_analysis_monitor import monitor, start_batch_analysis, end_batch_analysis, track_symbol_analysis
    from .robust_data_fetcher import fetch_stock_data, fetch_multiple_stocks
except ImportError:
    # Fallback for when running as standalone
    def setup_logging(**kwargs): pass
    def get_logger(name): return logging.getLogger(name)
    def operation_context(**kwargs):
        from contextlib import nullcontext
        return nullcontext()
    class MockMonitor:
        def record_data_fetch(self, *args, **kwargs): 
            # Mock implementation for monitoring data fetch operations
            pass
        def record_indicator_computation(self, *args, **kwargs): 
            # Mock implementation for monitoring indicator calculations
            pass
        def record_scoring(self, *args, **kwargs): 
            # Mock implementation for monitoring scoring operations
            pass
        def complete_symbol_analysis(self, *args, **kwargs): 
            # Mock implementation for monitoring symbol analysis completion
            pass
    monitor = MockMonitor()
    def start_batch_analysis(): return "mock_session"
    def end_batch_analysis(): return {}
    def track_symbol_analysis():
        from contextlib import nullcontext
        return nullcontext()
    def fetch_stock_data(symbol, **kwargs):
        import yfinance as yf
        return yf.Ticker(symbol).history(**kwargs)
    def fetch_multiple_stocks(symbols, **kwargs):
        return {s: fetch_stock_data(s, **kwargs) for s in symbols}

# Use enhanced data ingestion layer for robust data fetching
from .data.compat import enhanced_yfinance as yf
# Use new modular data fetchers
from .data import YahooDataFetcher, DataFetcherFactory

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import path utilities for cross-platform compatibility
from .common.paths import PathManager, get_data_path, get_output_path, ensure_dir

# Import shared enums from centralized location
try:
    from .common.enums import MarketRegime
    from .common.interfaces import IDataFetcher, IIndicator, IScorer
except ImportError:
    # Fallback for direct execution
    from src.common.enums import MarketRegime
    from src.common.interfaces import IDataFetcher, IIndicator, IScorer

# Import our enhanced modules
from advanced_indicators import AdvancedIndicator
# Use new modular indicators
from .indicators import IndicatorEngine
# We'll import CompositeScorer at runtime to avoid circular import issues
# from composite_scorer import CompositeScorer
from advanced_backtester import AdvancedBacktester
from risk_manager import RiskManager

# Import centralized configuration
from .config import SystemConfig, get_config
from .common.config import ConfigManager

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedEarlyWarningSystem:
    """Enhanced Early Warning System with advanced technical analysis"""
    
    def __init__(self, custom_stocks: Optional[List[str]] = None, 
                 input_file: Optional[str] = None,
                 batch_size: int = 50, timeout: int = 10,
                 config: Optional[SystemConfig] = None):
        """
        Initialize Enhanced Early Warning System
        
        Args:
            custom_stocks: List of custom stock symbols
            input_file: Path to file containing stock symbols
            batch_size: Number of stocks to process in each batch
            timeout: Seconds to wait between batches
            config: System configuration (optional, uses defaults if not provided)
        """
        # Use provided config or get default
        self.config = config or get_config()
        
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
        
        # Initialize risk manager with centralized config
        self.risk_manager = RiskManager(
            initial_capital=self.config.portfolio_capital,
            config=self.config
        )
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        self.logger.info("Enhanced Early Warning System initialized", extra={
            'stock_count': len(self.nse_stocks),
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'portfolio_capital': self.config.portfolio_capital
        })
    
    def _setup_output_directories(self) -> Dict[str, str]:
        """Setup output directories for reports and charts"""
        base_dir = Path(__file__).parent
        parent_dir = base_dir.parent
        
        output_dirs = {
            'reports': str(parent_dir / 'output' / 'reports'),
            'charts': str(parent_dir / 'output' / 'charts'),
            'backtests': str(parent_dir / 'output' / 'backtests')
        }
        
        # Create directories if they don't exist
        for dir_path in output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return output_dirs
    
    def _load_stock_symbols(self, custom_stocks: Optional[List[str]], 
                           input_file: Optional[str]) -> List[str]:
        """Load stock symbols from various sources"""
        if custom_stocks:
            logger.info(f"Using custom stock list: {len(custom_stocks)} stocks", 
                       extra={'source': 'custom', 'count': len(custom_stocks)})
            # Ensure all custom stocks have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in custom_stocks]
            return stocks
        
        if input_file and Path(input_file).exists():
            logger.info(f"Loading stocks from file: {input_file}", 
                       extra={'source': 'file', 'file_path': input_file})
            with open(input_file, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
            # Ensure all stocks from file have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in stocks]
            return stocks
        
        # Default: load from NSE symbols file
        base_dir = Path(__file__).parent
        parent_dir = base_dir.parent
        default_file = parent_dir / 'data' / 'nse_only_symbols.txt'
        
        if default_file.exists():
            logger.info(f"Loading stocks from default file: {default_file}", 
                       extra={'source': 'default_file', 'file_path': str(default_file)})
            with open(default_file, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
            # Ensure all stocks from default file have .NS suffix
            stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in stocks]
            return stocks[:35]  # Limit to first 35 for testing
        
        # Fallback: sample stocks
        logger.warning("Using fallback sample stocks", extra={'source': 'fallback'})
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime using NIFTY data"""
        try:
            logger.info("🌍 Detecting market regime using NIFTY data", 
                       extra={'operation': 'market_regime_detection'})
            
            # Fetch NIFTY data using enhanced fetcher
            nifty = yf.Ticker("^NSEI")
            data = nifty.history(period="3mo", auto_adjust=True)
            
            if data.empty:
                logger.warning("Could not fetch NIFTY data, using SIDEWAYS regime", 
                             extra={'fallback_regime': 'SIDEWAYS'})
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
            
            logger.info(f"📈 Market regime detected: {regime.value.upper()}", 
                       extra={'regime': regime.value, 'volatility': volatility, 
                             'current_price': current_price, 'ma20': ma20, 'ma50': ma50})
            return regime
            
        except Exception as e:
            logger.error(f"⚠️ Error detecting market regime: {e}", 
                        extra={'error_type': type(e).__name__}, exc_info=True)
            return MarketRegime.SIDEWAYS
            
    def analyze_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock with full indicator suite and comprehensive error handling"""
        
        with track_symbol_analysis(symbol):
            try:
                self.logger.debug(f"Starting analysis for {symbol}", extra={'symbol': symbol})
                
                # Get all technical indicators with timing
                indicator_start = time.time()
                indicators = self.indicators_engine.compute_all_indicators(symbol)
                indicator_duration = time.time() - indicator_start
                
                if indicators is None:
                    self.logger.warning(f"Failed to compute indicators for {symbol}", 
                                      extra={'symbol': symbol, 'duration': indicator_duration})
                    return None
                
                self.logger.debug(f"Computed indicators for {symbol}", extra={
                    'symbol': symbol,
                    'indicator_count': len(indicators),
                    'duration': indicator_duration
                })
                
                # Compute composite score using the centralized enum
                scoring_start = time.time()
                scoring_result = self.scorer.compute_composite_score(indicators, self.market_regime)
                scoring_duration = time.time() - scoring_start
                
                if scoring_result is None:
                    self.logger.warning(f"Failed to compute score for {symbol}", 
                                      extra={'symbol': symbol, 'duration': scoring_duration})
                    monitor.record_scoring(symbol, scoring_duration, False)
                    return None
                
                # Record successful scoring
                composite_score = scoring_result.get('composite_score', 0)
                monitor.record_scoring(symbol, scoring_duration, True, composite_score)
                
                # Check risk management constraints
                entry_price = indicators.get('current_price', 0)
                atr = indicators.get('atr', entry_price * 0.02)  # Default 2% if no ATR
                stop_loss = entry_price - (2.0 * atr)  # 2x ATR stop
                
                can_enter, risk_reason, quantity, risk_amount = self.risk_manager.can_enter_position(
                    symbol, entry_price, stop_loss, composite_score
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
                
                self.logger.info(f"Successfully analyzed {symbol}", extra={
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'can_enter_position': can_enter,
                    'indicator_duration': indicator_duration,
                    'scoring_duration': scoring_duration
                })
                
                return scoring_result
                
            except Exception as e:
                error_type = type(e).__name__
                self.logger.error(f"Error analyzing {symbol}: {e}", extra={
                    'symbol': symbol,
                    'error_type': error_type,
                    'market_regime': self.market_regime.value if hasattr(self.market_regime, 'value') else str(self.market_regime)
                }, exc_info=True)
                
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
            logger.debug(f"Generating enhanced chart for {symbol}", 
                        extra={'symbol': symbol, 'operation': 'chart_generation'})
            
            # Fetch data for charting using enhanced fetcher
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo", auto_adjust=True)
            
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol} chart: {len(data)} rows", 
                             extra={'symbol': symbol, 'data_points': len(data)})
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
            ax2.legend(loc=LEGEND_LOCATION_UPPER_RIGHT, fontsize=8)
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
            ax3.legend(loc=LEGEND_LOCATION_UPPER_RIGHT, fontsize=8)
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
            ax4.legend(loc=LEGEND_LOCATION_UPPER_RIGHT, fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = Path(self.output_dirs['charts']) / f"{symbol.replace('.NS', '')}_enhanced_chart.png"
            plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Enhanced chart saved: {chart_path}", 
                       extra={'symbol': symbol, 'chart_path': str(chart_path)})
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating enhanced chart for {symbol}: {e}", 
                        extra={'symbol': symbol, 'error_type': type(e).__name__}, 
                        exc_info=True)
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
                
                high_file = Path(self.output_dirs['reports']) / f'high_probability_enhanced_{timestamp}.csv'
                high_prob_df.to_csv(str(high_file), index=False)
                saved_files.append(str(high_file))
                logger.info(f"High probability report saved: {high_file}", 
                           extra={'report_type': 'high_probability', 'file_path': str(high_file)})
            
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
                
                comprehensive_file = Path(self.output_dirs['reports']) / f'comprehensive_analysis_{timestamp}.csv'
                comprehensive_df.to_csv(str(comprehensive_file), index=False)
                saved_files.append(str(comprehensive_file))
                logger.info(f"Comprehensive report saved: {comprehensive_file}", 
                           extra={'report_type': 'comprehensive', 'file_path': str(comprehensive_file)})
            
            # 3. Analysis summary report
            summary_file = Path(self.output_dirs['reports']) / f'analysis_summary_{timestamp}.txt'
            
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
            
            saved_files.append(str(summary_file))
            logger.info(f"Summary report saved: {summary_file}", 
                       extra={'report_type': 'summary', 'file_path': str(summary_file)})
            
        except Exception as e:
            logger.error(f"Error saving reports: {e}", 
                        extra={'error_type': type(e).__name__}, exc_info=True)
        
        return saved_files
    
    def run_enhanced_analysis(self) -> Dict[str, Any]:
        """Run the complete enhanced analysis pipeline"""
        logger.info("🚀 ENHANCED EARLY WARNING SYSTEM", 
                   extra={'operation': 'enhanced_analysis_start', 'stock_count': len(self.nse_stocks)})
        logger.info(f"Analyzing {len(self.nse_stocks)} stocks with advanced indicators", 
                   extra={'total_stocks': len(self.nse_stocks)})
        
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
            
            logger.info(f"Processing batch {batch_num+1}/{total_batches} ({len(batch)} stocks)", 
                       extra={'batch_num': batch_num + 1, 'total_batches': total_batches, 
                             'batch_size': len(batch), 'stocks_range': f"{start_idx+1}-{end_idx}"})
            
            for i, symbol in enumerate(batch):
                progress = (i + 1) / len(batch) * 100
                logger.debug(f"[{progress:.1f}%] Analyzing {symbol}", 
                           extra={'symbol': symbol, 'progress': progress, 'batch_position': i + 1})
                
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
                logger.info(f"Pausing for {self.timeout} seconds between batches", 
                           extra={'timeout_seconds': self.timeout, 'batch_completed': batch_num + 1})
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
        
        logger.info("📊 ANALYSIS COMPLETE", 
                   extra={'total_successful': len(all_results), 'total_stocks': len(self.nse_stocks), 
                         'reports_saved': len(saved_files), 'operation': 'analysis_complete'})
        
        return {
            'categorized_results': categorized_results,
            'analysis_summary': analysis_summary,
            'saved_files': saved_files
        }
    
    def display_analysis_results(self, categorized_results: Dict[str, List[Dict]], 
                               analysis_summary: Dict[str, Any]):
        """Display formatted analysis results"""
        logger.info("📈 ENHANCED ANALYSIS RESULTS", 
                   extra={'operation': 'results_display', 'market_regime': analysis_summary['market_regime']})
        
        logger.info(f"🌍 Market Regime: {analysis_summary['market_regime'].upper()}", 
                   extra={'market_regime': analysis_summary['market_regime']})
        logger.info(f"📊 Analysis Success Rate: {analysis_summary['success_rate']:.1f}%", 
                   extra={'success_rate': analysis_summary['success_rate']})
        logger.info(f"✅ Risk Approved Positions: {analysis_summary['risk_approved']} "
                   f"({analysis_summary['risk_approval_rate']:.1f}%)", 
                   extra={'risk_approved': analysis_summary['risk_approved'], 
                         'risk_approval_rate': analysis_summary['risk_approval_rate']})
        
        # High probability stocks
        if categorized_results['HIGH']:
            logger.info("🎯 HIGH PROBABILITY STOCKS (Score ≥ 70)", 
                       extra={'high_probability_count': len(categorized_results['HIGH'])})
            
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
            
            logger.info("High probability stocks data", 
                       extra={'high_stocks_table': high_df.to_string(index=False)})
        else:
            logger.warning("❌ No HIGH probability stocks found today.", 
                          extra={'high_probability_count': 0})
        
        # Medium probability stocks (top 10)
        if categorized_results['MEDIUM']:
            logger.info("📊 TOP MEDIUM PROBABILITY STOCKS (Score 45-69)", 
                       extra={'medium_probability_count': len(categorized_results['MEDIUM'])})
            
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
            
            logger.info("Medium probability stocks data", 
                       extra={'medium_stocks_table': medium_df.to_string(index=False)})
        
        logger.info(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   extra={'completion_time': datetime.now().isoformat()})
        logger.warning("⚠️  This is probability-based analysis. Always do your own research!", 
                      extra={'disclaimer': True})
    
    def generate_charts_for_top_stocks(self, categorized_results: Dict[str, List[Dict]]):
        """Generate enhanced charts for top-scoring stocks"""
        logger.info("📈 Generating enhanced charts for top stocks", 
                   extra={'operation': 'chart_generation', 'max_charts': 10})
        
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
        
        logger.info(f"Generated {chart_count} enhanced charts", 
                   extra={'charts_generated': chart_count, 'operation': 'chart_generation_complete'})
    
    def run_backtest_analysis(self, start_date: datetime = None, 
                            end_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Run backtesting analysis on historical signals"""
        if not self.analysis_results:
            logger.warning("No analysis results available for backtesting", 
                          extra={'operation': 'backtest_analysis'})
            return None
        
        logger.info("🔄 RUNNING BACKTEST ANALYSIS", 
                   extra={'operation': 'backtest_start', 'results_count': len(self.analysis_results)})
        
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
        backtest_config = SystemConfig(
            portfolio_capital=50000,  # 50K
            risk_per_trade=0.01,
            min_score_threshold=45
        )
        backtester = AdvancedBacktester(backtest_config)
        
        # Run backtest
        try:
            metrics = backtester.walk_forward_backtest(signals_data, start_date, end_date)
            
            # Generate and save backtest report
            report = backtester.generate_backtest_report(metrics, save_to_file=True)
            
            # Save backtest results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_file = Path(self.output_dirs['backtests']) / f'backtest_results_{timestamp}.csv'
            
            if 'trade_details' in metrics:
                trade_df = pd.DataFrame(metrics['trade_details'])
                trade_df.to_csv(str(backtest_file), index=False)
                logger.info(f"Backtest results saved: {backtest_file}", 
                           extra={'backtest_file': str(backtest_file), 'trade_count': len(trade_df)})
            
            logger.info("Backtest report generated", 
                       extra={'report_content': report, 'operation': 'backtest_complete'})
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}", 
                        extra={'error_type': type(e).__name__}, exc_info=True)
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
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            os.chdir(args.output_dir)
            output_path = Path(args.output_dir).resolve()
            logger.info(f"Output directory: {output_path}", 
                       extra={'output_directory': str(output_path)})
        except Exception as e:
            logger.error(f"Error setting output directory: {e}", 
                        extra={'error_type': type(e).__name__}, exc_info=True)
    
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
        
        logger.info("✅ Enhanced Early Warning System analysis completed successfully!", 
                   extra={'operation': 'main_analysis_complete'})
        
    except KeyboardInterrupt:
        logger.warning("⏹️ Analysis interrupted by user", 
                      extra={'interruption': 'keyboard_interrupt'})
    except Exception as e:
        logger.error(f"❌ Error running analysis: {e}", 
                    extra={'error_type': type(e).__name__}, exc_info=True)
        import traceback
        traceback.print_exc()