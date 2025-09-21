#!/usr/bin/env python3
"""
Enhanced Command Line Interface for NSE Stock Screener
Provides advanced CLI capabilities for power users and automation
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with comprehensive options"""
    parser = argparse.ArgumentParser(
        description='NSE Stock Screener - Enhanced Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with top 10 stocks
  python -m src.enhanced_cli --top 10

  # Analyze specific stocks
  python -m src.enhanced_cli -s "RELIANCE,TCS,INFY" --output-format json

  # Banking sector analysis with intelligent ranking
  python -m src.enhanced_cli --sector banking --intelligent --top 5

  # High-volume analysis with custom threshold
  python -m src.enhanced_cli --volume-threshold 1.5 --min-score 60

  # Batch analysis with custom output
  python -m src.enhanced_cli -f symbols.txt --batch-size 20 --output results.csv

  # Backtesting with specific date range
  python -m src.enhanced_cli --backtest --start-date 2024-01-01 --end-date 2024-12-31

  # Quick market overview
  python -m src.enhanced_cli --market-overview --regime-analysis
        """
    )

    # Input Selection Group
    input_group = parser.add_argument_group('Input Selection')
    input_group.add_argument('-s', '--stocks', type=str,
                            help='Comma-separated list of stock symbols (e.g., "RELIANCE,TCS,INFY")')
    input_group.add_argument('-f', '--file', type=str,
                            help='Path to file containing stock symbols (one per line)')
    input_group.add_argument('--sector', type=str, choices=['banking', 'technology', 'auto', 'pharma', 'fmcg'],
                            help='Analyze specific sector stocks')
    input_group.add_argument('--top', type=int, metavar='N',
                            help='Analyze top N stocks from intelligent ranking')
    input_group.add_argument('--random', type=int, metavar='N',
                            help='Analyze N random stocks from NSE list')

    # Analysis Configuration Group
    analysis_group = parser.add_argument_group('Analysis Configuration')
    analysis_group.add_argument('--intelligent', action='store_true',
                               help='Use intelligent ranking with technical analysis')
    analysis_group.add_argument('--min-score', type=int, default=45, metavar='SCORE',
                               help='Minimum score threshold for reporting (default: 45)')
    analysis_group.add_argument('--volume-threshold', type=float, default=1.0, metavar='RATIO',
                               help='Volume threshold ratio (default: 1.0)')
    analysis_group.add_argument('--regime-analysis', action='store_true',
                               help='Include market regime analysis')
    analysis_group.add_argument('--risk-analysis', action='store_true',
                               help='Include detailed risk analysis')

    # Processing Options Group
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('-b', '--batch-size', type=int, default=50, metavar='SIZE',
                                 help='Batch size for processing (default: 50)')
    processing_group.add_argument('-t', '--timeout', type=int, default=5, metavar='SECONDS',
                                 help='Timeout between batches in seconds (default: 5)')
    processing_group.add_argument('--parallel', type=int, metavar='THREADS',
                                 help='Number of parallel threads for analysis')

    # Output Configuration Group
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('-o', '--output', type=str, metavar='PATH',
                             help='Output file path (default: timestamped file in output/reports/)')
    output_group.add_argument('--output-format', type=str, choices=['csv', 'json', 'excel', 'text'],
                             default='csv', help='Output format (default: csv)')
    output_group.add_argument('--display-console', action='store_true',
                             help='Display comprehensive results table in console')
    output_group.add_argument('--charts', action='store_true',
                             help='Generate technical analysis charts')
    output_group.add_argument('--chart-format', type=str, choices=['png', 'pdf', 'svg'],
                             default='png', help='Chart format (default: png)')

    # Backtesting Group
    backtest_group = parser.add_argument_group('Backtesting Options')
    backtest_group.add_argument('--backtest', action='store_true',
                               help='Run backtesting analysis')
    backtest_group.add_argument('--start-date', type=str, metavar='YYYY-MM-DD',
                               help='Backtest start date (default: 1 year ago)')
    backtest_group.add_argument('--end-date', type=str, metavar='YYYY-MM-DD',
                               help='Backtest end date (default: today)')
    backtest_group.add_argument('--initial-capital', type=float, default=100000, metavar='AMOUNT',
                               help='Initial capital for backtesting (default: 100000)')

    # Special Operations Group
    special_group = parser.add_argument_group('Special Operations')
    special_group.add_argument('--market-overview', action='store_true',
                              help='Generate market overview report')
    special_group.add_argument('--update-symbols', action='store_true',
                              help='Update NSE symbols list from live data')
    special_group.add_argument('--validate-setup', action='store_true',
                              help='Validate system setup and dependencies')
    special_group.add_argument('--benchmark', action='store_true',
                              help='Run performance benchmark tests')

    # Verbosity and Logging Group
    log_group = parser.add_argument_group('Logging and Output')
    log_group.add_argument('-v', '--verbose', action='count', default=0,
                          help='Increase verbosity (-v, -vv, -vvv for more detail)')
    log_group.add_argument('-q', '--quiet', action='store_true',
                          help='Suppress non-essential output')
    log_group.add_argument('--log-file', type=str, metavar='PATH',
                          help='Log file path (default: no file logging)')
    log_group.add_argument('--no-color', action='store_true',
                          help='Disable colored output')

    return parser

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate argument combinations and requirements"""
    errors = []

    # Check for special operations that don't need input
    special_operations = [args.validate_setup, args.update_symbols, args.benchmark]
    if any(special_operations):
        # Special operations don't need input validation
        return

    # Check for mutually exclusive input options
    input_options = [args.stocks, args.file, args.sector, args.top, args.random]
    input_count = sum(1 for opt in input_options if opt is not None)
    
    if input_count == 0:
        errors.append("Must specify at least one input option: --stocks, --file, --sector, --top, or --random")
    elif input_count > 1:
        errors.append("Cannot specify multiple input options simultaneously")

    # Validate date format if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            errors.append("Invalid start date format. Use YYYY-MM-DD")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            errors.append("Invalid end date format. Use YYYY-MM-DD")

    # Validate numeric ranges
    if args.min_score < 0 or args.min_score > 100:
        errors.append("Minimum score must be between 0 and 100")

    if args.volume_threshold < 0:
        errors.append("Volume threshold must be positive")

    if args.batch_size < 1:
        errors.append("Batch size must be at least 1")

    if args.timeout < 0:
        errors.append("Timeout must be non-negative")

    # Check file existence
    if args.file and not Path(args.file).exists():
        errors.append(f"Input file not found: {args.file}")

    if errors:
        print("❌ Validation Errors:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)

def setup_logging_from_args(args: argparse.Namespace) -> logging.Logger:
    """Setup logging based on command line arguments"""
    # Determine log level based on verbosity
    if args.quiet:
        level = logging.WARNING
    elif args.verbose == 0:
        level = logging.INFO
    elif args.verbose == 1:
        level = logging.DEBUG
    else:
        level = logging.DEBUG

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s' if args.verbose > 0 else '%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(args.log_file)] if args.log_file else [])
        ]
    )

    return logging.getLogger(__name__)

def load_stocks_from_input(args: argparse.Namespace, logger: logging.Logger) -> List[str]:
    """Load stock symbols based on input arguments"""
    stocks = []

    try:
        if args.stocks:
            # Direct stock list
            stocks = [s.strip().upper() for s in args.stocks.split(',') if s.strip()]
            logger.info(f"Loaded {len(stocks)} stocks from command line")

        elif args.file:
            # File input
            with open(args.file, 'r') as f:
                stocks = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Loaded {len(stocks)} stocks from file: {args.file}")

        elif args.sector:
            # Sector-based selection
            from src.dynamic_watchlist_manager import DynamicWatchlistManager
            dwm = DynamicWatchlistManager()
            
            # Update data if needed
            stats = dwm.get_stats()
            if stats['cached_sector_data'] < 20:
                logger.info("Updating market data for sector analysis...")
                dwm.update_market_data(sample_size=100)
            
            # Get sector stocks
            if args.intelligent:
                from src.intelligent_watchlist_manager import IntelligentWatchlistManager
                iwm = IntelligentWatchlistManager()
                sector_stocks = iwm._analyze_sector_stocks(args.sector.upper())
                stocks = [item['symbol'] for item in sector_stocks[:args.top or 10]]
            else:
                sector_map = {
                    'banking': ['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK'],
                    'technology': ['TCS', 'INFY', 'WIPRO', 'HCL', 'TECH'],
                    'auto': ['MARUTI', 'HERO', 'BAJAJ', 'TATA', 'MAHINDRA'],
                    'pharma': ['SUN', 'CIPLA', 'DRREDDY', 'LUPIN', 'BIOCON'],
                    'fmcg': ['ITC', 'HUL', 'NESTLE', 'BRITANNIA', 'DABUR']
                }
                search_terms = sector_map.get(args.sector, [])
                for term in search_terms:
                    matches = dwm.search_symbol(term)
                    stocks.extend(matches[:2])
                stocks = list(dict.fromkeys(stocks))  # Remove duplicates
            
            logger.info(f"Loaded {len(stocks)} stocks from {args.sector} sector")

        elif args.top:
            # Top N intelligent ranking
            if args.intelligent:
                from src.intelligent_watchlist_manager import IntelligentWatchlistManager
                iwm = IntelligentWatchlistManager()
                
                # Update data if needed
                stats = iwm.get_stats()
                if stats['cached_sector_data'] < 50:
                    logger.info("Updating market data for intelligent ranking...")
                    iwm.update_market_data(sample_size=150)
                
                # Get top stocks with analytical ranking
                sector_data = iwm.get_intelligent_sector_watchlists(max_stocks_per_sector=args.top)
                all_stocks = []
                for sector_stocks in sector_data.values():
                    all_stocks.extend(sector_stocks)
                # Sort by score and take top N
                sorted_stocks = sorted(all_stocks, key=lambda x: x.get('composite_score', 0), reverse=True)
                stocks = [item['symbol'] for item in sorted_stocks[:args.top]]
                logger.info(f"Loaded top {len(stocks)} stocks using intelligent ranking")
            else:
                # Fallback to popular stocks
                popular_stocks = [
                    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                    "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS"
                ]
                stocks = popular_stocks[:args.top]
                logger.info(f"Loaded top {len(stocks)} popular stocks (fallback)")

        elif args.random:
            # Random selection
            from src.dynamic_watchlist_manager import DynamicWatchlistManager
            dwm = DynamicWatchlistManager()
            
            # Use the internal all_symbols list
            all_symbols = dwm.all_symbols
            import random
            stocks = random.sample(all_symbols, min(args.random, len(all_symbols)))
            logger.info(f"Loaded {len(stocks)} random stocks")

    except Exception as e:
        logger.error(f"Error loading stocks: {e}")
        return []

    # Ensure .NS suffix
    stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in stocks]
    return stocks

def run_analysis_with_args(args: argparse.Namespace, stocks: List[str], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Run analysis based on command line arguments"""
    try:
        from src.enhanced_early_warning_system import EnhancedEarlyWarningSystem
        
        logger.info(f"Starting analysis of {len(stocks)} stocks...")
        
        # Initialize system
        ews = EnhancedEarlyWarningSystem(
            custom_stocks=stocks,
            batch_size=args.batch_size,
            timeout=args.timeout
        )
        
        # Run main analysis
        results = ews.run_enhanced_analysis()
        
        # Generate market overview if requested
        if args.market_overview:
            logger.info("Generating market overview...")
            market_stats = generate_market_overview(stocks, logger)
            results['market_overview'] = market_stats
        
        # Run backtesting if requested
        if args.backtest:
            logger.info("Running backtesting analysis...")
            backtest_results = ews.run_backtest_analysis()
            results['backtest'] = backtest_results
        
        logger.info("✅ Analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None

def generate_market_overview(stocks: List[str], logger: logging.Logger) -> Dict[str, Any]:
    """Generate comprehensive market overview"""
    try:
        from src.composite_scorer import CompositeScorer
        
        logger.info("Analyzing market conditions...")
        
        overview = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks_analyzed': len(stocks),
            'market_sentiment': 'NEUTRAL',
            'regime': 'SIDEWAYS',
            'top_performers': [],
            'risk_alerts': []
        }
        
        # Analyze NIFTY for market regime
        try:
            import yfinance as yf
            nifty_data = yf.Ticker("^NSEI").history(period="3mo")
            if not nifty_data.empty:
                recent_return = ((nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-21]) - 1) * 100
                if recent_return > 3:
                    overview['regime'] = 'BULLISH'
                    overview['market_sentiment'] = 'POSITIVE'
                elif recent_return < -3:
                    overview['regime'] = 'BEARISH'
                    overview['market_sentiment'] = 'NEGATIVE'
        except Exception as e:
            logger.warning(f"Could not analyze NIFTY: {e}")
        
        logger.info(f"Market regime detected: {overview['regime']}")
        return overview
        
    except Exception as e:
        logger.error(f"Error generating market overview: {e}")
        return {}

def save_results(results: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> None:
    """Save analysis results in specified format"""
    try:
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"cli_analysis_{timestamp}.{args.output_format}"
        
        # Save based on format
        if args.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif args.output_format == 'csv':
            # Convert to comprehensive DataFrame for CSV with all columns
            if 'categorized_results' in results:
                # Extract all results from all categories
                all_analysis_results = []
                for level, stock_results in results['categorized_results'].items():
                    all_analysis_results.extend(stock_results)
                
                if all_analysis_results:
                    # Create comprehensive DataFrame with all columns like enhanced_early_warning_system
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
                        for result in all_analysis_results
                    ])
                    comprehensive_df.to_csv(output_path, index=False)
                else:
                    # Fallback to simple format if no comprehensive data
                    if 'summary' in results:
                        df = pd.DataFrame(results['summary'])
                        df.to_csv(output_path, index=False)
        
        elif args.output_format == 'excel':
            # Convert to comprehensive DataFrame for Excel with all columns
            if 'categorized_results' in results:
                # Extract all results from all categories
                all_analysis_results = []
                for level, stock_results in results['categorized_results'].items():
                    all_analysis_results.extend(stock_results)
                
                if all_analysis_results:
                    # Create comprehensive DataFrame with all columns like enhanced_early_warning_system
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
                        for result in all_analysis_results
                    ])
                    comprehensive_df.to_excel(output_path, index=False)
                else:
                    # Fallback to simple format if no comprehensive data
                    if 'summary' in results:
                        df = pd.DataFrame(results['summary'])
                        df.to_excel(output_path, index=False)
        
        elif args.output_format == 'text':
            with open(output_path, 'w') as f:
                f.write(f"NSE Stock Screener Analysis Report\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"{'='*50}\n\n")
                f.write(str(results))
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def display_comprehensive_results(results: Dict[str, Any], logger: logging.Logger) -> None:
    """Display comprehensive analysis results in a formatted table matching CSV export format"""
    try:
        if 'categorized_results' not in results:
            logger.warning("No categorized results found for display")
            return
        
        # Extract all results from all categories
        all_analysis_results = []
        for level, stock_results in results['categorized_results'].items():
            all_analysis_results.extend(stock_results)
        
        if not all_analysis_results:
            logger.info("No analysis results to display")
            return
        
        print("\n" + "="*180)
        print("NSE STOCK SCREENER - COMPREHENSIVE ANALYSIS RESULTS")
        print("="*180)
        
        # Comprehensive headers matching CSV format
        headers = [
            "Symbol", "Composite_Score", "Probability", "Current_Price", "Price_Change_%", 
            "Volume_Ratio", "Volume_Z_Score", "RSI", "MACD_Signal", "ADX", "ATR_%", 
            "Rel_Strength_20d", "Can_Enter", "Suggested_Qty", "Risk_Amount", "Stop_Loss"
        ]
        
        # Print headers with proper spacing for readability
        header_line = ""
        for header in headers:
            if header == "Symbol":
                header_line += f"{header:<10} | "
            elif header in ["Composite_Score", "Current_Price", "Price_Change_%", "Volume_Ratio", "Volume_Z_Score", "RSI", "ADX", "ATR_%", "Rel_Strength_20d", "Suggested_Qty", "Risk_Amount", "Stop_Loss"]:
                header_line += f"{header:<12} | "
            elif header in ["Probability", "MACD_Signal"]:
                header_line += f"{header:<11} | "
            else:
                header_line += f"{header:<9} | "
        
        print(header_line)
        print("-" * 180)
        
        # Print each stock's comprehensive data
        for result in all_analysis_results:
            symbol = result['symbol'].replace('.NS', '')[:10]
            score = result['composite_score']
            prob_level = result['probability_level']
            
            # Extract comprehensive indicators
            indicators = result['key_indicators']
            risk_mgmt = result['risk_management']
            
            current_price = indicators['current_price']
            price_change = indicators['price_change_pct']
            volume_ratio = indicators['volume_ratio']
            volume_z_score = indicators.get('volume_z_score', 0)
            rsi = indicators['rsi']
            macd_signal = indicators['macd_signal']
            adx = indicators['adx']
            atr_pct = indicators['atr_pct']
            rel_strength = indicators.get('rel_strength_20d', 0)
            can_enter = risk_mgmt['can_enter_position']
            suggested_qty = risk_mgmt.get('suggested_quantity', 0)
            risk_amount = risk_mgmt.get('risk_amount', 0)
            stop_loss = risk_mgmt.get('stop_loss_price', 0)
            
            # Format the comprehensive row data
            row_data = [
                f"{symbol:<10}",
                f"{score:<12.1f}",
                f"{prob_level:<11}",
                f"{current_price:<12.2f}",
                f"{price_change:<12.2f}",
                f"{volume_ratio:<12.2f}",
                f"{volume_z_score:<12.2f}",
                f"{rsi:<12.2f}",
                f"{macd_signal:<11}",
                f"{adx:<12.2f}",
                f"{atr_pct:<12.2f}",
                f"{rel_strength:<12.2f}",
                f"{'True' if can_enter else 'False':<9}",
                f"{suggested_qty:<12}",
                f"{risk_amount:<12.2f}",
                f"{stop_loss:<12.2f}"
            ]
            
            print(" | ".join(row_data))
        
        print("-" * 180)
        print(f"Total stocks analyzed: {len(all_analysis_results)}")
        
        # Summary by probability level with comprehensive statistics
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = len(results['categorized_results'].get(level, []))
            if count > 0:
                avg_score = sum(r['composite_score'] for r in results['categorized_results'][level]) / count
                risk_approved = len([r for r in results['categorized_results'][level] if r['risk_management']['can_enter_position']])
                print(f"{level} probability: {count} stocks (avg score: {avg_score:.1f}, risk approved: {risk_approved})")
        
        print("="*180)
        
    except Exception as e:
        logger.error(f"Error displaying comprehensive results: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main CLI entry point"""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup logging
    logger = setup_logging_from_args(args)
    
    # Handle special operations
    if args.validate_setup:
        from scripts.check_deps import main as check_deps
        check_deps()
        return
    
    if args.update_symbols:
        logger.info("Updating NSE symbols list...")
        try:
            from src.dynamic_watchlist_manager import DynamicWatchlistManager
            dwm = DynamicWatchlistManager()
            dwm.update_market_data(sample_size=200)
            logger.info("✅ NSE symbols updated successfully")
        except Exception as e:
            logger.error(f"❌ Failed to update symbols: {e}")
        return
    
    # Load stocks
    stocks = load_stocks_from_input(args, logger)
    if not stocks:
        logger.error("❌ No stocks to analyze")
        sys.exit(1)
    
    # Run analysis
    results = run_analysis_with_args(args, stocks, logger)
    if not results:
        logger.error("❌ Analysis failed")
        sys.exit(1)
    
    # Save results
    save_results(results, args, logger)
    
    # Display results to console if requested or if table format specified
    if args.display_console or args.output_format == 'table' or not args.output:
        display_comprehensive_results(results, logger)
    
    logger.info("🎉 CLI analysis completed successfully!")

if __name__ == "__main__":
    main()