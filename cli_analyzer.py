#!/usr/bin/env python3
"""
Command Line Interface for NSE Stock Screener
Provides powerful command-line options for advanced users
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='NSE Stock Screener - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s "RELIANCE"                    # Single stock analysis
  %(prog)s -s "RELIANCE,TCS,INFY"           # Multiple stocks
  %(prog)s --banking                        # Banking sector analysis
  %(prog)s --technology                     # Technology sector analysis
  %(prog)s --quick-scan                     # Quick market scan
  %(prog)s --min-score 70                   # High-score stocks only
  %(prog)s --format csv,json                # Multiple output formats
  %(prog)s --output-prefix "morning_scan"   # Custom file naming
        """)
    
    # Stock Selection Options
    stock_group = parser.add_argument_group('Stock Selection')
    stock_group.add_argument('-s', '--stocks', type=str, 
                            help='Comma-separated list of stock symbols (e.g., RELIANCE,TCS,INFY)')
    stock_group.add_argument('-f', '--file', type=str,
                            help='Path to file containing stock symbols (one per line)')
    stock_group.add_argument('--banking', action='store_true',
                            help='Analyze banking sector stocks')
    stock_group.add_argument('--technology', action='store_true',
                            help='Analyze technology sector stocks')
    stock_group.add_argument('--full-nse', action='store_true',
                            help='Analyze all NSE stocks (2155+ stocks)')
    
    # Analysis Options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--quick-scan', action='store_true',
                               help='Quick scan mode (faster, basic analysis)')
    analysis_group.add_argument('--backtest', action='store_true',
                               help='Include backtesting analysis')
    analysis_group.add_argument('--period', choices=['1m', '3m', '6m', '1y', '2y'], default='1y',
                               help='Analysis period (default: 1y)')
    
    # Filtering Options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--min-score', type=int, default=0,
                             help='Minimum composite score (0-100, default: 0)')
    filter_group.add_argument('--max-results', type=int, default=100,
                             help='Maximum number of results (default: 100)')
    filter_group.add_argument('--signal-strength', choices=['HIGH', 'MEDIUM', 'LOW'],
                             help='Filter by signal strength')
    
    # Output Options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--format', default='csv',
                             help='Output format: csv, json, xlsx, or comma-separated (default: csv)')
    output_group.add_argument('--output-prefix', type=str,
                             help='Prefix for output filenames')
    output_group.add_argument('--output-dir', type=str,
                             help='Output directory (default: output/reports/)')
    output_group.add_argument('--no-charts', action='store_true',
                             help='Skip chart generation')
    
    # Performance Options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--batch-size', type=int, default=20,
                           help='Batch size for processing (default: 20)')
    perf_group.add_argument('--timeout', type=int, default=30,
                           help='Timeout per stock in seconds (default: 30)')
    perf_group.add_argument('--parallel', action='store_true',
                           help='Enable parallel processing (experimental)')
    
    # Debug Options
    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
    debug_group.add_argument('--test-mode', action='store_true',
                            help='Test mode (dry run)')
    debug_group.add_argument('--verbose', action='store_true',
                            help='Verbose output')
    
    args = parser.parse_args()
    
    # Import here to avoid import issues at startup
    try:
        from enhanced_analysis_wrapper import create_enhanced_analyzer
        print("[OK] Enhanced analysis system loaded successfully")
    except ImportError as e:
        print(f"[ERROR] Error loading enhanced analysis: {e}")
        print("[INFO] Make sure enhanced_analysis_wrapper.py is available")
        return 1
    
    # Determine stock selection
    symbols = []
    description = "Command Line Analysis"
    
    if args.stocks:
        symbols = [s.strip().upper() for s in args.stocks.split(',') if s.strip()]
        if not all(s.endswith('.NS') for s in symbols):
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
        description = f"Custom Analysis - {len(symbols)} stocks"
        
    elif args.banking:
        try:
            from nse_sector_filter import filter_banking_stocks
            symbols = filter_banking_stocks(top_n=args.max_results)
            # Add .NS suffix for Yahoo Finance
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
            description = f"Banking Sector Analysis - {len(symbols)} stocks"
        except ImportError:
            print("[ERROR] Sector filtering not available, using fallback banking stocks")
            symbols = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
            description = "Banking Sector Analysis (Fallback)"
            
    elif args.technology:
        try:
            from nse_sector_filter import filter_technology_stocks
            symbols = filter_technology_stocks(top_n=args.max_results)
            # Add .NS suffix for Yahoo Finance
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
            description = f"Technology Sector Analysis - {len(symbols)} stocks"
        except ImportError:
            print("[ERROR] Sector filtering not available, using fallback technology stocks")
            symbols = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
            description = "Technology Sector Analysis (Fallback)"
            
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            if not all(s.endswith('.NS') for s in symbols):
                symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
            description = f"File Analysis - {len(symbols)} stocks from {args.file}"
        except FileNotFoundError:
            print(f"[ERROR] File not found: {args.file}")
            return 1
            
    elif args.full_nse:
        try:
            from nse_sector_filter import get_all_nse_symbols
            symbols = get_all_nse_symbols()
            # Add .NS suffix for Yahoo Finance
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
            description = f"Full NSE Analysis - {len(symbols)} stocks"
        except ImportError:
            print("[ERROR] NSE symbol loader not available")
            return 1
    else:
        print("[ERROR] No stock selection specified. Use -s, --banking, --technology, or --file")
        parser.print_help()
        return 1
    
    if not symbols:
        print("[ERROR] No symbols found for analysis")
        return 1
    
    # Apply filters
    if args.max_results and len(symbols) > args.max_results:
        symbols = symbols[:args.max_results]
        description += f" (limited to {args.max_results})"
    
    # Display analysis info
    print(f"\n[LAUNCH] NSE STOCK SCREENER - COMMAND LINE")
    print(f"[DATA] {description}")
    print(f"[TARGET] Analyzing {len(symbols)} stocks")
    if args.min_score > 0:
        print(f"[CHART] Minimum score filter: {args.min_score}")
    if args.signal_strength:
        print(f"[FAST] Signal strength filter: {args.signal_strength}")
    print()
    
    # Create analyzer
    analyzer = create_enhanced_analyzer()
    
    # Custom parameters handled by wrapper
    
    if args.test_mode:
        print("🧪 Test mode - no actual analysis will be performed")
        print(f"Would analyze: {', '.join([s.replace('.NS', '') for s in symbols[:5]])}")
        if len(symbols) > 5:
            print(f"... and {len(symbols) - 5} more stocks")
        return 0
    
    # Run analysis
    start_time = time.time()
    
    try:
        results_df = analyzer.analyze_multiple_symbols(symbols, description)
        
        if results_df is not None and not results_df.empty:
            # Apply score filter
            if args.min_score > 0:
                results_df = results_df[results_df['Composite_Score'] >= args.min_score]
            
            # Apply signal strength filter
            if args.signal_strength:
                # Calculate signal strength from score
                def get_signal_strength(score):
                    if score >= 75: return 'HIGH'
                    elif score >= 60: return 'MEDIUM'
                    else: return 'LOW'
                
                results_df['Signal_Strength'] = results_df['Composite_Score'].apply(get_signal_strength)
                results_df = results_df[results_df['Signal_Strength'] == args.signal_strength]
            
            # Display results summary
            elapsed_time = time.time() - start_time
            print(f"\n[OK] Analysis completed in {elapsed_time:.1f} seconds")
            print(f"[DATA] Results: {len(results_df)} stocks after filtering")
            
            if len(results_df) > 0:
                print(f"\n[TOP] TOP PERFORMERS:")
                top_results = results_df.head(5)
                for idx, row in top_results.iterrows():
                    symbol = row['Symbol'].replace('.NS', '')
                    score = row['Composite_Score']
                    price = row.get('Current_Price', 'N/A')
                    can_enter = row.get('Can_Enter', 'Unknown')
                    print(f"   {symbol:12} | Score: {score:5.1f} | Price: Rs{price} | Entry: {can_enter}")
            
        else:
            print("[ERROR] No analysis results generated")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    print(f"\n[FOLDER] Results saved to: output/reports/")
    print(f"[TIP] Use --min-score to filter results")
    print(f"[CHART] Use --banking or --technology for sector analysis")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())