#!/usr/bin/env python3
"""
Fixed Entry Point for NSE Stock Screener
Handles both package and standalone execution
"""

import sys
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import the original enhanced system
from src.enhanced_early_warning_system import EnhancedEarlyWarningSystem

def main() -> int:
    """Main entry point for the screener"""
    import argparse

    parser = argparse.ArgumentParser(description="NSE Stock Screener - Enhanced Analysis")
    parser.add_argument('-s', '--symbols', type=str, help='Comma-separated list of symbols (e.g., "RELIANCE,TCS,INFY")')
    parser.add_argument('-f', '--file', type=str, help='File containing stock symbols')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('-t', '--timeout', type=int, default=10, help='Timeout between batches')
    parser.add_argument('--test', action='store_true', help='Run test analysis on a single stock')

    args = parser.parse_args()

    # Handle test mode
    if args.test:
        print("🧪 Running test analysis...")
        custom_stocks = ["RELIANCE.NS"]
    elif args.symbols:
        # Parse comma-separated symbols and add .NS suffix if needed
        symbols = [s.strip() for s in args.symbols.split(',')]
        custom_stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
    else:
        custom_stocks = None

    try:
        # Initialize the system
        system = EnhancedEarlyWarningSystem(
            custom_stocks=custom_stocks,
            input_file=args.file,
            batch_size=args.batch_size,
            timeout=args.timeout
        )

        # Run the analysis
        print("🚀 Starting Enhanced NSE Stock Analysis...")
        system.run_analysis()

        # Show results summary
        if hasattr(system, 'analysis_results') and system.analysis_results:
            print(f"\n✅ Analysis completed! Found {len(system.analysis_results)} results.")
            print(f"📊 Results saved to: {system.output_dirs.get('reports', 'output/reports/')}")
        else:
            print("⚠️  No analysis results generated.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
