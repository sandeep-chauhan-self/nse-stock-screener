#!/usr/bin/env python3
"""
NSE Stock Screener - CLI Analyzer
Simple command-line interface for automation
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Main CLI analyzer entry point"""
    parser = argparse.ArgumentParser(description='NSE Stock Screener CLI')
    
    # Sector filters
    parser.add_argument('--banking', action='store_true', help='Analyze banking sector')
    parser.add_argument('--technology', action='store_true', help='Analyze technology sector')
    parser.add_argument('--pharma', action='store_true', help='Analyze pharma sector')
    
    # Score filters
    parser.add_argument('--min-score', type=int, default=60, help='Minimum composite score')
    parser.add_argument('--max-results', type=int, default=50, help='Maximum results to return')
    
    # Output options
    parser.add_argument('--output-prefix', type=str, default='cli_analysis', help='Output file prefix')
    parser.add_argument('--save-charts', action='store_true', help='Save charts')
    
    args = parser.parse_args()
    
    print(f"CLI Analyzer called with args: {args}")
    print("This is a placeholder CLI analyzer for automation testing")
    print("In production, this would call the enhanced_analysis_wrapper.py")
    
    # Create dummy output for automation testing
    output_dir = PROJECT_ROOT / 'output' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.output_prefix}_{timestamp}.csv"
    
    # Create minimal CSV for testing
    with open(output_file, 'w') as f:
        f.write("Symbol,Composite_Score,Can_Enter\n")
        f.write("RELIANCE.NS,65.2,True\n")
        f.write("TCS.NS,58.7,False\n")
        f.write("HDFC.NS,71.3,True\n")
    
    print(f"Created test output: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())