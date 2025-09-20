import logging
"""
Enhanced Stock Screening System - Usage Example
This demonstrates how to use the upgraded system with all new features
"""

from datetime import datetime
import os
import sys

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import the enhanced system
from enhanced_early_warning_system import EnhancedEarlyWarningSystem

def run_basic_analysis():
    """Run basic analysis with default settings"""
    print("=" * 60)
    print("ENHANCED STOCK SCREENING SYSTEM - BASIC ANALYSIS")
    print("=" * 60)
    
    # Initialize with default stocks (smaller list for demo)
    demo_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS', 'SUNPHARMA.NS', 'WIPRO.NS'
    ]
    
    ews = EnhancedEarlyWarningSystem(
        custom_stocks=demo_stocks,
        batch_size=5,  # Smaller batches for demo
        timeout=5
    )
    
    # Run analysis
    results = ews.run_enhanced_analysis()
    
    return results

def run_advanced_analysis():
    """Run analysis with custom settings and backtesting"""
    print("=" * 60)
    print("ENHANCED STOCK SCREENING SYSTEM - ADVANCED ANALYSIS")
    print("=" * 60)
    
    # Custom stock list
    custom_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'DRREDDY.NS'
    ]
    
    ews = EnhancedEarlyWarningSystem(
        custom_stocks=custom_stocks,
        batch_size=3,
        timeout=3
    )
    
    # Run main analysis
    results = ews.run_enhanced_analysis()
    
    # Run backtesting
    print("\n" + "=" * 60)
    print("RUNNING BACKTESTING ANALYSIS")
    print("=" * 60)
    
    backtest_results = ews.run_backtest_analysis()
    
    return results, backtest_results

def demonstrate_key_features():
    """Demonstrate key features of the upgraded system"""
    print("\n" + "=" * 80)
    print("KEY FEATURES OF THE ENHANCED SYSTEM")
    print("=" * 80)
    
    features = [
        ("🎯 Probabilistic Scoring", "0-100 composite score with HIGH/MEDIUM/LOW classification"),
        ("📊 Multi-Indicator Analysis", "Volume z-score, RSI, MACD, ADX, ATR, relative strength, volume profile"),
        ("⏰ Multi-Timeframe Confirmation", "Daily + weekly analysis to reduce false signals"),
        ("🌍 Market Regime Detection", "Adaptive thresholds based on bull/bear/neutral market conditions"),
        ("💰 Risk Management Integration", "Position sizing, stop losses, exposure limits"),
        ("📈 Advanced Backtesting", "Walk-forward testing with transaction costs and slippage"),
        ("📋 Comprehensive Reporting", "Detailed CSV reports with all metrics and risk analysis"),
        ("📊 Enhanced Charting", "Multi-panel technical analysis charts"),
        ("🔧 Volume Profile Approximation", "Breakout detection above high-volume price nodes"),
        ("⚡ Improved Volume Detection", "Statistical z-score analysis, not just simple ratios")
    ]
    
    for feature, description in features:
        print(f"{feature}: {description}")
    
    print("\n" + "=" * 80)
    print("SCORING BREAKDOWN (Total: 100 points + 10 bonus)")
    print("=" * 80)
    
    scoring = [
        ("Volume Analysis", "25 points", "Volume ratio + z-score"),
        ("Momentum Signals", "25 points", "RSI sweet spot + MACD signals"),
        ("Trend Strength", "15 points", "ADX + moving average alignment"),
        ("Volatility Analysis", "10 points", "ATR expansion + levels"),
        ("Relative Strength", "10 points", "Performance vs NIFTY index"),
        ("Volume Profile", "10 points", "Breakout above resistance levels"),
        ("Weekly Confirmation", "10 bonus", "Weekly timeframe confirmation")
    ]
    
    for component, points, description in scoring:
        print(f"{component:<20} {points:<10} {description}")
    
    print("\n" + "=" * 80)
    print("PROBABILITY THRESHOLDS")
    print("=" * 80)
    print("HIGH Probability: Score ≥ 70 (Strong multi-indicator confirmation)")
    print("MEDIUM Probability: Score 45-69 (Moderate signals)")
    print("LOW Probability: Score < 45 (Weak or mixed signals)")

if __name__ == "__main__":
    try:
        # Demonstrate key features
        demonstrate_key_features()
        
        # Ask user which demo to run
        print("\n" + "=" * 60)
        print("DEMO OPTIONS")
        print("=" * 60)
        print("1. Basic Analysis (10 stocks, quick demo)")
        print("2. Advanced Analysis (with backtesting)")
        print("3. Show features only (no analysis)")
        print("4. Go back to previous menu")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning basic analysis demo...")
            results = run_basic_analysis()
            print("\n✅ Basic analysis completed!")
            
        elif choice == "2":
            print("\nRunning advanced analysis demo...")
            results, backtest_results = run_advanced_analysis()
            print("\n✅ Advanced analysis with backtesting completed!")
            
        elif choice == "3":
            print("\nShowing system features...")
            demonstrate_key_features()
            print("\n✅ Features demonstration completed!")
            
        elif choice == "4":
            print("\n⬅️ Going back to previous menu...")
            exit(0)  # Exit to go back
            
        elif choice == "5":
            print("\n👋 Exiting...")
            exit(0)  # Exit program
            
        else:
            print("❌ Invalid choice. Running basic analysis...")
            results = run_basic_analysis()
        
        print("\n" + "=" * 80)
        print("USAGE SUMMARY")
        print("=" * 80)
        print("Command Line Usage:")
        logging.warning("python enhanced_early_warning_system.py                    # Default analysis")
        logging.warning("python enhanced_early_warning_system.py -f stocks.txt     # From file")
        logging.warning("python enhanced_early_warning_system.py -s RELIANCE,TCS   # Custom stocks")
        logging.warning("python enhanced_early_warning_system.py --backtest        # With backtesting")
        logging.warning("python enhanced_early_warning_system.py -b 20 -t 5        # Custom batch settings")
        
        print("\nOutput Files:")
        print("- output/reports/high_probability_enhanced_*.csv")
        print("- output/reports/comprehensive_analysis_*.csv")
        print("- output/reports/analysis_summary_*.txt")
        print("- output/charts/*_enhanced_chart.png")
        print("- output/backtests/backtest_results_*.csv")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logging.error(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()