import logging
"""
Enhanced Stock Screening System Launcher
Simple launcher script for the upgraded analysis system
"""

from pathlib import Path
import os
import sys

import subprocess

def main():
    print("🚀 ENHANCED STOCK SCREENING SYSTEM")
    print("=" * 50)
    print("Upgraded with probabilistic scoring, risk management, and advanced indicators")
    print()
    
    # Get the source directory
    src_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    required_files = [
        'enhanced_early_warning_system.py',
        'advanced_indicators.py',
        'composite_scorer.py',
        'risk_manager.py',
        'advanced_backtester.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not (src_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all system files are in the src/ directory")
        return
    
    print("✅ All system files found")
    print()
    
    # Menu options
    print("SELECT ANALYSIS TYPE:")
    print("1. Quick Demo (10 stocks, basic analysis)")
    print("2. Full Analysis (from stock list file)")
    print("3. Custom Stocks (enter symbols)")
    print("4. Advanced Analysis with Backtesting")
    print("5. Show System Features")
    print("6. Go back to previous menu")
    print("7. Exit")
    print()
    
    try:
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            print("\n🎯 Running Quick Demo...")
            run_quick_demo(src_dir)
            
        elif choice == "2":
            print("\n📋 Running Full Analysis...")
            run_full_analysis(src_dir)
            
        elif choice == "3":
            print("\n🎯 Custom Stocks Analysis...")
            run_custom_analysis(src_dir)
            
        elif choice == "4":
            print("\n📊 Advanced Analysis with Backtesting...")
            run_advanced_analysis(src_dir)
            
        elif choice == "5":
            print("\n📖 System Features...")
            show_features()
            
        elif choice == "6":
            print("\n⬅️ Going back to previous menu...")
            return False  # Signal to go back
            
        elif choice == "7":
            print("\n👋 Goodbye!")
            return True  # Signal to exit
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user")
    except Exception as e:
        logging.error(f"\n❌ Error: {e}")

def run_quick_demo(src_dir):
    """Run quick demo with default stocks"""
    cmd = [
        sys.executable,
        str(src_dir / "demo_enhanced_system.py")
    ]
    
    print("Launching demo...")
    subprocess.run(cmd, cwd=src_dir)

def run_full_analysis(src_dir):
    """Run analysis from stock list file"""
    # Check for stock list file
    stock_files = [
        "../data/nse_only_symbols.txt",
        "nse_only_symbols.txt",
        "stocks.txt"
    ]
    
    stock_file = None
    for file in stock_files:
        if os.path.exists(os.path.join(src_dir.parent if file.startswith("..") else src_dir, file.replace("../", ""))):
            stock_file = file
            break
    
    if not stock_file:
        print("❌ No stock list file found. Looking for:")
        for file in stock_files:
            print(f"   - {file}")
        print("\nPlease ensure you have a stock list file, or use option 3 for custom stocks.")
        return
    
    print(f"📋 Using stock file: {stock_file}")
    
    # Ask for batch settings
    print("\nBatch Settings (press Enter for defaults):")
    batch_size = input("Batch size (default 50): ").strip() or "50"
    timeout = input("Timeout between batches in seconds (default 10): ").strip() or "10"
    
    cmd = [
        sys.executable,
        str(src_dir / "enhanced_early_warning_system.py"),
        "-f", stock_file,
        "-b", batch_size,
        "-t", timeout
    ]
    
    print(f"Launching analysis with batch size {batch_size}, timeout {timeout}s...")
    subprocess.run(cmd, cwd=src_dir)

def run_custom_analysis(src_dir):
    """Run analysis with custom stock symbols"""
    print("Enter stock symbols (NSE symbols without .NS suffix)")
    print("Example: RELIANCE,TCS,HDFCBANK,INFY")
    print()
    
    stocks = input("Stock symbols (comma-separated): ").strip()
    
    if not stocks:
        print("❌ No stocks entered")
        return
    
    cmd = [
        sys.executable,
        str(src_dir / "enhanced_early_warning_system.py"),
        "-s", stocks
    ]
    
    print(f"Launching analysis for: {stocks}")
    subprocess.run(cmd, cwd=src_dir)

def run_advanced_analysis(src_dir):
    """Run advanced analysis with backtesting"""
    print("Advanced Analysis includes:")
    print("- Full indicator suite")
    print("- Risk management analysis")
    print("- Historical backtesting")
    print("- Comprehensive reporting")
    print()
    
    # Get input type
    input_type = input("Input type (1=file, 2=custom stocks): ").strip()
    
    cmd = [
        sys.executable,
        str(src_dir / "enhanced_early_warning_system.py"),
        "--backtest"
    ]
    
    if input_type == "1":
        stock_file = input("Stock file path (or press Enter for default): ").strip()
        if stock_file:
            cmd.extend(["-f", stock_file])
    elif input_type == "2":
        stocks = input("Stock symbols (comma-separated): ").strip()
        if stocks:
            cmd.extend(["-s", stocks])
    
    print("Launching advanced analysis...")
    subprocess.run(cmd, cwd=src_dir)

def show_features():
    """Show system features"""
    print("\n🌟 ENHANCED STOCK SCREENING SYSTEM FEATURES")
    print("=" * 60)
    
    features = [
        ("Multi-Indicator Analysis", [
            "Volume z-score (statistical significance)",
            "Enhanced RSI with regime-adaptive thresholds", 
            "MACD with histogram momentum",
            "ADX for trend strength measurement",
            "ATR for volatility analysis",
            "Relative strength vs NIFTY index",
            "Volume profile breakout detection"
        ]),
        
        ("Probabilistic Scoring System", [
            "0-100 composite score with weighted components",
            "HIGH (≥70), MEDIUM (45-69), LOW (<45) classification",
            "Weekly timeframe confirmation bonus",
            "Market regime adaptive scoring"
        ]),
        
        ("Risk Management Integration", [
            "ATR-based stop loss calculation",
            "Volatility-scaled position sizing",
            "Portfolio exposure limits",
            "Risk-reward ratio analysis",
            "Maximum position and exposure controls"
        ]),
        
        ("Advanced Backtesting", [
            "Walk-forward analysis methodology",
            "Transaction costs and slippage inclusion",
            "Comprehensive performance metrics",
            "Sharpe ratio, max drawdown, expectancy",
            "Statistical significance testing"
        ]),
        
        ("Market Regime Detection", [
            "Bull/Bear/Neutral market classification",
            "Adaptive indicator thresholds",
            "Index breadth analysis",
            "Volatility regime consideration"
        ]),
        
        ("Enhanced Reporting", [
            "Detailed CSV reports with all metrics",
            "Multi-panel technical analysis charts",
            "Risk management recommendations",
            "Component-wise score breakdown",
            "Executive summary reports"
        ])
    ]
    
    for category, items in features:
        print(f"\n🔹 {category}:")
        for item in items:
            print(f"   • {item}")
    
    print(f"\n💡 UPGRADE HIGHLIGHTS:")
    print("   • Detects TRUE unusual volume (z-score + rolling ratio)")
    print("   • Uses multi-indicator confirmation across 7+ signals")
    print("   • Multi-timeframe analysis (daily + weekly)")
    print("   • Adapts to market conditions automatically")
    print("   • Produces clear HIGH/MEDIUM/LOW probability buckets")
    print("   • Includes robust backtesting with realistic costs")
    print("   • Comprehensive risk management integration")
    
    input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()