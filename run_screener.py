#!/usr/bin/env python3
"""
NSE Stock Screener Package Runner
Fixes import issues by properly setting up the Python path and running as a package
"""

import sys
import os
from pathlib import Path

def setup_package_path():
    """Set up Python path to enable proper package imports"""
    # Get the project root directory
    project_root = Path(__file__).parent
    src_dir = project_root / "src"

    # Add both project root and src to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Set environment variable for package recognition
    os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{src_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    return project_root, src_dir

def run_enhanced_launcher():
    """Run the enhanced launcher with proper package setup"""
    project_root, src_dir = setup_package_path()

    print("🚀 NSE Stock Screener - Package Runner")
    print("=" * 50)
    print(f"Project Root: {project_root}")
    print(f"Source Directory: {src_dir}")
    print(f"Python Path: {sys.path[:3]}...")
    print("=" * 50)

    try:
        # Change to src directory for relative imports
        original_cwd = os.getcwd()
        os.chdir(src_dir)

        # Import and run the launcher
        from enhanced_launcher import main as launcher_main
        launcher_main()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import method...")

        try:
            # Alternative: Run as module
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "enhanced_launcher"
            ], cwd=src_dir, capture_output=False)
            return result.returncode == 0

        except Exception as e2:
            print(f"❌ Alternative method failed: {e2}")
            return False

    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False

    finally:
        # Restore original directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

    return True

def run_quick_analysis(symbols=None):
    """Run a quick analysis with proper package setup"""
    project_root, src_dir = setup_package_path()

    if symbols is None:
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    print(f"🎯 Running Quick Analysis for: {', '.join(symbols)}")

    try:
        # Change to src directory
        original_cwd = os.getcwd()
        os.chdir(src_dir)

        # Import modules with fallback
        try:
            from enhanced_early_warning_system import EnhancedEarlyWarningSystem
        except ImportError:
            print("Using simplified analysis...")
            return run_simplified_analysis(symbols)

        # Run analysis
        system = EnhancedEarlyWarningSystem(custom_stocks=symbols)
        results = system.run_analysis()

        print("✅ Analysis completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Trying simplified analysis...")
        return run_simplified_analysis(symbols)

    finally:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

def run_simplified_analysis(symbols):
    """Run simplified analysis without complex imports"""
    print("🔧 Running Simplified Analysis...")

    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        results = []

        for symbol in symbols:
            try:
                print(f"  Analyzing {symbol}...")

                # Fetch data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")

                if len(data) < 50:
                    print(f"    ⚠️ Insufficient data for {symbol}")
                    continue

                # Simple technical analysis
                current_price = data['Close'].iloc[-1]
                sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                sma_50 = data['Close'].rolling(50).mean().iloc[-1]

                # Volume analysis
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Simple RSI
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]

                # Simple scoring
                score = 50  # Base score

                # Price above moving averages
                if current_price > sma_20:
                    score += 10
                if current_price > sma_50:
                    score += 10

                # RSI conditions
                if 40 < current_rsi < 70:
                    score += 15
                elif current_rsi > 70:
                    score += 5  # Slightly overbought

                # Volume
                if volume_ratio > 1.5:
                    score += 15

                # Determine signal
                if score >= 70:
                    signal = "HIGH"
                elif score >= 60:
                    signal = "MEDIUM"
                else:
                    signal = "LOW"

                results.append({
                    'Symbol': symbol,
                    'Price': f"₹{current_price:.2f}",
                    'RSI': f"{current_rsi:.1f}",
                    'Volume_Ratio': f"{volume_ratio:.1f}x",
                    'Score': score,
                    'Signal': signal
                })

                print(f"    ✅ {symbol}: Score {score}, Signal {signal}")

            except Exception as e:
                print(f"    ❌ Failed to analyze {symbol}: {e}")

        # Display results
        if results:
            print("\n📊 Analysis Results:")
            print("-" * 60)
            for result in results:
                print(f"{result['Symbol']:12} | {result['Price']:>10} | RSI: {result['RSI']:>6} | "
                      f"Vol: {result['Volume_Ratio']:>6} | Score: {result['Score']:>3} | {result['Signal']}")

            # Save to CSV
            output_dir = Path("output/reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(results)
            output_file = output_dir / f"simplified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"❌ Simplified analysis failed: {e}")
        return False

def main():
    """Main entry point with options"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "launcher":
            return run_enhanced_launcher()
        elif command == "analysis":
            symbols = sys.argv[2:] if len(sys.argv) > 2 else None
            return run_quick_analysis(symbols)
        elif command == "test":
            return run_quick_analysis(["RELIANCE.NS"])
        else:
            print(f"Unknown command: {command}")
            return False

    # Interactive menu
    print("🚀 NSE Stock Screener - Package Runner")
    print("=" * 40)
    print("1. Run Interactive Launcher")
    print("2. Run Quick Analysis (3 stocks)")
    print("3. Run Test Analysis (RELIANCE only)")
    print("4. Custom Analysis (enter symbols)")
    print("5. Exit")
    print("=" * 40)

    try:
        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            return run_enhanced_launcher()
        elif choice == "2":
            return run_quick_analysis()
        elif choice == "3":
            return run_quick_analysis(["RELIANCE.NS"])
        elif choice == "4":
            symbols_input = input("Enter symbols (comma-separated): ").strip()
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
            if symbols:
                return run_quick_analysis(symbols)
            else:
                print("No symbols provided.")
                return False
        elif choice == "5":
            print("Goodbye!")
            return True
        else:
            print("Invalid choice.")
            return False

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)