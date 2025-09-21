#!/usr/bin/env python3
"""
Enhanced Interactive Launcher for NSE Stock Screener
Focuses on working functionality with user-friendly interface
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def clear_screen():
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print attractive header"""
    print("=" * 60)
    print("🚀 NSE STOCK SCREENER - Interactive Launcher")
    print("=" * 60)
    print("📊 Real-time Analysis | 🎯 Smart Scoring | 💰 Risk Management")
    print()

def print_menu():
    """Print main menu options"""
    print("📋 SELECT YOUR ANALYSIS:")
    print()
    print("1.  Banking Sector Analysis")
    print("2. 💻 Technology Sector Analysis")
    print("3. ⚡ Custom Stock Analysis")
    print("4. 📈 Portfolio Watchlist (Intelligent)")
    print("5. 🔍 Search & Analyze")
    print("6. 📊 View Recent Results")
    print("7. ℹ️  System Information")
    print("8. ❌ Exit")
    print()

def get_user_choice():
    """Get and validate user choice"""
    while True:
        try:
            choice = input("👉 Enter your choice (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 1-8.")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return 8

def run_analysis(symbols, description):
    """Run comprehensive enhanced analysis with given symbols"""
    try:
        # Import the enhanced analysis wrapper
        from enhanced_analysis_wrapper import create_enhanced_analyzer
        
        # Create analyzer and run comprehensive analysis
        analyzer = create_enhanced_analyzer()
        results_df = analyzer.analyze_multiple_symbols(symbols, description)
        
        if not results_df.empty:
            return True
        else:
            print("❌ No analysis results generated")
            return False
            
    except ImportError as e:
        print(f"⚠️ Enhanced analysis unavailable, falling back to simplified analysis...")
        print(f"Import error: {e}")
        
        # Fallback to simplified analysis
        try:
            from run_screener import run_simplified_analysis
            results = run_simplified_analysis(symbols)
            print(f"\n✅ Simplified analysis completed!")
            return True
        except Exception as e2:
            print(f"❌ Both enhanced and simplified analysis failed: {e2}")
            return False
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {e}")
        return False

def show_predefined_options():
    """Show production-ready intelligent watchlists with analytical ranking"""
    try:
        # Import the intelligent watchlist manager
        from src.intelligent_watchlist_manager import IntelligentWatchlistManager
        
        print("\n🧠 Loading intelligent watchlists with analytical ranking...")
        iwm = IntelligentWatchlistManager()
        
        # Check analysis status
        status = iwm.get_analysis_summary()
        print(f"📊 Analysis Status: {status['analysis_type']} ({status['market_regime']} market)")
        
        # Check if we need fresh data
        stats = iwm.get_stats()
        if stats['cached_sector_data'] < 50:
            print("📊 Fetching fresh market data (first time setup)...")
            iwm.update_market_data(sample_size=150)
        
        watchlists = iwm.list_intelligent_watchlists()
        
        if not watchlists:
            print("❌ No intelligent watchlists generated. Please check NSE data.")
            return None
        
        print(f"\n🧠 INTELLIGENT NSE WATCHLISTS ({len(watchlists)} available):")
        print(f"📊 Based on {stats['total_nse_symbols']} NSE symbols with analytical ranking")
        print("-" * 60)
        
        # Group watchlists by type for better organization
        sector_lists = [(i, id, name, desc, count, score) for i, (id, name, desc, count, score) in enumerate(watchlists, 1) if 'intelligent_sector_' in id]
        marketcap_lists = [(i, id, name, desc, count, score) for i, (id, name, desc, count, score) in enumerate(watchlists, 1) if 'intelligent_marketcap_' in id]
        special_lists = [(i, id, name, desc, count, score) for i, (id, name, desc, count, score) in enumerate(watchlists, 1) if id in ['intelligent_top_picks', 'intelligent_random']]
        
        # Display AI-ranked sector lists
        if sector_lists:
            print("🧠 AI-RANKED SECTOR WATCHLISTS:")
            for i, wl_id, name, description, count, score in sector_lists:
                score_display = f"(score: {score:.1f})" if score > 0 else "(unscored)"
                print(f"{i:2d}. {name} - {count} stocks {score_display}")
                print(f"     📝 {description}")
        
        # Display AI-enhanced market cap lists
        if marketcap_lists:
            print(f"\n💰 AI-ENHANCED MARKET CAP WATCHLISTS:")
            for i, wl_id, name, description, count, score in marketcap_lists:
                score_display = f"(score: {score:.1f})" if score > 0 else ""
                print(f"{i:2d}. {name} - {count} stocks {score_display}")
                print(f"     📝 {description}")
        
        # Display special AI lists
        if special_lists:
            print(f"\n⭐ AI SPECIAL WATCHLISTS:")
            for i, wl_id, name, description, count, score in special_lists:
                score_display = f"(score: {score:.1f})" if score > 0 else ""
                print(f"{i:2d}. {name} - {count} stocks {score_display}")
                print(f"     📝 {description}")
        
        print()
        
        # Get user choice
        while True:
            choice = input(f"\n👉 Select watchlist (1-{len(watchlists)}) or 'b' to go back: ").strip().lower()
            
            if choice == 'b':
                return None
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(watchlists):
                    selected_id = watchlists[choice_num - 1][0]
                    selected_watchlist = iwm.get_watchlist_with_analysis(selected_id)
                    
                    if selected_watchlist:
                        # Show preview of symbols with analysis info
                        symbols = selected_watchlist['symbols']
                        symbols_preview = ", ".join([s.replace('.NS', '') for s in symbols[:8]])
                        if len(symbols) > 8:
                            symbols_preview += f" (+{len(symbols)-8} more)"
                        
                        print(f"\n✅ Selected: {selected_watchlist['name']}")
                        print(f"📊 Analysis: {selected_watchlist.get('analysis_timestamp', 'Not analyzed')[:10]}")
                        print(f"🎯 Avg Score: {selected_watchlist.get('average_score', 'N/A')}")
                        print(f"📈 Stocks: {symbols_preview}")
                        
                        # Show detailed analysis if available
                        detailed_analysis = selected_watchlist.get('detailed_analysis', [])
                        if detailed_analysis:
                            print(f"\n🔬 Top Performers:")
                            for i, stock in enumerate(detailed_analysis[:3], 1):
                                score = stock.get('composite_score', 'N/A')
                                symbol = stock.get('symbol', '').replace('.NS', '')
                                print(f"   {i}. {symbol}: {score}")
                        
                        confirm = input("👉 Proceed with intelligent analysis? (y/n): ").lower()
                        if confirm == 'y':
                            return {
                                "name": selected_watchlist['name'],
                                "symbols": selected_watchlist['symbols'],
                                "description": selected_watchlist['description'],
                                "intelligent": True,
                                "analysis_data": selected_watchlist.get('detailed_analysis', [])
                            }
                        else:
                            continue  # Let user choose again
                    else:
                        print("❌ Error loading watchlist. Please try again.")
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(watchlists)} or 'b'.")
            except ValueError:
                print(f"❌ Invalid input. Please enter a number 1-{len(watchlists)} or 'b'.")
    
    except ImportError as e:
        print(f"❌ Error loading intelligent watchlist manager: {e}")
        print("💡 Falling back to basic watchlists...")
        return _show_basic_options()
    except Exception as e:
        print(f"❌ Error loading intelligent watchlists: {e}")
        return _show_basic_options()

def _show_basic_options():
    """Fallback basic watchlist options"""
    basic_options = {
        "1": {
            "name": "Quick Blue Chips",
            "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            "description": "Top 3 blue chip stocks"
        },
        "2": {
            "name": "Banking Trio",
            "symbols": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
            "description": "Major banking stocks"
        }
    }

    print("\n📋 BASIC WATCHLISTS:")
    for key, value in basic_options.items():
        symbols_str = ", ".join([s.replace('.NS', '') for s in value['symbols']])
        print(f"{key}. {value['name']} - {symbols_str}")

    choice = input("\n👉 Select (1-2) or 'b' to go back: ").strip()
    if choice == 'b':
        return None
    elif choice in basic_options:
        return basic_options[choice]
    else:
        return None

def get_custom_symbols():
    """Get custom symbols from user"""
    print("\n📝 CUSTOM SYMBOL ENTRY:")
    print("💡 Tips:")
    print("   • Use NSE symbols (e.g., RELIANCE, TCS, INFY)")
    print("   • Separate multiple symbols with commas")
    print("   • .NS suffix will be added automatically")
    print()

    while True:
        symbols_input = input("👉 Enter symbols: ").strip()
        if not symbols_input:
            print("❌ Please enter at least one symbol.")
            continue

        # Process symbols
        symbols = []
        for symbol in symbols_input.split(','):
            symbol = symbol.strip().upper()
            if symbol:
                if not symbol.endswith('.NS'):
                    symbol += '.NS'
                symbols.append(symbol)

        if symbols:
            return symbols
        else:
            print("❌ No valid symbols entered.")

def show_recent_results():
    """Show recent analysis results"""
    reports_dir = PROJECT_ROOT / "output" / "reports"

    if not reports_dir.exists():
        print("❌ No reports directory found.")
        return

    # Get recent CSV files
    csv_files = list(reports_dir.glob("simplified_analysis_*.csv"))

    if not csv_files:
        print("❌ No recent analysis results found.")
        return

    # Sort by modification time (newest first)
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print("\n📊 RECENT ANALYSIS RESULTS:")
    print("-" * 50)

    for i, file in enumerate(csv_files[:5], 1):  # Show last 5
        mtime = datetime.fromtimestamp(file.stat().st_mtime)
        print(f"{i}. {file.name}")
        print(f"   📅 {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    choice = input("👉 Enter number to view details (or press Enter to continue): ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(csv_files[:5]):
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[int(choice)-1])
            print(f"\n📈 ANALYSIS DETAILS:")
            print("-" * 50)
            for _, row in df.iterrows():
                print(f"🏢 {row['Symbol']:<12} | ₹{row['Price']:>8.2f} | Score: {row['Score']:>3} | {row['Signal']:>6}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

def show_system_info():
    """Show system information and status"""
    print("\n🔧 SYSTEM INFORMATION:")
    print("-" * 40)

    # Python version
    print(f"🐍 Python: {sys.version.split()[0]}")

    # Check dependencies
    try:
        import pandas as pd
        print(f"📊 Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas: Not installed")

    try:
        import numpy as np
        print(f"🔢 NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy: Not installed")

    try:
        import yfinance as yf
        print(f"💹 yfinance: {yf.__version__}")
    except ImportError:
        print("❌ yfinance: Not installed")

    try:
        import matplotlib
        print(f"📈 Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib: Not installed")

    # Project paths
    print(f"\n📁 Project Root: {PROJECT_ROOT}")

    # Output directories
    output_dir = PROJECT_ROOT / "output"
    reports_dir = output_dir / "reports"

    print(f"📂 Output Directory: {'✅' if output_dir.exists() else '❌'}")
    print(f"📄 Reports Directory: {'✅' if reports_dir.exists() else '❌'}")

    # Recent activity
    if reports_dir.exists():
        csv_files = list(reports_dir.glob("simplified_analysis_*.csv"))
        print(f"📊 Recent Reports: {len(csv_files)}")

def main():
    """Main interactive launcher"""
    try:
        while True:
            clear_screen()
            print_header()
            print_menu()

            choice = get_user_choice()

            if choice == 1:  # Banking Sector
                # Use comprehensive sector filtering from all NSE stocks
                try:
                    from nse_sector_filter import filter_banking_stocks
                    
                    print("� Filtering banking stocks from complete NSE universe...")
                    banking_stocks = filter_banking_stocks(top_n=20)  # Get top 20 banking stocks
                    
                    if banking_stocks:
                        run_analysis(banking_stocks, f"Comprehensive Banking Sector Analysis ({len(banking_stocks)} top stocks)")
                    else:
                        print("❌ No banking stocks found. Using fallback.")
                        symbols = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                        run_analysis(symbols, "Banking Sector Analysis (Fallback)")
                
                except ImportError as e:
                    print(f"❌ Error loading sector filter: {e}")
                    print("⚠️ Using fallback banking stocks")
                    symbols = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                    run_analysis(symbols, "Banking Sector Analysis (Fallback)")
                except Exception as e:
                    print(f"❌ Error in sector filtering: {e}")
                    print("⚠️ Using fallback banking stocks")
                    symbols = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                    run_analysis(symbols, "Banking Sector Analysis (Fallback)")

            elif choice == 2:  # Technology Sector
                # Use comprehensive sector filtering from all NSE stocks
                try:
                    from nse_sector_filter import filter_technology_stocks
                    
                    print("� Filtering technology stocks from complete NSE universe...")
                    tech_stocks = filter_technology_stocks(top_n=20)  # Get top 20 tech stocks
                    
                    if tech_stocks:
                        run_analysis(tech_stocks, f"Comprehensive Technology Sector Analysis ({len(tech_stocks)} top stocks)")
                    else:
                        print("❌ No technology stocks found. Using fallback.")
                        symbols = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
                        run_analysis(symbols, "Technology Sector Analysis (Fallback)")
                
                except ImportError as e:
                    print(f"❌ Error loading sector filter: {e}")
                    print("⚠️ Using fallback technology stocks")
                    symbols = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
                    run_analysis(symbols, "Technology Sector Analysis (Fallback)")
                except Exception as e:
                    print(f"❌ Error in sector filtering: {e}")
                    print("⚠️ Using fallback technology stocks")
                    symbols = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
                    run_analysis(symbols, "Technology Sector Analysis (Fallback)")

            elif choice == 3:  # Custom Analysis
                symbols = get_custom_symbols()
                if symbols:
                    run_analysis(symbols, f"Custom Analysis - {len(symbols)} stocks")

            elif choice == 4:  # Portfolio Watchlist
                watchlist = show_predefined_options()
                if watchlist:
                    run_analysis(watchlist["symbols"], f"Portfolio Analysis - {watchlist['name']}")

            elif choice == 5:  # Search & Analyze
                print("\n🔍 SEARCH & ANALYZE:")
                print("💡 Search across 2,100+ real NSE symbols...")
                search_term = input("👉 Search: ").strip().upper()
                if search_term:
                    try:
                        # Use dynamic watchlist manager for search
                        from src.dynamic_watchlist_manager import DynamicWatchlistManager
                        dwm = DynamicWatchlistManager()
                        matches = dwm.search_symbol(search_term)
                        
                        if matches:
                            # Show search results
                            print(f"\n✅ Found {len(matches)} matches:")
                            for i, symbol in enumerate(matches, 1):
                                print(f"  {i}. {symbol.replace('.NS', '')}")
                            
                            # Enhanced selection options
                            print(f"\n� ANALYSIS OPTIONS:")
                            print(f"   • Type 'all' to analyze all {len(matches)} stocks")
                            print(f"   • Type specific numbers (e.g., '3,6,9') to analyze selected stocks")
                            print(f"   • Type single number (e.g., '6') to analyze one stock")
                            print(f"   • Type 'n' to go back")
                            
                            selection = input("👉 Your choice: ").strip().lower()
                            
                            if selection == 'n':
                                continue
                            elif selection == 'all':
                                run_analysis(matches, f"Search Results - All {len(matches)} matches for '{search_term}'")
                            else:
                                # Parse selection
                                selected_symbols = []
                                try:
                                    # Handle comma-separated numbers or single number
                                    if ',' in selection:
                                        # Multiple selections
                                        indices = [int(x.strip()) for x in selection.split(',')]
                                    else:
                                        # Single selection
                                        indices = [int(selection)]
                                    
                                    # Validate indices and collect symbols
                                    valid_indices = []
                                    for idx in indices:
                                        if 1 <= idx <= len(matches):
                                            selected_symbols.append(matches[idx - 1])
                                            valid_indices.append(idx)
                                        else:
                                            print(f"⚠️ Invalid index: {idx} (valid range: 1-{len(matches)})")
                                    
                                    if selected_symbols:
                                        # Show selected stocks
                                        selected_names = [s.replace('.NS', '') for s in selected_symbols]
                                        print(f"\n✅ Selected {len(selected_symbols)} stock(s): {', '.join(selected_names)}")
                                        
                                        confirm = input("👉 Proceed with analysis? (y/n): ").lower()
                                        if confirm == 'y':
                                            description = f"Search Results - Selected stocks ({', '.join(selected_names)}) for '{search_term}'"
                                            run_analysis(selected_symbols, description)
                                    else:
                                        print("❌ No valid stocks selected.")
                                        
                                except ValueError:
                                    print("❌ Invalid format. Use numbers separated by commas (e.g., 3,6,9) or 'all'")
                        else:
                            print(f"❌ No matches found for '{search_term}' in NSE database.")
                            print("💡 Try: RELIANCE, TATA, INFY, HDFC, ICICI, SBI, BAJAJ, MARUTI")
                    except ImportError:
                        # Fallback to basic search
                        print("⚠️ Using basic search (dynamic search unavailable)")
                        common_stocks = {
                            "REL": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFOSYS.NS",
                            "HDFC": "HDFCBANK.NS", "ICICI": "ICICIBANK.NS", "SBI": "SBIN.NS",
                            "ITC": "ITC.NS", "BAJAJ": "BAJFINANCE.NS", "WIPRO": "WIPRO.NS"
                        }
                        matches = [v for k, v in common_stocks.items() if search_term in k]
                        if matches:
                            print(f"\n✅ Found matches: {', '.join([s.replace('.NS', '') for s in matches])}")
                            if input("👉 Analyze these? (y/n): ").lower() == 'y':
                                run_analysis(matches, f"Search Results - {search_term}")
                        else:
                            print("❌ No matches found. Try: REL, TCS, INFY, HDFC, ICICI, SBI, ITC, BAJAJ, WIPRO")

            elif choice == 6:  # View Recent Results
                show_recent_results()

            elif choice == 7:  # System Information
                show_system_info()

            elif choice == 8:  # Exit
                print("\n👋 Thank you for using NSE Stock Screener!")
                print("📈 Happy Trading!")
                break

            # Pause before returning to menu
            if choice != 8:
                input("\n⏸️  Press Enter to continue...")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔧 Please check your setup and try again.")

if __name__ == "__main__":
    main()