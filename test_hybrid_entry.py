#!/usr/bin/env python3
"""
Test script for hybrid entry system validation
Tests the complete hybrid entry system implementation
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path (though explicit imports are now used for clarity)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.constants import PROJECT_ROOT_PATH
from src.core import DataFetcher, DataValidator
from src.risk_manager import RiskManager
from src.optimal_entry_calculator import OptimalEntryCalculator
from src.strategic_entry import calculate_strategic_entry
from src.utils.marketcap_utils import detect_market_cap_from_volume

def test_hybrid_entry_system():
    """Test the complete hybrid entry system"""
    print("🧪 TESTING HYBRID ENTRY SYSTEM")
    print("=" * 50)

    # Test stocks with different scenarios
    test_stocks = [
        "BAJFINANCE.NS",  # High probability BUY signal
        "RELIANCE.NS",    # Large cap
        "ITC.NS",         # Mid cap
    ]

    results = []

    for symbol in test_stocks:
        print(f"\n📊 Testing {symbol}...")

        try:
            # Fetch data
            data = DataFetcher.fetch_stock_data(symbol, "3mo")
            if data is None or data.empty:
                print(f"❌ No data for {symbol}")
                continue

            # Validate data
            if not DataValidator.validate_ohlcv_data(data):
                print(f"❌ Invalid data for {symbol}")
                continue

            # Get current price and indicators
            current_price = data['Close'].iloc[-1]
            atr = data['Close'].rolling(14).std().iloc[-1]  # Simple ATR approximation

            # Create indicators dict for risk manager
            indicators = {
                'current_price': current_price,
                'atr': atr,
                'rsi': 65.0,  # Mock RSI for BUY signal
                'avg_volume': data['Volume'].mean()
            }

            # Initialize components
            risk_manager = RiskManager(initial_capital=100000)  # 1 lakh initial capital

            # Test hybrid system via risk manager
            risk_info = risk_manager.calculate_entry_stop_target(
                signal="BUY",  # Force BUY signal for testing
                current_price=current_price,
                indicators=indicators,
                symbol=symbol,
                historical_data=data
            )

            result = {
                'symbol': symbol,
                'current_price': current_price,
                'hybrid_entry': risk_info.get('entry_value'),
                'entry_method': risk_info.get('entry_method', 'UNAVAILABLE'),
                'validation_flag': risk_info.get('validation_flag', 'UNKNOWN'),
                'validation_message': risk_info.get('validation_message', ''),
                'calculation_method': risk_info.get('calculation_method', 'UNKNOWN'),
                'fallback_used': risk_info.get('fallback_used', 'UNKNOWN')
            }

            results.append(result)

            print(f"✅ {symbol}:")
            print(f"   Current: ₹{current_price:.2f}")
            print(f"   Hybrid Entry: ₹{risk_info.get('entry_value', 'N/A'):.2f}")
            print(f"   Method: {result['entry_method']}")
            print(f"   Validation: {result['validation_flag']}")
            if result['validation_message']:
                print(f"   Message: {result['validation_message']}")

        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue

    # Create test results DataFrame
    if results:
        df = pd.DataFrame(results)

        # Save test results
        test_file = PROJECT_ROOT_PATH / 'output' / 'test_hybrid_entry_results.csv'
        df.to_csv(str(test_file), index=False)
        print(f"\n💾 Test results saved: {test_file}")

        # Calculate CI gate metrics
        entries_equal_current = (df['hybrid_entry'] == df['current_price']).sum()
        total_entries = len(df)
        entries_equal_current_pct = (entries_equal_current / total_entries) * 100 if total_entries > 0 else 0

        print("\n🎯 CI GATE VALIDATION:")
        print(f"   Total entries tested: {total_entries}")
        print(f"   Entries equal to current price: {entries_equal_current}")
        print(f"   Percentage: {entries_equal_current_pct:.1f}%")
        print(f"   CI Gate (≤30%): {'✅ PASS' if entries_equal_current_pct <= 30 else '❌ FAIL'}")
        print(f"   Target (≤15%): {'✅ PASS' if entries_equal_current_pct <= 15 else '❌ FAIL'}")

        return {
            'entries_equal_current_count': entries_equal_current,
            'entries_equal_current_pct': entries_equal_current_pct,
            'total_tested': total_entries,
            'ci_gate_pass': entries_equal_current_pct <= 30,
            'target_pass': entries_equal_current_pct <= 15
        }

    return None

if __name__ == "__main__":
    test_results = test_hybrid_entry_system()

    if test_results:
        print("\n🏁 TEST SUMMARY:")
        print(f"   CI Gate Status: {'PASS' if test_results['ci_gate_pass'] else 'FAIL'}")
        print(f"   Target Status: {'PASS' if test_results['target_pass'] else 'FAIL'}")

        if test_results['ci_gate_pass']:
            print("🎉 Hybrid entry system successfully reduces current_price fallback!")
        else:
            print("⚠️  Hybrid entry system needs further optimization")
    else:
        print("❌ No test results generated")