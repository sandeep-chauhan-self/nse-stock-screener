#!/usr/bin/env python3
"""
Analysis script to validate hybrid entry system improvements
Compares current results with baseline to show reduction in current_price fallback
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def analyze_hybrid_entry_improvements():
    """Analyze the improvements from the hybrid entry system"""

    print("📊 HYBRID ENTRY SYSTEM IMPROVEMENT ANALYSIS")
    print("=" * 60)

    # Load the latest comprehensive analysis
    reports_dir = Path('output/reports')
    csv_files = list(reports_dir.glob('comprehensive_analysis_*.csv'))

    if not csv_files:
        print("❌ No comprehensive analysis files found")
        return

    # Get the latest file
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analyzing: {latest_csv}")

    # Load the data
    df = pd.read_csv(str(latest_csv))

    # Filter for BUY signals only (where entry calculation matters)
    buy_signals = df[df['Signal'] == 'BUY'].copy()

    if buy_signals.empty:
        print("❌ No BUY signals found in the analysis")
        return

    print(f"📈 Found {len(buy_signals)} BUY signals for analysis")

    # Calculate key metrics
    total_buy_signals = len(buy_signals)

    # Check entries equal to current price
    entries_equal_current = (buy_signals['Optimal_Entry'] == buy_signals['Current_Price']).sum()
    entries_equal_current_pct = (entries_equal_current / total_buy_signals) * 100

    # Check for different entry methods
    entry_methods = buy_signals['entry_method'].value_counts()
    validation_flags = buy_signals['validation_flag'].value_counts()

    # Check for strategic entries (not current price)
    strategic_entries = buy_signals[buy_signals['entry_method'] != 'CURRENT_PRICE']
    strategic_entries_pct = (len(strategic_entries) / total_buy_signals) * 100

    print("\n🎯 HYBRID ENTRY SYSTEM RESULTS:")
    print(f"   Total BUY signals analyzed: {total_buy_signals}")
    print(f"   Entries equal to current price: {entries_equal_current}")
    print(f"   Percentage equal to current price: {entries_equal_current_pct:.1f}%")

    print("\n📊 ENTRY METHOD DISTRIBUTION:")
    for method, count in entry_methods.items():
        pct = (count / total_buy_signals) * 100
        print(f"   {method}: {count} ({pct:.1f}%)")

    print("\n✅ VALIDATION RESULTS:")
    for flag, count in validation_flags.items():
        pct = (count / total_buy_signals) * 100
        status = "✅" if flag == "PASS" else "⚠️" if flag == "REVIEW" else "❌"
        print(f"   {status} {flag}: {count} ({pct:.1f}%)")

    # CI Gate validation
    ci_gate_pass = entries_equal_current_pct <= 30
    target_pass = entries_equal_current_pct <= 15

    print("\n🚦 CI GATE VALIDATION:")
    print(f"   CI Gate (≤30% entries = current price): {'✅ PASS' if ci_gate_pass else '❌ FAIL'}")
    print(f"   Target (≤15% entries = current price): {'✅ PASS' if target_pass else '❌ FAIL'}")

    # Show improvement from baseline (93.8% from the issue)
    baseline_pct = 93.8
    improvement = baseline_pct - entries_equal_current_pct

    print("\n📈 IMPROVEMENT METRICS:")
    print(f"   Baseline (Issue #2): {baseline_pct:.1f}% entries = current price")
    print(f"   Current result: {entries_equal_current_pct:.1f}% entries = current price")
    print(f"   Improvement: {improvement:.1f} percentage points")
    print(f"   Reduction factor: {baseline_pct / max(entries_equal_current_pct, 0.1):.1f}x")

    # Show sample entries that are different
    if not strategic_entries.empty:
        print("\n💡 SAMPLE IMPROVED ENTRIES:")
        sample = strategic_entries.head(3)[['Symbol', 'Current_Price', 'Optimal_Entry', 'entry_method', 'validation_flag']]
        for _, row in sample.iterrows():
            diff = row['Optimal_Entry'] - row['Current_Price']
            print(f"   {row['Symbol']}: ₹{row['Current_Price']:.2f} → ₹{row['Optimal_Entry']:.2f} "
                  f"(+₹{diff:.2f}) via {row['entry_method']} [{row['validation_flag']}]")

    # Summary
    print("\n🏁 SUMMARY:")
    if ci_gate_pass:
        print("   ✅ Hybrid entry system successfully implemented!")
        print("   ✅ CI gates passed - production ready!")
        print(f"   ✅ Reduced current_price fallback from {baseline_pct:.1f}% to {entries_equal_current_pct:.1f}%")
    else:
        print("   ⚠️  CI gates not fully met - further optimization needed")
        print(f"   ❌ Current: {entries_equal_current_pct:.1f}% (target: ≤30%)")

    return {
        'total_buy_signals': total_buy_signals,
        'entries_equal_current': entries_equal_current,
        'entries_equal_current_pct': entries_equal_current_pct,
        'ci_gate_pass': ci_gate_pass,
        'target_pass': target_pass,
        'improvement': improvement,
        'entry_methods': entry_methods.to_dict(),
        'validation_flags': validation_flags.to_dict()
    }

if __name__ == "__main__":
    results = analyze_hybrid_entry_improvements()

    if results:
        print("\n📋 RAW METRICS:")
        for key, value in results.items():
            print(f"   {key}: {value}")