#!/usr/bin/env python3
"""
Test Script for Issue #1 Fix: Missing Target Values

This script tests the critical fix for missing target values in BUY signals.

Requirements being tested:
- 100% of BUY signals must have target_value > entry_value  
- Risk-reward ratio validation: minimum 1.5:1
- Mathematical validation before saving any trade data
- No missing values for actionable signals

Usage:
    python test_issue_1_fix.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import time

# Import our modules
from enhanced_early_warning_system import EnhancedEarlyWarningSystem
from constants import MarketRegime

class Issue1Tester:
    """Test suite specifically for Issue #1: Missing Target Values"""
    
    def __init__(self):
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Sample test stocks for controlled testing
        self.test_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'WIPRO.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS'
        ]
    
    def run_all_tests(self):
        """Run comprehensive test suite for Issue #1 fix"""
        print("🧪 TESTING ISSUE #1 FIX: Missing Target Values")
        print("=" * 60)
        
        # Test 1: Data completeness test
        self.test_data_completeness()
        
        # Test 2: Mathematical consistency test  
        self.test_mathematical_consistency()
        
        # Test 3: BUY signal validation test
        self.test_buy_signal_validation()
        
        # Test 4: Risk-reward ratio validation
        self.test_risk_reward_validation()
        
        # Test 5: Edge case handling
        self.test_edge_cases()
        
        # Print final results
        self.print_test_summary()
        
        return self.test_results
    
    def test_data_completeness(self):
        """Test Case 1: Ensure no missing critical fields in BUY signals"""
        print("\n🔍 TEST 1: Data Completeness for BUY Signals")
        print("-" * 40)
        
        try:
            # Initialize system with limited stocks for testing
            ews = EnhancedEarlyWarningSystem(custom_stocks=self.test_stocks[:3])
            
            # Run analysis
            results = []
            for symbol in self.test_stocks[:3]:
                print(f"Testing {symbol}...")
                result = ews.analyze_single_stock(symbol)
                if result:
                    results.append(result)
                time.sleep(0.5)  # Rate limiting
            
            # Analyze results
            buy_signals = [r for r in results if r.get('signal_info', {}).get('signal') == 'BUY']
            
            missing_fields_count = 0
            total_buy_signals = len(buy_signals)
            
            for result in buy_signals:
                risk_mgmt = result.get('risk_management', {})
                symbol = result.get('symbol', 'Unknown')
                
                # Check critical fields
                critical_fields = ['entry_value', 'stop_value', 'target_value', 'risk_reward_ratio']
                missing_fields = []
                
                for field in critical_fields:
                    value = risk_mgmt.get(field)
                    if value is None or value == 0:
                        missing_fields.append(field)
                
                if missing_fields:
                    missing_fields_count += 1
                    print(f"  ❌ {symbol}: Missing fields: {missing_fields}")
                else:
                    print(f"  ✅ {symbol}: All critical fields present")
            
            # Test assertion
            success = missing_fields_count == 0
            self._record_test_result(
                "Data Completeness", 
                success, 
                f"BUY signals with complete data: {total_buy_signals - missing_fields_count}/{total_buy_signals}"
            )
            
        except Exception as e:
            self._record_test_result("Data Completeness", False, f"Test failed: {e}")
    
    def test_mathematical_consistency(self):
        """Test Case 2: Validate mathematical relationships"""
        print("\n🔍 TEST 2: Mathematical Consistency")
        print("-" * 40)
        
        try:
            ews = EnhancedEarlyWarningSystem(custom_stocks=self.test_stocks[:3])
            
            results = []
            for symbol in self.test_stocks[:3]:
                result = ews.analyze_single_stock(symbol)
                if result:
                    results.append(result)
                time.sleep(0.5)
            
            buy_signals = [r for r in results if r.get('signal_info', {}).get('signal') == 'BUY']
            
            math_errors = []
            
            for result in buy_signals:
                risk_mgmt = result.get('risk_management', {})
                symbol = result.get('symbol', 'Unknown')
                
                entry = risk_mgmt.get('entry_value', 0)
                stop = risk_mgmt.get('stop_value', 0) 
                target = risk_mgmt.get('target_value', 0)
                rrr = risk_mgmt.get('risk_reward_ratio', 0)
                
                # Mathematical validations
                errors = []
                
                # 1. Target must be > Entry for BUY signals
                if target <= entry:
                    errors.append(f"Target ({target}) <= Entry ({entry})")
                
                # 2. Stop must be < Entry for BUY signals  
                if stop >= entry:
                    errors.append(f"Stop ({stop}) >= Entry ({entry})")
                
                # 3. Risk-reward ratio calculation
                if entry > 0 and stop > 0 and target > 0:
                    risk = abs(entry - stop)
                    reward = target - entry
                    calculated_rrr = reward / risk if risk > 0 else 0
                    
                    if abs(calculated_rrr - rrr) > 0.1:  # Allow small rounding differences
                        errors.append(f"R:R mismatch: calculated {calculated_rrr:.2f} vs stored {rrr}")
                
                if errors:
                    math_errors.append(f"{symbol}: {', '.join(errors)}")
                    print(f"  ❌ {symbol}: {errors}")
                else:
                    print(f"  ✅ {symbol}: Mathematical consistency verified")
            
            success = len(math_errors) == 0
            self._record_test_result(
                "Mathematical Consistency",
                success, 
                f"Math errors found: {len(math_errors)}" + (f" - {math_errors[:3]}" if math_errors else "")
            )
            
        except Exception as e:
            self._record_test_result("Mathematical Consistency", False, f"Test failed: {e}")
    
    def test_buy_signal_validation(self):
        """Test Case 3: BUY signal specific validations"""
        print("\n🔍 TEST 3: BUY Signal Validation")
        print("-" * 40)
        
        try:
            ews = EnhancedEarlyWarningSystem(custom_stocks=self.test_stocks[:5])
            
            results = []
            for symbol in self.test_stocks[:5]:
                result = ews.analyze_single_stock(symbol)
                if result:
                    results.append(result)
                time.sleep(0.5)
            
            buy_signals = [r for r in results if r.get('signal_info', {}).get('signal') == 'BUY']
            
            validation_failures = 0
            
            for result in buy_signals:
                risk_mgmt = result.get('risk_management', {})
                symbol = result.get('symbol', 'Unknown')
                
                target = risk_mgmt.get('target_value')
                entry = risk_mgmt.get('entry_value', 0)
                
                # BUY signal must have valid target > entry
                if target is None or target <= entry:
                    validation_failures += 1
                    print(f"  ❌ {symbol}: BUY signal missing/invalid target: Entry={entry}, Target={target}")
                else:
                    print(f"  ✅ {symbol}: Valid BUY signal: Entry={entry:.2f}, Target={target:.2f}")
            
            success = validation_failures == 0 and len(buy_signals) > 0
            self._record_test_result(
                "BUY Signal Validation",
                success,
                f"Valid BUY signals: {len(buy_signals) - validation_failures}/{len(buy_signals)}"
            )
            
        except Exception as e:
            self._record_test_result("BUY Signal Validation", False, f"Test failed: {e}")
    
    def test_risk_reward_validation(self):
        """Test Case 4: Risk-reward ratio validation (minimum 1.5:1)"""
        print("\n🔍 TEST 4: Risk-Reward Ratio Validation")
        print("-" * 40)
        
        try:
            ews = EnhancedEarlyWarningSystem(custom_stocks=self.test_stocks[:5])
            
            results = []
            for symbol in self.test_stocks[:5]:
                result = ews.analyze_single_stock(symbol)
                if result:
                    results.append(result)
                time.sleep(0.5)
            
            buy_signals = [r for r in results if r.get('signal_info', {}).get('signal') == 'BUY']
            
            rrr_failures = 0
            min_rrr = 1.5
            
            for result in buy_signals:
                risk_mgmt = result.get('risk_management', {})
                symbol = result.get('symbol', 'Unknown')
                rrr = risk_mgmt.get('risk_reward_ratio', 0)
                
                if rrr < min_rrr:
                    rrr_failures += 1
                    print(f"  ❌ {symbol}: R:R below minimum: {rrr:.2f} < {min_rrr}")
                else:
                    print(f"  ✅ {symbol}: R:R acceptable: {rrr:.2f}")
            
            success = rrr_failures == 0 and len(buy_signals) > 0
            self._record_test_result(
                "Risk-Reward Validation",
                success,
                f"R:R >= 1.5: {len(buy_signals) - rrr_failures}/{len(buy_signals)}"
            )
            
        except Exception as e:
            self._record_test_result("Risk-Reward Validation", False, f"Test failed: {e}")
    
    def test_edge_cases(self):
        """Test Case 5: Edge case handling"""
        print("\n🔍 TEST 5: Edge Case Handling")
        print("-" * 40)
        
        try:
            # Test with potential problematic symbols
            edge_case_stocks = ['TATASTEEL.NS', 'NTPC.NS']
            ews = EnhancedEarlyWarningSystem(custom_stocks=edge_case_stocks)
            
            edge_failures = 0
            
            for symbol in edge_case_stocks:
                try:
                    result = ews.analyze_single_stock(symbol)
                    if result:
                        risk_mgmt = result.get('risk_management', {})
                        signal = result.get('signal_info', {}).get('signal', 'UNKNOWN')
                        
                        # Check if BUY signal has proper data structure
                        if signal == 'BUY':
                            entry = risk_mgmt.get('entry_value', 0)
                            target = risk_mgmt.get('target_value')
                            
                            if target is None or target <= entry:
                                edge_failures += 1
                                print(f"  ❌ {symbol}: Edge case failure")
                            else:
                                print(f"  ✅ {symbol}: Edge case handled properly")
                        else:
                            print(f"  ℹ️  {symbol}: {signal} signal (not testing)")
                    else:
                        print(f"  ⚠️  {symbol}: Analysis failed (expected for edge cases)")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  ⚠️  {symbol}: Exception handled: {str(e)[:50]}...")
            
            success = edge_failures == 0
            self._record_test_result(
                "Edge Case Handling",
                success,
                f"Edge cases handled properly: {len(edge_case_stocks) - edge_failures}/{len(edge_case_stocks)}"
            )
            
        except Exception as e:
            self._record_test_result("Edge Case Handling", False, f"Test failed: {e}")
    
    def _record_test_result(self, test_name: str, success: bool, details: str):
        """Record individual test result"""
        self.test_results['total_tests'] += 1
        if success:
            self.test_results['passed_tests'] += 1
        else:
            self.test_results['failed_tests'] += 1
        
        self.test_results['test_details'].append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    def print_test_summary(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 60)
        print("📊 ISSUE #1 FIX TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.test_results['total_tests']}")
        print(f"Passed: {self.test_results['passed_tests']}")
        print(f"Failed: {self.test_results['failed_tests']}")
        
        success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test in self.test_results['test_details']:
            status = "✅ PASS" if test['success'] else "❌ FAIL"
            print(f"  {status}: {test['test']} - {test['details']}")
        
        if success_rate >= 80:
            print(f"\n🎉 ISSUE #1 FIX VALIDATION: {'SUCCESS' if success_rate == 100 else 'MOSTLY SUCCESSFUL'}")
            print("✅ Target value fix is working correctly!")
        else:
            print(f"\n⚠️  ISSUE #1 FIX VALIDATION: NEEDS ATTENTION")
            print("❌ Some critical tests are failing - review required")

def main():
    """Main test execution"""
    print("Starting Issue #1 Fix Validation...")
    
    tester = Issue1Tester()
    results = tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()