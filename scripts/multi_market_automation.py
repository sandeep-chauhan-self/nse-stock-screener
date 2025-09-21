#!/usr/bin/env python3
"""
NSE Stock Screener - Multi-Market Automation
Advanced automation across multiple markets and timeframes
"""

import sys
import json
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from automation_manager import AutomationManager
from generate_summary import SummaryGenerator

class MultiMarketAutomation:
    """Advanced multi-market automation system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.config = self.load_config(config_file)
        self.base_manager = AutomationManager()
        self.results = {}
        
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load multi-market configuration"""
        default_config = {
            "markets": {
                "NSE": {
                    "enabled": True,
                    "symbols_file": "data/nse_only_symbols.txt",
                    "analysis_params": {
                        "min_score": 60,
                        "volume_threshold": 1.5,
                        "max_results": 100
                    }
                },
                "BSE": {
                    "enabled": False,
                    "symbols_file": "data/bse_symbols.txt", 
                    "analysis_params": {
                        "min_score": 65,
                        "volume_threshold": 2.0,
                        "max_results": 50
                    }
                }
            },
            "parallel_processing": {
                "enabled": True,
                "max_workers": 4,
                "batch_size": 50
            },
            "cross_market_analysis": {
                "arbitrage_detection": True,
                "correlation_analysis": True,
                "relative_strength": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                print(f"[WARNING] Config load error: {e}, using defaults")
        
        return default_config
    
    def run_multi_market_analysis(self, markets: List[str] = None) -> Dict:
        """Run analysis across multiple markets"""
        markets = markets or list(self.config["markets"].keys())
        enabled_markets = [m for m in markets if self.config["markets"][m]["enabled"]]
        
        print(f"[NETWORK] Starting multi-market analysis for: {', '.join(enabled_markets)}")
        
        results = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "markets_analyzed": enabled_markets,
            "start_time": datetime.now().isoformat(),
            "market_results": {},
            "cross_market_analysis": {},
            "summary": {}
        }
        
        # Run parallel market analysis
        if self.config["parallel_processing"]["enabled"]:
            results["market_results"] = self.run_parallel_market_analysis(enabled_markets)
        else:
            results["market_results"] = self.run_sequential_market_analysis(enabled_markets)
        
        # Cross-market analysis
        if len(enabled_markets) > 1:
            results["cross_market_analysis"] = self.run_cross_market_analysis(results["market_results"])
        
        # Generate summary
        results["summary"] = self.generate_multi_market_summary(results)
        
        # Save results
        self.save_multi_market_results(results)
        
        return results
    
    def run_parallel_market_analysis(self, markets: List[str]) -> Dict:
        """Run market analysis in parallel"""
        market_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config["parallel_processing"]["max_workers"]
        ) as executor:
            
            # Submit analysis jobs for each market
            future_to_market = {
                executor.submit(self.analyze_single_market, market): market 
                for market in markets
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_market):
                market = future_to_market[future]
                try:
                    result = future.result()
                    market_results[market] = result
                    print(f"[OK] {market} analysis completed")
                except Exception as e:
                    print(f"[ERROR] {market} analysis failed: {e}")
                    market_results[market] = {"error": str(e)}
        
        return market_results
    
    def run_sequential_market_analysis(self, markets: List[str]) -> Dict:
        """Run market analysis sequentially"""
        market_results = {}
        
        for market in markets:
            print(f"[ANALYSIS] Analyzing {market}...")
            try:
                result = self.analyze_single_market(market)
                market_results[market] = result
                print(f"[OK] {market} analysis completed")
            except Exception as e:
                print(f"[ERROR] {market} analysis failed: {e}")
                market_results[market] = {"error": str(e)}
        
        return market_results
    
    def analyze_single_market(self, market: str) -> Dict:
        """Analyze a single market"""
        market_config = self.config["markets"][market]
        
        # Simulate market-specific analysis
        # In production, this would call the actual analysis system
        result = {
            "market": market,
            "analysis_time": datetime.now().isoformat(),
            "symbols_analyzed": 150,  # Simulated
            "high_score_stocks": 12,   # Simulated
            "entry_ready": 8,          # Simulated
            "avg_score": 58.5,         # Simulated
            "top_performers": [
                {"symbol": f"{market}_STOCK1", "score": 78.5},
                {"symbol": f"{market}_STOCK2", "score": 76.2},
                {"symbol": f"{market}_STOCK3", "score": 74.8}
            ],
            "sector_analysis": {
                "Banking": {"count": 15, "avg_score": 62.3},
                "Technology": {"count": 12, "avg_score": 65.8},
                "Pharma": {"count": 8, "avg_score": 59.2}
            }
        }
        
        return result
    
    def run_cross_market_analysis(self, market_results: Dict) -> Dict:
        """Perform cross-market analysis"""
        cross_analysis = {
            "arbitrage_opportunities": [],
            "correlation_analysis": {},
            "relative_strength": {},
            "cross_market_trends": {}
        }
        
        markets = list(market_results.keys())
        
        # Arbitrage detection
        if self.config["cross_market_analysis"]["arbitrage_detection"]:
            cross_analysis["arbitrage_opportunities"] = self.detect_arbitrage_opportunities(market_results)
        
        # Correlation analysis
        if self.config["cross_market_analysis"]["correlation_analysis"]:
            cross_analysis["correlation_analysis"] = self.analyze_market_correlations(market_results)
        
        # Relative strength
        if self.config["cross_market_analysis"]["relative_strength"]:
            cross_analysis["relative_strength"] = self.analyze_relative_strength(market_results)
        
        return cross_analysis
    
    def detect_arbitrage_opportunities(self, market_results: Dict) -> List[Dict]:
        """Detect arbitrage opportunities across markets"""
        opportunities = []
        
        # Simulated arbitrage detection
        opportunities.append({
            "symbol": "RELIANCE",
            "nse_price": 2450.00,
            "bse_price": 2455.50,
            "spread": 5.50,
            "spread_pct": 0.22,
            "opportunity_type": "NSE_BUY_BSE_SELL"
        })
        
        return opportunities
    
    def analyze_market_correlations(self, market_results: Dict) -> Dict:
        """Analyze correlations between markets"""
        # Simulated correlation analysis
        return {
            "NSE_BSE_correlation": 0.85,
            "sector_correlations": {
                "Banking": 0.92,
                "Technology": 0.78,
                "Pharma": 0.65
            },
            "trend_alignment": "HIGH"
        }
    
    def analyze_relative_strength(self, market_results: Dict) -> Dict:
        """Analyze relative strength between markets"""
        # Simulated relative strength analysis
        return {
            "market_strength_ranking": ["NSE", "BSE"],
            "sector_strength": {
                "Technology": "NSE_STRONGER",
                "Banking": "EQUAL",
                "Pharma": "BSE_STRONGER"
            }
        }
    
    def generate_multi_market_summary(self, results: Dict) -> Dict:
        """Generate comprehensive multi-market summary"""
        market_results = results["market_results"]
        
        total_symbols = sum(r.get("symbols_analyzed", 0) for r in market_results.values() if "error" not in r)
        total_entry_ready = sum(r.get("entry_ready", 0) for r in market_results.values() if "error" not in r)
        
        avg_scores = [r.get("avg_score", 0) for r in market_results.values() if "error" not in r and r.get("avg_score")]
        overall_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
        
        summary = {
            "total_markets": len(results["markets_analyzed"]),
            "successful_markets": len([r for r in market_results.values() if "error" not in r]),
            "total_symbols_analyzed": total_symbols,
            "total_entry_ready": total_entry_ready,
            "overall_avg_score": round(overall_avg_score, 1),
            "cross_market_opportunities": len(results["cross_market_analysis"].get("arbitrage_opportunities", [])),
            "analysis_duration": (datetime.now() - datetime.fromisoformat(results["start_time"])).total_seconds()
        }
        
        return summary
    
    def save_multi_market_results(self, results: Dict):
        """Save multi-market analysis results"""
        output_dir = self.project_root / 'output' / 'multi_market'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / f"multi_market_analysis_{results['session_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[FOLDER] Multi-market results saved: {results_file}")
        
        # Generate summary report
        self.generate_multi_market_report(results)
    
    def generate_multi_market_report(self, results: Dict):
        """Generate human-readable multi-market report"""
        report_lines = [
            "[NETWORK] MULTI-MARKET ANALYSIS REPORT",
            "=" * 50,
            f"Session ID: {results['session_id']}",
            f"Analysis Time: {results['start_time']}",
            f"Markets: {', '.join(results['markets_analyzed'])}",
            "",
            "[ANALYSIS] SUMMARY METRICS",
            "-" * 30,
            f"Total Markets: {results['summary']['total_markets']}",
            f"Successful Markets: {results['summary']['successful_markets']}",
            f"Total Symbols: {results['summary']['total_symbols_analyzed']}",
            f"Entry Ready: {results['summary']['total_entry_ready']}",
            f"Overall Avg Score: {results['summary']['overall_avg_score']}",
            f"Analysis Duration: {results['summary']['analysis_duration']:.1f} seconds",
            "",
            "[SEARCH] MARKET BREAKDOWN",
            "-" * 30
        ]
        
        for market, market_result in results["market_results"].items():
            if "error" in market_result:
                report_lines.append(f"{market}: [ERROR] {market_result['error']}")
            else:
                report_lines.extend([
                    f"{market}:",
                    f"  Symbols: {market_result['symbols_analyzed']}",
                    f"  Entry Ready: {market_result['entry_ready']}",
                    f"  Avg Score: {market_result['avg_score']}",
                    f"  Top Performer: {market_result['top_performers'][0]['symbol']} ({market_result['top_performers'][0]['score']})",
                    ""
                ])
        
        # Cross-market analysis
        if results["cross_market_analysis"]:
            report_lines.extend([
                "[REFRESH] CROSS-MARKET ANALYSIS",
                "-" * 30,
                f"Arbitrage Opportunities: {len(results['cross_market_analysis'].get('arbitrage_opportunities', []))}",
                f"Market Correlation: {results['cross_market_analysis'].get('correlation_analysis', {}).get('NSE_BSE_correlation', 'N/A')}",
                ""
            ])
        
        # Save report
        output_dir = self.project_root / 'output' / 'multi_market'
        report_file = output_dir / f"multi_market_report_{results['session_id']}.txt"
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[LIST] Multi-market report saved: {report_file}")
        
        # Print summary to console
        print("\n" + "\n".join(report_lines[:20]))  # Print first 20 lines
        if len(report_lines) > 20:
            print("... (full report saved to file)")

def main():
    """Main multi-market automation entry point"""
    parser = argparse.ArgumentParser(description='Multi-Market Automation System')
    
    parser.add_argument('--markets', nargs='+', default=['NSE'], 
                        help='Markets to analyze (e.g., NSE BSE)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    try:
        # Create automation system
        automation = MultiMarketAutomation(config_file=args.config)
        
        # Override parallel processing if specified
        if args.parallel:
            automation.config["parallel_processing"]["enabled"] = True
            automation.config["parallel_processing"]["max_workers"] = args.workers
        
        # Run multi-market analysis
        results = automation.run_multi_market_analysis(markets=args.markets)
        
        print(f"\n[SUCCESS] Multi-market analysis completed successfully!")
        print(f"[ANALYSIS] {results['summary']['total_symbols_analyzed']} symbols analyzed across {results['summary']['total_markets']} markets")
        print(f"[TARGET] {results['summary']['total_entry_ready']} entry-ready opportunities found")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"[ERROR] Multi-market analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()