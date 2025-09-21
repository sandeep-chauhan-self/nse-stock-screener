#!/usr/bin/env python3
"""
NSE Stock Screener - Summary Generator
Creates summary reports from analysis results
"""

import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

class SummaryGenerator:
    """Generate summary reports from analysis results"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.reports_dir = self.project_root / 'output' / 'reports'
        self.logs_dir = self.project_root / 'logs'
        
    def get_recent_reports(self, days: int = 1) -> List[Path]:
        """Get recent analysis reports"""
        cutoff_date = datetime.now() - timedelta(days=days)
        reports = []
        
        if self.reports_dir.exists():
            for file in self.reports_dir.glob("enhanced_analysis_*.csv"):
                # Extract timestamp from filename
                try:
                    timestamp_str = file.stem.split('_')[-2] + file.stem.split('_')[-1]
                    file_date = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                    if file_date >= cutoff_date:
                        reports.append(file)
                except (ValueError, IndexError):
                    continue
        
        return sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def analyze_report(self, report_file: Path) -> Dict:
        """Analyze a single report file"""
        try:
            df = pd.read_csv(report_file)
            
            if df.empty:
                return {"error": "Empty report"}
            
            # Basic statistics
            total_stocks = len(df)
            entry_ready = len(df[df.get('Can_Enter', False) == True])
            avg_score = df['Composite_Score'].mean() if 'Composite_Score' in df.columns else 0
            max_score = df['Composite_Score'].max() if 'Composite_Score' in df.columns else 0
            
            # High-scoring stocks
            high_score_stocks = df[df['Composite_Score'] >= 70] if 'Composite_Score' in df.columns else pd.DataFrame()
            
            # Sector analysis (if possible)
            sectors = {}
            if 'Symbol' in df.columns:
                for symbol in df['Symbol']:
                    # Simple sector mapping based on common patterns
                    clean_symbol = symbol.replace('.NS', '')
                    if any(bank in clean_symbol.upper() for bank in ['BANK', 'HDFC', 'ICICI', 'AXIS', 'KOTAK']):
                        sectors['Banking'] = sectors.get('Banking', 0) + 1
                    elif any(tech in clean_symbol.upper() for tech in ['TCS', 'INFY', 'WIPRO', 'TECH', 'INFO']):
                        sectors['Technology'] = sectors.get('Technology', 0) + 1
                    else:
                        sectors['Others'] = sectors.get('Others', 0) + 1
            
            return {
                "file": report_file.name,
                "timestamp": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat(),
                "total_stocks": total_stocks,
                "entry_ready": entry_ready,
                "avg_score": round(avg_score, 1),
                "max_score": round(max_score, 1),
                "high_score_count": len(high_score_stocks),
                "sectors": sectors,
                "top_performers": df.nlargest(5, 'Composite_Score')[['Symbol', 'Composite_Score']].to_dict('records') if 'Composite_Score' in df.columns else []
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_daily_summary(self) -> Dict:
        """Generate daily summary report"""
        recent_reports = self.get_recent_reports(days=1)
        
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "report_count": len(recent_reports),
            "reports": [],
            "totals": {
                "stocks_analyzed": 0,
                "entry_ready": 0,
                "avg_score": 0,
                "high_score_stocks": 0
            }
        }
        
        total_scores = []
        
        for report_file in recent_reports:
            analysis = self.analyze_report(report_file)
            if "error" not in analysis:
                summary["reports"].append(analysis)
                summary["totals"]["stocks_analyzed"] += analysis["total_stocks"]
                summary["totals"]["entry_ready"] += analysis["entry_ready"]
                summary["totals"]["high_score_stocks"] += analysis["high_score_count"]
                total_scores.append(analysis["avg_score"])
        
        if total_scores:
            summary["totals"]["avg_score"] = round(sum(total_scores) / len(total_scores), 1)
        
        return summary
    
    def generate_weekly_summary(self) -> Dict:
        """Generate weekly summary report"""
        recent_reports = self.get_recent_reports(days=7)
        
        summary = {
            "week_ending": datetime.now().strftime("%Y-%m-%d"),
            "report_count": len(recent_reports),
            "daily_summaries": [],
            "weekly_totals": {
                "stocks_analyzed": 0,
                "entry_ready": 0,
                "avg_score": 0,
                "high_score_stocks": 0
            },
            "trends": {}
        }
        
        # Group reports by day
        daily_reports = {}
        for report_file in recent_reports:
            date_key = datetime.fromtimestamp(report_file.stat().st_mtime).strftime("%Y-%m-%d")
            if date_key not in daily_reports:
                daily_reports[date_key] = []
            daily_reports[date_key].append(report_file)
        
        # Analyze each day
        all_scores = []
        for date, reports in sorted(daily_reports.items()):
            day_summary = {
                "date": date,
                "reports": len(reports),
                "stocks_analyzed": 0,
                "entry_ready": 0,
                "avg_score": 0
            }
            
            day_scores = []
            for report_file in reports:
                analysis = self.analyze_report(report_file)
                if "error" not in analysis:
                    day_summary["stocks_analyzed"] += analysis["total_stocks"]
                    day_summary["entry_ready"] += analysis["entry_ready"]
                    day_scores.append(analysis["avg_score"])
            
            if day_scores:
                day_summary["avg_score"] = round(sum(day_scores) / len(day_scores), 1)
                all_scores.extend(day_scores)
            
            summary["daily_summaries"].append(day_summary)
            summary["weekly_totals"]["stocks_analyzed"] += day_summary["stocks_analyzed"]
            summary["weekly_totals"]["entry_ready"] += day_summary["entry_ready"]
            summary["weekly_totals"]["high_score_stocks"] += len([s for s in day_scores if s >= 70])
        
        if all_scores:
            summary["weekly_totals"]["avg_score"] = round(sum(all_scores) / len(all_scores), 1)
        
        return summary
    
    def save_summary(self, summary: Dict, filename: str):
        """Save summary to file"""
        summary_file = self.logs_dir / filename
        self.logs_dir.mkdir(exist_ok=True)
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"[OK] Summary saved to {summary_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save summary: {e}")
    
    def print_summary(self, summary: Dict):
        """Print summary to console"""
        if "date" in summary:  # Daily summary
            print(f"\n[ANALYSIS] DAILY SUMMARY - {summary['date']}")
            print("=" * 50)
            print(f"[FOLDER] Reports analyzed: {summary['report_count']}")
            print(f"[CHART] Stocks analyzed: {summary['totals']['stocks_analyzed']}")
            print(f"[TARGET] Entry-ready stocks: {summary['totals']['entry_ready']}")
            print(f"[STAR] Average score: {summary['totals']['avg_score']}")
            print(f"🏆 High-score stocks (70+): {summary['totals']['high_score_stocks']}")
            
            if summary['reports']:
                print(f"\n[LIST] Recent Reports:")
                for report in summary['reports'][-3:]:  # Show last 3
                    print(f"   • {report['file']}: {report['total_stocks']} stocks, avg {report['avg_score']}")
        
        elif "week_ending" in summary:  # Weekly summary
            print(f"\n📅 WEEKLY SUMMARY - Week ending {summary['week_ending']}")
            print("=" * 60)
            print(f"[FOLDER] Total reports: {summary['report_count']}")
            print(f"[CHART] Stocks analyzed: {summary['weekly_totals']['stocks_analyzed']}")
            print(f"[TARGET] Entry-ready stocks: {summary['weekly_totals']['entry_ready']}")
            print(f"[STAR] Average score: {summary['weekly_totals']['avg_score']}")
            print(f"🏆 High-score stocks (70+): {summary['weekly_totals']['high_score_stocks']}")
            
            if summary['daily_summaries']:
                print(f"\n[LIST] Daily Breakdown:")
                for day in summary['daily_summaries'][-5:]:  # Show last 5 days
                    print(f"   • {day['date']}: {day['stocks_analyzed']} stocks, avg {day['avg_score']}")

def main():
    """Main summary generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE Stock Screener - Summary Generator')
    parser.add_argument('--mode', choices=['daily', 'weekly'], default='daily',
                        help='Summary mode: daily or weekly')
    parser.add_argument('--save', action='store_true',
                        help='Save summary to file')
    
    args = parser.parse_args()
    
    generator = SummaryGenerator()
    
    try:
        if args.mode == 'daily':
            summary = generator.generate_daily_summary()
            filename = f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
        else:
            summary = generator.generate_weekly_summary()
            filename = f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Print summary
        generator.print_summary(summary)
        
        # Save if requested
        if args.save:
            generator.save_summary(summary, filename)
        
    except Exception as e:
        print(f"[ERROR] Summary generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()