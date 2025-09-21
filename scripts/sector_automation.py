#!/usr/bin/env python3
"""
NSE Stock Screener - Sector-Specific Automation
Specialized automation for different market sectors
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from automation_manager import AutomationManager

class SectorAutomation:
    """Sector-specific automation with specialized analysis"""
    
    SECTOR_DEFINITIONS = {
        "banking": {
            "symbols": [
                "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
                "INDUSINDBK", "FEDERALBNK", "RBLBANK", "BANDHANBNK", "IDFCFIRSTB"
            ],
            "key_indicators": ["npa_ratio", "credit_growth", "net_interest_margin", "casa_ratio"],
            "special_events": ["rbi_policy", "regulatory_changes", "credit_offtake"],
            "analysis_params": {
                "min_score": 65,
                "volume_threshold": 1.8,
                "focus_metrics": ["loan_growth", "deposit_growth", "operational_efficiency"]
            }
        },
        "technology": {
            "symbols": [
                "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MINDTREE",
                "LTTS", "COFORGE", "PERSISTENT", "OFSS", "CYIENT"
            ],
            "key_indicators": ["export_revenue", "digital_transformation", "cloud_adoption", "client_additions"],
            "special_events": ["quarterly_results", "large_deal_wins", "currency_impact"],
            "analysis_params": {
                "min_score": 60,
                "volume_threshold": 1.5,
                "focus_metrics": ["revenue_growth", "margin_expansion", "digital_revenue_mix"]
            }
        },
        "pharma": {
            "symbols": [
                "DRREDDY", "CIPLA", "BIOCON", "LUPIN", "SUNPHARMA", "AUROPHARMA",
                "TORNTPHARM", "ALKEM", "CADILAHC", "GLENMARK", "DIVISLAB"
            ],
            "key_indicators": ["fda_approvals", "drug_launches", "api_business", "us_generics"],
            "special_events": ["drug_approvals", "clinical_trials", "regulatory_issues"],
            "analysis_params": {
                "min_score": 70,
                "volume_threshold": 2.0,
                "focus_metrics": ["r_and_d_spending", "product_pipeline", "regulatory_compliance"]
            }
        },
        "energy": {
            "symbols": [
                "RELIANCE", "ONGC", "IOC", "BPCL", "HPCL", "GAIL", "OIL",
                "ADANIPORTS", "ADANIGREEN", "TATAPOWER", "NTPC", "POWERGRID"
            ],
            "key_indicators": ["crude_prices", "refining_margins", "gas_demand", "renewable_capacity"],
            "special_events": ["oil_price_movements", "government_policy", "green_energy_push"],
            "analysis_params": {
                "min_score": 58,
                "volume_threshold": 2.2,
                "focus_metrics": ["capacity_utilization", "energy_transition", "esg_compliance"]
            }
        },
        "auto": {
            "symbols": [
                "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
                "EICHERMOT", "ASHOKLEY", "BHARATFORG", "MOTHERSON", "BOSCHLTD"
            ],
            "key_indicators": ["vehicle_sales", "ev_adoption", "commodity_prices", "rural_demand"],
            "special_events": ["monthly_sales", "festive_season", "policy_changes"],
            "analysis_params": {
                "min_score": 62,
                "volume_threshold": 1.7,
                "focus_metrics": ["market_share", "ev_transition", "export_performance"]
            }
        },
        "fmcg": {
            "symbols": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
                "GODREJCP", "MARICO", "EMAMILTD", "COLPAL", "UBL"
            ],
            "key_indicators": ["rural_demand", "commodity_inflation", "distribution_reach", "brand_strength"],
            "special_events": ["quarterly_results", "rural_recovery", "input_cost_inflation"],
            "analysis_params": {
                "min_score": 68,
                "volume_threshold": 1.4,
                "focus_metrics": ["volume_growth", "margin_protection", "market_penetration"]
            }
        }
    }
    
    def __init__(self, sector: str, config_file: Optional[str] = None):
        self.sector = sector.lower()
        self.project_root = PROJECT_ROOT
        self.automation_manager = AutomationManager(config_file)
        
        if self.sector not in self.SECTOR_DEFINITIONS:
            raise ValueError(f"Unknown sector: {sector}. Available: {list(self.SECTOR_DEFINITIONS.keys())}")
        
        self.sector_config = self.SECTOR_DEFINITIONS[self.sector]
        self.results = {}
    
    def run_sector_analysis(self, **kwargs) -> Dict:
        """Run comprehensive sector-specific analysis"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🏭 Starting {self.sector.upper()} sector analysis - Session: {session_id}")
        
        analysis_result = {
            "session_id": session_id,
            "sector": self.sector,
            "start_time": datetime.now().isoformat(),
            "analysis_type": kwargs.get("analysis_type", "comprehensive"),
            "sector_overview": {},
            "stock_analysis": {},
            "sector_insights": {},
            "investment_recommendations": {},
            "risk_assessment": {}
        }
        
        try:
            # 1. Sector Overview Analysis
            analysis_result["sector_overview"] = self.analyze_sector_overview()
            
            # 2. Individual Stock Analysis
            analysis_result["stock_analysis"] = self.analyze_sector_stocks(**kwargs)
            
            # 3. Sector-Specific Insights
            analysis_result["sector_insights"] = self.generate_sector_insights()
            
            # 4. Investment Recommendations
            analysis_result["investment_recommendations"] = self.generate_investment_recommendations()
            
            # 5. Risk Assessment
            analysis_result["risk_assessment"] = self.assess_sector_risks()
            
            # Save results
            self.save_sector_analysis(analysis_result)
            
            print(f"[OK] {self.sector.upper()} sector analysis completed")
            
        except Exception as e:
            analysis_result["error"] = str(e)
            print(f"[ERROR] {self.sector.upper()} sector analysis failed: {e}")
        
        return analysis_result
    
    def analyze_sector_overview(self) -> Dict:
        """Analyze overall sector health and trends"""
        # Simulated sector overview analysis
        return {
            "sector_health": "POSITIVE",
            "trend_direction": "BULLISH",
            "market_cap_distribution": {
                "large_cap": 70,
                "mid_cap": 25,
                "small_cap": 5
            },
            "performance_metrics": {
                "ytd_performance": 15.2,
                "52_week_high_percentage": 68,
                "average_pe_ratio": 22.5,
                "average_pb_ratio": 3.2
            },
            "key_drivers": self.get_sector_drivers(),
            "headwinds": self.get_sector_headwinds()
        }
    
    def get_sector_drivers(self) -> List[str]:
        """Get sector-specific positive drivers"""
        drivers_map = {
            "banking": [
                "Economic recovery driving credit demand",
                "Declining NPAs and improved asset quality",
                "Rising interest rates benefiting NIMs",
                "Digital transformation reducing costs"
            ],
            "technology": [
                "Strong global IT spending growth",
                "Digital transformation acceleration",
                "Cloud adoption increasing",
                "Large deal pipeline robust"
            ],
            "pharma": [
                "Strong US generics performance",
                "Increasing R&D investments",
                "API business expansion",
                "Government healthcare initiatives"
            ],
            "energy": [
                "Rising oil prices benefiting upstream",
                "Government push for renewable energy",
                "Improving refining margins",
                "Infrastructure development"
            ],
            "auto": [
                "Rural demand recovery",
                "Festive season boost",
                "EV adoption increasing",
                "Export opportunities expanding"
            ],
            "fmcg": [
                "Rural demand stabilizing",
                "Premiumization trends",
                "Distribution network expansion",
                "Brand strength and loyalty"
            ]
        }
        
        return drivers_map.get(self.sector, ["General economic growth", "Market expansion"])
    
    def get_sector_headwinds(self) -> List[str]:
        """Get sector-specific challenges"""
        headwinds_map = {
            "banking": [
                "Rising interest rates affecting borrowers",
                "Potential asset quality concerns",
                "Regulatory compliance costs",
                "Competition from NBFCs and fintechs"
            ],
            "technology": [
                "Client budget optimization",
                "Currency headwinds",
                "Wage inflation pressures",
                "Visa restrictions impact"
            ],
            "pharma": [
                "Pricing pressures in US market",
                "Regulatory compliance costs",
                "Patent cliffs for innovator drugs",
                "Raw material cost inflation"
            ],
            "energy": [
                "Volatile crude oil prices",
                "Environmental regulations",
                "Transition to renewable energy",
                "Geopolitical risks"
            ],
            "auto": [
                "Semiconductor supply constraints",
                "Commodity price inflation",
                "EV transition costs",
                "Regulatory emission norms"
            ],
            "fmcg": [
                "Input cost inflation",
                "Rural demand weakness",
                "Competitive pressures",
                "Changing consumer preferences"
            ]
        }
        
        return headwinds_map.get(self.sector, ["Market competition", "Economic headwinds"])
    
    def analyze_sector_stocks(self, **kwargs) -> Dict:
        """Analyze individual stocks in the sector"""
        stocks = self.sector_config["symbols"]
        analysis_params = self.sector_config["analysis_params"].copy()
        
        # Override with any provided parameters
        analysis_params.update(kwargs)
        
        stock_results = {}
        
        for stock in stocks:
            print(f"  [ANALYSIS] Analyzing {stock}...")
            
            # Simulated individual stock analysis
            stock_results[stock] = {
                "symbol": stock,
                "composite_score": self.calculate_simulated_score(stock),
                "sector_rank": len([s for s in stocks if s < stock]) + 1,
                "key_metrics": self.get_stock_key_metrics(stock),
                "sector_specific_analysis": self.get_sector_specific_metrics(stock),
                "recommendation": self.get_stock_recommendation(stock),
                "price_targets": self.get_price_targets(stock)
            }
        
        return stock_results
    
    def calculate_simulated_score(self, stock: str) -> float:
        """Calculate simulated composite score for stock"""
        # Hash-based consistent simulation
        hash_val = hash(stock + self.sector) % 100
        base_score = 45 + (hash_val % 40)  # Score between 45-85
        
        # Sector-specific adjustments
        sector_adjustments = {
            "banking": 5,
            "technology": 3,
            "pharma": -2,
            "energy": 0,
            "auto": 2,
            "fmcg": 4
        }
        
        adjusted_score = base_score + sector_adjustments.get(self.sector, 0)
        return max(30, min(90, adjusted_score))  # Clamp between 30-90
    
    def get_stock_key_metrics(self, stock: str) -> Dict:
        """Get key financial metrics for stock"""
        # Simulated metrics
        return {
            "pe_ratio": 18.5 + (hash(stock) % 20),
            "pb_ratio": 2.1 + (hash(stock) % 5),
            "roe": 12.5 + (hash(stock) % 15),
            "debt_to_equity": 0.3 + (hash(stock) % 10) / 10,
            "revenue_growth": 8.5 + (hash(stock) % 25),
            "profit_margin": 12.0 + (hash(stock) % 20)
        }
    
    def get_sector_specific_metrics(self, stock: str) -> Dict:
        """Get sector-specific metrics for stock"""
        sector_metrics = {
            "banking": {
                "npa_ratio": 1.5 + (hash(stock) % 3),
                "casa_ratio": 35 + (hash(stock) % 20),
                "credit_growth": 10 + (hash(stock) % 15),
                "net_interest_margin": 3.2 + (hash(stock) % 2)
            },
            "technology": {
                "revenue_per_employee": 250000 + (hash(stock) % 100000),
                "digital_revenue_mix": 60 + (hash(stock) % 30),
                "client_concentration": 15 + (hash(stock) % 10),
                "employee_utilization": 75 + (hash(stock) % 20)
            },
            "pharma": {
                "us_revenue_mix": 40 + (hash(stock) % 30),
                "r_and_d_intensity": 8 + (hash(stock) % 7),
                "product_pipeline": 15 + (hash(stock) % 20),
                "regulatory_compliance_score": 85 + (hash(stock) % 10)
            },
            "energy": {
                "capacity_utilization": 70 + (hash(stock) % 25),
                "renewable_mix": 20 + (hash(stock) % 40),
                "refining_margin": 8 + (hash(stock) % 10),
                "carbon_intensity": 50 + (hash(stock) % 30)
            },
            "auto": {
                "market_share": 8 + (hash(stock) % 15),
                "ev_readiness_score": 60 + (hash(stock) % 30),
                "export_revenue_mix": 25 + (hash(stock) % 30),
                "plant_utilization": 75 + (hash(stock) % 20)
            },
            "fmcg": {
                "rural_revenue_mix": 35 + (hash(stock) % 25),
                "brand_strength_score": 75 + (hash(stock) % 20),
                "distribution_reach": 80 + (hash(stock) % 15),
                "volume_growth": 5 + (hash(stock) % 10)
            }
        }
        
        return sector_metrics.get(self.sector, {})
    
    def get_stock_recommendation(self, stock: str) -> str:
        """Get investment recommendation for stock"""
        score = self.calculate_simulated_score(stock)
        
        if score >= 75:
            return "BUY"
        elif score >= 65:
            return "ACCUMULATE"
        elif score >= 55:
            return "HOLD"
        elif score >= 45:
            return "REDUCE"
        else:
            return "SELL"
    
    def get_price_targets(self, stock: str) -> Dict:
        """Get price targets for stock"""
        # Simulated price targets
        current_price = 1000 + (hash(stock) % 2000)  # Simulated current price
        
        return {
            "current_price": current_price,
            "target_1m": current_price * (1 + 0.05),
            "target_3m": current_price * (1 + 0.12),
            "target_6m": current_price * (1 + 0.20),
            "target_12m": current_price * (1 + 0.35),
            "stop_loss": current_price * (1 - 0.08)
        }
    
    def generate_sector_insights(self) -> Dict:
        """Generate sector-specific insights"""
        return {
            "top_themes": self.get_sector_investment_themes(),
            "valuation_commentary": self.get_valuation_insights(),
            "technical_outlook": self.get_technical_sector_view(),
            "event_calendar": self.get_upcoming_events(),
            "peer_comparison": self.generate_peer_comparison()
        }
    
    def get_sector_investment_themes(self) -> List[str]:
        """Get key investment themes for the sector"""
        themes_map = {
            "banking": [
                "Credit cycle upturn benefiting banks",
                "Digital banking reducing operational costs",
                "Asset quality normalization post-COVID",
                "Rising interest rate environment"
            ],
            "technology": [
                "Digital transformation driving demand",
                "Cloud migration acceleration",
                "Automation and AI adoption",
                "Talent retention and skill upgradation"
            ],
            "pharma": [
                "Biosimilars opportunity in developed markets",
                "API supply chain diversification",
                "Digital health and telemedicine growth",
                "Vaccine manufacturing capabilities"
            ],
            "energy": [
                "Energy transition and renewable focus",
                "Petrochemical integration benefits",
                "Carbon capture and storage technologies",
                "Energy security and domestic production"
            ],
            "auto": [
                "Electric vehicle ecosystem development",
                "Shared mobility and connectivity",
                "Semiconductor and battery technology",
                "Export market diversification"
            ],
            "fmcg": [
                "Premiumization and brand portfolio",
                "Direct-to-consumer channel growth",
                "Health and wellness product focus",
                "Sustainable packaging initiatives"
            ]
        }
        
        return themes_map.get(self.sector, ["Growth opportunities", "Market expansion"])
    
    def get_valuation_insights(self) -> str:
        """Get sector valuation commentary"""
        valuation_map = {
            "banking": "Trading at attractive valuations with improving asset quality metrics",
            "technology": "Premium valuations justified by strong growth prospects and digital demand",
            "pharma": "Mixed valuations with US-focused companies trading at premium",
            "energy": "Cyclical valuations with focus on integrated players and renewable transition",
            "auto": "Reasonable valuations with EV transition creating differentiation",
            "fmcg": "Premium valuations supported by consistent performance and brand strength"
        }
        
        return valuation_map.get(self.sector, "Sector trading at fair valuations")
    
    def get_technical_sector_view(self) -> str:
        """Get technical analysis view for sector"""
        return "Sector showing bullish momentum with strong volume participation"
    
    def get_upcoming_events(self) -> List[Dict]:
        """Get upcoming sector-relevant events"""
        # Simulated upcoming events
        return [
            {
                "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                "event": f"{self.sector.title()} sector quarterly results",
                "impact": "HIGH"
            },
            {
                "date": (datetime.now() + timedelta(days=12)).strftime("%Y-%m-%d"),
                "event": "Regulatory policy announcement",
                "impact": "MEDIUM"
            }
        ]
    
    def generate_peer_comparison(self) -> Dict:
        """Generate peer comparison within sector"""
        stocks = self.sector_config["symbols"][:5]  # Top 5 for comparison
        
        comparison = {}
        for stock in stocks:
            comparison[stock] = {
                "score": self.calculate_simulated_score(stock),
                "rank": 0,  # Will be calculated
                "key_strength": f"{stock} leadership position",
                "key_concern": f"{stock} competitive pressure"
            }
        
        # Calculate ranks
        sorted_stocks = sorted(comparison.items(), key=lambda x: x[1]["score"], reverse=True)
        for rank, (stock, data) in enumerate(sorted_stocks, 1):
            comparison[stock]["rank"] = rank
        
        return comparison
    
    def generate_investment_recommendations(self) -> Dict:
        """Generate sector investment recommendations"""
        return {
            "top_picks": self.get_top_sector_picks(),
            "allocation_strategy": self.get_allocation_strategy(),
            "risk_considerations": self.get_risk_considerations(),
            "time_horizon": self.get_investment_time_horizon()
        }
    
    def get_top_sector_picks(self) -> List[Dict]:
        """Get top stock picks in sector"""
        stocks = self.sector_config["symbols"]
        scored_stocks = [(stock, self.calculate_simulated_score(stock)) for stock in stocks]
        top_stocks = sorted(scored_stocks, key=lambda x: x[1], reverse=True)[:3]
        
        return [
            {
                "symbol": stock,
                "score": score,
                "rationale": f"{stock} strong fundamentals and sector leadership"
            }
            for stock, score in top_stocks
        ]
    
    def get_allocation_strategy(self) -> str:
        """Get sector allocation strategy"""
        strategies = {
            "banking": "Overweight large private banks, selective PSU exposure",
            "technology": "Focus on large-cap with strong digital capabilities",
            "pharma": "Diversified approach across US generics and domestic formulations",
            "energy": "Balanced exposure to traditional and renewable energy",
            "auto": "Prefer companies with strong EV transition strategy",
            "fmcg": "Quality companies with rural recovery exposure"
        }
        
        return strategies.get(self.sector, "Balanced approach across sector leaders")
    
    def get_risk_considerations(self) -> List[str]:
        """Get key risk considerations"""
        return [
            "Regulatory policy changes",
            "Economic cycle sensitivity",
            "Competitive intensity",
            "Global market dynamics"
        ]
    
    def get_investment_time_horizon(self) -> str:
        """Get recommended investment time horizon"""
        return "Medium to long-term (1-3 years) for structural growth themes"
    
    def assess_sector_risks(self) -> Dict:
        """Assess sector-specific risks"""
        return {
            "regulatory_risk": "MEDIUM",
            "cyclical_risk": "MEDIUM",
            "competitive_risk": "HIGH",
            "technology_risk": "LOW",
            "execution_risk": "MEDIUM",
            "overall_risk_rating": "MEDIUM",
            "risk_mitigation": [
                "Diversification across sector leaders",
                "Focus on companies with strong moats",
                "Regular portfolio rebalancing"
            ]
        }
    
    def save_sector_analysis(self, results: Dict):
        """Save sector analysis results"""
        output_dir = self.project_root / 'output' / 'sector_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / f"{self.sector}_analysis_{results['session_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        self.generate_sector_report(results, output_dir)
        
        print(f"[FOLDER] {self.sector.title()} analysis saved: {results_file}")
    
    def generate_sector_report(self, results: Dict, output_dir: Path):
        """Generate human-readable sector report"""
        report_lines = [
            f"🏭 {self.sector.upper()} SECTOR ANALYSIS REPORT",
            "=" * 60,
            f"Session ID: {results['session_id']}",
            f"Analysis Date: {results['start_time'][:10]}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[ANALYSIS] SECTOR OVERVIEW",
            "-" * 30,
            f"Sector Health: {results['sector_overview']['sector_health']}",
            f"Trend Direction: {results['sector_overview']['trend_direction']}",
            f"YTD Performance: {results['sector_overview']['performance_metrics']['ytd_performance']}%",
            f"Average P/E Ratio: {results['sector_overview']['performance_metrics']['average_pe_ratio']}",
            "",
            "[TARGET] TOP STOCK PICKS",
            "-" * 30
        ]
        
        for pick in results['investment_recommendations']['top_picks']:
            report_lines.append(f"• {pick['symbol']}: Score {pick['score']:.1f} - {pick['rationale']}")
        
        report_lines.extend([
            "",
            "[SEARCH] KEY INVESTMENT THEMES",
            "-" * 30
        ])
        
        for theme in results['sector_insights']['top_themes']:
            report_lines.append(f"• {theme}")
        
        report_lines.extend([
            "",
            "[WARNING] RISK ASSESSMENT",
            "-" * 30,
            f"Overall Risk Rating: {results['risk_assessment']['overall_risk_rating']}",
            f"Key Risks: Regulatory, Cyclical, Competitive",
            "",
            "[INFO] INVESTMENT STRATEGY",
            "-" * 30,
            f"Allocation: {results['investment_recommendations']['allocation_strategy']}",
            f"Time Horizon: {results['investment_recommendations']['time_horizon']}",
            "",
            "[CHART] VALUATION COMMENTARY",
            "-" * 30,
            results['sector_insights']['valuation_commentary']
        ])
        
        # Save report
        report_file = output_dir / f"{self.sector}_sector_report_{results['session_id']}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[LIST] {self.sector.title()} sector report saved: {report_file}")

def main():
    """Main sector automation entry point"""
    parser = argparse.ArgumentParser(description='Sector-Specific Automation System')
    
    parser.add_argument('--sector', required=True,
                        choices=list(SectorAutomation.SECTOR_DEFINITIONS.keys()),
                        help='Sector to analyze')
    parser.add_argument('--analysis-type', default='comprehensive',
                        choices=['comprehensive', 'quick', 'deep'],
                        help='Type of analysis to perform')
    parser.add_argument('--min-score', type=float,
                        help='Minimum composite score filter')
    parser.add_argument('--focus', nargs='+',
                        help='Specific focus areas for analysis')
    
    args = parser.parse_args()
    
    try:
        # Create sector automation
        sector_automation = SectorAutomation(args.sector)
        
        # Prepare analysis parameters
        analysis_params = {
            "analysis_type": args.analysis_type
        }
        
        if args.min_score:
            analysis_params["min_score"] = args.min_score
        
        if args.focus:
            analysis_params["focus_areas"] = args.focus
        
        # Run sector analysis
        results = sector_automation.run_sector_analysis(**analysis_params)
        
        if "error" not in results:
            print(f"\n[SUCCESS] {args.sector.title()} sector analysis completed successfully!")
            
            # Show key highlights
            top_picks = results['investment_recommendations']['top_picks']
            print(f"[TARGET] Top picks: {', '.join([pick['symbol'] for pick in top_picks])}")
            
            sector_health = results['sector_overview']['sector_health']
            print(f"[ANALYSIS] Sector health: {sector_health}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ {args.sector.title()} sector analysis interrupted by user")
    except Exception as e:
        print(f"[ERROR] {args.sector.title()} sector analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()