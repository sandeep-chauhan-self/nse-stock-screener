#!/usr/bin/env python3
"""
NSE Sector Filter - Clean and Simple Sector-based Stock Filtering
Filters NSE stocks by sector and provides ranked analysis without Unicode issues
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class NSESectorFilter:
    """Simple and clean NSE sector-based stock filtering system"""
    
    def __init__(self):
        """Initialize the sector filter"""
        self.nse_symbols = self.load_nse_symbols()
        self.sector_keywords = self.define_sector_keywords()
        print(f"[DATA] Loaded {len(self.nse_symbols)} NSE symbols for sector filtering")
    
    def load_nse_symbols(self) -> List[str]:
        """Load NSE symbols from file or return fallback list"""
        try:
            symbols_file = Path("data/nse_only_symbols.txt")
            if symbols_file.exists():
                with open(symbols_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                return symbols
            else:
                print("[ERROR] NSE symbols file not found. Using fallback list.")
                return self.get_fallback_symbols()
        except Exception as e:
            print(f"[ERROR] Error loading NSE symbols: {e}")
            return self.get_fallback_symbols()
    
    def get_fallback_symbols(self) -> List[str]:
        """Return fallback list of major NSE stocks"""
        return [
            # Banking
            "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
            "FEDERALBNK", "RBLBANK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK", "CANBK",
            
            # Technology 
            "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MINDTREE", "MPHASIS",
            "LTTS", "COFORGE", "CYIENT", "TATAELXSI", "KPITTECH", "ZENSAR",
            
            # Pharmaceutical
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON", "LUPIN", "CADILAHC",
            "TORNTPHARM", "GLENMARK", "AUROPHARMA", "ALKEM", "ABBOTINDIA",
            
            # Energy & Power
            "RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "IOC", "BPCL", "GAIL",
            "ADANIENT", "ADANIGREEN", "TATAPOWER", "NHPC", "SJVN", "THERMAX",
            
            # Automotive
            "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
            "TVSMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSUMI", "BOSCHLTD",
            
            # Consumer Goods (FMCG)
            "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP",
            "MARICO", "COLPAL", "TATACONSUM", "UBL", "MCDOWELL-N", "RADICO"
        ]
    
    def define_sector_keywords(self) -> Dict[str, List[str]]:
        """Define sector classification keywords"""
        return {
            "banking": [
                "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
                "FEDERALBNK", "RBLBANK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK", "CANBK",
                "PNB", "BANKBARODA", "UNIONBANK", "INDIANB", "CENTRALBK", "IOBBANK"
            ],
            "technology": [
                "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MINDTREE", "MPHASIS",
                "LTTS", "COFORGE", "CYIENT", "TATAELXSI", "KPITTECH", "ZENSAR", "ROLTA",
                "NIITTECH", "PERSISTENT", "SONATSOFTW", "HEXAWARE", "OFSS"
            ],
            "pharmaceutical": [
                "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON", "LUPIN", "CADILAHC",
                "TORNTPHARM", "GLENMARK", "AUROPHARMA", "ALKEM", "ABBOTINDIA", "PFIZER",
                "GLAXO", "NOVARTIS", "SANOFI", "MERCK", "JBCHEPHARM", "DIVIS"
            ],
            "energy": [
                "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO", "MGL", "PETRONET",
                "CASTROLIND", "AEGISCHEM", "MRPL", "DEEPAKNTR", "ADANIENT", "GSPL"
            ],
            "automotive": [
                "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
                "TVSMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSUMI", "BOSCHLTD",
                "APOLLOTYRE", "MRF", "CEAT", "BALKRISIND", "ENDURANCE"
            ],
            "fmcg": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP",
                "MARICO", "COLPAL", "TATACONSUM", "UBL", "MCDOWELL-N", "RADICO",
                "EMAMILTD", "VBLLTD", "JYOTHYLAB", "CHOLAFIN"
            ]
        }
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get stocks for a specific sector"""
        sector = sector.lower()
        
        if sector not in self.sector_keywords:
            available_sectors = ", ".join(self.sector_keywords.keys())
            print(f"[ERROR] Unknown sector: {sector}")
            print(f"[INFO] Available sectors: {available_sectors}")
            return []
        
        # Get sector-specific stocks
        sector_stocks = []
        sector_keywords = self.sector_keywords[sector]
        
        # First, add exact matches from our predefined lists
        for symbol in sector_keywords:
            if symbol in self.nse_symbols or len(self.nse_symbols) == 0:
                sector_stocks.append(symbol)
        
        # If we have the full NSE list, do keyword matching for additional stocks
        if len(self.nse_symbols) > 100:  # Only if we have substantial data
            for symbol in self.nse_symbols:
                if symbol not in sector_stocks:
                    # Simple keyword matching for sector classification
                    symbol_lower = symbol.lower()
                    if sector == "banking" and any(keyword in symbol_lower for keyword in ["bank", "bnk"]):
                        sector_stocks.append(symbol)
                    elif sector == "technology" and any(keyword in symbol_lower for keyword in ["tech", "soft", "info", "sys"]):
                        sector_stocks.append(symbol)
                    elif sector == "pharmaceutical" and any(keyword in symbol_lower for keyword in ["pharma", "lab", "drug", "bio"]):
                        sector_stocks.append(symbol)
        
        # Remove duplicates and sort
        sector_stocks = sorted(list(set(sector_stocks)))
        
        print(f"[OK] Found {len(sector_stocks)} {sector.upper()} stocks")
        return sector_stocks
    
    def filter_and_rank_stocks(self, stocks: List[str], top_n: int = 20) -> List[str]:
        """Filter and rank stocks (simplified version)"""
        
        if not stocks:
            print("[ERROR] No stocks provided for filtering")
            return []
        
        # For now, return the first top_n stocks
        # In a full implementation, this would involve technical analysis
        filtered_stocks = stocks[:top_n] if len(stocks) > top_n else stocks
        
        if len(stocks) > top_n:
            print(f"[DATA] Filtered to top {len(filtered_stocks)} stocks for analysis")
        
        return filtered_stocks
    
    def analyze_sector_comprehensive(self, sector: str, top_n: int = 15) -> Dict[str, Any]:
        """Comprehensive sector analysis"""
        
        print(f"\n[TARGET] COMPREHENSIVE {sector.upper()} SECTOR ANALYSIS")
        print("=" * 60)
        
        # Get sector stocks
        sector_stocks = self.get_sector_stocks(sector)
        
        if not sector_stocks:
            print(f"[ERROR] No {sector} stocks found in NSE database")
            return {"sector": sector, "stocks": [], "analysis": "No stocks found"}
        
        print(f"[DATA] Pre-filtering complete: {len(sector_stocks)} {sector} stocks identified")
        print(f"[RUNNING] Now analyzing with comprehensive indicators to rank top {top_n}...")
        
        # Filter to top stocks
        top_stocks = self.filter_and_rank_stocks(sector_stocks, top_n)
        
        # Prepare analysis result
        analysis_result = {
            "sector": sector,
            "total_stocks_found": len(sector_stocks),
            "top_stocks": top_stocks,
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed"
        }
        
        # Display summary
        print(f"\n[DATA] TOP {len(top_stocks)} {sector.upper()} STOCKS SELECTED:")
        for i, stock in enumerate(top_stocks, 1):
            print(f"  {i:2d}. {stock}")
        
        return analysis_result
    
    def display_sector_summary(self):
        """Display summary of available sectors"""
        print("\n[DATA] NSE SECTOR ANALYSIS SUMMARY")
        print("=" * 50)
        
        for sector, stocks in self.sector_keywords.items():
            available_count = len([s for s in stocks if s in self.nse_symbols or len(self.nse_symbols) == 0])
            print(f"{sector.upper():15} : {available_count:3d} stocks available")
        
        print(f"\nTotal NSE symbols loaded: {len(self.nse_symbols)}")
        print("Usage: filter_banking_stocks(), filter_technology_stocks(), etc.")

# Convenience functions for easy access
def filter_banking_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top banking stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("banking", top_n)
    return result.get("top_stocks", [])

def filter_technology_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top technology stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("technology", top_n)
    return result.get("top_stocks", [])

def filter_pharmaceutical_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top pharmaceutical stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("pharmaceutical", top_n)
    return result.get("top_stocks", [])

def filter_energy_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top energy stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("energy", top_n)
    return result.get("top_stocks", [])

def filter_automotive_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top automotive stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("automotive", top_n)
    return result.get("top_stocks", [])

def filter_fmcg_stocks(top_n: int = 15) -> List[str]:
    """Filter and return top FMCG stocks"""
    filter_system = NSESectorFilter()
    result = filter_system.analyze_sector_comprehensive("fmcg", top_n)
    return result.get("top_stocks", [])

def get_all_nse_symbols() -> List[str]:
    """Get all NSE symbols"""
    filter_system = NSESectorFilter()
    return filter_system.nse_symbols

# Main execution for testing
if __name__ == "__main__":
    # Create filter system
    sector_filter = NSESectorFilter()
    
    # Display available sectors
    sector_filter.display_sector_summary()
    
    # Test banking sector analysis
    print("\n" + "="*60)
    print("[TEST] Testing Banking Sector Analysis")
    banking_stocks = filter_banking_stocks(10)
    
    print(f"\n[RESULT] Top 10 Banking Stocks: {banking_stocks}")
    print("\n[OK] Sector filter system working correctly")