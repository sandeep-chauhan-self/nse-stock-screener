#!/usr/bin/env python3
"""
Dynamic NSE Watchlist Manager - Fetches Real Market Data
Creates watchlists dynamically based on live NSE data, market cap, sector classification
"""

import json
import logging
import random
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import time

logger = logging.getLogger(__name__)

class DynamicWatchlistManager:
    """
    Production-ready dynamic watchlist management using real NSE data
    """
    
    def __init__(self, symbols_file: Optional[str] = None):
        """
        Initialize dynamic watchlist manager
        
        Args:
            symbols_file: Path to NSE symbols file (uses default if not provided)
        """
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Use provided symbols file or default NSE symbols
        if symbols_file:
            self.symbols_file = Path(symbols_file)
        else:
            self.symbols_file = self.project_root / "data" / "nse_only_symbols.txt"
        
        # Cache directory for storing fetched data
        self.cache_dir = self.project_root / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.sector_cache_file = self.cache_dir / "nse_sector_data.json"
        self.marketcap_cache_file = self.cache_dir / "nse_marketcap_data.json"
        
        self.all_symbols = []
        self.sector_data = {}
        self.marketcap_data = {}
        
        self._load_nse_symbols()
        self._load_cached_data()
    
    def _load_nse_symbols(self) -> None:
        """Load NSE symbols from file"""
        try:
            if not self.symbols_file.exists():
                logger.error(f"NSE symbols file not found: {self.symbols_file}")
                return
            
            with open(self.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            # Add .NS suffix for Yahoo Finance compatibility
            self.all_symbols = [f"{symbol}.NS" for symbol in symbols if symbol]
            
            logger.info(f"Loaded {len(self.all_symbols)} NSE symbols")
            print(f"📊 Loaded {len(self.all_symbols)} NSE symbols from {self.symbols_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading NSE symbols: {e}")
            self.all_symbols = []
    
    def _load_cached_data(self) -> None:
        """Load cached sector and market cap data"""
        try:
            # Load sector data
            if self.sector_cache_file.exists():
                with open(self.sector_cache_file, 'r') as f:
                    self.sector_data = json.load(f)
                logger.info(f"Loaded sector data for {len(self.sector_data)} symbols")
            
            # Load market cap data  
            if self.marketcap_cache_file.exists():
                with open(self.marketcap_cache_file, 'r') as f:
                    self.marketcap_data = json.load(f)
                logger.info(f"Loaded market cap data for {len(self.marketcap_data)} symbols")
                
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
    
    def _save_cached_data(self) -> None:
        """Save sector and market cap data to cache"""
        try:
            # Save sector data
            with open(self.sector_cache_file, 'w') as f:
                json.dump(self.sector_data, f, indent=2)
            
            # Save market cap data
            with open(self.marketcap_cache_file, 'w') as f:
                json.dump(self.marketcap_data, f, indent=2)
                
            logger.info("Cached data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving cached data: {e}")
    
    def fetch_stock_info_batch(self, symbols: List[str], batch_size: int = 50) -> Dict:
        """
        Fetch stock information in batches to avoid rate limiting
        
        Args:
            symbols: List of stock symbols to fetch
            batch_size: Number of symbols to fetch at once
            
        Returns:
            Dictionary with stock information
        """
        all_data = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        print(f"🔄 Fetching data for {len(symbols)} symbols in {total_batches} batches...")
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"  📥 Batch {batch_num}/{total_batches}: {len(batch)} symbols")
            
            try:
                # Create ticker string for batch download
                ticker_string = " ".join(batch)
                data = yf.download(ticker_string, period="5d", interval="1d", 
                                 group_by='ticker', progress=False, threads=True)
                
                # Process each symbol in the batch
                for symbol in batch:
                    try:
                        # Get ticker info
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        if info and 'symbol' in info:
                            all_data[symbol] = {
                                'sector': info.get('sector', 'Unknown'),
                                'industry': info.get('industry', 'Unknown'),
                                'marketCap': info.get('marketCap', 0),
                                'fullTimeEmployees': info.get('fullTimeEmployees', 0),
                                'longName': info.get('longName', symbol.replace('.NS', '')),
                                'shortName': info.get('shortName', symbol.replace('.NS', '')),
                                'lastUpdated': datetime.now().isoformat()
                            }
                        
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
                        continue
                
                # Be gentle with API
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                continue
        
        print(f"✅ Successfully fetched data for {len(all_data)} symbols")
        return all_data
    
    def update_market_data(self, sample_size: int = 200, force_refresh: bool = False) -> None:
        """
        Update market data for NSE stocks
        
        Args:
            sample_size: Number of random stocks to sample and update
            force_refresh: Whether to force refresh even if cache exists
        """
        print(f"🔄 Updating market data...")
        
        # Check if we need to update
        if not force_refresh and self.sector_data and self.marketcap_data:
            # Check cache age
            try:
                cache_age = datetime.now() - datetime.fromisoformat(
                    list(self.sector_data.values())[0].get('lastUpdated', '2000-01-01')
                )
                if cache_age.days < 7:  # Cache is less than 7 days old
                    print(f"📅 Using cached data (age: {cache_age.days} days)")
                    return
            except:
                pass  # If parsing fails, proceed with update
        
        # Sample symbols to avoid hitting rate limits
        if len(self.all_symbols) > sample_size:
            sample_symbols = random.sample(self.all_symbols, sample_size)
            print(f"📊 Sampling {sample_size} symbols from {len(self.all_symbols)} total")
        else:
            sample_symbols = self.all_symbols
        
        # Fetch data in batches
        stock_data = self.fetch_stock_info_batch(sample_symbols, batch_size=25)
        
        # Update sector and market cap data
        for symbol, data in stock_data.items():
            self.sector_data[symbol] = {
                'sector': data['sector'],
                'industry': data['industry'],
                'longName': data['longName'],
                'lastUpdated': data['lastUpdated']
            }
            
            self.marketcap_data[symbol] = {
                'marketCap': data['marketCap'],
                'employees': data['fullTimeEmployees'],
                'lastUpdated': data['lastUpdated']
            }
        
        # Save to cache
        self._save_cached_data()
        print(f"💾 Market data updated and cached")
    
    def get_sector_watchlists(self) -> Dict[str, List[str]]:
        """
        Create dynamic sector-based watchlists
        
        Returns:
            Dictionary with sector names as keys and symbol lists as values
        """
        sector_groups = {}
        
        for symbol, data in self.sector_data.items():
            sector = data.get('sector', 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)
        
        # Filter out sectors with too few stocks
        filtered_sectors = {
            sector: symbols for sector, symbols in sector_groups.items()
            if len(symbols) >= 3 and sector != 'Unknown'
        }
        
        return filtered_sectors
    
    def get_marketcap_watchlists(self) -> Dict[str, List[str]]:
        """
        Create dynamic market cap based watchlists
        
        Returns:
            Dictionary with market cap categories and symbol lists
        """
        # Sort stocks by market cap
        sorted_stocks = sorted(
            [(symbol, data.get('marketCap', 0)) for symbol, data in self.marketcap_data.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Filter out stocks with no market cap data
        valid_stocks = [(symbol, mc) for symbol, mc in sorted_stocks if mc > 0]
        
        if not valid_stocks:
            return {}
        
        # Create market cap categories
        total_stocks = len(valid_stocks)
        
        # Large cap (top 10%)
        large_cap_count = max(5, total_stocks // 10)
        large_cap = [symbol for symbol, _ in valid_stocks[:large_cap_count]]
        
        # Mid cap (next 20%)
        mid_cap_start = large_cap_count
        mid_cap_count = max(10, total_stocks // 5)
        mid_cap = [symbol for symbol, _ in valid_stocks[mid_cap_start:mid_cap_start + mid_cap_count]]
        
        # Small cap (next 30%)
        small_cap_start = mid_cap_start + mid_cap_count
        small_cap_count = max(15, (total_stocks * 3) // 10)
        small_cap = [symbol for symbol, _ in valid_stocks[small_cap_start:small_cap_start + small_cap_count]]
        
        return {
            'Large Cap': large_cap,
            'Mid Cap': mid_cap,
            'Small Cap': small_cap
        }
    
    def get_dynamic_watchlists(self) -> Dict[str, Dict]:
        """
        Generate all dynamic watchlists
        
        Returns:
            Dictionary with watchlist configurations
        """
        watchlists = {}
        
        # Sector-based watchlists
        sector_lists = self.get_sector_watchlists()
        for sector, symbols in sector_lists.items():
            # Limit to top 10 symbols per sector
            limited_symbols = symbols[:10]
            
            watchlists[f"sector_{sector.lower().replace(' ', '_').replace('&', 'and')}"] = {
                "name": f"{sector} Sector",
                "description": f"Top companies in {sector} sector",
                "symbols": limited_symbols,
                "sector": sector,
                "type": "sector",
                "count": len(limited_symbols)
            }
        
        # Market cap based watchlists
        marketcap_lists = self.get_marketcap_watchlists()
        for cap_type, symbols in marketcap_lists.items():
            # Limit to top 15 symbols per category
            limited_symbols = symbols[:15]
            
            watchlists[f"marketcap_{cap_type.lower().replace(' ', '_')}"] = {
                "name": f"{cap_type} Stocks",
                "description": f"Top {cap_type.lower()} companies by market capitalization",
                "symbols": limited_symbols,
                "sector": "Mixed",
                "type": "marketcap",
                "count": len(limited_symbols)
            }
        
        # Random sample watchlists
        if len(self.all_symbols) >= 20:
            # High volume random sample
            random_sample = random.sample(self.all_symbols, min(20, len(self.all_symbols)))
            watchlists["random_sample"] = {
                "name": "Random Market Sample",
                "description": "Random selection of NSE stocks for exploration",
                "symbols": random_sample,
                "sector": "Mixed",
                "type": "random",
                "count": len(random_sample)
            }
        
        # Top performers (if we have recent data)
        if self.marketcap_data:
            top_by_market_cap = sorted(
                [(symbol, data.get('marketCap', 0)) for symbol, data in self.marketcap_data.items()],
                key=lambda x: x[1], reverse=True
            )[:20]
            
            top_symbols = [symbol for symbol, _ in top_by_market_cap if symbol]
            
            if top_symbols:
                watchlists["nse_giants"] = {
                    "name": "NSE Giants",
                    "description": "Top 20 companies by market capitalization",
                    "symbols": top_symbols,
                    "sector": "Mixed",
                    "type": "giants",
                    "count": len(top_symbols)
                }
        
        return watchlists
    
    def list_dynamic_watchlists(self) -> List[Tuple[str, str, str, int]]:
        """
        Get list of all dynamic watchlists with basic info
        
        Returns:
            List of tuples: (id, name, description, count)
        """
        watchlists = self.get_dynamic_watchlists()
        result = []
        
        for wl_id, wl_data in watchlists.items():
            result.append((
                wl_id,
                wl_data.get('name', wl_id),
                wl_data.get('description', 'Dynamic watchlist'),
                wl_data.get('count', len(wl_data.get('symbols', [])))
            ))
        
        return result
    
    def get_watchlist_symbols(self, watchlist_id: str) -> List[str]:
        """
        Get symbols from a specific dynamic watchlist
        
        Args:
            watchlist_id: ID of the watchlist
            
        Returns:
            List of stock symbols
        """
        watchlists = self.get_dynamic_watchlists()
        watchlist = watchlists.get(watchlist_id)
        
        if watchlist:
            return watchlist.get('symbols', [])
        return []
    
    def get_watchlist(self, watchlist_id: str) -> Optional[Dict]:
        """
        Get specific watchlist by ID
        
        Args:
            watchlist_id: ID of the watchlist to retrieve
            
        Returns:
            Dictionary containing watchlist data or None if not found
        """
        watchlists = self.get_dynamic_watchlists()
        return watchlists.get(watchlist_id)
    
    def search_symbol(self, search_term: str) -> List[str]:
        """
        Search for symbols in the NSE database
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching symbols
        """
        search_term = search_term.upper().strip()
        matches = []
        
        # Direct symbol match
        for symbol in self.all_symbols:
            symbol_base = symbol.replace('.NS', '')
            if search_term in symbol_base:
                matches.append(symbol)
        
        # Search in company names if we have sector data
        for symbol, data in self.sector_data.items():
            long_name = data.get('longName', '').upper()
            if search_term in long_name and symbol not in matches:
                matches.append(symbol)
        
        return matches[:10]  # Limit to 10 matches
    
    def get_stats(self) -> Dict:
        """
        Get statistics about dynamic watchlists
        
        Returns:
            Dictionary with statistics
        """
        watchlists = self.get_dynamic_watchlists()
        sector_lists = self.get_sector_watchlists()
        marketcap_lists = self.get_marketcap_watchlists()
        
        return {
            "total_nse_symbols": len(self.all_symbols),
            "cached_sector_data": len(self.sector_data),
            "cached_marketcap_data": len(self.marketcap_data),
            "dynamic_watchlists": len(watchlists),
            "sector_watchlists": len(sector_lists),
            "marketcap_watchlists": len(marketcap_lists),
            "unique_sectors": len(set(data.get('sector') for data in self.sector_data.values())),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Usage example and testing
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Dynamic NSE Watchlist Manager Test")
    print("=" * 50)
    
    # Initialize manager
    dwm = DynamicWatchlistManager()
    
    # Check if we need to update market data
    print(f"📊 Loaded {len(dwm.all_symbols)} NSE symbols")
    
    if not dwm.sector_data or len(dwm.sector_data) < 50:
        print("🔄 Market data cache is empty or small, fetching sample data...")
        dwm.update_market_data(sample_size=100)  # Fetch data for 100 random stocks
    
    # Show dynamic watchlists
    watchlists = dwm.list_dynamic_watchlists()
    print(f"\n📋 Dynamic Watchlists Generated ({len(watchlists)}):")
    
    for wl_id, name, desc, count in watchlists:
        print(f"  • {name} ({count} stocks)")
        print(f"    {desc}")
        print()
    
    # Test search
    print("🔍 Search Test:")
    search_results = dwm.search_symbol("RELIANCE")
    print(f"  Search 'RELIANCE': {search_results}")
    
    search_results = dwm.search_symbol("TATA")
    print(f"  Search 'TATA': {search_results}")
    
    # Show statistics
    print("\n📈 Statistics:")
    stats = dwm.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")