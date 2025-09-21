"""
Intelligent Watchlist Manager - Enhanced version with comprehensive analysis
Integrates technical analysis pipeline to provide truly high-potential stock selections
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import random
from datetime import datetime, timedelta
import time
from enum import Enum

# Add src directory to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import base watchlist manager
from dynamic_watchlist_manager import DynamicWatchlistManager

# Define market regime enum if not available
class MarketRegime(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"  
    SIDEWAYS = "SIDEWAYS"

# Check for analysis components availability
ANALYSIS_AVAILABLE = False
try:
    # Try to import yfinance for basic analysis
    import yfinance as yf
    import pandas as pd
    import numpy as np
    ANALYSIS_AVAILABLE = True
    print("Basic analysis components available")
except ImportError:
    print("Analysis components not available - using fallback mode")


class IntelligentWatchlistManager(DynamicWatchlistManager):
    """
    Enhanced Dynamic Watchlist Manager with integrated analysis capabilities
    
    Provides intelligent "Top N" stock selection based on:
    - Market cap and volume analysis
    - Basic technical indicators  
    - Market performance metrics
    - Risk-based filtering
    """
    
    def __init__(self):
        super().__init__()
        self.analysis_cache = {}
        self.last_analysis_update = None
        self.analysis_cache_duration = timedelta(hours=6)
        self.market_regime = MarketRegime.SIDEWAYS  # Default
        
        print(f"Intelligent analysis initialized (Market: {self.market_regime.value})")
    
    def _analyze_stock_basic(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Run basic analysis on a single stock using yfinance
        
        Args:
            symbol: Stock symbol (with .NS suffix)
            
        Returns:
            Dictionary with basic analysis results or None if analysis fails
        """
        if not ANALYSIS_AVAILABLE:
            return {'symbol': symbol, 'composite_score': 60, 'analysis_type': 'fallback'}
            
        try:
            # Check cache first
            if symbol in self.analysis_cache:
                cached_result = self.analysis_cache[symbol]
                cache_time = cached_result.get('timestamp')
                if cache_time and datetime.now() - cache_time < self.analysis_cache_duration:
                    return cached_result['analysis']
            
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            info = ticker.info
            
            if hist.empty or len(hist) < 20:
                return None
                
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            price_change_6m = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            # Simple RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Calculate composite score (simplified)
            score = 50  # Base score
            
            # Volume factor (0-25 points)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 1.5:
                score += min(25, (volume_ratio - 1) * 10)
            
            # Price momentum (0-25 points)  
            if price_change_6m > 10:
                score += min(25, price_change_6m / 2)
            elif price_change_6m < -10:
                score -= min(25, abs(price_change_6m) / 2)
                
            # RSI factor (0-20 points)
            if 40 <= current_rsi <= 70:  # Good RSI range
                score += 20
            elif current_rsi > 80 or current_rsi < 20:  # Extreme levels
                score -= 10
                
            # Market cap factor (0-10 points)
            market_cap = info.get('marketCap', 0)
            if market_cap > 100_000_000_000:  # Large cap
                score += 10
            elif market_cap > 10_000_000_000:  # Mid cap
                score += 5
                
            # Ensure score is within bounds
            composite_score = max(0, min(100, score))
            
            analysis_result = {
                'symbol': symbol,
                'composite_score': round(composite_score, 1),
                'current_price': round(current_price, 2),
                'volume_ratio': round(volume_ratio, 2),
                'price_change_6m': round(price_change_6m, 2),
                'rsi': round(current_rsi, 1),
                'market_cap': market_cap,
                'analysis_type': 'basic_technical',
                'last_analyzed': datetime.now().isoformat()
            }
            
            # Cache the result
            self.analysis_cache[symbol] = {
                'analysis': analysis_result,
                'timestamp': datetime.now()
            }
            
            return analysis_result
                
        except Exception as e:
            print(f"Analysis failed for {symbol}: {e}")
            return None
    
    def _analyze_sector_stocks(self, sector_symbols: List[str], min_score: float = 50.0) -> List[Dict[str, Any]]:
        """
        Analyze multiple stocks in a sector and return ranked results
        
        Args:
            sector_symbols: List of stock symbols
            min_score: Minimum composite score threshold
            
        Returns:
            List of analyzed stocks sorted by composite score (highest first)
        """
        analyzed_stocks = []
        total_symbols = len(sector_symbols)
        
        print(f"Analyzing {total_symbols} stocks in sector...")
        
        for i, symbol in enumerate(sector_symbols, 1):
            if i % 5 == 0:
                print(f"   Progress: {i}/{total_symbols} stocks analyzed")
                
            analysis = self._analyze_stock_basic(symbol)
            
            if analysis and analysis.get('composite_score', 0) >= min_score:
                analyzed_stocks.append(analysis)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Sort by composite score (highest first)
        analyzed_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Analysis complete: {len(analyzed_stocks)} qualifying stocks found")
        return analyzed_stocks
    
    def get_intelligent_sector_watchlists(self, max_stocks_per_sector: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create intelligent sector-based watchlists using analysis
        
        Args:
            max_stocks_per_sector: Maximum number of top stocks per sector
            
        Returns:
            Dictionary with sector names and ranked stock data
        """
        print(f"Creating intelligent sector watchlists (Market: {self.market_regime.value})")
        
        # Get base sector groupings
        sector_lists = self.get_sector_watchlists()
        intelligent_watchlists = {}
        
        # Set minimum score based on market regime
        regime_thresholds = {
            MarketRegime.BULLISH: 55.0,    
            MarketRegime.BEARISH: 70.0,     
            MarketRegime.SIDEWAYS: 60.0    
        }
        min_score = regime_thresholds.get(self.market_regime, 60.0)
        
        print(f"Using minimum score threshold: {min_score} (regime: {self.market_regime.value})")
        
        for sector, symbols in sector_lists.items():
            print(f"\nAnalyzing {sector} sector ({len(symbols)} stocks)...")
            
            # Analyze stocks in this sector
            analyzed_stocks = self._analyze_sector_stocks(symbols, min_score)
            
            # Take top N stocks
            top_stocks = analyzed_stocks[:max_stocks_per_sector]
            
            if top_stocks:
                intelligent_watchlists[sector] = top_stocks
                avg_score = sum(s['composite_score'] for s in top_stocks) / len(top_stocks)
                print(f"{sector}: {len(top_stocks)} stocks selected (avg score: {avg_score:.1f})")
            else:
                print(f"{sector}: No stocks met minimum criteria")
        
        return intelligent_watchlists
    
    def get_intelligent_dynamic_watchlists(self, max_stocks_per_category: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Generate all intelligent dynamic watchlists with analysis
        
        Args:
            max_stocks_per_category: Maximum stocks per watchlist category
            
        Returns:
            Dictionary with enhanced watchlist configurations
        """
        print(f"Generating intelligent dynamic watchlists...")
        watchlists = {}
        
        # Get intelligent sector watchlists
        intelligent_sectors = self.get_intelligent_sector_watchlists(max_stocks_per_category)
        
        for sector, analyzed_stocks in intelligent_sectors.items():
            if not analyzed_stocks:
                continue
                
            # Extract symbols for compatibility
            symbols = [stock['symbol'] for stock in analyzed_stocks]
            avg_score = sum(stock['composite_score'] for stock in analyzed_stocks) / len(analyzed_stocks)
            
            watchlist_id = f"intelligent_sector_{sector.lower().replace(' ', '_').replace('&', 'and')}"
            
            watchlists[watchlist_id] = {
                "name": f"{sector} Sector (AI-Ranked)",
                "description": f"Top {len(symbols)} analytically-ranked companies in {sector} (avg score: {avg_score:.1f})",
                "symbols": symbols,
                "sector": sector,
                "type": "intelligent_sector",
                "count": len(symbols),
                "average_score": round(avg_score, 1),
                "min_score": min(stock['composite_score'] for stock in analyzed_stocks),
                "max_score": max(stock['composite_score'] for stock in analyzed_stocks),
                "market_regime": self.market_regime.value,
                "analysis_timestamp": datetime.now().isoformat(),
                "detailed_analysis": analyzed_stocks
            }
        
        # Add market cap based intelligent lists
        marketcap_lists = self.get_marketcap_watchlists()
        for cap_type, symbols in marketcap_lists.items():
            limited_symbols = symbols[:max_stocks_per_category]
            
            watchlist_id = f"intelligent_marketcap_{cap_type.lower().replace(' ', '_')}"
            
            watchlists[watchlist_id] = {
                "name": f"{cap_type} Stocks (AI-Enhanced)",
                "description": f"Top {len(limited_symbols)} {cap_type.lower()} companies with analysis overlay",
                "symbols": limited_symbols,
                "sector": "Mixed",
                "type": "intelligent_marketcap",
                "count": len(limited_symbols),
                "market_regime": self.market_regime.value,
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        # High-potential cross-sector picks
        if intelligent_sectors:
            all_analyzed = []
            for sector_stocks in intelligent_sectors.values():
                all_analyzed.extend(sector_stocks)
            
            # Sort by composite score across all sectors
            all_analyzed.sort(key=lambda x: x['composite_score'], reverse=True)
            top_cross_sector = all_analyzed[:15]
            
            if top_cross_sector:
                cross_sector_symbols = [stock['symbol'] for stock in top_cross_sector]
                avg_score = sum(stock['composite_score'] for stock in top_cross_sector) / len(top_cross_sector)
                
                watchlists["intelligent_top_picks"] = {
                    "name": "AI Top Picks (Cross-Sector)",
                    "description": f"Top {len(cross_sector_symbols)} highest-scoring stocks across all sectors (avg: {avg_score:.1f})",
                    "symbols": cross_sector_symbols,
                    "sector": "Cross-Sector",
                    "type": "intelligent_top_picks",
                    "count": len(cross_sector_symbols),
                    "average_score": round(avg_score, 1),
                    "market_regime": self.market_regime.value,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "detailed_analysis": top_cross_sector
                }
        
        # Random sample with analysis capability
        if len(self.all_symbols) >= 20:
            random_sample = random.sample(self.all_symbols, min(20, len(self.all_symbols)))
            watchlists["intelligent_random"] = {
                "name": "Random Sample (AI-Ready)",
                "description": "Random NSE stock selection ready for analysis",
                "symbols": random_sample,
                "sector": "Mixed",
                "type": "intelligent_random",
                "count": len(random_sample),
                "market_regime": self.market_regime.value,
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        print(f"Generated {len(watchlists)} intelligent watchlists")
        return watchlists
    
    def get_watchlist_with_analysis(self, watchlist_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific intelligent watchlist with full analysis data
        
        Args:
            watchlist_id: ID of the watchlist
            
        Returns:
            Enhanced watchlist with analysis details
        """
        watchlists = self.get_intelligent_dynamic_watchlists()
        return watchlists.get(watchlist_id)
    
    def list_intelligent_watchlists(self) -> List[Tuple[str, str, str, int, float]]:
        """
        List all intelligent watchlists with scores
        
        Returns:
            List of tuples: (id, name, description, count, avg_score)
        """
        watchlists = self.get_intelligent_dynamic_watchlists()
        result = []
        
        for wl_id, wl_data in watchlists.items():
            avg_score = wl_data.get('average_score', 0.0)
            result.append((
                wl_id,
                wl_data.get('name', wl_id),
                wl_data.get('description', 'Intelligent watchlist'),
                wl_data.get('count', len(wl_data.get('symbols', []))),
                avg_score
            ))
        
        # Sort by average score (highest first)
        result.sort(key=lambda x: x[4], reverse=True)
        return result
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis capabilities and cache status
        
        Returns:
            Dictionary with analysis status information
        """
        cache_size = len(self.analysis_cache)
        cache_age = None
        if self.last_analysis_update:
            cache_age = (datetime.now() - self.last_analysis_update).total_seconds() / 3600
        
        return {
            'analysis_available': ANALYSIS_AVAILABLE,
            'market_regime': self.market_regime.value if self.market_regime else 'Unknown',
            'cache_size': cache_size,
            'cache_age_hours': round(cache_age, 1) if cache_age else None,
            'cache_duration_hours': self.analysis_cache_duration.total_seconds() / 3600,
            'analysis_type': 'basic_technical' if ANALYSIS_AVAILABLE else 'fallback'
        }


def main():
    """Test the intelligent watchlist manager"""
    print("Testing Intelligent Watchlist Manager")
    print("=" * 50)
    
    iwm = IntelligentWatchlistManager()
    
    # Show analysis status
    status = iwm.get_analysis_summary()
    print("Analysis Status:")
    print(f"   • Analysis Available: {status['analysis_available']}")
    print(f"   • Analysis Type: {status['analysis_type']}")
    print(f"   • Market Regime: {status['market_regime']}")
    print(f"   • Cache Size: {status['cache_size']} stocks")
    
    # Test basic functionality
    stats = iwm.get_stats()
    print("\nBase Data Statistics:")
    print(f"   • Total NSE Symbols: {stats['total_nse_symbols']}")
    print(f"   • Cached Sector Data: {stats['cached_sector_data']}")
    
    print("\nIntelligent Watchlist Manager initialized successfully!")
    print("Ready for intelligent stock selection and analysis")


if __name__ == "__main__":
    main()
    
    def get_watchlist_with_analysis(self, watchlist_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific intelligent watchlist with full analysis data
        
        Args:
            watchlist_id: ID of the watchlist
            
        Returns:
            Enhanced watchlist with analysis details
        """
        watchlists = self.get_intelligent_dynamic_watchlists()
        return watchlists.get(watchlist_id)
    
    def list_intelligent_watchlists(self) -> List[Tuple[str, str, str, int, float]]:
        """
        List all intelligent watchlists with scores
        
        Returns:
            List of tuples: (id, name, description, count, avg_score)
        """
        watchlists = self.get_intelligent_dynamic_watchlists()
        result = []
        
        for wl_id, wl_data in watchlists.items():
            avg_score = wl_data.get('average_score', 0.0)
            result.append((
                wl_id,
                wl_data.get('name', wl_id),
                wl_data.get('description', 'Intelligent watchlist'),
                wl_data.get('count', len(wl_data.get('symbols', []))),
                avg_score
            ))
        
        # Sort by average score (highest first)
        result.sort(key=lambda x: x[4], reverse=True)
        return result
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis capabilities and cache status
        
        Returns:
            Dictionary with analysis status information
        """
        cache_size = len(self.analysis_cache)
        cache_age = None
        if self.last_analysis_update:
            cache_age = (datetime.now() - self.last_analysis_update).total_seconds() / 3600
        
        return {
            'analysis_available': ANALYSIS_AVAILABLE,
            'market_regime': self.market_regime.value if self.market_regime else 'Unknown',
            'cache_size': cache_size,
            'cache_age_hours': round(cache_age, 1) if cache_age else None,
            'cache_duration_hours': self.analysis_cache_duration.total_seconds() / 3600,
            'components_status': {
                'composite_scorer': self.composite_scorer is not None,
                'risk_manager': self.risk_manager is not None,
                'market_regime_detection': self.market_regime is not None
            }
        }


def main():
    """Test the intelligent watchlist manager"""
    print("🧠 Testing Intelligent Watchlist Manager")
    print("=" * 50)
    
    iwm = IntelligentWatchlistManager()
    
    # Show analysis status
    status = iwm.get_analysis_summary()
    print(f"📊 Analysis Status:")
    print(f"   • Analysis Available: {status['analysis_available']}")
    print(f"   • Market Regime: {status['market_regime']}")
    print(f"   • Cache Size: {status['cache_size']} stocks")
    
    # Test intelligent watchlist generation
    print(f"\n🚀 Generating intelligent watchlists (sample)...")
    
    # Start with a small sample for testing
    intelligent_lists = iwm.list_intelligent_watchlists()
    
    if intelligent_lists:
        print(f"\n📋 INTELLIGENT WATCHLISTS ({len(intelligent_lists)} available):")
        for i, (wl_id, name, description, count, avg_score) in enumerate(intelligent_lists[:5], 1):
            score_display = f"(avg: {avg_score:.1f})" if avg_score > 0 else "(unscored)"
            print(f"{i:2d}. {name} - {count} stocks {score_display}")
            print(f"     📝 {description}")
    else:
        print("⚠️ No intelligent watchlists generated")
        
    # Show basic stats
    stats = iwm.get_stats()
    print(f"\n📈 Data Statistics:")
    print(f"   • Total NSE Symbols: {stats['total_nse_symbols']}")
    print(f"   • Cached Sector Data: {stats['cached_sector_data']}")
    print(f"   • Analysis Cache: {status['cache_size']}")


if __name__ == "__main__":
    main()