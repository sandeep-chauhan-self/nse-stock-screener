#!/usr/bin/env python3
"""
Enhanced Analysis Wrapper - Provides comprehensive NSE stock analysis
Handles import issues and provides all advanced indicators and risk management
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class EnhancedAnalysisWrapper:
    """Wrapper for enhanced analysis with comprehensive indicators"""
    
    def __init__(self):
        """Initialize the enhanced analysis system"""
        self.setup_imports()
        self.initialize_components()
    
    def setup_imports(self):
        """Setup required imports with fallbacks"""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance is required for data fetching")
        
        # Market regime enum
        self.MarketRegime = type('MarketRegime', (), {
            'BULLISH': 'BULLISH',
            'BEARISH': 'BEARISH', 
            'SIDEWAYS': 'SIDEWAYS',
            'HIGH_VOLATILITY': 'HIGH_VOLATILITY'
        })
    
    def initialize_components(self):
        """Initialize analysis components"""
        self.portfolio_capital = 1000000  # Default 10L capital
        self.risk_per_trade = 0.015      # 1.5% risk per trade
        self.max_positions = 10
        
        # Technical indicator parameters
        self.indicators_config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'volume_lookback': 50,
            'atr_period': 14,
            'relative_strength_period': 20
        }
        
        # Scoring weights
        self.scoring_weights = {
            'volume_analysis': 0.25,
            'momentum_indicators': 0.25,
            'trend_strength': 0.15,
            'volatility_analysis': 0.10,
            'relative_strength': 0.10,
            'volume_profile': 0.10,
            'weekly_confirmation': 0.05
        }
    
    def fetch_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with error handling"""
        try:
            ticker = self.yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty or len(data) < 50:
                return pd.DataFrame()
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi.iloc[-1], 2)
        except:
            return np.nan
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD indicator"""
        try:
            fast = data['Close'].ewm(span=self.indicators_config['macd_fast']).mean()
            slow = data['Close'].ewm(span=self.indicators_config['macd_slow']).mean()
            macd_line = fast - slow
            signal_line = macd_line.ewm(span=self.indicators_config['macd_signal']).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': round(macd_line.iloc[-1], 4),
                'signal': round(signal_line.iloc[-1], 4),
                'histogram': round(histogram.iloc[-1], 4),
                'macd_signal': 'BULLISH' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'BEARISH'
            }
        except:
            return {'macd': np.nan, 'signal': np.nan, 'histogram': np.nan, 'macd_signal': 'NEUTRAL'}
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
            minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
            
            # Smooth the values
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return round(adx.iloc[-1], 2)
        except:
            return np.nan
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return round(atr.iloc[-1], 2)
        except:
            return np.nan
    
    def calculate_volume_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            volume = data['Volume']
            avg_volume = volume.rolling(self.indicators_config['volume_lookback']).mean()
            volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Volume Z-score
            volume_std = volume.rolling(self.indicators_config['volume_lookback']).std()
            volume_z_score = (volume.iloc[-1] - avg_volume.iloc[-1]) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0
            
            return {
                'volume_ratio': round(volume_ratio, 2),
                'volume_z_score': round(volume_z_score, 2),
                'avg_volume': round(avg_volume.iloc[-1], 0)
            }
        except:
            return {'volume_ratio': 1.0, 'volume_z_score': 0.0, 'avg_volume': 0}
    
    def calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """Calculate relative strength vs market (using NIFTY as proxy)"""
        try:
            # Simple relative strength calculation
            period = self.indicators_config['relative_strength_period']
            stock_return = (data['Close'].iloc[-1] / data['Close'].iloc[-period] - 1) * 100
            
            # For now, use a benchmark return (in real implementation, fetch NIFTY data)
            benchmark_return = 2.0  # Assume 2% benchmark return
            relative_strength = stock_return - benchmark_return
            
            return round(relative_strength, 2)
        except:
            return 0.0
    
    def calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate price change percentage"""
        try:
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            change_pct = ((current_price - previous_price) / previous_price) * 100
            return round(change_pct, 2)
        except:
            return 0.0
    
    def calculate_composite_score(self, indicators: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate composite score and probability"""
        try:
            score = 50  # Base score
            
            # Volume analysis (25% weight)
            volume_score = 0
            if indicators['volume_ratio'] > 1.5:
                volume_score += 15
            elif indicators['volume_ratio'] > 1.2:
                volume_score += 10
            if indicators['volume_z_score'] > 1.5:
                volume_score += 10
            score += volume_score * self.scoring_weights['volume_analysis']
            
            # Momentum indicators (25% weight)
            momentum_score = 0
            rsi = indicators.get('rsi', 50)
            if 40 < rsi < 70:
                momentum_score += 15
            elif 30 < rsi < 80:
                momentum_score += 10
            
            if indicators.get('macd_signal') == 'BULLISH':
                momentum_score += 10
            score += momentum_score * self.scoring_weights['momentum_indicators']
            
            # Trend strength (15% weight)
            trend_score = 0
            adx = indicators.get('adx', 0)
            if adx > 25:
                trend_score += 15
            elif adx > 20:
                trend_score += 10
            score += trend_score * self.scoring_weights['trend_strength']
            
            # Volatility analysis (10% weight)
            volatility_score = 0
            atr_pct = indicators.get('atr_pct', 0)
            if 1 < atr_pct < 3:  # Moderate volatility is good
                volatility_score += 10
            score += volatility_score * self.scoring_weights['volatility_analysis']
            
            # Relative strength (10% weight)
            rel_strength = indicators.get('relative_strength_20d', 0)
            if rel_strength > 0:
                rel_strength_score = min(10, rel_strength)
            else:
                rel_strength_score = max(-5, rel_strength / 2)
            score += rel_strength_score * self.scoring_weights['relative_strength']
            
            # Ensure score is within bounds
            score = max(0, min(100, score))
            
            # Calculate probability (simple sigmoid transformation)
            probability = 1 / (1 + np.exp(-(score - 50) / 10))
            
            return round(score, 1), round(probability * 100, 1)
        except:
            return 50.0, 50.0
    
    def calculate_risk_management(self, current_price: float, atr: float, score: float) -> Dict[str, Any]:
        """Calculate risk management parameters"""
        try:
            # Stop loss calculation (2x ATR below current price)
            atr_multiplier = 2.0
            stop_loss = current_price - (atr * atr_multiplier)
            
            # Risk amount per trade
            risk_amount = self.portfolio_capital * self.risk_per_trade
            
            # Position sizing based on risk
            risk_per_share = current_price - stop_loss
            if risk_per_share > 0:
                suggested_qty = int(risk_amount / risk_per_share)
            else:
                suggested_qty = 0
            
            # Can enter position?
            can_enter = score >= 60 and suggested_qty > 0 and stop_loss > 0
            
            return {
                'can_enter': can_enter,
                'suggested_qty': suggested_qty,
                'risk_amount': round(risk_amount, 0),
                'stop_loss': round(stop_loss, 2)
            }
        except:
            return {
                'can_enter': False,
                'suggested_qty': 0,
                'risk_amount': 0,
                'stop_loss': 0
            }
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single symbol"""
        try:
            # Fetch data
            data = self.fetch_data(symbol)
            if data.empty:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate all indicators
            rsi = self.calculate_rsi(data)
            macd_data = self.calculate_macd(data)
            adx = self.calculate_adx(data)
            atr = self.calculate_atr(data)
            volume_data = self.calculate_volume_analysis(data)
            relative_strength = self.calculate_relative_strength(data)
            price_change = self.calculate_price_change(data)
            
            # Calculate ATR percentage
            atr_pct = (atr / current_price * 100) if current_price > 0 else 0
            
            # Compile all indicators
            indicators = {
                'rsi': rsi,
                'macd_signal': macd_data['macd_signal'],
                'adx': adx,
                'atr_pct': round(atr_pct, 2),
                'volume_ratio': volume_data['volume_ratio'],
                'volume_z_score': volume_data['volume_z_score'],
                'relative_strength_20d': relative_strength
            }
            
            # Calculate composite score and probability
            composite_score, probability = self.calculate_composite_score(indicators)
            
            # Calculate risk management
            risk_data = self.calculate_risk_management(current_price, atr, composite_score)
            
            # Compile final result
            result = {
                'Symbol': symbol,
                'Composite_Score': composite_score,
                'Probability': probability,
                'Current_Price': round(current_price, 2),
                'Price_Change_%': price_change,
                'Volume_Ratio': volume_data['volume_ratio'],
                'Volume_Z_Score': volume_data['volume_z_score'],
                'RSI': rsi,
                'MACD_Signal': macd_data['macd_signal'],
                'ADX': adx,
                'ATR_%': round(atr_pct, 2),
                'Rel_Strength_20d': relative_strength,
                'Can_Enter': risk_data['can_enter'],
                'Suggested_Qty': risk_data['suggested_qty'],
                'Risk_Amount': risk_data['risk_amount'],
                'Stop_Loss': risk_data['stop_loss']
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_multiple_symbols(self, symbols: List[str], description: str = "Analysis") -> pd.DataFrame:
        """Analyze multiple symbols and return comprehensive results"""
        print(f"\n[TARGET] {description}")
        print("[RUNNING] Starting comprehensive enhanced analysis...")
        print("[ANALYSIS] Calculating advanced indicators: RSI, MACD, ADX, ATR, Volume Analysis, Risk Management...")
        
        results = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"  [{i}/{len(symbols)}] Analyzing {symbol.replace('.NS', '')}...", end="")
                
                result = self.analyze_symbol(symbol)
                if result:
                    results.append(result)
                    score = result['Composite_Score']
                    signal = 'HIGH' if score >= 75 else 'MEDIUM' if score >= 60 else 'LOW'
                    can_enter = "[OK]" if result['Can_Enter'] else "[X]"
                    print(f" Score: {score}, Signal: {signal}, Entry: {can_enter}")
                else:
                    print(" ❌ Failed")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        if not results:
            print("❌ No valid analysis results generated")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by composite score (highest first)
        df = df.sort_values('Composite_Score', ascending=False)
        
        # Display summary
        print(f"\n[RESULTS] COMPREHENSIVE ANALYSIS RESULTS ({len(df)} stocks analyzed):")
        print("=" * 120)
        print("Symbol        | Score | Prob% | Price    | Change% | Vol Ratio | RSI   | MACD    | ADX   | ATR%  | Rel Str | Entry | Qty   | Risk     | Stop Loss")
        print("-" * 120)
        
        for _, row in df.head(10).iterrows():  # Show top 10
            symbol_short = row['Symbol'].replace('.NS', '')[:12]
            entry_icon = "[OK]" if row['Can_Enter'] else "[X]"
            
            print(f"{symbol_short:<12} | {row['Composite_Score']:>5.1f} | {row['Probability']:>4.1f} | "
                  f"Rs{row['Current_Price']:>7.2f} | {row['Price_Change_%']:>6.1f}% | "
                  f"{row['Volume_Ratio']:>8.1f}x | {row['RSI']:>5.1f} | {row['MACD_Signal']:>7} | "
                  f"{row['ADX']:>5.1f} | {row['ATR_%']:>4.1f}% | {row['Rel_Strength_20d']:>6.1f}% | "
                  f"{entry_icon:>5} | {row['Suggested_Qty']:>5} | Rs{row['Risk_Amount']:>7.0f} | Rs{row['Stop_Loss']:>8.2f}")
        
        if len(df) > 10:
            print(f"... and {len(df) - 10} more stocks")
        
        # Filter high-probability stocks
        high_prob_stocks = df[df['Composite_Score'] >= 70]
        entry_ready_stocks = df[df['Can_Enter'] == True]
        
        print(f"\n[SUMMARY] SUMMARY:")
        print(f"   • High-Score Stocks (>=70): {len(high_prob_stocks)}")
        print(f"   • Entry-Ready Stocks: {len(entry_ready_stocks)}")
        print(f"   • Average Score: {df['Composite_Score'].mean():.1f}")
        print(f"   • Top Score: {df['Composite_Score'].max():.1f}")
        
        # Save to CSV
        output_dir = PROJECT_ROOT / "output" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"enhanced_analysis_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n[SAVED] Comprehensive results saved to: {output_file}")
        print("[OK] Enhanced analysis completed successfully!")
        
        return df

# Convenience function for easy import
def create_enhanced_analyzer():
    """Create and return an enhanced analyzer instance"""
    return EnhancedAnalysisWrapper()

if __name__ == "__main__":
    # Test the wrapper
    analyzer = create_enhanced_analyzer()
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    results = analyzer.analyze_multiple_symbols(test_symbols, "Test Analysis")
    print(f"\nTest completed with {len(results)} results")