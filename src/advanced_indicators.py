"""
Advanced Technical Indicators Engine
Implements comprehensive technical analysis indicators for the upgraded stock screening system
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedIndicator:
    """
    Comprehensive technical indicators calculator with focus on:
    - Volume anomaly detection (z-score + ratio)
    - Advanced momentum (RSI, MACD with improvements)
    - Trend strength (ADX, MA slopes)
    - Volatility (ATR, normalized)
    - Relative strength vs index
    - Volume profile approximation
    """
    
    def __init__(self, nifty_symbol: str = "^NSEI"):
        self.nifty_symbol = nifty_symbol
        self.nifty_data = None
        
    def get_nifty_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch NIFTY data for relative strength calculations"""
        try:
            if self.nifty_data is None:
                nifty = yf.Ticker(self.nifty_symbol)
                self.nifty_data = nifty.history(period=period)
            return self.nifty_data
        except Exception as e:
            print(f"Warning: Could not fetch NIFTY data: {e}")
            return None
    
    def compute_volume_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute advanced volume signals:
        - Volume ratio (current vs 20-day SMA)
        - Volume z-score (statistical significance)
        - Volume trend (5-day slope)
        """
        if len(data) < 25:
            return {'vol_ratio': np.nan, 'vol_z': np.nan, 'vol_trend': np.nan}
        
        volume = data['Volume']
        
        # Volume ratio (traditional)
        vol_sma20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else np.nan
        
        # Volume z-score (statistical significance)
        vol_mean20 = vol_sma20.iloc[-1]
        vol_std20 = volume.rolling(20).std().iloc[-1]
        vol_z = (volume.iloc[-1] - vol_mean20) / vol_std20 if vol_std20 > 0 else np.nan
        
        # Volume trend (5-day slope)
        if len(volume) >= 5:
            recent_volume = volume.tail(5).values
            x = np.arange(len(recent_volume))
            vol_trend = np.polyfit(x, recent_volume, 1)[0] / vol_mean20 if vol_mean20 > 0 else 0
        else:
            vol_trend = 0
        
        return {
            'vol_ratio': round(vol_ratio, 2) if not np.isnan(vol_ratio) else np.nan,
            'vol_z': round(vol_z, 2) if not np.isnan(vol_z) else np.nan,
            'vol_trend': round(vol_trend, 4) if not np.isnan(vol_trend) else np.nan
        }
    
    def compute_momentum_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute advanced momentum signals:
        - RSI (14-period Wilder's)
        - MACD with histogram momentum
        - MACD normalized strength
        """
        if len(data) < 35:
            return {'rsi': np.nan, 'macd': np.nan, 'macd_signal': np.nan, 
                   'macd_hist': np.nan, 'macd_strength': np.nan}
        
        close = data['Close']
        
        # RSI (Wilder's method)
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        # Use Wilder's smoothing (alpha = 1/14)
        alpha = 1/14
        up_ewm = up.ewm(alpha=alpha, adjust=False).mean()
        down_ewm = down.ewm(alpha=alpha, adjust=False).mean()
        
        rs = up_ewm / down_ewm
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD (12, 26, 9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        # MACD normalized strength
        macd_strength = (current_macd - current_signal) / abs(current_signal) if abs(current_signal) > 1e-8 else 0
        
        return {
            'rsi': round(current_rsi, 2) if not np.isnan(current_rsi) else np.nan,
            'macd': round(current_macd, 6) if not np.isnan(current_macd) else np.nan,
            'macd_signal': round(current_signal, 6) if not np.isnan(current_signal) else np.nan,
            'macd_hist': round(current_hist, 6) if not np.isnan(current_hist) else np.nan,
            'macd_strength': round(macd_strength, 4) if not np.isnan(macd_strength) else np.nan
        }
    
    def compute_trend_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute trend strength signals:
        - ADX (Average Directional Index)
        - Moving average slopes (20, 50)
        - MA crossover status
        """
        if len(data) < 55:
            return {'adx': np.nan, 'ma20_slope': np.nan, 'ma50_slope': np.nan, 'ma_crossover': 0}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # ADX calculation
        def calculate_adx(high, low, close, window=14):
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = np.where((high - high.shift()) > (low.shift() - low), 
                              np.maximum(high - high.shift(), 0), 0)
            dm_minus = np.where((low.shift() - low) > (high - high.shift()), 
                               np.maximum(low.shift() - low, 0), 0)
            
            dm_plus = pd.Series(dm_plus, index=high.index)
            dm_minus = pd.Series(dm_minus, index=high.index)
            
            # Smoothed values
            tr_smooth = tr.ewm(alpha=1/window, adjust=False).mean()
            dm_plus_smooth = dm_plus.ewm(alpha=1/window, adjust=False).mean()
            dm_minus_smooth = dm_minus.ewm(alpha=1/window, adjust=False).mean()
            
            # Directional Indicators
            di_plus = 100 * dm_plus_smooth / tr_smooth
            di_minus = 100 * dm_minus_smooth / tr_smooth
            
            # ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.ewm(alpha=1/window, adjust=False).mean()
            
            return adx.iloc[-1] if not adx.empty else np.nan
        
        current_adx = calculate_adx(high, low, close)
        
        # Moving averages and slopes
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        
        # MA slopes (normalized by price)
        if len(ma20) >= 5:
            ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / (5 * close.iloc[-1])
        else:
            ma20_slope = np.nan
            
        if len(ma50) >= 5:
            ma50_slope = (ma50.iloc[-1] - ma50.iloc[-5]) / (5 * close.iloc[-1])
        else:
            ma50_slope = np.nan
        
        # MA crossover (1 = MA20 > MA50, 0 = neutral, -1 = MA20 < MA50)
        if not np.isnan(ma20.iloc[-1]) and not np.isnan(ma50.iloc[-1]):
            ma_crossover = 1 if ma20.iloc[-1] > ma50.iloc[-1] else -1
        else:
            ma_crossover = 0
        
        return {
            'adx': round(current_adx, 2) if not np.isnan(current_adx) else np.nan,
            'ma20_slope': round(ma20_slope, 6) if not np.isnan(ma20_slope) else np.nan,
            'ma50_slope': round(ma50_slope, 6) if not np.isnan(ma50_slope) else np.nan,
            'ma_crossover': ma_crossover
        }
    
    def compute_volatility_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute volatility signals:
        - ATR (Average True Range)
        - ATR relative to price
        - Volatility trend
        """
        if len(data) < 20:
            return {'atr': np.nan, 'atr_pct': np.nan, 'atr_trend': np.nan}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (14-period Wilder's smoothing)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        current_atr = atr.iloc[-1]
        
        # ATR as percentage of price
        atr_pct = (current_atr / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else np.nan
        
        # ATR trend (increasing vs decreasing volatility)
        if len(atr) >= 5:
            atr_trend = (atr.iloc[-1] - atr.iloc[-5]) / atr.iloc[-5] if atr.iloc[-5] > 0 else 0
        else:
            atr_trend = 0
        
        return {
            'atr': round(current_atr, 4) if not np.isnan(current_atr) else np.nan,
            'atr_pct': round(atr_pct, 2) if not np.isnan(atr_pct) else np.nan,
            'atr_trend': round(atr_trend, 4) if not np.isnan(atr_trend) else np.nan
        }
    
    def compute_relative_strength(self, data: pd.DataFrame, symbol: str, period: int = 20) -> Dict[str, float]:
        """
        Compute relative strength vs NIFTY:
        - 20-day relative performance
        - 50-day relative performance (if available)
        """
        if len(data) < period + 5:
            return {'rel_strength_20d': np.nan, 'rel_strength_50d': np.nan}
        
        nifty_data = self.get_nifty_data()
        if nifty_data is None or len(nifty_data) < period + 5:
            return {'rel_strength_20d': np.nan, 'rel_strength_50d': np.nan}
        
        # Align dates
        stock_close = data['Close']
        common_dates = stock_close.index.intersection(nifty_data.index)
        
        if len(common_dates) < period + 1:
            return {'rel_strength_20d': np.nan, 'rel_strength_50d': np.nan}
        
        # Get aligned data
        stock_aligned = stock_close.loc[common_dates]
        nifty_aligned = nifty_data['Close'].loc[common_dates]
        
        # 20-day relative strength
        if len(stock_aligned) >= period:
            stock_return_20d = (stock_aligned.iloc[-1] - stock_aligned.iloc[-(period+1)]) / stock_aligned.iloc[-(period+1)]
            nifty_return_20d = (nifty_aligned.iloc[-1] - nifty_aligned.iloc[-(period+1)]) / nifty_aligned.iloc[-(period+1)]
            rel_strength_20d = (stock_return_20d - nifty_return_20d) * 100
        else:
            rel_strength_20d = np.nan
        
        # 50-day relative strength
        if len(stock_aligned) >= 50:
            stock_return_50d = (stock_aligned.iloc[-1] - stock_aligned.iloc[-51]) / stock_aligned.iloc[-51]
            nifty_return_50d = (nifty_aligned.iloc[-1] - nifty_aligned.iloc[-51]) / nifty_aligned.iloc[-51]
            rel_strength_50d = (stock_return_50d - nifty_return_50d) * 100
        else:
            rel_strength_50d = np.nan
        
        return {
            'rel_strength_20d': round(rel_strength_20d, 2) if not np.isnan(rel_strength_20d) else np.nan,
            'rel_strength_50d': round(rel_strength_50d, 2) if not np.isnan(rel_strength_50d) else np.nan
        }
    
    def compute_volume_profile_proxy(self, data: pd.DataFrame, lookback: int = 90) -> Dict[str, float]:
        """
        Compute volume profile approximation:
        - Identify high-volume price nodes
        - Check if current price is breaking above/below key levels
        """
        if len(data) < lookback:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}
        
        # Use recent data for volume profile
        recent_data = data.tail(lookback).copy()
        
        # Create price buckets (simplified volume profile)
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        num_buckets = min(20, len(recent_data) // 5)  # Adaptive bucket count
        
        if price_range <= 0 or num_buckets < 3:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}
        
        bucket_size = price_range / num_buckets
        
        # Calculate volume at each price level
        volume_at_price = {}
        
        for _, row in recent_data.iterrows():
            # Approximate volume distribution within the day's range
            low, high, volume = row['Low'], row['High'], row['Volume']
            if high > low:
                # Distribute volume across price buckets for this day
                start_bucket = int((low - recent_data['Low'].min()) // bucket_size)
                end_bucket = int((high - recent_data['Low'].min()) // bucket_size)
                
                buckets_touched = max(1, end_bucket - start_bucket + 1)
                volume_per_bucket = volume / buckets_touched
                
                for bucket in range(start_bucket, min(end_bucket + 1, num_buckets)):
                    if bucket not in volume_at_price:
                        volume_at_price[bucket] = 0
                    volume_at_price[bucket] += volume_per_bucket
        
        if not volume_at_price:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}
        
        # Find high volume nodes (top 20% by volume)
        volume_threshold = np.percentile(list(volume_at_price.values()), 80)
        high_volume_buckets = [bucket for bucket, vol in volume_at_price.items() if vol >= volume_threshold]
        
        # Convert buckets to price levels
        high_volume_prices = []
        for bucket in high_volume_buckets:
            price = recent_data['Low'].min() + (bucket + 0.5) * bucket_size
            high_volume_prices.append(price)
        
        # Check current price vs high volume levels
        current_price = data['Close'].iloc[-1]
        breakout_score = 0
        resistance_level = np.nan
        
        if high_volume_prices:
            # Find nearest resistance level above current price
            resistance_levels = [p for p in high_volume_prices if p > current_price]
            if resistance_levels:
                resistance_level = min(resistance_levels)
                # Score based on how close we are to breaking resistance
                distance_pct = (resistance_level - current_price) / current_price * 100
                if distance_pct < 2:  # Within 2% of resistance
                    breakout_score = 10
                elif distance_pct < 5:  # Within 5% of resistance
                    breakout_score = 5
            
            # Check if we've broken above a recent high-volume level
            broken_levels = [p for p in high_volume_prices if current_price > p * 1.02]  # 2% above
            if broken_levels:
                breakout_score = max(breakout_score, 8)
        
        return {
            'vp_breakout_score': breakout_score,
            'vp_resistance_level': round(resistance_level, 2) if not np.isnan(resistance_level) else np.nan
        }
    
    def compute_weekly_confirmation(self, symbol: str) -> Dict[str, Any]:
        """
        Compute weekly timeframe indicators for confirmation:
        - Weekly RSI trend
        - Weekly MACD signal
        - Weekly volume trend
        """
        try:
            # All symbols should already have the .NS suffix
            # Fetch weekly data
            ticker = yf.Ticker(symbol)
            weekly_data = ticker.history(period="6mo", interval="1wk")
            
            if weekly_data is None or len(weekly_data) < 15 or weekly_data.empty:
                return {'weekly_rsi_trend': 0, 'weekly_macd_bullish': False, 'weekly_vol_trend': 0}
            
            # Weekly RSI
            close = weekly_data['Close']
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            up_ewm = up.ewm(alpha=1/14, adjust=False).mean()
            down_ewm = down.ewm(alpha=1/14, adjust=False).mean()
            
            rs = up_ewm / down_ewm
            rsi = 100 - (100 / (1 + rs))
            
            # RSI trend (positive if increasing over last 3 weeks)
            if len(rsi) >= 3:
                rsi_trend = 1 if rsi.iloc[-1] > rsi.iloc[-3] else 0
            else:
                rsi_trend = 0
            
            # Weekly MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1] if len(macd_line) > 0 else False
            
            # Weekly volume trend
            volume = weekly_data['Volume']
            if len(volume) >= 4:
                vol_trend = 1 if volume.tail(2).mean() > volume.iloc[-4:-2].mean() else 0
            else:
                vol_trend = 0
            
            return {
                'weekly_rsi_trend': rsi_trend,
                'weekly_macd_bullish': macd_bullish,
                'weekly_vol_trend': vol_trend
            }
            
        except Exception as e:
            print(f"Warning: Could not compute weekly indicators for {symbol}: {e}")
            return {'weekly_rsi_trend': 0, 'weekly_macd_bullish': False, 'weekly_vol_trend': 0}
    
    def compute_all_indicators(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Compute all indicators for a given symbol
        Returns comprehensive dictionary of all technical indicators
        """
        try:
            # All symbols should already have the .NS suffix
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data is None or len(data) < 50 or data.empty:
                print(f"Insufficient data for {symbol}")
                return None
            
            # Compute all indicator groups
            volume_signals = self.compute_volume_signals(data)
            momentum_signals = self.compute_momentum_signals(data)
            trend_signals = self.compute_trend_signals(data)
            volatility_signals = self.compute_volatility_signals(data)
            relative_strength = self.compute_relative_strength(data, symbol)
            volume_profile = self.compute_volume_profile_proxy(data)
            weekly_confirm = self.compute_weekly_confirmation(symbol)
            
            # Combine all indicators
            all_indicators = {
                'symbol': symbol,
                'current_price': round(data['Close'].iloc[-1], 2),
                'price_change_pct': round(((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100, 2),
                **volume_signals,
                **momentum_signals,
                **trend_signals,
                **volatility_signals,
                **relative_strength,
                **volume_profile,
                **weekly_confirm
            }
            
            return all_indicators
            
        except Exception as e:
            print(f"Error computing indicators for {symbol}: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Test the indicators
    indicators_engine = AdvancedIndicator()
    
    # Test with a sample stock
    test_symbol = "RELIANCE.NS"
    print(f"Testing indicators for {test_symbol}...")
    
    result = indicators_engine.compute_all_indicators(test_symbol)
    
    if result:
        print(f"\nIndicators for {test_symbol}:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("Failed to compute indicators")