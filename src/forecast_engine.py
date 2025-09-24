"""
Forecast Engine Module
Estimates duration to reach target prices using multiple methodologies.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from core import DataFetcher
from constants import TRADING_CONSTANTS

class ForecastEngine:
    """
    Duration estimation for price targets using multiple methodologies:
    1. ATR-based speed estimation
    2. Historical volatility patterns
    3. Moving average trend speed
    """
    
    def __init__(self):
        self.cache = {}  # Cache for historical data
        self.cache_expiry = {}  # Cache expiry times
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    def estimate_duration(self,
                         symbol: str,
                         entry_price: float,
                         target_price: float,
                         indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate duration to reach target price using multiple methods.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price level
            target_price: Target price level
            indicators: Technical indicators dictionary
            
        Returns:
            Dictionary with duration estimates and confidence
        """
        try:
            # Calculate price move required
            price_move = abs(target_price - entry_price)
            move_direction = 1 if target_price > entry_price else -1
            
            if price_move <= 0:
                return self._default_forecast("Invalid price move")
            
            # Get historical data for analysis
            data = self._get_cached_data(symbol)
            if data is None or len(data) < 50:
                return self._default_forecast("Insufficient historical data")
            
            # Method 1: ATR-based estimation
            atr_estimate = self._estimate_by_atr(price_move, indicators, data)
            
            # Method 2: Historical volatility patterns
            volatility_estimate = self._estimate_by_volatility(
                symbol, price_move, move_direction, data
            )
            
            # Method 3: Moving average trend speed
            trend_estimate = self._estimate_by_trend(price_move, move_direction, data)
            
            # Combine estimates with weights
            combined_estimate = self._combine_estimates(
                atr_estimate, volatility_estimate, trend_estimate
            )
            
            return combined_estimate
            
        except Exception as e:
            return self._default_forecast(f"Error in duration estimation: {e}")
    
    def _get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached historical data or fetch new data"""
        current_time = datetime.now()
        
        # Check if cached data is still valid
        if (symbol in self.cache and 
            symbol in self.cache_expiry and 
            current_time < self.cache_expiry[symbol]):
            return self.cache[symbol]
        
        # Fetch new data
        data = DataFetcher.fetch_stock_data(symbol, period="2y")
        if data is not None:
            self.cache[symbol] = data
            self.cache_expiry[symbol] = current_time + self.cache_duration
        
        return data
    
    def _estimate_by_atr(self, 
                        price_move: float, 
                        indicators: Dict[str, Any], 
                        data: pd.DataFrame) -> Tuple[int, float]:
        """Estimate duration based on ATR (Average True Range)"""
        try:
            atr = indicators.get('atr', 0)
            if atr <= 0:
                # Calculate ATR from data if not available
                atr = self._calculate_atr(data)
            
            if atr <= 0:
                return 30, 0.3  # Default: 30 days, low confidence
            
            # Estimate: price_move / daily_atr_movement
            # Assumption: market moves roughly 0.5-1.0 ATR per day on average
            daily_atr_movement = atr * 0.7  # Conservative estimate
            estimated_days = max(1, int(price_move / daily_atr_movement))
            
            # Cap at reasonable limits
            estimated_days = min(120, max(3, estimated_days))  # 3-120 days
            
            # Confidence based on ATR reliability
            confidence = 0.7 if atr > 0 else 0.3
            
            return estimated_days, confidence
            
        except Exception:
            return 30, 0.2
    
    def _estimate_by_volatility(self, 
                               symbol: str,
                               price_move: float,
                               direction: int,
                               data: pd.DataFrame) -> Tuple[int, float]:
        """Estimate duration based on historical volatility patterns"""
        try:
            # Calculate historical moves similar to target move
            returns = data['Close'].pct_change().dropna()
            price_levels = data['Close']
            
            # Look for similar percentage moves in the past
            target_pct_move = price_move / price_levels.iloc[-1]
            
            # Find historical moves of similar magnitude
            similar_moves = []
            lookback_window = 20  # 20-day rolling window
            
            for i in range(lookback_window, len(price_levels)):
                window_start = price_levels.iloc[i - lookback_window]
                window_end = price_levels.iloc[i]
                window_move = abs(window_end - window_start) / window_start
                
                # If similar magnitude move (within 50% tolerance)
                if 0.5 * target_pct_move <= window_move <= 1.5 * target_pct_move:
                    # Check if direction matches
                    actual_direction = 1 if window_end > window_start else -1
                    if actual_direction == direction:
                        similar_moves.append(lookback_window)
            
            if similar_moves:
                avg_duration = np.mean(similar_moves)
                confidence = min(0.8, len(similar_moves) / 10)  # Higher confidence with more samples
                return int(avg_duration), confidence
            else:
                # Use volatility-based estimation
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Higher volatility = faster moves
                if volatility > 0.3:  # High volatility (>30%)
                    estimated_days = max(5, int(target_pct_move / volatility * 60))
                elif volatility > 0.2:  # Medium volatility (20-30%)
                    estimated_days = max(10, int(target_pct_move / volatility * 80))
                else:  # Low volatility (<20%)
                    estimated_days = max(15, int(target_pct_move / volatility * 100))
                
                estimated_days = min(90, estimated_days)
                return estimated_days, 0.5
                
        except Exception:
            return 25, 0.3
    
    def _estimate_by_trend(self, 
                          price_move: float,
                          direction: int,
                          data: pd.DataFrame) -> Tuple[int, float]:
        """Estimate duration based on moving average trend speed"""
        try:
            close_prices = data['Close']
            
            # Calculate trend using EMA20 and EMA50
            ema_20 = close_prices.ewm(span=20).mean()
            ema_50 = close_prices.ewm(span=50).mean()
            
            # Calculate recent trend speed (last 20 days)
            recent_ema20_slope = (ema_20.iloc[-1] - ema_20.iloc[-20]) / 20
            recent_ema50_slope = (ema_50.iloc[-1] - ema_50.iloc[-20]) / 20
            
            # Use the faster trend for estimation
            trend_speed = max(abs(recent_ema20_slope), abs(recent_ema50_slope))
            
            if trend_speed > 0:
                # Check if trend direction matches target direction
                trend_direction = 1 if recent_ema20_slope > 0 else -1
                
                if trend_direction == direction:
                    # Favorable trend
                    estimated_days = max(5, int(price_move / trend_speed))
                    confidence = 0.6
                else:
                    # Against trend - will take longer
                    estimated_days = max(15, int(price_move / trend_speed * 2))
                    confidence = 0.4
                
                estimated_days = min(100, estimated_days)
                return estimated_days, confidence
            else:
                # No clear trend
                return 30, 0.3
                
        except Exception:
            return 35, 0.2
    
    def _combine_estimates(self, 
                          atr_est: Tuple[int, float],
                          vol_est: Tuple[int, float],
                          trend_est: Tuple[int, float]) -> Dict[str, Any]:
        """Combine multiple estimates with confidence weighting"""
        
        estimates = [atr_est, vol_est, trend_est]
        method_names = ['ATR-based', 'Volatility-based', 'Trend-based']
        
        # Weight by confidence
        total_weight = sum(conf for _, conf in estimates)
        if total_weight == 0:
            return self._default_forecast("No reliable estimates")
        
        weighted_duration = sum(days * conf for days, conf in estimates) / total_weight
        final_duration = max(1, int(round(weighted_duration)))
        
        # Calculate overall confidence
        avg_confidence = total_weight / len(estimates)
        
        # Adjust confidence based on estimate agreement
        durations = [days for days, _ in estimates]
        duration_std = np.std(durations)
        duration_mean = np.mean(durations)
        
        # Lower confidence if estimates disagree significantly
        if duration_mean > 0:
            agreement_factor = 1 - min(0.5, duration_std / duration_mean)
            final_confidence = avg_confidence * agreement_factor
        else:
            final_confidence = avg_confidence
        
        # Provide breakdown for transparency
        estimate_breakdown = {}
        for i, (days, conf) in enumerate(estimates):
            estimate_breakdown[method_names[i]] = {
                'days': days,
                'confidence': round(conf, 2)
            }
        
        return {
            'estimated_duration_days': final_duration,
            'confidence': round(final_confidence, 2),
            'method_breakdown': estimate_breakdown,
            'consensus_level': 'High' if final_confidence >= 0.7 else 'Medium' if final_confidence >= 0.4 else 'Low'
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range from OHLC data"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else 0.0
            
        except Exception:
            return 0.0
    
    def _default_forecast(self, reason: str) -> Dict[str, Any]:
        """Return default forecast when calculation fails"""
        return {
            'estimated_duration_days': 21,  # Default 3 weeks
            'confidence': 0.1,
            'method_breakdown': {'Error': {'days': 21, 'confidence': 0.1}},
            'consensus_level': 'Low',
            'error_reason': reason
        }
    
    def forecast_batch(self, 
                      forecast_requests: list) -> Dict[str, Dict[str, Any]]:
        """Process multiple forecast requests efficiently"""
        results = {}
        
        for request in forecast_requests:
            symbol = request.get('symbol')
            entry_price = request.get('entry_price', 0)
            target_price = request.get('target_price', 0)
            indicators = request.get('indicators', {})
            
            if symbol and entry_price > 0 and target_price > 0:
                results[symbol] = self.estimate_duration(
                    symbol, entry_price, target_price, indicators
                )
            else:
                results[symbol] = self._default_forecast("Invalid request parameters")
        
        return results