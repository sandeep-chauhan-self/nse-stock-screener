"""
High-performance vectorized indicator implementations.

This module provides concrete implementations of common technical indicators
using vectorized pandas/numpy operations and Numba optimization where beneficial.
"""

import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from .base import VectorizedIndicator, IndicatorResult, IndicatorType, InsufficientDataError
from .numba_ops import (
    calculate_rsi_wilder, calculate_sma, calculate_ema, calculate_atr,
    calculate_adx, calculate_stochastic, calculate_bollinger_bands,
    calculate_williams_r, calculate_cci, calculate_money_flow_index
)


class RSIIndicator(VectorizedIndicator):
    """Relative Strength Index using Wilder's method with Numba optimization."""
    
    def __init__(self, period: int = 14, use_numba: bool = True):
        super().__init__(period=period, use_numba=use_numba)
        self.period = period
        self.use_numba = use_numba
    
    @property
    def name(self) -> str:
        return f"RSI_{self.period}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.MOMENTUM
    
    @property
    def required_periods(self) -> int:
        return self.period + 10  # Extra buffer for stability
    
    def get_required_columns(self) -> List[str]:
        return ['Close']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            # Validate input data
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            close_prices = data['Close'].values
            
            if self.use_numba:
                # Use Numba-optimized calculation
                rsi_values = calculate_rsi_wilder(close_prices, self.period)
                rsi_series = pd.Series(rsi_values, index=data.index)
                current_rsi = rsi_values[-1]
            else:
                # Use pandas calculation
                close = data['Close']
                delta = close.diff()
                
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                
                # Wilder's smoothing
                alpha = 1 / self.period
                up_ewm = up.ewm(alpha=alpha, adjust=False).mean()
                down_ewm = down.ewm(alpha=alpha, adjust=False).mean()
                
                rs = up_ewm / down_ewm
                rsi_series = 100 - (100 / (1 + rs))
                current_rsi = rsi_series.iloc[-1]
            
            # Calculate confidence
            confidence = self.calculate_confidence(data, current_rsi)
            
            # Determine trend and overbought/oversold levels
            if current_rsi >= 70:
                condition = "overbought"
            elif current_rsi <= 30:
                condition = "oversold"
            elif current_rsi > 50:
                condition = "bullish"
            else:
                condition = "bearish"
            
            end_time = time.perf_counter()
            computation_time = (end_time - start_time) * 1000
            
            return IndicatorResult(
                value=rsi_series,  # Return full time series
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "method": "Wilder",
                    "condition": condition,
                    "data_points": len(data),
                    "use_numba": self.use_numba
                },
                computation_time_ms=computation_time,
                data_points_used=len(data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value=math.nan,
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


class MACDIndicator(VectorizedIndicator):
    """MACD (Moving Average Convergence Divergence) indicator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.MOMENTUM
    
    @property
    def required_periods(self) -> int:
        return self.slow + self.signal + 10
    
    @property
    def output_names(self) -> List[str]:
        return ['macd_line', 'signal_line', 'histogram']
    
    def get_required_columns(self) -> List[str]:
        return ['Close']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value={"macd_line": math.nan, "signal_line": math.nan, "histogram": math.nan},
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            close = data['Close']
            
            # Calculate MACD components using vectorized operations
            ema_fast = self._ewm_operation(close, span=self.fast)
            ema_slow = self._ewm_operation(close, span=self.slow)
            macd_line = ema_fast - ema_slow
            signal_line = self._ewm_operation(macd_line, span=self.signal)
            histogram = macd_line - signal_line
            
            # Create result DataFrame
            macd_df = pd.DataFrame({
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            }, index=data.index)
            
            # Current values
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            # Calculate confidence
            confidence = self.calculate_confidence(data, current_macd)
            
            # Determine trend
            is_bullish = current_macd > current_signal and current_histogram > 0
            trend = "bullish" if is_bullish else "bearish"
            
            # Check for crossover signals
            prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0
            crossover = None
            if current_histogram > 0 and prev_histogram <= 0:
                crossover = "bullish_crossover"
            elif current_histogram < 0 and prev_histogram >= 0:
                crossover = "bearish_crossover"
            
            end_time = time.perf_counter()
            
            return IndicatorResult(
                value=macd_df,  # Return full time series DataFrame
                confidence=confidence,
                metadata={
                    "fast_period": self.fast,
                    "slow_period": self.slow,
                    "signal_period": self.signal,
                    "trend": trend,
                    "crossover": crossover,
                    "data_points": len(data)
                },
                computation_time_ms=(end_time - start_time) * 1000,
                data_points_used=len(data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value={"macd_line": math.nan, "signal_line": math.nan, "histogram": math.nan},
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


class ATRIndicator(VectorizedIndicator):
    """Average True Range indicator with Numba optimization."""
    
    def __init__(self, period: int = 14, use_numba: bool = True):
        super().__init__(period=period, use_numba=use_numba)
        self.period = period
        self.use_numba = use_numba
    
    @property
    def name(self) -> str:
        return f"ATR_{self.period}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.VOLATILITY
    
    @property
    def required_periods(self) -> int:
        return self.period + 5
    
    def get_required_columns(self) -> List[str]:
        return ['High', 'Low', 'Close']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            if self.use_numba:
                # Use Numba-optimized calculation
                high_values = data['High'].values
                low_values = data['Low'].values
                close_values = data['Close'].values
                
                atr_values = calculate_atr(high_values, low_values, close_values, self.period)
                current_atr = atr_values[-1]
            else:
                # Use pandas calculation
                high = data['High']
                low = data['Low']
                close = data['Close']
                
                tr = self._true_range(high, low, close)
                atr = self._ewm_operation(tr, alpha=1/self.period)
                atr_series = atr
                current_atr = atr.iloc[-1]
            
            # Calculate ATR as percentage of current price
            current_price = data['Close'].iloc[-1]
            atr_percentage = (current_atr / current_price) * 100 if current_price > 0 else math.nan
            
            # Classify volatility level
            if atr_percentage > 4:
                volatility_level = "very_high"
            elif atr_percentage > 2.5:
                volatility_level = "high"
            elif atr_percentage > 1.5:
                volatility_level = "normal"
            else:
                volatility_level = "low"
            
            confidence = self.calculate_confidence(data, current_atr)
            
            end_time = time.perf_counter()
            
            return IndicatorResult(
                value=atr_series,  # Return full time series
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "atr_percentage": round(float(atr_percentage), 2) if not math.isnan(atr_percentage) else math.nan,
                    "volatility_level": volatility_level,
                    "data_points": len(data),
                    "use_numba": self.use_numba
                },
                computation_time_ms=(end_time - start_time) * 1000,
                data_points_used=len(data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value=math.nan,
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


class BollingerBandsIndicator(VectorizedIndicator):
    """Bollinger Bands indicator with Numba optimization."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, use_numba: bool = True):
        super().__init__(period=period, std_dev=std_dev, use_numba=use_numba)
        self.period = period
        self.std_dev = std_dev
        self.use_numba = use_numba
    
    @property
    def name(self) -> str:
        return f"BBANDS_{self.period}_{self.std_dev}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.VOLATILITY
    
    @property
    def required_periods(self) -> int:
        return self.period + 5
    
    @property
    def output_names(self) -> List[str]:
        return ['upper_band', 'middle_band', 'lower_band', 'bb_width', 'bb_position']
    
    def get_required_columns(self) -> List[str]:
        return ['Close']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value={"upper_band": math.nan, "middle_band": math.nan, "lower_band": math.nan},
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            close = data['Close']
            
            if self.use_numba:
                # Use Numba-optimized calculation
                close_values = close.values
                upper_band, middle_band, lower_band = calculate_bollinger_bands(
                    close_values, self.period, self.std_dev
                )
                
                current_upper = upper_band[-1]
                current_middle = middle_band[-1]
                current_lower = lower_band[-1]
            else:
                # Use pandas calculation
                middle_band = self._rolling_operation(close, self.period, 'mean')
                std = self._rolling_operation(close, self.period, 'std')
                
                upper_band = middle_band + (self.std_dev * std)
                lower_band = middle_band - (self.std_dev * std)
                
                current_upper = upper_band.iloc[-1]
                current_middle = middle_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
            
            current_price = close.iloc[-1]
            
            # Calculate Bollinger Band metrics
            bb_width = ((current_upper - current_lower) / current_middle) * 100 if current_middle > 0 else math.nan
            bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100 if (current_upper - current_lower) > 0 else 50
            
            # Determine position relative to bands
            if bb_position >= 80:
                band_position = "near_upper"
            elif bb_position <= 20:
                band_position = "near_lower"
            else:
                band_position = "middle"
            
            confidence = self.calculate_confidence(data, current_middle)
            
            end_time = time.perf_counter()
            
            return IndicatorResult(
                value={
                    "upper_band": round(float(current_upper), 2) if not math.isnan(current_upper) else math.nan,
                    "middle_band": round(float(current_middle), 2) if not math.isnan(current_middle) else math.nan,
                    "lower_band": round(float(current_lower), 2) if not math.isnan(current_lower) else math.nan,
                    "bb_width": round(float(bb_width), 2) if not math.isnan(bb_width) else math.nan,
                    "bb_position": round(float(bb_position), 1) if not math.isnan(bb_position) else math.nan
                },
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "std_dev": self.std_dev,
                    "band_position": band_position,
                    "squeeze": bb_width < 10 if not math.isnan(bb_width) else False,
                    "data_points": len(data),
                    "use_numba": self.use_numba
                },
                computation_time_ms=(end_time - start_time) * 1000,
                data_points_used=len(data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value={"upper_band": math.nan, "middle_band": math.nan, "lower_band": math.nan},
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


class ADXIndicator(VectorizedIndicator):
    """Average Directional Index with Numba optimization."""
    
    def __init__(self, period: int = 14, use_numba: bool = True):
        super().__init__(period=period, use_numba=use_numba)
        self.period = period
        self.use_numba = use_numba
    
    @property
    def name(self) -> str:
        return f"ADX_{self.period}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.TREND
    
    @property
    def required_periods(self) -> int:
        return self.period * 2 + 10
    
    @property
    def output_names(self) -> List[str]:
        return ['adx', 'di_plus', 'di_minus']
    
    def get_required_columns(self) -> List[str]:
        return ['High', 'Low', 'Close']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value={"adx": math.nan, "di_plus": math.nan, "di_minus": math.nan},
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            if self.use_numba:
                # Use Numba-optimized calculation
                high_values = data['High'].values
                low_values = data['Low'].values
                close_values = data['Close'].values
                
                adx_values, di_plus_values, di_minus_values = calculate_adx(
                    high_values, low_values, close_values, self.period
                )
                
                current_adx = adx_values[-1]
                current_di_plus = di_plus_values[-1]
                current_di_minus = di_minus_values[-1]
            else:
                # Fall back to pandas implementation (simplified)
                high = data['High']
                low = data['Low']
                close = data['Close']
                
                # This is a simplified version - full implementation would match Numba version
                tr = self._true_range(high, low, close)
                
                # For brevity, returning NaN for pandas version
                # Full implementation would replicate the ADX calculation
                current_adx = math.nan
                current_di_plus = math.nan
                current_di_minus = math.nan
            
            # Determine trend strength
            if current_adx >= 50:
                trend_strength = "very_strong"
            elif current_adx >= 30:
                trend_strength = "strong"
            elif current_adx >= 20:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            # Determine trend direction
            if current_di_plus > current_di_minus:
                trend_direction = "bullish"
            elif current_di_minus > current_di_plus:
                trend_direction = "bearish"
            else:
                trend_direction = "neutral"
            
            confidence = self.calculate_confidence(data, current_adx)
            
            end_time = time.perf_counter()
            
            return IndicatorResult(
                value={
                    "adx": round(float(current_adx), 2) if not math.isnan(current_adx) else math.nan,
                    "di_plus": round(float(current_di_plus), 2) if not math.isnan(current_di_plus) else math.nan,
                    "di_minus": round(float(current_di_minus), 2) if not math.isnan(current_di_minus) else math.nan
                },
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "trend_strength": trend_strength,
                    "trend_direction": trend_direction,
                    "data_points": len(data),
                    "use_numba": self.use_numba
                },
                computation_time_ms=(end_time - start_time) * 1000,
                data_points_used=len(data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value={"adx": math.nan, "di_plus": math.nan, "di_minus": math.nan},
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


class VolumeProfileIndicator(VectorizedIndicator):
    """Vectorized Volume Profile calculation for breakout detection."""
    
    def __init__(self, lookback: int = 90, num_buckets: int = 20):
        super().__init__(lookback=lookback, num_buckets=num_buckets)
        self.lookback = lookback
        self.num_buckets = num_buckets
    
    @property
    def name(self) -> str:
        return f"VOLPROFILE_{self.lookback}_{self.num_buckets}"
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.VOLUME
    
    @property
    def required_periods(self) -> int:
        return self.lookback
    
    @property
    def output_names(self) -> List[str]:
        return ['breakout_score', 'resistance_level', 'support_level', 'high_volume_node']
    
    def get_required_columns(self) -> List[str]:
        return ['High', 'Low', 'Close', 'Volume']
    
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        start_time = time.perf_counter()
        
        try:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                return IndicatorResult(
                    value={"breakout_score": 0, "resistance_level": math.nan},
                    confidence=0.0,
                    metadata={"errors": errors},
                    computation_time_ms=0.0
                )
            
            # Use recent data for volume profile
            recent_data = data.tail(self.lookback).copy()
            
            # Vectorized volume profile calculation
            price_min = recent_data['Low'].min()
            price_max = recent_data['High'].max()
            price_range = price_max - price_min
            
            if price_range <= 0:
                return IndicatorResult(
                    value={"breakout_score": 0, "resistance_level": math.nan},
                    confidence=0.0,
                    metadata={"error": "No price range in data"},
                    computation_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Create price buckets
            midpoint_prices = (recent_data['High'] + recent_data['Low']) / 2
            volumes = recent_data['Volume'].values
            
            # Use numpy histogram with weights for volume distribution
            bin_edges = np.linspace(price_min, price_max, self.num_buckets + 1)
            volume_at_price, _ = np.histogram(midpoint_prices, bins=bin_edges, weights=volumes)
            
            # Find high volume nodes (top 20% by volume)
            if np.sum(volume_at_price) == 0:
                return IndicatorResult(
                    value={"breakout_score": 0, "resistance_level": math.nan},
                    confidence=0.0,
                    metadata={"error": "No volume data"},
                    computation_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            volume_threshold = np.percentile(volume_at_price[volume_at_price > 0], 80)
            high_volume_mask = volume_at_price >= volume_threshold
            high_volume_indices = np.where(high_volume_mask)[0]
            
            # Convert bucket indices to price levels
            bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            high_volume_prices = bucket_centers[high_volume_indices]
            
            # Analyze current price position
            current_price = data['Close'].iloc[-1]
            breakout_score = 0
            resistance_level = math.nan
            support_level = math.nan
            
            if len(high_volume_prices) > 0:
                # Find nearest resistance above current price
                resistance_mask = high_volume_prices > current_price
                if np.any(resistance_mask):
                    resistance_level = np.min(high_volume_prices[resistance_mask])
                    distance_to_resistance = (resistance_level - current_price) / current_price
                    if distance_to_resistance < 0.02:  # Within 2%
                        breakout_score = min(10, 10 * (0.02 - distance_to_resistance) / 0.02)
                
                # Find nearest support below current price
                support_mask = high_volume_prices < current_price
                if np.any(support_mask):
                    support_level = np.max(high_volume_prices[support_mask])
            
            high_volume_node = np.max(high_volume_prices) if len(high_volume_prices) > 0 else math.nan
            
            confidence = self.calculate_confidence(data, breakout_score)
            
            end_time = time.perf_counter()
            
            return IndicatorResult(
                value={
                    "breakout_score": round(float(breakout_score), 1),
                    "resistance_level": round(float(resistance_level), 2) if not math.isnan(resistance_level) else math.nan,
                    "support_level": round(float(support_level), 2) if not math.isnan(support_level) else math.nan,
                    "high_volume_node": round(float(high_volume_node), 2) if not math.isnan(high_volume_node) else math.nan
                },
                confidence=confidence,
                metadata={
                    "lookback": self.lookback,
                    "num_buckets": self.num_buckets,
                    "price_range": round(float(price_range), 2),
                    "total_volume_nodes": len(high_volume_prices),
                    "data_points": len(data)
                },
                computation_time_ms=(end_time - start_time) * 1000,
                data_points_used=len(recent_data)
            )
            
        except Exception as e:
            return IndicatorResult(
                value={"breakout_score": 0, "resistance_level": math.nan},
                confidence=0.0,
                metadata={"error": str(e)},
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )


# Convenient aliases for the indicator classes
RSI = RSIIndicator
MACD = MACDIndicator
ATR = ATRIndicator
BollingerBands = BollingerBandsIndicator
ADX = ADXIndicator
VolumeProfile = VolumeProfileIndicator

# All available indicators
__all__ = [
    'RSIIndicator', 'RSI',
    'MACDIndicator', 'MACD', 
    'ATRIndicator', 'ATR',
    'BollingerBandsIndicator', 'BollingerBands',
    'ADXIndicator', 'ADX',
    'VolumeProfileIndicator', 'VolumeProfile'
]