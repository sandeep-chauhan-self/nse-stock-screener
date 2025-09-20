"""
Technical indicators implementation following the IIndicator interface.

This module provides concrete implementations of technical indicators
with proper validation, error handling, and standardized results.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..common.interfaces import IIndicator, IndicatorResult, IIndicatorEngine
from ..common.config import get_config


class RSIIndicator:
    """Relative Strength Index indicator implementation."""
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period: RSI period (default: 14)
        """
        self.period = period
    
    @property
    def name(self) -> str:
        """Indicator name."""
        return f"RSI_{self.period}"
    
    @property 
    def required_periods(self) -> int:
        """Minimum periods required for calculation."""
        return self.period + 5  # Extra buffer for stability
    
    def compute(self, data: pd.DataFrame, **params) -> IndicatorResult:
        """
        Compute RSI using Wilder's method.
        
        Args:
            data: OHLCV DataFrame
            **params: Additional parameters
            
        Returns:
            IndicatorResult with RSI value
        """
        try:
            if len(data) < self.required_periods:
                return IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"error": "Insufficient data", "required": self.required_periods, "available": len(data)}
                )
            
            close = data['Close']
            delta = close.diff()
            
            # Separate gains and losses
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            # Use Wilder's smoothing (alpha = 1/period)
            alpha = 1 / self.period
            up_ewm = up.ewm(alpha=alpha, adjust=False).mean()
            down_ewm = down.ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate RSI
            rs = up_ewm / down_ewm
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(data, rsi)
            
            return IndicatorResult(
                value=round(float(current_rsi), 2) if not math.isnan(current_rsi) else math.nan,
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "method": "Wilder",
                    "data_points": len(data),
                    "trend": "bullish" if current_rsi > 50 else "bearish"
                }
            )
            
        except Exception as e:
            return IndicatorResult(
                value=math.nan,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, data: pd.DataFrame, rsi: pd.Series) -> float:
        """Calculate confidence score based on data quality."""
        try:
            # Base confidence from data availability
            base_confidence = min(1.0, len(data) / (self.required_periods * 2))
            
            # Reduce confidence if too many missing values
            missing_pct = data['Close'].isnull().sum() / len(data)
            missing_penalty = max(0.0, 1.0 - (missing_pct * 5))
            
            # Reduce confidence for extreme volatility in RSI
            rsi_volatility = rsi.tail(10).std() if len(rsi) > 10 else 50
            volatility_penalty = max(0.5, 1.0 - (rsi_volatility / 100))
            
            return base_confidence * missing_penalty * volatility_penalty
            
        except Exception:
            return 0.5  # Default moderate confidence


class ATRIndicator:
    """Average True Range indicator implementation."""
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            period: ATR period (default: 14)
        """
        self.period = period
    
    @property
    def name(self) -> str:
        """Indicator name."""
        return f"ATR_{self.period}"
    
    @property
    def required_periods(self) -> int:
        """Minimum periods required for calculation."""
        return self.period + 2
    
    def compute(self, data: pd.DataFrame, **params) -> IndicatorResult:
        """
        Compute Average True Range.
        
        Args:
            data: OHLCV DataFrame
            **params: Additional parameters
            
        Returns:
            IndicatorResult with ATR value
        """
        try:
            if len(data) < self.required_periods:
                return IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"error": "Insufficient data", "required": self.required_periods, "available": len(data)}
                )
            
            high = data['High']
            low = data['Low'] 
            close = data['Close']
            
            # Calculate True Range components
            prev_close = close.shift(1)
            tr_components = np.column_stack([
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            ])
            
            # True Range is the maximum of the three components
            tr = pd.Series(np.nanmax(tr_components, axis=1), index=high.index)
            
            # ATR using Wilder's smoothing
            atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
            current_atr = atr.iloc[-1]
            
            # Calculate ATR as percentage of price
            current_price = close.iloc[-1]
            atr_pct = (current_atr / current_price) * 100 if current_price > 0 else math.nan
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, atr)
            
            return IndicatorResult(
                value=round(float(current_atr), 4) if not math.isnan(current_atr) else math.nan,
                confidence=confidence,
                metadata={
                    "period": self.period,
                    "atr_percentage": round(float(atr_pct), 2) if not math.isnan(atr_pct) else math.nan,
                    "data_points": len(data),
                    "volatility_level": "high" if atr_pct > 3 else "normal" if atr_pct > 1 else "low"
                }
            )
            
        except Exception as e:
            return IndicatorResult(
                value=math.nan,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, data: pd.DataFrame, atr: pd.Series) -> float:
        """Calculate confidence score based on data quality."""
        try:
            base_confidence = min(1.0, len(data) / (self.required_periods * 1.5))
            
            # Check for missing OHLC data
            missing_cols = data[['High', 'Low', 'Close']].isnull().sum().sum()
            total_points = len(data) * 3
            missing_penalty = max(0.0, 1.0 - (missing_cols / total_points * 10))
            
            return base_confidence * missing_penalty
            
        except Exception:
            return 0.5


class MACDIndicator:
    """MACD (Moving Average Convergence Divergence) indicator implementation."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @property
    def name(self) -> str:
        """Indicator name."""
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"
    
    @property
    def required_periods(self) -> int:
        """Minimum periods required for calculation."""
        return self.slow + self.signal + 5
    
    def compute(self, data: pd.DataFrame, **params) -> IndicatorResult:
        """
        Compute MACD line, signal line, and histogram.
        
        Args:
            data: OHLCV DataFrame  
            **params: Additional parameters
            
        Returns:
            IndicatorResult with MACD components
        """
        try:
            if len(data) < self.required_periods:
                return IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"error": "Insufficient data", "required": self.required_periods, "available": len(data)}
                )
            
            close = data['Close']
            
            # Calculate MACD components
            exp_fast = close.ewm(span=self.fast, adjust=False).mean()
            exp_slow = close.ewm(span=self.slow, adjust=False).mean()
            macd_line = exp_fast - exp_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Current values
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            # Calculate confidence
            confidence = self._calculate_confidence(data)
            
            # Determine trend
            is_bullish = current_macd > current_signal and current_histogram > 0
            
            return IndicatorResult(
                value=round(float(current_macd), 4) if not math.isnan(current_macd) else math.nan,
                confidence=confidence,
                metadata={
                    "macd_line": round(float(current_macd), 4) if not math.isnan(current_macd) else math.nan,
                    "signal_line": round(float(current_signal), 4) if not math.isnan(current_signal) else math.nan,
                    "histogram": round(float(current_histogram), 4) if not math.isnan(current_histogram) else math.nan,
                    "fast_period": self.fast,
                    "slow_period": self.slow,
                    "signal_period": self.signal,
                    "bullish": is_bullish,
                    "data_points": len(data)
                }
            )
            
        except Exception as e:
            return IndicatorResult(
                value=math.nan,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality."""
        try:
            base_confidence = min(1.0, len(data) / (self.required_periods * 1.5))
            
            missing_pct = data['Close'].isnull().sum() / len(data)
            missing_penalty = max(0.0, 1.0 - (missing_pct * 5))
            
            return base_confidence * missing_penalty
            
        except Exception:
            return 0.5


class IndicatorEngine:
    """Engine for computing multiple technical indicators."""
    
    def __init__(self):
        """Initialize indicator engine with default indicators."""
        self.indicators = {}
        self.config = get_config().config.indicators
        
        # Register default indicators
        self._register_default_indicators()
    
    def _register_default_indicators(self) -> None:
        """Register default technical indicators."""
        self.register_indicator(RSIIndicator(self.config.rsi_period))
        self.register_indicator(ATRIndicator(self.config.atr_period))
        self.register_indicator(MACDIndicator(
            self.config.macd_fast,
            self.config.macd_slow, 
            self.config.macd_signal
        ))
    
    def register_indicator(self, indicator: IIndicator) -> None:
        """
        Register a new indicator.
        
        Args:
            indicator: Indicator to register
        """
        self.indicators[indicator.name] = indicator
    
    def compute_all(self, symbol: str, data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Compute all registered indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            
        Returns:
            Dictionary mapping indicator names to results
        """
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                result = indicator.compute(data)
                results[name] = result
                
            except Exception as e:
                results[name] = IndicatorResult(
                    value=math.nan,
                    confidence=0.0,
                    metadata={"error": f"Computation failed: {e}"}
                )
        
        return results
    
    def get_indicator(self, name: str) -> Optional[IIndicator]:
        """Get indicator by name."""
        return self.indicators.get(name)
    
    def list_indicators(self) -> List[str]:
        """Get list of registered indicator names."""
        return list(self.indicators.keys())


# Export public components
__all__ = [
    'RSIIndicator',
    'ATRIndicator',
    'MACDIndicator',
    'IndicatorEngine',
    'IndicatorResult'
]