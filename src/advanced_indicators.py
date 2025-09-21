"""
Advanced Technical Indicators Engine
Implements comprehensive technical analysis indicators for the upgraded stock screening system
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Union
import warnings
import logging
from functools import lru_cache, wraps
import time

# Import our logging and monitoring infrastructure
try:
    from .logging_config import get_logger, timed_operation, operation_context
    from .stock_analysis_monitor import monitor
    from .robust_data_fetcher import fetch_stock_data
    from .data.validation import (
        DataContract, DataValidator, safe_float, safe_bool,
        is_valid_numeric, replace_invalid_with_nan
    )
    # Import new modular interfaces
    from .common.interfaces import IIndicator, IndicatorResult
    from .indicators import IndicatorEngine
except ImportError:
    # Fallback for when running as standalone
    def get_logger(name):
        return logging.getLogger(name)
    def timed_operation(name):
        def decorator(func):
            return func
        return decorator
    def operation_context(name, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    class MockMonitor:
        def record_indicator_computation(self, *args, **kwargs): pass
    monitor = MockMonitor()
    def fetch_stock_data(symbol, **kwargs):
        import yfinance as yf
        return yf.Ticker(symbol).history(**kwargs)

    # Mock data validation imports
    class DataContract:
        IndicatorDict = dict
    class DataValidator:
        pass
    def safe_float(x, default, name):
        return x if not math.isnan(x) else default
    def safe_bool(x, default, name):
        return x if x is not None else default
    def is_valid_numeric(x):
        return not math.isnan(x) if isinstance(x, (int, float)) else False
    def replace_invalid_with_nan(x):
        return x

# Use yfinance for data fetching
try:
    import yfinance as yf
    # Verify yfinance is working
    pass
except ImportError:
    logger = logging.getLogger(__name__)
    logger.critical("yfinance not installed. Run: pip install yfinance")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Store timing info in the instance if available
        if args and hasattr(args[0], '_timing_stats'):
            if not hasattr(args[0], '_timing_stats'):
                args[0]._timing_stats = {}
            args[0]._timing_stats[func.__name__] = execution_time

        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class AdvancedIndicator:
    """
    Comprehensive technical indicators calculator with focus on:
    - Volume anomaly detection (z-score + ratio)
    - Advanced momentum (RSI, MACD with improvements)
    - Trend strength (ADX, MA slopes)
    - Volatility (ATR, normalized)
    - Relative strength vs index
    - Volume profile approximation

    Performance optimized: Cached data fetching, vectorized operations, timing monitoring
    """

    def __init__(self, nifty_symbol: str = "^NSEI"):
        self.nifty_symbol = nifty_symbol
        self.nifty_data = None
        # Cache for weekly data to avoid repeated API calls
        self._weekly_data_cache = {}
        self._cache_timestamp = {}
        self._cache_timeout = 3600  # 1 hour cache timeout
        # Performance monitoring
        self._timing_stats = {}
        self._performance_enabled = True

    def enable_performance_monitoring(self, enabled: bool = True):
        """Enable or disable performance monitoring"""
        self._performance_enabled = enabled
        if not enabled:
            self._timing_stats.clear()

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for all timed operations"""
        return self._timing_stats.copy()

    def clear_performance_stats(self):
        """Clear accumulated performance statistics"""
        self._timing_stats.clear()

    @lru_cache(maxsize=32)
    def get_nifty_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch NIFTY data for relative strength calculations with LRU caching"""
        try:
            if self.nifty_data is None:
                logger.debug(f"Fetching NIFTY data with period: {period}")
                nifty = yf.Ticker(self.nifty_symbol)
                self.nifty_data = nifty.history(period=period, auto_adjust=True)
                logger.debug(f"Fetched {len(self.nifty_data)} NIFTY data points")
            return self.nifty_data
        except Exception as e:
            logger.warning(f"Could not fetch NIFTY data: {e}")
            return None

    def _get_weekly_data_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get weekly data with intelligent caching to avoid repeated API calls
        Cache expires after 1 hour to balance performance vs data freshness
        """
        current_time = time.time()

        # Check if we have valid cached data
        if (symbol in self._weekly_data_cache and
            symbol in self._cache_timestamp and
            current_time - self._cache_timestamp[symbol] < self._cache_timeout):
            return self._weekly_data_cache[symbol]

        # Fetch fresh data
        try:
            logger.debug(f"Fetching weekly data for {symbol} (cache miss/expired)")
            ticker = yf.Ticker(symbol)
            weekly_data = ticker.history(period="6mo", interval="1wk", auto_adjust=True)

            if weekly_data is not None and not weekly_data.empty and len(weekly_data) >= 15:
                # Cache the data
                self._weekly_data_cache[symbol] = weekly_data
                self._cache_timestamp[symbol] = current_time
                logger.debug(f"Cached {len(weekly_data)} weekly data points for {symbol}")
                return weekly_data
            else:
                logger.warning(f"Insufficient weekly data for {symbol}: {len(weekly_data) if weekly_data is not None else 0} rows")
                return None

        except Exception as e:
            logger.warning(f"Error fetching weekly data for {symbol}: {e}")
            return None

    @timing_decorator
    def compute_volume_signals(self, data: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """
        Compute advanced volume signals with standardized validation:
        - Volume ratio (current vs 20-day SMA)
        - Volume z-score (statistical significance)
        - Volume trend (5-day slope)
        - Volume breakout signals

        Performance optimized: Consolidated rolling calculations
        Data contract: Returns float for numeric values, bool for signals, math.nan for missing
        """
        if len(data) < 25:
            return {
                'vol_ratio': math.nan,
                'vol_z': math.nan,
                'vol_trend': math.nan,
                'volume_increasing': False,
                'volume_breakout': False
            }

        try:
            volume = data['Volume']

            # OPTIMIZED: Compute all rolling statistics in one pass
            vol_rolling_stats = volume.rolling(20).agg(['mean', 'std'])
            vol_mean20 = safe_float(vol_rolling_stats['mean'].iloc[-1], math.nan, 'vol_mean20')
            vol_std20 = safe_float(vol_rolling_stats['std'].iloc[-1], math.nan, 'vol_std20')
            current_volume = safe_float(volume.iloc[-1], math.nan, 'current_volume')

            # Volume ratio (traditional)
            vol_ratio = (current_volume / vol_mean20) if is_valid_numeric(vol_mean20) and vol_mean20 > 0 else math.nan

            # Volume z-score (statistical significance)
            vol_z = ((current_volume - vol_mean20) / vol_std20) if (is_valid_numeric(vol_std20) and vol_std20 > 0 and is_valid_numeric(vol_mean20)) else math.nan

            # Volume trend (5-day slope) - vectorized approach
            if len(volume) >= 5:
                try:
                    recent_volume = volume.tail(5).values
                    # Replace any NaN values with forward fill
                    recent_volume = replace_invalid_with_nan(recent_volume)
                    if not np.all(np.isnan(recent_volume)):
                        x = np.arange(len(recent_volume))
                        vol_trend_raw = np.polyfit(x, recent_volume, 1)[0]
                        vol_trend = (vol_trend_raw / vol_mean20) if is_valid_numeric(vol_mean20) and vol_mean20 > 0 else 0
                    else:
                        vol_trend = math.nan
                except Exception as e:
                    logger.warning(f"Error computing volume trend: {e}")
                    vol_trend = math.nan
            else:
                vol_trend = math.nan

            # Volume signals with safe boolean conversion
            vol_5_avg = safe_float(volume.rolling(5).mean().iloc[-1], math.nan, 'vol_5_avg')
            volume_increasing = safe_bool(
                vol_5_avg > vol_mean20 if is_valid_numeric(vol_5_avg) and is_valid_numeric(vol_mean20) else None,
                False, 'volume_increasing'
            )

            # High volume breakout (1.5x above average)
            volume_breakout = safe_bool(
                vol_ratio > 1.5 if is_valid_numeric(vol_ratio) else None,
                False, 'volume_breakout'
            )

            return {
                'vol_ratio': safe_float(vol_ratio, math.nan, 'vol_ratio'),
                'vol_z': safe_float(vol_z, math.nan, 'vol_z'),
                'vol_trend': safe_float(vol_trend, math.nan, 'vol_trend'),
                'volume_increasing': volume_increasing,
                'volume_breakout': volume_breakout
            }

        except Exception as e:
            logger.error(f"Error in compute_volume_signals: {e}", exc_info=True)
            return {
                'vol_ratio': math.nan,
                'vol_z': math.nan,
                'vol_trend': math.nan,
                'volume_increasing': False,
                'volume_breakout': False
            }

    @timing_decorator
    def compute_momentum_signals(self, data: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """
        Compute advanced momentum signals with standardized validation:
        - RSI (14-period Wilder's)
        - MACD with histogram momentum
        - MACD normalized strength
        - Momentum signal flags

        Data contract: Returns float for numeric values, bool for signals, math.nan for missing
        """
        if len(data) < 35:
            return {
                'rsi': math.nan,
                'macd': math.nan,
                'macd_signal': math.nan,
                'macd_hist': math.nan,
                'macd_strength': math.nan,
                'rsi_bullish': False,
                'macd_bullish': False
            }

        try:
            close = data['Close']

            # RSI (Wilder's method) with safe calculations
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)

            # Use Wilder's smoothing (alpha = 1/14)
            alpha = 1/14
            up_ewm = up.ewm(alpha=alpha, adjust=False).mean()
            down_ewm = down.ewm(alpha=alpha, adjust=False).mean()

            # Safe RSI calculation
            rs = up_ewm / down_ewm
            rsi = 100 - (100 / (1 + rs))
            current_rsi = safe_float(rsi.iloc[-1], math.nan, 'current_rsi')

            # MACD (12, 26, 9) with safe calculations
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            current_macd = safe_float(macd_line.iloc[-1], math.nan, 'current_macd')
            current_signal = safe_float(signal_line.iloc[-1], math.nan, 'current_signal')
            current_hist = safe_float(histogram.iloc[-1], math.nan, 'current_hist')

            # MACD normalized strength with safe division
            if is_valid_numeric(current_signal) and abs(current_signal) > 1e-8:
                macd_strength = (current_macd - current_signal) / abs(current_signal) if is_valid_numeric(current_macd) else math.nan
            else:
                macd_strength = 0.0

            # Signal flags with safe boolean conversion
            rsi_bullish = safe_bool(
                30 <= current_rsi <= 70 if is_valid_numeric(current_rsi) else None,
                False, 'rsi_bullish'
            )

            macd_bullish = safe_bool(
                current_macd > current_signal if is_valid_numeric(current_macd) and is_valid_numeric(current_signal) else None,
                False, 'macd_bullish'
            )

            return {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_hist': current_hist,
                'macd_strength': safe_float(macd_strength, math.nan, 'macd_strength'),
                'rsi_bullish': rsi_bullish,
                'macd_bullish': macd_bullish
            }

        except Exception as e:
            logger.error(f"Error in compute_momentum_signals: {e}", exc_info=True)
            return {
                'rsi': math.nan,
                'macd': math.nan,
                'macd_signal': math.nan,
                'macd_hist': math.nan,
                'macd_strength': math.nan,
                'rsi_bullish': False,
                'macd_bullish': False
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

        Performance optimized: Vectorized True Range calculation
        """
        if len(data) < 20:
            return {'atr': np.nan, 'atr_pct': np.nan, 'atr_trend': np.nan}

        high = data['High']
        low = data['Low']
        close = data['Close']

        # OPTIMIZED: Vectorized True Range calculation
        prev_close = close.shift(1)
        tr_components = np.column_stack([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])

        # Use numpy to find max across components (faster than pandas concat)
        tr = pd.Series(np.nanmax(tr_components, axis=1), index=high.index)

        # ATR (14-period Wilder's smoothing)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        current_atr = atr.iloc[-1]

        # ATR as percentage of price
        atr_pct = (current_atr / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else np.nan

        # ATR trend (increasing vs decreasing volatility) - vectorized
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

    @timing_decorator
    def compute_volume_profile_proxy(self, data: pd.DataFrame, lookback: int = 90) -> Dict[str, float]:
        """
        Compute volume profile approximation using vectorized operations:
        - Identify high-volume price nodes
        - Check if current price is breaking above/below key levels

        Performance optimized: Replaced iterrows() with numpy.histogram weighted approach
        """
        if len(data) < lookback:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}

        # Use recent data for volume profile
        recent_data = data.tail(lookback).copy()

        # Create price buckets (simplified volume profile)
        price_min = recent_data['Low'].min()
        price_max = recent_data['High'].max()
        price_range = price_max - price_min
        num_buckets = min(20, len(recent_data) // 5)  # Adaptive bucket count

        if price_range <= 0 or num_buckets < 3:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}

        # VECTORIZED APPROACH: Use numpy operations instead of iterrows()
        # Create midpoint prices for each bar (representative price)
        midpoint_prices = (recent_data['High'] + recent_data['Low']) / 2
        volumes = recent_data['Volume'].values

        # Use numpy.histogram with weights for volume distribution
        bin_edges = np.linspace(price_min, price_max, num_buckets + 1)
        volume_at_price, _ = np.histogram(midpoint_prices, bins=bin_edges, weights=volumes)

        # For more accurate distribution, also consider price range impact
        # Distribute volume based on price range within each day
        low_vals = recent_data['Low'].values
        high_vals = recent_data['High'].values

        # Vectorized bucket assignment for low and high prices
        low_buckets = np.digitize(low_vals, bin_edges) - 1
        high_buckets = np.digitize(high_vals, bin_edges) - 1

        # Clip bucket indices to valid range
        low_buckets = np.clip(low_buckets, 0, num_buckets - 1)
        high_buckets = np.clip(high_buckets, 0, num_buckets - 1)

        # Enhanced volume distribution accounting for price spread
        volume_distribution = np.zeros(num_buckets)
        for i in range(len(volumes)):
            bucket_span = max(1, high_buckets[i] - low_buckets[i] + 1)
            volume_per_bucket = volumes[i] / bucket_span

            # Distribute volume across touched buckets
            for bucket in range(low_buckets[i], min(high_buckets[i] + 1, num_buckets)):
                volume_distribution[bucket] += volume_per_bucket

        # Combine both approaches (weighted average)
        final_volume_at_price = 0.7 * volume_distribution + 0.3 * volume_at_price

        if np.sum(final_volume_at_price) == 0:
            return {'vp_breakout_score': 0, 'vp_resistance_level': np.nan}

        # Find high volume nodes (top 20% by volume) - vectorized
        volume_threshold = np.percentile(final_volume_at_price[final_volume_at_price > 0], 80)
        high_volume_mask = final_volume_at_price >= volume_threshold
        high_volume_bucket_indices = np.where(high_volume_mask)[0]

        # Convert buckets to price levels - vectorized
        bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        high_volume_prices = bucket_centers[high_volume_bucket_indices]

        # Check current price vs high volume levels - vectorized
        current_price = data['Close'].iloc[-1]
        breakout_score = 0
        resistance_level = np.nan

        if len(high_volume_prices) > 0:
            # Find nearest resistance level above current price - vectorized
            resistance_mask = high_volume_prices > current_price
            if np.any(resistance_mask):
                resistance_level = np.min(high_volume_prices[resistance_mask])
                # Score based on how close we are to breaking resistance
                distance_pct = (resistance_level - current_price) / current_price * 100
                if distance_pct < 2:  # Within 2% of resistance
                    breakout_score = 10
                elif distance_pct < 5:  # Within 5% of resistance
                    breakout_score = 5

            # Check if we've broken above a recent high-volume level - vectorized
            broken_mask = current_price > high_volume_prices * 1.02  # 2% above
            if np.any(broken_mask):
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

        Performance optimized: Uses cached weekly data to avoid repeated API calls
        """
        weekly_data = self._get_weekly_data_cached(symbol)

        if weekly_data is None:
            return {'weekly_rsi_trend': 0, 'weekly_macd_bullish': False, 'weekly_vol_trend': 0}

        try:
            # Weekly RSI - vectorized calculations
            close = weekly_data['Close']
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)

            up_ewm = up.ewm(alpha=1/14, adjust=False).mean()
            down_ewm = down.ewm(alpha=1/14, adjust=False).mean()

            rs = up_ewm / down_ewm
            rsi = 100 - (100 / (1 + rs))

            # RSI trend (positive if increasing over last 3 weeks)
            rsi_trend = 1 if len(rsi) >= 3 and rsi.iloc[-1] > rsi.iloc[-3] else 0

            # Weekly MACD - vectorized calculations
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            macd_bullish = bool(len(macd_line) > 0 and macd_line.iloc[-1] > signal_line.iloc[-1])

            # Weekly volume trend - vectorized calculation
            volume = weekly_data['Volume']
            vol_trend = 0
            if len(volume) >= 4:
                recent_avg = volume.tail(2).mean()
                previous_avg = volume.iloc[-4:-2].mean()
                vol_trend = 1 if recent_avg > previous_avg else 0

            return {
                'weekly_rsi_trend': rsi_trend,
                'weekly_macd_bullish': macd_bullish,
                'weekly_vol_trend': vol_trend
            }

        except Exception as e:
            logger.warning(f"Error computing weekly indicators for {symbol}: {e}")
            return {'weekly_rsi_trend': 0, 'weekly_macd_bullish': False, 'weekly_vol_trend': 0}

    @timed_operation("compute_all_indicators")
    def compute_all_indicators(self, symbol: str, period: str = "6mo") -> Optional[DataContract.IndicatorDict]:
        """
        Compute all indicators for a given symbol with standardized data validation.

        Returns comprehensive dictionary of all technical indicators according to DataContract.

        Data Contract:
        - Returns None only for critical failures (no data, computation errors)
        - All numeric indicators return float type with math.nan for missing values
        - Boolean indicators return actual bool type
        - String indicators return str type with empty string for missing
        - All values are validated and sanitized before return

        Performance optimized: Includes timing monitoring for batch operations
        Corporate action aware: Uses auto_adjust=True for proper split/dividend handling
        """

        with operation_context("compute_indicators", symbol=symbol, period=period):
            start_time = time.time()

            try:
                # Use our robust data fetcher instead of direct yfinance
                logger.debug(f"Fetching data for {symbol}", extra={'symbol': symbol, 'period': period})
                data = fetch_stock_data(symbol, period=period)

                if data is None or len(data) < 50 or data.empty:
                    error_msg = f"Insufficient data for {symbol}"
                    logger.warning(error_msg, extra={
                        'symbol': symbol,
                        'data_length': len(data) if data is not None else 0,
                        'period': period
                    })

                    # Record monitoring metrics
                    duration = time.time() - start_time
                    monitor.record_indicator_computation(symbol, duration, False, "insufficient_data")
                    return None

                logger.debug(f"Computing indicators for {symbol}", extra={
                    'symbol': symbol,
                    'data_points': len(data),
                    'date_range': f"{data.index[0].date()} to {data.index[-1].date()}"
                })

                # Compute all indicator groups with individual timing
                indicator_start = time.time()

                volume_signals = self.compute_volume_signals(data)
                momentum_signals = self.compute_momentum_signals(data)
                trend_signals = self.compute_trend_signals(data)
                volatility_signals = self.compute_volatility_signals(data)
                relative_strength = self.compute_relative_strength(data, symbol)
                volume_profile = self.compute_volume_profile_proxy(data)
                weekly_confirm = self.compute_weekly_confirmation(symbol)

                indicator_duration = time.time() - indicator_start

                # Build raw indicators dictionary with safe value handling
                try:
                    current_price = safe_float(data['Close'].iloc[-1], math.nan, 'current_price')
                    prev_price = safe_float(data['Close'].iloc[-2], math.nan, 'prev_price')
                    price_change_pct = ((current_price - prev_price) / prev_price * 100) if is_valid_numeric(prev_price) and prev_price != 0 else math.nan

                    raw_indicators = {
                        'symbol': str(symbol),
                        'current_price': current_price,
                        'price_change_pct': safe_float(price_change_pct, math.nan, 'price_change_pct'),
                        **volume_signals,
                        **momentum_signals,
                        **trend_signals,
                        **volatility_signals,
                        **relative_strength,
                        **volume_profile,
                        **weekly_confirm
                    }
                except Exception as e:
                    logger.error(f"Error building indicators dictionary for {symbol}: {e}",
                               extra={'symbol': symbol}, exc_info=True)
                    monitor.record_indicator_computation(symbol, time.time() - start_time, False, "build_error")
                    return None

                # Validate the complete indicators dictionary using our data contract
                validator = DataValidator()
                validated_indicators = validator.validate_indicators_dict(raw_indicators, symbol)

                if validated_indicators is None:
                    logger.error(f"Critical validation failure for {symbol} indicators")
                    monitor.record_indicator_computation(symbol, time.time() - start_time, False, "validation_failure")
                    return None

                # Log validation results
                validator.log_validation_results(symbol)

                # Log successful completion
                total_duration = time.time() - start_time
                logger.info(f"Successfully computed indicators for {symbol}", extra={
                    'symbol': symbol,
                    'indicator_count': len(validated_indicators),
                    'data_points': len(data),
                    'total_duration': total_duration,
                    'indicator_computation_time': indicator_duration,
                    'validation_warnings': len([r for r in validator.validation_results if r.severity.value == 'warning']),
                    'validation_errors': len([r for r in validator.validation_results if r.severity.value in ['error', 'critical']])
                })

                # Record monitoring metrics
                monitor.record_indicator_computation(symbol, total_duration, True)

                return validated_indicators

            except Exception as e:
                total_duration = time.time() - start_time
                error_type = type(e).__name__

                logger.error(f"Error computing indicators for {symbol}: {e}", extra={
                    'symbol': symbol,
                    'error_type': error_type,
                    'duration': total_duration,
                    'period': period
                }, exc_info=True)

                # Record monitoring metrics
                monitor.record_indicator_computation(symbol, total_duration, False, error_type)

                return None

# Example usage and testing
if __name__ == "__main__":
    # Test the indicators
    from .logging_config import setup_logging

    setup_logging(level="INFO", console_output=True)
    logger = logging.getLogger(__name__)

    indicators_engine = AdvancedIndicator()

    # Test with a sample stock
    test_symbol = "RELIANCE.NS"
    logger.info(f"Testing indicators for {test_symbol}", extra={'symbol': test_symbol})

    result = indicators_engine.compute_all_indicators(test_symbol)

    if result:
        logger.info(f"Successfully computed indicators for {test_symbol}",
                   extra={'symbol': test_symbol, 'indicator_count': len(result)})
        for key, value in result.items():
            logger.debug(f"Indicator {key}: {value}", extra={'symbol': test_symbol, 'indicator': key, 'value': value})
    else:
        logger.error("Failed to compute indicators", extra={'symbol': test_symbol})