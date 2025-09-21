"""
Numba-optimized operations for maximum performance.
This module provides JIT-compiled functions for performance-critical
calculations that cannot be efficiently vectorized with pandas/numpy alone.
"""
import math
import numpy as np
from numba import jit, njit
from typing import Tuple[str, ...], Optional
@njit
def calculate_rsi_wilder(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate RSI using Wilder's method with Numba optimization.
    Args:
        prices: Array of closing prices
        period: RSI period
    Returns:
        Array of RSI values
    """
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initialize result array
    rsi = np.full(n, np.nan)

    # Calculate initial averages
    initial_avg_gain = np.mean(gains[:period])
    initial_avg_loss = np.mean(losses[:period])
    if initial_avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = initial_avg_gain / initial_avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Use Wilder's smoothing for subsequent values
    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss
    alpha = 1.0 / period
    for i in range(period + 1, n):

        # Wilder's smoothing: new_avg = (old_avg * (period-1) + new_value) / period
        # Equivalent to: new_avg = old_avg + alpha * (new_value - old_avg)
        avg_gain = avg_gain + alpha * (gains[i-1] - avg_gain)
        avg_loss = avg_loss + alpha * (losses[i-1] - avg_loss)
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi
@njit
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average with Numba optimization.
    Args:
        prices: Array of prices
        period: Moving average period
    Returns:
        Array of SMA values
    """
    n = len(prices)
    if n < period:
        return np.full(n, np.nan)
    sma = np.full(n, np.nan)

    # Calculate first SMA
    sma[period-1] = np.mean(prices[:period])

    # Use rolling calculation for efficiency
    for i in range(period, n):
        sma[i] = sma[i-1] + (prices[i] - prices[i-period]) / period
    return sma
@njit
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average with Numba optimization.
    Args:
        prices: Array of prices
        period: EMA period
    Returns:
        Array of EMA values
    """
    n = len(prices)
    if n == 0:
        return np.array([])
    alpha = 2.0 / (period + 1.0)
    ema = np.full(n, np.nan)

    # Initialize with first price
    ema[0] = prices[0]

    # Calculate EMA iteratively
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema
@njit
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Average True Range with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: ATR period
    Returns:
        Array of ATR values
    """
    n = len(high)
    if n < 2:
        return np.full(n, np.nan)

    # Calculate True Range
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)

    # Calculate ATR using Wilder's smoothing
    atr = np.full(n, np.nan)
    if n >= period + 1:

        # Initial ATR is simple average of first 'period' TR values
        atr[period] = np.mean(tr[1:period+1])

        # Use Wilder's smoothing for subsequent values
        alpha = 1.0 / period
        for i in range(period + 1, n):
            atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])
    return atr
@njit
def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADX, DI+, and DI- with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: ADX period
    Returns:
        Tuple[str, ...] of (ADX, DI+, DI-) arrays
    """
    n = len(high)
    if n < period + 1:
        nan_array = np.full(n, np.nan)
        return nan_array, nan_array, nan_array

    # Calculate True Range and Directional Movement
    tr = np.full(n, np.nan)
    dm_plus = np.full(n, np.nan)
    dm_minus = np.full(n, np.nan)
    for i in range(1, n):

        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)

        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0.0

    # Smooth TR and DM using Wilder's method
    tr_smooth = np.full(n, np.nan)
    dm_plus_smooth = np.full(n, np.nan)
    dm_minus_smooth = np.full(n, np.nan)

    # Initial values (sum of first 'period' values)
    if n >= period + 1:
        tr_smooth[period] = np.sum(tr[1:period+1])
        dm_plus_smooth[period] = np.sum(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.sum(dm_minus[1:period+1])

        # Wilder's smoothing
        for i in range(period + 1, n):
            tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/period + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - dm_plus_smooth[i-1]/period + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - dm_minus_smooth[i-1]/period + dm_minus[i]

    # Calculate DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(period, n):
        if tr_smooth[i] != 0:
            di_plus[i] = 100.0 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100.0 * dm_minus_smooth[i] / tr_smooth[i]

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum != 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    # Smooth DX to get ADX
    if n >= 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2*period])
        alpha = 1.0 / period
        for i in range(2 * period, n):
            adx[i] = adx[i-1] + alpha * (dx[i] - adx[i-1])
    return adx, di_plus, di_minus
@njit
def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator %K and %D with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        k_period: %K period
        d_period: %D smoothing period
    Returns:
        Tuple[str, ...] of (%K, %D) arrays
    """
    n = len(high)
    if n < k_period:
        nan_array = np.full(n, np.nan)
        return nan_array, nan_array
    k_values = np.full(n, np.nan)

    # Calculate %K
    for i in range(k_period - 1, n):
        period_high = np.max(high[i-k_period+1:i+1])
        period_low = np.min(low[i-k_period+1:i+1])
        if period_high != period_low:
            k_values[i] = 100.0 * (close[i] - period_low) / (period_high - period_low)
        else:
            k_values[i] = 50.0
  # Neutral value when no range
    # Calculate %D (SMA of %K)
    d_values = calculate_sma(k_values, d_period)
    return k_values, d_values
@njit
def calculate_bollinger_bands(prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands with Numba optimization.
    Args:
        prices: Array of prices
        period: Moving average period
        std_dev: Standard deviation multiplier
    Returns:
        Tuple[str, ...] of (upper_band, middle_band, lower_band) arrays
    """
    n = len(prices)
    if n < period:
        nan_array = np.full(n, np.nan)
        return nan_array, nan_array, nan_array

    # Calculate moving average
    sma = calculate_sma(prices, period)

    # Calculate rolling standard deviation
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        period_prices = prices[i-period+1:i+1]
        period_mean = sma[i]

        # Calculate variance manually
        variance = 0.0
        for j in range(period):
            diff = period_prices[j] - period_mean
            variance += diff * diff
        variance /= period
        std[i] = math.sqrt(variance)

    # Calculate bands
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band
@njit
def calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Williams %R with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: Williams %R period
    Returns:
        Array of Williams %R values
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    for i in range(period - 1, n):
        period_high = np.max(high[i-period+1:i+1])
        period_low = np.min(low[i-period+1:i+1])
        if period_high != period_low:
            williams_r[i] = -100.0 * (period_high - close[i]) / (period_high - period_low)
        else:
            williams_r[i] = -50.0
  # Neutral value
    return williams_r
@njit
def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Commodity Channel Index with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: CCI period
    Returns:
        Array of CCI values
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)

    # Calculate typical price
    typical_price = (high + low + close) / 3.0

    # Calculate CCI
    cci = np.full(n, np.nan)
    for i in range(period - 1, n):

        # Calculate SMA of typical price
        tp_sma = np.mean(typical_price[i-period+1:i+1])

        # Calculate mean deviation
        mean_deviation = 0.0
        for j in range(i-period+1, i+1):
            mean_deviation += abs(typical_price[j] - tp_sma)
        mean_deviation /= period
        if mean_deviation != 0:
            cci[i] = (typical_price[i] - tp_sma) / (0.015 * mean_deviation)
        else:
            cci[i] = 0.0
    return cci
@njit
def calculate_money_flow_index(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              volume: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Money Flow Index with Numba optimization.
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        volume: Array of volume
        period: MFI period
    Returns:
        Array of MFI values
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)

    # Calculate typical price and money flow
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    mfi = np.full(n, np.nan)
    for i in range(period, n):
        positive_flow = 0.0
        negative_flow = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
  # Skip first element since we need previous typical price
                if typical_price[j] > typical_price[j-1]:
                    positive_flow += money_flow[j]
                elif typical_price[j] < typical_price[j-1]:
                    negative_flow += money_flow[j]
        if negative_flow == 0:
            mfi[i] = 100.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio))
    return mfi
