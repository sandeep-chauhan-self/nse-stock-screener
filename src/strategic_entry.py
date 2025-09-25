# strategic_entry.py
# Deterministic strategic entry logic module

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Any
import logging

from .strategic_entry_constants import *
from .core import DataValidator

logger = logging.getLogger(__name__)

def calculate_strategic_entry(symbol: str,
                              current_price: float,
                              indicators: dict,
                              prices: Sequence[float],
                              volumes: Sequence[int],
                              metadata: dict,
                              signal: str,
                              config: dict) -> dict:
    """
    Calculate strategic entry using multiple fallback strategies.

    Returns a dict with entry details:
    {
      "entry_value": float,
      "entry_method": str,
      "order_type": str,
      "clamp_reason": Optional[str],
      "messages": List[str],
      "debug": dict
    }
    """
    messages = []
    debug = {}

    # Validate inputs
    if not isinstance(current_price, (int, float)) or current_price <= 0 or np.isnan(current_price):
        return _create_unavailable_result("Invalid current price", debug)

    if not prices or len(prices) < 20:
        return _create_unavailable_result("Insufficient price history", debug)

    # Extract required indicators
    atr = indicators.get('atr', 0)
    rsi = indicators.get('rsi', 50)
    volume_last = volumes[-1] if volumes else 0

    # Determine market cap proxy
    avg_volume = metadata.get('avg_volume', 0)
    is_large_cap = avg_volume >= LARGE_CAP_VOLUME_THRESHOLD

    debug.update({
        'is_large_cap': is_large_cap,
        'avg_volume': avg_volume,
        'atr': atr,
        'rsi': rsi,
        'current_price': current_price
    })

    # Strategy 1: Breakout entry
    breakout_result = _calculate_breakout_entry(
        prices, volumes, current_price, atr, volume_last, debug
    )
    if breakout_result['success']:
        entry_value = breakout_result['entry']
        result = {
            'entry_value': entry_value,
            'entry_method': 'BREAKOUT',
            'order_type': 'MARKET',
            'clamp_reason': None,
            'messages': messages + breakout_result['messages'],
            'debug': debug
        }
        return _apply_clamping(result, current_price, atr, is_large_cap, signal)

    # Strategy 2: Support/Pullback entry
    support_result = _calculate_support_pullback_entry(
        prices, current_price, atr, rsi, volume_last, debug
    )
    if support_result['success']:
        entry_value = support_result['entry']
        result = {
            'entry_value': entry_value,
            'entry_method': 'SUPPORT_PULLBACK',
            'order_type': 'LIMIT',
            'clamp_reason': None,
            'messages': messages + support_result['messages'],
            'debug': debug
        }
        return _apply_clamping(result, current_price, atr, is_large_cap, signal)

    # Strategy 3: ATR-based fallback
    atr_result = _calculate_atr_fallback_entry(
        current_price, atr, is_large_cap, debug
    )
    if atr_result['success']:
        entry_value = atr_result['entry']
        result = {
            'entry_value': entry_value,
            'entry_method': 'ATR_FALLBACK',
            'order_type': 'LIMIT',
            'clamp_reason': None,
            'messages': messages + atr_result['messages'],
            'debug': debug
        }
        return _apply_clamping(result, current_price, atr, is_large_cap, signal)

    # Strategy 4: Emergency current price fallback
    messages.append("All strategic entry methods failed, using current price as emergency fallback")
    result = {
        'entry_value': current_price,
        'entry_method': 'CURRENT_PRICE',
        'order_type': 'MARKET',
        'clamp_reason': None,
        'messages': messages,
        'debug': debug
    }

    logger.warning(f"{symbol}: Emergency fallback to current price. Debug: {debug}")
    return result

def _calculate_breakout_entry(prices: Sequence[float], volumes: Sequence[int],
                             current_price: float, atr: float, volume_last: int,
                             debug: dict) -> dict:
    """Calculate breakout entry point."""
    try:
        if len(prices) < ROLLING_HIGH_DAYS + 5:
            return {'success': False, 'entry': None, 'messages': ['Insufficient data for breakout']}

        # Calculate rolling high
        rolling_high = max(prices[-ROLLING_HIGH_DAYS:])

        # Volume confirmation
        if len(volumes) >= VOL_WINDOW:
            avg_volume = np.mean(volumes[-VOL_WINDOW:])
            volume_threshold = VOLUME_MULTIPLIER * avg_volume
        else:
            volume_threshold = 0  # Skip volume check if insufficient data

        debug['breakout'] = {
            'rolling_high': rolling_high,
            'volume_last': volume_last,
            'volume_threshold': volume_threshold
        }

        # Breakout conditions
        price_breakout = current_price > rolling_high
        volume_breakout = volume_last > volume_threshold if volume_threshold > 0 else True

        if price_breakout and volume_breakout:
            # Add small buffer above breakout level
            buffer = min(0.1 * atr, 0.01 * current_price)  # Min tick size proxy
            entry = rolling_high + buffer

            return {
                'success': True,
                'entry': entry,
                'messages': [f'Breakout entry: price {current_price:.2f} > high {rolling_high:.2f}, volume confirmed']
            }

        return {
            'success': False,
            'entry': None,
            'messages': [f'Breakout conditions not met: price_breakout={price_breakout}, volume_breakout={volume_breakout}']
        }

    except Exception as e:
        return {'success': False, 'entry': None, 'messages': [f'Breakout calculation error: {str(e)}']}

def _calculate_support_pullback_entry(prices: Sequence[float], current_price: float,
                                     atr: float, rsi: float, volume_last: int,
                                     debug: dict) -> dict:
    """Calculate support/pullback entry point."""
    try:
        if len(prices) < SUPPORT_LOOKBACK + 5:
            return {'success': False, 'entry': None, 'messages': ['Insufficient data for support analysis']}

        # Find recent swing low (simplified)
        lookback_prices = np.array(prices[-SUPPORT_LOOKBACK:])
        min_idx = np.argmin(lookback_prices)
        swing_low = lookback_prices[min_idx]

        # Check if current price is near support (within 2% of swing low)
        support_zone = swing_low * 1.02
        near_support = current_price <= support_zone

        # Momentum check
        rsi_ok = RSI_PULLBACK_MIN <= rsi <= RSI_PULLBACK_MAX

        debug['support'] = {
            'swing_low': swing_low,
            'support_zone': support_zone,
            'near_support': near_support,
            'rsi_ok': rsi_ok
        }

        if near_support and rsi_ok:
            # Add buffer above support
            buffer = 0.2 * atr
            entry = swing_low + buffer

            return {
                'success': True,
                'entry': entry,
                'messages': [f'Support pullback entry: near {swing_low:.2f} with RSI {rsi:.1f}']
            }

        return {
            'success': False,
            'entry': None,
            'messages': [f'Support conditions not met: near_support={near_support}, rsi_ok={rsi_ok}']
        }

    except Exception as e:
        return {'success': False, 'entry': None, 'messages': [f'Support calculation error: {str(e)}']}

def _calculate_atr_fallback_entry(current_price: float, atr: float,
                                 is_large_cap: bool, debug: dict) -> dict:
    """Calculate ATR-based fallback entry."""
    try:
        if atr <= 0:
            return {'success': False, 'entry': None, 'messages': ['Invalid ATR for fallback']}

        # ATR multiplier based on market cap
        k = ATR_K_LARGE if is_large_cap else ATR_K_SMALL

        # Calculate pullback entry
        entry = current_price - k * atr

        debug['atr_fallback'] = {
            'k': k,
            'atr': atr,
            'entry': entry
        }

        if entry > 0 and entry < current_price:
            return {
                'success': True,
                'entry': entry,
                'messages': [f'ATR fallback entry: {current_price:.2f} - {k:.1f}*{atr:.2f} = {entry:.2f}']
            }

        return {
            'success': False,
            'entry': None,
            'messages': [f'ATR fallback invalid: entry {entry:.2f} not reasonable']
        }

    except Exception as e:
        return {'success': False, 'entry': None, 'messages': [f'ATR fallback error: {str(e)}']}

def _apply_clamping(result: dict, current_price: float, atr: float,
                   is_large_cap: bool, signal: str) -> dict:
    """Apply clamping logic to entry value."""
    entry_value = result['entry_value']

    # Determine clamp parameters
    pct_limit = MAX_PCT_LARGE if is_large_cap else MAX_PCT_SMALL
    n_atr = N_ATR_LARGE if is_large_cap else N_ATR_SMALL

    # Calculate bounds - CRITICAL FIX: For BUY signals, entry should NOT be above current price
    if signal.upper() == 'BUY':
        lower_bound = max(
            current_price * (1 - pct_limit),
            current_price - n_atr * atr
        )
        upper_bound = current_price  # FIXED: BUY entries should not exceed current price
    else:  # SELL
        upper_bound = min(
            current_price * (1 + pct_limit),
            current_price + n_atr * atr
        )
        lower_bound = current_price * (1 - pct_limit)

    # Apply clamping
    original_entry = entry_value
    if entry_value < lower_bound:
        entry_value = lower_bound
        clamp_reason = f'Clamped up from {original_entry:.2f} to lower bound {lower_bound:.2f}'
    elif entry_value > upper_bound:
        entry_value = upper_bound
        clamp_reason = f'Clamped down from {original_entry:.2f} to upper bound {upper_bound:.2f}'
    else:
        clamp_reason = None

    result['entry_value'] = entry_value
    result['clamp_reason'] = clamp_reason

    if clamp_reason:
        result['messages'].append(clamp_reason)

    result['debug']['clamping'] = {
        'original_entry': original_entry,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'clamped': clamp_reason is not None
    }

    return result

def _create_unavailable_result(reason: str, debug: dict) -> dict:
    """Create unavailable result dict."""
    return {
        'entry_value': 0.0,
        'entry_method': 'UNAVAILABLE',
        'order_type': 'MARKET',
        'clamp_reason': None,
        'messages': [reason],
        'debug': debug
    }