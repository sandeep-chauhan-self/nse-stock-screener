# utils/marketcap_utils.py
# Market capitalization detection utilities

from typing import Optional
import numpy as np

def detect_market_cap_from_volume(avg_volume: float) -> str:
    """
    Detect market cap category based on average volume (proxy method).

    Args:
        avg_volume: Average daily volume

    Returns:
        'LARGE', 'MID', or 'SMALL'
    """
    if avg_volume >= 2_000_000:  # Large cap threshold
        return 'LARGE'
    elif avg_volume >= 500_000:  # Mid cap threshold
        return 'MID'
    else:
        return 'SMALL'

def is_large_cap(avg_volume: float) -> bool:
    """
    Check if stock is large cap based on volume proxy.

    Args:
        avg_volume: Average daily volume

    Returns:
        True if large cap, False otherwise
    """
    return avg_volume >= 2_000_000

def get_market_cap_multiplier(market_cap_category: str) -> float:
    """
    Get ATR multiplier based on market cap category.

    Args:
        market_cap_category: 'LARGE', 'MID', or 'SMALL'

    Returns:
        ATR multiplier for entry calculations
    """
    multipliers = {
        'LARGE': 0.5,
        'MID': 1.0,
        'SMALL': 1.5
    }
    return multipliers.get(market_cap_category, 1.0)