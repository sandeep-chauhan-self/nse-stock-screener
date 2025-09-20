"""
Volume Threshold Configuration for Composite Scoring

This module provides configurable volume threshold settings that support
both regime-adaptive multipliers and percentile-based thresholds.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

from .enums import MarketRegime


@dataclass
class VolumeThresholdConfig:
    """
    Configuration for volume threshold calculations.
    
    Supports both multiplier-based and percentile-based approaches
    for determining extreme volume conditions.
    """
    
    # Regime-specific base thresholds (traditional approach)
    regime_base_thresholds: Dict[MarketRegime, float] = None
    
    # Regime-specific extreme multipliers (fixes the 1.67 bug)
    regime_extreme_multipliers: Dict[MarketRegime, float] = None
    
    # Percentile-based thresholds (recommended approach)
    use_percentile_thresholds: bool = False
    extreme_percentile: float = 99.0  # 99th percentile for extreme detection
    high_percentile: float = 95.0     # 95th percentile for high detection
    
    # Historical lookback for percentile calculation
    percentile_lookback_days: int = 252  # 1 year of trading days
    
    # Minimum samples required for percentile calculation
    min_samples_for_percentiles: int = 50
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.regime_base_thresholds is None:
            self.regime_base_thresholds = {
                MarketRegime.BULLISH: 2.5,
                MarketRegime.SIDEWAYS: 3.0,
                MarketRegime.BEARISH: 4.0,
                MarketRegime.HIGH_VOLATILITY: 5.0
            }
        
        if self.regime_extreme_multipliers is None:
            # Corrected multipliers that actually implement the intended logic
            self.regime_extreme_multipliers = {
                MarketRegime.BULLISH: 2.0,      # 2.5 * 2.0 = 5.0x (easier in bull market)
                MarketRegime.SIDEWAYS: 2.0,     # 3.0 * 2.0 = 6.0x (neutral baseline)
                MarketRegime.BEARISH: 2.5,      # 4.0 * 2.5 = 10.0x (harder in bear market)
                MarketRegime.HIGH_VOLATILITY: 2.0  # 5.0 * 2.0 = 10.0x (high vol needs confirmation)
            }
    
    def get_extreme_threshold(self, regime: MarketRegime) -> float:
        """
        Calculate the extreme volume threshold for the given regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Extreme volume threshold (base_threshold * extreme_multiplier)
        """
        base = self.regime_base_thresholds[regime]
        multiplier = self.regime_extreme_multipliers[regime]
        return base * multiplier
    
    def get_high_threshold(self, regime: MarketRegime) -> float:
        """
        Get the high volume threshold (base threshold) for the given regime.
        
        Args:
            regime: Market regime
            
        Returns:
            High volume threshold
        """
        return self.regime_base_thresholds[regime]
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration settings.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check that all regimes have thresholds
        for regime in MarketRegime:
            if regime not in self.regime_base_thresholds:
                errors.append(f"Missing base threshold for regime: {regime}")
            
            if regime not in self.regime_extreme_multipliers:
                errors.append(f"Missing extreme multiplier for regime: {regime}")
        
        # Check threshold values are positive
        for regime, threshold in self.regime_base_thresholds.items():
            if threshold <= 0:
                errors.append(f"Base threshold for {regime} must be positive: {threshold}")
        
        # Check multipliers are >= 1.0
        for regime, multiplier in self.regime_extreme_multipliers.items():
            if multiplier < 1.0:
                errors.append(f"Extreme multiplier for {regime} must be >= 1.0: {multiplier}")
        
        # Check percentile values
        if not (0 < self.extreme_percentile <= 100):
            errors.append(f"Extreme percentile must be 0 < value <= 100: {self.extreme_percentile}")
        
        if not (0 < self.high_percentile <= 100):
            errors.append(f"High percentile must be 0 < value <= 100: {self.high_percentile}")
        
        if self.high_percentile >= self.extreme_percentile:
            errors.append(f"High percentile ({self.high_percentile}) must be < extreme percentile ({self.extreme_percentile})")
        
        return errors


class VolumeThresholdCalculator:
    """
    Calculator for volume threshold analysis supporting multiple approaches.
    """
    
    def __init__(self, config: VolumeThresholdConfig = None):
        """
        Initialize the calculator with configuration.
        
        Args:
            config: Volume threshold configuration (uses defaults if None)
        """
        self.config = config or VolumeThresholdConfig()
        
        # Validate configuration
        errors = self.config.validate_config()
        if errors:
            raise ValueError(f"Invalid volume threshold configuration: {'; '.join(errors)}")
        
        # Cache for percentile calculations
        self._percentile_cache: Dict[str, Dict[str, float]] = {}
    
    def calculate_thresholds_multiplier_based(self, regime: MarketRegime) -> Tuple[float, float]:
        """
        Calculate volume thresholds using regime-specific multipliers.
        
        Args:
            regime: Current market regime
            
        Returns:
            Tuple of (high_threshold, extreme_threshold)
        """
        high_threshold = self.config.get_high_threshold(regime)
        extreme_threshold = self.config.get_extreme_threshold(regime)
        
        return high_threshold, extreme_threshold
    
    def calculate_thresholds_percentile_based(self, 
                                            historical_vol_ratios: pd.Series,
                                            symbol: str = "UNKNOWN") -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate volume thresholds using historical percentiles.
        
        Args:
            historical_vol_ratios: Historical volume ratio series
            symbol: Symbol name for caching (optional)
            
        Returns:
            Tuple of (high_threshold, extreme_threshold) or (None, None) if insufficient data
        """
        # Check if we have enough data
        valid_ratios = historical_vol_ratios.dropna()
        if len(valid_ratios) < self.config.min_samples_for_percentiles:
            return None, None
        
        # Check cache first
        cache_key = f"{symbol}_{len(valid_ratios)}"
        if cache_key in self._percentile_cache:
            cached = self._percentile_cache[cache_key]
            return cached['high'], cached['extreme']
        
        # Calculate percentiles
        high_threshold = np.percentile(valid_ratios, self.config.high_percentile)
        extreme_threshold = np.percentile(valid_ratios, self.config.extreme_percentile)
        
        # Cache the results
        self._percentile_cache[cache_key] = {
            'high': high_threshold,
            'extreme': extreme_threshold
        }
        
        return high_threshold, extreme_threshold
    
    def get_volume_score_and_level(self, 
                                  vol_ratio: float,
                                  regime: MarketRegime,
                                  historical_vol_ratios: Optional[pd.Series] = None,
                                  symbol: str = "UNKNOWN") -> Tuple[int, str]:
        """
        Calculate volume ratio score and level using the configured approach.
        
        Args:
            vol_ratio: Current volume ratio
            regime: Current market regime
            historical_vol_ratios: Historical volume ratios (for percentile approach)
            symbol: Symbol name (for caching)
            
        Returns:
            Tuple of (score, level_description)
        """
        if np.isnan(vol_ratio):
            return 0, 'NO_DATA'
        
        # Determine which approach to use
        if self.config.use_percentile_thresholds and historical_vol_ratios is not None:
            high_threshold, extreme_threshold = self.calculate_thresholds_percentile_based(
                historical_vol_ratios, symbol
            )
            
            # Fall back to multiplier-based if percentiles unavailable
            if high_threshold is None or extreme_threshold is None:
                high_threshold, extreme_threshold = self.calculate_thresholds_multiplier_based(regime)
                approach_used = "multiplier_fallback"
            else:
                approach_used = "percentile"
        else:
            high_threshold, extreme_threshold = self.calculate_thresholds_multiplier_based(regime)
            approach_used = "multiplier"
        
        # Calculate score based on thresholds
        if vol_ratio >= extreme_threshold:
            score = 10
            level = f'EXTREME_{approach_used.upper()}'
        elif vol_ratio >= high_threshold:
            score = 5
            level = f'HIGH_{approach_used.upper()}'
        else:
            score = 0
            level = 'LOW'
        
        return score, level
    
    def get_threshold_info(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get detailed threshold information for debugging/reporting.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary with threshold details
        """
        base_threshold = self.config.get_high_threshold(regime)
        extreme_threshold = self.config.get_extreme_threshold(regime)
        multiplier = self.config.regime_extreme_multipliers[regime]
        
        return {
            'regime': regime.value,
            'base_threshold': base_threshold,
            'extreme_multiplier': multiplier,
            'extreme_threshold': extreme_threshold,
            'calculation': f"{base_threshold} * {multiplier} = {extreme_threshold}"
        }


# Default configuration instance
DEFAULT_VOLUME_CONFIG = VolumeThresholdConfig()

# Export key components
__all__ = [
    'VolumeThresholdConfig',
    'VolumeThresholdCalculator', 
    'DEFAULT_VOLUME_CONFIG'
]