"""
Liquidity Validation System for FS.5 Implementation

This module provides comprehensive liquidity validation including:
- Average daily volume calculations and classification
- Liquidity tier assignments with appropriate risk limits
- Volume-based position size constraints
- Market impact estimation and validation
- Real-time liquidity monitoring and alerts

Features:
- Multi-timeframe volume analysis (1D, 5D, 20D averages)
- Liquidity tier classification with dynamic limits
- Market impact estimation before trade execution
- Volume concentration risk monitoring
- Integration with position sizing and risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from collections import defaultdict

from .risk_config import RiskConfig, LiquidityTier


class LiquidityAlert(Enum):
    """Types of liquidity alerts."""
    LOW_VOLUME = "low_volume"
    HIGH_CONCENTRATION = "high_concentration"
    MARKET_IMPACT_WARNING = "market_impact_warning"
    ILLIQUID_POSITION = "illiquid_position"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROP = "volume_drop"


class LiquidityValidationResult(NamedTuple):
    """Result of liquidity validation with detailed analysis."""
    approved: bool
    liquidity_tier: LiquidityTier
    max_position_value: float
    estimated_market_impact: float
    volume_concentration: float
    alerts: List[LiquidityAlert]
    analysis_details: Dict[str, float]
    recommendations: List[str]


@dataclass
class VolumeData:
    """Historical volume data for liquidity analysis."""
    symbol: str
    dates: List[datetime]
    volumes: List[float]  # Daily volumes in rupees
    prices: List[float]   # Daily closing prices
    
    def __post_init__(self):
        """Validate volume data consistency."""
        if not (len(self.dates) == len(self.volumes) == len(self.prices)):
            raise ValueError("Dates, volumes, and prices must have same length")
        
        if len(self.volumes) == 0:
            raise ValueError("Volume data cannot be empty")


@dataclass
class LiquidityMetrics:
    """Calculated liquidity metrics for a stock."""
    symbol: str
    avg_daily_volume_1d: float
    avg_daily_volume_5d: float
    avg_daily_volume_20d: float
    volume_volatility: float
    price_impact_coefficient: float
    liquidity_tier: LiquidityTier
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    last_updated: datetime = field(default_factory=datetime.now)


class LiquidityValidator:
    """
    Comprehensive liquidity validation system implementing FS.5 requirements.
    
    Provides volume analysis, liquidity classification, market impact estimation,
    and position size validation based on liquidity constraints.
    """
    
    def __init__(self, risk_config: RiskConfig):
        """
        Initialize liquidity validator with risk configuration.
        
        Args:
            risk_config: Risk management configuration
        """
        self.config = risk_config
        self.logger = logging.getLogger(__name__)
        
        # Caches for performance
        self._metrics_cache: Dict[str, LiquidityMetrics] = {}
        self._volume_cache: Dict[str, VolumeData] = {}
        
        # Alert tracking
        self._active_alerts: Dict[str, List[LiquidityAlert]] = defaultdict(list)
        self._alert_history: List[Dict] = []
        
        # Performance tracking
        self._validation_count = 0
        self._rejection_count = 0
        
    def validate_position_liquidity(
        self,
        symbol: str,
        position_value: float,
        volume_data: Optional[VolumeData] = None
    ) -> LiquidityValidationResult:
        """
        Validate position size against liquidity constraints.
        
        Args:
            symbol: Stock symbol
            position_value: Proposed position value in rupees
            volume_data: Optional historical volume data
            
        Returns:
            LiquidityValidationResult with approval status and analysis
        """
        self._validation_count += 1
        
        try:
            # Get or calculate liquidity metrics
            metrics = self._get_or_calculate_metrics(symbol, volume_data)
            
            # Perform liquidity validation
            result = self._perform_validation(symbol, position_value, metrics)
            
            # Track rejections
            if not result.approved:
                self._rejection_count += 1
            
            # Update alerts
            self._update_alerts(symbol, result.alerts)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Liquidity validation failed for {symbol}: {e}")
            # Return conservative result on error
            return LiquidityValidationResult(
                approved=False,
                liquidity_tier=LiquidityTier.ILLIQUID,
                max_position_value=0.0,
                estimated_market_impact=1.0,  # High impact
                volume_concentration=1.0,     # High concentration
                alerts=[LiquidityAlert.LOW_VOLUME],
                analysis_details={},
                recommendations=["Unable to validate liquidity - consider smaller position"]
            )
    
    def calculate_liquidity_metrics(self, volume_data: VolumeData) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics from volume data.
        
        Args:
            volume_data: Historical volume and price data
            
        Returns:
            LiquidityMetrics with calculated values
        """
        symbol = volume_data.symbol
        volumes = np.array(volume_data.volumes)
        prices = np.array(volume_data.prices)
        
        # Calculate average volumes for different periods
        avg_1d = float(volumes[-1]) if len(volumes) > 0 else 0.0
        avg_5d = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else avg_1d
        avg_20d = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else avg_5d
        
        # Volume volatility (coefficient of variation)
        volume_volatility = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else float('inf')
        
        # Price impact coefficient (simplified model)
        price_impact_coef = self._estimate_price_impact_coefficient(volumes, prices)
        
        # Determine liquidity tier
        liquidity_tier = self.config.get_liquidity_tier(avg_20d)
        
        # Volume trend analysis
        volume_trend = self._analyze_volume_trend(volumes)
        
        return LiquidityMetrics(
            symbol=symbol,
            avg_daily_volume_1d=avg_1d,
            avg_daily_volume_5d=avg_5d,
            avg_daily_volume_20d=avg_20d,
            volume_volatility=volume_volatility,
            price_impact_coefficient=price_impact_coef,
            liquidity_tier=liquidity_tier,
            volume_trend=volume_trend
        )
    
    def estimate_market_impact(
        self,
        symbol: str,
        position_value: float,
        metrics: Optional[LiquidityMetrics] = None
    ) -> float:
        """
        Estimate market impact of a proposed trade.
        
        Args:
            symbol: Stock symbol
            position_value: Position value in rupees
            metrics: Optional pre-calculated liquidity metrics
            
        Returns:
            Estimated market impact as percentage of position value
        """
        if metrics is None:
            metrics = self._metrics_cache.get(symbol)
            if metrics is None:
                self.logger.warning(f"No metrics available for {symbol}, using conservative estimate")
                return 0.1  # 10% conservative estimate
        
        # Base impact calculation using sqrt model
        # Impact = coefficient * sqrt(position_value / avg_daily_volume)
        avg_volume = metrics.avg_daily_volume_20d
        if avg_volume <= 0:
            return 1.0  # 100% impact if no volume data
        
        volume_ratio = position_value / avg_volume
        base_impact = metrics.price_impact_coefficient * np.sqrt(volume_ratio)
        
        # Adjust based on liquidity tier
        tier_multipliers = {
            LiquidityTier.HIGHLY_LIQUID: 0.5,
            LiquidityTier.MODERATELY_LIQUID: 1.0,
            LiquidityTier.LOW_LIQUID: 2.0,
            LiquidityTier.ILLIQUID: 5.0
        }
        
        tier_multiplier = tier_multipliers.get(metrics.liquidity_tier, 3.0)
        adjusted_impact = base_impact * tier_multiplier
        
        # Cap impact at reasonable levels
        return min(adjusted_impact, 0.5)  # Max 50% impact
    
    def get_maximum_position_value(
        self,
        symbol: str,
        max_impact_pct: float = 0.05,
        metrics: Optional[LiquidityMetrics] = None
    ) -> float:
        """
        Calculate maximum position value for given impact tolerance.
        
        Args:
            symbol: Stock symbol
            max_impact_pct: Maximum acceptable market impact (default 5%)
            metrics: Optional pre-calculated liquidity metrics
            
        Returns:
            Maximum position value in rupees
        """
        if metrics is None:
            metrics = self._metrics_cache.get(symbol)
            if metrics is None:
                return 0.0  # Conservative - no position if no data
        
        lc = self.config.liquidity_config
        
        # Volume-based constraint
        max_by_volume = metrics.avg_daily_volume_20d * lc.max_position_vs_avg_volume
        
        # Liquidity tier constraint
        tier_limit_pct = lc.liquidity_tier_limits[metrics.liquidity_tier]
        # Assuming total capital is available in config
        max_by_tier = self.config.portfolio_capital * tier_limit_pct
        
        # Impact-based constraint
        # Solve: max_impact_pct = price_impact_coef * sqrt(position / avg_volume)
        # position = (max_impact_pct / price_impact_coef)^2 * avg_volume
        if metrics.price_impact_coefficient > 0:
            impact_ratio = max_impact_pct / metrics.price_impact_coefficient
            max_by_impact = (impact_ratio ** 2) * metrics.avg_daily_volume_20d
        else:
            max_by_impact = float('inf')
        
        # Return most restrictive constraint
        return min(max_by_volume, max_by_tier, max_by_impact)
    
    def check_volume_concentration(
        self,
        portfolio_positions: Dict[str, float],
        new_symbol: str,
        new_position_value: float
    ) -> Tuple[float, List[str]]:
        """
        Check volume concentration risk across portfolio.
        
        Args:
            portfolio_positions: Dict of symbol -> position_value
            new_symbol: Symbol for new position
            new_position_value: Value of new position
            
        Returns:
            Tuple of (concentration_ratio, warnings)
        """
        warnings_list = []
        
        # Create hypothetical portfolio with new position
        test_portfolio = portfolio_positions.copy()
        test_portfolio[new_symbol] = test_portfolio.get(new_symbol, 0) + new_position_value
        
        # Calculate volume-weighted concentration
        total_volume_impact = 0.0
        
        for symbol, position_value in test_portfolio.items():
            metrics = self._metrics_cache.get(symbol)
            if metrics is None:
                warnings_list.append(f"No liquidity metrics for {symbol}")
                continue
            
            # Calculate volume utilization for this position
            volume_utilization = position_value / metrics.avg_daily_volume_20d
            total_volume_impact += volume_utilization
        
        # Calculate concentration ratio
        concentration_ratio = total_volume_impact / len(test_portfolio) if test_portfolio else 0
        
        # Check concentration thresholds
        if concentration_ratio > 0.2:  # 20% average volume utilization
            warnings_list.append(
                f"High volume concentration: {concentration_ratio:.1%} average utilization"
            )
        
        # Check for illiquid positions
        illiquid_count = sum(
            1 for symbol in test_portfolio
            if self._metrics_cache.get(symbol, LiquidityMetrics('', 0, 0, 0, 0, 0, LiquidityTier.ILLIQUID, '')).liquidity_tier == LiquidityTier.ILLIQUID
        )
        
        if illiquid_count > 2:
            warnings_list.append(f"Too many illiquid positions: {illiquid_count}")
        
        return concentration_ratio, warnings_list
    
    def update_volume_data(self, symbol: str, volume_data: VolumeData) -> None:
        """
        Update volume data and recalculate metrics for a symbol.
        
        Args:
            symbol: Stock symbol
            volume_data: New volume data
        """
        # Store volume data
        self._volume_cache[symbol] = volume_data
        
        # Recalculate metrics
        metrics = self.calculate_liquidity_metrics(volume_data)
        self._metrics_cache[symbol] = metrics
        
        self.logger.info(f"Updated liquidity metrics for {symbol}: tier={metrics.liquidity_tier.value}")
    
    def get_liquidity_alerts(self, symbol: Optional[str] = None) -> Dict[str, List[LiquidityAlert]]:
        """
        Get active liquidity alerts.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary of symbol -> list of alerts
        """
        if symbol:
            return {symbol: self._active_alerts.get(symbol, [])}
        return dict(self._active_alerts)
    
    def clear_alerts(self, symbol: Optional[str] = None) -> None:
        """
        Clear liquidity alerts.
        
        Args:
            symbol: Optional symbol to clear (clears all if None)
        """
        if symbol:
            self._active_alerts.pop(symbol, None)
        else:
            self._active_alerts.clear()
        
        self.logger.info(f"Cleared liquidity alerts for {symbol or 'all symbols'}")
    
    def get_validation_stats(self) -> Dict[str, float]:
        """Get liquidity validation statistics."""
        return {
            'total_validations': self._validation_count,
            'rejections': self._rejection_count,
            'approval_rate': (
                (self._validation_count - self._rejection_count) / self._validation_count
                if self._validation_count > 0 else 0
            ),
            'symbols_tracked': len(self._metrics_cache),
            'active_alerts': sum(len(alerts) for alerts in self._active_alerts.values())
        }
    
    # Private methods
    
    def _get_or_calculate_metrics(
        self,
        symbol: str,
        volume_data: Optional[VolumeData]
    ) -> LiquidityMetrics:
        """Get cached metrics or calculate from provided data."""
        # Check cache first
        if symbol in self._metrics_cache:
            cached_metrics = self._metrics_cache[symbol]
            # Check if cache is still fresh (within cache hours)
            cache_age = datetime.now() - cached_metrics.last_updated
            cache_limit = timedelta(hours=self.config.nse_lot_config.lot_size_cache_hours)
            
            if cache_age < cache_limit:
                return cached_metrics
        
        # Calculate new metrics if data provided
        if volume_data is not None:
            metrics = self.calculate_liquidity_metrics(volume_data)
            self._metrics_cache[symbol] = metrics
            return metrics
        
        # Use cached data or create default
        if symbol in self._metrics_cache:
            return self._metrics_cache[symbol]
        
        # Create conservative default metrics
        return LiquidityMetrics(
            symbol=symbol,
            avg_daily_volume_1d=0.0,
            avg_daily_volume_5d=0.0,
            avg_daily_volume_20d=0.0,
            volume_volatility=float('inf'),
            price_impact_coefficient=0.1,
            liquidity_tier=LiquidityTier.ILLIQUID,
            volume_trend='unknown'
        )
    
    def _perform_validation(
        self,
        symbol: str,
        position_value: float,
        metrics: LiquidityMetrics
    ) -> LiquidityValidationResult:
        """Perform the actual liquidity validation."""
        lc = self.config.liquidity_config
        alerts = []
        recommendations = []
        analysis_details = {}
        
        # Check if liquidity checks are enabled
        if not lc.enable_liquidity_checks:
            return LiquidityValidationResult(
                approved=True,
                liquidity_tier=metrics.liquidity_tier,
                max_position_value=position_value,
                estimated_market_impact=0.0,
                volume_concentration=0.0,
                alerts=[],
                analysis_details={'liquidity_checks_disabled': True},
                recommendations=[]
            )
        
        # Calculate constraints and metrics
        max_position = self.get_maximum_position_value(symbol, metrics=metrics)
        market_impact = self.estimate_market_impact(symbol, position_value, metrics)
        volume_utilization = position_value / metrics.avg_daily_volume_20d if metrics.avg_daily_volume_20d > 0 else float('inf')
        
        # Store analysis details
        analysis_details.update({
            'avg_daily_volume': metrics.avg_daily_volume_20d,
            'volume_utilization': volume_utilization,
            'max_position_allowed': max_position,
            'volume_volatility': metrics.volume_volatility,
            'price_impact_coefficient': metrics.price_impact_coefficient
        })
        
        # Check minimum volume requirement
        if metrics.avg_daily_volume_20d < lc.min_avg_daily_volume:
            alerts.append(LiquidityAlert.LOW_VOLUME)
            recommendations.append(f"Volume below minimum: {metrics.avg_daily_volume_20d:.0f} < {lc.min_avg_daily_volume:.0f}")
        
        # Check position size vs volume
        if volume_utilization > lc.max_position_vs_avg_volume:
            alerts.append(LiquidityAlert.HIGH_CONCENTRATION)
            recommendations.append(
                f"Position too large vs volume: {volume_utilization:.1%} > {lc.max_position_vs_avg_volume:.1%}"
            )
        
        # Check market impact
        if market_impact > 0.05:  # 5% impact threshold
            alerts.append(LiquidityAlert.MARKET_IMPACT_WARNING)
            recommendations.append(f"High market impact estimated: {market_impact:.1%}")
        
        # Check liquidity tier constraints
        if metrics.liquidity_tier == LiquidityTier.ILLIQUID:
            alerts.append(LiquidityAlert.ILLIQUID_POSITION)
            recommendations.append("Stock classified as illiquid - consider avoiding")
        
        # Volume trend warnings
        if metrics.volume_trend == 'decreasing':
            alerts.append(LiquidityAlert.VOLUME_DROP)
            recommendations.append("Declining volume trend detected")
        
        # Approval logic
        approved = (
            position_value <= max_position and
            market_impact <= 0.1 and  # 10% max impact
            metrics.avg_daily_volume_20d >= lc.min_avg_daily_volume and
            metrics.liquidity_tier != LiquidityTier.ILLIQUID
        )
        
        return LiquidityValidationResult(
            approved=approved,
            liquidity_tier=metrics.liquidity_tier,
            max_position_value=max_position,
            estimated_market_impact=market_impact,
            volume_concentration=volume_utilization,
            alerts=alerts,
            analysis_details=analysis_details,
            recommendations=recommendations
        )
    
    def _estimate_price_impact_coefficient(
        self,
        volumes: np.ndarray,
        prices: np.ndarray
    ) -> float:
        """Estimate price impact coefficient from historical data."""
        if len(volumes) < 10 or len(prices) < 10:
            return 0.05  # Default coefficient
        
        # Calculate price volatility
        returns = np.diff(np.log(prices))
        price_volatility = np.std(returns)
        
        # Calculate volume volatility
        volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
        
        # Simple model: impact coefficient is related to volatility and inverse liquidity
        # Higher volatility and lower/more variable volume -> higher impact
        base_coefficient = 0.01  # 1% base
        volatility_adjustment = price_volatility * 10  # Scale up volatility
        volume_adjustment = min(volume_cv, 2.0)  # Cap volume variability effect
        
        coefficient = base_coefficient * (1 + volatility_adjustment + volume_adjustment)
        
        # Reasonable bounds
        return max(0.005, min(coefficient, 0.2))  # Between 0.5% and 20%
    
    def _analyze_volume_trend(self, volumes: np.ndarray) -> str:
        """Analyze volume trend from recent data."""
        if len(volumes) < 10:
            return 'insufficient_data'
        
        # Compare recent vs older periods
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:-5])
        
        if recent_avg > older_avg * 1.2:
            return 'increasing'
        elif recent_avg < older_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _update_alerts(self, symbol: str, new_alerts: List[LiquidityAlert]) -> None:
        """Update alert tracking for a symbol."""
        self._active_alerts[symbol] = new_alerts
        
        # Add to alert history
        if new_alerts:
            self._alert_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'alerts': [alert.value for alert in new_alerts]
            })
            
            # Keep only recent history (last 1000 entries)
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]


# Utility functions

def create_volume_data_from_dataframe(df: pd.DataFrame, symbol: str) -> VolumeData:
    """
    Create VolumeData from pandas DataFrame.
    
    Args:
        df: DataFrame with columns ['date', 'volume', 'close']
        symbol: Stock symbol
        
    Returns:
        VolumeData instance
    """
    return VolumeData(
        symbol=symbol,
        dates=pd.to_datetime(df['date']).tolist(),
        volumes=df['volume'].tolist(),
        prices=df['close'].tolist()
    )


def calculate_portfolio_liquidity_score(
    positions: Dict[str, float],
    liquidity_validator: LiquidityValidator
) -> Tuple[float, Dict[str, str]]:
    """
    Calculate overall portfolio liquidity score.
    
    Args:
        positions: Dict of symbol -> position_value
        liquidity_validator: Validator instance with cached metrics
        
    Returns:
        Tuple of (liquidity_score, symbol_classifications)
    """
    if not positions:
        return 1.0, {}
    
    tier_scores = {
        LiquidityTier.HIGHLY_LIQUID: 1.0,
        LiquidityTier.MODERATELY_LIQUID: 0.7,
        LiquidityTier.LOW_LIQUID: 0.4,
        LiquidityTier.ILLIQUID: 0.1
    }
    
    total_value = sum(positions.values())
    weighted_score = 0.0
    classifications = {}
    
    for symbol, position_value in positions.items():
        metrics = liquidity_validator._metrics_cache.get(symbol)
        if metrics:
            tier = metrics.liquidity_tier
            score = tier_scores[tier]
            weight = position_value / total_value
            weighted_score += score * weight
            classifications[symbol] = tier.value
        else:
            # Conservative classification for unknown symbols
            classifications[symbol] = LiquidityTier.ILLIQUID.value
            weighted_score += 0.1 * (position_value / total_value)
    
    return weighted_score, classifications


# Export main components
__all__ = [
    'LiquidityValidator',
    'LiquidityValidationResult',
    'LiquidityAlert',
    'VolumeData',
    'LiquidityMetrics',
    'create_volume_data_from_dataframe',
    'calculate_portfolio_liquidity_score',
]