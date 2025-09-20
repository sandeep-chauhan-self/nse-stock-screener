import logging
"""
Composite Scoring System
Implements the probabilistic scoring framework (0-100) with weighted components
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from .common.enums import MarketRegime, ProbabilityLevel
from .common.volume_thresholds import VolumeThresholdCalculator, DEFAULT_VOLUME_CONFIG

# Data validation imports
from .data.validation import DataContract, DataValidator, safe_float, safe_bool, is_valid_numeric

# Set up logger
logger = logging.getLogger(__name__)

class CompositeScorer:
    """
    Implements the composite scoring system with:
    - Volume scoring (25 points)
    - Momentum scoring (25 points) 
    - Trend strength scoring (15 points)
    - Volatility scoring (10 points)
    - Relative strength scoring (10 points)
    - Volume profile scoring (10 points)
    - Weekly confirmation bonus (up to 10 points)
    - Market regime adjustments
    """
    
    def __init__(self):
        # Scoring weights (total = 100 base points)
        self.weights = {
            'volume': 25,
            'momentum': 25,
            'trend': 15,
            'volatility': 10,
            'relative_strength': 10,
            'volume_profile': 10
        }
        
        # Weekly confirmation bonus
        self.weekly_bonus = 10
        
        # Probability thresholds
        self.thresholds = {
            'high': 70,
            'medium': 45
        }
        
        # Volume threshold calculator for regime-specific volume analysis
        self.volume_calculator = VolumeThresholdCalculator(DEFAULT_VOLUME_CONFIG)
        
        # Track current symbol for volume calculator context
        self.current_symbol = None
        
        # Market regime adjustments
        self.regime_adjustments = {
            MarketRegime.BULLISH: {'rsi_min': 58, 'rsi_max': 82, 'vol_threshold': 2.5},
            MarketRegime.SIDEWAYS: {'rsi_min': 60, 'rsi_max': 80, 'vol_threshold': 3.0},
            MarketRegime.BEARISH: {'rsi_min': 62, 'rsi_max': 78, 'vol_threshold': 4.0},
            MarketRegime.HIGH_VOLATILITY: {'rsi_min': 65, 'rsi_max': 75, 'vol_threshold': 5.0}  # Added for completeness
        }
    
    def set_current_symbol(self, symbol: str):
        """
        Set the current symbol being analyzed for volume calculator context.
        
        Args:
            symbol: Stock symbol being analyzed
        """
        self.current_symbol = symbol
    
    def score_volume_component(self, indicators: Dict[str, Any], regime: MarketRegime = MarketRegime.SIDEWAYS) -> Tuple[int, Dict[str, Any]]:
        """
        Score volume component (25 points max) with safe value extraction:
        - Volume z-score: 15 points max
        - Volume ratio: 10 points max
        """
        score = 0
        breakdown = {}
        
        # Safe value extraction with validation
        vol_z = safe_float(indicators.get('vol_z', np.nan), np.nan, 'vol_z')
        vol_ratio = safe_float(indicators.get('vol_ratio', np.nan), np.nan, 'vol_ratio')
        
        # Get regime-adjusted thresholds
        regime_settings = self.regime_adjustments[regime]
        vol_threshold = regime_settings.get('vol_threshold', 1.5)
        
        # Volume z-score scoring with safe numeric checks
        if is_valid_numeric(vol_z):
            if vol_z >= 3:
                z_score = 15
                breakdown['vol_z_level'] = 'EXTREME'
            elif vol_z >= 2:
                z_score = 10
                breakdown['vol_z_level'] = 'HIGH'
            elif vol_z >= 1:
                z_score = 5
                breakdown['vol_z_level'] = 'MEDIUM'
            else:
                z_score = 0
                breakdown['vol_z_level'] = 'LOW'
        else:
            z_score = 0
            breakdown['vol_z_level'] = 'NO_DATA'
        
        # Volume ratio scoring (adjusted for regime using new threshold calculator)
        if is_valid_numeric(vol_ratio):
            # Use the volume threshold calculator for proper regime-specific thresholds
            ratio_score, level = self.volume_calculator.get_volume_score_and_level(
                vol_ratio=vol_ratio,
                regime=regime,
                historical_vol_ratios=None,  # Could be enhanced with historical data later
                symbol=self.current_symbol or 'UNKNOWN'
            )
            breakdown['vol_ratio_level'] = level
        else:
            ratio_score = 0
            breakdown['vol_ratio_level'] = 'NO_DATA'
        
        score = z_score + ratio_score
        breakdown['vol_z_score'] = z_score
        breakdown['vol_ratio_score'] = ratio_score
        breakdown['total_volume_score'] = score
        
        return score, breakdown
    
    def score_momentum_component(self, indicators: Dict[str, Any], regime: MarketRegime = MarketRegime.SIDEWAYS) -> Tuple[int, Dict[str, Any]]:
        """
        Score momentum component (25 points max):
        - RSI sweet spot: 12 points max
        - MACD bullish signals: 13 points max
        """
        score = 0
        breakdown = {}
        
        rsi = indicators.get('rsi', np.nan)
        macd = indicators.get('macd', np.nan)
        macd_signal = indicators.get('macd_signal', np.nan)
        macd_hist = indicators.get('macd_hist', np.nan)
        macd_strength = indicators.get('macd_strength', np.nan)
        
        # Get regime-adjusted RSI thresholds
        regime_settings = self.regime_adjustments[regime]
        rsi_min = regime_settings['rsi_min']
        rsi_max = regime_settings['rsi_max']
        
        # RSI scoring (regime-adjusted)
        if not np.isnan(rsi):
            if rsi_min + 10 <= rsi <= rsi_max - 2:  # Sweet spot (e.g., 70-78 in neutral)
                rsi_score = 12
                breakdown['rsi_level'] = 'SWEET_SPOT'
            elif rsi_min <= rsi <= rsi_max:  # Acceptable range
                rsi_score = 8
                breakdown['rsi_level'] = 'ACCEPTABLE'
            elif rsi > rsi_max:  # Overbought caution
                rsi_score = 6
                breakdown['rsi_level'] = 'OVERBOUGHT'
            elif rsi < rsi_min and rsi > 50:  # Mild momentum
                rsi_score = 4
                breakdown['rsi_level'] = 'MILD'
            else:
                rsi_score = 0
                breakdown['rsi_level'] = 'WEAK'
        else:
            rsi_score = 0
            breakdown['rsi_level'] = 'NO_DATA'
        
        # MACD scoring
        macd_score = 0
        if not np.isnan(macd) and not np.isnan(macd_signal):
            # MACD above signal line (bullish crossover)
            if macd > macd_signal:
                macd_score += 7
                breakdown['macd_crossover'] = 'BULLISH'
            else:
                breakdown['macd_crossover'] = 'BEARISH'
            
            # MACD above zero line
            if macd > 0:
                macd_score += 3
                breakdown['macd_position'] = 'ABOVE_ZERO'
            else:
                breakdown['macd_position'] = 'BELOW_ZERO'
            
            # MACD histogram momentum
            if not np.isnan(macd_hist) and macd_hist > 0:
                macd_score += 3
                breakdown['macd_momentum'] = 'POSITIVE'
            else:
                breakdown['macd_momentum'] = 'NEGATIVE'
        else:
            breakdown['macd_crossover'] = 'NO_DATA'
            breakdown['macd_position'] = 'NO_DATA'
            breakdown['macd_momentum'] = 'NO_DATA'
        
        score = rsi_score + macd_score
        breakdown['rsi_score'] = rsi_score
        breakdown['macd_score'] = macd_score
        breakdown['total_momentum_score'] = score
        
        return score, breakdown
    
    def score_trend_component(self, indicators: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Score trend strength component (15 points max):
        - ADX strength: 8 points max
        - MA alignment: 7 points max
        """
        score = 0
        breakdown = {}
        
        adx = indicators.get('adx', np.nan)
        ma_crossover = indicators.get('ma_crossover', 0)
        ma20_slope = indicators.get('ma20_slope', np.nan)
        ma50_slope = indicators.get('ma50_slope', np.nan)
        
        # ADX scoring
        if not np.isnan(adx):
            if adx > 35:
                adx_score = 8
                breakdown['adx_level'] = 'VERY_STRONG'
            elif adx > 25:
                adx_score = 6
                breakdown['adx_level'] = 'STRONG'
            elif adx > 20:
                adx_score = 3
                breakdown['adx_level'] = 'MODERATE'
            else:
                adx_score = 0
                breakdown['adx_level'] = 'WEAK'
        else:
            adx_score = 0
            breakdown['adx_level'] = 'NO_DATA'
        
        # MA alignment scoring
        ma_score = 0
        if ma_crossover == 1:  # MA20 > MA50
            ma_score += 4
            breakdown['ma_alignment'] = 'BULLISH'
        elif ma_crossover == -1:
            breakdown['ma_alignment'] = 'BEARISH'
        else:
            breakdown['ma_alignment'] = 'NEUTRAL'
        
        # MA slope scoring
        if not np.isnan(ma20_slope) and ma20_slope > 0:
            ma_score += 3
            breakdown['ma_slope'] = 'POSITIVE'
        elif not np.isnan(ma20_slope):
            breakdown['ma_slope'] = 'NEGATIVE'
        else:
            breakdown['ma_slope'] = 'NO_DATA'
        
        score = adx_score + ma_score
        breakdown['adx_score'] = adx_score
        breakdown['ma_score'] = ma_score
        breakdown['total_trend_score'] = score
        
        return score, breakdown
    
    def score_volatility_component(self, indicators: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Score volatility component (10 points max):
        - ATR expansion: 6 points max
        - ATR level: 4 points max
        """
        score = 0
        breakdown = {}
        
        atr_trend = indicators.get('atr_trend', np.nan)
        atr_pct = indicators.get('atr_pct', np.nan)
        
        # ATR trend scoring (expansion suggests upcoming move)
        if not np.isnan(atr_trend):
            if atr_trend > 0.2:  # 20% increase in ATR
                atr_trend_score = 6
                breakdown['atr_trend_level'] = 'EXPANDING'
            elif atr_trend > 0:
                atr_trend_score = 3
                breakdown['atr_trend_level'] = 'MILD_EXPANSION'
            else:
                atr_trend_score = 0
                breakdown['atr_trend_level'] = 'CONTRACTING'
        else:
            atr_trend_score = 0
            breakdown['atr_trend_level'] = 'NO_DATA'
        
        # ATR absolute level scoring
        if not np.isnan(atr_pct):
            if atr_pct > 3:  # High volatility
                atr_level_score = 4
                breakdown['atr_level'] = 'HIGH'
            elif atr_pct > 1.5:
                atr_level_score = 2
                breakdown['atr_level'] = 'MODERATE'
            else:
                atr_level_score = 0
                breakdown['atr_level'] = 'LOW'
        else:
            atr_level_score = 0
            breakdown['atr_level'] = 'NO_DATA'
        
        score = atr_trend_score + atr_level_score
        breakdown['atr_trend_score'] = atr_trend_score
        breakdown['atr_level_score'] = atr_level_score
        breakdown['total_volatility_score'] = score
        
        return score, breakdown
    
    def score_relative_strength_component(self, indicators: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Score relative strength component (10 points max):
        - 20-day relative performance vs NIFTY
        """
        score = 0
        breakdown = {}
        
        rel_strength_20d = indicators.get('rel_strength_20d', np.nan)
        rel_strength_50d = indicators.get('rel_strength_50d', np.nan)
        
        # 20-day relative strength (primary)
        if not np.isnan(rel_strength_20d):
            if rel_strength_20d > 10:  # Significantly outperforming
                rel_score_20d = 8
                breakdown['rel_20d_level'] = 'STRONG_OUTPERFORM'
            elif rel_strength_20d > 5:
                rel_score_20d = 6
                breakdown['rel_20d_level'] = 'OUTPERFORM'
            elif rel_strength_20d > 0:
                rel_score_20d = 3
                breakdown['rel_20d_level'] = 'MILD_OUTPERFORM'
            else:
                rel_score_20d = 0
                breakdown['rel_20d_level'] = 'UNDERPERFORM'
        else:
            rel_score_20d = 0
            breakdown['rel_20d_level'] = 'NO_DATA'
        
        # 50-day relative strength (bonus)
        if not np.isnan(rel_strength_50d) and rel_strength_50d > 5:
            rel_score_50d = 2
            breakdown['rel_50d_level'] = 'LONG_TERM_STRENGTH'
        else:
            rel_score_50d = 0
            breakdown['rel_50d_level'] = 'WEAK_OR_NO_DATA'
        
        score = rel_score_20d + rel_score_50d
        breakdown['rel_20d_score'] = rel_score_20d
        breakdown['rel_50d_score'] = rel_score_50d
        breakdown['total_rel_strength_score'] = score
        
        return score, breakdown
    
    def score_volume_profile_component(self, indicators: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Score volume profile component (10 points max):
        - Breakout above high-volume nodes
        """
        score = 0
        breakdown = {}
        
        vp_breakout_score = indicators.get('vp_breakout_score', 0)
        vp_resistance_level = indicators.get('vp_resistance_level', np.nan)
        
        # Direct scoring from volume profile analysis
        score = min(vp_breakout_score, 10)  # Cap at 10 points
        
        if score >= 8:
            breakdown['vp_level'] = 'CLEAR_BREAKOUT'
        elif score >= 5:
            breakdown['vp_level'] = 'APPROACHING_RESISTANCE'
        elif score > 0:
            breakdown['vp_level'] = 'MILD_SIGNAL'
        else:
            breakdown['vp_level'] = 'NO_SIGNAL'
        
        breakdown['vp_score'] = score
        breakdown['resistance_level'] = vp_resistance_level
        
        return score, breakdown
    
    def score_weekly_confirmation(self, indicators: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Score weekly confirmation bonus (10 points max):
        - Weekly RSI trend
        - Weekly MACD bullish
        - Weekly volume trend
        """
        score = 0
        breakdown = {}
        
        weekly_rsi_trend = indicators.get('weekly_rsi_trend', 0)
        weekly_macd_bullish = indicators.get('weekly_macd_bullish', False)
        weekly_vol_trend = indicators.get('weekly_vol_trend', 0)
        
        # Weekly RSI trending up
        if weekly_rsi_trend:
            score += 4
            breakdown['weekly_rsi'] = 'TRENDING_UP'
        else:
            breakdown['weekly_rsi'] = 'NOT_TRENDING'
        
        # Weekly MACD bullish
        if weekly_macd_bullish:
            score += 4
            breakdown['weekly_macd'] = 'BULLISH'
        else:
            breakdown['weekly_macd'] = 'BEARISH'
        
        # Weekly volume trending up
        if weekly_vol_trend:
            score += 2
            breakdown['weekly_volume'] = 'INCREASING'
        else:
            breakdown['weekly_volume'] = 'DECREASING'
        
        breakdown['total_weekly_score'] = score
        
        return score, breakdown
    
    def detect_market_regime(self, nifty_data: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Detect current market regime based on NIFTY indicators
        Simple implementation - can be enhanced with more sophisticated logic
        """
        if nifty_data is None or len(nifty_data) < 50:
            return MarketRegime.SIDEWAYS
        
        try:
            # Calculate NIFTY indicators
            close = nifty_data['Close']
            
            # Check trend: 20-day MA vs 50-day MA
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            
            # Check recent performance
            recent_return = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
            
            # Check volatility
            daily_returns = close.pct_change().dropna()
            volatility = daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Regime detection logic
            if ma20 > ma50 and recent_return > 3 and volatility < 25:
                return MarketRegime.BULLISH
            elif ma20 < ma50 and recent_return < -3:
                return MarketRegime.BEARISH
            elif volatility > 25:
                return MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logging.error(f"Warning: Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS
    
    def compute_composite_score(self, indicators: Dict[str, Any], regime: MarketRegime = MarketRegime.SIDEWAYS) -> Dict[str, Any]:
        """
        Compute the complete composite score with detailed breakdown and input validation.
        
        Args:
            indicators: Dictionary of indicators - must conform to DataContract.IndicatorDict
            regime: Market regime for scoring adjustment
            
        Returns:
            Complete score breakdown or None for critical validation failures
        """
        
        # Input validation at ingestion point
        if indicators is None:
            logging.warning("Received None indicators in compute_composite_score")
            return None
        
        # Validate indicators using our data contract
        validator = DataValidator()
        validated_indicators = validator.validate_indicators_dict(indicators, indicators.get('symbol', 'UNKNOWN'))
        
        if validated_indicators is None:
            logging.error(f"Critical validation failure for indicators in composite scoring: {indicators.get('symbol', 'UNKNOWN')}")
            return None
        
        # Log any validation warnings
        if validator.validation_results:
            validation_warnings = [r for r in validator.validation_results if r.severity.value == 'warning']
            validation_errors = [r for r in validator.validation_results if r.severity.value in ['error', 'critical']]
            
            if validation_warnings:
                logging.warning(f"Validation warnings for {validated_indicators.get('symbol', 'UNKNOWN')}: {len(validation_warnings)} issues")
            
            if validation_errors:
                logging.error(f"Validation errors for {validated_indicators.get('symbol', 'UNKNOWN')}: {len(validation_errors)} critical issues")
        
        # Use validated indicators for scoring with fallback handling
        try:
            total_score = 0
            full_breakdown = {
                'symbol': validated_indicators.get('symbol', 'UNKNOWN'),
                'market_regime': regime.value,
                'components': {},
                'validation_warnings': len([r for r in validator.validation_results if r.severity.value == 'warning']) if validator.validation_results else 0
            }
            
            # Score each component with graceful degradation
            vol_score, vol_breakdown = self.score_volume_component(validated_indicators, regime)
            momentum_score, momentum_breakdown = self.score_momentum_component(validated_indicators, regime)
            trend_score, trend_breakdown = self.score_trend_component(validated_indicators)
            volatility_score, volatility_breakdown = self.score_volatility_component(validated_indicators)
            rel_strength_score, rel_breakdown = self.score_relative_strength_component(validated_indicators)
            vp_score, vp_breakdown = self.score_volume_profile_component(validated_indicators)
            weekly_score, weekly_breakdown = self.score_weekly_confirmation(validated_indicators)
            
            # Calculate total with safe arithmetic
            component_scores = [vol_score, momentum_score, trend_score, volatility_score, rel_strength_score, vp_score, weekly_score]
            valid_scores = [score for score in component_scores if is_valid_numeric(score)]
            
            if not valid_scores:
                logging.warning(f"No valid component scores for {validated_indicators.get('symbol', 'UNKNOWN')}")
                total_score = 0
            else:
                total_score = safe_float(sum(valid_scores), 0, 'total_score')
        
        except Exception as e:
            logging.error(f"Error computing composite score for {validated_indicators.get('symbol', 'UNKNOWN')}: {e}", exc_info=True)
            return None
        
        # Store breakdown
        full_breakdown['components'] = {
            'volume': vol_breakdown,
            'momentum': momentum_breakdown,
            'trend': trend_breakdown,
            'volatility': volatility_breakdown,
            'relative_strength': rel_breakdown,
            'volume_profile': vp_breakdown,
            'weekly_confirmation': weekly_breakdown
        }
        
        # Determine probability level
        if total_score >= self.thresholds['high']:
            probability = ProbabilityLevel.HIGH
        elif total_score >= self.thresholds['medium']:
            probability = ProbabilityLevel.MEDIUM
        else:
            probability = ProbabilityLevel.LOW
        
        # Final result
        result = {
            'symbol': indicators.get('symbol', 'UNKNOWN'),
            'composite_score': min(100, int(total_score)),  # Cap at 100
            'probability_level': probability.value,
            'market_regime': regime.value,
            'component_scores': {
                'volume': vol_score,
                'momentum': momentum_score,
                'trend': trend_score,
                'volatility': volatility_score,
                'relative_strength': rel_strength_score,
                'volume_profile': vp_score,
                'weekly_confirmation': weekly_score
            },
            'key_indicators': {
                'current_price': indicators.get('current_price', np.nan),
                'price_change_pct': indicators.get('price_change_pct', np.nan),
                'volume_ratio': indicators.get('vol_ratio', np.nan),
                'volume_z_score': indicators.get('vol_z', np.nan),
                'rsi': indicators.get('rsi', np.nan),
                'macd_signal': 'BULLISH' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'BEARISH',
                'adx': indicators.get('adx', np.nan),
                'atr_pct': indicators.get('atr_pct', np.nan),
                'relative_strength_20d': indicators.get('rel_strength_20d', np.nan)
            },
            'detailed_breakdown': full_breakdown
        }
        
        return result

# Example usage and testing
if __name__ == "__main__":
    # Test the scoring system
    scorer = CompositeScorer()
    
    # Sample indicators data (you would get this from AdvancedIndicator)
    sample_indicators = {
        'symbol': 'RELIANCE.NS',
        'current_price': 2500.0,
        'price_change_pct': 2.5,
        'vol_ratio': 4.2,
        'vol_z': 2.8,
        'rsi': 68.5,
        'macd': 15.2,
        'macd_signal': 12.8,
        'macd_hist': 2.4,
        'adx': 28.5,
        'ma_crossover': 1,
        'ma20_slope': 0.002,
        'atr_pct': 2.1,
        'atr_trend': 0.15,
        'rel_strength_20d': 7.3,
        'vp_breakout_score': 8,
        'weekly_rsi_trend': 1,
        'weekly_macd_bullish': True,
        'weekly_vol_trend': 1
    }
    
    # Test scoring
    result = scorer.compute_composite_score(sample_indicators, MarketRegime.SIDEWAYS)
    
    if result:
        logger.info(f"Composite Score for {result['symbol']}: {result['composite_score']}/100", 
                   extra={'symbol': result['symbol'], 'composite_score': result['composite_score'],
                         'probability_level': result['probability_level'], 'market_regime': result['market_regime']})
        logger.info("Component scores breakdown", 
                   extra={'component_scores': result['component_scores']})
    else:
        logger.error("Failed to compute composite score")