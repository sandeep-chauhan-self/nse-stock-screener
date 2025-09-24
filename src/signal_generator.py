"""
Signal Generator Module
Generates BUY/HOLD/AVOID signals based on composite scores and market regime adjustments.
"""

import numpy as np
from typing import Dict, Any, Optional

from constants import MarketRegime, REGIME_ADJUSTMENTS

class SignalGenerator:
    """
    Signal generation based on composite scores with market regime adjustments.
    
    Signal Types:
    - BUY: High probability entry opportunity
    - HOLD: Medium probability, wait for better setup
    - AVOID: Low probability, avoid entry
    """
    
    def __init__(self):
        # Base thresholds for different market regimes
        self.regime_thresholds = {
            MarketRegime.BULLISH: {
                'buy_threshold': 65,     # Lower threshold in bull market
                'hold_threshold': 40,    # More lenient hold
            },
            MarketRegime.BEARISH: {
                'buy_threshold': 75,     # Higher threshold in bear market
                'hold_threshold': 50,    # Stricter hold requirements
            },
            MarketRegime.SIDEWAYS: {
                'buy_threshold': 70,     # Standard threshold
                'hold_threshold': 45,    # Standard hold
            },
            MarketRegime.HIGH_VOLATILITY: {
                'buy_threshold': 75,     # Higher threshold in volatile market
                'hold_threshold': 50,    # Stricter requirements
            }
        }
    
    def generate_signal(self, 
                       composite_score: int, 
                       indicators: Dict[str, Any], 
                       regime: MarketRegime = MarketRegime.SIDEWAYS) -> Dict[str, Any]:
        """
        Generate trading signal based on composite score and confirmations.
        
        Args:
            composite_score: 0-100 weighted composite score
            indicators: Dictionary of technical indicators
            regime: Current market regime
            
        Returns:
            Dictionary with signal, confidence, and reasoning
        """
        try:
            # Get regime-adjusted thresholds
            thresholds = self.regime_thresholds.get(regime, self.regime_thresholds[MarketRegime.SIDEWAYS])
            buy_threshold = thresholds['buy_threshold']
            hold_threshold = thresholds['hold_threshold']
            
            # Primary signal based on composite score
            if composite_score >= buy_threshold:
                primary_signal = "BUY"
                base_confidence = min(100, (composite_score - buy_threshold) / (100 - buy_threshold) * 50 + 50)
            elif composite_score >= hold_threshold:
                primary_signal = "HOLD"
                base_confidence = (composite_score - hold_threshold) / (buy_threshold - hold_threshold) * 30 + 30
            else:
                primary_signal = "AVOID"
                base_confidence = max(10, 30 - (hold_threshold - composite_score) / hold_threshold * 20)
            
            # Apply confirmations and filters
            signal_result = self._apply_confirmations(
                primary_signal, base_confidence, indicators, regime
            )
            
            # Add reasoning
            signal_result['reasoning'] = self._build_reasoning(
                composite_score, primary_signal, indicators, regime, thresholds
            )
            
            return signal_result
            
        except Exception as e:
            print(f"ERROR in SignalGenerator.generate_signal: {e}")
            import traceback
            traceback.print_exc()
            return {
                'signal': 'AVOID',
                'confidence': 0,
                'reasoning': f'Error in signal generation: {e}',
                'volume_confirmed': False,
                'trend_confirmed': False,
                'momentum_confirmed': False
            }
    
    def _apply_confirmations(self, 
                           signal: str, 
                           confidence: float, 
                           indicators: Dict[str, Any], 
                           regime: MarketRegime) -> Dict[str, Any]:
        """Apply additional confirmations to strengthen or weaken signal"""
        
        confirmations = {
            'volume_confirmed': False,
            'trend_confirmed': False,
            'momentum_confirmed': False
        }
        
        # Volume confirmation
        vol_ratio = indicators.get('vol_ratio', 1.0)
        vol_z = indicators.get('vol_z', 0.0)
        
        if vol_ratio >= 1.5 or vol_z >= 1.0:  # Above average volume
            confirmations['volume_confirmed'] = True
            if signal == "BUY":
                confidence = min(100, confidence + 5)
        
        # Trend confirmation
        adx = indicators.get('adx', 0)
        ma_signal = indicators.get('ma_signal', 'NEUTRAL')
        
        # Fallback to ma_crossover if ma_signal not available
        if ma_signal == 'NEUTRAL':
            ma_crossover = indicators.get('ma_crossover', 0)
            if ma_crossover > 0:
                ma_signal = 'BULLISH'
            elif ma_crossover < 0:
                ma_signal = 'BEARISH'
        
        if adx >= 25 and ma_signal in ['BULLISH', 'STRONG_BULLISH']:
            confirmations['trend_confirmed'] = True
            if signal == "BUY":
                confidence = min(100, confidence + 5)
        
        # Momentum confirmation
        rsi = indicators.get('rsi', 50)
        macd_signal = indicators.get('macd_signal', 'NEUTRAL')
        weekly_confirmation = indicators.get('weekly_rsi_trend', 0)
        
        # Handle both string and numeric MACD signals
        macd_bullish = False
        if isinstance(macd_signal, str):
            macd_bullish = macd_signal == 'BULLISH'
        elif isinstance(macd_signal, (int, float)):
            macd_bullish = macd_signal > 0
        
        # RSI in favorable range and MACD bullish
        if 45 <= rsi <= 75 and macd_bullish and weekly_confirmation > 0:
            confirmations['momentum_confirmed'] = True
            if signal == "BUY":
                confidence = min(100, confidence + 5)
        
        # Signal degradation for lack of confirmations
        confirmed_count = sum(confirmations.values())
        if signal == "BUY" and confirmed_count == 0:
            signal = "HOLD"
            confidence = max(30, confidence - 15)
        elif signal == "HOLD" and confirmed_count == 0:
            confidence = max(20, confidence - 10)
        
        # Market regime adjustments
        if regime == MarketRegime.BEARISH and signal == "BUY":
            confidence = max(20, confidence - 10)  # More conservative in bear market
        elif regime == MarketRegime.HIGH_VOLATILITY:
            confidence = max(20, confidence - 5)   # Reduce confidence in volatile markets
        
        return {
            'signal': signal,
            'confidence': round(confidence, 1),
            **confirmations
        }
    
    def _build_reasoning(self, 
                        score: int, 
                        signal: str, 
                        indicators: Dict[str, Any], 
                        regime: MarketRegime,
                        thresholds: Dict[str, int]) -> str:
        """Build human-readable reasoning for the signal"""
        
        reasoning_parts = []
        
        # Score-based reasoning
        reasoning_parts.append(f"Composite score: {score}/100")
        
        if signal == "BUY":
            reasoning_parts.append(f"Exceeds {regime.value} buy threshold ({thresholds['buy_threshold']})")
        elif signal == "HOLD":
            reasoning_parts.append(f"Between hold ({thresholds['hold_threshold']}) and buy ({thresholds['buy_threshold']}) thresholds")
        else:
            reasoning_parts.append(f"Below hold threshold ({thresholds['hold_threshold']})")
        
        # Key indicator highlights
        rsi = indicators.get('rsi', 0)
        if rsi > 0:
            if rsi >= 70:
                reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi <= 30:
                reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            else:
                reasoning_parts.append(f"RSI neutral ({rsi:.1f})")
        
        macd_signal = indicators.get('macd_signal', 'NEUTRAL')
        if macd_signal != 'NEUTRAL' and hasattr(macd_signal, 'lower'):
            reasoning_parts.append(f"MACD {macd_signal.lower()}")
        elif isinstance(macd_signal, (int, float)) and macd_signal != 0:
            if macd_signal > 0:
                reasoning_parts.append("MACD bullish")
            else:
                reasoning_parts.append("MACD bearish")
        
        vol_ratio = indicators.get('vol_ratio', 1.0)
        if vol_ratio >= 2.0:
            reasoning_parts.append(f"High volume ({vol_ratio:.1f}x avg)")
        elif vol_ratio >= 1.5:
            reasoning_parts.append(f"Above avg volume ({vol_ratio:.1f}x)")
        
        # Market regime context
        reasoning_parts.append(f"Market: {regime.value}")
        
        return "; ".join(reasoning_parts)
    
    def get_signal_summary(self, signals: list) -> Dict[str, Any]:
        """Generate summary statistics for a batch of signals"""
        if not signals:
            return {'total': 0, 'buy': 0, 'hold': 0, 'avoid': 0}
        
        signal_counts = {'BUY': 0, 'HOLD': 0, 'AVOID': 0}
        total_confidence = 0
        confirmations = {'volume': 0, 'trend': 0, 'momentum': 0}
        
        for signal_result in signals:
            signal_type = signal_result.get('signal', 'AVOID')
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            total_confidence += signal_result.get('confidence', 0)
            
            if signal_result.get('volume_confirmed', False):
                confirmations['volume'] += 1
            if signal_result.get('trend_confirmed', False):
                confirmations['trend'] += 1
            if signal_result.get('momentum_confirmed', False):
                confirmations['momentum'] += 1
        
        total_signals = len(signals)
        avg_confidence = total_confidence / total_signals if total_signals > 0 else 0
        
        return {
            'total': total_signals,
            'buy': signal_counts['BUY'],
            'hold': signal_counts['HOLD'],
            'avoid': signal_counts['AVOID'],
            'avg_confidence': round(avg_confidence, 1),
            'volume_confirmation_rate': round(confirmations['volume'] / total_signals * 100, 1),
            'trend_confirmation_rate': round(confirmations['trend'] / total_signals * 100, 1),
            'momentum_confirmation_rate': round(confirmations['momentum'] / total_signals * 100, 1)
        }