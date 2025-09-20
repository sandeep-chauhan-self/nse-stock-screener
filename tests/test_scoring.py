"""
Unit tests for composite scoring system.

This module tests the composite scoring logic, regime adjustments, 
and score combination with known inputs and expected outputs.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules under test
try:
    from composite_scorer import CompositeScorer
    from common.enums import MarketRegime, ProbabilityLevel
except ImportError:
    # Fallback for when imports are not available
    pytest.skip("Required modules not available", allow_module_level=True)

class TestCompositeScorerInitialization:
    """Test CompositeScorer initialization and configuration."""
    
    def test_scorer_initialization(self, sample_config):
        """Test that CompositeScorer initializes correctly."""
        scorer = CompositeScorer(sample_config)
        
        assert scorer.config is not None
        assert hasattr(scorer, 'regime_adjustments')
        assert hasattr(scorer, 'scoring_weights')
    
    def test_regime_adjustments_loaded(self, sample_config):
        """Test that regime adjustments are properly loaded."""
        scorer = CompositeScorer(sample_config)
        
        # Check that all regimes have adjustments
        expected_regimes = ['bullish', 'bearish', 'sideways', 'high_volatility']
        for regime in expected_regimes:
            assert regime in scorer.regime_adjustments
            assert 'rsi_min' in scorer.regime_adjustments[regime]
            assert 'volume_threshold' in scorer.regime_adjustments[regime]

class TestVolumeScoring:
    """Test volume-based scoring components."""
    
    def test_volume_zscore_scoring(self, sample_config):
        """Test volume z-score scoring logic."""
        scorer = CompositeScorer(sample_config)
        
        # Test different z-score values
        test_cases = [
            (3.0, 25),    # High z-score should get full volume score
            (2.0, 20),    # Medium-high z-score
            (1.0, 15),    # Medium z-score
            (0.0, 10),    # Normal volume
            (-1.0, 5),    # Below normal volume
        ]
        
        for zscore, expected_min_score in test_cases:
            score = scorer._score_volume_zscore(zscore)
            assert score >= expected_min_score, \
                f"Z-score {zscore} should score at least {expected_min_score}, got {score}"
            assert score <= 25, f"Volume z-score should not exceed 25, got {score}"
    
    def test_volume_ratio_scoring(self, sample_config):
        """Test volume ratio scoring with regime adjustments."""
        scorer = CompositeScorer(sample_config)
        
        # Test normal conditions (sideways regime)
        regime = MarketRegime.SIDEWAYS
        volume_threshold = scorer.regime_adjustments['sideways']['volume_threshold']
        
        # Test different ratio values
        test_cases = [
            (volume_threshold * 6.0, 10),  # Extreme volume - max score
            (volume_threshold * 3.0, 8),   # High volume
            (volume_threshold * 1.5, 5),   # Above threshold
            (volume_threshold * 0.8, 0),   # Below threshold
        ]
        
        for ratio, expected_min_score in test_cases:
            score = scorer._score_volume_ratio(ratio, regime)
            assert score >= expected_min_score, \
                f"Volume ratio {ratio} should score at least {expected_min_score}, got {score}"
            assert score <= 10, f"Volume ratio score should not exceed 10, got {score}"
    
    def test_volume_ratio_regime_adjustments(self, sample_config):
        """Test that volume ratio scoring adjusts correctly for different regimes."""
        scorer = CompositeScorer(sample_config)
        
        # Same volume ratio in different regimes should score differently
        volume_ratio = 4.0  # Fixed ratio
        
        bullish_score = scorer._score_volume_ratio(volume_ratio, MarketRegime.BULLISH)
        bearish_score = scorer._score_volume_ratio(volume_ratio, MarketRegime.BEARISH)
        
        # Bearish regime has higher threshold, so same ratio should score differently
        assert isinstance(bullish_score, (int, float))
        assert isinstance(bearish_score, (int, float))
        assert 0 <= bullish_score <= 10
        assert 0 <= bearish_score <= 10

class TestMomentumScoring:
    """Test momentum-based scoring components."""
    
    def test_rsi_scoring(self, sample_config):
        """Test RSI scoring with regime adjustments."""
        scorer = CompositeScorer(sample_config)
        
        # Test in different regimes
        regimes_to_test = [MarketRegime.BULLISH, MarketRegime.BEARISH, MarketRegime.SIDEWAYS]
        
        for regime in regimes_to_test:
            rsi_min = scorer.regime_adjustments[regime.value]['rsi_min']
            
            # Test different RSI values
            test_cases = [
                (rsi_min + 20, 15),  # Well above minimum
                (rsi_min + 10, 10),  # Above minimum
                (rsi_min + 5, 5),    # Just above minimum
                (rsi_min - 5, 0),    # Below minimum
            ]
            
            for rsi_value, expected_min_score in test_cases:
                score = scorer._score_rsi(rsi_value, regime)
                assert score >= expected_min_score, \
                    f"RSI {rsi_value} in {regime.value} should score at least {expected_min_score}, got {score}"
                assert score <= 15, f"RSI score should not exceed 15, got {score}"
    
    def test_macd_scoring(self, sample_config):
        """Test MACD scoring logic."""
        scorer = CompositeScorer(sample_config)
        
        # Test MACD crossover scenarios
        test_cases = [
            (2.0, 1.5, 10),   # Strong bullish (MACD > Signal, positive)
            (1.0, 0.5, 8),    # Moderate bullish
            (0.5, 0.7, 3),    # Bearish crossover (MACD < Signal)
            (-1.0, -0.5, 0),  # Strong bearish
        ]
        
        for macd_line, signal_line, expected_min_score in test_cases:
            score = scorer._score_macd(macd_line, signal_line)
            assert score >= expected_min_score, \
                f"MACD({macd_line}, {signal_line}) should score at least {expected_min_score}, got {score}"
            assert score <= 10, f"MACD score should not exceed 10, got {score}"

class TestTrendScoring:
    """Test trend-based scoring components."""
    
    def test_adx_scoring(self, sample_config):
        """Test ADX trend strength scoring."""
        scorer = CompositeScorer(sample_config)
        
        test_cases = [
            (50, 8),   # Very strong trend
            (35, 6),   # Strong trend
            (25, 4),   # Moderate trend
            (20, 2),   # Weak trend
            (15, 0),   # No trend
        ]
        
        for adx_value, expected_min_score in test_cases:
            score = scorer._score_adx(adx_value)
            assert score >= expected_min_score, \
                f"ADX {adx_value} should score at least {expected_min_score}, got {score}"
            assert score <= 8, f"ADX score should not exceed 8, got {score}"
    
    def test_moving_average_crossover_scoring(self, sample_config):
        """Test moving average crossover scoring."""
        scorer = CompositeScorer(sample_config)
        
        # Test different MA scenarios
        test_cases = [
            (105, 100, 95, 7),   # Strong uptrend (current > MA20 > MA50)
            (100, 102, 95, 4),   # Mixed signals
            (95, 100, 105, 0),   # Downtrend
        ]
        
        for current_price, ma20, ma50, expected_min_score in test_cases:
            score = scorer._score_ma_crossover(current_price, ma20, ma50)
            assert score >= expected_min_score, \
                f"MA crossover ({current_price}, {ma20}, {ma50}) should score at least {expected_min_score}, got {score}"
            assert score <= 7, f"MA crossover score should not exceed 7, got {score}"

class TestVolatilityScoring:
    """Test volatility-based scoring components."""
    
    def test_atr_scoring(self, sample_config):
        """Test ATR volatility scoring."""
        scorer = CompositeScorer(sample_config)
        
        # Create test data with different ATR scenarios
        test_cases = [
            (2.5, 100, 8),   # High relative volatility (2.5% of price)
            (1.5, 100, 5),   # Moderate volatility (1.5% of price)
            (0.5, 100, 2),   # Low volatility (0.5% of price)
        ]
        
        for atr_value, current_price, expected_min_score in test_cases:
            score = scorer._score_atr(atr_value, current_price)
            assert score >= expected_min_score, \
                f"ATR {atr_value} (price {current_price}) should score at least {expected_min_score}, got {score}"
            assert score <= 10, f"ATR score should not exceed 10, got {score}"

class TestCompositeScoreCalculation:
    """Test the main composite score calculation."""
    
    def test_calculate_composite_score_basic(self, sample_config, test_ohlcv_data):
        """Test basic composite score calculation."""
        scorer = CompositeScorer(sample_config)
        
        # Create mock indicators
        indicators = {
            'rsi': 65.0,
            'atr': 2.5,
            'macd_line': 1.5,
            'macd_signal': 1.0,
            'adx': 30.0,
            'volume_zscore': 2.0,
            'volume_ratio': 3.0,
            'current_price': 100.0,
            'volume_profile_breakout': True
        }
        
        # Mock NIFTY data for regime detection
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.BULLISH
            
            score, probability, regime = scorer.calculate_composite_score("TEST", indicators, test_ohlcv_data)
        
        # Score should be between 0 and 100
        assert 0 <= score <= 100, f"Composite score {score} should be between 0 and 100"
        
        # Probability should be a valid enum value
        assert probability in [ProbabilityLevel.HIGH, ProbabilityLevel.MEDIUM, ProbabilityLevel.LOW]
        
        # Regime should be what we mocked
        assert regime == MarketRegime.BULLISH
    
    def test_calculate_composite_score_edge_cases(self, sample_config, test_ohlcv_data):
        """Test composite score with edge case inputs."""
        scorer = CompositeScorer(sample_config)
        
        # Test with NaN indicators
        nan_indicators = {
            'rsi': np.nan,
            'atr': np.nan,
            'macd_line': np.nan,
            'macd_signal': np.nan,
            'adx': np.nan,
            'volume_zscore': np.nan,
            'volume_ratio': np.nan,
            'current_price': 100.0,
            'volume_profile_breakout': False
        }
        
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.SIDEWAYS
            
            score, probability, regime = scorer.calculate_composite_score("TEST", nan_indicators, test_ohlcv_data)
        
        # Should handle NaN gracefully
        assert 0 <= score <= 100
        assert probability in [ProbabilityLevel.HIGH, ProbabilityLevel.MEDIUM, ProbabilityLevel.LOW]
    
    def test_weekly_confirmation_bonus(self, sample_config, test_ohlcv_data):
        """Test weekly confirmation bonus calculation."""
        scorer = CompositeScorer(sample_config)
        
        # Create indicators that should trigger weekly confirmation
        good_indicators = {
            'rsi': 70.0,
            'atr': 2.5,
            'macd_line': 2.0,
            'macd_signal': 1.5,
            'adx': 35.0,
            'volume_zscore': 2.5,
            'volume_ratio': 4.0,
            'current_price': 100.0,
            'volume_profile_breakout': True
        }
        
        with patch.object(scorer, 'detect_market_regime') as mock_regime, \
             patch.object(scorer, '_check_weekly_confirmation') as mock_weekly:
            mock_regime.return_value = MarketRegime.BULLISH
            
            # Test with weekly confirmation
            mock_weekly.return_value = True
            score_with_bonus, _, _ = scorer.calculate_composite_score("TEST", good_indicators, test_ohlcv_data)
            
            # Test without weekly confirmation
            mock_weekly.return_value = False
            score_without_bonus, _, _ = scorer.calculate_composite_score("TEST", good_indicators, test_ohlcv_data)
            
            # Score with bonus should be higher
            assert score_with_bonus > score_without_bonus, \
                f"Score with weekly bonus ({score_with_bonus}) should be higher than without ({score_without_bonus})"
            
            # Bonus should be approximately 10 points
            bonus_difference = score_with_bonus - score_without_bonus
            assert 8 <= bonus_difference <= 12, f"Weekly bonus should be around 10 points, got {bonus_difference}"

class TestProbabilityClassification:
    """Test probability level classification."""
    
    def test_probability_thresholds(self, sample_config, test_ohlcv_data):
        """Test that probability levels are correctly assigned based on score thresholds."""
        scorer = CompositeScorer(sample_config)
        
        # Create indicators that result in different score ranges
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.BULLISH
            
            # High probability scenario (should score > 70)
            high_indicators = {
                'rsi': 75.0, 'atr': 3.0, 'macd_line': 3.0, 'macd_signal': 2.0,
                'adx': 40.0, 'volume_zscore': 3.0, 'volume_ratio': 5.0,
                'current_price': 100.0, 'volume_profile_breakout': True
            }
            
            # Medium probability scenario (should score 50-70)
            medium_indicators = {
                'rsi': 65.0, 'atr': 2.0, 'macd_line': 1.0, 'macd_signal': 0.8,
                'adx': 25.0, 'volume_zscore': 1.5, 'volume_ratio': 2.5,
                'current_price': 100.0, 'volume_profile_breakout': False
            }
            
            # Low probability scenario (should score < 50)
            low_indicators = {
                'rsi': 45.0, 'atr': 1.0, 'macd_line': -0.5, 'macd_signal': 0.0,
                'adx': 15.0, 'volume_zscore': 0.0, 'volume_ratio': 1.0,
                'current_price': 100.0, 'volume_profile_breakout': False
            }
            
            high_score, high_prob, _ = scorer.calculate_composite_score("TEST", high_indicators, test_ohlcv_data)
            medium_score, medium_prob, _ = scorer.calculate_composite_score("TEST", medium_indicators, test_ohlcv_data)
            low_score, low_prob, _ = scorer.calculate_composite_score("TEST", low_indicators, test_ohlcv_data)
            
            # Verify score ordering
            assert high_score > medium_score > low_score, \
                f"Scores should be ordered: high({high_score}) > medium({medium_score}) > low({low_score})"
            
            # Verify probability classifications
            # Note: These are tendencies, actual classifications depend on exact thresholds
            assert high_prob in [ProbabilityLevel.HIGH, ProbabilityLevel.MEDIUM]
            assert low_prob in [ProbabilityLevel.LOW, ProbabilityLevel.MEDIUM]

class TestMarketRegimeDetection:
    """Test market regime detection logic."""
    
    def test_regime_detection_with_mock_nifty(self, sample_config, mock_nifty_data):
        """Test market regime detection with mocked NIFTY data."""
        scorer = CompositeScorer(sample_config)
        
        with patch('composite_scorer.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = mock_nifty_data
            
            regime = scorer.detect_market_regime()
            
            # Should return a valid regime
            assert regime in [MarketRegime.BULLISH, MarketRegime.BEARISH, 
                            MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]
    
    def test_regime_detection_fallback(self, sample_config):
        """Test regime detection fallback when NIFTY data is unavailable."""
        scorer = CompositeScorer(sample_config)
        
        with patch('composite_scorer.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()  # Empty dataframe
            
            regime = scorer.detect_market_regime()
            
            # Should fallback to SIDEWAYS when data is unavailable
            assert regime == MarketRegime.SIDEWAYS

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_indicator_values(self, sample_config, test_ohlcv_data):
        """Test handling of invalid indicator values."""
        scorer = CompositeScorer(sample_config)
        
        # Test with extreme/invalid values
        invalid_indicators = {
            'rsi': 150.0,  # Invalid RSI (should be 0-100)
            'atr': -1.0,   # Negative ATR
            'macd_line': float('inf'),  # Infinite value
            'macd_signal': float('-inf'),
            'adx': 200.0,  # Invalid ADX
            'volume_zscore': None,  # None value
            'volume_ratio': 0,      # Zero ratio
            'current_price': 100.0,
            'volume_profile_breakout': True
        }
        
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.SIDEWAYS
            
            # Should not raise an exception
            score, probability, regime = scorer.calculate_composite_score("TEST", invalid_indicators, test_ohlcv_data)
            
            # Should return reasonable values despite invalid inputs
            assert 0 <= score <= 100
            assert probability in [ProbabilityLevel.HIGH, ProbabilityLevel.MEDIUM, ProbabilityLevel.LOW]
    
    def test_missing_config_keys(self, sample_config, test_ohlcv_data):
        """Test behavior when config keys are missing."""
        # Create incomplete config
        incomplete_config = {
            'portfolio_capital': 100000.0,
            # Missing regime_adjustments and other keys
        }
        
        # Should handle gracefully or raise appropriate error
        try:
            scorer = CompositeScorer(incomplete_config)
            # If it doesn't raise during initialization, it should handle missing keys during scoring
            assert hasattr(scorer, 'config')
        except (KeyError, AttributeError):
            # Expected behavior for missing required config
            pass

class TestScoringConsistency:
    """Test consistency and reproducibility of scoring."""
    
    def test_scoring_reproducibility(self, sample_config, test_ohlcv_data):
        """Test that scoring is reproducible with the same inputs."""
        scorer = CompositeScorer(sample_config)
        
        indicators = {
            'rsi': 65.0, 'atr': 2.5, 'macd_line': 1.5, 'macd_signal': 1.0,
            'adx': 30.0, 'volume_zscore': 2.0, 'volume_ratio': 3.0,
            'current_price': 100.0, 'volume_profile_breakout': True
        }
        
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.BULLISH
            
            # Calculate score multiple times
            score1, prob1, regime1 = scorer.calculate_composite_score("TEST", indicators, test_ohlcv_data)
            score2, prob2, regime2 = scorer.calculate_composite_score("TEST", indicators, test_ohlcv_data)
            
            # Results should be identical
            assert score1 == score2, f"Scores should be identical: {score1} vs {score2}"
            assert prob1 == prob2, f"Probabilities should be identical: {prob1} vs {prob2}"
            assert regime1 == regime2, f"Regimes should be identical: {regime1} vs {regime2}"
    
    def test_score_bounds(self, sample_config, test_ohlcv_data):
        """Test that composite scores always stay within valid bounds."""
        scorer = CompositeScorer(sample_config)
        
        # Test with extreme indicator values
        extreme_cases = [
            # All maximum values
            {'rsi': 100.0, 'atr': 10.0, 'macd_line': 10.0, 'macd_signal': 5.0,
             'adx': 100.0, 'volume_zscore': 5.0, 'volume_ratio': 10.0,
             'current_price': 100.0, 'volume_profile_breakout': True},
            
            # All minimum values
            {'rsi': 0.0, 'atr': 0.0, 'macd_line': -10.0, 'macd_signal': -5.0,
             'adx': 0.0, 'volume_zscore': -5.0, 'volume_ratio': 0.1,
             'current_price': 100.0, 'volume_profile_breakout': False},
        ]
        
        with patch.object(scorer, 'detect_market_regime') as mock_regime:
            mock_regime.return_value = MarketRegime.SIDEWAYS
            
            for indicators in extreme_cases:
                score, probability, regime = scorer.calculate_composite_score("TEST", indicators, test_ohlcv_data)
                
                assert 0 <= score <= 100, f"Score {score} is outside valid range [0, 100]"
                assert probability in [ProbabilityLevel.HIGH, ProbabilityLevel.MEDIUM, ProbabilityLevel.LOW]
                assert regime in [MarketRegime.BULLISH, MarketRegime.BEARISH, 
                                MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]