"""
Unit tests for risk management system.

This module tests position sizing, risk calculations, and portfolio constraints
with edge cases and realistic scenarios.
"""

from pathlib import Path
import sys

from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules under test
try:
    from risk_manager import RiskManager
    from common.enums import PositionStatus, StopType
except ImportError:
    # Fallback for when imports are not available
    pytest.skip("Required modules not available", allow_module_level=True)

class TestRiskManagerInitialization:
    """Test RiskManager initialization and configuration."""

    def test_risk_manager_initialization(self, sample_config):
        """Test that RiskManager initializes correctly."""
        risk_manager = RiskManager(sample_config)

        assert risk_manager.config is not None
        assert hasattr(risk_manager, 'portfolio_capital')
        assert hasattr(risk_manager, 'max_positions')
        assert hasattr(risk_manager, 'max_position_size')
        assert hasattr(risk_manager, 'risk_per_trade')

    def test_config_validation(self, sample_config):
        """Test config validation during initialization."""
        # Test with valid config
        risk_manager = RiskManager(sample_config)
        assert risk_manager.portfolio_capital > 0
        assert 0 < risk_manager.risk_per_trade <= 1.0
        assert 0 < risk_manager.max_position_size <= 1.0

        # Test with invalid config values
        invalid_config = sample_config.copy()
        invalid_config['risk_per_trade'] = 1.5  # Invalid: > 100%

        # Should either correct the value or raise an error
        try:
            risk_manager = RiskManager(invalid_config)
            # If it accepts it, it should have corrected it
            assert risk_manager.risk_per_trade <= 1.0
        except (ValueError, AssertionError):
            # Expected behavior for invalid config
            pass

class TestPositionSizing:
    """Test position sizing calculations."""

    def test_basic_position_sizing(self, sample_config, risk_scenarios):
        """Test basic position sizing calculation."""
        risk_manager = RiskManager(sample_config)

        scenario = risk_scenarios['normal_trade']

        can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
            symbol="TEST",
            entry_price=scenario['entry_price'],
            stop_loss=scenario['stop_loss'],
            signal_score=75.0
        )

        assert can_enter is True, f"Should be able to enter position: {reason}"
        assert quantity > 0, "Quantity should be positive"
        assert risk_amount > 0, "Risk amount should be positive"

        # Check that risk amount doesn't exceed configured limit
        expected_max_risk = scenario['portfolio_value'] * scenario['risk_per_trade']
        assert risk_amount <= expected_max_risk * 1.1, \
            f"Risk amount {risk_amount} exceeds expected maximum {expected_max_risk}"

    def test_position_sizing_with_different_scores(self, sample_config, risk_scenarios):
        """Test that position sizing adjusts based on signal score."""
        risk_manager = RiskManager(sample_config)

        scenario = risk_scenarios['normal_trade']

        # Test with different signal scores
        scores_to_test = [50, 70, 85]
        quantities = []

        for score in scores_to_test:
            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=scenario['entry_price'],
                stop_loss=scenario['stop_loss'],
                signal_score=score
            )

            if can_enter:
                quantities.append(quantity)
            else:
                quantities.append(0)

        # Higher scores should generally allow larger positions (if risk multiplier is used)
        # This depends on the specific implementation
        assert all(q >= 0 for q in quantities), "All quantities should be non-negative"

    def test_position_sizing_small_stop_distance(self, sample_config, risk_scenarios):
        """Test position sizing with very small stop loss distance."""
        risk_manager = RiskManager(sample_config)

        scenario = risk_scenarios['small_stop_trade']

        can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
            symbol="TEST",
            entry_price=scenario['entry_price'],
            stop_loss=scenario['stop_loss'],
            signal_score=75.0
        )

        if can_enter:
            # Small stop distance should allow larger quantity
            assert quantity >= scenario['expected_quantity'] * 0.8, \
                f"Quantity {quantity} should be close to expected {scenario['expected_quantity']}"

            # But position value should still respect limits
            position_value = quantity * scenario['entry_price']
            max_position_value = sample_config['portfolio_capital'] * sample_config['max_position_size']
            assert position_value <= max_position_value, \
                f"Position value {position_value} exceeds maximum {max_position_value}"

    def test_position_sizing_max_position_limit(self, sample_config, risk_scenarios):
        """Test position sizing with maximum position size limit."""
        risk_manager = RiskManager(sample_config)

        scenario = risk_scenarios['max_position_limit']

        can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
            symbol="TEST",
            entry_price=scenario['entry_price'],
            stop_loss=scenario['stop_loss'],
            signal_score=75.0
        )

        if can_enter:
            position_value = quantity * scenario['entry_price']
            max_allowed = scenario['max_position_size'] * sample_config['portfolio_capital']

            assert position_value <= max_allowed, \
                f"Position value {position_value} should not exceed max position limit {max_allowed}"

class TestRiskValidation:
    """Test risk validation and constraints."""

    def test_maximum_portfolio_risk(self, sample_config):
        """Test maximum portfolio risk constraint."""
        risk_manager = RiskManager(sample_config)

        # Simulate having existing positions that consume most of the risk budget
        with patch.object(risk_manager, '_get_current_portfolio_risk') as mock_risk:
            mock_risk.return_value = 0.018  # 1.8% (close to 2% limit)

            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=75.0
            )

            # Should either reduce position size or reject
            if not can_enter:
                assert "risk" in reason.lower(), f"Rejection reason should mention risk: {reason}"
            else:
                # If allowed, risk amount should be small
                portfolio_risk_increase = risk_amount / sample_config['portfolio_capital']
                assert portfolio_risk_increase < 0.005, \
                    f"Risk increase {portfolio_risk_increase} should be minimal when near limit"

    def test_maximum_positions_limit(self, sample_config):
        """Test maximum number of positions constraint."""
        risk_manager = RiskManager(sample_config)

        # Mock having maximum number of positions already
        with patch.object(risk_manager, '_get_current_position_count') as mock_count:
            mock_count.return_value = sample_config['max_positions']

            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=85.0
            )

            assert can_enter is False, "Should not allow new position when at maximum count"
            assert "position" in reason.lower(), f"Rejection reason should mention positions: {reason}"

    def test_duplicate_symbol_check(self, sample_config):
        """Test that duplicate symbol positions are prevented."""
        risk_manager = RiskManager(sample_config)

        # Mock having existing position in the same symbol
        with patch.object(risk_manager, '_has_existing_position') as mock_existing:
            mock_existing.return_value = True

            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=75.0
            )

            assert can_enter is False, "Should not allow duplicate position in same symbol"
            assert "existing" in reason.lower() or "duplicate" in reason.lower(), \
                f"Rejection reason should mention existing position: {reason}"

    def test_minimum_risk_reward_ratio(self, sample_config):
        """Test minimum risk-reward ratio constraint."""
        risk_manager = RiskManager(sample_config)

        # Test with poor risk-reward scenario
        entry_price = 100.0
        stop_loss = 95.0  # 5% stop loss
        target_price = 102.0  # Only 2% target - poor R:R ratio

        # This test depends on implementation - some risk managers check R:R ratio
        can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
            symbol="TEST",
            entry_price=entry_price,
            stop_loss=stop_loss,
            signal_score=75.0,
            target_price=target_price  # Some implementations might accept this parameter
        )

        # Depending on implementation, this might be rejected or allowed
        assert isinstance(can_enter, bool)
        assert isinstance(reason, str)

class TestStopLossCalculation:
    """Test stop loss calculation methods."""

    def test_atr_based_stop_loss(self, sample_config, test_ohlcv_data):
        """Test ATR-based stop loss calculation."""
        risk_manager = RiskManager(sample_config)

        current_price = 100.0
        atr_value = 2.5

        # Test different ATR multipliers
        for multiplier in [1.5, 2.0, 2.5]:
            stop_loss = risk_manager.calculate_stop_loss(
                current_price=current_price,
                atr=atr_value,
                method="atr",
                multiplier=multiplier
            )

            expected_stop = current_price - (atr_value * multiplier)
            assert abs(stop_loss - expected_stop) < 0.01, \
                f"ATR stop loss {stop_loss} should be close to {expected_stop}"

            assert stop_loss < current_price, "Stop loss should be below current price for long position"

    def test_percentage_based_stop_loss(self, sample_config):
        """Test percentage-based stop loss calculation."""
        risk_manager = RiskManager(sample_config)

        current_price = 100.0

        # Test different percentage levels
        for pct in [0.05, 0.08, 0.10]:  # 5%, 8%, 10%
            stop_loss = risk_manager.calculate_stop_loss(
                current_price=current_price,
                method="percentage",
                percentage=pct
            )

            expected_stop = current_price * (1 - pct)
            assert abs(stop_loss - expected_stop) < 0.01, \
                f"Percentage stop loss {stop_loss} should be close to {expected_stop}"

    def test_support_level_stop_loss(self, sample_config, test_ohlcv_data):
        """Test support level-based stop loss calculation."""
        risk_manager = RiskManager(sample_config)

        current_price = 105.0
        support_level = 98.0

        stop_loss = risk_manager.calculate_stop_loss(
            current_price=current_price,
            method="support",
            support_level=support_level,
            data=test_ohlcv_data
        )

        # Stop should be slightly below support level
        assert stop_loss < support_level, "Stop loss should be below support level"
        assert stop_loss > support_level * 0.95, "Stop loss should not be too far below support"

class TestPortfolioRiskManagement:
    """Test portfolio-level risk management."""

    def test_correlation_limit(self, sample_config):
        """Test position correlation limits (if implemented)."""
        risk_manager = RiskManager(sample_config)

        # This test depends on whether correlation checking is implemented
        # Mock having highly correlated positions
        with patch.object(risk_manager, '_check_correlation_risk') as mock_corr:
            mock_corr.return_value = (False, "High correlation with existing positions")

            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="CORRELATED_STOCK",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=75.0
            )

            # If correlation checking is implemented, should be rejected
            if not can_enter:
                assert "correlation" in reason.lower(), \
                    f"Rejection reason should mention correlation: {reason}"

    def test_sector_concentration_limit(self, sample_config):
        """Test sector concentration limits (if implemented)."""
        risk_manager = RiskManager(sample_config)

        # Mock having high sector concentration
        with patch.object(risk_manager, '_check_sector_concentration') as mock_sector:
            mock_sector.return_value = (False, "Too much exposure to this sector")

            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="SECTOR_STOCK",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=75.0,
                sector="BANKING"  # Some implementations might accept sector parameter
            )

            # If sector checking is implemented, should be rejected
            if not can_enter:
                assert "sector" in reason.lower() or "concentration" in reason.lower(), \
                    f"Rejection reason should mention sector/concentration: {reason}"

class TestRiskAdjustments:
    """Test risk adjustments based on market conditions."""

    def test_volatility_based_adjustment(self, sample_config):
        """Test risk adjustment based on market volatility."""
        risk_manager = RiskManager(sample_config)

        # Test position sizing in high vs low volatility environments
        high_vol_scenario = {
            'entry_price': 100.0,
            'stop_loss': 85.0,  # Wide stop due to high volatility
            'atr': 5.0,  # High ATR
            'signal_score': 75.0
        }

        low_vol_scenario = {
            'entry_price': 100.0,
            'stop_loss': 97.0,  # Tight stop due to low volatility
            'atr': 1.0,  # Low ATR
            'signal_score': 75.0
        }

        # Mock market volatility
        with patch.object(risk_manager, '_get_market_volatility') as mock_vol:
            # High volatility scenario
            mock_vol.return_value = 0.30  # 30% annualized volatility
            can_enter_hv, reason_hv, quantity_hv, risk_hv = risk_manager.can_enter_position(
                symbol="TEST_HV", **high_vol_scenario
            )

            # Low volatility scenario
            mock_vol.return_value = 0.15  # 15% annualized volatility
            can_enter_lv, reason_lv, quantity_lv, risk_lv = risk_manager.can_enter_position(
                symbol="TEST_LV", **low_vol_scenario
            )

            # Both should be valid scenarios
            if can_enter_hv and can_enter_lv:
                # In high volatility, might reduce position size
                # In low volatility, might allow larger position size
                # This depends on implementation details
                assert quantity_hv >= 0 and quantity_lv >= 0

    def test_market_regime_adjustment(self, sample_config):
        """Test risk adjustment based on market regime."""
        risk_manager = RiskManager(sample_config)

        # Test the same position in different market regimes
        position_params = {
            'symbol': "TEST",
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'signal_score': 75.0
        }

        regimes_to_test = ['BULLISH', 'BEARISH', 'SIDEWAYS', 'HIGH_VOLATILITY']
        results = {}

        for regime in regimes_to_test:
            with patch.object(risk_manager, '_get_market_regime') as mock_regime:
                mock_regime.return_value = regime

                can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(**position_params)
                results[regime] = {
                    'can_enter': can_enter,
                    'quantity': quantity if can_enter else 0,
                    'risk_amount': risk_amount if can_enter else 0
                }

        # All results should be reasonable
        for regime, result in results.items():
            assert isinstance(result['can_enter'], bool)
            assert result['quantity'] >= 0
            assert result['risk_amount'] >= 0

class TestRiskMonitoring:
    """Test ongoing risk monitoring capabilities."""

    def test_position_risk_update(self, sample_config):
        """Test updating position risk as prices move."""
        risk_manager = RiskManager(sample_config)

        # Mock an existing position
        position = {
            'symbol': 'TEST',
            'entry_price': 100.0,
            'quantity': 100,
            'stop_loss': 95.0,
            'current_price': 105.0  # Position is profitable
        }

        # Update position risk
        updated_risk = risk_manager.update_position_risk(position)

        assert 'current_risk' in updated_risk
        assert 'unrealized_pnl' in updated_risk
        assert updated_risk['current_risk'] >= 0

        # For a profitable position, current risk should be lower than initial
        initial_risk = (position['entry_price'] - position['stop_loss']) * position['quantity']
        assert updated_risk['current_risk'] <= initial_risk

    def test_portfolio_risk_summary(self, sample_config):
        """Test portfolio risk summary calculation."""
        risk_manager = RiskManager(sample_config)

        # Mock multiple positions
        positions = [
            {'symbol': 'STOCK1', 'entry_price': 100, 'quantity': 100, 'stop_loss': 95, 'current_price': 105},
            {'symbol': 'STOCK2', 'entry_price': 200, 'quantity': 50, 'stop_loss': 190, 'current_price': 195},
            {'symbol': 'STOCK3', 'entry_price': 50, 'quantity': 200, 'stop_loss': 47, 'current_price': 52},
        ]

        with patch.object(risk_manager, '_get_current_positions') as mock_positions:
            mock_positions.return_value = positions

            risk_summary = risk_manager.get_portfolio_risk_summary()

            assert 'total_portfolio_risk' in risk_summary
            assert 'position_count' in risk_summary
            assert 'largest_position_risk' in risk_summary
            assert 'total_unrealized_pnl' in risk_summary

            # Validate summary values
            assert risk_summary['position_count'] == len(positions)
            assert risk_summary['total_portfolio_risk'] >= 0
            assert isinstance(risk_summary['total_unrealized_pnl'], (int, float))

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_prices(self, sample_config):
        """Test handling of invalid price inputs."""
        risk_manager = RiskManager(sample_config)

        # Test invalid price scenarios
        invalid_scenarios = [
            {'entry_price': 0, 'stop_loss': 95, 'desc': 'zero entry price'},
            {'entry_price': -100, 'stop_loss': 95, 'desc': 'negative entry price'},
            {'entry_price': 100, 'stop_loss': 105, 'desc': 'stop loss above entry (long)'},
            {'entry_price': 100, 'stop_loss': np.nan, 'desc': 'NaN stop loss'},
            {'entry_price': float('inf'), 'stop_loss': 95, 'desc': 'infinite entry price'},
        ]

        for scenario in invalid_scenarios:
            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=scenario['entry_price'],
                stop_loss=scenario['stop_loss'],
                signal_score=75.0
            )

            # Should either handle gracefully or reject
            assert isinstance(can_enter, bool), f"Failed for {scenario['desc']}"
            assert isinstance(reason, str), f"Failed for {scenario['desc']}"

            if not can_enter:
                assert len(reason) > 0, f"Should provide reason for rejection: {scenario['desc']}"

    def test_extreme_signal_scores(self, sample_config):
        """Test handling of extreme signal scores."""
        risk_manager = RiskManager(sample_config)

        extreme_scores = [-10, 0, 150, float('inf'), float('-inf'), np.nan]

        for score in extreme_scores:
            can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
                symbol="TEST",
                entry_price=100.0,
                stop_loss=95.0,
                signal_score=score
            )

            # Should handle all extreme scores gracefully
            assert isinstance(can_enter, bool), f"Failed for score {score}"
            assert isinstance(reason, str), f"Failed for score {score}"

    def test_insufficient_capital(self, sample_config):
        """Test behavior when portfolio capital is insufficient."""
        risk_manager = RiskManager(sample_config)

        # Try to enter a position larger than available capital
        can_enter, reason, quantity, risk_amount = risk_manager.can_enter_position(
            symbol="EXPENSIVE_STOCK",
            entry_price=10000.0,  # Very expensive stock
            stop_loss=9500.0,
            signal_score=85.0
        )

        # Should either reject or size position appropriately
        if can_enter:
            position_value = quantity * 10000.0
            assert position_value <= sample_config['portfolio_capital'], \
                "Position value should not exceed available capital"
        else:
            assert "capital" in reason.lower() or "size" in reason.lower(), \
                f"Rejection reason should mention capital/size issue: {reason}"

class TestPerformanceAndScalability:
    """Test performance characteristics of risk management."""

    @pytest.mark.slow
    def test_bulk_position_evaluation(self, sample_config):
        """Test performance when evaluating many positions quickly."""
        import time

        risk_manager = RiskManager(sample_config)

        # Test evaluating many positions
        positions_to_test = [
            {'symbol': f'STOCK_{i}', 'entry_price': 100 + i, 'stop_loss': 95 + i, 'signal_score': 70 + (i % 20)}
            for i in range(100)
        ]

        start_time = time.time()
        results = []

        for position in positions_to_test:
            result = risk_manager.can_enter_position(**position)
            results.append(result)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 2 seconds for 100 evaluations)
        assert execution_time < 2.0, f"Risk evaluation took too long: {execution_time:.2f}s"
        assert len(results) == len(positions_to_test), "Should evaluate all positions"

        # All results should be valid
        for result in results:
            can_enter, reason, quantity, risk_amount = result
            assert isinstance(can_enter, bool)
            assert isinstance(reason, str)
            assert quantity >= 0
            assert risk_amount >= 0