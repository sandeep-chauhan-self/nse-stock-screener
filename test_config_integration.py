import logging
"""
Configuration Integration Tests
Tests for the centralized configuration system to ensure all modules
work correctly with the new SystemConfig approach.
"""

from datetime import datetime
from pathlib import Path
import json
import os

import tempfile
import unittest

from src.config import SystemConfig, ConfigManager, get_config, set_config, load_config_from_environment

class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig class functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear any environment variables
        env_vars_to_clear = [f"NSE_{field.upper()}" for field in SystemConfig.__dataclass_fields__]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = SystemConfig()
        
        # Test key defaults
        self.assertEqual(config.portfolio_capital, 1000000.0)
        self.assertEqual(config.max_positions, 10)
        self.assertEqual(config.risk_per_trade, 0.01)
        self.assertEqual(config.max_position_size, 0.10)
        self.assertEqual(config.transaction_cost, 0.0005)
        self.assertEqual(config.slippage, 0.0005)
        
        # Test validation passes
        config.validate()
    
    def test_configuration_validation(self):
        """Test configuration validation logic"""
        # Test valid config
        config = SystemConfig()
        config.validate()  # Should not raise
        
        # Test invalid configurations
        with self.assertRaises(ValueError):
            config = SystemConfig(portfolio_capital=-1000)
            config.validate()
        
        with self.assertRaises(ValueError):
            config = SystemConfig(risk_per_trade=0.5)  # 50% risk too high
            config.validate()
        
        with self.assertRaises(ValueError):
            config = SystemConfig(max_position_size=0.8)  # 80% position size too high
            config.validate()
        
        with self.assertRaises(ValueError):
            config = SystemConfig(max_positions=-5)
            config.validate()
    
    def test_logical_consistency_validation(self):
        """Test logical consistency between parameters"""
        # Test case where max_position_size * max_positions > 200% (extreme)
        with self.assertRaises(ValueError):
            config = SystemConfig(max_position_size=0.5, max_positions=5)  # 250% theoretical
            config.validate()
        
        # Test case where risk_per_trade > max_position_size
        with self.assertRaises(ValueError):
            config = SystemConfig(risk_per_trade=0.15, max_position_size=0.10)
            config.validate()
    
    def test_environment_override(self):
        """Test environment variable override functionality"""
        # Set environment variables
        os.environ["NSE_PORTFOLIO_CAPITAL"] = "2000000"
        os.environ["NSE_MAX_POSITIONS"] = "15"
        os.environ["NSE_RISK_PER_TRADE"] = "0.02"
        
        config = SystemConfig.from_environment()
        
        self.assertEqual(config.portfolio_capital, 2000000.0)
        self.assertEqual(config.max_positions, 15)
        self.assertEqual(config.risk_per_trade, 0.02)
    
    def test_json_serialization(self):
        """Test JSON export and import"""
        config = SystemConfig(portfolio_capital=1500000, max_positions=8)  # Use valid combination
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['portfolio_capital'], 1500000)
        self.assertEqual(config_dict['max_positions'], 8)
        
        # Test from_dict
        new_config = SystemConfig.from_dict(config_dict)
        self.assertEqual(new_config.portfolio_capital, 1500000)
        self.assertEqual(new_config.max_positions, 8)
        
        # Test JSON round trip
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            config.to_json(temp_file)
            loaded_config = SystemConfig.from_json(temp_file)
            
            self.assertEqual(loaded_config.portfolio_capital, 1500000)
            self.assertEqual(loaded_config.max_positions, 8)
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows

class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality"""
    
    def test_config_manager_basic(self):
        """Test basic ConfigManager operations"""
        config = SystemConfig(portfolio_capital=800000)
        manager = ConfigManager(config)
        
        self.assertEqual(manager.config.portfolio_capital, 800000)
    
    def test_config_manager_update(self):
        """Test configuration updates"""
        manager = ConfigManager()
        
        # Test valid update
        manager.update_config(portfolio_capital=2000000, max_positions=8)  # Use valid combination
        self.assertEqual(manager.config.portfolio_capital, 2000000)
        self.assertEqual(manager.config.max_positions, 8)
        
        # Test invalid field
        with self.assertRaises(ValueError):
            manager.update_config(invalid_field=123)
    
    def test_config_sections(self):
        """Test configuration section getters"""
        manager = ConfigManager()
        
        # Test risk settings
        risk_settings = manager.get_risk_settings()
        self.assertIn('risk_per_trade', risk_settings)
        self.assertIn('max_portfolio_risk', risk_settings)
        self.assertIn('max_position_size', risk_settings)
        
        # Test backtest settings
        backtest_settings = manager.get_backtest_settings()
        self.assertIn('initial_capital', backtest_settings)
        self.assertIn('transaction_cost', backtest_settings)
        self.assertIn('slippage', backtest_settings)
        
        # Test data settings
        data_settings = manager.get_data_settings()
        self.assertIn('batch_size', data_settings)
        self.assertIn('request_timeout', data_settings)

class TestModuleIntegration(unittest.TestCase):
    """Test integration with refactored modules"""
    
    def test_risk_manager_integration(self):
        """Test RiskManager with centralized config"""
        try:
            # This will fail if imports are broken
            from src.risk_manager import RiskManager
            from src.config import SystemConfig
            
            config = SystemConfig(
                portfolio_capital=500000,
                max_positions=5,
                risk_per_trade=0.015
            )
            
            risk_manager = RiskManager(config.portfolio_capital, config)
            
            # Test that configuration is properly used
            self.assertEqual(risk_manager.config.portfolio_capital, 500000)
            self.assertEqual(risk_manager.config.max_positions, 5)
            self.assertEqual(risk_manager.config.risk_per_trade, 0.015)
            
            # Test risk limits check
            limits = risk_manager.check_portfolio_risk_limits()
            self.assertIn('max_positions', limits)
            self.assertEqual(limits['max_positions'], 5)
            
        except ImportError as e:
            self.fail(f"Failed to import RiskManager: {e}")
    
    def test_backtester_integration(self):
        """Test AdvancedBacktester with centralized config"""
        try:
            from src.advanced_backtester import AdvancedBacktester
            
            config = SystemConfig(
                portfolio_capital=750000,
                risk_per_trade=0.02,
                max_position_size=0.08,  # 8% position size with 10 positions = 80% max
                transaction_cost=0.001
            )
            
            backtester = AdvancedBacktester(config)
            
            # Test that configuration is properly used
            self.assertEqual(backtester.config.portfolio_capital, 750000)
            self.assertEqual(backtester.config.risk_per_trade, 0.02)
            self.assertEqual(backtester.config.max_position_size, 0.08)
            self.assertEqual(backtester.config.transaction_cost, 0.001)
            
        except ImportError as e:
            self.fail(f"Failed to import AdvancedBacktester: {e}")

class TestConfigurationConsistency(unittest.TestCase):
    """Test that configuration naming is consistent across modules"""
    
    def test_canonical_field_names(self):
        """Ensure all modules use the same canonical field names"""
        config = SystemConfig()
        
        # Test that we have canonical names (not the old inconsistent ones)
        self.assertTrue(hasattr(config, 'max_positions'))  # Not max_concurrent_positions
        self.assertTrue(hasattr(config, 'portfolio_capital'))  # Not initial_capital
        self.assertTrue(hasattr(config, 'risk_per_trade'))
        self.assertTrue(hasattr(config, 'max_position_size'))
        self.assertTrue(hasattr(config, 'transaction_cost'))
        self.assertTrue(hasattr(config, 'slippage'))
        
        # Ensure old inconsistent names don't exist
        self.assertFalse(hasattr(config, 'max_concurrent_positions'))
        self.assertFalse(hasattr(config, 'initial_capital'))

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Configuration Integration Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSystemConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'✅' if result.wasSuccessful() else '❌'} Tests Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    logging.error(f"   Errors: {len(result.errors)}")
    
    if not result.wasSuccessful():
        print("\n❌ Test Failures:")
        for test, traceback in result.failures + result.errors:
            logging.error(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)