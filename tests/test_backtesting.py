"""
Unit tests for backtesting system.
This module tests backtest logic, transaction costs, P&L calculations,
and various backtesting scenarios with controlled data.
"""
from datetime import datetime, timedelta
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
    from advanced_backtester import AdvancedBacktester, Trade, BacktestResults
except ImportError:
    # Fallback for when imports are not available
    pytest.skip("Required modules not available", allow_module_level=True)
class TestBacktesterInitialization:
    """Test AdvancedBacktester initialization and configuration."""
    def test_backtester_initialization(self, sample_config):
        """Test that AdvancedBacktester initializes correctly."""
        backtester = AdvancedBacktester(sample_config)
        assert backtester.config is not None
        assert hasattr(backtester, 'initial_capital')
        assert hasattr(backtester, 'transaction_cost')
        assert hasattr(backtester, 'slippage')
        assert backtester.initial_capital > 0
    def test_config_validation(self, sample_config):
        """Test config validation during initialization."""
        # Test with valid config
        backtester = AdvancedBacktester(sample_config)
        assert backtester.transaction_cost >= 0
        assert backtester.slippage >= 0
        # Test with extreme values
        extreme_config = sample_config.copy()
        extreme_config['transaction_cost'] = 0.1  # 10% transaction cost
        backtester = AdvancedBacktester(extreme_config)
        # Should initialize but might warn about extreme values
        assert backtester.transaction_cost == 0.1
class TestTradeExecution:
    """Test trade execution logic."""
    def test_long_trade_execution(self, sample_config, backtest_test_data):
        """Test execution of a long trade."""
        backtester = AdvancedBacktester(sample_config)
        # Execute a long trade
        entry_date = backtest_test_data.index[20]  # Day 20 - start of uptrend
        entry_price = backtest_test_data.loc[entry_date, 'Close']
        stop_loss = entry_price * 0.95  # 5% stop
        target = entry_price * 1.10  # 10% target
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=100,
            direction="long",
            stop_loss=stop_loss,
            target=target,
            data=backtest_test_data
        )
        assert trade is not None
        assert trade.symbol == "TEST"
        assert trade.entry_date == entry_date
        assert trade.entry_price == entry_price
        assert trade.quantity == 100
        assert trade.direction == "long"
        assert trade.stop_loss == stop_loss
        assert trade.target == target
    def test_short_trade_execution(self, sample_config, backtest_test_data):
        """Test execution of a short trade."""
        backtester = AdvancedBacktester(sample_config)
        # Execute a short trade during downtrend
        entry_date = backtest_test_data.index[60]  # Day 60 - start of downtrend
        entry_price = backtest_test_data.loc[entry_date, 'Close']
        stop_loss = entry_price * 1.05  # 5% stop (above entry for short)
        target = entry_price * 0.90  # 10% target (below entry for short)
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=100,
            direction="short",
            stop_loss=stop_loss,
            target=target,
            data=backtest_test_data
        )
        assert trade is not None
        assert trade.direction == "short"
        assert trade.stop_loss > trade.entry_price  # Stop above entry for short
        assert trade.target < trade.entry_price  # Target below entry for short
    def test_trade_exit_conditions(self, sample_config, backtest_test_data):
        """Test different trade exit conditions."""
        backtester = AdvancedBacktester(sample_config)
        # Test profit target hit
        entry_date = backtest_test_data.index[20]
        entry_price = backtest_test_data.loc[entry_date, 'Close']
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=100,
            direction="long",
            stop_loss=entry_price * 0.90,
            target=entry_price * 1.05,  # Conservative target
            data=backtest_test_data
        )
        # During uptrend, should hit target
        if trade.exit_date is not None:
            assert trade.exit_price > trade.entry_price  # Profitable exit
            assert abs(trade.pnl - (trade.exit_price - trade.entry_price) * trade.quantity) < 1.0
    def test_stop_loss_hit(self, sample_config, backtest_test_data):
        """Test stop loss execution."""
        backtester = AdvancedBacktester(sample_config)
        # Enter during downtrend with tight stop
        entry_date = backtest_test_data.index[65]  # During downtrend
        entry_price = backtest_test_data.loc[entry_date, 'Close']
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=100,
            direction="long",
            stop_loss=entry_price * 0.98,  # Very tight stop
            target=entry_price * 1.20,  # Ambitious target
            data=backtest_test_data
        )
        # Should hit stop loss during downtrend
        if trade.exit_date is not None and trade.exit_reason == "stop_loss":
            assert trade.exit_price < trade.entry_price  # Loss
            assert trade.pnl < 0  # Negative P&L
class TestTransactionCosts:
    """Test transaction cost calculations."""
    def test_commission_calculation(self, sample_config):
        """Test commission calculation on trades."""
        backtester = AdvancedBacktester(sample_config)
        entry_price = 100.0
        quantity = 100
        trade_value = entry_price * quantity
        # Calculate commission for entry
        entry_commission = backtester.calculate_commission(trade_value, "entry")
        expected_commission = trade_value * sample_config['transaction_cost']
        assert abs(entry_commission - expected_commission) < 0.01, \
            f"Entry commission {entry_commission} should equal {expected_commission}"
        # Calculate commission for exit
        exit_commission = backtester.calculate_commission(trade_value, "exit")
        assert abs(exit_commission - expected_commission) < 0.01, \
            f"Exit commission {exit_commission} should equal {expected_commission}"
    def test_slippage_calculation(self, sample_config):
        """Test slippage calculation."""
        backtester = AdvancedBacktester(sample_config)
        expected_price = 100.0
        quantity = 100
        # Test entry slippage (should increase cost for long)
        actual_entry_price = backtester.apply_slippage(expected_price, quantity, "entry", "long")
        expected_slippage = expected_price * sample_config['slippage']
        assert actual_entry_price >= expected_price, "Entry slippage should increase cost for long"
        assert abs(actual_entry_price - (expected_price + expected_slippage)) < 0.01
        # Test exit slippage (should decrease proceeds for long)
        actual_exit_price = backtester.apply_slippage(expected_price, quantity, "exit", "long")
        assert actual_exit_price <= expected_price, "Exit slippage should decrease proceeds for long"
    def test_total_trade_costs(self, sample_config, backtest_test_data):
        """Test total trade cost calculation."""
        backtester = AdvancedBacktester(sample_config)
        # Execute a complete trade
        entry_date = backtest_test_data.index[20]
        entry_price = backtest_test_data.loc[entry_date, 'Close']
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=100,
            direction="long",
            stop_loss=entry_price * 0.95,
            target=entry_price * 1.10,
            data=backtest_test_data
        )
        if trade.exit_date is not None:
            # Calculate expected total costs
            entry_value = trade.entry_price * trade.quantity
            exit_value = trade.exit_price * trade.quantity
            expected_costs = (
                entry_value * sample_config['transaction_cost'] +  # Entry commission
                exit_value * sample_config['transaction_cost'] +   # Exit commission
                entry_value * sample_config['slippage'] +          # Entry slippage
                exit_value * sample_config['slippage']             # Exit slippage
            )
            # Total costs should be reasonable relative to trade size
            assert trade.total_costs > 0, "Total costs should be positive"
            assert trade.total_costs <= expected_costs * 1.1, "Costs should be within expected range"
class TestBacktestMetrics:
    """Test backtest performance metrics calculation."""
    def test_returns_calculation(self, sample_config, backtest_test_data):
        """Test returns calculation for completed trades."""
        backtester = AdvancedBacktester(sample_config)
        # Execute multiple trades
        trades = []
        # Profitable trade
        profitable_trade = Trade(
            symbol="TEST1",
            entry_date=backtest_test_data.index[20],
            entry_price=100.0,
            quantity=100,
            direction="long",
            stop_loss=95.0,
            target=110.0
        )
        profitable_trade.exit_date = backtest_test_data.index[25]
        profitable_trade.exit_price = 108.0
        profitable_trade.exit_reason = "target"
        profitable_trade.calculate_pnl()
        trades.append(profitable_trade)
        # Loss trade
        loss_trade = Trade(
            symbol="TEST2",
            entry_date=backtest_test_data.index[60],
            entry_price=120.0,
            quantity=100,
            direction="long",
            stop_loss=114.0,
            target=130.0
        )
        loss_trade.exit_date = backtest_test_data.index[65]
        loss_trade.exit_price = 115.0
        loss_trade.exit_reason = "stop_loss"
        loss_trade.calculate_pnl()
        trades.append(loss_trade)
        # Calculate metrics
        metrics = backtester.calculate_performance_metrics(trades)
        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics
        # Validate win rate
        expected_win_rate = 1.0 / 2.0  # 1 win out of 2 trades
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01
    def test_drawdown_calculation(self, sample_config):
        """Test maximum drawdown calculation."""
        backtester = AdvancedBacktester(sample_config)
        # Create equity curve with known drawdown
        equity_curve = pd.Series([
            100000, 105000, 110000, 108000, 112000,  # Peak at 112000
            110000, 105000, 102000, 98000,  # Drawdown to 98000
            101000, 106000, 115000  # Recovery
        ])
        max_drawdown = backtester.calculate_max_drawdown(equity_curve)
        # Maximum drawdown should be from 112000 to 98000 = 14000 / 112000 = 12.5%
        expected_drawdown = (112000 - 98000) / 112000
        assert abs(max_drawdown - expected_drawdown) < 0.01, \
            f"Max drawdown {max_drawdown} should be approximately {expected_drawdown}"
    def test_sharpe_ratio_calculation(self, sample_config):
        """Test Sharpe ratio calculation."""
        backtester = AdvancedBacktester(sample_config)
        # Create return series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005, 0.02, -0.005, 0.01])
        sharpe_ratio = backtester.calculate_sharpe_ratio(returns, risk_free_rate=0.03)
        # Sharpe ratio should be a reasonable number
        assert isinstance(sharpe_ratio, (int, float))
        assert not np.isnan(sharpe_ratio), "Sharpe ratio should not be NaN"
        # For positive average excess returns, Sharpe should be positive
        if returns.mean() > 0.03 / 252:  # Risk-free rate adjusted for daily returns
            assert sharpe_ratio > 0, "Sharpe ratio should be positive for outperforming returns"
class TestWalkForwardAnalysis:
    """Test walk-forward analysis functionality."""
    def test_walk_forward_split(self, sample_config, backtest_test_data):
        """Test walk-forward data splitting."""
        backtester = AdvancedBacktester(sample_config)
        # Split data into training and testing periods
        train_size = 60  # 60 days training
        test_size = 20   # 20 days testing
        splits = backtester.create_walk_forward_splits(
            data=backtest_test_data,
            train_size=train_size,
            test_size=test_size,
            step_size=10
        )
        assert len(splits) > 0, "Should create at least one split"
        for i, (train_data, test_data) in enumerate(splits):
            assert len(train_data) == train_size, f"Training set {i} should have {train_size} periods"
            assert len(test_data) <= test_size, f"Testing set {i} should have at most {test_size} periods"
            # Training data should come before testing data
            assert train_data.index[-1] < test_data.index[0], \
                f"Training data should end before testing data starts in split {i}"
    def test_walk_forward_backtest(self, sample_config, backtest_test_data):
        """Test walk-forward backtesting."""
        backtester = AdvancedBacktester(sample_config)
        # Mock strategy that generates signals
        def mock_strategy(data):
            """Simple momentum strategy for testing."""
            signals = []
            if len(data) >= 20:
                ma20 = data['Close'].rolling(20).mean()
                current_price = data['Close'].iloc[-1]
                ma_current = ma20.iloc[-1]
                if current_price > ma_current * 1.02:  # 2% above MA
                    signals.append({
                        'symbol': 'TEST',
                        'entry_price': current_price,
                        'quantity': 100,
                        'direction': 'long',
                        'stop_loss': current_price * 0.95,
                        'target': current_price * 1.10
                    })
            return signals
        # Run walk-forward backtest
        results = backtester.run_walk_forward_backtest(
            data=backtest_test_data,
            strategy_func=mock_strategy,
            train_size=40,
            test_size=15,
            step_size=10
        )
        assert isinstance($1, list), "Results should be a list of backtest results"
        for result in results:
            assert 'period_start' in result
            assert 'period_end' in result
            assert 'trades' in result
            assert 'metrics' in result
class TestRiskManagementIntegration:
    """Test integration with risk management during backtesting."""
    def test_position_sizing_in_backtest(self, sample_config, backtest_test_data):
        """Test that position sizing is applied correctly during backtesting."""
        backtester = AdvancedBacktester(sample_config)
        # Mock risk manager
        mock_risk_manager = MagicMock()
        mock_risk_manager.can_enter_position.return_value = (True, "Approved", 150, 750)
        with patch.object(backtester, 'risk_manager', mock_risk_manager):
            trade = backtester.execute_trade(
                symbol="TEST",
                entry_date=backtest_test_data.index[20],
                entry_price=100.0,
                quantity=100,  # Requested quantity
                direction="long",
                stop_loss=95.0,
                target=110.0,
                data=backtest_test_data
            )
            # Should use risk manager's approved quantity
            assert trade.quantity == 150, f"Should use risk manager quantity {150}, got {trade.quantity}"
    def test_portfolio_risk_limits(self, sample_config, backtest_test_data):
        """Test portfolio risk limits during backtesting."""
        backtester = AdvancedBacktester(sample_config)
        # Mock scenario where risk limit is reached
        mock_risk_manager = MagicMock()
        mock_risk_manager.can_enter_position.return_value = (False, "Maximum risk reached", 0, 0)
        with patch.object(backtester, 'risk_manager', mock_risk_manager):
            trade = backtester.execute_trade(
                symbol="TEST",
                entry_date=backtest_test_data.index[20],
                entry_price=100.0,
                quantity=100,
                direction="long",
                stop_loss=95.0,
                target=110.0,
                data=backtest_test_data
            )
            # Trade should be rejected
            assert trade is None, "Trade should be None when risk manager rejects"
class TestBacktestReporting:
    """Test backtest reporting and output generation."""
    def test_trade_log_generation(self, sample_config, backtest_test_data):
        """Test generation of detailed trade log."""
        backtester = AdvancedBacktester(sample_config)
        # Create sample trades
        trades = []
        for i in range(5):
            trade = Trade(
                symbol=f"TEST{i}",
                entry_date=backtest_test_data.index[20 + i * 10],
                entry_price=100.0 + i,
                quantity=100,
                direction="long",
                stop_loss=95.0 + i,
                target=110.0 + i
            )
            trade.exit_date = backtest_test_data.index[25 + i * 10]
            trade.exit_price = 105.0 + i
            trade.exit_reason = "target" if i % 2 == 0 else "stop_loss"
            trade.calculate_pnl()
            trades.append(trade)
        # Generate trade log
        trade_log = backtester.generate_trade_log(trades)
        assert isinstance(trade_log, pd.DataFrame), "Trade log should be a DataFrame"
        assert len(trade_log) == len(trades), "Trade log should have entry for each trade"
        expected_columns = ['symbol', 'entry_date', 'entry_price', 'exit_date',
                          'exit_price', 'quantity', 'direction', 'pnl', 'exit_reason']
        for col in expected_columns:
            assert col in trade_log.columns, f"Missing column: {col}"
    def test_equity_curve_generation(self, sample_config, backtest_test_data):
        """Test equity curve generation."""
        backtester = AdvancedBacktester(sample_config)
        # Create sample trades with known P&L
        trades = [
            {'date': backtest_test_data.index[20], 'pnl': 500},
            {'date': backtest_test_data.index[30], 'pnl': -200},
            {'date': backtest_test_data.index[40], 'pnl': 800},
            {'date': backtest_test_data.index[50], 'pnl': -100},
        ]
        equity_curve = backtester.generate_equity_curve(trades, initial_capital=100000)
        assert isinstance(equity_curve, pd.Series), "Equity curve should be a Series"
        assert equity_curve.iloc[0] == 100000, "Should start with initial capital"
        # Check cumulative calculation
        expected_final = 100000 + 500 - 200 + 800 - 100
        assert equity_curve.iloc[-1] == expected_final, \
            f"Final equity {equity_curve.iloc[-1]} should equal {expected_final}"
    def test_performance_summary(self, sample_config):
        """Test performance summary generation."""
        backtester = AdvancedBacktester(sample_config)
        # Mock trades with known outcomes
        winning_trade = MagicMock()
        winning_trade.pnl = 1000
        winning_trade.direction = "long"
        winning_trade.exit_reason = "target"
        losing_trade = MagicMock()
        losing_trade.pnl = -500
        losing_trade.direction = "long"
        losing_trade.exit_reason = "stop_loss"
        trades = [winning_trade, losing_trade]
        summary = backtester.generate_performance_summary(trades, initial_capital=100000)
        assert 'total_trades' in summary
        assert 'winning_trades' in summary
        assert 'losing_trades' in summary
        assert 'win_rate' in summary
        assert 'total_pnl' in summary
        assert 'final_capital' in summary
        # Validate calculations
        assert summary['total_trades'] == 2
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == 0.5
        assert summary['total_pnl'] == 500  # 1000 - 500
        assert summary['final_capital'] == 100500  # 100000 + 500
class TestErrorHandling:
    """Test error handling in backtesting."""
    def test_invalid_data_handling(self, sample_config):
        """Test handling of invalid or insufficient data."""
        backtester = AdvancedBacktester(sample_config)
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=datetime.now(),
            entry_price=100.0,
            quantity=100,
            direction="long",
            stop_loss=95.0,
            target=110.0,
            data=empty_data
        )
        assert trade is None, "Should return None for empty data"
    def test_missing_price_data(self, sample_config, backtest_test_data):
        """Test handling of missing price data during trade execution."""
        backtester = AdvancedBacktester(sample_config)
        # Create data with NaN values
        corrupted_data = backtest_test_data.copy()
        corrupted_data.loc[corrupted_data.index[25:35], 'Close'] = np.nan
        trade = backtester.execute_trade(
            symbol="TEST",
            entry_date=corrupted_data.index[20],
            entry_price=100.0,
            quantity=100,
            direction="long",
            stop_loss=95.0,
            target=110.0,
            data=corrupted_data
        )
        # Should handle gracefully - either complete the trade or exit appropriately
        if trade is not None:
            assert not np.isnan(trade.entry_price), "Entry price should not be NaN"
    def test_extreme_market_conditions(self, sample_config):
        """Test backtesting in extreme market conditions."""
        backtester = AdvancedBacktester(sample_config)
        # Create data with extreme volatility
        dates = pd.date_range('2023-01-01', periods=50)
        extreme_data = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100 + abs(np.random.normal(0, 20)) for _ in range(50)],
            'Low': [100 - abs(np.random.normal(0, 20)) for _ in range(50)],
            'Close': [100 + np.random.normal(0, 15) for _ in range(50)],
            'Volume': [100000] * 50
        }, index=dates)
        # Ensure realistic price relationships
        for i in range(len(extreme_data)):
            high = extreme_data.iloc[i]['High']
            low = extreme_data.iloc[i]['Low']
            close = extreme_data.iloc[i]['Close']
            # Adjust to ensure High >= Close >= Low
            extreme_data.iloc[i, extreme_data.columns.get_loc('High')] = max(high, close)
            extreme_data.iloc[i, extreme_data.columns.get_loc('Low')] = min(low, close)
        trade = backtester.execute_trade(
            symbol="EXTREME",
            entry_date=extreme_data.index[10],
            entry_price=extreme_data.loc[extreme_data.index[10], 'Close'],
            quantity=100,
            direction="long",
            stop_loss=extreme_data.loc[extreme_data.index[10], 'Close'] * 0.90,
            target=extreme_data.loc[extreme_data.index[10], 'Close'] * 1.20,
            data=extreme_data
        )
        # Should handle extreme conditions gracefully
        if trade is not None and trade.exit_date is not None:
            assert not np.isnan(trade.pnl), "P&L should be calculable even in extreme conditions"
class TestPerformanceOptimization:
    """Test performance optimization features."""
    @pytest.mark.slow
    def test_large_dataset_backtest(self, sample_config):
        """Test backtesting performance with large datasets."""
        import time
        backtester = AdvancedBacktester(sample_config)
        # Create large dataset (2 years of daily data)
        dates = pd.date_range('2023-01-01', periods=500)
        large_data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 500),
            'High': np.random.normal(105, 5, 500),
            'Low': np.random.normal(95, 5, 500),
            'Close': np.random.normal(100, 5, 500),
            'Volume': np.random.normal(100000, 20000, 500)
        }, index=dates)
        # Ensure realistic OHLC relationships
        for i in range(len(large_data)):
            o, h, l, c = large_data.iloc[i][['Open', 'High', 'Low', 'Close']]
            large_data.iloc[i, large_data.columns.get_loc('High')] = max(o, h, l, c)
            large_data.iloc[i, large_data.columns.get_loc('Low')] = min(o, h, l, c)
        start_time = time.time()
        # Execute multiple trades
        trades = []
        for i in range(0, 400, 50):  # Execute trade every 50 days
            trade = backtester.execute_trade(
                symbol=f"STOCK_{i}",
                entry_date=large_data.index[i],
                entry_price=large_data.iloc[i]['Close'],
                quantity=100,
                direction="long",
                stop_loss=large_data.iloc[i]['Close'] * 0.95,
                target=large_data.iloc[i]['Close'] * 1.10,
                data=large_data.iloc[i:i+50]  # Limited data window
            )
            if trade:
                trades.append(trade)
        end_time = time.time()
        execution_time = end_time - start_time
        # Should complete in reasonable time
        assert execution_time < 10.0, f"Large dataset backtest took too long: {execution_time:.2f}s"
        assert len(trades) > 0, "Should complete at least some trades"
