"""
Walk-Forward Analysis and Bias Avoidance Module
===============================================

This module implements sophisticated bias avoidance mechanisms for backtesting:
- Walk-forward analysis with train/test/validation splits
- Rolling rebalancing windows
- Strict lookahead prevention
- Out-of-sample testing
- Cross-validation for robustness

Ensures backtest results are statistically sound and free from common biases.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any, Iterator
import warnings
from copy import deepcopy

logger = logging.getLogger(__name__)

# =====================================================================================
# DATA STRUCTURES AND ENUMS
# =====================================================================================

class SplitType(Enum):
    """Types of data splits for bias avoidance"""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    OUT_OF_SAMPLE = "out_of_sample"

class RebalanceFrequency(Enum):
    """Rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class DataSplit:
    """Represents a data split with metadata"""
    split_type: SplitType
    start_date: datetime
    end_date: datetime
    split_id: str

    # Optional metadata
    train_window_days: Optional[int] = None
    test_window_days: Optional[int] = None
    validation_window_days: Optional[int] = None

    @property
    def duration_days(self) -> int:
        """Calculate duration in days"""
        return (self.end_date - self.start_date).days

    def contains_date(self, date: datetime) -> bool:
        """Check if date is within this split"""
        return self.start_date <= date <= self.end_date

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""

    # Window sizes (in trading days)
    train_window_days: int = 252  # 1 year training
    test_window_days: int = 63    # 3 months testing
    validation_window_days: int = 21  # 1 month validation

    # Gap between windows to prevent lookahead
    gap_days: int = 1

    # Minimum data requirements
    min_train_window_days: int = 126  # 6 months minimum
    min_test_window_days: int = 21    # 1 month minimum

    # Rebalancing parameters
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    force_rebalance_on_split: bool = True

    # Out-of-sample testing
    out_of_sample_ratio: float = 0.20  # 20% for final validation
    enable_out_of_sample: bool = True

    # Cross-validation parameters
    enable_cross_validation: bool = False
    cv_folds: int = 5
    cv_overlap_allowed: bool = False

    # Lookahead prevention
    strict_lookahead_prevention: bool = True
    max_future_data_days: int = 0  # No future data allowed

    def validate(self) -> None:
        """Validate configuration parameters"""
        errors = []

        if self.train_window_days < self.min_train_window_days:
            errors.append(f"train_window_days ({self.train_window_days}) "
                         f"< min_train_window_days ({self.min_train_window_days})")

        if self.test_window_days < self.min_test_window_days:
            errors.append(f"test_window_days ({self.test_window_days}) "
                         f"< min_test_window_days ({self.min_test_window_days})")

        if not 0 <= self.out_of_sample_ratio <= 0.5:
            errors.append("out_of_sample_ratio must be between 0 and 0.5")

        if self.cv_folds < 2:
            errors.append("cv_folds must be >= 2")

        if self.gap_days < 0:
            errors.append("gap_days must be >= 0")

        if errors:
            raise ValueError("WalkForwardConfig validation failed:\n" +
                           "\n".join(f"  - {e}" for e in errors))

@dataclass
class BiasCheck:
    """Results of bias checking"""
    check_name: str
    passed: bool
    message: str
    severity: str = "warning"  # "info", "warning", "error"
    recommendation: str = ""

# =====================================================================================
# LOOKAHEAD PREVENTION
# =====================================================================================

class LookaheadDetector:
    """
    Detects potential lookahead bias in backtesting data and signals
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.violations: List[Dict[str, Any]] = []

    def check_data_timestamps(self, data: pd.DataFrame,
                            signal_date: datetime) -> List[BiasCheck]:
        """
        Check if data contains future information relative to signal date

        Args:
            data: DataFrame with datetime index
            signal_date: Date when signal was generated

        Returns:
            List of bias check results
        """
        checks = []

        if not isinstance(data.index, pd.DatetimeIndex):
            checks.append(BiasCheck(
                "timestamp_format",
                False,
                "Data index is not DatetimeIndex - cannot verify timestamps",
                "error",
                "Ensure data has proper datetime index"
            ))
            return checks

        # Check for future data
        future_data = data[data.index > signal_date]
        if len(future_data) > 0:
            latest_future = future_data.index.max()
            days_ahead = (latest_future - signal_date).days

            if days_ahead > self.config.max_future_data_days:
                checks.append(BiasCheck(
                    "future_data_detected",
                    False,
                    f"Data contains {len(future_data)} rows with future timestamps, "
                    f"up to {days_ahead} days ahead of signal date {signal_date.date()}",
                    "error",
                    "Remove future data or adjust signal generation logic"
                ))
            else:
                checks.append(BiasCheck(
                    "future_data_within_tolerance",
                    True,
                    f"Future data found but within {self.config.max_future_data_days} day tolerance",
                    "info"
                ))
        else:
            checks.append(BiasCheck(
                "no_future_data",
                True,
                "No future data detected - timestamps are valid",
                "info"
            ))

        return checks

    def check_signal_consistency(self, signals: pd.DataFrame) -> List[BiasCheck]:
        """
        Check signal data for consistency and potential lookahead bias

        Args:
            signals: DataFrame with signal data and timestamps

        Returns:
            List of bias check results
        """
        checks = []

        if signals.empty:
            checks.append(BiasCheck(
                "empty_signals",
                False,
                "No signals provided for bias checking",
                "warning"
            ))
            return checks

        # Check for duplicate timestamps
        if 'timestamp' in signals.columns:
            duplicates = signals['timestamp'].duplicated().sum()
            if duplicates > 0:
                checks.append(BiasCheck(
                    "duplicate_timestamps",
                    False,
                    f"Found {duplicates} duplicate timestamps in signals",
                    "warning",
                    "Review signal generation logic for timestamp handling"
                ))

        # Check for chronological ordering
        if 'timestamp' in signals.columns:
            timestamps = pd.to_datetime(signals['timestamp'])
            if not timestamps.is_monotonic_increasing:
                checks.append(BiasCheck(
                    "non_chronological_signals",
                    False,
                    "Signals are not in chronological order",
                    "error",
                    "Sort signals by timestamp before backtesting"
                ))

        # Check for reasonable signal values
        numeric_columns = signals.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if signals[col].isna().all():
                checks.append(BiasCheck(
                    f"all_nan_column_{col}",
                    False,
                    f"Column '{col}' contains only NaN values",
                    "warning"
                ))
            elif (signals[col] == signals[col].iloc[0]).all():
                checks.append(BiasCheck(
                    f"constant_column_{col}",
                    False,
                    f"Column '{col}' has constant values - may indicate data issue",
                    "warning"
                ))

        return checks

    def validate_backtest_data(self, data: Dict[str, pd.DataFrame],
                             signals: pd.DataFrame) -> List[BiasCheck]:
        """
        Comprehensive validation of backtest data for bias prevention

        Args:
            data: Dictionary of symbol -> price data
            signals: DataFrame with signals

        Returns:
            List of all bias check results
        """
        all_checks = []

        # Check signals
        all_checks.extend(self.check_signal_consistency(signals))

        # Check each symbol's data
        for symbol, symbol_data in data.items():
            if symbol_data.empty:
                all_checks.append(BiasCheck(
                    f"empty_data_{symbol}",
                    False,
                    f"No data available for symbol {symbol}",
                    "error"
                ))
                continue

            # Check for missing required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in symbol_data.columns]
            if missing_columns:
                all_checks.append(BiasCheck(
                    f"missing_columns_{symbol}",
                    False,
                    f"Symbol {symbol} missing required columns: {missing_columns}",
                    "error"
                ))

            # Check for data consistency
            if len(symbol_data) > 1:
                # Check for non-decreasing timestamps
                if not symbol_data.index.is_monotonic_increasing:
                    all_checks.append(BiasCheck(
                        f"non_monotonic_timestamps_{symbol}",
                        False,
                        f"Symbol {symbol} has non-monotonic timestamps",
                        "error"
                    ))

                # Check for reasonable OHLC relationships
                invalid_ohlc = (
                    (symbol_data['High'] < symbol_data['Low']) |
                    (symbol_data['Open'] > symbol_data['High']) |
                    (symbol_data['Open'] < symbol_data['Low']) |
                    (symbol_data['Close'] > symbol_data['High']) |
                    (symbol_data['Close'] < symbol_data['Low'])
                )
                if invalid_ohlc.any():
                    all_checks.append(BiasCheck(
                        f"invalid_ohlc_{symbol}",
                        False,
                        f"Symbol {symbol} has {invalid_ohlc.sum()} rows with invalid OHLC relationships",
                        "error"
                    ))

        return all_checks

# =====================================================================================
# WALK-FORWARD ANALYSIS ENGINE
# =====================================================================================

class WalkForwardEngine:
    """
    Main engine for walk-forward analysis with bias avoidance
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward engine

        Args:
            config: Walk-forward configuration
        """
        config.validate()
        self.config = config
        self.lookahead_detector = LookaheadDetector(config)
        self.splits: List[DataSplit] = []
        self.bias_checks: List[BiasCheck] = []

        logger.info(f"Initialized WalkForwardEngine with {config.train_window_days} day training, "
                   f"{config.test_window_days} day testing windows")

    def create_walk_forward_splits(self, start_date: datetime,
                                 end_date: datetime) -> List[DataSplit]:
        """
        Create walk-forward splits for the given date range

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            List of DataSplit objects
        """
        splits = []

        # Reserve out-of-sample data if enabled
        if self.config.enable_out_of_sample:
            total_days = (end_date - start_date).days
            oos_days = int(total_days * self.config.out_of_sample_ratio)
            oos_start = end_date - timedelta(days=oos_days)

            # Create out-of-sample split
            splits.append(DataSplit(
                split_type=SplitType.OUT_OF_SAMPLE,
                start_date=oos_start,
                end_date=end_date,
                split_id=f"OOS_{oos_start.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            ))

            # Adjust end date for walk-forward splits
            end_date = oos_start - timedelta(days=self.config.gap_days)

        # Create walk-forward splits
        current_date = start_date
        split_counter = 1

        while current_date < end_date:
            # Calculate train window
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_window_days)

            if train_end > end_date:
                break  # Not enough data for complete train window

            # Calculate test window (after gap)
            test_start = train_end + timedelta(days=self.config.gap_days)
            test_end = test_start + timedelta(days=self.config.test_window_days)

            if test_end > end_date:
                break  # Not enough data for complete test window

            # Create train split
            train_split = DataSplit(
                split_type=SplitType.TRAIN,
                start_date=train_start,
                end_date=train_end,
                split_id=f"TRAIN_{split_counter}_{train_start.strftime('%Y%m%d')}",
                train_window_days=self.config.train_window_days
            )
            splits.append(train_split)

            # Create test split
            test_split = DataSplit(
                split_type=SplitType.TEST,
                start_date=test_start,
                end_date=test_end,
                split_id=f"TEST_{split_counter}_{test_start.strftime('%Y%m%d')}",
                test_window_days=self.config.test_window_days
            )
            splits.append(test_split)

            # Optional validation split
            if self.config.validation_window_days > 0:
                val_start = test_end + timedelta(days=self.config.gap_days)
                val_end = val_start + timedelta(days=self.config.validation_window_days)

                if val_end <= end_date:
                    val_split = DataSplit(
                        split_type=SplitType.VALIDATION,
                        start_date=val_start,
                        end_date=val_end,
                        split_id=f"VAL_{split_counter}_{val_start.strftime('%Y%m%d')}",
                        validation_window_days=self.config.validation_window_days
                    )
                    splits.append(val_split)

            # Move to next window (rolling forward by test window)
            current_date = test_start
            split_counter += 1

        self.splits = splits
        logger.info(f"Created {len(splits)} walk-forward splits")

        return splits

    def get_rebalance_dates(self, start_date: datetime,
                          end_date: datetime) -> List[datetime]:
        """
        Generate rebalancing dates based on frequency configuration

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of rebalancing dates
        """
        rebalance_dates = []

        # Generate dates based on frequency
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            rebalance_dates = self._generate_daily_dates(start_date, end_date)
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            rebalance_dates = self._generate_weekly_dates(start_date, end_date)
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            rebalance_dates = self._generate_monthly_dates(start_date, end_date)
        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            rebalance_dates = self._generate_quarterly_dates(start_date, end_date)

        # Add split boundaries if forced rebalancing is enabled
        if self.config.force_rebalance_on_split:
            split_dates = [
                split.start_date for split in self.splits
                if split.split_type in [SplitType.TEST, SplitType.VALIDATION]
            ]
            rebalance_dates.extend(split_dates)

        # Sort and deduplicate
        rebalance_dates = sorted(set(rebalance_dates))

        logger.info(f"Generated {len(rebalance_dates)} rebalancing dates "
                   f"({self.config.rebalance_frequency.value} frequency)")

        return rebalance_dates

    def _generate_daily_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate daily rebalancing dates"""
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        return dates

    def _generate_weekly_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate weekly rebalancing dates (Mondays)"""
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 0:  # Monday
                dates.append(current_date)
            current_date += timedelta(days=1)
        return dates

    def _generate_monthly_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate monthly rebalancing dates (first of month)"""
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.day == 1:
                dates.append(current_date)
            current_date += timedelta(days=1)
        return dates

    def _generate_quarterly_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate quarterly rebalancing dates"""
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.month in [1, 4, 7, 10] and current_date.day == 1:
                dates.append(current_date)
            current_date += timedelta(days=1)
        return dates

    def validate_data_for_bias(self, data: Dict[str, pd.DataFrame],
                             signals: pd.DataFrame) -> bool:
        """
        Run comprehensive bias validation on backtest data

        Args:
            data: Price data by symbol
            signals: Signal data

        Returns:
            True if data passes all critical bias checks
        """
        self.bias_checks = self.lookahead_detector.validate_backtest_data(data, signals)

        # Categorize results
        errors = [c for c in self.bias_checks if c.severity == "error" and not c.passed]
        warnings = [c for c in self.bias_checks if c.severity == "warning" and not c.passed]
        info = [c for c in self.bias_checks if c.severity == "info"]

        # Log results
        logger.info(f"Bias validation complete: {len(errors)} errors, "
                   f"{len(warnings)} warnings, {len(info)} info messages")

        for error in errors:
            logger.error(f"BIAS ERROR: {error.check_name}: {error.message}")

        for warning in warnings:
            logger.warning(f"BIAS WARNING: {warning.check_name}: {warning.message}")

        # Return True only if no critical errors
        return len(errors) == 0

    def get_split_data(self, data: pd.DataFrame, split: DataSplit) -> pd.DataFrame:
        """
        Extract data for a specific split with strict date filtering

        Args:
            data: Full dataset
            split: Data split definition

        Returns:
            Filtered dataset for the split
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for split filtering")

        # Apply strict date filtering
        mask = (data.index >= split.start_date) & (data.index <= split.end_date)
        split_data = data[mask].copy()

        # Additional validation
        if self.config.strict_lookahead_prevention:
            # Ensure no data leaks into the future
            future_mask = data.index > split.end_date
            if future_mask.any():
                logger.debug(f"Removed {future_mask.sum()} future data points "
                           f"beyond split end date {split.end_date}")

        logger.debug(f"Split {split.split_id}: extracted {len(split_data)} rows "
                    f"from {split.start_date.date()} to {split.end_date.date()}")

        return split_data

    def run_walk_forward_analysis(self,
                                backtest_function: Callable,
                                data: Dict[str, pd.DataFrame],
                                signals: pd.DataFrame,
                                **backtest_kwargs) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis

        Args:
            backtest_function: Function to run backtest on each split
            data: Price data by symbol
            signals: Signal data
            **backtest_kwargs: Additional arguments for backtest function

        Returns:
            Comprehensive walk-forward analysis results
        """
        # Validate data for bias
        if not self.validate_data_for_bias(data, signals):
            logger.error("Data failed bias validation - proceeding with warnings")

        # Ensure splits are created
        if not self.splits:
            logger.error("No splits created - call create_walk_forward_splits first")
            return {'error': 'No splits available'}

        results = {
            'splits': {},
            'summary': {
                'total_splits': len(self.splits),
                'train_splits': len([s for s in self.splits if s.split_type == SplitType.TRAIN]),
                'test_splits': len([s for s in self.splits if s.split_type == SplitType.TEST]),
                'validation_splits': len([s for s in self.splits if s.split_type == SplitType.VALIDATION]),
                'out_of_sample_splits': len([s for s in self.splits if s.split_type == SplitType.OUT_OF_SAMPLE])
            },
            'bias_checks': self.bias_checks,
            'performance_by_split_type': {},
            'config': self.config
        }

        # Run backtest on each split
        for split in self.splits:
            logger.info(f"Running backtest on split {split.split_id} "
                       f"({split.split_type.value}: {split.start_date.date()} to {split.end_date.date()})")

            try:
                # Prepare data for this split
                split_data = {}
                for symbol, symbol_data in data.items():
                    split_data[symbol] = self.get_split_data(symbol_data, split)

                # Filter signals for this split
                signal_mask = (
                    (pd.to_datetime(signals['timestamp']) >= split.start_date) &
                    (pd.to_datetime(signals['timestamp']) <= split.end_date)
                )
                split_signals = signals[signal_mask].copy()

                # Run backtest with bias-aware data
                split_result = backtest_function(
                    split_data,
                    split_signals,
                    split=split,
                    **backtest_kwargs
                )

                results['splits'][split.split_id] = {
                    'split_info': split,
                    'result': split_result,
                    'data_rows': sum(len(df) for df in split_data.values()),
                    'signal_count': len(split_signals)
                }

            except Exception as e:
                logger.error(f"Error running backtest on split {split.split_id}: {e}")
                results['splits'][split.split_id] = {
                    'split_info': split,
                    'error': str(e)
                }

        # Aggregate performance by split type
        for split_type in SplitType:
            split_results = [
                r['result'] for r in results['splits'].values()
                if r['split_info'].split_type == split_type and 'result' in r
            ]

            if split_results:
                # Calculate aggregate metrics (implementation depends on backtest_function output format)
                results['performance_by_split_type'][split_type.value] = {
                    'count': len(split_results),
                    'results': split_results
                }

        logger.info(f"Walk-forward analysis complete: processed {len(results['splits'])} splits")

        return results

# =====================================================================================
# CROSS-VALIDATION SUPPORT
# =====================================================================================

class TimeSeriesCrossValidator:
    """
    Time-series aware cross-validation to prevent data leakage
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def create_cv_splits(self, start_date: datetime, end_date: datetime) -> List[Tuple[DataSplit, DataSplit]]:
        """
        Create time-series cross-validation splits

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of (train_split, test_split) tuples
        """
        total_days = (end_date - start_date).days
        fold_size = total_days // self.config.cv_folds

        cv_splits = []

        for fold in range(self.config.cv_folds):
            # Calculate fold boundaries
            fold_start = start_date + timedelta(days=fold * fold_size)
            fold_end = fold_start + timedelta(days=fold_size)

            if fold_end > end_date:
                fold_end = end_date

            # Train on data before fold, test on fold
            train_split = DataSplit(
                split_type=SplitType.TRAIN,
                start_date=start_date,
                end_date=fold_start - timedelta(days=self.config.gap_days),
                split_id=f"CV_TRAIN_{fold}"
            )

            test_split = DataSplit(
                split_type=SplitType.TEST,
                start_date=fold_start,
                end_date=fold_end,
                split_id=f"CV_TEST_{fold}"
            )

            cv_splits.append((train_split, test_split))

        return cv_splits

# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def create_default_walk_forward_config() -> WalkForwardConfig:
    """Create default walk-forward configuration"""
    return WalkForwardConfig()

def validate_backtest_data(data: Dict[str, pd.DataFrame],
                         signals: pd.DataFrame,
                         config: Optional[WalkForwardConfig] = None) -> List[BiasCheck]:
    """
    Standalone function to validate backtest data for bias

    Args:
        data: Price data by symbol
        signals: Signal data
        config: Optional walk-forward configuration

    Returns:
        List of bias check results
    """
    if config is None:
        config = create_default_walk_forward_config()

    detector = LookaheadDetector(config)
    return detector.validate_backtest_data(data, signals)

# =====================================================================================
# EXAMPLE USAGE
# =====================================================================================

if __name__ == "__main__":
    # Example usage of walk-forward analysis

    # Create configuration
    config = WalkForwardConfig(
        train_window_days=252,
        test_window_days=63,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
        enable_out_of_sample=True
    )

    # Initialize engine
    wf_engine = WalkForwardEngine(config)

    # Create splits for date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    splits = wf_engine.create_walk_forward_splits(start_date, end_date)

    print(f"Created {len(splits)} splits:")
    for split in splits[:5]:  # Show first 5
        print(f"  {split.split_id}: {split.split_type.value} "
              f"{split.start_date.date()} to {split.end_date.date()}")

    # Example bias validation
    rng = np.random.default_rng(42)
    date_range = pd.date_range(start_date, end_date, freq='W')
    sample_signals = pd.DataFrame({
        'timestamp': date_range,
        'symbol': 'RELIANCE.NS',
        'signal_score': rng.random(len(date_range)) * 100
    })

    sample_data = {
        'RELIANCE.NS': pd.DataFrame({
            'Open': rng.random(100) * 2500 + 2400,
            'High': rng.random(100) * 2500 + 2450,
            'Low': rng.random(100) * 2500 + 2350,
            'Close': rng.random(100) * 2500 + 2400,
            'Volume': rng.integers(100000, 1000000, 100)
        }, index=pd.date_range(start_date, periods=100, freq='D'))
    }

    # Validate for bias
    bias_checks = validate_backtest_data(sample_data, sample_signals, config)
    print(f"\nBias validation: {len(bias_checks)} checks performed")
    for check in bias_checks:
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.check_name}: {check.message}")