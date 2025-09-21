"""
Historical stress event datasets for indicator validation.

This module provides curated datasets of major market stress events
for validating indicator behavior and ensuring robust computation
during extreme market conditions.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressEvent:
    """Definition of a market stress event for validation."""
    name: str
    description: str
    start_date: date
    end_date: date
    event_type: str  # 'crash', 'volatility', 'correction', 'rally'
    severity: str   # 'mild', 'moderate', 'severe', 'extreme'
    characteristics: List[str]  # Key market characteristics during event
    expected_indicators: Dict[str, Dict[str, float]]  # Expected indicator ranges


# Define major stress events for validation
STRESS_EVENTS = [
    StressEvent(
        name="2008_financial_crisis",
        description="Global Financial Crisis - Lehman Brothers collapse and market crash",
        start_date=date(2008, 9, 1),
        end_date=date(2009, 3, 31),
        event_type="crash",
        severity="extreme",
        characteristics=[
            "extreme_volatility", "high_volume", "trending_down",
            "correlation_breakdown", "liquidity_crisis"
        ],
        expected_indicators={
            "RSI": {"min": 10, "max": 90, "typical_low": 20, "typical_high": 35},
            "ATR": {"min": 2, "max": 15, "typical": 8},  # % of price
            "ADX": {"min": 20, "max": 80, "typical": 45},
            "VIX": {"min": 20, "max": 80, "typical": 45}
        }
    ),

    StressEvent(
        name="2020_covid_crash",
        description="COVID-19 pandemic market crash and recovery",
        start_date=date(2020, 2, 15),
        end_date=date(2020, 5, 31),
        event_type="crash",
        severity="severe",
        characteristics=[
            "extreme_volatility", "v_shaped_recovery", "sector_rotation",
            "central_bank_intervention", "high_correlation"
        ],
        expected_indicators={
            "RSI": {"min": 15, "max": 85, "typical_low": 25, "typical_high": 40},
            "ATR": {"min": 3, "max": 20, "typical": 12},
            "ADX": {"min": 25, "max": 70, "typical": 40},
            "MACD": {"extreme_divergence": True}
        }
    ),

    StressEvent(
        name="2022_inflation_volatility",
        description="Inflation concerns and interest rate volatility",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 10, 31),
        event_type="volatility",
        severity="moderate",
        characteristics=[
            "regime_uncertainty", "sector_rotation", "growth_value_divergence",
            "commodity_spike", "currency_volatility"
        ],
        expected_indicators={
            "RSI": {"min": 25, "max": 75, "typical_range": [35, 65]},
            "ATR": {"min": 1, "max": 8, "typical": 4},
            "ADX": {"min": 15, "max": 50, "typical": 25},
            "correlation": {"breakdown_periods": True}
        }
    ),

    StressEvent(
        name="2018_trade_war",
        description="US-China trade war volatility",
        start_date=date(2018, 3, 1),
        end_date=date(2018, 12, 31),
        event_type="volatility",
        severity="moderate",
        characteristics=[
            "news_driven", "sector_specific", "international_divergence",
            "commodity_impact", "uncertainty_premium"
        ],
        expected_indicators={
            "RSI": {"min": 30, "max": 70, "whipsaw": True},
            "ATR": {"min": 1, "max": 6, "typical": 3},
            "ADX": {"min": 10, "max": 40, "trending_periods": False}
        }
    ),

    StressEvent(
        name="2015_china_devaluation",
        description="China currency devaluation and emerging market stress",
        start_date=date(2015, 8, 1),
        end_date=date(2015, 9, 30),
        event_type="correction",
        severity="moderate",
        characteristics=[
            "emerging_market_contagion", "currency_crisis", "commodity_crash",
            "flight_to_quality", "correlation_spike"
        ],
        expected_indicators={
            "RSI": {"min": 25, "max": 75, "oversold_rebounds": True},
            "ATR": {"min": 1.5, "max": 8, "spike_pattern": True},
            "ADX": {"min": 15, "max": 55, "directional_strength": True}
        }
    )
]


class StressTestDataGenerator:
    """
    Generates synthetic market data for stress testing indicators.

    Creates realistic OHLCV data that exhibits the characteristics
    of major historical stress events for comprehensive validation.
    """

    def __init__(self, base_price: float = 100.0, random_seed: int = 42):
        self.base_price = base_price
        self.rng = np.random.RandomState(random_seed)

    def generate_normal_market(self,
                              days: int = 252,
                              daily_volatility: float = 0.015,
                              trend: float = 0.0001) -> pd.DataFrame:
        """Generate normal market conditions data."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Generate price movements
        returns = self.rng.normal(trend, daily_volatility, days)
        prices = self.base_price * np.exp(np.cumsum(returns))

        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            # Daily volatility for OHLC generation
            daily_vol = daily_volatility * 0.5

            open_price = price * (1 + self.rng.normal(0, daily_vol * 0.5))
            close_price = price

            high_factor = abs(self.rng.normal(0, daily_vol))
            low_factor = abs(self.rng.normal(0, daily_vol))

            high_price = max(open_price, close_price) * (1 + high_factor)
            low_price = min(open_price, close_price) * (1 - low_factor)

            # Volume (with some correlation to price movement)
            volume_base = 1000000
            volume_factor = 1 + abs(returns[i]) * 5  # Higher volume on big moves
            volume = int(volume_base * volume_factor * (1 + self.rng.normal(0, 0.3)))

            data.append({
                'Date': dates[i],
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': max(volume, 1000)  # Minimum volume
            })

        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df

    def generate_crash_scenario(self,
                               days: int = 60,
                               max_drawdown: float = -0.35,
                               volatility_spike: float = 5.0) -> pd.DataFrame:
        """Generate market crash scenario similar to 2008 or 2020."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Create crash pattern: initial decline, panic, stabilization
        crash_days = days // 3
        panic_days = days // 3
        recovery_days = days - crash_days - panic_days

        returns = []

        # Initial decline phase
        for i in range(crash_days):
            progress = i / crash_days
            base_return = max_drawdown * 0.3 / crash_days  # 30% of total decline
            volatility = 0.02 + 0.03 * progress  # Increasing volatility
            daily_return = self.rng.normal(base_return, volatility)
            returns.append(daily_return)

        # Panic phase
        for i in range(panic_days):
            progress = i / panic_days
            base_return = max_drawdown * 0.5 / panic_days  # 50% of total decline
            volatility = 0.05 + 0.05 * (1 - abs(progress - 0.5) * 2)  # Peak volatility mid-phase
            daily_return = self.rng.normal(base_return, volatility)
            returns.append(daily_return)

        # Stabilization phase
        for i in range(recovery_days):
            progress = i / recovery_days
            base_return = max_drawdown * 0.2 / recovery_days  # Remaining decline
            volatility = 0.04 * (1 - progress * 0.5)  # Decreasing volatility
            daily_return = self.rng.normal(base_return, volatility)
            returns.append(daily_return)

        # Generate prices and OHLCV
        prices = self.base_price * np.exp(np.cumsum(returns))
        return self._generate_ohlcv_from_prices(dates, prices, returns, volatility_spike)

    def generate_high_volatility_scenario(self,
                                        days: int = 120,
                                        base_volatility: float = 0.04) -> pd.DataFrame:
        """Generate high volatility regime with frequent direction changes."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        returns = []
        for i in range(days):
            # Create regime changes every 10-20 days
            regime_day = i % 15
            if regime_day < 5:
                # Trending up
                mean_return = 0.002
                volatility = base_volatility * 0.8
            elif regime_day < 10:
                # Trending down
                mean_return = -0.002
                volatility = base_volatility * 1.2
            else:
                # Sideways with high volatility
                mean_return = 0.0
                volatility = base_volatility * 1.5

            daily_return = self.rng.normal(mean_return, volatility)
            returns.append(daily_return)

        prices = self.base_price * np.exp(np.cumsum(returns))
        return self._generate_ohlcv_from_prices(dates, prices, returns, 2.0)

    def generate_trending_scenario(self,
                                 days: int = 180,
                                 total_return: float = 0.25,
                                 volatility: float = 0.02) -> pd.DataFrame:
        """Generate strong trending market."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        daily_trend = total_return / days
        returns = []

        for i in range(days):
            # Add some momentum (trending gets stronger over time)
            momentum_factor = 1 + (i / days) * 0.5
            trend_component = daily_trend * momentum_factor

            # Add noise
            noise = self.rng.normal(0, volatility)
            daily_return = trend_component + noise
            returns.append(daily_return)

        prices = self.base_price * np.exp(np.cumsum(returns))
        return self._generate_ohlcv_from_prices(dates, prices, returns, 1.0)

    def _generate_ohlcv_from_prices(self,
                                   dates: pd.DatetimeIndex,
                                   prices: np.ndarray,
                                   returns: List[float],
                                   volume_multiplier: float = 1.0) -> pd.DataFrame:
        """Generate OHLCV data from price series."""
        data = []

        for i, (date, price, ret) in enumerate(zip(dates, prices, returns)):
            # Intraday volatility based on daily return magnitude
            intraday_vol = min(abs(ret) * 2, 0.1)  # Cap at 10%

            # Generate OHLC
            if i == 0:
                open_price = price
            else:
                # Open with some gap based on overnight sentiment
                gap = self.rng.normal(0, intraday_vol * 0.3)
                open_price = prices[i-1] * (1 + gap)

            close_price = price

            # High and Low based on intraday volatility
            high_move = abs(self.rng.normal(0, intraday_vol))
            low_move = abs(self.rng.normal(0, intraday_vol))

            high_price = max(open_price, close_price) * (1 + high_move)
            low_price = min(open_price, close_price) * (1 - low_move)

            # Volume correlated with price movement and volatility
            volume_base = 1000000
            volume_factor = (1 + abs(ret) * 10) * volume_multiplier
            volume_noise = 1 + self.rng.normal(0, 0.5)
            volume = int(volume_base * volume_factor * volume_noise)

            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': max(volume, 1000)
            })

        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df


class ValidationDataset:
    """
    Comprehensive validation dataset for indicator testing.

    Provides both historical stress event data and synthetic scenarios
    for thorough indicator validation and regression testing.
    """

    def __init__(self):
        self.stress_events = STRESS_EVENTS
        self.data_generator = StressTestDataGenerator()
        self._cached_datasets = {}

    def get_stress_event(self, event_name: str) -> Optional[StressEvent]:
        """Get stress event definition by name."""
        for event in self.stress_events:
            if event.name == event_name:
                return event
        return None

    def list_stress_events(self) -> List[str]:
        """Get list of available stress event names."""
        return [event.name for event in self.stress_events]

    def generate_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive test scenarios for validation."""
        scenarios = {}

        # Normal market conditions
        scenarios['normal_market'] = self.data_generator.generate_normal_market(
            days=252, daily_volatility=0.015, trend=0.0001
        )

        # Bull market
        scenarios['bull_market'] = self.data_generator.generate_trending_scenario(
            days=180, total_return=0.30, volatility=0.018
        )

        # Bear market
        scenarios['bear_market'] = self.data_generator.generate_trending_scenario(
            days=150, total_return=-0.25, volatility=0.025
        )

        # Market crash
        scenarios['market_crash'] = self.data_generator.generate_crash_scenario(
            days=60, max_drawdown=-0.40, volatility_spike=6.0
        )

        # High volatility regime
        scenarios['high_volatility'] = self.data_generator.generate_high_volatility_scenario(
            days=120, base_volatility=0.045
        )

        # Low volatility regime
        scenarios['low_volatility'] = self.data_generator.generate_normal_market(
            days=180, daily_volatility=0.008, trend=0.0002
        )

        # Range-bound market
        scenarios['sideways_market'] = self.data_generator.generate_high_volatility_scenario(
            days=200, base_volatility=0.02
        )

        return scenarios

    def get_known_values_dataset(self) -> Dict[str, Tuple[pd.DataFrame, Dict[str, float]]]:
        """
        Generate datasets with known indicator values for unit testing.

        Returns:
            Dictionary mapping scenario names to (data, expected_values) tuples
        """
        known_values = {}

        # Simple constant price scenario (RSI should be 50, ATR should be 0)
        constant_data = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100] * 50,
            'Low': [100] * 50,
            'Close': [100] * 50,
            'Volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))

        known_values['constant_price'] = (constant_data, {
            'RSI_14': 50.0,  # RSI should be neutral
            'ATR_14': 0.0,   # No volatility
            'ADX_14': 0.0    # No trend
        })

        # Simple trending scenario
        trending_prices = np.linspace(100, 110, 50)  # 10% linear increase
        trending_data = pd.DataFrame({
            'Open': trending_prices,
            'High': trending_prices * 1.01,
            'Low': trending_prices * 0.99,
            'Close': trending_prices,
            'Volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))

        known_values['linear_trend'] = (trending_data, {
            'RSI_14': 70.0,  # Should be overbought (approximately)
            'ATR_14': 1.0,   # Low volatility due to smooth trend
            'ADX_14': 25.0   # Moderate trend strength
        })

        return known_values

    def save_datasets_to_files(self, output_dir: Union[str, Path]) -> None:
        """Save all validation datasets to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save test scenarios
        scenarios = self.generate_test_scenarios()
        for name, data in scenarios.items():
            file_path = output_dir / f"{name}.csv"
            data.to_csv(file_path)
            logger.info(f"Saved {name} dataset to {file_path}")

        # Save known values datasets
        known_values = self.get_known_values_dataset()
        for name, (data, expected) in known_values.items():
            file_path = output_dir / f"{name}.csv"
            data.to_csv(file_path)

            # Save expected values
            expected_file = output_dir / f"{name}_expected.json"
            import json
            with open(expected_file, 'w') as f:
                json.dump(expected, f, indent=2)

            logger.info(f"Saved {name} dataset and expected values")

    def load_dataset(self, name: str, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load a dataset by name or from file."""
        if file_path:
            return pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Generate if not cached
        if name not in self._cached_datasets:
            scenarios = self.generate_test_scenarios()
            if name in scenarios:
                self._cached_datasets[name] = scenarios[name]
            else:
                raise ValueError(f"Unknown dataset: {name}")

        return self._cached_datasets[name]


# Global validation dataset instance
_validation_dataset = ValidationDataset()


def get_validation_dataset() -> ValidationDataset:
    """Get the global validation dataset instance."""
    return _validation_dataset


def create_stress_test_data(output_dir: Union[str, Path] = "validation_data") -> None:
    """Create and save stress test datasets to files."""
    dataset = get_validation_dataset()
    dataset.save_datasets_to_files(output_dir)
    print(f"Created validation datasets in {output_dir}")


if __name__ == "__main__":
    # Create validation datasets when run as script
    create_stress_test_data()