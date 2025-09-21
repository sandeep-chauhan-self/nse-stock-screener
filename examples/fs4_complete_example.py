"""
FS.4 Composite Scoring & Model Governance - Example Usage

This file demonstrates how to use the complete FS.4 scoring system including:
- Configuration setup with YAML
- Dynamic scoring engine usage
- Parameter persistence
- Weight optimization with calibration harness

Requirements covered:
- Transparent, auditable scoring with tunable parameters
- Configuration-driven scoring logic with persistence
- Calibration harness for weight optimization
- Database tracking of configurations and performance
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.scoring import (
    ScoringConfig,
    ScoringEngine,
    ParameterStore,
    CalibrationHarness,
    create_default_config,
    ValidationPeriod,
    ScoringMethod,
    ConditionOperator
)


def create_sample_configuration():
    """Create a sample configuration demonstrating FS.4 capabilities."""

    # Create default configuration
    config = create_default_config()

    # Customize some components for demonstration
    for component in config.components:
        if component.name == "momentum_composite":
            component.weight = 0.25
            component.method_config.high_threshold = 1.5  # Z-score threshold
        elif component.name == "volume_analysis":
            component.weight = 0.20
        elif component.name == "technical_strength":
            component.weight = 0.30
        elif component.name == "relative_performance":
            component.weight = 0.25

    # Add custom bonus/penalty rules
    from src.scoring.scoring_schema import BonusPenaltyRule

    high_volume_bonus = BonusPenaltyRule(
        name="high_volume_bonus",
        description="Bonus for exceptional volume (>3x average)",
        condition="vol_ratio > 3.0",
        operator=ConditionOperator.GREATER_THAN,
        value=3.0,
        bonus=5.0
    )

    config.bonus_penalty_rules.append(high_volume_bonus)

    return config


def generate_sample_data():
    """Generate sample market data for demonstration."""
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ADANIGREEN', 'BHARTIARTL']

    historical_data = {}
    rng = np.random.default_rng(42)

    for symbol in symbols:
        # Generate 1 year of daily data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

        # Generate realistic price data with some trend
        base_price = rng.uniform(100, 2000)
        price_changes = rng.normal(0.001, 0.025, len(dates))  # Daily returns
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        volume_base = rng.uniform(1000000, 10000000)
        volumes = rng.uniform(0.5, 2.0, len(dates)) * volume_base

        data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'High': [p * rng.uniform(1.0, 1.05) for p in prices],
            'Low': [p * rng.uniform(0.95, 1.0) for p in prices],
            'Open': prices  # Simplified
        }, index=dates)

        historical_data[symbol] = data

    return historical_data


def generate_sample_indicators(symbol, data):
    """Generate sample technical indicators for a symbol."""
    if len(data) < 20:
        return {}

    close_prices = data['Close']
    volumes = data['Volume']

    # Calculate some basic indicators (simplified versions)
    sma_20 = close_prices.rolling(20).mean().iloc[-1]
    sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else sma_20

    current_price = close_prices.iloc[-1]
    price_vs_sma20 = (current_price - sma_20) / sma_20 * 100

    avg_volume = volumes.rolling(20).mean().iloc[-1]
    vol_ratio = volumes.iloc[-1] / avg_volume

    # Mock some additional indicators
    rng = np.random.default_rng(hash(symbol) % 2**32)

    return {
        'symbol': symbol,
        'current_price': current_price,
        'momentum_composite': price_vs_sma20 / 10,  # Normalize to z-score-like
        'vol_ratio': vol_ratio,
        'rsi': rng.uniform(20, 80),
        'macd': rng.normal(0, 5),
        'macd_signal': rng.normal(0, 3),
        'adx': rng.uniform(10, 50),
        'atr_pct': rng.uniform(0.5, 3.0),
        'rel_strength_20d': rng.normal(0, 5),
        'sma_20': sma_20,
        'sma_50': sma_50
    }


def demonstrate_scoring_engine():
    """Demonstrate the scoring engine with sample data."""
    print("=== FS.4 Scoring Engine Demonstration ===")

    # Create configuration
    config = create_sample_configuration()
    print(f"Created configuration with {len(config.components)} components")

    # Initialize scoring engine
    engine = ScoringEngine(config)
    print("Initialized scoring engine")

    # Generate sample data
    historical_data = generate_sample_data()
    print(f"Generated sample data for {len(historical_data)} symbols")

    # Score each symbol
    results = []
    for symbol, data in historical_data.items():
        indicators = generate_sample_indicators(symbol, data)

        if indicators:
            result = engine.score(symbol, indicators)
            results.append(result)

            print(f"\n{symbol}:")
            print(f"  Total Score: {result.total_score:.2f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Probability Level: {result.probability_level}")
            print(f"  Components: {len(result.component_scores)}")

            # Show top scoring components
            top_components = sorted(result.component_scores,
                                  key=lambda x: x.weighted_score, reverse=True)[:3]
            for comp in top_components:
                print(f"    {comp.name}: {comp.weighted_score:.2f}")

    return config, results, historical_data


def demonstrate_parameter_persistence():
    """Demonstrate parameter store and database persistence."""
    print("\n=== Parameter Persistence Demonstration ===")

    # Initialize parameter store
    db_path = "data/temp/scoring_demo.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    param_store = ParameterStore(db_path)
    print(f"Initialized parameter store: {db_path}")

    # Create and save configuration
    config = create_sample_configuration()
    config_id = param_store.save_config(
        config,
        metadata={
            'description': 'Demo configuration for FS.4',
            'created_by': 'example_script',
            'use_case': 'demonstration'
        }
    )

    print(f"Saved configuration with ID: {config_id}")

    # Track some performance metrics
    param_store.track_config_performance(config_id, "demo_backtest", 0.15)
    param_store.track_config_performance(config_id, "demo_sharpe", 1.25)

    # List saved configurations
    try:
        configs = param_store.get_best_configs(metric_name="demo_backtest", limit=5)
        print(f"Configurations found: {len(configs)}")

        for config_id, score in configs:
            config_data, _ = param_store.get_config(config_id)
            if config_data:
                print(f"  Config {config_id}: {config_data.name} (score: {score:.3f})")
    except Exception as e:
        print(f"Note: Could not list configurations: {e}")
        # Get database stats instead
        stats = param_store.get_database_stats()
        print(f"Database stats: {stats}")

    return param_store, config_id


def demonstrate_calibration_harness():
    """Demonstrate weight optimization with calibration harness."""
    print("\n=== Calibration Harness Demonstration ===")

    # Create base configuration
    config = create_sample_configuration()
    historical_data = generate_sample_data()

    # Initialize parameter store for tracking
    param_store = ParameterStore("data/temp/scoring_demo.db")

    # Create calibration harness
    calibrator = CalibrationHarness(config, param_store)
    print("Initialized calibration harness")

    # Define validation periods
    validation_periods = [
        ValidationPeriod(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            name="training_period"
        ),
        ValidationPeriod(
            start_date=datetime(2023, 7, 1),
            end_date=datetime(2024, 1, 1),
            name="validation_period"
        )
    ]

    print("Starting weight optimization...")

    # Run optimization (small grid for demo)
    result = calibrator.optimize_weights(
        historical_data=historical_data,
        method='grid_search',
        objective='win_rate',
        validation_periods=validation_periods,
        steps=3  # Small grid for demonstration
    )

    print("Optimization completed!")
    print(f"  Method: {result.method_used}")
    print(f"  Objective: {result.objective_function}")
    print(f"  Best Score: {result.best_score:.4f}")
    print(f"  Total Evaluations: {result.total_evaluations}")
    print(f"  Time: {result.optimization_time_seconds:.1f} seconds")

    # Show optimized weights
    print("\nOptimized Component Weights:")
    for component in result.best_config.get_enabled_components():
        print(f"  {component.name}: {component.weight:.3f}")

    # Get optimization summary
    summary = calibrator.get_optimization_summary(result)
    print("\nOptimization Summary:")
    print(f"  Score Range: {summary['score_statistics']['min']:.4f} - {summary['score_statistics']['max']:.4f}")
    print(f"  Score Mean: {summary['score_statistics']['mean']:.4f}")
    print(f"  Improvement: {summary['improvement']['absolute']:.4f} ({summary['improvement']['relative_pct']:.1f}%)")

    return result


def demonstrate_yaml_configuration():
    """Demonstrate YAML configuration loading and saving."""
    print("\n=== YAML Configuration Demonstration ===")

    # Create sample configuration
    config = create_sample_configuration()

    # Save to YAML
    yaml_path = "data/temp/sample_scoring_config.yaml"
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    try:
        from src.scoring.scoring_schema import save_config_to_file, load_config_from_file

        save_config_to_file(config, yaml_path)
        print(f"Saved configuration to: {yaml_path}")

        # Load back from YAML
        loaded_config = load_config_from_file(yaml_path)
        print(f"Loaded configuration: {loaded_config.name} v{loaded_config.version}")
        print(f"  Components: {len(loaded_config.components)}")
        print(f"  Rules: {len(loaded_config.bonus_penalty_rules)}")

    except ImportError:
        print("YAML support requires PyYAML package")
        print("Install with: pip install PyYAML")


def main():
    """Main demonstration function."""
    print("FS.4 Composite Scoring & Model Governance - Complete Example")
    print("=" * 60)

    try:
        # 1. Demonstrate scoring engine
        _, _, _ = demonstrate_scoring_engine()

        # 2. Demonstrate parameter persistence
        _, _ = demonstrate_parameter_persistence()

        # 3. Demonstrate calibration harness
        demonstrate_calibration_harness()

        # 4. Demonstrate YAML configuration
        demonstrate_yaml_configuration()

        print("\n" + "=" * 60)
        print("FS.4 Demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Configuration-driven scoring with tunable parameters")
        print("✅ Dynamic scoring engine with component breakdown")
        print("✅ Database persistence for configurations and tracking")
        print("✅ Weight optimization with calibration harness")
        print("✅ YAML configuration support")
        print("✅ Transparent, auditable scoring system")

        print("\nFiles created:")
        print("  - Database: data/temp/scoring_demo.db")
        print("  - Config: data/temp/sample_scoring_config.yaml")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()