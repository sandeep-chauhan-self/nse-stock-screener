"""
Scoring Package for NSE Stock Screener

This package implements FS.4 Composite Scoring & Model Governance with:
- Configuration-driven scoring with YAML/JSON support
- Parameter persistence with database integration  
- Calibration harness for weight optimization
- Transparent, auditable scoring with detailed breakdowns

Key Components:
- scoring_schema.py: Configuration schema and validation
- scoring_engine.py: Main scoring engine implementation
- parameter_store.py: Database persistence for configurations
- calibration.py: Weight optimization and parameter tuning

Usage:
    from src.scoring import ScoringEngine, ScoringConfig, CalibrationHarness
    
    # Load configuration
    config = create_default_config()
    
    # Create engine
    engine = ScoringEngine(config)
    
    # Score indicators
    result = engine.score(symbol="RELIANCE", indicators=indicators)
    
    # Optimize weights
    calibrator = CalibrationHarness(config)
    optimized = calibrator.optimize_weights(historical_data)
"""

from .scoring_schema import (
    ScoringSchema,
    ScoringConfig,
    ComponentConfig,
    BonusPenaltyRule,
    RegimeAdjustment,
    ScoringMethod,
    ConditionOperator,
    ThresholdConfig,
    PercentileConfig,
    ZScoreConfig,
    LinearConfig,
    CustomConfig,
    create_default_config
)

from .scoring_engine import (
    ScoringEngine,
    ScoringResult,
    ComponentScore,
    BonusPenaltyResult
)

from .parameter_store import (
    ParameterStore
)

from .calibration import (
    CalibrationHarness,
    OptimizationResult,
    ValidationPeriod,
    ObjectiveFunction,
    SharpeRatioObjective,
    WinRateObjective
)

__all__ = [
    # Main classes
    'ScoringSchema',
    'ScoringConfig',
    'ScoringEngine',
    'ParameterStore',
    'CalibrationHarness',
    'ComponentConfig',
    'BonusPenaltyRule',
    'RegimeAdjustment',
    
    # Result classes
    'ScoringResult',
    'ComponentScore', 
    'BonusPenaltyResult',
    'OptimizationResult',
    'ValidationPeriod',
    
    # Objective functions
    'ObjectiveFunction',
    'SharpeRatioObjective',
    'WinRateObjective',
    
    # Enums
    'ScoringMethod',
    'ConditionOperator',
    
    # Configuration types
    'ThresholdConfig',
    'PercentileConfig',
    'ZScoreConfig',
    'LinearConfig',
    'CustomConfig',
    
    # Utilities
    'create_default_config'
]

__version__ = "1.0.0"