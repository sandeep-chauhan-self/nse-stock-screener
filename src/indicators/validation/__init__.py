"""
Validation package for indicator testing and validation.

This package provides comprehensive testing and validation capabilities
for technical indicators, including stress testing, regression testing,
performance validation, and accuracy verification.
"""

from .stress_data import (
    ValidationDataset,
    StressEvent,
    StressTestDataGenerator,
    get_validation_dataset,
    create_stress_test_data,
    STRESS_EVENTS
)

from .test_suite import (
    IndicatorTestCase,
    RSITests,
    MACDTests,
    ATRTests,
    PerformanceTests,
    StressTests,
    ValidationTestSuite,
    run_validation_tests
)

from .utils import (
    ValidationMetrics,
    RegressionTestResult,
    IndicatorValidator,
    ValidationReportGenerator,
    validate_indicator_suite
)

__all__ = [
    # Data generation and datasets
    'ValidationDataset',
    'StressEvent', 
    'StressTestDataGenerator',
    'get_validation_dataset',
    'create_stress_test_data',
    'STRESS_EVENTS',
    
    # Test framework
    'IndicatorTestCase',
    'RSITests',
    'MACDTests', 
    'ATRTests',
    'PerformanceTests',
    'StressTests',
    'ValidationTestSuite',
    'run_validation_tests',
    
    # Validation utilities
    'ValidationMetrics',
    'RegressionTestResult',
    'IndicatorValidator',
    'ValidationReportGenerator',
    'validate_indicator_suite'
]

# Version information
__version__ = "1.0.0"
__author__ = "Indicator Engine Team"
__description__ = "Comprehensive validation framework for technical indicators"