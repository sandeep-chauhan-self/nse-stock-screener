"""
Validation utilities and helpers for indicator testing.

This module provides utilities for indicator validation, regression testing,
and performance benchmarking across different market scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import logging
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from ..base import BaseIndicator, IndicatorResult
from ..engine import IndicatorEngine
from .stress_data import ValidationDataset, StressEvent

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for indicator validation."""
    accuracy_score: float
    stability_score: float
    performance_score: float
    confidence_score: float
    overall_score: float
    test_timestamp: datetime
    scenario: str
    indicator_name: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['test_timestamp'] = self.test_timestamp.isoformat()
        return data


@dataclass
class RegressionTestResult:
    """Result of regression testing for an indicator."""
    indicator_name: str
    baseline_version: str
    current_version: str
    scenarios_tested: List[str]
    differences: Dict[str, float]
    max_difference: float
    passed: bool
    details: Dict[str, Any]


class IndicatorValidator:
    """
    Comprehensive validator for technical indicators.
    
    Provides methods for accuracy validation, stability testing,
    performance benchmarking, and regression testing.
    """
    
    def __init__(self, validation_dataset: Optional[ValidationDataset] = None):
        self.validation_dataset = validation_dataset or ValidationDataset()
        self.engine = IndicatorEngine()
        
    def validate_accuracy(self, 
                         indicator: BaseIndicator,
                         scenario: str = 'normal_market',
                         known_values: Optional[Dict[str, float]] = None) -> float:
        """
        Validate indicator accuracy against known values or expected ranges.
        
        Args:
            indicator: Indicator to validate
            scenario: Test scenario name
            known_values: Expected values for validation
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        data = self.validation_dataset.load_dataset(scenario)
        result = indicator.calculate(data)
        
        if known_values:
            return self._compare_with_known_values(result, known_values)
        else:
            return self._validate_expected_ranges(result, scenario)
    
    def validate_stability(self, 
                          indicator: BaseIndicator,
                          scenarios: Optional[List[str]] = None) -> float:
        """
        Validate indicator stability across different market scenarios.
        
        Args:
            indicator: Indicator to validate
            scenarios: List of scenarios to test
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        if scenarios is None:
            scenarios = ['normal_market', 'high_volatility', 'market_crash', 'bull_market']
        
        stability_scores = []
        
        for scenario in scenarios:
            data = self.validation_dataset.load_dataset(scenario)
            result = indicator.calculate(data)
            
            # Calculate stability metrics
            nan_ratio = result.values.isna().sum() / len(result.values)
            confidence_score = result.confidence
            
            # Check for extreme outliers
            if isinstance(result.values, pd.Series):
                clean_values = result.values.dropna()
                if len(clean_values) > 0:
                    q1, q3 = clean_values.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = ((clean_values < (q1 - 3 * iqr)) | 
                               (clean_values > (q3 + 3 * iqr))).sum()
                    outlier_ratio = outliers / len(clean_values)
                else:
                    outlier_ratio = 1.0
            else:
                outlier_ratio = 0.0  # For multi-column results
            
            # Combine metrics
            scenario_stability = (
                (1 - nan_ratio) * 0.4 +
                confidence_score * 0.4 +
                (1 - min(outlier_ratio, 1.0)) * 0.2
            )
            stability_scores.append(scenario_stability)
        
        return np.mean(stability_scores)
    
    def validate_performance(self, 
                           indicator: BaseIndicator,
                           data_sizes: Optional[List[int]] = None,
                           target_time_per_1k: float = 0.1) -> float:
        """
        Validate indicator performance across different data sizes.
        
        Args:
            indicator: Indicator to validate
            data_sizes: List of data sizes to test
            target_time_per_1k: Target time per 1000 data points (seconds)
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 2500, 5000]
        
        performance_scores = []
        base_data = self.validation_dataset.load_dataset('normal_market')
        
        for size in data_sizes:
            if size > len(base_data):
                # Extend data by replication if needed
                test_data = pd.concat([base_data] * (size // len(base_data) + 1))[:size]
            else:
                test_data = base_data[:size]
            
            # Measure calculation time
            start_time = time.time()
            result = indicator.calculate(test_data)
            calc_time = time.time() - start_time
            
            # Calculate performance score
            time_per_1k = calc_time * 1000 / size
            if time_per_1k <= target_time_per_1k:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (time_per_1k - target_time_per_1k) / target_time_per_1k)
            
            performance_scores.append(score)
        
        return np.mean(performance_scores)
    
    def comprehensive_validation(self, 
                               indicator: BaseIndicator,
                               scenario: str = 'normal_market') -> ValidationMetrics:
        """
        Perform comprehensive validation of an indicator.
        
        Args:
            indicator: Indicator to validate
            scenario: Primary test scenario
            
        Returns:
            Comprehensive validation metrics
        """
        # Get indicator parameters
        params = {}
        if hasattr(indicator, '__dict__'):
            params = {k: v for k, v in indicator.__dict__.items() 
                     if not k.startswith('_') and not callable(v)}
        
        # Run validation tests
        accuracy_score = self.validate_accuracy(indicator, scenario)
        stability_score = self.validate_stability(indicator)
        performance_score = self.validate_performance(indicator)
        
        # Calculate confidence from base result
        data = self.validation_dataset.load_dataset(scenario)
        result = indicator.calculate(data)
        confidence_score = result.confidence
        
        # Calculate overall score
        overall_score = (
            accuracy_score * 0.3 +
            stability_score * 0.3 +
            performance_score * 0.2 +
            confidence_score * 0.2
        )
        
        return ValidationMetrics(
            accuracy_score=accuracy_score,
            stability_score=stability_score,
            performance_score=performance_score,
            confidence_score=confidence_score,
            overall_score=overall_score,
            test_timestamp=datetime.now(),
            scenario=scenario,
            indicator_name=type(indicator).__name__,
            parameters=params
        )
    
    def regression_test(self,
                       indicator: BaseIndicator,
                       baseline_results: Dict[str, IndicatorResult],
                       tolerance: float = 1e-6) -> RegressionTestResult:
        """
        Perform regression testing against baseline results.
        
        Args:
            indicator: Current indicator implementation
            baseline_results: Baseline results by scenario
            tolerance: Numerical tolerance for differences
            
        Returns:
            Regression test results
        """
        scenarios_tested = list(baseline_results.keys())
        differences = {}
        max_difference = 0.0
        details = {}
        
        for scenario, baseline_result in baseline_results.items():
            data = self.validation_dataset.load_dataset(scenario)
            current_result = indicator.calculate(data)
            
            # Compare results
            diff = self._calculate_result_difference(baseline_result, current_result)
            differences[scenario] = diff
            max_difference = max(max_difference, diff)
            
            # Store detailed comparison
            details[scenario] = {
                'baseline_confidence': baseline_result.confidence,
                'current_confidence': current_result.confidence,
                'difference': diff,
                'values_compared': len(baseline_result.values)
            }
        
        passed = max_difference <= tolerance
        
        return RegressionTestResult(
            indicator_name=type(indicator).__name__,
            baseline_version="unknown",  # Could be parameterized
            current_version="current",
            scenarios_tested=scenarios_tested,
            differences=differences,
            max_difference=max_difference,
            passed=passed,
            details=details
        )
    
    def _compare_with_known_values(self, 
                                  result: IndicatorResult,
                                  known_values: Dict[str, float]) -> float:
        """Compare indicator result with known expected values."""
        if isinstance(result.values, pd.Series):
            final_value = result.values.iloc[-1]
            if pd.isna(final_value):
                return 0.0
            
            # Find matching known value
            indicator_name = None
            for key in known_values.keys():
                if key.startswith(type(result).__name__):
                    indicator_name = key
                    break
            
            if indicator_name:
                expected = known_values[indicator_name]
                # Calculate relative error
                if expected != 0:
                    error = abs(final_value - expected) / abs(expected)
                else:
                    error = abs(final_value)
                
                return max(0.0, 1.0 - error)
        
        return 0.5  # Default score for complex results
    
    def _validate_expected_ranges(self, 
                                result: IndicatorResult,
                                scenario: str) -> float:
        """Validate against expected ranges for scenario."""
        # Define expected ranges by indicator type and scenario
        ranges = self._get_expected_ranges(type(result).__name__, scenario)
        
        if not ranges:
            return 0.8  # Default score when no ranges defined
        
        if isinstance(result.values, pd.Series):
            clean_values = result.values.dropna()
            if len(clean_values) == 0:
                return 0.0
            
            # Check if values fall within expected ranges
            min_val, max_val = clean_values.min(), clean_values.max()
            
            if 'min' in ranges and min_val < ranges['min']:
                return 0.3
            if 'max' in ranges and max_val > ranges['max']:
                return 0.3
            
            return 0.9
        
        return 0.7  # Default for multi-column results
    
    def _get_expected_ranges(self, indicator_name: str, scenario: str) -> Dict[str, float]:
        """Get expected value ranges for indicator and scenario."""
        ranges_map = {
            ('RSI', 'normal_market'): {'min': 20, 'max': 80},
            ('RSI', 'market_crash'): {'min': 5, 'max': 95},
            ('ATR', 'normal_market'): {'min': 0, 'max': float('inf')},
            ('ATR', 'high_volatility'): {'min': 0, 'max': float('inf')},
            ('ADX', 'normal_market'): {'min': 0, 'max': 100},
        }
        
        return ranges_map.get((indicator_name, scenario), {})
    
    def _calculate_result_difference(self, 
                                   baseline: IndicatorResult,
                                   current: IndicatorResult) -> float:
        """Calculate numerical difference between two indicator results."""
        if isinstance(baseline.values, pd.Series) and isinstance(current.values, pd.Series):
            # Align series by index
            aligned_baseline, aligned_current = baseline.values.align(current.values)
            
            # Compare non-NaN values
            mask = ~(aligned_baseline.isna() | aligned_current.isna())
            if mask.sum() == 0:
                return float('inf')  # No comparable values
            
            diff = np.abs(aligned_baseline[mask] - aligned_current[mask])
            return float(diff.max())
        
        elif isinstance(baseline.values, pd.DataFrame) and isinstance(current.values, pd.DataFrame):
            max_diff = 0.0
            for col in baseline.values.columns:
                if col in current.values.columns:
                    baseline_col = baseline.values[col]
                    current_col = current.values[col]
                    
                    aligned_baseline, aligned_current = baseline_col.align(current_col)
                    mask = ~(aligned_baseline.isna() | aligned_current.isna())
                    
                    if mask.sum() > 0:
                        diff = np.abs(aligned_baseline[mask] - aligned_current[mask])
                        max_diff = max(max_diff, float(diff.max()))
            
            return max_diff
        
        return float('inf')  # Incomparable types


class ValidationReportGenerator:
    """
    Generate comprehensive validation reports for indicator testing.
    """
    
    def __init__(self):
        self.validator = IndicatorValidator()
    
    def generate_indicator_report(self, 
                                indicator: BaseIndicator,
                                scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate validation report for a single indicator."""
        if scenarios is None:
            scenarios = ['normal_market', 'bull_market', 'bear_market', 'high_volatility']
        
        report = {
            'indicator_name': type(indicator).__name__,
            'test_timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'summary': {}
        }
        
        all_metrics = []
        
        for scenario in scenarios:
            try:
                metrics = self.validator.comprehensive_validation(indicator, scenario)
                report['scenarios'][scenario] = metrics.to_dict()
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed validation for {scenario}: {e}")
                report['scenarios'][scenario] = {'error': str(e)}
        
        # Generate summary
        if all_metrics:
            report['summary'] = {
                'avg_accuracy': np.mean([m.accuracy_score for m in all_metrics]),
                'avg_stability': np.mean([m.stability_score for m in all_metrics]),
                'avg_performance': np.mean([m.performance_score for m in all_metrics]),
                'avg_confidence': np.mean([m.confidence_score for m in all_metrics]),
                'overall_score': np.mean([m.overall_score for m in all_metrics]),
                'scenarios_tested': len(all_metrics),
                'passed_threshold': all(m.overall_score >= 0.7 for m in all_metrics)
            }
        
        return report
    
    def generate_suite_report(self, 
                            indicators: Dict[str, BaseIndicator],
                            output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate validation report for a suite of indicators."""
        suite_report = {
            'test_timestamp': datetime.now().isoformat(),
            'indicators_tested': list(indicators.keys()),
            'individual_reports': {},
            'summary': {}
        }
        
        all_scores = []
        
        for name, indicator in indicators.items():
            try:
                report = self.generate_indicator_report(indicator)
                suite_report['individual_reports'][name] = report
                
                if 'summary' in report and 'overall_score' in report['summary']:
                    all_scores.append(report['summary']['overall_score'])
                    
            except Exception as e:
                logger.error(f"Failed to generate report for {name}: {e}")
                suite_report['individual_reports'][name] = {'error': str(e)}
        
        # Generate suite summary
        if all_scores:
            suite_report['summary'] = {
                'total_indicators': len(indicators),
                'successful_tests': len(all_scores),
                'average_score': np.mean(all_scores),
                'passing_indicators': sum(1 for score in all_scores if score >= 0.7),
                'failing_indicators': sum(1 for score in all_scores if score < 0.7),
                'suite_passed': all(score >= 0.7 for score in all_scores)
            }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(suite_report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_file}")
        
        return suite_report
    
    def save_baseline_results(self, 
                            indicators: Dict[str, BaseIndicator],
                            output_dir: Union[str, Path]) -> None:
        """Save baseline results for regression testing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scenarios = ['normal_market', 'bull_market', 'bear_market', 'high_volatility']
        
        for name, indicator in indicators.items():
            indicator_dir = output_dir / name
            indicator_dir.mkdir(exist_ok=True)
            
            for scenario in scenarios:
                try:
                    data = self.validator.validation_dataset.load_dataset(scenario)
                    result = indicator.calculate(data)
                    
                    # Save result values
                    values_file = indicator_dir / f"{scenario}_values.csv"
                    if isinstance(result.values, pd.DataFrame):
                        result.values.to_csv(values_file)
                    else:
                        result.values.to_frame('value').to_csv(values_file)
                    
                    # Save metadata
                    metadata_file = indicator_dir / f"{scenario}_metadata.json"
                    metadata = {
                        'confidence': result.confidence,
                        'metadata': result.metadata,
                        'indicator_type': type(indicator).__name__
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    
                except Exception as e:
                    logger.error(f"Failed to save baseline for {name}/{scenario}: {e}")
        
        logger.info(f"Baseline results saved to {output_dir}")


def validate_indicator_suite(indicators: Dict[str, BaseIndicator],
                           output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to validate a suite of indicators.
    
    Args:
        indicators: Dictionary of indicator name to indicator instance
        output_file: Optional file to save the report
        
    Returns:
        Validation report dictionary
    """
    generator = ValidationReportGenerator()
    return generator.generate_suite_report(indicators, output_file)


if __name__ == "__main__":
    # Example usage
    from ..vectorized import RSI, MACD, ATR
    
    # Create test indicators
    test_indicators = {
        'RSI_14': RSI(period=14),
        'RSI_21': RSI(period=21),
        'MACD': MACD(),
        'ATR_14': ATR(period=14)
    }
    
    # Run validation
    print("Running indicator validation...")
    report = validate_indicator_suite(test_indicators, "validation_report.json")
    
    # Print summary
    if 'summary' in report:
        summary = report['summary']
        print(f"\nValidation Summary:")
        print(f"  Total indicators: {summary.get('total_indicators', 0)}")
        print(f"  Average score: {summary.get('average_score', 0):.3f}")
        print(f"  Passing indicators: {summary.get('passing_indicators', 0)}")
        print(f"  Suite passed: {summary.get('suite_passed', False)}")
    else:
        print("No summary available")