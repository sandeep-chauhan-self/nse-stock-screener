"""
Enhanced data validation framework for ensuring data quality and integrity.

FS.2 Data Infrastructure & ETL - Validation & Health Checks Implementation:
- Comprehensive validation rules (freshness, consistency, discrepancy detection)
- Health monitoring with alerting for data pipeline components
- Cross-provider discrepancy detection with configurable tolerance
- Standardized error handling and validation result reporting
- Corporate action detection and data quality scoring
- Automated freshness monitoring with T+1 compliance alerts
"""

import math
import json
import logging
from datetime import datetime
from typing import Optional, Any, Tuple, Union, TypeVar

import numpy as np
import pandas as pd

# Import validation framework components
from .validation_core import (
    ValidationLevel, ValidationStatus, ValidationIssue,
    EnhancedValidationResult, ValidationRule
)

# Import specific validation rules
from .validation_rules import (
    FreshnessCheck, EnhancedConsistencyCheck, CrossProviderDiscrepancyCheck
)

# Configure logger
logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnhancedDataValidator(object):
    """Enhanced data validation engine with comprehensive rule management."""

    def __init__(self) -> None:
        self.rules: list[ValidationRule] = []
        self.add_default_rules()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_default_rules(self) -> None:
        """Add enhanced validation rules for FS.2 compliance."""
        self.rules = [
            # T+1 compliance
            FreshnessCheck(max_age_hours=25, market_hours_only=True),
            EnhancedConsistencyCheck(),
            CrossProviderDiscrepancyCheck(tolerance_pct=2.0)
        ]

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)
        self.logger.info(f"Added validation rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """Remove a validation rule by name."""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        removed_count = original_count - len(self.rules)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} validation rule(s): {rule_name}")

    def validate_data(self, symbol: str, data: pd.DataFrame, data_type: str = "ohlcv",
                     metadata: dict[str, Any] = None) -> EnhancedValidationResult:
        """Validate data using all applicable rules."""
        result = EnhancedValidationResult(symbol=symbol, data_type=data_type, status=ValidationStatus.PASSED)
        result.metadata.update({
            'data_rows': len(data),
            'data_columns': list(data.columns) if not data.empty else [],
            'validation_timestamp': datetime.now().isoformat()
        })

        for rule in self.rules:
            if not rule.enabled or not rule.is_applicable(symbol, data_type):
                continue

            try:
                issues = rule.validate(symbol, data, metadata)
                result.issues.extend(issues)

                # Log rule execution
                if issues:
                    self.logger.debug(f"Rule {rule.name} found {len(issues)} issues for {symbol}")

            except Exception as e:
                self.logger.error(f"Validation rule {rule.name} failed for {symbol}: {e}")
                result.add_issue(
                    rule_name=rule.name,
                    level=ValidationLevel.ERROR,
                    message=f"Validation rule execution failed: {e}",
                    error_type="rule_execution_failure"
                )

        # Determine overall status
        if result.has_errors:
            result.status = ValidationStatus.FAILED

        # Add summary metadata
        result.metadata.update({
            'total_issues': len(result.issues),
            'critical_issues': len(result.get_issues_by_level(ValidationLevel.CRITICAL)),
            'error_issues': len(result.get_issues_by_level(ValidationLevel.ERROR)),
            'warning_issues': len(result.get_issues_by_level(ValidationLevel.WARNING)),
            'info_issues': len(result.get_issues_by_level(ValidationLevel.INFO))
        })

        return result

    def validate_multiple_sources(self, symbol: str, data_sources: dict[str, pd.DataFrame],
                                 data_type: str = "ohlcv") -> EnhancedValidationResult:
        """Validate data from multiple sources with cross-provider analysis."""
        result = self._initialize_multi_source_result(symbol, data_type, data_sources)

        # Run cross-provider discrepancy checks
        self._run_cross_provider_checks(symbol, data_sources, result)

        # Run single-source validations on each source
        self._run_single_source_validations(symbol, data_sources, data_type, result)

        # Finalize result
        self._finalize_multi_source_result(result, data_sources)

        return result

    @staticmethod
    def _initialize_multi_source_result(symbol: str, data_type: str,
                                       data_sources: dict[str, pd.DataFrame]) -> EnhancedValidationResult:
        """Initialize validation result for multi-source analysis."""
        result = EnhancedValidationResult(symbol=symbol, data_type=data_type, status=ValidationStatus.PASSED)
        result.metadata.update({
            'sources': list(data_sources.keys()),
            'source_data_counts': {src: len(data) for src, data in data_sources.items()},
            'validation_timestamp': datetime.now().isoformat()
        })
        return result

    def _run_cross_provider_checks(self, symbol: str, data_sources: dict[str, pd.DataFrame],
                                  result: EnhancedValidationResult) -> None:
        """Run cross-provider discrepancy checks."""
        for rule in self.rules:
            if not (isinstance(rule, CrossProviderDiscrepancyCheck) and rule.enabled):
                continue

            try:
                issues = rule.validate_multiple_sources(symbol, data_sources)
                result.issues.extend(issues)

                if issues:
                    self.logger.info(f"Cross-provider analysis found {len(issues)} discrepancies for {symbol}")

            except Exception as e:
                self.logger.error(f"Multi-source validation failed for {symbol}: {e}")
                result.add_issue(
                    rule_name=rule.name,
                    level=ValidationLevel.ERROR,
                    message=f"Multi-source validation failed: {e}",
                    error_type="multi_source_validation_failure"
                )

    def _run_single_source_validations(self, symbol: str, data_sources: dict[str, pd.DataFrame],
                                      data_type: str, result: EnhancedValidationResult) -> None:
        """Run single-source validations on each data source."""
        for source_name, data in data_sources.items():
            if data.empty:
                continue

            single_result = self.validate_data(symbol, data, data_type)

            # Add source context to issues
            for issue in single_result.issues:
                issue.metadata['data_source'] = source_name

            result.issues.extend(single_result.issues)

    @staticmethod
    def _finalize_multi_source_result(result: EnhancedValidationResult,
                                     data_sources: dict[str, pd.DataFrame]) -> None:
        """Finalize multi-source validation result."""
        # Determine overall status
        if result.has_errors:
            result.status = ValidationStatus.FAILED

        # Add summary metadata
        result.metadata.update({
            'total_issues': len(result.issues),
            'issues_by_source': {src: len([i for i in result.issues
                                         if i.metadata.get('data_source') == src])
                                for src in data_sources.keys()}
        })

    def create_validation_report(self, results: list[EnhancedValidationResult]) -> dict[str, Any]:
        """Create comprehensive validation report with analytics."""
        if not results:
            return {"error": "No validation results provided"}

        total_symbols = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)

        all_issues = []
        for result in results:
            all_issues.extend(result.issues)

        # Issues by level
        issues_by_level = {}
        for level in ValidationLevel:
            issues_by_level[level.value] = len([i for i in all_issues if i.level == level])

        # Issues by rule
        issues_by_rule = {}
        for issue in all_issues:
            rule_name = issue.rule_name
            if rule_name not in issues_by_rule:
                issues_by_rule[rule_name] = 0
            issues_by_rule[rule_name] += 1

        # Failed symbols analysis
        failed_symbols = [r.symbol for r in results if r.status == ValidationStatus.FAILED]
        critical_symbols = [r.symbol for r in results
                           if any(i.level == ValidationLevel.CRITICAL for i in r.issues)]

        report = {
            "summary": {
                "total_symbols": total_symbols,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "pass_rate": (passed / total_symbols * 100) if total_symbols > 0 else 0,
                "validation_timestamp": datetime.now().isoformat()
            },
            "quality_metrics": {
                "issues_by_level": issues_by_level,
                "issues_by_rule": dict(sorted(issues_by_rule.items(),
                                            key=lambda x: x[1], reverse=True)),
                "total_issues": len(all_issues),
                "avg_issues_per_symbol": len(all_issues) / total_symbols if total_symbols > 0 else 0
            },
            "problem_symbols": {
                # Top 20
                "failed_symbols": failed_symbols[:20],
                # Top 10
                "critical_symbols": critical_symbols[:10],
                "symbols_with_warnings": [r.symbol for r in results if r.has_warnings][:15]
            },
            "recommendations": self._generate_recommendations(results),
            "detailed_issues": self._get_top_issues(all_issues, top_n=15)
        }

        return report

    @staticmethod
    def _generate_recommendations(results: list[EnhancedValidationResult]) -> list[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Analyze common issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)

        # Count issue types
        freshness_issues = len([i for i in all_issues if 'freshness' in i.rule_name])
        consistency_issues = len([i for i in all_issues if 'consistency' in i.rule_name])
        discrepancy_issues = len([i for i in all_issues if 'discrepancy' in i.rule_name])

        total_symbols = len(results)

        # >10% have freshness issues
        if freshness_issues > total_symbols * 0.1:
            recommendations.append("⚠️ Data freshness issues detected - "
                                  "review data update schedules and T+1 compliance")

        # >5% have consistency issues
        if consistency_issues > total_symbols * 0.05:
            recommendations.append("🔍 Data consistency problems found - "
                                  "validate OHLCV relationships and check for data corruption")

        if discrepancy_issues > 0:
            recommendations.append("📊 Provider discrepancies detected - "
                                  "review data source reliability and consider implementing consensus mechanisms")

        # Check for critical issues
        critical_count = len([i for i in all_issues if i.level == ValidationLevel.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count} critical issues require immediate attention")

        # General health recommendations
        pass_rate = len([r for r in results if r.status == ValidationStatus.PASSED]) / total_symbols * 100
        if pass_rate < 95:
            quality_msg = (f"📈 Overall data quality ({pass_rate:.1f}% pass rate) needs improvement - "
                          f"consider stricter data validation at ingestion")
            recommendations.append(quality_msg)

        if not recommendations:
            recommendations.append("✅ Data quality appears healthy - continue monitoring")

        return recommendations

    @staticmethod
    def _get_top_issues(issues: list[ValidationIssue], top_n: int = 15) -> list[dict[str, Any]]:
        """Get most common and impactful validation issues."""
        issue_patterns = {}

        for issue in issues:
            # Create pattern key based on rule and message type
            pattern_key = f"{issue.rule_name}:{issue.level.value}"

            if pattern_key not in issue_patterns:
                issue_patterns[pattern_key] = {
                    "rule_name": issue.rule_name,
                    "level": issue.level.value,
                    "count": 0,
                    "affected_symbols": set(),
                    "example_message": issue.message,
                    "first_seen": issue.timestamp,
                    "last_seen": issue.timestamp
                }

            pattern = issue_patterns[pattern_key]
            pattern["count"] += 1
            pattern["affected_symbols"].add(issue.symbol)
            pattern["last_seen"] = max(pattern["last_seen"], issue.timestamp)

        # Sort by impact (critical > error > warning > info, then by count)
        level_priority = {
            "critical": 4, "error": 3, "warning": 2, "info": 1
        }

        sorted_patterns = sorted(
            issue_patterns.values(),
            key=lambda x: (level_priority.get(x["level"], 0), x["count"]),
            reverse=True
        )

        # Convert to serializable format
        top_issues = []
        for pattern in sorted_patterns[:top_n]:
            top_issues.append({
                "rule_name": pattern["rule_name"],
                "level": pattern["level"],
                "count": pattern["count"],
                "affected_symbols_count": len(pattern["affected_symbols"]),
                # First 5 symbols
                "example_symbols": list(pattern["affected_symbols"])[:5],
                "example_message": pattern["example_message"],
                "first_seen": pattern["first_seen"].isoformat(),
                "last_seen": pattern["last_seen"].isoformat()
            })

        return top_issues

    def validate(self, data: pd.DataFrame, symbol: str) -> dict[str, Any]:
        """
        Backward compatibility method for legacy interface.

        Args:
            data: OHLCV DataFrame to validate
            symbol: Symbol being validated

        Returns:
            Validation report dict with status and issues
        """
        result = self.validate_data(symbol, data)

        # Convert to legacy format
        status = "valid"
        if result.has_errors:
            status = "error"
        elif result.has_warnings:
            status = "warning"

        issues = [issue.message for issue in result.issues]

        return {
            "status": status,
            "issues": issues,
            "symbol": symbol,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata
        }

    @staticmethod
    def validate_indicators_dict(indicators: Optional[dict[str, Any]],
                                symbol: str = "UNKNOWN") -> Optional[dict[str, Any]]:
        """
        Backward compatibility method for validating indicator dictionaries.

        Args:
            indicators: Dictionary of indicators to validate
            symbol: Symbol being validated

        Returns:
            Validated and cleaned indicators dictionary, or None on critical failure
        """
        # Use the standalone function for backward compatibility
        return validate_indicators_dict(indicators, symbol)


# Standalone utility functions for backward compatibility

def validate_data_types(data: pd.DataFrame) -> bool:
    """Validate that data contains required numeric columns."""
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_columns):
        return False
    
    try:
        for col in required_columns:
            pd.to_numeric(data[col], errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def detect_data_anomalies(data: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Detect anomalies in OHLCV data."""
    anomalies = {
        'gaps': [],
        'spikes': [],
        'inconsistencies': [],
        'zero_volume_days': 0
    }
    
    if data.empty or len(data) < 2:
        return anomalies
    
    try:
        # Detect price spikes (>20% daily change)
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().abs()
            spike_threshold = 0.20
            spike_days = data[returns > spike_threshold]
            if not spike_days.empty:
                anomalies['spikes'] = spike_days.index.tolist()
        
        # Detect zero volume days
        if 'Volume' in data.columns:
            zero_volume = (data['Volume'] == 0).sum()
            anomalies['zero_volume_days'] = int(zero_volume)
        
        # Detect OHLC inconsistencies
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            inconsistent = (
                (data['High'] < data[['Open', 'Close']].max(axis=1)) |
                (data['Low'] > data[['Open', 'Close']].min(axis=1))
            )
            if inconsistent.any():
                anomalies['inconsistencies'] = data[inconsistent].index.tolist()
                
    except Exception as e:
        logger.warning(f"Anomaly detection failed for {symbol}: {e}")
    
    return anomalies


def validate_indicators_dict(indicators: Optional[dict[str, Any]],
                            symbol: str = "UNKNOWN") -> Optional[dict[str, Any]]:
    """
    Validate and clean a dictionary of technical indicators.
    
    Args:
        indicators: Dictionary containing technical indicator values
        symbol: Symbol being validated (for logging)
        
    Returns:
        Cleaned indicators dictionary, or None if critical validation fails
    """
    if indicators is None:
        return None
        
    cleaned = {}
    
    try:
        for key, value in indicators.items():
            processed_value = _process_indicator_value(key, value, symbol)
            if processed_value is not None:
                cleaned[key] = processed_value
                
    except Exception as e:
        logger.error(f"Failed to validate indicators for {symbol}: {e}")
        return None
        
    return cleaned if cleaned else None


def _process_indicator_value(key: str, value: Union[int, float, Any], symbol: str) -> Optional[Union[int, float]]:
    """Process and validate a single indicator value."""
    # Skip None values or invalid numeric values
    if value is None or not _is_valid_numeric_value(value):
        if value is not None:
            logger.debug(f"Skipping invalid indicator {key} for {symbol}: {value}")
        return None
        
    # Handle numpy types
    converted_value = _convert_numpy_value(value)
    if converted_value is None:
        return None
        
    # Validate numeric ranges for common indicators
    return converted_value if _validate_indicator_range(key, converted_value, symbol) else None


def _is_valid_numeric_value(value: Union[int, float, Any]) -> bool:
    """Check if value is a valid numeric value (not NaN or inf)."""
    if isinstance(value, (int, float)):
        return not (math.isnan(value) or math.isinf(value))
    return True


def _convert_numpy_value(value: Union[int, float, Any]) -> Optional[Union[int, float]]:
    """Convert numpy types to Python types."""
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            return None
    return value


def _validate_indicator_range(key: str, value: Union[int, float], symbol: str) -> bool:
    """Validate that indicator value is within expected range."""
    key_lower = key.lower()
    
    if key_lower in ['rsi']:
        if not 0 <= value <= 100:
            logger.warning(f"RSI value {value} out of range [0,100] for {symbol}")
            return False
            
    elif key_lower in ['bb_percent', 'stoch_k', 'stoch_d']:
        if not 0 <= value <= 100:
            logger.warning(f"{key} value {value} out of expected range [0,100] for {symbol}")
            # Don't skip, as these can occasionally go outside normal range
            
    return True


def calculate_data_quality_score(validation_result: EnhancedValidationResult) -> float:
    """
    Calculate a data quality score (0-100) based on validation results.
    
    Args:
        validation_result: Enhanced validation result
        
    Returns:
        Quality score from 0 (worst) to 100 (best)
    """
    if validation_result.status == ValidationStatus.PASSED and not validation_result.issues:
        return 100.0
    
    score = 100.0
    
    # Deduct points based on issue severity
    for issue in validation_result.issues:
        if issue.level == ValidationLevel.CRITICAL:
            # Critical issues are severe
            score -= 25.0
        elif issue.level == ValidationLevel.ERROR:
            score -= 10.0
        elif issue.level == ValidationLevel.WARNING:
            score -= 5.0
        elif issue.level == ValidationLevel.INFO:
            score -= 1.0
    
    # Ensure score doesn't go below 0
    return max(0.0, score)
