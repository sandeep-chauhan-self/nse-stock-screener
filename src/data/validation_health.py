"""
Health monitoring components for data validation.

This module contains health monitoring infrastructure:
- HealthCheckResult: Data structure for health check results
- DataHealthMonitor: Main health monitoring coordinator
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

import pandas as pd

from .validation_core import ValidationLevel, EnhancedValidationResult

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult(object):
    """Result of health check operation."""
    
    passed: bool
    message: str
    details: dict[str, Any]
    timestamp: datetime
    check_type: str
    
    def __post_init__(self) -> None:
        """Ensure timestamp is set if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict


class DataHealthMonitor(object):
    """Monitor overall data health across validation results."""

    def __init__(self, alert_thresholds: dict[str, float] = None) -> None:
        """Initialize health monitor."""
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.health_history: list[HealthCheckResult] = []
        
    @staticmethod
    def _get_default_thresholds() -> dict[str, float]:
        """Get default alert thresholds."""
        return {
            'critical_error_rate': 5.0,  # %
            'total_error_rate': 20.0,    # %
            'data_staleness_hours': 25.0, # hours
            'validation_failure_rate': 10.0  # %
        }

    def assess_overall_health(self, validation_results: list[EnhancedValidationResult]) -> HealthCheckResult:
        """Assess overall health of data pipeline."""
        if not validation_results:
            return HealthCheckResult(
                passed=False,
                message="No validation results available for health assessment",
                details={},
                timestamp=datetime.now(),
                check_type="overall_health"
            )

        # Calculate health metrics
        metrics = self._calculate_health_metrics(validation_results)
        
        # Determine health status
        health_status = self._determine_health_status(metrics)
        
        # Generate health message
        message = self._generate_health_message(health_status, metrics)
        
        result = HealthCheckResult(
            passed=health_status['is_healthy'],
            message=message,
            details={
                'metrics': metrics,
                'thresholds': self.alert_thresholds,
                'failing_checks': health_status['failing_checks']
            },
            timestamp=datetime.now(),
            check_type="overall_health"
        )
        
        self.health_history.append(result)
        return result

    def _calculate_health_metrics(self, results: list[EnhancedValidationResult]) -> dict[str, float]:
        """Calculate health metrics from validation results."""
        total_symbols = len(results)
        total_issues = sum(len(result.issues) for result in results)
        
        critical_issues = sum(1 for result in results
                              for issue in result.issues
                              if issue.level == ValidationLevel.CRITICAL)

        error_issues = sum(1 for result in results
                          for issue in result.issues
                          if issue.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR])
        
        failed_validations = sum(1 for result in results if not result.passed)
        
        # Calculate rates
        critical_rate = (critical_issues / total_symbols * 100) if total_symbols > 0 else 0
        error_rate = (error_issues / total_symbols * 100) if total_symbols > 0 else 0
        failure_rate = (failed_validations / total_symbols * 100) if total_symbols > 0 else 0
        
        # Calculate staleness metrics
        staleness_info = self._calculate_staleness_metrics(results)
        
        return {
            'total_symbols': total_symbols,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'error_issues': error_issues,
            'failed_validations': failed_validations,
            'critical_error_rate': critical_rate,
            'total_error_rate': error_rate,
            'validation_failure_rate': failure_rate,
            'average_staleness_hours': staleness_info['avg_staleness'],
            'max_staleness_hours': staleness_info['max_staleness'],
            'stale_data_count': staleness_info['stale_count']
        }

    def _calculate_staleness_metrics(self, results: list[EnhancedValidationResult]) -> dict[str, float]:
        """Calculate data staleness metrics."""
        staleness_values = []
        stale_count = 0
        
        for result in results:
            for issue in result.issues:
                if 'age_hours' in issue.metadata:
                    age_hours = issue.metadata['age_hours']
                    staleness_values.append(age_hours)
                    if age_hours > self.alert_thresholds['data_staleness_hours']:
                        stale_count += 1
        
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        max_staleness = max(staleness_values) if staleness_values else 0
        
        return {
            'avg_staleness': avg_staleness,
            'max_staleness': max_staleness,
            'stale_count': stale_count
        }

    def _determine_health_status(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Determine overall health status."""
        failing_checks = []
        
        # Check critical error rate
        if metrics['critical_error_rate'] > self.alert_thresholds['critical_error_rate']:
            failing_checks.append({
                'check': 'critical_error_rate',
                'value': metrics['critical_error_rate'],
                'threshold': self.alert_thresholds['critical_error_rate']
            })
        
        # Check total error rate
        if metrics['total_error_rate'] > self.alert_thresholds['total_error_rate']:
            failing_checks.append({
                'check': 'total_error_rate',
                'value': metrics['total_error_rate'],
                'threshold': self.alert_thresholds['total_error_rate']
            })
        
        # Check validation failure rate
        if metrics['validation_failure_rate'] > self.alert_thresholds['validation_failure_rate']:
            failing_checks.append({
                'check': 'validation_failure_rate',
                'value': metrics['validation_failure_rate'],
                'threshold': self.alert_thresholds['validation_failure_rate']
            })
        
        # Check data staleness
        if metrics['max_staleness_hours'] > self.alert_thresholds['data_staleness_hours']:
            failing_checks.append({
                'check': 'data_staleness_hours',
                'value': metrics['max_staleness_hours'],
                'threshold': self.alert_thresholds['data_staleness_hours']
            })
        
        is_healthy = len(failing_checks) == 0
        
        return {
            'is_healthy': is_healthy,
            'failing_checks': failing_checks,
            'health_score': self._calculate_health_score(metrics)
        }

    def _calculate_health_score(self, metrics: dict[str, float]) -> float:
        """Calculate a health score from 0-100."""
        # Start with perfect score
        score = 100.0
        
        # Deduct points for issues
        # Max 30 points deduction
        score -= min(metrics['critical_error_rate'] * 2, 30)
        # Max 25 points deduction
        score -= min(metrics['total_error_rate'], 25)
        # Max 20 points deduction
        score -= min(metrics['validation_failure_rate'], 20)
        
        # Deduct for staleness
        if metrics['max_staleness_hours'] > self.alert_thresholds['data_staleness_hours']:
            staleness_penalty = min((metrics['max_staleness_hours'] -
                                   self.alert_thresholds['data_staleness_hours']) / 24 * 10, 25)
            score -= staleness_penalty
        
        return max(score, 0.0)

    @staticmethod
    def _generate_health_message(health_status: dict[str, Any],
                                metrics: dict[str, float]) -> str:
        """Generate health status message."""
        if health_status['is_healthy']:
            return (f"Data pipeline is healthy. Processed {metrics['total_symbols']} symbols "
                   f"with {metrics['total_issues']} total issues. "
                   f"Health score: {health_status['health_score']:.1f}/100")
        
        failing_checks = health_status['failing_checks']
        failed_check_names = [check['check'] for check in failing_checks]
        
        return (f"Data pipeline health issues detected. "
               f"Failing checks: {', '.join(failed_check_names)}. "
               f"Health score: {health_status['health_score']:.1f}/100")

    def get_health_trend(self, hours: int = 24) -> dict[str, Any]:
        """Get health trend over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return {
                'trend': 'no_data',
                'message': f"No health checks in the last {hours} hours"
            }
        
        # Calculate trend
        health_scores = [check.details.get('metrics', {}).get('health_score', 0)
                        for check in recent_checks
                        if 'metrics' in check.details]
        
        if len(health_scores) < 2:
            return {
                'trend': 'insufficient_data',
                'message': "Insufficient data for trend analysis"
            }
        
        # Simple trend calculation
        first_half = health_scores[:len(health_scores)//2]
        second_half = health_scores[len(health_scores)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        trend_direction = 'improving' if avg_second > avg_first else 'declining'
        trend_magnitude = abs(avg_second - avg_first)
        
        return {
            'trend': trend_direction,
            'magnitude': trend_magnitude,
            'current_score': health_scores[-1],
            'period_avg': sum(health_scores) / len(health_scores),
            'checks_count': len(recent_checks),
            'message': f"Health trend over {hours}h: {trend_direction} "
                      f"(magnitude: {trend_magnitude:.1f} points)"
        }

    def save_health_report(self, filepath: Union[str, Path]) -> None:
        """Save health history to file."""
        filepath = Path(filepath)
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'thresholds': self.alert_thresholds,
            'health_history': [check.to_dict() for check in self.health_history]
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Health report saved to {filepath}")

    def load_health_history(self, filepath: Union[str, Path]) -> None:
        """Load health history from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Health history file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            self.alert_thresholds = report_data.get('thresholds', self.alert_thresholds)
            
            # Load health history
            for check_data in report_data.get('health_history', []):
                check_data['timestamp'] = datetime.fromisoformat(check_data['timestamp'])
                self.health_history.append(HealthCheckResult(**check_data))
            
            logger.info(f"Loaded {len(self.health_history)} health check records from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading health history from {filepath}: {e}")

    def clear_old_history(self, days: int = 30) -> None:
        """Clear health history older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        old_count = len(self.health_history)
        
        self.health_history = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time
        ]
        
        new_count = len(self.health_history)
        removed_count = old_count - new_count
        
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} old health check records (older than {days} days)")

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics of health monitoring."""
        if not self.health_history:
            return {'message': 'No health check history available'}
        
        total_checks = len(self.health_history)
        passed_checks = sum(1 for check in self.health_history if check.passed)
        failed_checks = total_checks - passed_checks
        
        recent_check = self.health_history[-1] if self.health_history else None
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'last_check': recent_check.to_dict() if recent_check else None,
            'monitoring_period_days': (
                (self.health_history[-1].timestamp - self.health_history[0].timestamp).days
                if len(self.health_history) > 1 else 0
            )
        }
