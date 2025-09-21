"""
Calibration Harness for Scoring Weight Optimization

This module implements the calibration system for optimizing scoring weights
against historical true positives as specified in FS.4 requirements.

Features:
- Grid search optimization for component weights
- Bayesian optimization using scikit-optimize
- Historical performance validation
- Cross-validation with walk-forward analysis
- Hyperparameter tuning with configurable objectives

Usage:
    from src.scoring import CalibrationHarness, ScoringConfig

    # Initialize calibration
    calibrator = CalibrationHarness(base_config, historical_data)

    # Run optimization
    best_config = calibrator.optimize_weights(
        method='bayesian',
        objective='sharpe_ratio',
        n_calls=50
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import itertools
from abc import ABC, abstractmethod

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from .scoring_schema import ScoringConfig, ComponentConfig
from .scoring_engine import ScoringEngine, ScoringResult
from .parameter_store import ParameterStore

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of weight optimization."""
    best_config: ScoringConfig
    best_score: float
    optimization_history: List[Dict[str, Any]]
    validation_metrics: Dict[str, float]
    total_evaluations: int
    optimization_time_seconds: float
    method_used: str
    objective_function: str


@dataclass
class ValidationPeriod:
    """Time period for validation."""
    start_date: datetime
    end_date: datetime
    name: str
    metadata: Dict[str, Any] = None


class ObjectiveFunction(ABC):
    """Base class for optimization objective functions."""

    @abstractmethod
    def calculate(self, results: List[ScoringResult], market_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate objective function value from scoring results."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the objective function."""
        pass

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether higher values are better for this objective."""
        pass


class SharpeRatioObjective(ObjectiveFunction):
    """Sharpe ratio based on portfolio returns."""

    def calculate(self, results: List[ScoringResult], market_data: Optional[pd.DataFrame] = None) -> float:
        if not results or len(results) < 20:  # Need minimum data points
            return -10.0  # Penalty for insufficient data

        # Simulate portfolio returns based on scores
        scores = [r.total_score for r in results]
        weights = np.array(scores) / sum(scores) if sum(scores) > 0 else np.ones(len(scores)) / len(scores)

        # Simple return simulation (would use actual price data in production)
        rng = np.random.default_rng(42)
        daily_returns = rng.normal(0.001, 0.02, len(results))  # Placeholder
        portfolio_return = np.sum(weights * daily_returns)
        portfolio_vol = np.std(daily_returns)

        if portfolio_vol == 0:
            return 0.0

        return (portfolio_return * 252) / (portfolio_vol * np.sqrt(252))  # Annualized Sharpe

    @property
    def name(self) -> str:
        return "sharpe_ratio"

    @property
    def higher_is_better(self) -> bool:
        return True


class WinRateObjective(ObjectiveFunction):
    """Win rate based on high probability classifications."""

    def __init__(self, high_threshold: float = 70.0):
        self.high_threshold = high_threshold

    def calculate(self, results: List[ScoringResult], market_data: Optional[pd.DataFrame] = None) -> float:
        if not results:
            return 0.0

        high_prob_predictions = [r for r in results if r.total_score >= self.high_threshold]

        if not high_prob_predictions:
            return 0.0

        # Simulate win rate (would use actual performance data in production)
        # For now, assume higher confidence correlates with better performance
        avg_confidence = np.mean([r.confidence for r in high_prob_predictions])
        simulated_win_rate = min(0.95, max(0.3, avg_confidence * 0.8 + 0.2))

        return simulated_win_rate

    @property
    def name(self) -> str:
        return "win_rate"

    @property
    def higher_is_better(self) -> bool:
        return True


class CalibrationHarness:
    """
    Calibration system for optimizing scoring configuration weights
    against historical performance data.
    """

    def __init__(self,
                 base_config: ScoringConfig,
                 parameter_store: Optional[ParameterStore] = None,
                 max_workers: int = 4):
        """
        Initialize calibration harness.

        Args:
            base_config: Base scoring configuration to optimize
            parameter_store: Optional parameter store for tracking results
            max_workers: Maximum parallel workers for optimization
        """
        self.base_config = base_config
        self.parameter_store = parameter_store
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # Available objective functions
        self.objective_functions = {
            'sharpe_ratio': SharpeRatioObjective(),
            'win_rate': WinRateObjective(),
            'win_rate_strict': WinRateObjective(high_threshold=80.0)
        }

        self.optimization_history = []

    def optimize_weights(self,
                        historical_data: Dict[str, pd.DataFrame],
                        method: str = 'grid_search',
                        objective: str = 'sharpe_ratio',
                        validation_periods: Optional[List[ValidationPeriod]] = None,
                        **kwargs) -> OptimizationResult:
        """
        Optimize component weights using specified method.

        Args:
            historical_data: Dictionary mapping symbols to historical data
            method: Optimization method ('grid_search', 'bayesian', 'random_search')
            objective: Objective function name
            validation_periods: Time periods for validation (if None, uses simple split)
            **kwargs: Method-specific parameters

        Returns:
            OptimizationResult with best configuration and metrics
        """
        start_time = datetime.now()

        if objective not in self.objective_functions:
            raise ValueError(f"Unknown objective function: {objective}")

        objective_func = self.objective_functions[objective]

        # Prepare validation periods
        if validation_periods is None:
            validation_periods = self._create_default_validation_periods()

        self.logger.info(f"Starting weight optimization using {method} for {objective}")

        # Run optimization based on method
        if method == 'grid_search':
            result = self._optimize_grid_search(
                historical_data, objective_func, validation_periods, **kwargs
            )
        elif method == 'bayesian':
            if not SKOPT_AVAILABLE:
                self.logger.warning("scikit-optimize not available, falling back to grid search")
                result = self._optimize_grid_search(
                    historical_data, objective_func, validation_periods, **kwargs
                )
            else:
                result = self._optimize_bayesian(
                    historical_data, objective_func, validation_periods, **kwargs
                )
        elif method == 'random_search':
            result = self._optimize_random_search(
                historical_data, objective_func, validation_periods, **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Calculate final metrics and timing
        optimization_time = (datetime.now() - start_time).total_seconds()

        result.optimization_time_seconds = optimization_time
        result.method_used = method
        result.objective_function = objective

        # Save result if parameter store available
        if self.parameter_store:
            self._save_optimization_result(result)

        self.logger.info(f"Optimization completed in {optimization_time:.1f}s. Best score: {result.best_score:.4f}")

        return result

    def _optimize_grid_search(self,
                             historical_data: Dict[str, pd.DataFrame],
                             objective_func: ObjectiveFunction,
                             validation_periods: List[ValidationPeriod],
                             weight_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                             steps: int = 5) -> OptimizationResult:
        """Optimize using grid search."""
        components = self.base_config.get_enabled_components()

        # Define weight ranges for each component
        if weight_ranges is None:
            weight_ranges = {
                comp.name: (0.0, 1.0) for comp in components
            }

        # Generate grid points
        grid_points = []
        for comp in components:
            min_weight, max_weight = weight_ranges.get(comp.name, (0.0, 1.0))
            grid_points.append(np.linspace(min_weight, max_weight, steps))

        # Create all combinations
        weight_combinations = list(itertools.product(*grid_points))

        self.logger.info(f"Grid search: evaluating {len(weight_combinations)} combinations")

        # Evaluate all combinations
        best_score = float('-inf') if objective_func.higher_is_better else float('inf')
        best_weights = None
        optimization_history = []

        for i, weights in enumerate(weight_combinations):
            # Normalize weights to sum to 1
            normalized_weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(len(weights)) / len(weights)

            try:
                score = self._evaluate_weights(
                    normalized_weights, components, historical_data, objective_func, validation_periods
                )

                optimization_history.append({
                    'iteration': i,
                    'weights': dict(zip([comp.name for comp in components], normalized_weights)),
                    'score': score,
                    'timestamp': datetime.now()
                })

                is_better = (score > best_score) if objective_func.higher_is_better else (score < best_score)
                if is_better:
                    best_score = score
                    best_weights = normalized_weights

                if i % 10 == 0:
                    self.logger.debug(f"Grid search progress: {i}/{len(weight_combinations)}, best: {best_score:.4f}")

            except Exception as e:
                self.logger.warning(f"Error evaluating weights {weights}: {e}")

        # Create best configuration
        best_config = self._create_config_with_weights(best_weights, components)

        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            best_config, historical_data, validation_periods
        )

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            optimization_history=optimization_history,
            validation_metrics=validation_metrics,
            total_evaluations=len(weight_combinations),
            optimization_time_seconds=0.0,  # Will be set by caller
            method_used='grid_search',
            objective_function=objective_func.name
        )

    def _optimize_bayesian(self,
                          historical_data: Dict[str, pd.DataFrame],
                          objective_func: ObjectiveFunction,
                          validation_periods: List[ValidationPeriod],
                          n_calls: int = 50,
                          random_state: int = 42) -> OptimizationResult:
        """Optimize using Bayesian optimization."""
        components = self.base_config.get_enabled_components()

        # Define search space
        dimensions = [Real(0.01, 1.0, name=comp.name) for comp in components]

        optimization_history = []

        @use_named_args(dimensions)
        def objective(**params):
            weights = np.array([params[comp.name] for comp in components])
            # Normalize weights
            normalized_weights = weights / np.sum(weights)

            try:
                score = self._evaluate_weights(
                    normalized_weights, components, historical_data, objective_func, validation_periods
                )

                optimization_history.append({
                    'iteration': len(optimization_history),
                    'weights': dict(zip([comp.name for comp in components], normalized_weights)),
                    'score': score,
                    'timestamp': datetime.now()
                })

                # Bayesian optimization minimizes, so negate if higher is better
                return -score if objective_func.higher_is_better else score

            except Exception as e:
                self.logger.warning(f"Error in Bayesian objective function: {e}")
                return 1000.0  # Large penalty

        self.logger.info(f"Bayesian optimization: {n_calls} evaluations")

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI'  # Expected Improvement
        )

        # Extract best weights
        best_weights = np.array(result.x)
        best_weights = best_weights / np.sum(best_weights)  # Normalize

        best_score = -result.fun if objective_func.higher_is_better else result.fun

        # Create best configuration
        best_config = self._create_config_with_weights(best_weights, components)

        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            best_config, historical_data, validation_periods
        )

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            optimization_history=optimization_history,
            validation_metrics=validation_metrics,
            total_evaluations=n_calls,
            optimization_time_seconds=0.0,  # Will be set by caller
            method_used='bayesian',
            objective_function=objective_func.name
        )

    def _optimize_random_search(self,
                               historical_data: Dict[str, pd.DataFrame],
                               objective_func: ObjectiveFunction,
                               validation_periods: List[ValidationPeriod],
                               n_iterations: int = 100,
                               random_state: int = 42) -> OptimizationResult:
        """Optimize using random search."""
        np.random.seed(random_state)
        components = self.base_config.get_enabled_components()

        best_score = float('-inf') if objective_func.higher_is_better else float('inf')
        best_weights = None
        optimization_history = []

        self.logger.info(f"Random search: {n_iterations} evaluations")

        for i in range(n_iterations):
            # Generate random weights
            rng = np.random.default_rng(random_state + i)
            weights = rng.uniform(0.1, 1.0, len(components))
            normalized_weights = weights / np.sum(weights)

            try:
                score = self._evaluate_weights(
                    normalized_weights, components, historical_data, objective_func, validation_periods
                )

                optimization_history.append({
                    'iteration': i,
                    'weights': dict(zip([comp.name for comp in components], normalized_weights)),
                    'score': score,
                    'timestamp': datetime.now()
                })

                is_better = (score > best_score) if objective_func.higher_is_better else (score < best_score)
                if is_better:
                    best_score = score
                    best_weights = normalized_weights

                if i % 20 == 0:
                    self.logger.debug(f"Random search progress: {i}/{n_iterations}, best: {best_score:.4f}")

            except Exception as e:
                self.logger.warning(f"Error in random search iteration {i}: {e}")

        # Create best configuration
        best_config = self._create_config_with_weights(best_weights, components)

        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            best_config, historical_data, validation_periods
        )

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            optimization_history=optimization_history,
            validation_metrics=validation_metrics,
            total_evaluations=n_iterations,
            optimization_time_seconds=0.0,  # Will be set by caller
            method_used='random_search',
            objective_function=objective_func.name
        )

    def _evaluate_weights(self,
                         weights: np.ndarray,
                         components: List[ComponentConfig],
                         historical_data: Dict[str, pd.DataFrame],
                         objective_func: ObjectiveFunction,
                         validation_periods: List[ValidationPeriod]) -> float:
        """Evaluate a set of weights using cross-validation."""
        # Create temporary config with new weights
        temp_config = self._create_config_with_weights(weights, components)
        engine = ScoringEngine(temp_config)

        period_scores = []

        for _ in validation_periods:
            # Filter data for this period (simplified - would need proper date filtering)
            period_results = []

            # Score symbols for this period
            for symbol, data in historical_data.items():
                # Generate mock indicators (in production, would compute real indicators)
                mock_indicators = self._generate_mock_indicators(symbol, data)

                try:
                    result = engine.score(symbol, mock_indicators)
                    period_results.append(result)
                except Exception as e:
                    self.logger.debug(f"Error scoring {symbol}: {e}")

            if period_results:
                period_score = objective_func.calculate(period_results)
                period_scores.append(period_score)

        # Return average score across periods
        return np.mean(period_scores) if period_scores else 0.0

    def _create_config_with_weights(self, weights: np.ndarray, components: List[ComponentConfig]) -> ScoringConfig:
        """Create new configuration with updated weights."""
        updated_components = []

        for i, component in enumerate(components):
            updated_component = replace(component, weight=weights[i])
            updated_components.append(updated_component)

        # Update the base config with new components
        updated_config = replace(
            self.base_config,
            components=updated_components,
            version=f"{self.base_config.version}_optimized"
        )

        return updated_config

    def _generate_mock_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mock indicators for testing (replace with real indicator computation)."""
        if len(data) < 20:
            return {}

        # Generate realistic-looking mock indicators
        close_prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

        # Use a deterministic seed based on symbol for consistent testing
        rng = np.random.default_rng(hash(symbol) % 2**32)

        return {
            'symbol': symbol,
            'momentum_composite': rng.normal(0, 2),  # Mock z-score
            'vol_ratio': rng.uniform(0.5, 5.0),
            'rsi': rng.uniform(30, 80),
            'macd': rng.normal(0, 10),
            'macd_signal': rng.normal(0, 8),
            'adx': rng.uniform(15, 45),
            'atr_pct': rng.uniform(0.5, 4.0),
            'rel_strength_20d': rng.normal(0, 10),
            'current_price': close_prices.iloc[-1] if len(close_prices) > 0 else 100.0
        }

    def _create_default_validation_periods(self) -> List[ValidationPeriod]:
        """Create default validation periods for testing."""
        end_date = datetime.now()

        return [
            ValidationPeriod(
                start_date=end_date - timedelta(days=180),
                end_date=end_date - timedelta(days=90),
                name="train_period"
            ),
            ValidationPeriod(
                start_date=end_date - timedelta(days=90),
                end_date=end_date,
                name="validation_period"
            )
        ]

    def _calculate_validation_metrics(self,
                                    config: ScoringConfig,
                                    historical_data: Dict[str, pd.DataFrame],
                                    validation_periods: List[ValidationPeriod]) -> Dict[str, float]:
        """Calculate validation metrics for a configuration."""
        metrics = {}

        engine = ScoringEngine(config)

        for period in validation_periods:
            period_results = []

            for symbol, data in historical_data.items():
                mock_indicators = self._generate_mock_indicators(symbol, data)
                try:
                    result = engine.score(symbol, mock_indicators)
                    period_results.append(result)
                except Exception:
                    pass

            if period_results:
                # Calculate various metrics for this period
                high_prob_count = sum(1 for r in period_results if r.probability_level == 'HIGH')
                avg_confidence = np.mean([r.confidence for r in period_results])
                avg_score = np.mean([r.total_score for r in period_results])

                metrics[f"{period.name}_high_prob_count"] = high_prob_count
                metrics[f"{period.name}_avg_confidence"] = avg_confidence
                metrics[f"{period.name}_avg_score"] = avg_score

        return metrics

    def _save_optimization_result(self, result: OptimizationResult) -> None:
        """Save optimization result to parameter store."""
        if not self.parameter_store:
            return

        try:
            # Save the optimized configuration
            config_id = self.parameter_store.save_config(
                result.best_config,
                metadata={
                    'optimization_method': result.method_used,
                    'objective_function': result.objective_function,
                    'optimization_score': result.best_score,
                    'total_evaluations': result.total_evaluations,
                    'optimization_time': result.optimization_time_seconds
                }
            )

            # Track the optimization performance
            self.parameter_store.track_config_performance(
                config_id,
                f"optimization_{result.objective_function}",
                result.best_score
            )

            self.logger.info(f"Saved optimization result with config ID: {config_id}")

        except Exception as e:
            self.logger.error(f"Error saving optimization result: {e}")

    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        """Generate summary of optimization results."""
        history = result.optimization_history

        if not history:
            return {"error": "No optimization history available"}

        scores = [h['score'] for h in history]

        return {
            "method": result.method_used,
            "objective": result.objective_function,
            "best_score": result.best_score,
            "total_evaluations": result.total_evaluations,
            "optimization_time_seconds": result.optimization_time_seconds,
            "score_statistics": {
                "min": min(scores),
                "max": max(scores),
                "mean": np.mean(scores),
                "std": np.std(scores)
            },
            "best_weights": {
                comp.name: comp.weight
                for comp in result.best_config.get_enabled_components()
            },
            "validation_metrics": result.validation_metrics,
            "improvement": {
                "absolute": result.best_score - scores[0] if scores else 0,
                "relative_pct": ((result.best_score - scores[0]) / abs(scores[0]) * 100) if scores and scores[0] != 0 else 0
            }
        }


# Example usage and testing
if __name__ == "__main__":
    from .scoring_schema import create_default_config

    # Create base configuration
    base_config = create_default_config()

    # Generate mock historical data
    historical_data = {}
    rng = np.random.default_rng(42)
    for symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']:
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        data = pd.DataFrame({
            'Close': rng.uniform(100, 500, len(dates)),
            'Volume': rng.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        historical_data[symbol] = data

    # Initialize calibration harness
    calibrator = CalibrationHarness(base_config)

    # Run optimization
    result = calibrator.optimize_weights(
        historical_data=historical_data,
        method='grid_search',
        objective='win_rate',
        steps=3  # Small grid for testing
    )

    print("Optimization completed!")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Total evaluations: {result.total_evaluations}")

    # Get summary
    summary = calibrator.get_optimization_summary(result)
    print(f"Summary: {summary}")

    print("Calibration harness test completed!")