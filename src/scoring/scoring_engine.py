"""
Configuration-Driven Scoring Engine

This module implements the main scoring engine that reads YAML/JSON configurations
and applies dynamic scoring logic as specified in FS.4 requirements.

The engine supports:
- Dynamic component scoring based on configuration
- Bonus/penalty rule evaluation
- Market regime adjustments
- Parallel processing for performance
- Comprehensive audit trails

Usage:
    from src.scoring import ScoringEngine, ScoringSchema
    
    # Load configuration
    schema = ScoringSchema()
    config = schema.load_config("config.yaml")
    
    # Create and use engine
    engine = ScoringEngine(config)
    result = engine.score(symbol="RELIANCE", indicators=indicators, regime=regime)
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
from dataclasses import dataclass, asdict

from .scoring_schema import (
    ScoringConfig, ComponentConfig, BonusPenaltyRule, RegimeAdjustment,
    ScoringMethod, ConditionOperator, ThresholdConfig, PercentileConfig,
    ZScoreConfig, LinearConfig, CustomConfig
)

try:
    from ..common.enums import MarketRegime, ProbabilityLevel
    from ..common.interfaces import ScoreBreakdown
    from ..data.validation import safe_float, safe_bool, is_valid_numeric
except ImportError:
    # Fallback imports for standalone testing
    from src.common.enums import MarketRegime, ProbabilityLevel
    from src.common.interfaces import ScoreBreakdown
    from src.data.validation import safe_float, safe_bool, is_valid_numeric

logger = logging.getLogger(__name__)


@dataclass
class ComponentScore:
    """Individual component scoring result."""
    name: str
    raw_score: float
    weighted_score: float
    confidence: float
    method_used: str
    indicator_key: str
    raw_value: Union[float, int, bool, None]
    metadata: Dict[str, Any]


@dataclass
class BonusPenaltyResult:
    """Bonus/penalty rule evaluation result."""
    rule_name: str
    applied: bool
    value: float
    condition_met: bool
    condition_value: Union[float, int, str, bool, None]
    metadata: Dict[str, Any]


@dataclass
class ScoringResult:
    """Complete scoring result with detailed breakdown."""
    symbol: str
    total_score: float
    probability_level: str
    market_regime: str
    config_hash: str
    timestamp: datetime
    
    # Component details
    component_scores: List[ComponentScore]
    bonus_penalty_results: List[BonusPenaltyResult]
    
    # Metadata
    total_weight: float
    normalized_score: float
    confidence: float
    regime_adjustments_applied: bool
    
    # Audit trail
    indicators_used: List[str]
    indicators_missing: List[str]
    validation_warnings: List[str]
    processing_time_ms: float


class ScoringEngine:
    """
    Configuration-driven scoring engine that evaluates stock indicators
    against flexible, configurable scoring rules.
    """
    
    def __init__(self, config: ScoringConfig):
        """
        Initialize scoring engine with configuration.
        
        Args:
            config: Validated ScoringConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance optimizations
        self._cache = {} if config.cache_intermediate_results else None
        self._compiled_conditions = {}  # Pre-compile bonus/penalty conditions
        
        # Pre-process configuration for performance
        self._preprocess_config()
        
        self.logger.info(f"Initialized scoring engine: {config.name} v{config.version}")
    
    def _preprocess_config(self) -> None:
        """Pre-process configuration for optimal runtime performance."""
        # Pre-compile regex patterns for bonus/penalty conditions
        for rule in self.config.bonus_penalty_rules:
            if rule.enabled:
                try:
                    # Parse condition string into structured format
                    parsed_condition = self._parse_condition(rule.condition)
                    self._compiled_conditions[rule.name] = parsed_condition
                except Exception as e:
                    self.logger.warning(f"Failed to parse condition for rule {rule.name}: {e}")
        
        # Validate component indicator mappings
        for component in self.config.get_enabled_components():
            if not component.indicator_key and not component.fallback_keys:
                self.logger.warning(f"Component {component.name} has no indicator mapping")
    
    def _parse_condition(self, condition: str) -> Dict[str, Any]:
        """Parse condition string into structured format for evaluation."""
        condition = condition.strip()
        
        # Try parsing different condition types
        parsed = self._try_parse_between(condition)
        if parsed:
            return parsed
            
        parsed = self._try_parse_in_operator(condition)
        if parsed:
            return parsed
            
        parsed = self._try_parse_comparison(condition)
        if parsed:
            return parsed
        
        raise ValueError(f"Could not parse condition: {condition}")
    
    def _try_parse_between(self, condition: str) -> Optional[Dict[str, Any]]:
        """Try to parse 'between' operator condition."""
        between_match = re.match(r'(\w+)\s+between\s+([0-9.]+)\s+([0-9.]+)', condition, re.IGNORECASE)
        if between_match:
            return {
                "indicator": between_match.group(1),
                "operator": "between",
                "value": [float(between_match.group(2)), float(between_match.group(3))]
            }
        return None
    
    def _try_parse_in_operator(self, condition: str) -> Optional[Dict[str, Any]]:
        """Try to parse 'in' operator condition."""
        in_match = re.match(r'(\w+)\s+in\s+\[(.*?)\]', condition, re.IGNORECASE)
        if in_match:
            values_str = in_match.group(2)
            values = []
            for val in values_str.split(','):
                val = val.strip().strip("'\"")
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(val)
            
            return {
                "indicator": in_match.group(1),
                "operator": "in",
                "value": values
            }
        return None
    
    def _try_parse_comparison(self, condition: str) -> Optional[Dict[str, Any]]:
        """Try to parse standard comparison operators."""
        operators = ['>=', '<=', '==', '!=', '>', '<']
        for op in operators:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    indicator = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str.strip("'\"")
                    
                    return {
                        "indicator": indicator,
                        "operator": op,
                        "value": value
                    }
        return None
    
    def score(self, 
             symbol: str, 
             indicators: Dict[str, Any], 
             regime: MarketRegime = MarketRegime.SIDEWAYS,
             metadata: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score a symbol using indicators and current market regime."""
        start_time = datetime.now()
        
        try:
            # Initialize scoring context
            scoring_context = self._initialize_scoring_context(symbol)
            
            # Score components
            self._score_all_components(indicators, scoring_context)
            
            # Apply regime adjustments and bonuses
            final_score = self._calculate_final_score(scoring_context, regime, indicators)
            
            # Create and return result
            return self._create_scoring_result(scoring_context, final_score, regime, start_time)
            
        except Exception as e:
            self.logger.error(f"Error scoring {symbol}: {e}", exc_info=True)
            return self._create_error_result(symbol, regime, start_time, str(e), indicators)
    
    def _initialize_scoring_context(self, symbol: str) -> Dict[str, Any]:
        """Initialize scoring context for tracking results."""
        return {
            'symbol': symbol,
            'component_scores': [],
            'bonus_penalty_results': [],
            'validation_warnings': [],
            'indicators_used': [],
            'indicators_missing': [],
            'total_raw_score': 0.0,
            'total_weight': 0.0
        }
    
    def _score_all_components(self, indicators: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Score all enabled components and update context."""
        for component in self.config.get_enabled_components():
            try:
                comp_result = self._score_component(component, indicators)
                context['component_scores'].append(comp_result)
                
                context['total_raw_score'] += comp_result.weighted_score
                context['total_weight'] += component.weight
                
                if comp_result.indicator_key:
                    context['indicators_used'].append(comp_result.indicator_key)
                    
            except Exception as e:
                self.logger.warning(f"Failed to score component {component.name}: {e}")
                context['validation_warnings'].append(f"Component {component.name}: {str(e)}")
                self._track_missing_indicators(component, indicators, context)
    
    def _track_missing_indicators(self, component: ComponentConfig, indicators: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Track missing indicators for a component."""
        missing_keys = [component.indicator_key] + component.fallback_keys
        for key in missing_keys:
            if key and key not in indicators:
                context['indicators_missing'].append(key)
    
    def _calculate_final_score(self, context: Dict[str, Any], regime: MarketRegime, indicators: Dict[str, Any]) -> float:
        """Calculate final score with regime adjustments and bonuses."""
        # Apply regime adjustments
        regime_adjusted_score, regime_applied = self._apply_regime_adjustments(
            context['total_raw_score'], context['component_scores'], regime
        )
        context['regime_adjustments_applied'] = regime_applied
        
        # Evaluate bonus/penalty rules
        bonus_penalty_total = self._evaluate_all_bonus_penalties(indicators, context)
        
        # Calculate final score with bounds
        final_score = regime_adjusted_score + bonus_penalty_total
        return max(self.config.min_total_score, min(self.config.max_total_score, final_score))
    
    def _evaluate_all_bonus_penalties(self, indicators: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate all bonus/penalty rules and return total adjustment."""
        bonus_penalty_total = 0.0
        
        for rule in self.config.bonus_penalty_rules:
            if rule.enabled:
                try:
                    bp_result = self._evaluate_bonus_penalty(rule, indicators)
                    context['bonus_penalty_results'].append(bp_result)
                    
                    if bp_result.applied:
                        bonus_penalty_total += bp_result.value
                        
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate rule {rule.name}: {e}")
                    context['validation_warnings'].append(f"Bonus/Penalty {rule.name}: {str(e)}")
        
        return bonus_penalty_total
    
    def _create_scoring_result(self, context: Dict[str, Any], final_score: float, regime: MarketRegime, start_time: datetime) -> ScoringResult:
        """Create the final scoring result from context."""
        normalized_score = (final_score / self.config.max_total_score) * 100.0
        probability_level = self._classify_probability(final_score)
        confidence = self._calculate_confidence(
            context['component_scores'], 
            context['indicators_used'], 
            context['indicators_missing']
        )
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ScoringResult(
            symbol=context['symbol'],
            total_score=final_score,
            probability_level=probability_level,
            market_regime=regime.value,
            config_hash=self.config.get_config_hash(),
            timestamp=start_time,
            component_scores=context['component_scores'],
            bonus_penalty_results=context['bonus_penalty_results'],
            total_weight=context['total_weight'],
            normalized_score=normalized_score,
            confidence=confidence,
            regime_adjustments_applied=context.get('regime_adjustments_applied', False),
            indicators_used=list(set(context['indicators_used'])),
            indicators_missing=list(set(context['indicators_missing'])),
            validation_warnings=context['validation_warnings'],
            processing_time_ms=processing_time
        )
    
    def _create_error_result(self, symbol: str, regime: MarketRegime, start_time: datetime, error_msg: str, indicators: Optional[Dict[str, Any]]) -> ScoringResult:
        """Create an error result when scoring fails."""
        return ScoringResult(
            symbol=symbol,
            total_score=0.0,
            probability_level="LOW",
            market_regime=regime.value,
            config_hash=self.config.get_config_hash(),
            timestamp=start_time,
            component_scores=[],
            bonus_penalty_results=[],
            total_weight=0.0,
            normalized_score=0.0,
            confidence=0.0,
            regime_adjustments_applied=False,
            indicators_used=[],
            indicators_missing=list(indicators.keys()) if indicators else [],
            validation_warnings=[f"Critical error: {error_msg}"],
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
        )
    
    def _score_component(self, component: ComponentConfig, indicators: Dict[str, Any]) -> ComponentScore:
        """Score a single component using its configuration."""
        # Get indicator value
        indicator_value, indicator_key = self._get_indicator_value(component, indicators)
        
        if indicator_value is None:
            return ComponentScore(
                name=component.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                method_used=component.method.value,
                indicator_key=indicator_key or "missing",
                raw_value=None,
                metadata={"error": "Indicator value not found"}
            )
        
        # Score using configured method
        raw_score, confidence, metadata = self._apply_scoring_method(
            component.method, indicator_value, component
        )
        
        # Apply weight
        weighted_score = raw_score * component.weight
        
        return ComponentScore(
            name=component.name,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            method_used=component.method.value,
            indicator_key=indicator_key or component.indicator_key,
            raw_value=indicator_value,
            metadata=metadata
        )
    
    def _get_indicator_value(self, component: ComponentConfig, indicators: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Get indicator value using primary key or fallbacks."""
        # Try primary indicator key
        if component.indicator_key and component.indicator_key in indicators:
            value = indicators[component.indicator_key]
            if is_valid_numeric(value) or isinstance(value, (bool, str)):
                return value, component.indicator_key
        
        # Try fallback keys
        for fallback_key in component.fallback_keys:
            if fallback_key in indicators:
                value = indicators[fallback_key]
                if is_valid_numeric(value) or isinstance(value, (bool, str)):
                    return value, fallback_key
        
        return None, None
    
    def _apply_scoring_method(self, 
                            method: ScoringMethod, 
                            value: Any, 
                            component: ComponentConfig) -> Tuple[float, float, Dict[str, Any]]:
        """Apply the configured scoring method to a value."""
        metadata = {"raw_value": value, "method": method.value}
        
        try:
            if method == ScoringMethod.THRESHOLD:
                return self._score_threshold(value, component.threshold_config, metadata)
            
            elif method == ScoringMethod.PERCENTILE:
                return self._score_percentile(value, component.percentile_config, metadata)
            
            elif method == ScoringMethod.ZSCORE:
                return self._score_zscore(value, component.zscore_config, metadata)
            
            elif method == ScoringMethod.LINEAR:
                return self._score_linear(value, component.linear_config, metadata)
            
            elif method == ScoringMethod.CUSTOM:
                return self._score_custom(component.custom_config, metadata)
            
            else:
                raise ValueError(f"Unsupported scoring method: {method}")
                
        except Exception as e:
            self.logger.warning(f"Error applying {method.value} scoring: {e}")
            metadata["error"] = str(e)
            return 0.0, 0.0, metadata
    
    def _score_threshold(self, value: Any, config: ThresholdConfig, metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Score using threshold-based method."""
        if not is_valid_numeric(value):
            return config.default_score, 0.0, metadata
        
        numeric_value = float(value)
        
        # Find matching threshold level
        for level in config.levels:
            min_val = level.get("min", float("-inf"))
            max_val = level.get("max", float("inf"))
            
            if min_val <= numeric_value <= max_val:
                score = level["score"]
                metadata.update({
                    "threshold_level": level,
                    "matched_range": f"{min_val} <= {numeric_value} <= {max_val}"
                })
                return float(score), 1.0, metadata
        
        # No threshold matched
        metadata["matched_range"] = "none"
        return config.default_score, 0.5, metadata
    
    def _score_percentile(self, value: Any, config: PercentileConfig, metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Score using percentile-based method."""
        if not is_valid_numeric(value):
            return 0.0, 0.0, metadata
        
        numeric_value = float(value)
        
        # For now, use static thresholds - in full implementation, 
        # this would calculate percentiles from historical data
        for level_name, percentile in sorted(config.percentile_thresholds.items(), 
                                           key=lambda x: x[1], reverse=True):
            # Simple percentile approximation - would need historical data for proper implementation
            if numeric_value >= percentile / 25.0:  # Rough approximation
                score = config.scores.get(level_name, 0.0)
                metadata.update({
                    "percentile_level": level_name,
                    "percentile_threshold": percentile,
                    "approximated": True
                })
                return score, 0.8, metadata
        
        # Below all thresholds
        lowest_level = min(config.scores.keys(), key=lambda k: config.percentile_thresholds[k])
        score = config.scores[lowest_level] * 0.5  # Reduced score for very low values
        metadata.update({
            "percentile_level": "below_minimum",
            "approximated": True
        })
        return score, 0.6, metadata
    
    def _score_zscore(self, value: Any, config: ZScoreConfig, metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Score using z-score based method."""
        if not is_valid_numeric(value):
            return 0.0, 0.0, metadata
        
        numeric_value = float(value)
        
        # Calculate z-score (simplified - would need historical data for proper mean/std)
        # For now, assume value itself is already a z-score or use simple thresholds
        abs_value = abs(numeric_value)
        
        for level_name, threshold in sorted(config.thresholds.items(), 
                                          key=lambda x: x[1], reverse=True):
            if abs_value >= threshold:
                score = config.scores.get(level_name, 0.0)
                metadata.update({
                    "zscore_level": level_name,
                    "zscore_threshold": threshold,
                    "abs_zscore": abs_value
                })
                return score, 0.9, metadata
        
        # Below all thresholds
        lowest_level = min(config.scores.keys(), key=lambda k: config.thresholds[k])
        score = config.scores[lowest_level] * 0.3
        metadata.update({
            "zscore_level": "below_minimum",
            "abs_zscore": abs_value
        })
        return score, 0.7, metadata
    
    def _score_linear(self, value: Any, config: LinearConfig, metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Score using linear scaling method."""
        if not is_valid_numeric(value):
            return config.min_score, 0.0, metadata
        
        numeric_value = float(value)
        
        # Clamp value to configured range
        clamped_value = max(config.min_value, min(config.max_value, numeric_value))
        
        # Linear interpolation
        value_range = config.max_value - config.min_value
        score_range = config.max_score - config.min_score
        
        if value_range == 0:
            score = config.min_score
        else:
            normalized = (clamped_value - config.min_value) / value_range
            if config.invert:
                normalized = 1.0 - normalized
            score = config.min_score + (normalized * score_range)
        
        metadata.update({
            "linear_config": asdict(config),
            "clamped_value": clamped_value,
            "normalized": normalized if 'normalized' in locals() else 0.0
        })
        
        return score, 1.0, metadata
    
    def _score_custom(self, config: CustomConfig, metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Score using custom function."""
        # This would call registered custom scoring functions
        # For now, return default score
        metadata.update({
            "custom_function": config.function_name,
            "parameters": config.parameters,
            "error": "Custom scoring not implemented yet"
        })
        return 0.0, 0.5, metadata
    
    def _apply_regime_adjustments(self, base_score: float, component_scores: List[ComponentScore], regime: MarketRegime) -> Tuple[float, bool]:
        """Apply market regime adjustments to the base score."""
        regime_adjustment = None
        
        # Find matching regime adjustment
        for adjustment in self.config.regime_adjustments:
            if adjustment.enabled and adjustment.regime == regime.value:
                regime_adjustment = adjustment
                break
        
        if not regime_adjustment:
            return base_score, False
        
        adjusted_score = base_score
        
        # Apply weight multipliers to component scores
        for comp_score in component_scores:
            multiplier = regime_adjustment.weight_multipliers.get(comp_score.name, 1.0)
            if abs(multiplier - 1.0) > 1e-9:  # Use epsilon comparison
                adjustment_amount = comp_score.weighted_score * (multiplier - 1.0)
                adjusted_score += adjustment_amount
        
        return adjusted_score, True
    
    def _evaluate_bonus_penalty(self, rule: BonusPenaltyRule, indicators: Dict[str, Any]) -> BonusPenaltyResult:
        """Evaluate a bonus/penalty rule against indicators."""
        if rule.name not in self._compiled_conditions:
            return self._create_bonus_penalty_error(rule.name, "Condition not compiled")
        
        condition = self._compiled_conditions[rule.name]
        indicator_key = condition["indicator"]
        
        # Check if indicator exists
        if indicator_key not in indicators:
            return self._create_bonus_penalty_error(rule.name, f"Indicator {indicator_key} not found")
        
        actual_value = indicators[indicator_key]
        
        try:
            condition_met = self._evaluate_condition(condition, actual_value)
            
            return BonusPenaltyResult(
                rule_name=rule.name,
                applied=condition_met,
                value=rule.bonus if condition_met else 0.0,
                condition_met=condition_met,
                condition_value=actual_value,
                metadata={
                    "operator": condition["operator"],
                    "expected_value": condition["value"],
                    "condition": rule.condition
                }
            )
            
        except Exception as e:
            return self._create_bonus_penalty_error(rule.name, f"Evaluation error: {e}", actual_value)
    
    def _create_bonus_penalty_error(self, rule_name: str, error_msg: str, actual_value: Any = None) -> BonusPenaltyResult:
        """Create a bonus/penalty error result."""
        return BonusPenaltyResult(
            rule_name=rule_name,
            applied=False,
            value=0.0,
            condition_met=False,
            condition_value=actual_value,
            metadata={"error": error_msg}
        )
    
    def _evaluate_condition(self, condition: Dict[str, Any], actual_value: Any) -> bool:
        """Evaluate a parsed condition against an actual value."""
        operator = condition["operator"]
        expected_value = condition["value"]
        
        if operator in [">", "<", ">=", "<=", "==", "!="]:
            return self._evaluate_numeric_condition(operator, actual_value, expected_value)
        elif operator == "between":
            return self._evaluate_between_condition(actual_value, expected_value)
        elif operator in ["in", "not_in"]:
            return self._evaluate_membership_condition(operator, actual_value, expected_value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def _evaluate_numeric_condition(self, operator: str, actual: Any, expected: Any) -> bool:
        """Evaluate numeric comparison conditions."""
        if not is_valid_numeric(actual):
            return False
        
        actual_float = float(actual)
        expected_float = float(expected)
        
        if operator == ">":
            return actual_float > expected_float
        elif operator == "<":
            return actual_float < expected_float
        elif operator == ">=":
            return actual_float >= expected_float
        elif operator == "<=":
            return actual_float <= expected_float
        elif operator == "==":
            return abs(actual_float - expected_float) < 1e-9
        elif operator == "!=":
            return abs(actual_float - expected_float) >= 1e-9
        else:
            return False
    
    def _evaluate_between_condition(self, actual: Any, expected: List[float]) -> bool:
        """Evaluate 'between' condition."""
        if not is_valid_numeric(actual) or len(expected) != 2:
            return False
        return expected[0] <= float(actual) <= expected[1]
    
    def _evaluate_membership_condition(self, operator: str, actual: Any, expected: List[Any]) -> bool:
        """Evaluate 'in' or 'not_in' conditions."""
        if operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected
        else:
            return False
    
    def _classify_probability(self, score: float) -> str:
        """Classify score into probability level."""
        thresholds = self.config.probability_thresholds
        
        if score >= thresholds["high"]:
            return "HIGH"
        elif score >= thresholds["medium"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence(self, component_scores: List[ComponentScore], indicators_used: List[str], indicators_missing: List[str]) -> float:
        """Calculate overall confidence based on data quality and component performance."""
        if not component_scores:
            return 0.0
        
        # Component confidence average
        component_confidence = sum(cs.confidence for cs in component_scores) / len(component_scores)
        
        # Data availability factor
        total_indicators = len(indicators_used) + len(indicators_missing)
        if total_indicators == 0:
            data_availability = 0.0
        else:
            data_availability = len(indicators_used) / total_indicators
        
        # Combined confidence
        overall_confidence = (component_confidence * 0.7) + (data_availability * 0.3)
        
        return min(1.0, max(0.0, overall_confidence))
    
    def score_batch(self, 
                   symbols_and_indicators: List[Tuple[str, Dict[str, Any]]], 
                   regime: MarketRegime = MarketRegime.SIDEWAYS,
                   max_workers: Optional[int] = None) -> List[ScoringResult]:
        """
        Score multiple symbols in parallel for improved performance.
        
        Args:
            symbols_and_indicators: List of (symbol, indicators) tuples
            regime: Market regime to apply
            max_workers: Maximum number of worker threads
            
        Returns:
            List of scoring results in same order as input
        """
        if not self.config.parallel_processing or len(symbols_and_indicators) < 4:
            # Process sequentially for small batches or if parallel processing disabled
            return [self.score(symbol, indicators, regime) 
                   for symbol, indicators in symbols_and_indicators]
        
        # Parallel processing
        max_workers = max_workers or min(8, len(symbols_and_indicators))
        results = [None] * len(symbols_and_indicators)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.score, symbol, indicators, regime): idx
                for idx, (symbol, indicators) in enumerate(symbols_and_indicators)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    symbol = symbols_and_indicators[idx][0]
                    self.logger.error(f"Error scoring {symbol} in batch: {e}")
                    # Create error result
                    results[idx] = ScoringResult(
                        symbol=symbol,
                        total_score=0.0,
                        probability_level="LOW",
                        market_regime=regime.value,
                        config_hash=self.config.get_config_hash(),
                        timestamp=datetime.now(),
                        component_scores=[],
                        bonus_penalty_results=[],
                        total_weight=0.0,
                        normalized_score=0.0,
                        confidence=0.0,
                        regime_adjustments_applied=False,
                        indicators_used=[],
                        indicators_missing=[],
                        validation_warnings=[f"Batch processing error: {str(e)}"],
                        processing_time_ms=0.0
                    )
        
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for reporting."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "config_hash": self.config.get_config_hash(),
            "components": len(self.config.get_enabled_components()),
            "bonus_penalty_rules": len([r for r in self.config.bonus_penalty_rules if r.enabled]),
            "regime_adjustments": len([r for r in self.config.regime_adjustments if r.enabled]),
            "probability_thresholds": self.config.probability_thresholds,
            "max_score": self.config.max_total_score,
            "parallel_processing": self.config.parallel_processing
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the scoring engine with a sample configuration
    from .scoring_schema import create_default_config
    
    # Create default configuration
    config = create_default_config()
    
    # Initialize engine
    engine = ScoringEngine(config)
    
    # Sample indicators
    sample_indicators = {
        'symbol': 'RELIANCE.NS',
        'momentum_composite': 2.5,  # Z-score for momentum
        'vol_ratio': 4.2,  # Volume ratio
        'rsi': 68.5,
        'macd': 15.2,
        'atr_pct': 1.8
    }
    
    # Score the sample
    result = engine.score(
        symbol="RELIANCE",
        indicators=sample_indicators,
        regime=MarketRegime.SIDEWAYS
    )
    
    print(f"Scoring Result for {result.symbol}:")
    print(f"Total Score: {result.total_score:.2f} ({result.probability_level})")
    print(f"Components: {len(result.component_scores)}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")