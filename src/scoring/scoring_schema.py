"""
Scoring Configuration Schema Definition

This module defines the schema and validation for configuration-driven scoring.
It supports YAML/JSON scoring configurations with weights, methods, lookbacks,
and bonus/penalty rules as specified in FS.4 requirements.

Usage:
    from src.scoring.scoring_schema import ScoringSchema, ScoringConfig
    
    # Load and validate config
    schema = ScoringSchema()
    config = schema.load_config("config.yaml")
    
    # Use config in scoring engine
    engine = ScoringEngine(config)
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Supported scoring methods for components."""
    ZSCORE = "zscore"
    PERCENTILE = "percentile"
    THRESHOLD = "threshold"
    LINEAR = "linear"
    LOG_SCALE = "log_scale"
    CUSTOM = "custom"


class ConditionOperator(Enum):
    """Operators for bonus/penalty conditions."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based scoring."""
    levels: List[Dict[str, Union[float, int]]]  # [{"min": 0, "max": 25, "score": 5}, ...]
    default_score: float = 0.0
    
    def __post_init__(self):
        """Validate threshold configuration."""
        if not self.levels:
            raise ValueError("Threshold levels cannot be empty")
        
        for level in self.levels:
            required_keys = {"score"}
            if not required_keys.issubset(level.keys()):
                raise ValueError(f"Threshold level must contain keys: {required_keys}")


@dataclass
class PercentileConfig:
    """Configuration for percentile-based scoring."""
    lookback_periods: int = 252  # Trading days for percentile calculation
    percentile_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 25.0,
        "medium": 50.0,
        "high": 75.0,
        "extreme": 95.0
    })
    scores: Dict[str, float] = field(default_factory=lambda: {
        "low": 2.0,
        "medium": 5.0,
        "high": 8.0,
        "extreme": 10.0
    })


@dataclass
class ZScoreConfig:
    """Configuration for z-score based scoring."""
    lookback_periods: int = 252
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0
    })
    scores: Dict[str, float] = field(default_factory=lambda: {
        "low": 3.0,
        "medium": 6.0,
        "high": 10.0
    })
    cap_extreme: bool = True  # Cap extreme values


@dataclass
class LinearConfig:
    """Configuration for linear scaling."""
    min_value: float
    max_value: float
    min_score: float = 0.0
    max_score: float = 10.0
    invert: bool = False  # Invert scaling (higher values = lower scores)


@dataclass
class CustomConfig:
    """Configuration for custom scoring functions."""
    function_name: str  # Name of custom function to call
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BonusPenaltyRule:
    """Configuration for bonus/penalty rules."""
    name: str
    description: str
    condition: str  # e.g., "atr < 2.0", "rsi between 60 80", "sector in ['FINANCE', 'TECH']"
    operator: ConditionOperator
    value: Union[float, int, str, List[Union[float, int, str]]]
    bonus: float = 0.0  # Positive for bonus, negative for penalty
    max_applications: int = 1  # How many times this rule can be applied
    enabled: bool = True
    
    def __post_init__(self):
        """Validate bonus/penalty rule."""
        if abs(self.bonus) < 1e-9:  # Use epsilon comparison for floating point
            raise ValueError("Bonus/penalty value cannot be zero")


@dataclass
class ComponentConfig:
    """Configuration for a single scoring component."""
    name: str
    weight: float
    method: ScoringMethod
    lookback: int = 20
    enabled: bool = True
    
    # Method-specific configurations
    threshold_config: Optional[ThresholdConfig] = None
    percentile_config: Optional[PercentileConfig] = None
    zscore_config: Optional[ZScoreConfig] = None
    linear_config: Optional[LinearConfig] = None
    custom_config: Optional[CustomConfig] = None
    
    # Component-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Data source mapping
    indicator_key: str = ""  # Key in indicators dict
    fallback_keys: List[str] = field(default_factory=list)  # Fallback indicator keys
    
    def __post_init__(self):
        """Validate component configuration."""
        if self.weight < 0:
            raise ValueError(f"Component weight must be non-negative: {self.weight}")
        
        if not self.indicator_key and not self.fallback_keys:
            raise ValueError(f"Component {self.name} must have indicator_key or fallback_keys")
        
        # Validate method-specific config is provided
        method_configs = {
            ScoringMethod.THRESHOLD: self.threshold_config,
            ScoringMethod.PERCENTILE: self.percentile_config,
            ScoringMethod.ZSCORE: self.zscore_config,
            ScoringMethod.LINEAR: self.linear_config,
            ScoringMethod.CUSTOM: self.custom_config
        }
        
        required_config = method_configs.get(self.method)
        if required_config is None and self.method != ScoringMethod.CUSTOM:
            raise ValueError(f"Component {self.name} method {self.method.value} requires specific configuration")


@dataclass
class RegimeAdjustment:
    """Market regime-specific adjustments."""
    regime: str  # MarketRegime value
    weight_multipliers: Dict[str, float] = field(default_factory=dict)  # Component name -> multiplier
    threshold_adjustments: Dict[str, float] = field(default_factory=dict)  # Component name -> adjustment
    bonus_penalty_multiplier: float = 1.0
    enabled: bool = True


@dataclass
class ScoringConfig:
    """Complete scoring configuration."""
    # Metadata
    name: str
    version: str
    description: str
    created_by: str
    created_at: str
    
    # Core configuration
    components: List[ComponentConfig]
    bonus_penalty_rules: List[BonusPenaltyRule] = field(default_factory=list)
    regime_adjustments: List[RegimeAdjustment] = field(default_factory=list)
    
    # Global settings
    max_total_score: float = 100.0
    min_total_score: float = 0.0
    probability_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 70.0,
        "medium": 45.0,
        "low": 0.0
    })
    
    # Validation settings
    require_minimum_data: bool = True
    minimum_indicators: int = 3
    fail_on_missing_data: bool = False
    
    # Performance settings
    cache_intermediate_results: bool = True
    parallel_processing: bool = False
    
    def __post_init__(self):
        """Validate complete scoring configuration."""
        if not self.components:
            raise ValueError("Scoring configuration must have at least one component")
        
        # Validate total weights
        total_weight = sum(comp.weight for comp in self.components if comp.enabled)
        if total_weight <= 0:
            raise ValueError("Total component weights must be positive")
        
        # Validate component names are unique
        component_names = [comp.name for comp in self.components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("Component names must be unique")
        
        # Validate probability thresholds
        thresholds = self.probability_thresholds
        if not (thresholds["low"] <= thresholds["medium"] <= thresholds["high"]):
            raise ValueError("Probability thresholds must be in ascending order")
    
    def get_config_hash(self) -> str:
        """Generate MD5 hash of configuration for tracking."""
        # Create a normalized dict representation
        config_dict = {
            "name": self.name,
            "version": self.version,
            "components": [
                {
                    "name": comp.name,
                    "weight": comp.weight,
                    "method": comp.method.value,
                    "lookback": comp.lookback,
                    "enabled": comp.enabled,
                    "indicator_key": comp.indicator_key,
                    "parameters": comp.parameters
                }
                for comp in self.components
            ],
            "bonus_penalty_rules": [
                {
                    "name": rule.name,
                    "condition": rule.condition,
                    "operator": rule.operator.value,
                    "value": rule.value,
                    "bonus": rule.bonus,
                    "enabled": rule.enabled
                }
                for rule in self.bonus_penalty_rules
            ],
            "probability_thresholds": self.probability_thresholds,
            "max_total_score": self.max_total_score
        }
        
        # Generate hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_enabled_components(self) -> List[ComponentConfig]:
        """Get list of enabled components."""
        return [comp for comp in self.components if comp.enabled]
    
    def get_component_by_name(self, name: str) -> Optional[ComponentConfig]:
        """Get component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None


class ScoringSchema:
    """Schema loader and validator for scoring configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Union[str, Path]) -> ScoringConfig:
        """
        Load and validate scoring configuration from file.
        
        Args:
            config_path: Path to YAML or JSON configuration file
            
        Returns:
            Validated ScoringConfig object
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration based on file extension
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return self._parse_config(config_data)
    
    def _parse_config(self, config_data: Dict[str, Any]) -> ScoringConfig:
        """Parse and validate configuration data."""
        try:
            # Parse components
            components = []
            for comp_data in config_data.get('components', []):
                component = self._parse_component(comp_data)
                components.append(component)
            
            # Parse bonus/penalty rules
            bonus_penalty_rules = []
            for rule_data in config_data.get('bonus_penalty_rules', []):
                rule = self._parse_bonus_penalty_rule(rule_data)
                bonus_penalty_rules.append(rule)
            
            # Parse regime adjustments
            regime_adjustments = []
            for regime_data in config_data.get('regime_adjustments', []):
                adjustment = self._parse_regime_adjustment(regime_data)
                regime_adjustments.append(adjustment)
            
            # Create main config
            config = ScoringConfig(
                name=config_data.get('name', 'Unnamed'),
                version=config_data.get('version', '1.0'),
                description=config_data.get('description', ''),
                created_by=config_data.get('created_by', 'Unknown'),
                created_at=config_data.get('created_at', ''),
                components=components,
                bonus_penalty_rules=bonus_penalty_rules,
                regime_adjustments=regime_adjustments,
                max_total_score=config_data.get('max_total_score', 100.0),
                min_total_score=config_data.get('min_total_score', 0.0),
                probability_thresholds=config_data.get('probability_thresholds', {
                    "high": 70.0, "medium": 45.0, "low": 0.0
                }),
                require_minimum_data=config_data.get('require_minimum_data', True),
                minimum_indicators=config_data.get('minimum_indicators', 3),
                fail_on_missing_data=config_data.get('fail_on_missing_data', False),
                cache_intermediate_results=config_data.get('cache_intermediate_results', True),
                parallel_processing=config_data.get('parallel_processing', False)
            )
            
            self.logger.info(f"Successfully loaded scoring configuration: {config.name} v{config.version}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error parsing scoring configuration: {e}")
            raise ValueError(f"Invalid scoring configuration: {e}")
    
    def _parse_component(self, comp_data: Dict[str, Any]) -> ComponentConfig:
        """Parse component configuration."""
        method_str = comp_data.get('method', 'threshold')
        try:
            method = ScoringMethod(method_str)
        except ValueError:
            raise ValueError(f"Invalid scoring method: {method_str}")
        
        # Parse method-specific configuration
        method_configs = {}
        
        if method == ScoringMethod.THRESHOLD and 'threshold_config' in comp_data:
            method_configs['threshold_config'] = ThresholdConfig(**comp_data['threshold_config'])
        
        elif method == ScoringMethod.PERCENTILE and 'percentile_config' in comp_data:
            method_configs['percentile_config'] = PercentileConfig(**comp_data['percentile_config'])
        
        elif method == ScoringMethod.ZSCORE and 'zscore_config' in comp_data:
            method_configs['zscore_config'] = ZScoreConfig(**comp_data['zscore_config'])
        
        elif method == ScoringMethod.LINEAR and 'linear_config' in comp_data:
            method_configs['linear_config'] = LinearConfig(**comp_data['linear_config'])
        
        elif method == ScoringMethod.CUSTOM and 'custom_config' in comp_data:
            method_configs['custom_config'] = CustomConfig(**comp_data['custom_config'])
        
        return ComponentConfig(
            name=comp_data['name'],
            weight=comp_data['weight'],
            method=method,
            lookback=comp_data.get('lookback', 20),
            enabled=comp_data.get('enabled', True),
            indicator_key=comp_data.get('indicator_key', ''),
            fallback_keys=comp_data.get('fallback_keys', []),
            parameters=comp_data.get('parameters', {}),
            **method_configs
        )
    
    def _parse_bonus_penalty_rule(self, rule_data: Dict[str, Any]) -> BonusPenaltyRule:
        """Parse bonus/penalty rule configuration."""
        operator_str = rule_data.get('operator', '>')
        try:
            operator = ConditionOperator(operator_str)
        except ValueError:
            raise ValueError(f"Invalid condition operator: {operator_str}")
        
        return BonusPenaltyRule(
            name=rule_data['name'],
            description=rule_data.get('description', ''),
            condition=rule_data['condition'],
            operator=operator,
            value=rule_data['value'],
            bonus=rule_data['bonus'],
            max_applications=rule_data.get('max_applications', 1),
            enabled=rule_data.get('enabled', True)
        )
    
    def _parse_regime_adjustment(self, regime_data: Dict[str, Any]) -> RegimeAdjustment:
        """Parse regime adjustment configuration."""
        return RegimeAdjustment(
            regime=regime_data['regime'],
            weight_multipliers=regime_data.get('weight_multipliers', {}),
            threshold_adjustments=regime_data.get('threshold_adjustments', {}),
            bonus_penalty_multiplier=regime_data.get('bonus_penalty_multiplier', 1.0),
            enabled=regime_data.get('enabled', True)
        )
    
    def save_config(self, config: ScoringConfig, output_path: Union[str, Path]) -> None:
        """
        Save scoring configuration to file.
        
        Args:
            config: ScoringConfig to save
            output_path: Output file path (YAML or JSON)
        """
        output_path = Path(output_path)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        # Save based on file extension
        if output_path.suffix.lower() in ['.yml', '.yaml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        self.logger.info(f"Saved scoring configuration to: {output_path}")
    
    def _config_to_dict(self, config: ScoringConfig) -> Dict[str, Any]:
        """Convert ScoringConfig to dictionary for serialization."""
        # This is a simplified conversion - a full implementation would
        # need to handle all nested dataclasses properly
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "created_by": config.created_by,
            "created_at": config.created_at,
            "components": [
                {
                    "name": comp.name,
                    "weight": comp.weight,
                    "method": comp.method.value,
                    "lookback": comp.lookback,
                    "enabled": comp.enabled,
                    "indicator_key": comp.indicator_key,
                    "fallback_keys": comp.fallback_keys,
                    "parameters": comp.parameters
                }
                for comp in config.components
            ],
            "bonus_penalty_rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "operator": rule.operator.value,
                    "value": rule.value,
                    "bonus": rule.bonus,
                    "max_applications": rule.max_applications,
                    "enabled": rule.enabled
                }
                for rule in config.bonus_penalty_rules
            ],
            "probability_thresholds": config.probability_thresholds,
            "max_total_score": config.max_total_score,
            "min_total_score": config.min_total_score
        }


# Example configuration factory
def create_default_config() -> ScoringConfig:
    """Create a default scoring configuration based on current composite_scorer.py."""
    from datetime import datetime
    
    components = [
        ComponentConfig(
            name="momentum",
            weight=0.4,
            method=ScoringMethod.ZSCORE,
            lookback=90,
            indicator_key="momentum_composite",
            fallback_keys=["rsi", "macd"],
            zscore_config=ZScoreConfig(
                lookback_periods=90,
                thresholds={"low": 1.0, "medium": 2.0, "high": 3.0},
                scores={"low": 5.0, "medium": 15.0, "high": 25.0}
            )
        ),
        ComponentConfig(
            name="volume",
            weight=0.2,
            method=ScoringMethod.PERCENTILE,
            lookback=20,
            indicator_key="vol_ratio",
            fallback_keys=["vol_z"],
            percentile_config=PercentileConfig(
                lookback_periods=20,
                percentile_thresholds={"low": 50.0, "medium": 75.0, "high": 90.0, "extreme": 95.0},
                scores={"low": 2.0, "medium": 5.0, "high": 8.0, "extreme": 10.0}
            )
        )
    ]
    
    bonus_rules = [
        BonusPenaltyRule(
            name="atr_risk_adjust",
            description="Bonus for low volatility stocks",
            condition="atr_pct < 2.0",
            operator=ConditionOperator.LESS_THAN,
            value=2.0,
            bonus=0.1
        )
    ]
    
    return ScoringConfig(
        name="Default NSE Scoring",
        version="1.0",
        description="Default configuration based on existing composite scorer",
        created_by="System",
        created_at=datetime.now().isoformat(),
        components=components,
        bonus_penalty_rules=bonus_rules
    )