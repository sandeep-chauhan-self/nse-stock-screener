"""
Base classes and interfaces for high-performance indicator engine.

This module provides the foundational abstractions for building
vectorized, configurable technical indicators with comprehensive
validation and performance optimization.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class IndicatorType(Enum):
    """Indicator classification types."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    OVERLAY = "overlay"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Fail on any data quality issues
    NORMAL = "normal"      # Warn on minor issues, fail on major ones
    PERMISSIVE = "permissive"  # Continue with best-effort calculation


@dataclass
class IndicatorResult:
    """Standardized result container for indicator calculations."""
    
    # Core result data
    value: Union[float, Dict[str, float], np.ndarray] = math.nan
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    computation_time_ms: Optional[float] = None
    data_points_used: Optional[int] = None
    
    # Validation information
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if result is valid (has numeric value)."""
        if isinstance(self.value, dict):
            return any(not math.isnan(v) if isinstance(v, (int, float)) else False 
                      for v in self.value.values())
        elif isinstance(self.value, np.ndarray):
            return not np.isnan(self.value).all()
        else:
            return not math.isnan(self.value) if isinstance(self.value, (int, float)) else False
    
    @property
    def values(self) -> Union[pd.Series, pd.DataFrame]:
        """Convenience property to access result values as pandas object."""
        if isinstance(self.value, dict):
            # Convert dict to DataFrame for multi-column indicators
            return pd.DataFrame([self.value]) if len(self.value) > 1 else pd.Series(self.value)
        elif isinstance(self.value, (pd.Series, pd.DataFrame)):
            return self.value
        elif isinstance(self.value, np.ndarray):
            return pd.Series(self.value)
        else:
            # Single value - return as Series
            return pd.Series([self.value])
    
    @property
    def has_warnings(self) -> bool:
        """Check if result has warnings."""
        return len(self.warnings) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if result has errors."""
        return len(self.errors) > 0
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "computation_time_ms": self.computation_time_ms,
            "data_points_used": self.data_points_used
        }
        
        if self.warnings:
            result["warnings"] = self.warnings
        if self.errors:
            result["errors"] = self.errors
            
        return result


class IndicatorError(Exception):
    """Base exception for indicator-related errors."""
    pass


class InsufficientDataError(IndicatorError):
    """Raised when insufficient data is provided for calculation."""
    
    def __init__(self, required: int, available: int, indicator_name: str = ""):
        self.required = required
        self.available = available
        self.indicator_name = indicator_name
        super().__init__(
            f"Insufficient data for {indicator_name}: "
            f"required {required}, available {available}"
        )


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    Provides the standard interface and common functionality for
    high-performance, configurable indicator implementations.
    """
    
    def __init__(self, **params):
        """
        Initialize indicator with parameters.
        
        Args:
            **params: Indicator-specific parameters
        """
        self.params = params
        self.validation_level = ValidationLevel.NORMAL
        self._performance_stats = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this indicator instance."""
        pass
    
    @property
    @abstractmethod
    def indicator_type(self) -> IndicatorType:
        """Classification of this indicator."""
        pass
    
    @property
    @abstractmethod
    def required_periods(self) -> int:
        """Minimum number of data periods required for calculation."""
        pass
    
    @property
    def output_names(self) -> List[str]:
        """Names of output values (for multi-value indicators)."""
        return [self.name]
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """
        Compute indicator value(s) from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            **kwargs: Additional computation parameters
            
        Returns:
            IndicatorResult with computed values and metadata
        """
        pass
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """
        Convenience method that delegates to compute().
        
        Args:
            data: DataFrame with OHLCV columns
            **kwargs: Additional computation parameters
            
        Returns:
            IndicatorResult with computed values and metadata
        """
        return self.compute(data, **kwargs)
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data quality and completeness.
        
        Args:
            data: Input OHLCV DataFrame
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check minimum data requirement
        if len(data) < self.required_periods:
            errors.append(f"Insufficient data: {len(data)} < {self.required_periods}")
        
        # Check for required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for excessive missing values
        for col in required_cols:
            if col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct > 0.1:  # More than 10% missing
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(f"Too many missing values in {col}: {missing_pct:.1%}")
                    elif self.validation_level == ValidationLevel.NORMAL and missing_pct > 0.2:
                        errors.append(f"Excessive missing values in {col}: {missing_pct:.1%}")
        
        return len(errors) == 0, errors
    
    def get_required_columns(self) -> List[str]:
        """Get list of required DataFrame columns."""
        return ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def calculate_confidence(self, data: pd.DataFrame, result_value: Any) -> float:
        """
        Calculate confidence score based on data quality and result stability.
        
        Args:
            data: Input data used for calculation
            result_value: Computed indicator value
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from data availability
        periods_ratio = min(1.0, len(data) / (self.required_periods * 2))
        
        # Penalty for missing data
        required_cols = self.get_required_columns()
        total_cells = len(data) * len(required_cols)
        missing_cells = sum(data[col].isnull().sum() for col in required_cols if col in data.columns)
        missing_penalty = max(0.0, 1.0 - (missing_cells / total_cells * 5))
        
        # Penalty for extreme values (indicator-specific)
        stability_score = self._assess_result_stability(result_value)
        
        return periods_ratio * missing_penalty * stability_score
    
    def _assess_result_stability(self, result_value: Any) -> float:
        """Assess stability/reasonableness of computed result."""
        # Default implementation - subclasses can override
        if isinstance(result_value, (int, float)):
            if math.isnan(result_value) or math.isinf(result_value):
                return 0.0
            return 1.0
        elif isinstance(result_value, dict):
            valid_values = [v for v in result_value.values() 
                          if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
            return len(valid_values) / len(result_value) if result_value else 0.0
        else:
            return 0.8  # Moderate confidence for other types
    
    def set_validation_level(self, level: ValidationLevel) -> None:
        """Set data validation strictness level."""
        self.validation_level = level
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this indicator."""
        return self._performance_stats.copy()
    
    def __str__(self) -> str:
        """String representation of indicator."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"
    
    def __repr__(self) -> str:
        """Detailed representation of indicator."""
        return f"{self.__class__.__name__}(name='{self.name}', type={self.indicator_type.value}, params={self.params})"


class VectorizedIndicator(BaseIndicator):
    """
    Base class for vectorized indicators using pandas/numpy operations.
    
    Provides common vectorized operations and optimizations for
    high-performance indicator calculations.
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self._use_cache = params.get('use_cache', True)
        self._cache = {}
    
    def _get_cache_key(self, data: pd.DataFrame, **kwargs) -> str:
        """Generate cache key for computation results."""
        data_hash = str(hash(tuple(data.index))) + str(hash(tuple(data.values.flatten())))
        params_hash = str(hash(tuple(sorted(self.params.items()))))
        kwargs_hash = str(hash(tuple(sorted(kwargs.items()))))
        return f"{self.name}_{data_hash}_{params_hash}_{kwargs_hash}"
    
    def _safe_division(self, numerator: pd.Series, denominator: pd.Series, 
                      default_value: float = 0.0) -> pd.Series:
        """Perform safe division avoiding division by zero."""
        return numerator.div(denominator).fillna(default_value).replace([np.inf, -np.inf], default_value)
    
    def _rolling_operation(self, series: pd.Series, window: int, 
                          operation: str = 'mean', min_periods: Optional[int] = None) -> pd.Series:
        """Perform vectorized rolling operations."""
        if min_periods is None:
            min_periods = max(1, window // 2)
        
        rolling = series.rolling(window=window, min_periods=min_periods)
        
        if operation == 'mean':
            return rolling.mean()
        elif operation == 'std':
            return rolling.std()
        elif operation == 'max':
            return rolling.max()
        elif operation == 'min':
            return rolling.min()
        elif operation == 'sum':
            return rolling.sum()
        elif operation == 'median':
            return rolling.median()
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")
    
    def _ewm_operation(self, series: pd.Series, span: Optional[int] = None, 
                      alpha: Optional[float] = None, adjust: bool = False) -> pd.Series:
        """Perform exponential weighted moving operations."""
        if span is not None:
            return series.ewm(span=span, adjust=adjust).mean()
        elif alpha is not None:
            return series.ewm(alpha=alpha, adjust=adjust).mean()
        else:
            raise ValueError("Either span or alpha must be provided for EWM operation")
    
    def _true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range vectorized."""
        prev_close = close.shift(1)
        tr_components = pd.DataFrame({
            'hl': high - low,
            'hc': (high - prev_close).abs(),
            'lc': (low - prev_close).abs()
        })
        return tr_components.max(axis=1)
    
    def _typical_price(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate typical price (HLC/3)."""
        return (high + low + close) / 3
    
    def _weighted_close(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate weighted close (HLCC/4)."""
        return (high + low + 2 * close) / 4