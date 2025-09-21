"""
Core validation framework - base classes, enums, and data structures.
This module contains the foundational components for the data validation system:
- ValidationLevel and ValidationStatus enums
- ValidationIssue and EnhancedValidationResult data classes
- ValidationRule abstract base class
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
class ValidationLevel(Enum):
    """Validation severity levels for enhanced reporting."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
@dataclass
class ValidationIssue(object):
    """Individual validation issue with detailed context."""
    rule_name: str
    level: ValidationLevel
    message: str
    symbol: str
    data_type: str = "ohlcv"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=Dict[str, Any])
@dataclass
class EnhancedValidationResult(object):
    """Enhanced result of data validation with comprehensive reporting."""
    symbol: str
    data_type: str
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=List[str])
    metadata: Dict[str, Any] = field(default_factory=Dict[str, Any])
    @property
    def has_errors(self) -> bool:
        """Check if result contains error or critical level issues."""
        return any(issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
                   for issue in self.issues)
    @property
    def has_warnings(self) -> bool:
        """Check if result contains warning level issues."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)
    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """Get all issues of a specific level."""
        return [issue for issue in self.issues if issue.level == level]
    def add_issue(self, rule_name: str, level: ValidationLevel, message: str, **metadata) -> None:
        """Add a validation issue to the result."""
        self.issues.append(ValidationIssue(
            rule_name=rule_name,
            level=level,
            message=message,
            symbol=self.symbol,
            data_type=self.data_type,
            metadata=metadata
        ))
class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.ERROR,
                 enabled: bool = True, config: Dict[str, Any] = None) -> None:
        self.name = name
        self.level = level
        self.enabled = enabled
        self.config = config or {}
    @abstractmethod
    def validate(self, symbol: str, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate data and return List[str] of issues."""
        pass
    def is_applicable(self, symbol: str, data_type: str) -> bool:
        """Check if this rule applies to the given symbol/data type."""
        return True
