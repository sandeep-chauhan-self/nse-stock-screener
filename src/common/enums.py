"""
Centralized Enums for NSE Stock Screener

This module contains all shared enum definitions used across the stock screening system.
Centralizing enums here prevents circular imports and ensures consistency across modules.

Usage:
    from src.common.enums import MarketRegime, ProbabilityLevel, PositionStatus, StopType

    # Use the enums normally
    regime = MarketRegime.BULLISH
    status = PositionStatus.OPEN
"""

from enum import Enum
from typing import Dict, Any


class MarketRegime(Enum):
    """
    Market regime classification for adaptive threshold adjustments.

    Values represent different market conditions that affect
    indicator thresholds and scoring parameters.
    """
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

    def __str__(self) -> str:
        """Return the string value of the enum for display purposes."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> 'MarketRegime':
        """
        Create MarketRegime from string value, case-insensitive.

        Args:
            value: String representation of the regime

        Returns:
            MarketRegime enum value

        Raises:
            ValueError: If the string doesn't match any regime
        """
        try:
            return cls(value.lower())
        except ValueError:
            # Try case-insensitive lookup
            for regime in cls:
                if regime.value.lower() == value.lower():
                    return regime
            raise ValueError(f"Unknown market regime: {value}")


class ProbabilityLevel(Enum):
    """
    Signal probability classification levels.

    Used by the composite scoring system to categorize
    signal strength into discrete probability buckets.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __str__(self) -> str:
        """Return the string value of the enum for display purposes."""
        return self.value

    @property
    def score_threshold(self) -> int:
        """
        Return the minimum composite score threshold for this probability level.

        Returns:
            Minimum score threshold (0-100)
        """
        thresholds = {
            ProbabilityLevel.HIGH: 70,
            ProbabilityLevel.MEDIUM: 45,
            ProbabilityLevel.LOW: 0
        }
        return thresholds[self]


class PositionStatus(Enum):
    """
    Position lifecycle status for risk management.

    Tracks the current state of trading positions
    in the risk management system.
    """
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

    def __str__(self) -> str:
        """Return the string value of the enum for display purposes."""
        return self.value


class StopType(Enum):
    """
    Stop loss type classification for position management.

    Defines different stop loss strategies used
    in progressive risk management.
    """
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"

    def __str__(self) -> str:
        """Return the string value of the enum for display purposes."""
        return self.value


# Utility functions for enum validation and conversion
def validate_enum_consistency() -> Dict[str, Any]:
    """
    Validate that all enum values are consistent and unique.

    Returns:
        Dictionary with validation results including any issues found
    """
    validation_results = {
        "status": "success",
        "issues": [],
        "enum_counts": {}
    }

    # Check for value uniqueness within each enum
    enums_to_check = [MarketRegime, ProbabilityLevel, PositionStatus, StopType]

    for enum_class in enums_to_check:
        enum_name = enum_class.__name__
        values = [member.value for member in enum_class]
        validation_results["enum_counts"][enum_name] = len(values)

        # Check for duplicate values
        if len(values) != len(set(values)):
            validation_results["issues"].append(
                f"Duplicate values found in {enum_name}: {values}"
            )
            validation_results["status"] = "warning"

    return validation_results


def get_all_enum_info() -> Dict[str, Dict[str, str]]:
    """
    Get comprehensive information about all defined enums.

    Returns:
        Dictionary mapping enum names to their member information
    """
    enum_info = {}

    for enum_class in [MarketRegime, ProbabilityLevel, PositionStatus, StopType]:
        enum_name = enum_class.__name__
        enum_info[enum_name] = {
            member.name: member.value for member in enum_class
        }

    return enum_info


# Export all enums for easy importing
__all__ = [
    'MarketRegime',
    'ProbabilityLevel',
    'PositionStatus',
    'StopType',
    'validate_enum_consistency',
    'get_all_enum_info'
]