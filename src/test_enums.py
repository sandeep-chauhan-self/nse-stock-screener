"""
Unit tests for centralized enums
Run with: python -m pytest test_enums.py -v
Or directly: python test_enums.py
"""
from pathlib import Path
import sys
import unittest

# Add src directory to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))
from common.enums import MarketRegime, ProbabilityLevel, PositionStatus, StopType
class TestMarketRegime(unittest.TestCase):
    """Test MarketRegime enum functionality."""
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(MarketRegime.BULLISH.value, "bullish")
        self.assertEqual(MarketRegime.BEARISH.value, "bearish")
        self.assertEqual(MarketRegime.SIDEWAYS.value, "sideways")
        self.assertEqual(MarketRegime.HIGH_VOLATILITY.value, "high_volatility")
    def test_string_representation(self):
        """Test string representation."""
        self.assertEqual(str(MarketRegime.BULLISH), "bullish")
        self.assertEqual(str(MarketRegime.BEARISH), "bearish")
    def test_from_string_method(self):
        """Test from_string class method."""
        self.assertEqual(MarketRegime.from_string("bullish"), MarketRegime.BULLISH)
        self.assertEqual(MarketRegime.from_string("BULLISH"), MarketRegime.BULLISH)
        self.assertEqual(MarketRegime.from_string("Bullish"), MarketRegime.BULLISH)
        with self.assertRaises(ValueError):
            MarketRegime.from_string("invalid_regime")
class TestProbabilityLevel(unittest.TestCase):
    """Test ProbabilityLevel enum functionality."""
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(ProbabilityLevel.HIGH.value, "HIGH")
        self.assertEqual(ProbabilityLevel.MEDIUM.value, "MEDIUM")
        self.assertEqual(ProbabilityLevel.LOW.value, "LOW")
    def test_score_thresholds(self):
        """Test score threshold property."""
        self.assertEqual(ProbabilityLevel.HIGH.score_threshold, 70)
        self.assertEqual(ProbabilityLevel.MEDIUM.score_threshold, 45)
        self.assertEqual(ProbabilityLevel.LOW.score_threshold, 0)
class TestPositionStatus(unittest.TestCase):
    """Test PositionStatus enum functionality."""
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(PositionStatus.OPEN.value, "open")
        self.assertEqual(PositionStatus.CLOSED.value, "closed")
        self.assertEqual(PositionStatus.PENDING.value, "pending")
class TestStopType(unittest.TestCase):
    """Test StopType enum functionality."""
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(StopType.INITIAL.value, "initial")
        self.assertEqual(StopType.BREAKEVEN.value, "breakeven")
        self.assertEqual(StopType.TRAILING.value, "trailing")
class TestEnumUtilities(unittest.TestCase):
    """Test enum utility functions."""
    def test_validate_enum_consistency(self):
        """Test enum validation function."""
        from common.enums import validate_enum_consistency
        result = validate_enum_consistency()
        self.assertIn("status", result)
        self.assertIn("issues", result)
        self.assertIn("enum_counts", result)

        # Should have 4 enum classes
        self.assertEqual(len(result["enum_counts"]), 4)
    def test_get_all_enum_info(self):
        """Test enum info retrieval function."""
        from common.enums import get_all_enum_info
        info = get_all_enum_info()
        self.assertIn("MarketRegime", info)
        self.assertIn("ProbabilityLevel", info)
        self.assertIn("PositionStatus", info)
        self.assertIn("StopType", info)

        # Check MarketRegime has correct members
        self.assertIn("BULLISH", info["MarketRegime"])
        self.assertEqual(info["MarketRegime"]["BULLISH"], "bullish")
if __name__ == "__main__":

    # Run tests when script is executed directly
    unittest.main(verbosity=2)
