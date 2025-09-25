# strategic_entry_constants.py
# Configuration constants for strategic entry calculations

# Breakout detection parameters
ROLLING_HIGH_DAYS = 20
VOL_WINDOW = 20
VOLUME_MULTIPLIER = 1.5

# Support/Pullback parameters
SUPPORT_LOOKBACK = 20
RSI_PULLBACK_MIN = 30
RSI_PULLBACK_MAX = 70

# ATR-based fallback parameters
ATR_K_LARGE = 0.5
ATR_K_SMALL = 1.5

# Market cap detection (volume-based proxy)
LARGE_CAP_VOLUME_THRESHOLD = 2_000_000

# Entry clamping parameters
MAX_PCT_LARGE = 0.05
MAX_PCT_SMALL = 0.10
N_ATR_LARGE = 1.0
N_ATR_SMALL = 2.5

# CI gates and validation thresholds
MAX_ENTRIES_EQUAL_CURRENT_PCT = 0.30
TARGET_ENTRIES_EQUAL_CURRENT_PCT = 0.15

# Risk-reward ratio validation
MIN_RISK_REWARD_RATIO = 1.5
RRR_TOLERANCE = 0.05

# Entry method constants
ENTRY_METHODS = [
    'MONTE_CARLO',
    'BREAKOUT',
    'SUPPORT_PULLBACK',
    'ATR_FALLBACK',
    'CURRENT_PRICE',
    'UNAVAILABLE'
]

# Order type constants
ORDER_TYPES = ['MARKET', 'LIMIT']

# Validation flags
VALIDATION_FLAGS = ['PASS', 'FAIL', 'REVIEW']