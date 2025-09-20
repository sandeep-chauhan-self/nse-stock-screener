# NSE Stock Screener - AI Coding Agent Instructions

## Architecture Overview

This is a probabilistic stock screening system that analyzes NSE-listed stocks using technical indicators, composite scoring, and risk management. The system produces actionable **entry/exit/stop-loss signals** through a multi-layered analysis pipeline.

### Core Data Flow
```
Stock Symbols → Technical Indicators → Composite Scoring → Risk Management → Actionable Signals
     ↓              ↓                    ↓                  ↓                ↓
fetch_symbols → advanced_indicators → composite_scorer → risk_manager → reports/charts
```

### Key Components
- **`enhanced_early_warning_system.py`**: Main orchestrator that coordinates the analysis pipeline
- **`advanced_indicators.py`**: Computes 15+ technical indicators (RSI, MACD, ADX, volume analysis, etc.)
- **`composite_scorer.py`**: Probabilistic scoring system (0-100) with HIGH/MEDIUM/LOW classification
- **`risk_manager.py`**: Position sizing, stop-loss calculation, exposure limits
- **`advanced_backtester.py`**: Walk-forward backtesting with realistic transaction costs

## Critical Project Patterns

### Enum Management and Circular Imports
The system uses `MarketRegime` enum across multiple files. **Always import from a single source** to avoid type mismatches:
```python
# CORRECT: Import from enhanced_early_warning_system
from enhanced_early_warning_system import MarketRegime

# AVOID: Creating duplicate enums or type conversions
```

### Yahoo Finance Data Handling
All price data comes through `yfinance`. **Always handle missing data gracefully**:
```python
# Standard pattern for data fetching
ticker = yf.Ticker(symbol)
data = ticker.history(period="1y")
if data.empty or len(data) < minimum_required:
    return None  # Fail gracefully
```

### Composite Scoring Logic
The scoring system uses weighted components totaling 100 points:
- Volume (25): z-score + ratio analysis
- Momentum (25): RSI + MACD 
- Trend (15): ADX + MA crossover
- Volatility (10): ATR analysis
- Relative Strength (10): vs NIFTY performance
- Volume Profile (10): breakout detection
- Weekly Confirmation (+10 bonus)

**Critical**: Regime-adjusted thresholds change scoring behavior. Example:
```python
# Bullish regime: looser RSI requirements
regime_settings = self.regime_adjustments[regime]
rsi_min = regime_settings['rsi_min']  # 58 in bullish, 62 in bearish
```

### Risk Management Integration
Every signal MUST pass through risk management validation:
```python
can_enter, risk_reason, quantity, risk_amount = self.risk_manager.can_enter_position(
    symbol, entry_price, stop_loss, composite_score
)
```

## File Organization Conventions

### Path Handling
**Always use pathlib for cross-platform compatibility**:
```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]  # Get repo root
output_path = BASE_DIR / 'output' / 'reports' / filename
```

### Configuration Pattern
Configuration scattered across multiple files needs centralization. When adding config:
```python
# Pattern: Use dataclass for type safety
@dataclass
class SystemConfig:
    portfolio_capital: float = 100000.0
    max_positions: int = 10
    risk_per_trade: float = 0.01
```

### Output Structure
```
output/
├── reports/        # CSV analysis results with timestamps
├── charts/         # PNG technical analysis charts  
└── backtests/      # Backtest results and trade logs
```

## Key Workflows

### Running Analysis
Primary entry points:
- **Interactive**: `python src/enhanced_early_warning_system.py`
- **Batch**: `scripts/enhanced_launcher.bat` (Windows)
- **Custom symbols**: `python src/enhanced_early_warning_system.py -s "RELIANCE,TCS,INFY"`

### Dependencies
Check with: `python check_deps.py`
Required: `yfinance`, `pandas`, `numpy`, `matplotlib`, `requests`, `beautifulsoup4`

### Data Sources
- **Symbol lists**: `data/nse_only_symbols.txt` (NSE symbols)
- **Market data**: Yahoo Finance API (`yfinance`)
- **Index data**: NIFTY 50 (`^NSEI`) for relative strength calculations

## Technical Debt Hotspots

### Volume Threshold Bug
In `composite_scorer.py`, line with `vol_ratio >= vol_threshold * 1.67` has mismatched comment claiming "5x becomes 8.3x". Fix by:
```python
extreme_multiplier = regime_settings.get("extreme_multiplier", 5.0)
if vol_ratio >= vol_threshold * extreme_multiplier:
```

### Performance Issues
Replace Python loops with vectorized operations, especially in volume profile calculations:
```python
# SLOW: iterrows() pattern
for index, row in data.iterrows():
    # per-row processing

# FAST: vectorized with numpy
hist, edges = np.histogram(prices, bins=num_buckets, weights=volumes)
```

### Error Handling Pattern
Replace `print()` statements with proper logging:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Analyzing {symbol}...")  # Instead of print()
```

## Data Validation Patterns

### Indicator Return Contracts
Functions return `Dict[str, Union[float, int, bool]]` with `np.nan` for missing numeric values:
```python
def compute_rsi(data: pd.DataFrame) -> float:
    if len(data) < 15:
        return np.nan  # Not enough data
    # ... computation
    return round(rsi_value, 2)
```

### Market Regime Detection
Based on NIFTY analysis with fallback to SIDEWAYS:
```python
def detect_market_regime(self) -> MarketRegime:
    try:
        # NIFTY-based logic
        if ma20 > ma50 and recent_return > 3:
            return MarketRegime.BULLISH
        # ... other conditions
    except Exception:
        return MarketRegime.SIDEWAYS  # Safe fallback
```

## Testing Approach

### Unit Test Pattern
For deterministic testing, use small CSV fixtures:
```python
def test_rsi_calculation():
    # Use known OHLC data with expected RSI output
    test_data = pd.read_csv('fixtures/rsi_test_data.csv')
    result = compute_rsi(test_data)
    assert abs(result - 65.23) < 0.01  # Expected RSI value
```

## Integration Points

### Yahoo Finance Rate Limiting
Batch processing with delays between requests:
```python
for symbol in batch:
    result = analyze_stock(symbol)
    time.sleep(0.5)  # Rate limiting
```

### NSE Data Scraping
Fragile dependency on NSE website structure. Always include fallback symbol lists and validate data freshness.

## Common Gotchas

1. **Enum type mismatches**: Always use consistent enum imports
2. **Path separators**: Use `pathlib.Path` not hardcoded `\\` or `/`
3. **Data freshness**: Yahoo Finance can return stale/missing data
4. **Lookahead bias**: Ensure indicators only use historical data
5. **Corporate actions**: Use adjusted close prices for return calculations
6. **Volume normalization**: Different exchanges have different volume scales

Focus changes on the core pipeline: indicators → scoring → risk management → actionable outputs. The system's value is in producing reliable entry/exit/stop-loss signals, not just screening.