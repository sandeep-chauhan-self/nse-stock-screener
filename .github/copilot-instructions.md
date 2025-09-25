# NSE Stock Screener - AI Agent Instructions

## Architecture Overview

This is a sophisticated **multi-layered stock analysis system** with probabilistic scoring, designed around a **centralized constants + modular engines** pattern. The core architecture flows: `constants.py` → `core.py` → specialized engines → main analysis systems.

### Key Architectural Patterns

**Centralized Configuration**: All constants, enums, and paths flow through `src/constants.py` using `PROJECT_ROOT_PATH` as the anchor. Never hardcode paths - always use `FILE_CONSTANTS` and `Path` objects.

**Engine-based Design**: The system uses specialized engines that can be composed:
- `AdvancedIndicator` - 15+ technical indicators with volume anomaly detection
- `CompositeScorer` - Probabilistic 0-100 scoring with HIGH/MEDIUM/LOW buckets  
- `RiskManager` - Position sizing, stop losses, exposure limits
- `AdvancedBacktester` - Walk-forward validation with transaction costs
- `ForecastEngine` - Monte Carlo simulations with regime-adaptive parameters

**Market Regime Adaptation**: All thresholds adapt based on `MarketRegime` enum (BULLISH/BEARISH/SIDEWAYS/HIGH_VOLATILITY). The system detects regime via NIFTY analysis and adjusts RSI ranges, volume thresholds, etc.

## Critical Developer Workflows

### Running Analysis
```bash
# Main entry points (Windows-focused)
start.bat                    # Interactive menu system
scripts/stocks_analysis.bat  # Analysis with multiple modes
python src/enhanced_early_warning_system.py --stocks "RELIANCE.NS,TCS.NS"
```

### Key File Patterns
- **Input**: `data/nse_only_symbols.txt` (NSE symbols), custom lists via `--stocks` or `--file`
- **Output**: `output/reports/` (CSV), `output/charts/` (PNG), `output/backtests/` (analysis)
- **All NSE symbols must have `.NS` suffix** - use `StockLoader.ensure_ns_suffix()`

### Batch Processing Pattern
The system is designed for **large-scale analysis** (2000+ stocks) with rate limiting:
```python
# Always use rate limiting for yfinance API calls
from core import RateLimiter
limiter = RateLimiter(delay=0.5)  # From TRADING_CONSTANTS
limiter.wait_if_needed()
```

## Project-Specific Conventions

### Data Validation Standards
Every data fetch **must** use `DataValidator.validate_ohlcv_data()`. The system has strict quality thresholds:
- Minimum 50 data points for general analysis
- Price relationship validation (High ≥ Close ≥ Low)
- Maximum 10% missing data tolerance

### Scoring System Logic
The **composite scoring system is the core differentiator**:
```python
# Component weights (from SCORING_CONSTANTS)
volume: 25 points      # z-score + ratio detection
momentum: 25 points    # RSI + MACD with regime adjustments  
trend: 15 points       # ADX + moving average slopes
volatility: 10 points  # ATR-based calculations
relative_strength: 10  # vs NIFTY benchmark
volume_profile: 10     # Breakout detection
weekly_confirmation: +10 bonus  # Multi-timeframe validation
```

### Error Handling Pattern
Use centralized error messages from `constants.py` and **graceful degradation**:
```python
if data is None or data.empty:
    print(f"⚠️ {ERROR_MESSAGES['INSUFFICIENT_DATA']}")
    return default_safe_value
```

## Integration Points & Dependencies

### External APIs
- **Yahoo Finance (yfinance)**: Primary data source with rate limiting
- **NSE Symbol Lists**: Via `Equity_all.py` and `fetch_stock_symbols.py`
- **NIFTY Index**: Used for market regime detection and relative strength

### Cross-Component Communication
```python
# Standard pattern for engine integration
regime = MarketRegimeDetector.detect_regime()
indicators = AdvancedIndicator().compute_all_indicators(data)
score = CompositeScorer().calculate_composite_score(indicators, regime)
risk_params = RiskManager().calculate_position_size(score, current_price)
```

### File Dependencies
- `constants.py` → `core.py` → all engines (strict import hierarchy)
- **Windows batch files** drive the user workflow - maintain `.bat` compatibility
- Output directories auto-created via `PathManager.setup_output_directories()`

## Critical Implementation Notes

### Monte Carlo Integration
The `ForecastEngine` uses regime-adaptive Monte Carlo with **different parameter sets per market regime**. Access via `MONTE_CARLO_PARAMETERS['regime_indicator_adjustments'][regime]`.

### Multi-timeframe Confirmation
Weekly data analysis provides **confirmation bonus (+10 points)**. Always fetch both daily and weekly data for complete analysis:
```python
daily_data = DataFetcher.fetch_stock_data(symbol, "1y")
weekly_data = get_weekly_data(symbol)  # From core.py
```

### Risk Management Integration
The `RiskManager` calculates position sizes using **ATR-based stops** and **portfolio-level exposure limits**. Always validate risk parameters before position entry.

This system is designed for **institutional-grade analysis** with proper risk management, backtesting, and statistical validation. Maintain these standards when extending functionality.