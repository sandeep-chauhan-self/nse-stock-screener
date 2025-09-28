"""
Constants and Enums for NSE Stock Screener
Centralized location for all shared constants, enums, and configuration values.
"""

from enum import Enum
from typing import Dict, Any
from pathlib import Path

# Project Base Path - All relative paths calculated from this
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

class MarketRegime(Enum):
    """Market regime classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

# Trading Constants
TRADING_CONSTANTS = {
    'DEFAULT_BATCH_SIZE': 50,
    'DEFAULT_TIMEOUT': 5,
    'DEFAULT_INITIAL_CAPITAL': 50000,  # 10 Lakh
    'MAX_CHARTS_LIMIT': 10,
    'RATE_LIMIT_DELAY': 0.5,
    'BATCH_RATE_LIMIT_DELAY': 1.0,
    'NSE_SUFFIX': '.NS',
    'NIFTY_SYMBOL': '^NSEI'
}

# Risk Management Constants
RISK_CONSTANTS = {
    'DEFAULT_MAX_PORTFOLIO_RISK': 0.02,  # 2%
    'DEFAULT_MAX_POSITION_SIZE': 0.005,  # 0.5%
    'DEFAULT_MAX_DAILY_LOSS': 0.01,  # 1%
    'DEFAULT_MAX_CONCURRENT_POSITIONS': 10,
    'DEFAULT_RISK_PER_TRADE': 0.01,  # 1%
    'DEFAULT_ATR_MULTIPLIER': 2.0,
    'DEFAULT_TARGET_PROFIT_PCT': 0.025  # 2.5%
}

# Technical Indicator Constants
INDICATOR_CONSTANTS = {
    'RSI_PERIOD': 14,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ADX_PERIOD': 14,
    'ADX_STRONG_TREND': 25,
    'ATR_PERIOD': 14,
    'VOLUME_SMA_PERIOD': 20,
    'MA_SHORT_PERIOD': 20,
    'MA_LONG_PERIOD': 50,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    'STOCH_K_PERIOD': 14,
    'STOCH_D_PERIOD': 3,
    'CCI_PERIOD': 20,
    'WILLIAMS_R_PERIOD': 14,
    'MFI_PERIOD': 14
}

# Scoring System Constants
SCORING_CONSTANTS = {
    'COMPONENT_WEIGHTS': {
        'volume': 25,
        'momentum': 25,
        'trend': 15,
        'volatility': 10,
        'relative_strength': 10,
        'volume_profile': 10,
        'weekly_confirmation': 10  # Bonus component
    },
    'PROBABILITY_THRESHOLDS': {
        'HIGH': 70,
        'MEDIUM': 45,
        'LOW': 0
    },
    'MAX_COMPOSITE_SCORE': 100
}

# Market Regime Adjustments
REGIME_ADJUSTMENTS = {
    MarketRegime.BULLISH: {
        'rsi_min': 58,
        'rsi_max': 85,
        'volume_threshold': 2.0,
        'extreme_multiplier': 4.0,
        'trend_strength_min': 0.6,
        'volatility_tolerance': 1.2
    },
    MarketRegime.BEARISH: {
        'rsi_min': 62,
        'rsi_max': 80,
        'volume_threshold': 2.5,
        'extreme_multiplier': 5.0,
        'trend_strength_min': 0.7,
        'volatility_tolerance': 0.8
    },
    MarketRegime.SIDEWAYS: {
        'rsi_min': 60,
        'rsi_max': 75,
        'volume_threshold': 2.2,
        'extreme_multiplier': 4.5,
        'trend_strength_min': 0.65,
        'volatility_tolerance': 1.0
    },
    MarketRegime.HIGH_VOLATILITY: {
        'rsi_min': 55,
        'rsi_max': 85,
        'volume_threshold': 3.0,
        'extreme_multiplier': 6.0,
        'trend_strength_min': 0.5,
        'volatility_tolerance': 1.5
    }
}

# Data Quality Constants
DATA_QUALITY_CONSTANTS = {
    'MIN_DATA_POINTS_RSI': 15,
    'MIN_DATA_POINTS_MACD': 35,
    'MIN_DATA_POINTS_ADX': 28,
    'MIN_DATA_POINTS_GENERAL': 50,
    'MIN_CHART_DATA_POINTS': 50,
    'VOLATILITY_THRESHOLD': 0.25,  # 25% annual volatility for high vol regime
    'MAX_MISSING_DATA_PCT': 0.1  # 10% missing data tolerance
}

# File and Directory Constants
FILE_CONSTANTS = {
    'DEFAULT_STOCK_FILE': 'nse_only_symbols.txt',
    'OUTPUT_DIRS': ['reports', 'charts', 'backtests'],
    'CHART_DPI': 300,
    'CHART_SIZE': (16, 12),
    'FALLBACK_STOCKS': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'],
    'DEFAULT_STOCK_LIMIT': 35,  # Limit for testing
    # Path Constants using PROJECT_ROOT_PATH
    'DATA_DIR': PROJECT_ROOT_PATH / 'data',
    'OUTPUT_DIR': PROJECT_ROOT_PATH / 'output',
    'SRC_DIR': PROJECT_ROOT_PATH / 'src',
    'REPORTS_DIR': PROJECT_ROOT_PATH / 'output' / 'reports',
    'CHARTS_DIR': PROJECT_ROOT_PATH / 'output' / 'charts',
    'BACKTESTS_DIR': PROJECT_ROOT_PATH / 'output' / 'backtests',
    'DEFAULT_STOCK_FILE_PATH': PROJECT_ROOT_PATH / 'data' / 'nse_only_symbols.txt'
}

# Error Messages
ERROR_MESSAGES = {
    'INSUFFICIENT_DATA': "Insufficient data for analysis",
    'DATA_FETCH_FAILED': "Failed to fetch data from Yahoo Finance",
    'INDICATOR_COMPUTATION_FAILED': "Failed to compute technical indicators",
    'SCORING_FAILED': "Failed to compute composite score",
    'RISK_CHECK_FAILED': "Failed to perform risk management check",
    'CHART_GENERATION_FAILED': "Failed to generate chart",
    'FILE_NOT_FOUND': "Required file not found",
    'INVALID_SYMBOL': "Invalid stock symbol format"
}

# Success Messages
SUCCESS_MESSAGES = {
    'SYSTEM_INITIALIZED': "✅ Enhanced Early Warning System initialized",
    'ANALYSIS_COMPLETE': "📊 ANALYSIS COMPLETE",
    'CHART_SAVED': "Enhanced chart saved",
    'REPORT_SAVED': "Report saved successfully",
    'BACKTEST_COMPLETE': "Backtest analysis completed"
}

# Monte Carlo Configuration Constants
MONTE_CARLO_PARAMETERS = {
    # Simulation Parameters (Optimized for batch processing 2145 stocks)
    'simulation_paths': 3000,          # Reduced from 5000 for faster batch processing
    'horizon_days': 7,                 # Default forecast horizon (trading days)
    'min_historical_days': 120,        # Minimum days for GBM (fallback at 60)
    'preferred_historical_days': 250,   # Preferred days for stable GBM parameters
    
    # Scoring Parameters
    'greed_factor': 1.2,               # Profit weighting multiplier
    'min_probability_threshold': 0.15,  # Minimum hit probability (15%)
    'indicator_confidence_weight': 0.7, # Weight of indicators vs pure probability
    'max_price_deviation': 0.15,       # ±15% max deviation from current price
    
    # Risk & Performance Parameters
    'volatility_cap': 3.0,            # Maximum daily move (3x historical sigma)
    'max_grid_points': 1000,          # Maximum price discretization points
    'cache_duration_hours': 2,         # Cache results for 2 hours (for 2145 stocks)
    'max_execution_time_seconds': 10,  # Extended time per stock for large batches
    'early_termination_threshold': 0.05, # Stop if <5% favorable probability
    'min_simulation_paths': 2000,      # Minimum paths even with early termination
    
    # Extreme Price Protection
    'min_absolute_price': 0.01,        # Floor price (₹0.01)
    'upper_price_multiplier': 10.0,    # Max 10x current price
    'historical_max_multiplier': 2.0,  # Max 2x historical maximum
    'historical_min_multiplier': 0.5,  # Min 0.5x historical minimum
    
    # Indicator Weights (must sum to 1.0)
    'indicator_weights': {
        'rsi': 0.25,           # RSI/Stochastic indicators
        'macd': 0.20,          # MACD/Momentum indicators  
        'volume': 0.20,        # Volume anomaly indicators
        'bollinger': 0.15,     # Bollinger Bands
        'adx': 0.10,           # ADX/Trend strength
        'ema': 0.10            # EMA/Moving average indicators
    },
    
    # Market Regime Weights Adjustment
    'regime_indicator_adjustments': {
        MarketRegime.BULLISH: {
            'macd': 0.30, 'adx': 0.20, 'rsi': 0.15, 'volume': 0.20, 'bollinger': 0.10, 'ema': 0.05
        },
        MarketRegime.BEARISH: {
            'rsi': 0.35, 'bollinger': 0.20, 'macd': 0.15, 'volume': 0.15, 'adx': 0.10, 'ema': 0.05  
        },
        MarketRegime.SIDEWAYS: {
            'rsi': 0.25, 'macd': 0.20, 'volume': 0.20, 'bollinger': 0.15, 'adx': 0.10, 'ema': 0.10
        },
        MarketRegime.HIGH_VOLATILITY: {
            'volume': 0.30, 'bollinger': 0.25, 'rsi': 0.20, 'macd': 0.15, 'adx': 0.05, 'ema': 0.05
        }
    },
    
    # Target Price Determination
    'target_price_fallback_pct': 0.10,  # 10% above current as last resort
    'min_profit_margin': 0.02,          # Minimum 2% profit margin
    'max_target_multiplier': 3.0        # Max 3x current price as target
}

# Display Format Constants
DISPLAY_CONSTANTS = {
    'CURRENCY_SYMBOL': '₹',
    'PERCENTAGE_FORMAT': '{:.1f}%',
    'PRICE_FORMAT': '{:.1f}',
    'SCORE_FORMAT': '{:.0f}',
    'RATIO_FORMAT': '{:.1f}x',
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    'FILE_DATE_FORMAT': '%Y%m%d_%H%M%S'
}