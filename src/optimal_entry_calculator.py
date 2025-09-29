"""
Optimal Entry Calculator - Probability-Weighted Entry Optimization Module

This module implements Monte Carlo simulation-based optimal entry price calculation
using probability-weighted scoring that combines:
1. Monte Carlo price path simulation (GBM)
2. Technical indicator favorability scoring
3. Market regime adjustments
4. Greed factor for profit optimization

Author: AI Enhanced Stock Screener
Date: September 2025
"""

import numpy as np
import pandas as pd
import math
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, cast
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import constants and core functionality
from constants import (
    MONTE_CARLO_PARAMETERS, MarketRegime, INDICATOR_CONSTANTS,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)
from core import DataFetcher, DataValidator

@dataclass
class OptimalEntryResult:
    """Result container for optimal entry calculations"""
    optimal_entry: float
    hit_probability: float
    indicator_confidence: float
    monte_carlo_paths: int
    data_confidence: str
    fallback_used: str
    cache_key: str
    execution_time_ms: float
    extreme_clip_count: int
    min_sim_price: float
    max_sim_price: float
    target_price: float
    greed_factor_applied: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'optimal_entry': round(self.optimal_entry, 2),
            'hit_probability': round(self.hit_probability, 3),
            'indicator_confidence': round(self.indicator_confidence, 1),
            'monte_carlo_paths': self.monte_carlo_paths,
            'data_confidence': self.data_confidence,
            'fallback_used': self.fallback_used,
            'execution_time_ms': round(self.execution_time_ms, 1),
            'extreme_clip_count': self.extreme_clip_count,
            'target_price': round(self.target_price, 2),
            'greed_factor_applied': self.greed_factor_applied
        }

class OptimalEntryCalculator:
    """
    Monte Carlo-based optimal entry price calculator with technical indicator weighting.
    
    Features:
    - Geometric Brownian Motion (GBM) price simulation
    - Technical indicator favorability scoring
    - Market regime-adjusted weights
    - Greed factor for profit optimization
    - Comprehensive caching and fallback strategies
    """
    
    def __init__(self):
        self.config = MONTE_CARLO_PARAMETERS
        self.cache = {}  # In-memory cache
        self.cache_timestamps = {}  # Cache expiry tracking
        
    def calculate_optimal_entry(self,
                              symbol: str,
                              current_price: float,
                              historical_data: Optional[pd.DataFrame],
                              indicators: Dict[str, Any],
                              market_regime: MarketRegime,
                              risk_bounds: Optional[Tuple[float, float]] = None,
                              target_price: Optional[float] = None,
                              horizon_days: Optional[int] = None,
                              signal: Optional[str] = None) -> OptimalEntryResult:
        """
        Calculate probability-weighted optimal entry price.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            historical_data: OHLCV historical data (pandas DataFrame)
            indicators: Technical indicators dictionary
            market_regime: Current market regime
            risk_bounds: Optional (min_price, max_price) risk limits
            target_price: Optional target price for greed factor
            horizon_days: Optional custom horizon (default from config)
            
        Returns:
            OptimalEntryResult with all calculation details
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(symbol, current_price, historical_data, indicators):
                return self._create_fallback_result(symbol, current_price, "Invalid inputs", start_time)
            
            # Check cache first
            cache_key = self._generate_cache_key(symbol, current_price, market_regime, horizon_days or self.config['horizon_days'])
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                cached_result.execution_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Determine horizon
            horizon = horizon_days or self.config['horizon_days']
            
            # Check data sufficiency and get confidence level
            data_confidence = self._assess_data_confidence(historical_data)
            
            if data_confidence == "INSUFFICIENT":
                return self._create_fallback_result(symbol, current_price, "Insufficient historical data", start_time)
            
            # Determine target price for greed factor
            final_target_price = self._determine_target_price(current_price, indicators, target_price)
            
            # Generate price grid within risk bounds
            price_grid = self._generate_price_grid(current_price, historical_data, risk_bounds, signal)
            
            if len(price_grid) < 5:  # Need at least 5 price points
                return self._create_fallback_result(symbol, current_price, "Price grid too small", start_time)
            
            # Run Monte Carlo simulation based on data confidence
            if data_confidence in ["HIGH", "MEDIUM"]:
                # Full Monte Carlo with GBM
                simulation_result = self._run_monte_carlo_simulation(
                    current_price, historical_data, horizon, price_grid
                )
            else:  # LOW confidence
                # Bootstrap simulation
                simulation_result = self._run_bootstrap_simulation(
                    current_price, historical_data, horizon, price_grid
                )
            
            # Calculate indicator favorability scores for each price
            indicator_scores = self._calculate_indicator_favorability_scores(
                price_grid, current_price, indicators, market_regime
            )
            
            # Combine probabilities with indicator scores
            weighted_scores = self._calculate_weighted_scores(
                price_grid, simulation_result['probabilities'], indicator_scores, 
                current_price, final_target_price
            )
            
            # Find optimal entry
            optimal_idx = np.argmax(weighted_scores)
            optimal_entry = price_grid[optimal_idx]
            hit_probability = simulation_result['probabilities'][optimal_idx]
            indicator_confidence = indicator_scores[optimal_idx]
            
            # Validate minimum probability threshold
            if hit_probability < self.config['min_probability_threshold']:
                return self._create_fallback_result(symbol, current_price, "Probability below threshold", start_time)
            
            # Create result
            result = OptimalEntryResult(
                optimal_entry=optimal_entry,
                hit_probability=hit_probability,
                indicator_confidence=indicator_confidence,
                monte_carlo_paths=simulation_result['paths_used'],
                data_confidence=data_confidence,
                fallback_used="Monte Carlo",
                cache_key=cache_key,
                execution_time_ms=(time.time() - start_time) * 1000,
                extreme_clip_count=simulation_result.get('extreme_clip_count', 0),
                min_sim_price=simulation_result.get('min_sim_price', current_price * 0.8),
                max_sim_price=simulation_result.get('max_sim_price', current_price * 1.2),
                target_price=final_target_price,
                greed_factor_applied=final_target_price > current_price
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
            return result
            
        except Exception as e:
            print(f"Error in optimal entry calculation for {symbol}: {e}")
            return self._create_fallback_result(symbol, current_price, f"Calculation error: {str(e)}", start_time)
    
    def monte_carlo_optimal_entry(self, symbol: str, historical_prices: pd.DataFrame, params: dict) -> dict:
        """
        Calculate Monte Carlo optimal entry with validation and success flag.
        
        Returns:
        {
          "success": bool,
          "entry": float or None,
          "reason": str,
          "debug": {...}
        }
        """
        try:
            if historical_prices is None or historical_prices.empty:
                return {
                    "success": False,
                    "entry": None,
                    "reason": "No historical price data",
                    "debug": {}
                }
            
            # Extract current price
            current_price = historical_prices['Close'].iloc[-1]
            
            # Create minimal indicators for calculation
            indicators = self._create_minimal_indicators(historical_prices)
            
            # Run optimal entry calculation
            result = self.calculate_optimal_entry(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_prices,
                indicators=indicators,
                market_regime=MarketRegime.BULLISH,  # Default regime
                risk_bounds=None,
                target_price=None,
                horizon_days=params.get('horizon_days', self.config['horizon_days']),
                signal=params.get('signal')  # Pass signal parameter
            )
            
            # Validate result
            if result.hit_probability >= self.config['min_probability_threshold']:
                # Additional sanity checks
                entry = result.optimal_entry
                
                # Check if entry is reasonable (within bounds)
                min_reasonable = current_price * 0.8
                max_reasonable = current_price * 1.2
                
                if min_reasonable <= entry <= max_reasonable:
                    return {
                        "success": True,
                        "entry": entry,
                        "reason": f"Monte Carlo success: prob={result.hit_probability:.3f}",
                        "debug": {
                            "hit_probability": result.hit_probability,
                            "indicator_confidence": result.indicator_confidence,
                            "data_confidence": result.data_confidence,
                            "paths_used": result.monte_carlo_paths
                        }
                    }
                else:
                    return {
                        "success": False,
                        "entry": None,
                        "reason": f"Entry {entry:.2f} outside reasonable bounds [{min_reasonable:.2f}, {max_reasonable:.2f}]",
                        "debug": {"entry": entry, "bounds": [min_reasonable, max_reasonable]}
                    }
            else:
                return {
                    "success": False,
                    "entry": None,
                    "reason": f"Hit probability {result.hit_probability:.3f} below threshold {self.config['min_probability_threshold']}",
                    "debug": {"hit_probability": result.hit_probability}
                }
                
        except Exception as e:
            return {
                "success": False,
                "entry": None,
                "reason": f"Monte Carlo calculation error: {str(e)}",
                "debug": {"error": str(e)}
            }
    
    def _create_minimal_indicators(self, data: pd.DataFrame) -> dict:
        """Create minimal indicators for Monte Carlo calculation."""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # Simple RSI approximation
            rsi = 50  # Neutral default
            
            # Simple Bollinger Bands
            sma = float(cast(pd.Series, close.rolling(20).mean()).iloc[-1])
            std = float(cast(pd.Series, close.rolling(20).std()).iloc[-1])
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            
            # Volume ratio
            avg_volume = float(cast(pd.Series, volume.rolling(20).mean()).iloc[-1])
            current_volume = float(volume.iloc[-1])
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': sma,
                'vol_ratio': vol_ratio,
                'vol_zscore': 0.0,  # Neutral
                'macd_hist': 0.0,   # Neutral
                'macd_strength': 0.0,
                'adx': 20,          # Neutral trend
                'trend_strength': 0.0,
                'ema_fast': close.iloc[-1],
                'ema_slow': close.iloc[-1],
                'ma_slope_fast': 0.0
            }
            
        except Exception:
            # Return neutral indicators
            return {
                'rsi': 50,
                'bb_upper': data['Close'].iloc[-1] * 1.02,
                'bb_lower': data['Close'].iloc[-1] * 0.98,
                'bb_middle': data['Close'].iloc[-1],
                'vol_ratio': 1.0,
                'vol_zscore': 0.0,
                'macd_hist': 0.0,
                'macd_strength': 0.0,
                'adx': 20,
                'trend_strength': 0.0,
                'ema_fast': data['Close'].iloc[-1],
                'ema_slow': data['Close'].iloc[-1],
                'ma_slope_fast': 0.0
            }
    
    def _validate_inputs(self, symbol: str, current_price: float, 
                        historical_data: Optional[pd.DataFrame], indicators: Dict[str, Any]) -> bool:
        """Validate all inputs for calculation"""
        try:
            if not symbol or current_price <= 0:
                return False
            
            if historical_data is None or len(historical_data) < 30:
                return False
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in historical_data.columns for col in required_columns):
                return False
            
            if not isinstance(indicators, dict) or len(indicators) < 5:
                return False
            
            return True
        except Exception:
            return False
    
    def _generate_cache_key(self, symbol: str, current_price: float, 
                           market_regime: MarketRegime, horizon_days: int) -> str:
        """Generate comprehensive cache key"""
        try:
            # Create config hash
            config_items = [
                str(self.config['simulation_paths']),
                str(self.config['greed_factor']),
                str(self.config['indicator_confidence_weight']),
                str(self.config['volatility_cap'])
            ]
            config_hash = hashlib.md5('|'.join(config_items).encode()).hexdigest()[:8]
            
            # Round price to avoid minor fluctuations
            price_rounded = round(current_price, 2)
            
            return f"{symbol}_{price_rounded}_{horizon_days}_{market_regime.value}_{config_hash}"
        except Exception:
            return f"{symbol}_{current_price}_{horizon_days}_default"
    
    def _get_cached_result(self, cache_key: str) -> Optional[OptimalEntryResult]:
        """Retrieve cached result if still valid"""
        try:
            if cache_key in self.cache and cache_key in self.cache_timestamps:
                cache_time = self.cache_timestamps[cache_key]
                cache_duration = timedelta(hours=self.config['cache_duration_hours'])
                
                if datetime.now() - cache_time < cache_duration:
                    return self.cache[cache_key]
            
            # Remove expired cache
            if cache_key in self.cache:
                del self.cache[cache_key]
            if cache_key in self.cache_timestamps:
                del self.cache_timestamps[cache_key]
            
            return None
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: OptimalEntryResult):
        """Cache calculation result"""
        try:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Cleanup old cache entries (keep last 2500 for large stock analysis)
            if len(self.cache) > 2500:
                oldest_keys = sorted(self.cache_timestamps.keys(), 
                                   key=lambda k: self.cache_timestamps[k])[:500]
                for key in oldest_keys:
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
        except Exception:
            pass  # Cache failure shouldn't break calculation
    
    def _assess_data_confidence(self, historical_data: Optional[pd.DataFrame]) -> str:
        """Assess confidence level based on historical data availability"""
        try:
            if historical_data is None:
                return "INSUFFICIENT"
            data_length = len(historical_data)
            
            if data_length >= self.config['preferred_historical_days']:
                return "HIGH"
            elif data_length >= self.config['min_historical_days']:
                return "MEDIUM"  
            elif data_length >= 60:
                return "LOW"
            else:
                return "INSUFFICIENT"
        except Exception:
            return "INSUFFICIENT"
    
    def _determine_target_price(self, current_price: float, indicators: Dict[str, Any], 
                               target_price: Optional[float]) -> float:
        """Determine target price using fallback chain"""
        try:
            # 1. Use provided target price if valid
            if target_price and target_price > current_price * (1 + self.config['min_profit_margin']):
                max_reasonable = current_price * self.config['max_target_multiplier']
                return min(target_price, max_reasonable)
            
            # 2. Try Volume Profile resistance
            vp_resistance = indicators.get('vp_resistance_level')
            if vp_resistance and not np.isnan(vp_resistance) and vp_resistance > current_price:
                return min(vp_resistance, current_price * self.config['max_target_multiplier'])
            
            # 3. Try Bollinger Upper Band
            bb_upper = indicators.get('bb_upper')
            if bb_upper and not np.isnan(bb_upper) and bb_upper > current_price:
                return min(bb_upper, current_price * self.config['max_target_multiplier'])
            
            # 4. Fallback to percentage above current price
            fallback_target = current_price * (1 + self.config['target_price_fallback_pct'])
            return fallback_target
            
        except Exception:
            return current_price * (1 + self.config['target_price_fallback_pct'])
    
    def _generate_price_grid(self, current_price: float, historical_data: Optional[pd.DataFrame],
                           risk_bounds: Optional[Tuple[float, float]], signal: Optional[str] = None) -> np.ndarray:
        """Generate discretized price grid for scoring"""
        try:
            if historical_data is None or len(historical_data) == 0:
                # Fallback to simple range around current price
                deviation = current_price * 0.05  # 5% range
                return np.linspace(current_price * 0.95, current_price * 1.05, 20)
            
            # Calculate ATR-based envelope
            atr = self._calculate_atr(historical_data)
            hist_min = historical_data['Low'].min()
            hist_max = historical_data['High'].max()
            
            # Determine bounds
            lower_limit = max(
                hist_min * self.config['historical_min_multiplier'],
                current_price - (5 * atr),
                self.config['min_absolute_price']
            )
            
            # CRITICAL FIX: For BUY signals, entry should not exceed current price
            if signal and signal.upper() == 'BUY':
                upper_limit = current_price  # FIXED: BUY entries cannot exceed current price
            else:
                upper_limit = min(
                    hist_max * self.config['historical_max_multiplier'],
                    current_price + (5 * atr),
                    current_price * self.config['upper_price_multiplier']
                )
            
            # Apply risk bounds if provided
            if risk_bounds:
                lower_limit = max(lower_limit, risk_bounds[0])
                upper_limit = min(upper_limit, risk_bounds[1])
            
            # Generate grid with adaptive step size
            step_size = self._get_step_size(current_price)
            
            # Ensure grid doesn't exceed max points
            max_points = self.config['max_grid_points']
            while (upper_limit - lower_limit) / step_size > max_points:
                step_size *= 2
            
            # Generate price points
            prices = []
            current = lower_limit
            while current <= upper_limit:
                prices.append(round(current, 8))
                current += step_size
            
            return np.array(prices)
            
        except Exception as e:
            # Fallback grid
            return np.linspace(current_price * 0.95, current_price * 1.05, 20)
    
    def _get_step_size(self, price: float) -> float:
        """Get adaptive step size based on price level"""
        if price < 1.0:
            return 0.01
        elif price < 10.0:
            return 0.05
        elif price < 50.0:
            return 0.10
        elif price < 200.0:
            return 0.50
        elif price < 500.0:
            return 1.00
        else:
            return 2.00
    
    def _calculate_atr(self, data: Optional[pd.DataFrame], period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if data is None or len(data) < period:
                return 1.0  # Fallback ATR value
                
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = float(cast(pd.Series, true_range.rolling(window=period).mean()).iloc[-1])
            
            return atr if not np.isnan(atr) else data['Close'].iloc[-1] * 0.02
        except Exception:
            return data['Close'].iloc[-1] * 0.02 if (data is not None and not data.empty) else 1.0
    
    def _run_monte_carlo_simulation(self, current_price: float, historical_data: Optional[pd.DataFrame],
                                  horizon_days: int, price_grid: np.ndarray) -> Dict[str, Any]:
        """Run Monte Carlo simulation using GBM"""
        try:
            if historical_data is None or len(historical_data) < 30:
                # Fallback simulation with default parameters
                mu = 0.0005  # Small positive drift
                sigma = 0.02  # 2% daily volatility
                hist_min = current_price * 0.7
                hist_max = current_price * 1.5
            else:
                # Calculate GBM parameters from historical data
                returns = cast(pd.Series, historical_data['Close'].pct_change().dropna())
                mu = float(returns.mean())  # Daily drift
                sigma = float(returns.std())  # Daily volatility
                hist_min = float(cast(pd.Series, historical_data['Low']).min())
                hist_max = float(cast(pd.Series, historical_data['High']).max())
            
            # Cap volatility for stability
            vol_cap = self.config['volatility_cap']
            min_price, max_price = self._compute_simulation_bounds(current_price, hist_min, hist_max)
            
            # Run simulations
            n_paths = self.config['simulation_paths']
            paths_completed = 0
            extreme_clip_count = 0
            min_prices_reached = []
            
            # Early termination check
            early_check_interval = self.config['min_simulation_paths']
            should_continue = True
            
            for i in range(0, n_paths, early_check_interval):
                batch_end = min(i + early_check_interval, n_paths)
                batch_size = batch_end - i
                
                # Simulate batch
                for _ in range(batch_size):
                    path_min = self._simulate_single_path(
                        current_price, mu, sigma, horizon_days, min_price, max_price, vol_cap
                    )
                    
                    if path_min['clipped']:
                        extreme_clip_count += 1
                    
                    min_prices_reached.append(path_min['min_price'])
                    paths_completed += 1
                
                # Early termination check after minimum paths
                if paths_completed >= early_check_interval and should_continue:
                    temp_probabilities = self._calculate_hit_probabilities(min_prices_reached, price_grid)
                    favorable_prob = np.max(temp_probabilities)
                    
                    if favorable_prob < self.config['early_termination_threshold']:
                        should_continue = False
                        break
            
            # Calculate final probabilities
            probabilities = self._calculate_hit_probabilities(min_prices_reached, price_grid)
            
            return {
                'probabilities': probabilities,
                'paths_used': paths_completed,
                'extreme_clip_count': extreme_clip_count,
                'min_sim_price': min_price,
                'max_sim_price': max_price
            }
            
        except Exception as e:
            print(f"Monte Carlo simulation error: {e}")
            return self._fallback_probability_calculation(current_price, price_grid)
    
    def _simulate_single_path(self, start_price: float, mu: float, sigma: float,
                            days: int, min_price: float, max_price: float, vol_cap: float) -> Dict[str, Any]:
        """Simulate single GBM price path"""
        try:
            price = start_price
            min_reached = start_price
            clipped = False
            
            for _ in range(days):
                # GBM step with volatility capping
                z = np.random.normal()
                z = np.clip(z, -vol_cap, vol_cap)  # Cap extreme moves
                
                price = price * np.exp((mu - 0.5 * sigma**2) + sigma * z)
                
                # Apply bounds
                if price < min_price:
                    price = min_price
                    clipped = True
                elif price > max_price:
                    price = max_price
                    clipped = True
                
                min_reached = min(min_reached, price)
            
            return {'min_price': min_reached, 'clipped': clipped}
            
        except Exception:
            return {'min_price': start_price, 'clipped': False}
    
    def _run_bootstrap_simulation(self, current_price: float, historical_data: Optional[pd.DataFrame],
                                horizon_days: int, price_grid: np.ndarray) -> Dict[str, Any]:
        """Run bootstrap simulation for low-confidence data"""
        try:
            if historical_data is None or len(historical_data) < 30:
                return self._fallback_probability_calculation(current_price, price_grid)
                
            # Get historical daily returns
            returns = historical_data['Close'].pct_change().dropna()
            
            if len(returns) < 30:
                return self._fallback_probability_calculation(current_price, price_grid)
            
            # Bootstrap simulation
            n_paths = min(self.config['simulation_paths'], 2000)  # Reduce for bootstrap
            min_prices_reached = []
            
            for _ in range(n_paths):
                price = current_price
                min_reached = current_price
                
                for _ in range(horizon_days):
                    # Sample random historical return
                    random_return = np.random.choice(returns)
                    price = price * (1 + random_return)
                    
                    # Simple bounds
                    price = max(price, current_price * 0.5)
                    price = min(price, current_price * 2.0)
                    
                    min_reached = min(min_reached, price)
                
                min_prices_reached.append(min_reached)
            
            probabilities = self._calculate_hit_probabilities(min_prices_reached, price_grid)
            
            return {
                'probabilities': probabilities,
                'paths_used': n_paths,
                'extreme_clip_count': 0,
                'min_sim_price': current_price * 0.5,
                'max_sim_price': current_price * 2.0
            }
            
        except Exception:
            return self._fallback_probability_calculation(current_price, price_grid)
    
    def _calculate_hit_probabilities(self, min_prices_reached: List[float], 
                                   price_grid: np.ndarray) -> np.ndarray:
        """Calculate probability of hitting each price level"""
        try:
            probabilities = np.zeros(len(price_grid))
            
            for i, target_price in enumerate(price_grid):
                hits = sum(1 for min_price in min_prices_reached if min_price <= target_price)
                probabilities[i] = hits / len(min_prices_reached)
            
            return probabilities
        except Exception:
            return np.ones(len(price_grid)) * 0.1  # Fallback uniform probability
    
    def _compute_simulation_bounds(self, current_price: float, hist_min: float, hist_max: float) -> Tuple[float, float]:
        """Compute simulation price bounds"""
        min_price = max(
            self.config['min_absolute_price'],
            hist_min * self.config['historical_min_multiplier']
        )
        max_price = min(
            current_price * self.config['upper_price_multiplier'],
            hist_max * self.config['historical_max_multiplier']
        )
        return min_price, max_price
    
    def _calculate_indicator_favorability_scores(self, price_grid: np.ndarray, current_price: float,
                                               indicators: Dict[str, Any], market_regime: MarketRegime) -> np.ndarray:
        """Calculate indicator favorability scores for each price level"""
        try:
            scores = np.zeros(len(price_grid))
            
            # Get regime-adjusted weights
            weights = self.config['regime_indicator_adjustments'].get(
                market_regime, self.config['indicator_weights']
            )
            
            for i, price in enumerate(price_grid):
                # Price-relative indicators (full recalculation)
                rsi_score = self._calculate_rsi_favorability(price, current_price, indicators)
                bollinger_score = self._calculate_bollinger_favorability(price, indicators)
                volume_score = self._calculate_volume_favorability(price, current_price, indicators)
                
                # Trend/momentum indicators (extrapolation)
                macd_score = self._extrapolate_macd_score(price, current_price, indicators)
                adx_score = self._extrapolate_adx_score(price, current_price, indicators)
                ema_score = self._extrapolate_ema_score(price, current_price, indicators)
                
                # Combine with weights
                weighted_score = (
                    weights.get('rsi', 0.25) * rsi_score +
                    weights.get('macd', 0.20) * macd_score +
                    weights.get('volume', 0.20) * volume_score +
                    weights.get('bollinger', 0.15) * bollinger_score +
                    weights.get('adx', 0.10) * adx_score +
                    weights.get('ema', 0.10) * ema_score
                )
                
                scores[i] = weighted_score * 100  # Scale to 0-100
            
            return scores
            
        except Exception as e:
            print(f"Error calculating indicator scores: {e}")
            return np.ones(len(price_grid)) * 50  # Neutral fallback
    
    def _calculate_rsi_favorability(self, price: float, current_price: float, indicators: Dict[str, Any]) -> float:
        """Calculate RSI favorability for hypothetical price"""
        try:
            current_rsi = indicators.get('rsi', 50)
            
            # Estimate RSI direction based on price movement
            if price < current_price * 0.97:  # 3% below current
                # Lower price likely means oversold condition = favorable for long
                return 1.0 if current_rsi < 40 else 0.7 if current_rsi < 50 else 0.3
            elif price > current_price * 1.03:  # 3% above current
                # Higher price likely means overbought = less favorable
                return 0.2 if current_rsi > 70 else 0.5 if current_rsi > 60 else 0.8
            else:
                # Near current price
                return 0.6 if 30 < current_rsi < 70 else 0.4
        except Exception:
            return 0.5
    
    def _calculate_bollinger_favorability(self, price: float, indicators: Dict[str, Any]) -> float:
        """Calculate Bollinger Bands favorability"""
        try:
            bb_upper = indicators.get('bb_upper', price * 1.02)
            bb_lower = indicators.get('bb_lower', price * 0.98)
            bb_middle = indicators.get('bb_middle', price)
            
            if bb_upper <= bb_lower:
                return 0.5  # Invalid bands
            
            # Calculate position within bands
            if price <= bb_lower:
                return 1.0  # At or below lower band = very favorable
            elif price <= bb_middle:
                return 0.8  # Below middle = favorable
            elif price <= bb_upper:
                return 0.4  # Above middle = less favorable
            else:
                return 0.1  # Above upper band = unfavorable
        except Exception:
            return 0.5
    
    def _calculate_volume_favorability(self, price: float, current_price: float, indicators: Dict[str, Any]) -> float:
        """Calculate volume favorability"""
        try:
            volume_ratio = indicators.get('vol_ratio', 1.0)
            volume_zscore = indicators.get('vol_zscore', 0.0)
            
            # Higher volume generally supports moves toward that price
            price_distance = abs(price - current_price) / current_price
            
            # High volume + price move = more favorable
            if volume_ratio > 2.0 and price_distance > 0.02:
                return 0.9
            elif volume_ratio > 1.5:
                return 0.7
            elif volume_ratio > 1.0:
                return 0.6
            else:
                return 0.4  # Low volume = less favorable
        except Exception:
            return 0.5
    
    def _extrapolate_macd_score(self, price: float, current_price: float, indicators: Dict[str, Any]) -> float:
        """Extrapolate MACD favorability"""
        try:
            macd_hist = indicators.get('macd_hist', 0)
            macd_strength = indicators.get('macd_strength', 0)
            
            price_change_pct = (price - current_price) / current_price
            
            # MACD bullish + price above current = favorable
            if macd_hist > 0 and price_change_pct > 0:
                return min(1.0, 0.5 + abs(macd_strength) * 0.5)
            elif macd_hist < 0 and price_change_pct < 0:
                return 0.3  # Bearish MACD + lower price = somewhat favorable for shorts
            else:
                return 0.5  # Neutral
        except Exception:
            return 0.5
    
    def _extrapolate_adx_score(self, price: float, current_price: float, indicators: Dict[str, Any]) -> float:
        """Extrapolate ADX trend strength favorability"""
        try:
            adx = indicators.get('adx', 20)
            trend_strength = indicators.get('trend_strength', 0)
            
            # Strong trend + movement in trend direction = favorable
            if adx > 25:  # Strong trend
                if trend_strength > 0 and price > current_price:
                    return 0.8  # Uptrend + higher price
                elif trend_strength < 0 and price < current_price:
                    return 0.6  # Downtrend + lower price
                else:
                    return 0.4  # Against trend
            else:
                return 0.5  # Weak trend = neutral
        except Exception:
            return 0.5
    
    def _extrapolate_ema_score(self, price: float, current_price: float, indicators: Dict[str, Any]) -> float:
        """Extrapolate EMA/Moving average favorability"""
        try:
            ema_fast = indicators.get('ema_fast', current_price)
            ema_slow = indicators.get('ema_slow', current_price)
            ma_slope_fast = indicators.get('ma_slope_fast', 0)
            
            # EMA crossover direction
            if ema_fast > ema_slow:  # Bullish crossover
                return 0.8 if price >= current_price else 0.4
            elif ema_fast < ema_slow:  # Bearish crossover
                return 0.3 if price >= current_price else 0.6
            else:
                return 0.5  # No clear direction
        except Exception:
            return 0.5
    
    def _calculate_weighted_scores(self, price_grid: np.ndarray, probabilities: np.ndarray,
                                 indicator_scores: np.ndarray, current_price: float,
                                 target_price: float) -> np.ndarray:
        """Calculate final weighted scores combining probability, indicators, and greed factor"""
        try:
            # Base weighted score
            confidence_weight = self.config['indicator_confidence_weight']
            base_scores = (
                (1 - confidence_weight) * probabilities * 100 +  # Probability component
                confidence_weight * indicator_scores  # Indicator component
            )
            
            # Apply greed factor if target price is valid
            if target_price > current_price * (1 + self.config['min_profit_margin']):
                greed_factors = np.zeros(len(price_grid))
                
                for i, price in enumerate(price_grid):
                    if price < target_price:
                        # Higher score for entries farther from target (more profit potential)
                        greed_factors[i] = (target_price - price) / target_price
                    else:
                        greed_factors[i] = 0.1  # Small positive value for prices above target
                
                # Apply greed factor
                greed_multiplier = self.config['greed_factor']
                final_scores = base_scores * (1 + greed_multiplier * greed_factors)
            else:
                final_scores = base_scores
            
            return final_scores
            
        except Exception:
            return indicator_scores  # Fallback to indicator scores only
    
    def _fallback_probability_calculation(self, current_price: float, price_grid: np.ndarray) -> Dict[str, Any]:
        """Fallback probability calculation using simple volatility assumptions"""
        try:
            # Simple assumption: higher probability for prices closer to current
            probabilities = np.zeros(len(price_grid))
            
            for i, price in enumerate(price_grid):
                distance = abs(price - current_price) / current_price
                # Linear decay with distance
                probabilities[i] = max(0.05, 0.5 - distance * 2)
            
            return {
                'probabilities': probabilities,
                'paths_used': 0,
                'extreme_clip_count': 0,
                'min_sim_price': current_price * 0.9,
                'max_sim_price': current_price * 1.1
            }
        except Exception:
            uniform_prob = 0.2
            return {
                'probabilities': np.full(len(price_grid), uniform_prob),
                'paths_used': 0,
                'extreme_clip_count': 0,
                'min_sim_price': current_price * 0.9,
                'max_sim_price': current_price * 1.1
            }
    
    def _create_fallback_result(self, symbol: str, current_price: float, reason: str, start_time: float) -> OptimalEntryResult:
        """Create fallback result when calculation fails"""
        return OptimalEntryResult(
            optimal_entry=current_price,
            hit_probability=0.1,
            indicator_confidence=50.0,
            monte_carlo_paths=0,
            data_confidence="LOW",
            fallback_used=f"ATR-based entry ({reason})",
            cache_key="",
            execution_time_ms=(time.time() - start_time) * 1000,
            extreme_clip_count=0,
            min_sim_price=current_price * 0.9,
            max_sim_price=current_price * 1.1,
            target_price=current_price * 1.05,
            greed_factor_applied=False
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the optimal entry calculator
    calculator = OptimalEntryCalculator()
    
    # Create sample data for testing
    import yfinance as yf
    
    symbol = "RELIANCE.NS"
    try:
        data = yf.download(symbol, period="1y")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            
            # Sample indicators
            indicators = {
                'rsi': 45.5,
                'macd_hist': 2.3,
                'macd_strength': 0.6,
                'bb_upper': current_price * 1.02,
                'bb_lower': current_price * 0.98,
                'bb_middle': current_price,
                'vol_ratio': 1.8,
                'vol_zscore': 1.2,
                'adx': 28,
                'trend_strength': 0.3,
                'ema_fast': current_price * 1.001,
                'ema_slow': current_price * 0.999,
                'ma_slope_fast': 0.5
            }
            
            # Test calculation
            result = calculator.calculate_optimal_entry(
                symbol=symbol,
                current_price=current_price,
                historical_data=data,
                indicators=indicators,
                market_regime=MarketRegime.BULLISH
            )
            
            print(f"\n=== Optimal Entry Analysis for {symbol} ===")
            print(f"Current Price: ₹{current_price:.2f}")
            print(f"Optimal Entry: ₹{result.optimal_entry:.2f}")
            print(f"Hit Probability: {result.hit_probability:.1%}")
            print(f"Indicator Confidence: {result.indicator_confidence:.1f}")
            print(f"Monte Carlo Paths: {result.monte_carlo_paths}")
            print(f"Data Confidence: {result.data_confidence}")
            print(f"Fallback Used: {result.fallback_used}")
            print(f"Execution Time: {result.execution_time_ms:.1f}ms")
            
        else:
            print("Failed to fetch test data")
    except Exception as e:
        print(f"Error in test: {e}")