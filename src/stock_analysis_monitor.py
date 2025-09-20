"""
Stock Analysis Metrics and Monitoring System
Tracks specific metrics for the NSE Stock Screener operations
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import threading

from .logging_config import get_logger, metrics


@dataclass
class SymbolAnalysisMetrics:
    """Metrics for individual symbol analysis"""
    symbol: str
    start_time: float
    end_time: Optional[float] = None
    indicators_computed: bool = False
    score_computed: bool = False
    chart_generated: bool = False
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    data_fetch_time: Optional[float] = None
    indicator_compute_time: Optional[float] = None
    scoring_time: Optional[float] = None
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def was_successful(self) -> bool:
        return not self.error_occurred and self.score_computed


@dataclass
class BatchAnalysisSession:
    """Tracks metrics for a complete batch analysis session"""
    session_id: str
    start_time: float
    symbols_requested: List[str] = field(default_factory=list)
    symbols_completed: List[str] = field(default_factory=list)
    symbols_failed: List[str] = field(default_factory=list)
    total_data_fetch_time: float = 0.0
    total_indicator_time: float = 0.0
    total_scoring_time: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    end_time: Optional[float] = None
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        total = len(self.symbols_requested)
        if total == 0:
            return 0.0
        return len(self.symbols_completed) / total
    
    @property
    def symbols_per_minute(self) -> float:
        duration_minutes = self.total_duration / 60.0
        if duration_minutes == 0:
            return 0.0
        return len(self.symbols_completed) / duration_minutes


class StockAnalysisMonitor:
    """
    Advanced monitoring system for stock analysis operations
    Tracks performance metrics, error patterns, and operational health
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        
        # Session tracking
        self.current_session: Optional[BatchAnalysisSession] = None
        self.completed_sessions: deque = deque(maxlen=50)  # Keep last 50 sessions
        
        # Symbol-level tracking
        self.symbol_metrics: Dict[str, SymbolAnalysisMetrics] = {}
        
        # Operational metrics
        self.data_fetch_failures = defaultdict(int)
        self.indicator_failures = defaultdict(int)
        self.yfinance_rate_limits = 0
        self.network_errors = 0
        
        # Performance baselines
        self.target_symbols_per_minute = 30  # Configurable target
        self.max_acceptable_failure_rate = 0.15  # 15%
        
        # Historical performance tracking
        self.daily_metrics: Dict[str, Dict] = defaultdict(dict)
        
    def start_batch_session(self, symbols: List[str], session_id: Optional[str] = None) -> str:
        """Start tracking a new batch analysis session"""
        if session_id is None:
            session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            # End current session if exists
            if self.current_session:
                self.end_batch_session()
            
            self.current_session = BatchAnalysisSession(
                session_id=session_id,
                start_time=time.time(),
                symbols_requested=symbols.copy()
            )
            
            self.logger.info(
                f"Started batch analysis session",
                extra={
                    'session_id': session_id,
                    'symbol_count': len(symbols),
                    'operation': 'batch_start'
                }
            )
            
            # Record in global metrics
            metrics.increment_counter('batch_sessions_started')
            
        return session_id
    
    def end_batch_session(self) -> Optional[Dict[str, Any]]:
        """End current batch session and return summary"""
        with self._lock:
            if not self.current_session:
                return None
            
            self.current_session.end_time = time.time()
            session_summary = self._generate_session_summary(self.current_session)
            
            # Move to completed sessions
            self.completed_sessions.append(self.current_session)
            self.current_session = None
            
            self.logger.info(
                "Ended batch analysis session",
                extra={
                    'session_id': session_summary['session_id'],
                    'duration': session_summary['total_duration'],
                    'success_rate': session_summary['success_rate'],
                    'symbols_per_minute': session_summary['symbols_per_minute'],
                    'operation': 'batch_end'
                }
            )
            
            # Record performance metrics
            metrics.record_duration('batch_session_duration', session_summary['total_duration'])
            metrics.increment_counter('batch_sessions_completed')
            
            return session_summary
    
    def start_symbol_analysis(self, symbol: str) -> SymbolAnalysisMetrics:
        """Start tracking analysis for a specific symbol"""
        symbol_metrics = SymbolAnalysisMetrics(
            symbol=symbol,
            start_time=time.time()
        )
        
        with self._lock:
            self.symbol_metrics[symbol] = symbol_metrics
            
            # Add to current session if active
            if self.current_session:
                if symbol not in self.current_session.symbols_completed:
                    # Symbol starting analysis
                    pass
        
        self.logger.debug(
            f"Started analysis for {symbol}",
            extra={'symbol': symbol, 'operation': 'symbol_start'}
        )
        
        return symbol_metrics
    
    def record_data_fetch(self, symbol: str, duration: float, success: bool, error_type: Optional[str] = None):
        """Record data fetching metrics"""
        with self._lock:
            if symbol in self.symbol_metrics:
                self.symbol_metrics[symbol].data_fetch_time = duration
                
                if not success:
                    self.symbol_metrics[symbol].error_occurred = True
                    self.symbol_metrics[symbol].error_type = error_type
                    self.data_fetch_failures[error_type or 'unknown'] += 1
            
            # Add to session metrics
            if self.current_session:
                self.current_session.total_data_fetch_time += duration
                
                if not success:
                    self.current_session.error_counts[f'data_fetch_{error_type or "unknown"}'] += 1
        
        # Global metrics
        metrics.record_duration('data_fetch_duration', duration)
        if not success:
            metrics.increment_counter('data_fetch_failures')
            metrics.record_error('data_fetch_errors', {
                'symbol': symbol,
                'error_type': error_type,
                'duration': duration
            })
        
        self.logger.debug(
            f"Data fetch for {symbol}: {'success' if success else 'failed'}",
            extra={
                'symbol': symbol,
                'duration': duration,
                'success': success,
                'error_type': error_type,
                'operation': 'data_fetch'
            }
        )
    
    def record_indicator_computation(self, symbol: str, duration: float, success: bool, error_type: Optional[str] = None):
        """Record indicator computation metrics"""
        with self._lock:
            if symbol in self.symbol_metrics:
                self.symbol_metrics[symbol].indicator_compute_time = duration
                self.symbol_metrics[symbol].indicators_computed = success
                
                if not success:
                    self.symbol_metrics[symbol].error_occurred = True
                    self.symbol_metrics[symbol].error_type = error_type
                    self.indicator_failures[error_type or 'unknown'] += 1
            
            # Add to session metrics
            if self.current_session:
                self.current_session.total_indicator_time += duration
                
                if not success:
                    self.current_session.error_counts[f'indicator_{error_type or "unknown"}'] += 1
        
        # Global metrics
        metrics.record_duration('indicator_compute_duration', duration)
        if not success:
            metrics.increment_counter('indicator_failures')
            metrics.record_error('indicator_errors', {
                'symbol': symbol,
                'error_type': error_type,
                'duration': duration
            })
        
        self.logger.debug(
            f"Indicator computation for {symbol}: {'success' if success else 'failed'}",
            extra={
                'symbol': symbol,
                'duration': duration,
                'success': success,
                'error_type': error_type,
                'operation': 'indicator_compute'
            }
        )
    
    def record_scoring(self, symbol: str, duration: float, success: bool, score: Optional[float] = None):
        """Record scoring computation metrics"""
        with self._lock:
            if symbol in self.symbol_metrics:
                self.symbol_metrics[symbol].scoring_time = duration
                self.symbol_metrics[symbol].score_computed = success
                
                if not success:
                    self.symbol_metrics[symbol].error_occurred = True
            
            # Add to session metrics
            if self.current_session:
                self.current_session.total_scoring_time += duration
        
        # Global metrics
        metrics.record_duration('scoring_duration', duration)
        if score is not None:
            metrics.record_duration('scores_computed', score)  # Track score distribution
        
        self.logger.debug(
            f"Scoring for {symbol}: {'success' if success else 'failed'}",
            extra={
                'symbol': symbol,
                'duration': duration,
                'success': success,
                'score': score,
                'operation': 'scoring'
            }
        )
    
    def complete_symbol_analysis(self, symbol: str, success: bool):
        """Mark symbol analysis as complete"""
        with self._lock:
            if symbol in self.symbol_metrics:
                self.symbol_metrics[symbol].end_time = time.time()
                
                if not success:
                    self.symbol_metrics[symbol].error_occurred = True
            
            # Update session tracking
            if self.current_session:
                if success:
                    if symbol not in self.current_session.symbols_completed:
                        self.current_session.symbols_completed.append(symbol)
                else:
                    if symbol not in self.current_session.symbols_failed:
                        self.current_session.symbols_failed.append(symbol)
        
        # Global metrics
        if success:
            metrics.increment_counter('symbols_completed')
        else:
            metrics.increment_counter('symbols_failed')
        
        self.logger.info(
            f"Completed analysis for {symbol}: {'success' if success else 'failed'}",
            extra={
                'symbol': symbol,
                'success': success,
                'operation': 'symbol_complete'
            }
        )
    
    def record_network_error(self, error_type: str, details: Dict[str, Any]):
        """Record network-related errors"""
        with self._lock:
            if 'rate limit' in error_type.lower():
                self.yfinance_rate_limits += 1
            else:
                self.network_errors += 1
        
        metrics.record_error('network_errors', {
            'error_type': error_type,
            **details
        })
        
        self.logger.warning(
            f"Network error: {error_type}",
            extra={
                'error_type': error_type,
                'operation': 'network_error',
                **details
            }
        )
    
    def _generate_session_summary(self, session: BatchAnalysisSession) -> Dict[str, Any]:
        """Generate comprehensive session summary"""
        return {
            'session_id': session.session_id,
            'start_time': datetime.fromtimestamp(session.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(session.end_time).isoformat() if session.end_time else None,
            'total_duration': session.total_duration,
            'symbols_requested': len(session.symbols_requested),
            'symbols_completed': len(session.symbols_completed),
            'symbols_failed': len(session.symbols_failed),
            'success_rate': session.success_rate,
            'symbols_per_minute': session.symbols_per_minute,
            'avg_data_fetch_time': (session.total_data_fetch_time / len(session.symbols_requested) 
                                   if session.symbols_requested else 0),
            'avg_indicator_time': (session.total_indicator_time / len(session.symbols_requested) 
                                  if session.symbols_requested else 0),
            'avg_scoring_time': (session.total_scoring_time / len(session.symbols_requested) 
                               if session.symbols_requested else 0),
            'error_counts': dict(session.error_counts),
            'performance_vs_target': {
                'target_symbols_per_minute': self.target_symbols_per_minute,
                'actual_symbols_per_minute': session.symbols_per_minute,
                'performance_ratio': session.symbols_per_minute / self.target_symbols_per_minute if self.target_symbols_per_minute > 0 else 0,
                'acceptable_failure_rate': self.max_acceptable_failure_rate,
                'actual_failure_rate': 1 - session.success_rate,
                'within_failure_threshold': (1 - session.success_rate) <= self.max_acceptable_failure_rate
            }
        }
    
    def get_operational_health(self) -> Dict[str, Any]:
        """Get current operational health status"""
        with self._lock:
            # Recent session performance
            recent_sessions = list(self.completed_sessions)[-5:]  # Last 5 sessions
            avg_success_rate = (sum(s.success_rate for s in recent_sessions) / len(recent_sessions) 
                              if recent_sessions else 0)
            avg_symbols_per_minute = (sum(s.symbols_per_minute for s in recent_sessions) / len(recent_sessions) 
                                    if recent_sessions else 0)
            
            # Error analysis
            total_data_failures = sum(self.data_fetch_failures.values())
            total_indicator_failures = sum(self.indicator_failures.values())
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'current_session': {
                    'active': self.current_session is not None,
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'duration': self.current_session.total_duration if self.current_session else 0,
                    'symbols_completed': len(self.current_session.symbols_completed) if self.current_session else 0,
                    'symbols_failed': len(self.current_session.symbols_failed) if self.current_session else 0
                },
                'recent_performance': {
                    'sessions_analyzed': len(recent_sessions),
                    'avg_success_rate': avg_success_rate,
                    'avg_symbols_per_minute': avg_symbols_per_minute,
                    'performance_vs_target': avg_symbols_per_minute / self.target_symbols_per_minute if self.target_symbols_per_minute > 0 else 0
                },
                'error_summary': {
                    'total_data_fetch_failures': total_data_failures,
                    'total_indicator_failures': total_indicator_failures,
                    'yfinance_rate_limits': self.yfinance_rate_limits,
                    'network_errors': self.network_errors,
                    'data_failure_types': dict(self.data_fetch_failures),
                    'indicator_failure_types': dict(self.indicator_failures)
                },
                'health_indicators': {
                    'success_rate_healthy': avg_success_rate >= (1 - self.max_acceptable_failure_rate),
                    'performance_healthy': avg_symbols_per_minute >= (self.target_symbols_per_minute * 0.8),
                    'error_rate_acceptable': total_data_failures + total_indicator_failures < 10,  # Configurable threshold
                    'overall_healthy': None  # Will be calculated below
                }
            }
            
            # Calculate overall health
            health_checks = [
                health_status['health_indicators']['success_rate_healthy'],
                health_status['health_indicators']['performance_healthy'],
                health_status['health_indicators']['error_rate_acceptable']
            ]
            health_status['health_indicators']['overall_healthy'] = all(health_checks)
            
            return health_status
    
    def export_metrics(self, output_path: str):
        """Export comprehensive metrics to JSON file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'operational_health': self.get_operational_health(),
            'completed_sessions': [self._generate_session_summary(session) 
                                 for session in self.completed_sessions],
            'global_metrics': metrics.get_metrics_summary(),
            'symbol_metrics': {
                symbol: {
                    'symbol': metrics.symbol,
                    'total_duration': metrics.total_duration,
                    'indicators_computed': metrics.indicators_computed,
                    'score_computed': metrics.score_computed,
                    'was_successful': metrics.was_successful,
                    'error_type': metrics.error_type,
                    'data_fetch_time': metrics.data_fetch_time,
                    'indicator_compute_time': metrics.indicator_compute_time,
                    'scoring_time': metrics.scoring_time
                }
                for symbol, metrics in self.symbol_metrics.items()
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(
            f"Metrics exported to {output_path}",
            extra={'output_path': output_path, 'operation': 'metrics_export'}
        )


# Global instance
monitor = StockAnalysisMonitor()


# Convenience functions for easy integration
def start_batch_analysis(symbols: List[str], session_id: Optional[str] = None) -> str:
    """Start monitoring a batch analysis session"""
    return monitor.start_batch_session(symbols, session_id)


def end_batch_analysis() -> Optional[Dict[str, Any]]:
    """End current batch analysis session"""
    return monitor.end_batch_session()


def track_symbol_analysis(symbol: str):
    """Context manager for tracking individual symbol analysis"""
    from contextlib import contextmanager
    
    @contextmanager
    def symbol_tracker():
        symbol_metrics = monitor.start_symbol_analysis(symbol)
        success = False
        try:
            yield symbol_metrics
            success = True
        except Exception:
            success = False
            raise
        finally:
            monitor.complete_symbol_analysis(symbol, success)
    
    return symbol_tracker()


def get_health_status() -> Dict[str, Any]:
    """Get current operational health status"""
    return monitor.get_operational_health()


if __name__ == "__main__":
    # Demo the monitoring system
    from .logging_config import setup_logging
    
    setup_logging(level="INFO", console_output=True)
    
    # Simulate a batch analysis
    symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
    session_id = start_batch_analysis(symbols)
    
    for symbol in symbols:
        with track_symbol_analysis(symbol):
            # Simulate data fetch
            import random
            time.sleep(0.1)
            monitor.record_data_fetch(symbol, 0.1, True)
            
            # Simulate indicator computation
            time.sleep(0.05)
            monitor.record_indicator_computation(symbol, 0.05, True)
            
            # Simulate scoring
            time.sleep(0.02)
            score = random.uniform(40, 80)
            monitor.record_scoring(symbol, 0.02, True, score)
    
    # End session and show results
    summary = end_batch_analysis()
    print("Session Summary:")
    print(json.dumps(summary, indent=2))
    
    # Show health status
    health = get_health_status()
    print("\nHealth Status:")
    print(json.dumps(health, indent=2))