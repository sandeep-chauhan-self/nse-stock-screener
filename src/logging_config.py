"""
Centralized Logging and Observability System
Provides structured logging, metrics collection, and retry mechanisms for the NSE Stock Screener
"""

import logging
import logging.config
import json
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sys


@dataclass
class MetricData:
    """Structure for collecting metrics"""
    count: int = 0
    total_time: float = 0.0
    errors: int = 0
    last_updated: Optional[datetime] = None


class CorrelationIdManager:
    """Thread-local correlation ID management"""
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get or create correlation ID for current thread"""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())[:8]
        return cls._local.correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread"""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls):
        """Clear correlation ID for current thread"""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')


class StructuredJSONFormatter(logging.Formatter):
    """Custom JSON formatter with correlation IDs and structured fields"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': CorrelationIdManager.get_correlation_id(),
            'thread': threading.current_thread().name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'retry_count'):
            log_entry['retry_count'] = record.retry_count
        
        return json.dumps(log_entry)


class MetricsCollector:
    """Thread-safe metrics collection system"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, MetricData] = defaultdict(MetricData)
        self._recent_errors: deque = deque(maxlen=100)  # Store recent errors
        self._start_time = time.time()
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric"""
        with self._lock:
            self._metrics[metric_name].count += value
            self._metrics[metric_name].last_updated = datetime.now()
    
    def record_duration(self, metric_name: str, duration: float):
        """Record execution duration"""
        with self._lock:
            metric = self._metrics[metric_name]
            metric.count += 1
            metric.total_time += duration
            metric.last_updated = datetime.now()
    
    def record_error(self, metric_name: str, error_details: Dict[str, Any]):
        """Record error occurrence"""
        with self._lock:
            self._metrics[metric_name].errors += 1
            self._metrics[metric_name].last_updated = datetime.now()
            
            # Store recent error details
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'metric': metric_name,
                'details': error_details,
                'correlation_id': CorrelationIdManager.get_correlation_id()
            }
            self._recent_errors.append(error_record)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            summary = {
                'uptime_seconds': time.time() - self._start_time,
                'metrics': {},
                'recent_errors': list(self._recent_errors)[-10:],  # Last 10 errors
                'generated_at': datetime.now().isoformat()
            }
            
            for name, metric in self._metrics.items():
                summary['metrics'][name] = {
                    'count': metric.count,
                    'total_time': metric.total_time,
                    'avg_time': metric.total_time / metric.count if metric.count > 0 else 0,
                    'errors': metric.errors,
                    'error_rate': metric.errors / metric.count if metric.count > 0 else 0,
                    'last_updated': metric.last_updated.isoformat() if metric.last_updated else None
                }
            
            return summary
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self._metrics.clear()
            self._recent_errors.clear()
            self._start_time = time.time()


class RetryManager:
    """Intelligent retry mechanism for handling transient failures"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.base_delay * (2 ** retry_count)
        return min(delay, self.max_delay)
    
    def is_retryable_error(self, exception: Exception) -> bool:
        """Determine if an error is worth retrying"""
        retryable_types = (
            ConnectionError,
            TimeoutError,
            OSError,  # Network issues
        )
        
        # Check for specific yfinance/network errors
        error_str = str(exception).lower()
        retryable_messages = [
            'connection',
            'timeout',
            'temporary',
            'rate limit',
            'network',
            'unavailable',
            'service temporarily'
        ]
        
        return (isinstance(exception, retryable_types) or 
                any(msg in error_str for msg in retryable_messages))
    
    def retry_with_backoff(self, operation_name: str, retryable_func: Callable, *args, **kwargs):
        """Execute function with retry logic and exponential backoff"""
        correlation_id = CorrelationIdManager.get_correlation_id()
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = retryable_func(*args, **kwargs)
                
                # Log successful operation
                duration = time.time() - start_time
                self.logger.info(
                    f"Operation succeeded: {operation_name}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'retry_count': attempt
                    }
                )
                
                # Record metrics
                metrics.record_duration(f"{operation_name}_duration", duration)
                if attempt > 0:
                    metrics.increment_counter(f"{operation_name}_retries_succeeded")
                
                return result
                
            except Exception as e:
                is_last_attempt = attempt == self.max_retries
                
                if not self.is_retryable_error(e) or is_last_attempt:
                    # Log final failure
                    self.logger.error(
                        f"Operation failed permanently: {operation_name}",
                        extra={
                            'operation': operation_name,
                            'error_type': type(e).__name__,
                            'retry_count': attempt,
                            'final_attempt': is_last_attempt
                        },
                        exc_info=True
                    )
                    
                    # Record error metrics
                    metrics.record_error(f"{operation_name}_failures", {
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'retry_count': attempt,
                        'correlation_id': correlation_id
                    })
                    
                    raise e
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                self.logger.warning(
                    f"Operation failed, retrying in {delay:.1f}s: {operation_name}",
                    extra={
                        'operation': operation_name,
                        'error_type': type(e).__name__,
                        'retry_count': attempt,
                        'delay_seconds': delay
                    }
                )
                
                metrics.increment_counter(f"{operation_name}_retries_attempted")
                time.sleep(delay)


# Global instances
metrics = MetricsCollector()
retry_manager = RetryManager()


@contextmanager
def operation_context(operation_name: str, **extra_fields):
    """Context manager for tracking operations with metrics and logging"""
    correlation_id = str(uuid.uuid4())[:8]
    CorrelationIdManager.set_correlation_id(correlation_id)
    
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    logger.info(
        f"Starting operation: {operation_name}",
        extra={'operation': operation_name, **extra_fields}
    )
    
    try:
        yield correlation_id
        
        duration = time.time() - start_time
        logger.info(
            f"Completed operation: {operation_name}",
            extra={'operation': operation_name, 'duration': duration, **extra_fields}
        )
        
        metrics.record_duration(f"{operation_name}_duration", duration)
        metrics.increment_counter(f"{operation_name}_completed")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed operation: {operation_name}",
            extra={
                'operation': operation_name,
                'duration': duration,
                'error_type': type(e).__name__,
                **extra_fields
            },
            exc_info=True
        )
        
        metrics.record_error(f"{operation_name}_failures", {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'duration': duration,
            'correlation_id': correlation_id,
            **extra_fields
        })
        
        raise
    
    finally:
        CorrelationIdManager.clear_correlation_id()


def with_retry(operation_name: str, max_retries: int = 3):
    """Decorator for adding retry logic to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_manager.retry_with_backoff(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator


def timed_operation(operation_name: str):
    """Decorator for timing operations and recording metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(
                    f"Function executed: {func.__name__}",
                    extra={'operation': operation_name, 'duration': duration}
                )
                
                metrics.record_duration(f"{operation_name}_duration", duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
                
                metrics.record_error(f"{operation_name}_failures", {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'function': func.__name__,
                    'duration': duration
                })
                
                raise
        
        return wrapper
    return decorator


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Configure centralized logging system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use structured JSON format instead of plain text
        log_file: Optional log file path
        console_output: Whether to output to console
    """
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': StructuredJSONFormatter,
            },
        },
        'handlers': {},
        'loggers': {
            '': {  # Root logger
                'handlers': [],
                'level': level,
                'propagate': False
            }
        }
    }
    
    # Console handler
    if console_output:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'formatter': 'json' if json_format else 'standard',
            'stream': 'ext://sys.stdout'
        }
        config['loggers']['']['handlers'].append('console')
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json' if json_format else 'standard',
            'filename': str(log_path),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5
        }
        config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(config)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            'level': level,
            'json_format': json_format,
            'log_file': log_file,
            'console_output': console_output
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


def log_metrics_summary():
    """Log current metrics summary"""
    logger = logging.getLogger(__name__)
    summary = metrics.get_metrics_summary()
    
    logger.info(
        "Metrics Summary",
        extra={
            'operation': 'metrics_summary',
            'uptime_seconds': summary['uptime_seconds'],
            'total_metrics': len(summary['metrics']),
            'recent_errors': len(summary['recent_errors'])
        }
    )
    
    return summary


# Utility functions for backward compatibility and easy migration
def replace_print_with_log(message: str, level: str = "INFO", **extra):
    """Helper function to replace print() calls"""
    logger = logging.getLogger('migration')
    log_func = getattr(logger, level.lower())
    log_func(message, extra=extra)


if __name__ == "__main__":
    # Demo and test the logging system
    setup_logging(level="DEBUG", json_format=True, console_output=True)
    
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.info("Testing basic logging functionality")
    
    # Test operation context
    with operation_context("demo_operation", test_param="value"):
        logger.info("Inside operation context")
        time.sleep(0.1)
    
    # Test metrics
    metrics.increment_counter("demo_counter", 5)
    metrics.record_duration("demo_operation", 0.5)
    
    # Test retry decorator
    @with_retry("demo_retry_operation", max_retries=2)
    def flaky_function():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Simulated network error")
        return "Success!"
    
    try:
        result = flaky_function()
        logger.info(f"Flaky function result: {result}")
    except Exception:
        logger.error("Flaky function failed permanently")
    
    # Print metrics summary
    summary = log_metrics_summary()
    print(json.dumps(summary, indent=2))