"""
Monitoring and Alerting System for NSE Stock Screener
This module provides comprehensive monitoring capabilities with Prometheus metrics
export, Grafana dashboard support, and intelligent alerting for production deployment.
Key Features:
- Prometheus metrics export for latency, error rates, and throughput
- Custom metrics for financial analysis operations
- Health checks and system monitoring
- Alert management and notification system
- Performance regression detection
- Real-time system observability
"""
import logging
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict[str, Any], List[str], Any, Optional, Callable, Union
import json
import os
import socket
import sys
import psutil
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry
from prometheus_client.core import REGISTRY
import requests

# Import logging and configuration
try:
    from ..logging_config import get_logger
    from ..config.settings import MonitoringConfig
except ImportError:

    # Fallback imports
    def get_logger(name): return logging.getLogger(name)
    @dataclass
    class MonitoringConfig:
        prometheus_port: int = 8000
        metrics_prefix: str = "nse_screener"
        enable_alerts: bool = True
        alert_webhook_url: Optional[str] = None
        health_check_interval: int = 60
logger = get_logger(__name__)
@dataclass
class MetricConfig:
    """Configuration for monitoring metrics."""
    prometheus_port: int = 8000
    metrics_prefix: str = "nse_screener"
    enable_custom_registry: bool = False
    export_system_metrics: bool = True
    metric_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate_threshold": 0.05,
  # 5% error rate
        "latency_p95_threshold": 10.0,
  # 10 seconds
        "memory_usage_threshold": 0.8,
  # 80% memory usage
        "cpu_usage_threshold": 0.9,
  # 90% CPU usage
    })
class PrometheusMetrics:
    """Prometheus metrics collector for stock screener operations."""
    def __init__(self, config: Optional[MetricConfig] = None) -> None:
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry() if config and config.enable_custom_registry else REGISTRY
        self.prefix = self.config.metrics_prefix

        # Initialize metrics
        self._init_performance_metrics()
        self._init_business_metrics()
        self._init_system_metrics()

        # Start Prometheus HTTP server
        self.http_server = None
        self.start_prometheus_server()
        logger.info(f"Prometheus metrics initialized with prefix '{self.prefix}'")
    def _init_performance_metrics(self):
        """Initialize performance-related metrics."""

        # Processing time histogram
        self.processing_time = Histogram(
            f'{self.prefix}_processing_duration_seconds',
            'Time spent processing operations',
            ['operation', 'status'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')]
        )

        # Request counter
        self.requests_total = Counter(
            f'{self.prefix}_requests_total',
            'Total number of requests',
            ['operation', 'status'],
            registry=self.registry
        )

        # Error counter
        self.errors_total = Counter(
            f'{self.prefix}_errors_total',
            'Total number of errors',
            ['operation', 'error_type'],
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            f'{self.prefix}_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        self.cache_misses = Counter(
            f'{self.prefix}_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )

        # Memory usage gauge
        self.memory_usage = Gauge(
            f'{self.prefix}_memory_usage_bytes',
            'Current memory usage in bytes',
            ['component'],
            registry=self.registry
        )
    def _init_business_metrics(self):
        """Initialize business logic metrics."""

        # Symbol processing metrics
        self.symbols_processed = Counter(
            f'{self.prefix}_symbols_processed_total',
            'Total symbols processed',
            ['status'],
            registry=self.registry
        )

        # Analysis results
        self.analysis_signals = Counter(
            f'{self.prefix}_analysis_signals_total',
            'Total analysis signals generated',
            ['signal_type'],
            registry=self.registry
        )

        # Portfolio metrics
        self.portfolio_value = Gauge(
            f'{self.prefix}_portfolio_value',
            'Current portfolio value',
            registry=self.registry
        )
        self.positions_count = Gauge(
            f'{self.prefix}_positions_count',
            'Number of current positions',
            registry=self.registry
        )

        # Data quality metrics
        self.data_freshness = Gauge(
            f'{self.prefix}_data_freshness_seconds',
            'Age of the latest data in seconds',
            ['data_source'],
            registry=self.registry
        )
        self.data_quality_score = Gauge(
            f'{self.prefix}_data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry
        )
    def _init_system_metrics(self):
        """Initialize system-level metrics."""
        if not self.config.export_system_metrics:
            return

        # CPU usage
        self.cpu_usage = Gauge(
            f'{self.prefix}_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        # Memory usage
        self.system_memory_usage = Gauge(
            f'{self.prefix}_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )

        # Disk usage
        self.disk_usage = Gauge(
            f'{self.prefix}_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )

        # Network I/O
        self.network_bytes_sent = Counter(
            f'{self.prefix}_network_bytes_sent_total',
            'Total network bytes sent',
            registry=self.registry
        )
        self.network_bytes_received = Counter(
            f'{self.prefix}_network_bytes_received_total',
            'Total network bytes received',
            registry=self.registry
        )
    def start_prometheus_server(self):
        """Start Prometheus HTTP server for metrics export."""
        try:

            # Check if port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', self.config.prometheus_port))
            sock.close()
            if result == 0:
                logger.warning(f"Port {self.config.prometheus_port} already in use for Prometheus")
                return

            # Start server
            start_http_server(self.config.prometheus_port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    @contextmanager
    def timer(self, operation: str, **labels):
        """Context manager for timing operations."""
        start_time = time.time()
        status = "success"
        try:
            yield
        except Exception as e:
            status = "error"
            self.errors_total.labels(operation=operation, error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.processing_time.labels(operation=operation, status=status).observe(duration)
            self.requests_total.labels(operation=operation, status=status).inc()
    def record_symbol_processing(self, symbol: str, status: str, processing_time: float):
        """Record symbol processing metrics."""
        self.symbols_processed.labels(status=status).inc()
        self.processing_time.labels(operation="symbol_analysis", status=status).observe(processing_time)
    def record_analysis_signal(self, signal_type: str):
        """Record analysis signal generation."""
        self.analysis_signals.labels(signal_type=signal_type).inc()
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    def update_portfolio_metrics(self, value: float, positions: int):
        """Update portfolio-related metrics."""
        self.portfolio_value.Set[str](value)
        self.positions_count.Set[str](positions)
    def update_data_quality(self, data_source: str, freshness_seconds: float, quality_score: float):
        """Update data quality metrics."""
        self.data_freshness.labels(data_source=data_source).Set[str](freshness_seconds)
        self.data_quality_score.labels(data_source=data_source).Set[str](quality_score)
    def update_system_metrics(self):
        """Update system resource metrics."""
        if not self.config.export_system_metrics:
            return
        try:

            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.Set[str](cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.Set[str](memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.Set[str](disk_percent)

            # Network I/O
            net_io = psutil.net_io_counters()
            self.network_bytes_sent.inc(net_io.bytes_sent)
            self.network_bytes_received.inc(net_io.bytes_recv)
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
class AlertManager:
    """Alert management system for monitoring thresholds and notifications."""
    def __init__(self, config: MetricConfig, webhook_url: Optional[str] = None) -> None:
        self.config = config
        self.webhook_url = webhook_url
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        self.suppressed_alerts = Set[str]()

        # Alert thresholds
        self.thresholds = config.alert_thresholds

        # Metrics tracking for alerting
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))
    def check_performance_alerts(self, metrics: PrometheusMetrics):
        """Check for performance-based alerts."""
        alerts = []

        # Check error rate
        try:
            error_rate = self._calculate_error_rate(metrics)
            if error_rate > self.thresholds["error_rate_threshold"]:
                alerts.append({
                    "severity": "critical",
                    "alert": "high_error_rate",
                    "message": f"Error rate {error_rate:.2%} exceeds threshold {self.thresholds['error_rate_threshold']:.2%}",
                    "value": error_rate,
                    "threshold": self.thresholds["error_rate_threshold"]
                })
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")

        # Check system resource usage
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent / 100
            if cpu_usage > self.thresholds["cpu_usage_threshold"] * 100:
                alerts.append({
                    "severity": "warning",
                    "alert": "high_cpu_usage",
                    "message": f"CPU usage {cpu_usage:.1f}% exceeds threshold {self.thresholds['cpu_usage_threshold']*100:.1f}%",
                    "value": cpu_usage / 100,
                    "threshold": self.thresholds["cpu_usage_threshold"]
                })
            if memory_usage > self.thresholds["memory_usage_threshold"]:
                alerts.append({
                    "severity": "warning",
                    "alert": "high_memory_usage",
                    "message": f"Memory usage {memory_usage:.1%} exceeds threshold {self.thresholds['memory_usage_threshold']:.1%}",
                    "value": memory_usage,
                    "threshold": self.thresholds["memory_usage_threshold"]
                })
        except Exception as e:
            logger.error(f"Error checking system metrics: {e}")

        # Process alerts
        for alert in alerts:
            self.process_alert(alert)
    def _calculate_error_rate(self, metrics: PrometheusMetrics) -> float:
        """Calculate current error rate from metrics."""

        # This is a simplified calculation - in production, you'd query the actual metrics
        # For now, return a placeholder value
        return 0.01
  # 1% error rate
    def process_alert(self, alert: Dict[str, Any]):
        """Process and potentially send an alert."""
        alert_key = f"{alert['alert']}_{alert.get('component', 'system')}"

        # Check if alert is suppressed
        if alert_key in self.suppressed_alerts:
            return

        # Check if this is a new alert or escalation
        current_time = datetime.now()
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]

            # Don't re-send within 5 minutes unless it's critical
            if (current_time - last_alert["timestamp"]).seconds < 300 and alert["severity"] != "critical":
                return

        # Add timestamp and send
        alert["timestamp"] = current_time
        alert["alert_id"] = f"{alert_key}_{int(current_time.timestamp())}"
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Send notification
        self.send_alert_notification(alert)
        logger.warning(f"Alert triggered: {alert['alert']} - {alert['message']}")
    def send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification via webhook or other channels."""
        if not self.webhook_url:
            return
        try:
            payload = {
                "alert_id": alert["alert_id"],
                "severity": alert["severity"],
                "alert_type": alert["alert"],
                "message": alert["message"],
                "timestamp": alert["timestamp"].isoformat(),
                "value": alert.get("value"),
                "threshold": alert.get("threshold"),
                "source": "nse-stock-screener"
            }
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                logger.info(f"Alert notification sent successfully: {alert['alert_id']}")
            else:
                logger.error(f"Failed to send alert notification: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    def suppress_alert(self, alert_type: str, duration_minutes: int = 60):
        """Temporarily suppress an alert type."""
        self.suppressed_alerts.add(alert_type)

        # Schedule removal of suppression
        def remove_suppression():
            time.sleep(duration_minutes * 60)
            self.suppressed_alerts.discard(alert_type)
        threading.Thread(target=remove_suppression, daemon=True).start()
        logger.info(f"Alert '{alert_type}' suppressed for {duration_minutes} minutes")
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        recent_alerts = [a for a in self.alert_history if
                        (datetime.now() - a["timestamp"]).total_seconds() / 3600 < 24]
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert["alert"]] += 1
        return {
            "total_alerts_24h": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "suppressed_alerts": len(self.suppressed_alerts),
            "alert_types": Dict[str, Any](alert_counts),
            "latest_alerts": List[str](self.alert_history)[-5:]
  # Last 5 alerts
        }
class HealthChecker:
    """System health monitoring and checks."""
    def __init__(self, metrics: PrometheusMetrics) -> None:
        self.metrics = metrics
        self.health_checks = {}
        self.last_check_time = {}

        # Register default health checks
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("data_freshness", self._check_data_freshness)
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a custom health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = "healthy"
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                if result.get("status") != "healthy":
                    overall_status = "unhealthy"
                self.last_check_time[name] = datetime.now()
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_status = "unhealthy"
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        status = "healthy"
        messages = []
        if cpu_percent > 90:
            status = "unhealthy"
            messages.append(f"High CPU usage: {cpu_percent:.1f}%")
        if memory.percent > 85:
            status = "unhealthy"
            messages.append(f"High memory usage: {memory.percent:.1f}%")
        return {
            "status": status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        }
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        status = "healthy"
        if usage_percent > 90:
            status = "unhealthy"
        elif usage_percent > 80:
            status = "warning"
        return {
            "status": status,
            "usage_percent": usage_percent,
            "free_gb": disk.free / (1024**3),
            "total_gb": disk.total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check if data is fresh and up-to-date."""

        # This would check actual data sources in production
        # For now, return a placeholder
        return {
            "status": "healthy",
            "last_update": datetime.now().isoformat(),
            "age_minutes": 5,
            "timestamp": datetime.now().isoformat()
        }
class MonitoringSystem:
    """Main monitoring system that coordinates all monitoring components."""
    def __init__(self, config: Optional[MetricConfig] = None) -> None:
        self.config = config or MetricConfig()
        self.metrics = PrometheusMetrics(self.config)
        self.alert_manager = AlertManager(self.config)
        self.health_checker = HealthChecker(self.metrics)

        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        logger.info("Monitoring system initialized")
    def start_monitoring(self, check_interval: int = 60):
        """Start background monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        self.monitoring_active = True
        def monitoring_loop():
            while self.monitoring_active:
                try:

                    # Update system metrics
                    self.metrics.update_system_metrics()

                    # Run health checks
                    health_status = self.health_checker.run_health_checks()

                    # Check for alerts
                    self.alert_manager.check_performance_alerts(self.metrics)

                    # Log health status
                    if health_status["overall_status"] != "healthy":
                        logger.warning(f"System health: {health_status['overall_status']}")
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(check_interval)
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Background monitoring started (interval: {check_interval}s)")
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_status = self.health_checker.run_health_checks()
        alert_summary = self.alert_manager.get_alert_summary()
        return {
            "system_health": health_status,
            "alerts": alert_summary,
            "metrics_endpoint": f"http://localhost:{self.config.prometheus_port}/metrics",
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now().isoformat()
        }

# Global monitoring instance
monitoring_system = None
def initialize_monitoring(config: Optional[MetricConfig] = None) -> MonitoringSystem:
    """Initialize global monitoring system."""
    global monitoring_system
    if monitoring_system is None:
        monitoring_system = MonitoringSystem(config)
        monitoring_system.start_monitoring()
    return monitoring_system
def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get the global monitoring system instance."""
    return monitoring_system

# Decorators for easy monitoring integration
def monitored_operation(operation_name: str):
    """Decorator to add monitoring to operations."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            system = get_monitoring_system()
            if system:
                with system.metrics.timer(operation_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
if __name__ == "__main__":

    # Example usage and testing
    print("Testing monitoring system...")

    # Initialize monitoring
    config = MetricConfig(
        prometheus_port=8001,
  # Use different port for testing
        export_system_metrics=True
    )
    monitoring = initialize_monitoring(config)

    # Test metrics recording
    with monitoring.metrics.timer("test_operation"):
        time.sleep(0.1)
    monitoring.metrics.record_analysis_signal("BUY")
    monitoring.metrics.record_cache_operation("indicator_cache", hit=True)
    monitoring.metrics.update_portfolio_metrics(100000.0, 5)

    # Get system status
    status = monitoring.get_system_status()
    print(f"System status: {status['system_health']['overall_status']}")
    print(f"Active alerts: {status['alerts']['active_alerts']}")
    print(f"Metrics endpoint: {status['metrics_endpoint']}")

    # Simulate running for a bit
    print("Running for 5 seconds...")
    time.sleep(5)

    # Stop monitoring
    monitoring.stop_monitoring()
    print("Monitoring system test completed")
