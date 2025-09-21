"""
NSE Stock Screener - Enterprise Logging System
Structured logging with retention policies, RBAC, and compliance features.
"""
import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict[str, Any], Any, Optional, List[str], Union
import traceback
import threading
from dataclasses import dataclass, field
from enum import Enum
import hashlib
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import warnings
    warnings.warn("structlog not available, falling back to standard logging")
class LogLevel(Enum):
    """Standard log levels with security categorization."""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5
class SecurityCategory(Enum):
    """Security event categories for compliance tracking."""
    AUTHENTICATION = "auth"
    AUTHORIZATION = "authz"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "config"
    SYSTEM = "system"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    BUSINESS = "business"
@dataclass
class LogConfig:
    """Configuration for enterprise logging system."""

    # Basic configuration
    log_level: str = "INFO"
    log_format: str = "json"
  # json, structured, or standard
    # File logging
    log_dir: str = "/app/logs"
    max_file_size: int = 100 * 1024 * 1024
  # 100MB
    backup_count: int = 10

    # Retention policy
    retention_days: int = 90
    archive_after_days: int = 30

    # Security and compliance
    enable_audit_log: bool = True
    enable_security_log: bool = True
    mask_sensitive_data: bool = True

    # Performance
    async_logging: bool = True
    buffer_size: int = 1000

    # Integration
    enable_syslog: bool = False
    syslog_address: str = "localhost:514"
    enable_prometheus_metrics: bool = True

    # Environment-specific
    environment: str = "production"
    application_name: str = "nse-screener"
    version: str = "1.0.0"
class SecurityAuditLogger:
    """Dedicated logger for security and audit events."""
    def __init__(self, config: LogConfig) -> None:
        self.config = config
        self.logger = self._setup_security_logger()
        self._lock = threading.Lock()
    def _setup_security_logger(self):
        """Setup dedicated security audit logger."""
        logger = logging.getLogger("nse_screener.security")
        logger.setLevel(logging.INFO)

        # Security log file with restricted permissions
        security_log_path = Path(self.config.log_dir) / "security_audit.log"
        security_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler for security logs
        handler = logging.handlers.RotatingFileHandler(
            security_log_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )

        # JSON formatter for security events
        formatter = JsonSecurityFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    def log_security_event(
        self,
        category: SecurityCategory,
        event: str,
        severity: str = "INFO",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log a security event with standardized format."""
        with self._lock:
            event_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "category": category.value,
                "event": event,
                "severity": severity,
                "user_id": user_id,
                "session_id": session_id,
                "source_ip": source_ip,
                "details": details or {},
                "environment": self.config.environment,
                "application": self.config.application_name
            }

            # Calculate event hash for integrity
            event_hash = self._calculate_event_hash(event_data)
            event_data["event_hash"] = event_hash
            self.logger.info("Security Event", extra={"security_event": event_data})
    def _calculate_event_hash(self, event_data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of event for integrity verification."""

        # Remove hash field if present
        data_copy = event_data.copy()
        data_copy.pop("event_hash", None)

        # Create deterministic string representation
        data_string = json.dumps(data_copy, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
class JsonSecurityFormatter(logging.Formatter):
    """JSON formatter for security events with PII masking."""
    def __init__(self) -> None:
        super().__init__()
        self.sensitive_fields = {
            "password", "token", "api_key", "secret", "credential",
            "ssn", "credit_card", "bank_account", "auth"
        }
    def format(self, record):
        """Format log record as JSON with security considerations."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }

        # Add extra fields from record
        if hasattr(record, "security_event"):
            log_entry.update(record.security_event)

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Mask sensitive data
        self._mask_sensitive_data(log_entry)
        return json.dumps(log_entry, default=str)
    def _mask_sensitive_data(self, data: Union[Dict[str, Any], List[str], str], path: str = ""):
        """Recursively mask sensitive data in log entries."""
        if isinstance($1, Dict[str, Any]):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    data[key] = self._mask_value(str(value))
                elif isinstance(value, (Dict[str, Any], List[str])):
                    self._mask_sensitive_data(value, current_path)
        elif isinstance($1, List[str]):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                if isinstance(item, (Dict[str, Any], List[str])):
                    self._mask_sensitive_data(item, current_path)
    def _mask_value(self, value: str) -> str:
        """Mask sensitive values."""
        if len(value) <= 4:
            return "****"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
class EnterpriseLogger:
    """
    Enterprise-grade logging system for NSE Stock Screener.
    Features:
    - Structured JSON logging
    - Security audit trail
    - Retention policies
    - Performance metrics
    - RBAC compliance
    """
    def __init__(self, config: Optional[LogConfig] = None) -> None:
        self.config = config or LogConfig()
        self.security_logger = SecurityAuditLogger(self.config)
        self.main_logger = self._setup_main_logger()
        self.metrics = LogMetrics() if self.config.enable_prometheus_metrics else None

        # Setup retention policy
        self._setup_retention_policy()

        # Register cleanup handlers
        import atexit
        atexit.register(self._cleanup)
    def _setup_main_logger(self):
        """Setup main application logger."""
        if STRUCTLOG_AVAILABLE:
            return self._setup_structlog()
        else:
            return self._setup_standard_logging()
    def _setup_structlog(self):
        """Setup structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(self.config.log_level.upper())
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger("nse_screener")
    def _setup_standard_logging(self):
        """Setup standard Python logging with JSON formatter."""
        logger = logging.getLogger("nse_screener")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_dir / "application.log",
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        app_handler.setFormatter(JsonSecurityFormatter())
        logger.addHandler(app_handler)

        # Console handler for development
        if self.config.environment != "production":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)
        return logger
    def _setup_retention_policy(self):
        """Setup log retention and archival policies."""
        self.retention_thread = threading.Thread(
            target=self._retention_worker,
            daemon=True,
            name="LogRetentionWorker"
        )
        self.retention_thread.start()
    def _retention_worker(self):
        """Background worker for log retention and cleanup."""
        import time
        while True:
            try:
                self._enforce_retention_policy()

                # Run retention check daily
                time.sleep(24 * 60 * 60)
            except Exception as e:
                self.main_logger.error(f"Retention policy error: {e}")
                time.sleep(60 * 60)
  # Retry in 1 hour on error
    def _enforce_retention_policy(self):
        """Enforce log retention and archival policies."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        archive_date = datetime.now() - timedelta(days=self.config.archive_after_days)
        for log_file in log_dir.glob("*.log*"):
            try:
                file_stat = log_file.stat()
                file_date = datetime.fromtimestamp(file_stat.st_mtime)
                if file_date < cutoff_date:

                    # Delete old log files
                    log_file.unlink()
                    self.main_logger.info(f"Deleted old log file: {log_file}")
                elif file_date < archive_date:

                    # Archive old log files (compress)
                    self._archive_log_file(log_file)
            except Exception as e:
                self.main_logger.error(f"Error processing log file {log_file}: {e}")
    def _archive_log_file(self, log_file: Path):
        """Archive (compress) old log files."""
        import gzip
        import shutil
        archived_file = log_file.with_suffix(log_file.suffix + ".gz")
        if not archived_file.exists():
            with open(log_file, 'rb') as f_in:
                with gzip.open(archived_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log_file.unlink()
            self.main_logger.info(f"Archived log file: {log_file} -> {archived_file}")
    def _cleanup(self):
        """Cleanup resources on shutdown."""
        if hasattr(self, 'retention_thread') and self.retention_thread.is_alive():

            # Note: Can't join daemon thread on shutdown, just log
            self.main_logger.info("Logging system shutting down")

    # Public logging interface
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", message, **kwargs)
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("info", message, **kwargs)
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", message, **kwargs)
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("error", message, **kwargs)
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("critical", message, **kwargs)
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        if self.metrics:
            self.metrics.increment_log_count(level)

        # Add context information
        context = {
            "application": self.config.application_name,
            "version": self.config.version,
            "environment": self.config.environment,
            **kwargs
        }

        # Log to main logger
        if STRUCTLOG_AVAILABLE:
            getattr(self.main_logger, level)(message, **context)
        else:
            getattr(self.main_logger, level)(message, extra=context)
    def audit(
        self,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit event for compliance."""
        self.security_logger.log_security_event(
            category=SecurityCategory.AUDIT,
            event=f"AUDIT: {action} on {resource}",
            severity="INFO" if success else "WARNING",
            user_id=user_id,
            details={
                "action": action,
                "resource": resource,
                "success": success,
                **(details or {})
            }
        )
    def security_event(
        self,
        category: SecurityCategory,
        event: str,
        severity: str = "INFO",
        **kwargs
    ):
        """Log security event."""
        self.security_logger.log_security_event(
            category=category,
            event=event,
            severity=severity,
            **kwargs
        )
class LogMetrics:
    """Prometheus metrics for logging system."""
    def __init__(self) -> None:
        try:
            from prometheus_client import Counter, Histogram
            self.log_count = Counter(
                'nse_screener_log_total',
                'Total number of log messages',
                ['level', 'logger']
            )
            self.log_duration = Histogram(
                'nse_screener_log_duration_seconds',
                'Time spent logging messages'
            )
        except ImportError:
            self.log_count = None
            self.log_duration = None
    def increment_log_count(self, level: str, logger: str = "main"):
        """Increment log count metric."""
        if self.log_count:
            self.log_count.labels(level=level, logger=logger).inc()

# Global logger instance
_enterprise_logger: Optional[EnterpriseLogger] = None
def get_logger() -> EnterpriseLogger:
    """Get the global enterprise logger instance."""
    global _enterprise_logger
    if _enterprise_logger is None:

        # Load configuration from environment
        config = LogConfig(
            log_level=os.getenv("NSE_SCREENER_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("NSE_SCREENER_LOGS_PATH", "/app/logs"),
            environment=os.getenv("NSE_SCREENER_ENV", "production"),
            enable_prometheus_metrics=os.getenv("NSE_SCREENER_PROMETHEUS_ENABLED", "true").lower() == "true"
        )
        _enterprise_logger = EnterpriseLogger(config)
    return _enterprise_logger
def init_logging(config: Optional[LogConfig] = None) -> EnterpriseLogger:
    """Initialize enterprise logging with custom configuration."""
    global _enterprise_logger
    _enterprise_logger = EnterpriseLogger(config)
    return _enterprise_logger
if __name__ == "__main__":

    # Example usage and testing
    logger = get_logger()

    # Test different log levels
    logger.info("NSE Stock Screener starting up", component="main")
    logger.debug("Debug information", module="test")
    logger.warning("This is a warning", category="system")
    logger.error("This is an error", error_code="E001")

    # Test audit logging
    logger.audit("LOGIN", "user_session", user_id="user123", success=True)

    # Test security events
    logger.security_event(
        SecurityCategory.AUTHENTICATION,
        "User login attempt",
        severity="INFO",
        user_id="user123",
        source_ip="192.168.1.100"
    )
    print("Logging test completed. Check log files in /app/logs/")
