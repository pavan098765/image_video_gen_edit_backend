# logging_config.py
"""
Structured Logging Configuration

Provides:
- JSON formatted logging for production
- Human-readable logging for development
- Separate security audit logging
- Log rotation support
- Non-sensitive data filtering
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, field


# Sensitive fields that should be redacted
SENSITIVE_FIELDS: Set[str] = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credentials",
    "access_key",
    "secret_key",
    "private_key",
    "signature",
    "x-signature",
    "x-fingerprint",
}


def is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data"""
    key_lower = key.lower()
    return any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS)


def redact_sensitive(data: Any, depth: int = 0) -> Any:
    """
    Recursively redact sensitive fields from data structures.
    
    Args:
        data: Data to redact
        depth: Current recursion depth (to prevent infinite loops)
        
    Returns:
        Redacted data structure
    """
    if depth > 10:  # Prevent infinite recursion
        return "[MAX_DEPTH]"
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if is_sensitive_key(str(key)):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_sensitive(value, depth + 1)
        return result
    elif isinstance(data, (list, tuple)):
        return [redact_sensitive(item, depth + 1) for item in data]
    else:
        return data


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs logs in JSON format suitable for log aggregation systems
    like ELK, Splunk, or Cloudwatch.
    """
    
    def __init__(
        self, 
        include_stack_trace: bool = True,
        redact_sensitive: bool = True
    ):
        super().__init__()
        self.include_stack_trace = include_stack_trace
        self.redact_sensitive = redact_sensitive
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process/thread info
        log_entry["process"] = record.process
        log_entry["thread"] = record.thread
        
        # Add exception info if present
        if record.exc_info and self.include_stack_trace:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName"
            }:
                if self.redact_sensitive:
                    value = redact_sensitive(value) if isinstance(value, (dict, list)) else value
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    
    Provides color-coded log levels for better readability.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        # Apply colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        result = super().format(record)
        
        # Add exception info
        if record.exc_info:
            result += f"\n{''.join(traceback.format_exception(*record.exc_info))}"
        
        return result


class SecurityAuditFormatter(logging.Formatter):
    """
    Security audit log formatter.
    
    Specialized format for security-related events with
    consistent structure for compliance/auditing.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": getattr(record, "event_type", "security"),
            "severity": record.levelname,
            "message": record.getMessage(),
            "source_ip": getattr(record, "source_ip", None),
            "user_agent": getattr(record, "user_agent", None),
            "path": getattr(record, "path", None),
            "method": getattr(record, "method", None),
            "actor": getattr(record, "actor", None),
            "target": getattr(record, "target", None),
            "outcome": getattr(record, "outcome", None),
            "details": getattr(record, "details", None),
        }
        
        # Remove None values
        audit_entry = {k: v for k, v in audit_entry.items() if v is not None}
        
        return json.dumps(audit_entry, default=str)


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    
    # General settings
    level: str = "INFO"
    json_format: bool = False  # True for production
    
    # Console output
    console_enabled: bool = True
    console_colors: bool = True
    
    # File output
    file_enabled: bool = True
    file_path: str = "logs/controller.log"
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    
    # Security audit log
    security_log_enabled: bool = True
    security_log_path: str = "logs/security.log"
    security_log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    security_log_backup_count: int = 10
    
    # Log levels per logger
    logger_levels: Dict[str, str] = field(default_factory=lambda: {
        "uvicorn": "INFO",
        "uvicorn.access": "WARNING",
        "uvicorn.error": "ERROR",
        "sqlalchemy.engine": "WARNING",
        "httpx": "WARNING",
        "httpcore": "WARNING",
    })


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        config: Optional LoggingConfig instance. If not provided,
                settings are loaded from environment variables.
    """
    if config is None:
        config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            json_format=os.getenv("LOG_JSON_FORMAT", "false").lower() == "true",
            console_enabled=os.getenv("LOG_CONSOLE", "true").lower() == "true",
            console_colors=os.getenv("LOG_COLORS", "true").lower() == "true",
            file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
            file_path=os.getenv("LOG_FILE_PATH", "logs/controller.log"),
            security_log_enabled=os.getenv("SECURITY_LOG_ENABLED", "true").lower() == "true",
            security_log_path=os.getenv("SECURITY_LOG_PATH", "logs/security.log"),
        )
    
    # Create log directories
    for path in [config.file_path, config.security_log_path]:
        log_dir = Path(path).parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        if config.json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColoredFormatter(use_colors=config.console_colors))
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.file_enabled:
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
        )
        if config.json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
            ))
        root_logger.addHandler(file_handler)
    
    # Security audit logger
    if config.security_log_enabled:
        security_logger = logging.getLogger("security")
        security_logger.setLevel(logging.INFO)
        security_logger.propagate = False  # Don't propagate to root
        
        security_handler = RotatingFileHandler(
            config.security_log_path,
            maxBytes=config.security_log_max_bytes,
            backupCount=config.security_log_backup_count,
        )
        security_handler.setFormatter(SecurityAuditFormatter())
        security_logger.addHandler(security_handler)
    
    # Configure specific logger levels
    for logger_name, level in config.logger_levels.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Log startup
    root_logger.info(
        "Logging configured",
        extra={
            "config": {
                "level": config.level,
                "json_format": config.json_format,
                "console_enabled": config.console_enabled,
                "file_enabled": config.file_enabled,
                "file_path": config.file_path if config.file_enabled else None,
                "security_log_enabled": config.security_log_enabled,
            }
        }
    )


class AuditLogger:
    """
    Security audit logger utility class.
    
    Provides convenience methods for logging security-related events
    with consistent structure.
    """
    
    def __init__(self):
        self._logger = logging.getLogger("security")
    
    def log_event(
        self,
        event_type: str,
        message: str,
        level: int = logging.INFO,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security audit event"""
        self._logger.log(
            level,
            message,
            extra={
                "event_type": event_type,
                "source_ip": source_ip,
                "user_agent": user_agent,
                "path": path,
                "method": method,
                "actor": actor,
                "target": target,
                "outcome": outcome,
                "details": redact_sensitive(details) if details else None,
            }
        )
    
    def auth_success(
        self,
        source_ip: str,
        auth_type: str,
        actor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log successful authentication"""
        self.log_event(
            event_type="auth_success",
            message=f"Authentication successful ({auth_type})",
            source_ip=source_ip,
            actor=actor,
            outcome="success",
            details=details,
        )
    
    def auth_failure(
        self,
        source_ip: str,
        auth_type: str,
        reason: str,
        actor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log authentication failure"""
        self.log_event(
            event_type="auth_failure",
            message=f"Authentication failed ({auth_type}): {reason}",
            level=logging.WARNING,
            source_ip=source_ip,
            actor=actor,
            outcome="failure",
            details=details,
        )
    
    def ip_banned(
        self,
        source_ip: str,
        reason: str,
        permanent: bool = False,
        duration: Optional[int] = None,
    ) -> None:
        """Log IP ban"""
        ban_type = "permanent" if permanent else f"temporary ({duration}s)"
        self.log_event(
            event_type="ip_banned",
            message=f"IP banned ({ban_type}): {reason}",
            level=logging.WARNING,
            source_ip=source_ip,
            outcome="banned",
            details={"permanent": permanent, "duration": duration, "reason": reason},
        )
    
    def ip_unbanned(self, source_ip: str, reason: Optional[str] = None) -> None:
        """Log IP unban"""
        self.log_event(
            event_type="ip_unbanned",
            message=f"IP unbanned: {source_ip}",
            source_ip=source_ip,
            outcome="unbanned",
            details={"reason": reason} if reason else None,
        )
    
    def rate_limited(
        self,
        source_ip: str,
        path: str,
        requests_count: int,
    ) -> None:
        """Log rate limiting event"""
        self.log_event(
            event_type="rate_limited",
            message=f"Rate limit exceeded ({requests_count} requests)",
            level=logging.WARNING,
            source_ip=source_ip,
            path=path,
            outcome="blocked",
            details={"requests_count": requests_count},
        )
    
    def suspicious_activity(
        self,
        source_ip: str,
        activity_type: str,
        path: Optional[str] = None,
        user_agent: Optional[str] = None,
        score: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log suspicious activity"""
        self.log_event(
            event_type="suspicious_activity",
            message=f"Suspicious activity detected: {activity_type}",
            level=logging.WARNING,
            source_ip=source_ip,
            path=path,
            user_agent=user_agent,
            outcome="flagged",
            details={**(details or {}), "suspicion_score": score} if score else details,
        )
    
    def worker_registered(
        self,
        worker_id: str,
        pod_uuid: str,
        source_ip: str,
        model: str,
    ) -> None:
        """Log worker registration"""
        self.log_event(
            event_type="worker_registered",
            message=f"Worker registered: {worker_id}",
            source_ip=source_ip,
            actor=worker_id,
            target=pod_uuid,
            outcome="success",
            details={"model": model},
        )
    
    def job_created(
        self,
        job_id: str,
        model: str,
        source_ip: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        """Log job creation"""
        self.log_event(
            event_type="job_created",
            message=f"Job created: {job_id}",
            source_ip=source_ip,
            actor=actor,
            target=job_id,
            outcome="success",
            details={"model": model},
        )
    
    def job_completed(
        self,
        job_id: str,
        worker_id: str,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Log job completion"""
        self.log_event(
            event_type="job_completed",
            message=f"Job completed: {job_id}",
            actor=worker_id,
            target=job_id,
            outcome="success",
            details={"duration_seconds": duration_seconds} if duration_seconds else None,
        )
    
    def job_failed(
        self,
        job_id: str,
        worker_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log job failure"""
        self.log_event(
            event_type="job_failed",
            message=f"Job failed: {job_id}",
            level=logging.ERROR,
            actor=worker_id,
            target=job_id,
            outcome="failure",
            details={"error": error} if error else None,
        )
    
    def pod_created(
        self,
        pod_uuid: str,
        model: str,
        machine_id: Optional[str] = None,
    ) -> None:
        """Log pod creation"""
        self.log_event(
            event_type="pod_created",
            message=f"Pod created: {pod_uuid}",
            target=pod_uuid,
            outcome="success",
            details={"model": model, "machine_id": machine_id},
        )
    
    def pod_destroyed(
        self,
        pod_uuid: str,
        reason: Optional[str] = None,
    ) -> None:
        """Log pod destruction"""
        self.log_event(
            event_type="pod_destroyed",
            message=f"Pod destroyed: {pod_uuid}",
            target=pod_uuid,
            outcome="success",
            details={"reason": reason} if reason else None,
        )


# Global audit logger instance
audit_logger = AuditLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def get_audit_logger() -> AuditLogger:
    """Get the security audit logger"""
    return audit_logger

