# db.py
"""
Database Layer

Provides:
- SQLAlchemy models with proper indexes
- Database session management with connection pooling
- Timezone-aware datetime handling (UTC enforcement)
- Audit log table for security events
- Migration support ready (Alembic compatible)
"""

import os
from datetime import datetime, timezone
from typing import Optional, Generator, Any, Dict
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Text,
    Boolean,
    Float,
    Index,
    event,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.pool import QueuePool, StaticPool

Base = declarative_base()


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime"""
    return datetime.now(timezone.utc)


def utc_naive_now() -> datetime:
    """Return current UTC time as naive datetime (for SQLite compatibility)"""
    return datetime.utcnow()


# =============================================================================
# Worker Model
# =============================================================================

class Worker(Base):
    """
    Tracks container pods, status, heartbeat, and metrics.
    
    Note: container_id column stores QuickPod pod_uuid (canonical pod id).
    Column name kept for backward compatibility.
    """
    __tablename__ = "workers"

    worker_id = Column(String(128), primary_key=True, index=True)
    container_id = Column(String(128), index=True)  # Stores pod_uuid
    endpoint = Column(String(512), nullable=True)
    model = Column(String(64), index=True)
    is_warm_pod = Column(Boolean, default=False)
    status = Column(String(32), default="ready", index=True)
    health_status = Column(String(32), default="healthy", index=True)
    
    # Timestamps (UTC)
    last_heartbeat = Column(DateTime, default=utc_naive_now)
    last_active = Column(DateTime, default=utc_naive_now)
    last_job_finished_at = Column(DateTime, default=utc_naive_now)
    created_at = Column(DateTime, default=utc_naive_now)
    updated_at = Column(DateTime, default=utc_naive_now, onupdate=utc_naive_now)
    
    # Resource/metrics fields
    info = Column(MutableDict.as_mutable(JSON), nullable=True)
    resource_metrics = Column(JSON, nullable=True)
    resource_limits = Column(JSON, nullable=True)
    
    # Tracking fields
    error_count = Column(Integer, default=0)
    queue_size = Column(Integer, default=0)
    total_jobs_processed = Column(Integer, default=0)
    job_id_counts = Column(MutableDict.as_mutable(JSON), default=dict)
    machine_id = Column(String(64), nullable=True, index=True)

    __table_args__ = (
        Index("ix_worker_model_status", "model", "status"),
        Index("ix_worker_container", "container_id"),
        Index("ix_worker_heartbeat", "last_heartbeat"),
        Index("ix_worker_active", "last_active"),
        Index("ix_worker_health", "health_status"),
        Index("ix_worker_warm_model", "is_warm_pod", "model"),
    )

    @property
    def pod_uuid(self) -> str:
        """Alias for container_id, which stores the QuickPod pod_uuid."""
        return self.container_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "worker_id": self.worker_id,
            "container_id": self.container_id,
            "pod_uuid": self.pod_uuid,
            "endpoint": self.endpoint,
            "model": self.model,
            "is_warm_pod": self.is_warm_pod,
            "status": self.status,
            "health_status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "error_count": self.error_count,
            "queue_size": self.queue_size,
            "total_jobs_processed": self.total_jobs_processed,
            "machine_id": self.machine_id,
        }


# =============================================================================
# Job Model
# =============================================================================

class Job(Base):
    """
    Tracks job state, assignments, and results.
    """
    __tablename__ = "jobs"

    job_id = Column(String(128), primary_key=True, index=True)
    model = Column(String(64), index=True)
    prompt = Column(Text, nullable=True)
    params = Column(MutableDict.as_mutable(JSON), default=dict, nullable=True)
    
    # Status tracking
    status = Column(String(32), default="pending", index=True)
    worker_id = Column(String(128), nullable=True, index=True)
    
    # Timestamps (UTC)
    created_at = Column(DateTime, default=utc_naive_now)
    updated_at = Column(DateTime, default=utc_naive_now, onupdate=utc_naive_now)
    completed_at = Column(DateTime, nullable=True)
    timeout_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)  # When job actually started processing
    queued_at = Column(DateTime, nullable=True)  # When job entered QUEUED state
    dispatched_at = Column(DateTime, nullable=True)  # When job entered DISPATCHING state
    
    # Timing metrics (in seconds)
    total_time_seconds = Column(Float, nullable=True)  # created_at to completed_at
    processing_time_seconds = Column(Float, nullable=True)  # started_at to completed_at
    queue_time_seconds = Column(Float, nullable=True)  # queued_at to started_at
    dispatch_time_seconds = Column(Float, nullable=True)  # dispatched_at to started_at
    
    # Retry management
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Results
    result_r2_key = Column(String(512), nullable=True)
    output_key = Column(String(512), nullable=True)
    output_url = Column(String(1024), nullable=True)
    
    # Error tracking
    error = Column(Text, nullable=True)
    error_log = Column(MutableList.as_mutable(JSON), nullable=True, default=list)
    
    # Client notification
    fcm_token = Column(String(512), nullable=True)
    notification_sent = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_job_model_status", "model", "status"),
        Index("ix_job_worker", "worker_id"),
        Index("ix_job_created", "created_at"),
        Index("ix_job_updated", "updated_at"),
        Index("ix_job_timeout", "timeout_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "model": self.model,
            "prompt": self.prompt,
            "params": self.params,
            "status": self.status,
            "worker_id": self.worker_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "retry_count": self.retry_count,
            "result_r2_key": self.result_r2_key,
            "output_url": self.output_url,
            "error": self.error,
            "error_log": self.error_log,
            # Timing metrics
            "total_time_seconds": self.total_time_seconds,
            "processing_time_seconds": self.processing_time_seconds,
            "queue_time_seconds": self.queue_time_seconds,
            "dispatch_time_seconds": self.dispatch_time_seconds,
        }


# =============================================================================
# Device Model
# =============================================================================

class Device(Base):
    """
    Tracks authenticated client devices.
    """
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_uuid = Column(String(128), unique=False, index=True, nullable=True)
    fingerprint_hash = Column(String(64), index=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    last_seen = Column(DateTime, default=utc_naive_now, onupdate=utc_naive_now, nullable=False)
    
    # Tracking
    request_count = Column(Integer, default=0, nullable=False)
    banned = Column(Boolean, default=False, nullable=False)
    ban_reason = Column(String(256), nullable=True)
    
    # Metadata
    meta = Column(MutableDict.as_mutable(JSON), nullable=True)

    __table_args__ = (
        Index("ix_device_lookup", "device_uuid", "fingerprint_hash"),
        Index("ix_device_banned", "banned"),
        Index("ix_device_last_seen", "last_seen"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "device_uuid": self.device_uuid,
            "fingerprint_hash": self.fingerprint_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "request_count": self.request_count,
            "banned": self.banned,
        }


# =============================================================================
# Audit Log Model
# =============================================================================

class AuditLog(Base):
    """
    Security and operational audit log.
    
    Records significant events for compliance and debugging.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Event classification
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), default="info", index=True)  # info, warning, error
    
    # Actor and target
    actor = Column(String(128), nullable=True, index=True)  # worker_id, system, admin
    actor_ip = Column(String(64), nullable=True, index=True)
    target = Column(String(256), nullable=True, index=True)  # job_id, pod_uuid, ip
    target_type = Column(String(32), nullable=True)  # job, pod, worker, ip
    
    # Event details
    message = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=utc_naive_now, nullable=False, index=True)

    __table_args__ = (
        Index("ix_audit_event_time", "event_type", "timestamp"),
        Index("ix_audit_actor_time", "actor", "timestamp"),
        Index("ix_audit_target_time", "target", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "actor": self.actor,
            "actor_ip": self.actor_ip,
            "target": self.target,
            "target_type": self.target_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# =============================================================================
# IP Ban Model
# =============================================================================

class IPBan(Base):
    """
    Persisted IP bans.
    """
    __tablename__ = "ip_bans"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ip = Column(String(64), nullable=False, unique=True, index=True)
    reason = Column(String(256), nullable=True)
    permanent = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    created_by = Column(String(64), nullable=True)  # admin user or "system"

    __table_args__ = (
        Index("ix_ipban_expires", "expires_at"),
    )


# =============================================================================
# Database Engine and Session Management
# =============================================================================

class DatabaseManager:
    """
    Database connection and session management.
    
    Provides:
    - Connection pooling for concurrent requests
    - Context manager for safe session handling
    - Migration-ready schema initialization
    """
    
    _instance: Optional['DatabaseManager'] = None
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite:///./controller.db"
        )
        self._engine = None
        self._session_factory = None
    
    @classmethod
    def get_instance(cls, database_url: Optional[str] = None) -> 'DatabaseManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(database_url)
        return cls._instance
    
    @property
    def engine(self):
        """Get or create database engine with connection pooling"""
        if self._engine is None:
            is_sqlite = self.database_url.startswith("sqlite")
            
            if is_sqlite:
                # SQLite-specific configuration
                self._engine = create_engine(
                    self.database_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,  # Single connection for SQLite
                    echo=os.getenv("DB_ECHO", "false").lower() == "true",
                )
            else:
                # PostgreSQL/MySQL configuration with connection pooling
                pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
                max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
                pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
                pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))
                
                self._engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_recycle=pool_recycle,
                    echo=os.getenv("DB_ECHO", "false").lower() == "true",
                )
        
        return self._engine
    
    @property
    def session_factory(self):
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        return self._session_factory
    
    def init_db(self):
        """Initialize database schema"""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                # use session
                session.commit()
        """
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_db_dependency(self) -> Generator[Session, None, None]:
        """
        FastAPI dependency for database sessions.
        
        Usage in routes:
            @app.get("/items")
            def get_items(db: Session = Depends(get_db)):
                ...
        """
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()


# =============================================================================
# Global Instances (for backward compatibility)
# =============================================================================

# These are initialized lazily to support testing and configuration

_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager.get_instance()
    return _db_manager


def init_db(database_url: Optional[str] = None):
    """Initialize database with schema"""
    global _db_manager
    if database_url:
        _db_manager = DatabaseManager(database_url)
    else:
        _db_manager = get_db_manager()
    _db_manager.init_db()


def get_session() -> Generator[Session, None, None]:
    """Get a database session (FastAPI dependency compatible)"""
    return get_db_manager().get_db_dependency()


# Backward compatibility aliases
def SessionLocal():
    """Create a new session (legacy compatibility)"""
    return get_db_manager().session_factory()


# =============================================================================
# Helper Functions
# =============================================================================

def create_audit_log(
    session: Session,
    event_type: str,
    message: Optional[str] = None,
    actor: Optional[str] = None,
    actor_ip: Optional[str] = None,
    target: Optional[str] = None,
    target_type: Optional[str] = None,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """
    Create an audit log entry.
    
    Args:
        session: Database session
        event_type: Type of event (e.g., "job_created", "pod_destroyed")
        message: Human-readable message
        actor: Who performed the action
        actor_ip: IP address of actor
        target: What was affected
        target_type: Type of target (job, pod, worker, ip)
        severity: info, warning, or error
        details: Additional JSON details
        
    Returns:
        Created AuditLog instance
    """
    log = AuditLog(
        event_type=event_type,
        message=message,
        actor=actor,
        actor_ip=actor_ip,
        target=target,
        target_type=target_type,
        severity=severity,
        details=details,
        timestamp=utc_naive_now(),
    )
    session.add(log)
    return log


def get_or_create_ip_ban(
    session: Session,
    ip: str,
    reason: Optional[str] = None,
    permanent: bool = False,
    expires_at: Optional[datetime] = None,
    created_by: Optional[str] = None,
) -> IPBan:
    """
    Get existing IP ban or create new one.
    
    Args:
        session: Database session
        ip: IP address to ban
        reason: Reason for ban
        permanent: Whether ban is permanent
        expires_at: When ban expires (if not permanent)
        created_by: Who created the ban
        
    Returns:
        IPBan instance
    """
    existing = session.query(IPBan).filter(IPBan.ip == ip).first()
    if existing:
        # Update existing ban
        existing.reason = reason or existing.reason
        existing.permanent = permanent
        existing.expires_at = expires_at
        return existing
    
    # Create new ban
    ban = IPBan(
        ip=ip,
        reason=reason,
        permanent=permanent,
        expires_at=expires_at,
        created_by=created_by,
    )
    session.add(ban)
    return ban


def remove_ip_ban(session: Session, ip: str) -> bool:
    """
    Remove an IP ban.
    
    Args:
        session: Database session
        ip: IP address to unban
        
    Returns:
        True if ban was removed, False if not found
    """
    ban = session.query(IPBan).filter(IPBan.ip == ip).first()
    if ban:
        session.delete(ban)
        return True
    return False


def get_active_ip_bans(session: Session) -> list:
    """
    Get all active (non-expired) IP bans.
    
    Args:
        session: Database session
        
    Returns:
        List of active IPBan instances
    """
    now = utc_naive_now()
    return session.query(IPBan).filter(
        (IPBan.permanent == True) | 
        (IPBan.expires_at == None) | 
        (IPBan.expires_at > now)
    ).all()


def cleanup_expired_bans(session: Session) -> int:
    """
    Remove expired IP bans.
    
    Args:
        session: Database session
        
    Returns:
        Number of bans removed
    """
    now = utc_naive_now()
    result = session.query(IPBan).filter(
        IPBan.permanent == False,
        IPBan.expires_at != None,
        IPBan.expires_at <= now
    ).delete()
    return result

