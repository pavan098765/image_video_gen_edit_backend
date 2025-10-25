# db.py
import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Text,
    Boolean,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.mutable import MutableDict

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./controller.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Worker(Base):
    __tablename__ = "workers"

    worker_id = Column(String, primary_key=True, index=True)
    container_id = Column(
        String,
        index=True,
    )  # stores QuickPod pod_uuid (canonical pod id, legacy column name kept for compatibility)
    endpoint = Column(String, nullable=True)
    model = Column(String, index=True)
    is_warm_pod = Column(Boolean, default=False)
    status = Column(String, default="ready", index=True)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    info = Column(MutableDict.as_mutable(JSON), nullable=True)
    health_status = Column(String, default="healthy", index=True)
    last_active = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Resource/metrics fields
    resource_metrics = Column(JSON, nullable=True)  # JSON dict of metrics
    error_count = Column(Integer, default=0)
    queue_size = Column(Integer, default=0)
    resource_limits = Column(JSON, nullable=True)  # optional limits per worker
    
    # Job tracking fields
    total_jobs_processed = Column(Integer, default=0)
    job_id_counts = Column(MutableDict.as_mutable(JSON), default=dict)  # Track jobs per worker
    machine_id = Column(String, nullable=True, index=True)  # Store machine ID for machine-based strategy

    __table_args__ = (
        Index("ix_worker_model_status", "model", "status"),
        Index("ix_worker_container", "container_id"),
        Index("ix_worker_heartbeat", "last_heartbeat"),
        Index("ix_worker_active", "last_active"),
        Index("ix_worker_health", "health_status"),
    )

    @property
    def pod_uuid(self) -> str:
        """Alias for container_id, which actually stores the QuickPod pod_uuid."""
        return self.container_id


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)
    model = Column(String, index=True)
    prompt = Column(String, nullable=True)
    params = Column(MutableDict.as_mutable(JSON), default=dict, nullable=True)
    status = Column(String, default="pending", index=True)
    worker_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    timeout_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    result_r2_key = Column(String, nullable=True)
    output_key = Column(String, nullable=True)
    output_url = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    error_log = Column(JSON, nullable=True, default=list)


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_uuid = Column(String, unique=False, index=True, nullable=True)
    fingerprint_hash = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    request_count = Column(Integer, default=0, nullable=False)
    banned = Column(Boolean, default=False, nullable=False)
    meta = Column(MutableDict.as_mutable(JSON), nullable=True)

    __table_args__ = (
        Index("ix_device_lookup", "device_uuid", "fingerprint_hash"),
        Index("ix_device_banned", "banned"),
        Index("ix_device_last_seen", "last_seen"),
    )


def init_db():
    Base.metadata.create_all(bind=engine)
