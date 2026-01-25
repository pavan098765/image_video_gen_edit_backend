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
    
    # Cost tracking - GPU infrastructure cost
    gpu_hourly_rate = Column(Float, nullable=True)  # $/hour rate from offer
    offer_id = Column(Integer, nullable=True)  # QuickPod offer ID
    gpu_type = Column(String(128), nullable=True)  # e.g., "NVIDIA GeForce RTX 3090"
    destroyed_at = Column(DateTime, nullable=True)  # When pod was destroyed
    total_pod_cost_usd = Column(Float, nullable=True)  # Total cost for pod lifetime

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

    # Credit tracking - for atomic credit operations
    user_id = Column(String(128), ForeignKey("users.id"), nullable=True, index=True)
    credits_locked = Column(Integer, default=0, nullable=False)  # Credits deducted at submit
    credits_refunded = Column(Integer, default=0, nullable=False)  # Credits refunded on failure
    
    # Cost tracking - GPU infrastructure cost for this job
    # Note: job_cost_usd = processing_cost + pod_overhead_share_usd (final total cost)
    # To get processing cost only: job_cost_usd - pod_overhead_share_usd
    pod_uuid = Column(String(128), nullable=True, index=True)  # Pod that processed this job
    gpu_hourly_rate = Column(Float, nullable=True)  # $/hour rate for the GPU used
    job_cost_usd = Column(Float, nullable=True)  # TOTAL cost (processing + overhead) in USD
    pod_overhead_share_usd = Column(Float, nullable=True)  # Share of pod startup/shutdown overhead

    __table_args__ = (
        Index("ix_job_model_status", "model", "status"),
        Index("ix_job_worker", "worker_id"),
        Index("ix_job_created", "created_at"),
        Index("ix_job_updated", "updated_at"),
        Index("ix_job_timeout", "timeout_at"),
        Index("ix_job_user", "user_id"),
        Index("ix_job_pod", "pod_uuid"),
    )

    @property
    def processing_cost_usd(self) -> float:
        """Get processing cost only (total cost minus overhead)"""
        if self.job_cost_usd is None:
            return 0.0
        overhead = self.pod_overhead_share_usd or 0.0
        return max(0.0, self.job_cost_usd - overhead)
    
    @property
    def total_cost_usd(self) -> float:
        """Alias for job_cost_usd for clarity (total = processing + overhead)"""
        return self.job_cost_usd or 0.0
    
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
            # Cost metrics
            "job_cost_usd": self.job_cost_usd,  # Total cost (processing + overhead)
            "processing_cost_usd": self.processing_cost_usd,  # Processing cost only
            "pod_overhead_share_usd": self.pod_overhead_share_usd,  # Overhead portion
            "gpu_hourly_rate": self.gpu_hourly_rate,
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
# User Model
# =============================================================================

class User(Base):
    """
    User account with credits system.
    Linked to Firebase Authentication via firebase_uid.
    """
    __tablename__ = "users"

    id = Column(String(128), primary_key=True)
    firebase_uid = Column(String(256), unique=True, nullable=False, index=True)
    email = Column(String(256), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    last_seen = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Credits
    total_credits = Column(Integer, default=0, nullable=False)
    bonus_granted = Column(Integer, default=0, nullable=False)
    
    # Signup bonus tracking (prevents abuse on reinstall/re-signup)
    signup_bonus_granted = Column(Boolean, default=False, nullable=False)
    
    # Daily spinner tracking
    last_daily_spin_time = Column(DateTime, nullable=True)
    
    # Rewarded ad progress (for "X ads = 1 credit" feature)
    pending_ad_watches = Column(Integer, default=0, nullable=False)
    
    # Ad-Free Purchase Status
    is_ad_free = Column(Boolean, default=False, nullable=False)
    ad_free_purchased_at = Column(DateTime, nullable=True)
    ad_free_product_id = Column(String(100), nullable=True)
    
    # VIP Subscription Status
    # VIP is time-limited (weekly/monthly) unlike ad-free which is permanent
    is_vip = Column(Boolean, default=False, nullable=False)
    vip_expires_at = Column(DateTime, nullable=True)  # When VIP expires (NULL = not VIP)
    vip_product_id = Column(String(100), nullable=True)  # Last VIP product purchased
    vip_purchased_at = Column(DateTime, nullable=True)  # Last VIP purchase time
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    credit_transactions = relationship("CreditTransaction", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")
    daily_spins = relationship("DailySpin", back_populates="user")
    rewarded_ad_events = relationship("RewardedAdEvent", back_populates="user")
    ad_free_purchases = relationship("AdFreePurchase", back_populates="user")
    purchases = relationship("Purchase", back_populates="user")
    vip_purchases = relationship("VipPurchase", back_populates="user")

    # Note: firebase_uid already has index=True on the column definition
    # No need for additional __table_args__ index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "firebase_uid": self.firebase_uid,
            "email": self.email,
            "total_credits": self.total_credits,
            "bonus_granted": self.bonus_granted,
            "signup_bonus_granted": self.signup_bonus_granted,
            "is_active": self.is_active,
            "is_ad_free": self.is_ad_free,
            "ad_free_purchased_at": self.ad_free_purchased_at.isoformat() if self.ad_free_purchased_at else None,
            "ad_free_product_id": self.ad_free_product_id,
            "is_vip": self.is_vip,
            "vip_expires_at": self.vip_expires_at.isoformat() if self.vip_expires_at else None,
            "vip_product_id": self.vip_product_id,
            "vip_purchased_at": self.vip_purchased_at.isoformat() if self.vip_purchased_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }
    
    def is_vip_active(self) -> bool:
        """Check if user has active VIP subscription (not expired)"""
        if not self.is_vip:
            return False
        if self.vip_expires_at is None:
            return False
        return self.vip_expires_at > utc_naive_now()


# =============================================================================
# Credit Transaction Model
# =============================================================================

class CreditTransaction(Base):
    """
    Credit transaction log for auditing.
    """
    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), nullable=False)
    
    delta = Column(Integer, nullable=False)  # Positive = add, negative = deduct
    reason = Column(String(256), nullable=False)
    
    job_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="credit_transactions")

    __table_args__ = (
        Index("ix_credit_user_time", "user_id", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "delta": self.delta,
            "reason": self.reason,
            "job_id": self.job_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# Subscription Model
# =============================================================================

class Subscription(Base):
    """
    User subscription for premium features.
    """
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)

    provider = Column(String(32), nullable=False)  # e.g., "stripe", "google_play", "apple"
    plan_id = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)  # e.g., "active", "cancelled", "expired"

    started_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    renewed_at = Column(DateTime, nullable=True)

    provider_ref = Column(String(256), nullable=True)  # External subscription ID
    last_event = Column(String(64), nullable=True)  # Last webhook event type

    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")

    __table_args__ = (
        Index("ix_subscriptions_user", "user_id"),
        Index("ix_subscriptions_status", "status"),
    )


# =============================================================================
# App Configuration Model
# =============================================================================

class AppConfig(Base):
    """
    Dynamic application configuration stored in database.
    Allows changing settings without redeploying.
    """
    __tablename__ = "app_config"

    key = Column(String(128), primary_key=True)
    value = Column(String(512), nullable=False)
    description = Column(String(512), nullable=True)
    updated_at = Column(DateTime, default=utc_naive_now, onupdate=utc_naive_now)


# Default configuration values
DEFAULT_APP_CONFIG = {
    "credits_per_rewarded_ad": ("1", "Credits given per rewarded ad view"),
    "daily_ad_credit_limit": ("30", "Maximum ad credits a user can earn per day"),
    "ad_cooldown_seconds": ("30", "Minimum seconds between ad reward claims"),
}


def get_app_config(db: "Session", key: str, default: str = None) -> Optional[str]:
    """Get a configuration value from the database."""
    config = db.query(AppConfig).filter(AppConfig.key == key).first()
    if config:
        return config.value
    return default


def get_app_config_int(db: "Session", key: str, default: int = 0) -> int:
    """Get a configuration value as integer."""
    value = get_app_config(db, key, str(default))
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def set_app_config(db: "Session", key: str, value: str, description: str = None):
    """Set a configuration value in the database."""
    config = db.query(AppConfig).filter(AppConfig.key == key).first()
    if config:
        config.value = value
        if description:
            config.description = description
    else:
        config = AppConfig(key=key, value=value, description=description)
        db.add(config)
    db.commit()


def init_default_config(db: "Session"):
    """Initialize default configuration values if they don't exist."""
    for key, (value, description) in DEFAULT_APP_CONFIG.items():
        existing = db.query(AppConfig).filter(AppConfig.key == key).first()
        if not existing:
            config = AppConfig(key=key, value=value, description=description)
            db.add(config)
    db.commit()


# =============================================================================
# Rewarded Ad Event Model (Anti-Abuse Tracking)
# =============================================================================

class RewardedAdEvent(Base):
    """
    Detailed tracking of rewarded ad events for anti-abuse and analytics.
    """
    __tablename__ = "rewarded_ad_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)
    
    # Reward details
    credits = Column(Integer, nullable=False)
    ad_provider = Column(String(32), nullable=False)  # "admob", "unity"
    
    # SSV verification data
    ad_unit_id = Column(String(128), nullable=True)
    transaction_id = Column(String(256), nullable=True, unique=True)  # Google's reward ID
    signature_verified = Column(Boolean, default=False, nullable=False)
    
    # Anti-abuse tracking
    ip_address = Column(String(64), nullable=True)
    device_id = Column(String(256), nullable=True)
    user_agent = Column(String(512), nullable=True)
    
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="rewarded_ad_events")

    __table_args__ = (
        Index("ix_rewarded_ad_user_time", "user_id", "created_at"),
        Index("ix_rewarded_ad_transaction", "transaction_id"),
    )


# =============================================================================
# Daily Spin Model (Audit/Analytics)
# =============================================================================

class DailySpin(Base):
    """
    Daily spin audit log for tracking and analytics.
    """
    __tablename__ = "daily_spins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)
    
    credits_won = Column(Integer, nullable=False)
    device_id = Column(String(256), nullable=True)  # For fraud detection
    
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="daily_spins")

    __table_args__ = (
        Index("ix_daily_spin_user_time", "user_id", "created_at"),
    )


# =============================================================================
# Ad-Free Purchase Model (Audit Trail)
# =============================================================================

class AdFreePurchase(Base):
    """
    Ad-free purchase audit trail.
    
    Stores verified in-app purchases for:
    - Audit trail and dispute resolution
    - Preventing replay attacks (order_id must be unique)
    - Analytics on purchase patterns
    
    SECURITY MODEL (Consumable Product):
    - Google Play: Product is consumable (anyone can buy anytime)
    - Backend: One Firebase account = one ad-free grant
    - order_id is UNIQUE to prevent replay attacks
    - Each Firebase account can only have ad-free granted once
    """
    __tablename__ = "ad_free_purchases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)
    firebase_uid = Column(String(256), nullable=False, index=True)
    
    # Purchase details
    product_id = Column(String(100), nullable=False)
    purchase_token = Column(String(500), nullable=False)  # Token from store
    platform = Column(String(20), nullable=False)  # 'android' or 'ios'
    device_id = Column(String(256), nullable=True)  # Hashed device ID for abuse prevention
    
    # Store verification data - order_id is CRITICAL for replay attack prevention
    order_id = Column(String(256), nullable=True, unique=True, index=True)  # UNIQUE: Prevent replay attacks
    store_data = Column(MutableDict.as_mutable(JSON), nullable=True)  # Raw response from store
    
    # Timestamps
    verified_at = Column(DateTime, default=utc_naive_now, nullable=False)
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="ad_free_purchases")

    __table_args__ = (
        Index("ix_ad_free_user", "user_id"),
        Index("ix_ad_free_firebase", "firebase_uid"),
        Index("ix_ad_free_order", "order_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "firebase_uid": self.firebase_uid,
            "product_id": self.product_id,
            "platform": self.platform,
            "order_id": self.order_id,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# VIP Purchase Model (Time-Limited VIP Subscriptions)
# =============================================================================

class VipPurchase(Base):
    """
    Tracks VIP subscription purchases.
    
    VIP is different from ad-free:
    - VIP is TIME-LIMITED (weekly/monthly) - expires after duration
    - VIP benefits: unlimited concurrent generations, bulk edit, free upscale
    - Multiple purchases allowed (each extends VIP period)
    - Uses consumable products that grant VIP time
    
    Security:
    - order_id is UNIQUE to prevent duplicate VIP time grants
    - Server tracks expiration, not client
    - Expiration checked on every protected operation
    
    Product IDs (suggested):
    - vip_pack_weekly: 7 days VIP
    - vip_pack_monthly: 30 days VIP
    """
    __tablename__ = "vip_purchases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)
    firebase_uid = Column(String(256), nullable=False, index=True)
    
    # Purchase details
    product_id = Column(String(100), nullable=False, index=True)
    purchase_token = Column(String(500), nullable=False)  # Token from store
    platform = Column(String(20), nullable=False)  # 'android' or 'ios'
    device_id = Column(String(256), nullable=True)  # Hashed device ID for fraud detection
    
    # Store verification data - order_id is CRITICAL for replay attack prevention
    order_id = Column(String(256), nullable=True, unique=True, index=True)  # UNIQUE: Prevent replay attacks
    store_data = Column(MutableDict.as_mutable(JSON), nullable=True)  # Raw response from store
    
    # VIP duration tracking
    duration_days = Column(Integer, nullable=False)  # How many days this purchase grants
    vip_started_at = Column(DateTime, nullable=False)  # When VIP period started
    vip_expires_at = Column(DateTime, nullable=False)  # When VIP period ends
    
    # Timestamps
    verified_at = Column(DateTime, default=utc_naive_now, nullable=False)
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="vip_purchases")

    __table_args__ = (
        Index("ix_vip_user", "user_id"),
        Index("ix_vip_firebase", "firebase_uid"),
        Index("ix_vip_order", "order_id"),
        Index("ix_vip_expires", "vip_expires_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "firebase_uid": self.firebase_uid,
            "product_id": self.product_id,
            "platform": self.platform,
            "order_id": self.order_id,
            "duration_days": self.duration_days,
            "vip_started_at": self.vip_started_at.isoformat() if self.vip_started_at else None,
            "vip_expires_at": self.vip_expires_at.isoformat() if self.vip_expires_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# Purchase Model (Consumable Purchases - Credits/Gems)
# =============================================================================

class Purchase(Base):
    """
    Tracks all consumable purchases (credit/gem packs).
    
    Security features:
    - order_id is UNIQUE to prevent duplicate credit awarding
    - purchase_token is hashed for privacy but available for verification
    - raw store_data is stored for dispute resolution
    
    Workflow:
    1. Client sends purchase_token → Server verifies with Google Play
    2. Server checks order_id doesn't exist (duplicate prevention)
    3. Server credits user atomically with purchase record creation
    4. Server consumes purchase on Google Play (allows re-purchase)
    """
    __tablename__ = "purchases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.id"), index=True, nullable=False)
    firebase_uid = Column(String(256), nullable=False, index=True)
    
    # Order identification (UNIQUE - critical for duplicate prevention)
    order_id = Column(String(256), unique=True, nullable=True, index=True)  # Google Play order ID
    purchase_token_hash = Column(String(64), nullable=False)  # SHA256 hash for privacy
    
    # Product details
    product_id = Column(String(100), nullable=False, index=True)
    platform = Column(String(20), nullable=False)  # 'android' or 'ios'
    device_id = Column(String(256), nullable=True)  # Hashed device ID for fraud detection
    
    # Credit tracking
    credits_awarded = Column(Integer, default=0, nullable=False)
    
    # Purchase state
    purchase_state = Column(String(50), default="verified", nullable=False)  # verified, refunded, disputed
    consumed = Column(Boolean, default=False, nullable=False)  # Whether consumed on Google Play
    is_test_purchase = Column(Boolean, default=False, nullable=False)  # purchaseType == 0
    
    # Store verification data
    store_data = Column(MutableDict.as_mutable(JSON), nullable=True)  # Raw response for auditing
    purchase_time_ms = Column(String(20), nullable=True)  # Timestamp from store (milliseconds)
    
    # Timestamps
    verified_at = Column(DateTime, default=utc_naive_now, nullable=False)
    created_at = Column(DateTime, default=utc_naive_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="purchases")

    __table_args__ = (
        Index("ix_purchase_user", "user_id"),
        Index("ix_purchase_firebase", "firebase_uid"),
        Index("ix_purchase_order", "order_id"),
        Index("ix_purchase_product", "product_id"),
        Index("ix_purchase_state", "purchase_state"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "firebase_uid": self.firebase_uid,
            "order_id": self.order_id,
            "product_id": self.product_id,
            "platform": self.platform,
            "credits_awarded": self.credits_awarded,
            "purchase_state": self.purchase_state,
            "is_test_purchase": self.is_test_purchase,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


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
    """Initialize database with schema and default config"""
    global _db_manager
    if database_url:
        _db_manager = DatabaseManager(database_url)
    else:
        _db_manager = get_db_manager()
    _db_manager.init_db()
    
    # Initialize default app configuration
    with _db_manager.get_session() as session:
        init_default_config(session)


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

