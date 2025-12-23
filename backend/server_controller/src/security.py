# security.py
"""
Enhanced Security Layer

Provides:
- API key verification
- Device fingerprint + HMAC-SHA256 signature validation
- IP banning (temporary & permanent) with persistence
- Rate limiting per IP
- Device registration & tracking
- Worker IP whitelist (auto-add on register)
- Security audit trail
- Brute-force detection
"""

import os
import hmac
import hashlib
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set, Tuple
from pathlib import Path
import logging

from db import (
    SessionLocal, 
    Device, 
    IPBan, 
    create_audit_log,
    get_or_create_ip_ban,
    remove_ip_ban,
    get_active_ip_bans,
    cleanup_expired_bans,
    utc_naive_now,
)

logger = logging.getLogger("security")


# =============================================================================
# Configuration
# =============================================================================

class SecurityConfig:
    """Security configuration loaded from environment"""
    
    def __init__(self):
        self.reload()
    
    def reload(self):
        """Reload configuration from environment"""
        self.shared_secret = os.getenv("CONTROLLER_SHARED_SECRET", "")
        self.api_keys = set(
            k.strip() 
            for k in os.getenv("CONTROLLER_API_KEYS", "").split(",") 
            if k.strip()
        )
        self.device_signing_secret = os.getenv(
            "DEVICE_SIGNING_SECRET",
            os.getenv("CONTROLLER_SHARED_SECRET", "")
        )
        self.device_timestamp_window = int(os.getenv("DEVICE_TIMESTAMP_WINDOW", "300"))
        self.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "40"))
        self.ban_duration = int(os.getenv("BAN_DURATION", "300"))
        self.max_body_size = int(os.getenv("MAX_BODY_SIZE", "10000000"))
        self.suspicious_score_threshold = int(os.getenv("SUSPICIOUS_SCORE_THRESHOLD", "5"))
        self.permanent_ban_on_threshold = os.getenv("PERMANENT_BAN_ON_THRESHOLD", "true").lower() == "true"
        self.trusted_worker_ip_ttl = int(os.getenv("TRUSTED_WORKER_IP_TTL", "86400"))
        self.brute_force_threshold = int(os.getenv("BRUTE_FORCE_THRESHOLD", "5"))
        self.brute_force_window = int(os.getenv("BRUTE_FORCE_WINDOW", "300"))
        self.bans_file_path = os.getenv("BANS_FILE_PATH", "banned_ips.json")


# Global config instance
security_config = SecurityConfig()


# =============================================================================
# Worker HMAC Signature
# =============================================================================

def sign_payload_bytes(payload_bytes: bytes) -> str:
    """
    Sign payload bytes with HMAC-SHA256.
    
    Args:
        payload_bytes: Raw bytes to sign
        
    Returns:
        Signature string in format "sha256=<hexdigest>"
    """
    if not security_config.shared_secret:
        raise ValueError("CONTROLLER_SHARED_SECRET not set")
    
    mac = hmac.new(
        security_config.shared_secret.encode(),
        payload_bytes,
        hashlib.sha256
    )
    return "sha256=" + mac.hexdigest()


def verify_signature(header_signature: str, payload_bytes: bytes) -> bool:
    """
    Verify HMAC-SHA256 signature from worker.
    
    Accepts signatures generated from either compact or pretty JSON encoding.
    
    Args:
        header_signature: Signature from X-Signature header
        payload_bytes: Raw request body bytes
        
    Returns:
        True if signature is valid
    """
    if not security_config.shared_secret or not header_signature:
        return False
    
    try:
        algo, hexsig = header_signature.split("=", 1)
    except (ValueError, AttributeError):
        return False
    
    if algo != "sha256":
        return False
    
    # Primary check: raw body as received
    mac = hmac.new(
        security_config.shared_secret.encode(),
        payload_bytes,
        hashlib.sha256
    )
    if hmac.compare_digest(mac.hexdigest(), hexsig):
        return True
    
    # Fallback: normalize JSON and re-sign (for workers that sign compact JSON)
    try:
        import json
        data = json.loads(payload_bytes.decode("utf-8"))
        alt_bytes = json.dumps(data).encode("utf-8")
        alt_mac = hmac.new(
            security_config.shared_secret.encode(),
            alt_bytes,
            hashlib.sha256
        )
        if hmac.compare_digest(alt_mac.hexdigest(), hexsig):
            return True
    except Exception:
        pass
    
    return False


# =============================================================================
# Device Authentication
# =============================================================================

def hash_fingerprint(raw_fingerprint: str) -> str:
    """
    Hash device fingerprint with SHA256.
    
    Args:
        raw_fingerprint: Raw fingerprint string from client
        
    Returns:
        SHA256 hex digest of the fingerprint
    """
    return hashlib.sha256(raw_fingerprint.encode("utf-8")).hexdigest()


def validate_device_signature(
    fingerprint_hash: str, 
    timestamp: str, 
    signature_hex: str
) -> bool:
    """
    Validate device signature.
    
    Signature expected: HMAC-SHA256(DEVICE_SIGNING_SECRET, f"{fingerprint_hash}:{timestamp}")
    
    Args:
        fingerprint_hash: SHA256 hash of device fingerprint
        timestamp: Unix timestamp as string
        signature_hex: Hex signature from client
        
    Returns:
        True if signature is valid and timestamp is within window
    """
    if not security_config.device_signing_secret:
        return False
    
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False
    
    now = int(time.time())
    if abs(now - ts) > security_config.device_timestamp_window:
        logger.debug(f"Device timestamp too old: {now - ts}s")
        return False
    
    msg = f"{fingerprint_hash}:{timestamp}".encode("utf-8")
    mac = hmac.new(
        security_config.device_signing_secret.encode("utf-8"),
        msg,
        hashlib.sha256
    )
    expected = mac.hexdigest()
    
    return hmac.compare_digest(expected, signature_hex)


# =============================================================================
# API Key Verification
# =============================================================================

def verify_api_key(auth_header: Optional[str]) -> bool:
    """
    Verify API key from Authorization header.
    
    Args:
        auth_header: Authorization header value (Bearer <key>)
        
    Returns:
        True if API key is valid
    """
    if not auth_header:
        return False
    
    parts = auth_header.split()
    if len(parts) != 2:
        return False
    
    scheme, key = parts
    if scheme.lower() != "bearer":
        return False
    
    return key in security_config.api_keys


# =============================================================================
# Device Management
# =============================================================================

def find_device_by_fingerprint(fingerprint_hash: str) -> Optional[Device]:
    """Find device by fingerprint hash"""
    db = SessionLocal()
    try:
        return (
            db.query(Device)
            .filter(Device.fingerprint_hash == fingerprint_hash)
            .order_by(Device.last_seen.desc())
            .first()
        )
    finally:
        db.close()


def find_device_by_uuid(device_uuid: str) -> Optional[Device]:
    """Find device by UUID"""
    db = SessionLocal()
    try:
        return (
            db.query(Device)
            .filter(Device.device_uuid == device_uuid)
            .order_by(Device.last_seen.desc())
            .first()
        )
    finally:
        db.close()


def register_or_update_device(
    fingerprint_hash: str,
    device_uuid: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Register a new device or update existing one.
    
    Args:
        fingerprint_hash: SHA256 hash of device fingerprint
        device_uuid: Optional device UUID
        meta: Optional metadata to store
        
    Returns:
        Dict with device_uuid and fingerprint_hash
    """
    db = SessionLocal()
    try:
        now = utc_naive_now()
        
        # Try to find by fingerprint first
        dev = (
            db.query(Device)
            .filter(Device.fingerprint_hash == fingerprint_hash)
            .order_by(Device.last_seen.desc())
            .first()
        )
        
        if dev:
            # Update existing device
            dev.last_seen = now
            dev.request_count = (dev.request_count or 0) + 1
            if device_uuid and not dev.device_uuid:
                dev.device_uuid = device_uuid
            if meta:
                dev.meta = {**(dev.meta or {}), **meta}
            db.commit()
            return {
                "device_uuid": dev.device_uuid,
                "fingerprint_hash": dev.fingerprint_hash
            }
        
        # Try to find by UUID if provided
        if device_uuid:
            existing = (
                db.query(Device)
                .filter(Device.device_uuid == device_uuid)
                .order_by(Device.last_seen.desc())
                .first()
            )
            if existing:
                existing.fingerprint_hash = fingerprint_hash
                existing.last_seen = now
                existing.request_count = (existing.request_count or 0) + 1
                if meta:
                    existing.meta = {**(existing.meta or {}), **meta}
                db.commit()
                return {
                    "device_uuid": existing.device_uuid,
                    "fingerprint_hash": existing.fingerprint_hash
                }
        
        # Create new device
        new_device = Device(
            device_uuid=device_uuid,
            fingerprint_hash=fingerprint_hash,
            created_at=now,
            last_seen=now,
            request_count=1,
            meta=meta
        )
        db.add(new_device)
        db.commit()
        
        return {
            "device_uuid": new_device.device_uuid,
            "fingerprint_hash": new_device.fingerprint_hash
        }
    finally:
        db.close()


def is_device_banned(fingerprint_hash: str) -> bool:
    """Check if a device is banned"""
    db = SessionLocal()
    try:
        dev = (
            db.query(Device)
            .filter(Device.fingerprint_hash == fingerprint_hash)
            .first()
        )
        return dev is not None and dev.banned
    finally:
        db.close()


def ban_device(fingerprint_hash: str, reason: str) -> bool:
    """Ban a device by fingerprint hash"""
    db = SessionLocal()
    try:
        dev = (
            db.query(Device)
            .filter(Device.fingerprint_hash == fingerprint_hash)
            .first()
        )
        if dev:
            dev.banned = True
            dev.ban_reason = reason
            db.commit()
            return True
        return False
    finally:
        db.close()


# =============================================================================
# IP Ban Management
# =============================================================================

class IPBanManager:
    """
    Manages IP bans with in-memory caching and database persistence.
    """
    
    def __init__(self):
        # In-memory cache: ip -> {expiry, permanent, reason, created_at}
        self._bans: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._bans_file = Path(security_config.bans_file_path)
        
        # Rate limiting: ip -> [timestamps]
        self._rate_limiter: Dict[str, list] = {}
        
        # Suspicion scores: ip -> score
        self._suspicion_scores: Dict[str, int] = {}
        
        # Trusted worker IPs: ip -> expiry_timestamp
        self._trusted_worker_ips: Dict[str, float] = {}
        
        # Failed auth attempts for brute-force detection: ip -> [timestamps]
        self._failed_auth_attempts: Dict[str, list] = {}
        
        # Static trusted IPs
        self._static_trusted_ips: Set[str] = set()
    
    async def load_from_db(self):
        """Load bans from database on startup"""
        db = SessionLocal()
        try:
            bans = get_active_ip_bans(db)
            async with self._lock:
                for ban in bans:
                    self._bans[ban.ip] = {
                        "expiry": ban.expires_at.timestamp() if ban.expires_at else None,
                        "permanent": ban.permanent,
                        "reason": ban.reason,
                        "created_at": ban.created_at.timestamp() if ban.created_at else time.time(),
                    }
            logger.info(f"Loaded {len(bans)} IP bans from database")
        finally:
            db.close()
    
    async def load_from_file(self):
        """Load bans from JSON file (legacy support)"""
        async with self._lock:
            if not self._bans_file.exists():
                return
            try:
                with open(self._bans_file, "r") as f:
                    loaded = json.load(f)
                for ip, info in loaded.items():
                    if ip not in self._bans:
                        self._bans[ip] = info
                logger.info(f"Loaded {len(loaded)} IP bans from file")
            except Exception as e:
                logger.error(f"Failed to load bans from file: {e}")
    
    async def save_to_file(self):
        """Save bans to JSON file"""
        async with self._lock:
            try:
                tmp_path = self._bans_file.with_suffix(".tmp")
                with open(tmp_path, "w") as f:
                    json.dump(self._bans, f)
                tmp_path.replace(self._bans_file)
            except Exception as e:
                logger.error(f"Failed to save bans to file: {e}")
    
    def is_banned(self, ip: str) -> bool:
        """Check if an IP is banned"""
        # Check static trusted IPs
        if ip in self._static_trusted_ips:
            return False
        
        info = self._bans.get(ip)
        if not info:
            return False
        
        if info.get("permanent"):
            return True
        
        expiry = info.get("expiry")
        if expiry and expiry > time.time():
            return True
        
        # Expired - clean up
        self._bans.pop(ip, None)
        return False
    
    async def ban_ip(
        self,
        ip: str,
        reason: str,
        duration: Optional[int] = None,
        permanent: bool = False,
        persist: bool = True,
    ):
        """
        Ban an IP address.
        
        Args:
            ip: IP address to ban
            reason: Reason for ban
            duration: Ban duration in seconds (if not permanent)
            permanent: Whether ban is permanent
            persist: Whether to persist to database
        """
        now = time.time()
        
        async with self._lock:
            if permanent:
                self._bans[ip] = {
                    "expiry": None,
                    "permanent": True,
                    "reason": reason,
                    "created_at": now,
                }
                logger.warning(f"IP banned (permanent): {ip} - {reason}")
            else:
                expiry = now + (duration or security_config.ban_duration)
                self._bans[ip] = {
                    "expiry": expiry,
                    "permanent": False,
                    "reason": reason,
                    "created_at": now,
                }
                logger.warning(f"IP banned until {expiry}: {ip} - {reason}")
        
        if persist:
            # Persist to database
            db = SessionLocal()
            try:
                expires_at = None
                if not permanent and duration:
                    expires_at = datetime.fromtimestamp(now + duration)
                
                get_or_create_ip_ban(
                    db,
                    ip=ip,
                    reason=reason,
                    permanent=permanent,
                    expires_at=expires_at,
                    created_by="system"
                )
                
                # Create audit log
                create_audit_log(
                    db,
                    event_type="ip_banned",
                    message=f"IP banned: {ip}",
                    target=ip,
                    target_type="ip",
                    severity="warning",
                    details={
                        "reason": reason,
                        "permanent": permanent,
                        "duration": duration,
                    }
                )
                
                db.commit()
            finally:
                db.close()
            
            await self.save_to_file()
    
    async def unban_ip(self, ip: str) -> bool:
        """Remove an IP ban"""
        async with self._lock:
            if ip not in self._bans:
                return False
            self._bans.pop(ip, None)
        
        # Remove from database
        db = SessionLocal()
        try:
            removed = remove_ip_ban(db, ip)
            if removed:
                create_audit_log(
                    db,
                    event_type="ip_unbanned",
                    message=f"IP unbanned: {ip}",
                    target=ip,
                    target_type="ip",
                )
                db.commit()
                logger.info(f"IP unbanned: {ip}")
            return removed
        finally:
            db.close()
    
    def check_rate_limit(self, ip: str) -> bool:
        """
        Check if IP is within rate limit.
        
        Returns True if request is allowed, False if rate limited.
        """
        now = time.time()
        timestamps = self._rate_limiter.get(ip, [])
        
        # Remove timestamps older than 1 minute
        timestamps = [t for t in timestamps if now - t < 60]
        
        if len(timestamps) >= security_config.rate_limit_per_minute:
            self._rate_limiter[ip] = timestamps
            return False
        
        timestamps.append(now)
        self._rate_limiter[ip] = timestamps
        return True
    
    async def increase_suspicion(self, ip: str, amount: int, reason: str):
        """
        Increase suspicion score for an IP.
        
        May trigger a ban if threshold is exceeded.
        """
        score = self._suspicion_scores.get(ip, 0) + amount
        self._suspicion_scores[ip] = score
        
        logger.info(
            f"Suspicion {score}/{security_config.suspicious_score_threshold} "
            f"for {ip} - {reason}"
        )
        
        if score >= security_config.suspicious_score_threshold:
            await self.ban_ip(
                ip,
                f"suspicious behavior (score={score})",
                permanent=security_config.permanent_ban_on_threshold
            )
    
    def record_failed_auth(self, ip: str):
        """Record a failed authentication attempt for brute-force detection"""
        now = time.time()
        attempts = self._failed_auth_attempts.get(ip, [])
        
        # Remove old attempts
        window = security_config.brute_force_window
        attempts = [t for t in attempts if now - t < window]
        attempts.append(now)
        self._failed_auth_attempts[ip] = attempts
        
        return len(attempts)
    
    async def check_brute_force(self, ip: str) -> bool:
        """
        Check for brute-force attack pattern.
        
        Returns True if brute-force detected (and bans the IP).
        """
        attempts = self._failed_auth_attempts.get(ip, [])
        
        if len(attempts) >= security_config.brute_force_threshold:
            await self.ban_ip(
                ip,
                f"brute-force attack detected ({len(attempts)} failed attempts)",
                permanent=True
            )
            return True
        
        return False
    
    def add_trusted_worker_ip(self, ip: str, ttl: Optional[int] = None):
        """
        Add an IP to trusted worker list.
        
        Args:
            ip: Worker IP address
            ttl: Time-to-live in seconds (default from config)
        """
        ttl = ttl or security_config.trusted_worker_ip_ttl
        self._trusted_worker_ips[ip] = time.time() + ttl
        logger.info(f"Added trusted worker IP: {ip} (TTL: {ttl}s)")
    
    def is_trusted_worker_ip(self, ip: str) -> bool:
        """Check if IP is a trusted worker"""
        expiry = self._trusted_worker_ips.get(ip)
        if not expiry:
            return False
        
        if expiry > time.time():
            return True
        
        # Expired
        self._trusted_worker_ips.pop(ip, None)
        return False
    
    def add_static_trusted_ip(self, ip: str):
        """Add a permanently trusted IP"""
        self._static_trusted_ips.add(ip)
    
    def remove_static_trusted_ip(self, ip: str):
        """Remove a permanently trusted IP"""
        self._static_trusted_ips.discard(ip)
    
    async def cleanup_expired(self):
        """Clean up expired bans and stale data"""
        now = time.time()
        
        # Clean up expired bans
        async with self._lock:
            expired = [
                ip for ip, info in self._bans.items()
                if not info.get("permanent") 
                and info.get("expiry") 
                and info["expiry"] <= now
            ]
            for ip in expired:
                self._bans.pop(ip, None)
        
        # Clean up expired trusted worker IPs
        expired_workers = [
            ip for ip, expiry in self._trusted_worker_ips.items()
            if expiry <= now
        ]
        for ip in expired_workers:
            self._trusted_worker_ips.pop(ip, None)
        
        # Clean up old rate limit data
        old_ips = [
            ip for ip, timestamps in self._rate_limiter.items()
            if all(now - t > 60 for t in timestamps)
        ]
        for ip in old_ips:
            self._rate_limiter.pop(ip, None)
        
        # Clean up old suspicion scores (reset after 1 hour)
        old_suspicion = [
            ip for ip, score in self._suspicion_scores.items()
            if score < security_config.suspicious_score_threshold
        ]
        for ip in old_suspicion:
            self._suspicion_scores.pop(ip, None)
        
        # Clean up database
        db = SessionLocal()
        try:
            cleaned = cleanup_expired_bans(db)
            if cleaned:
                db.commit()
                logger.info(f"Cleaned up {cleaned} expired bans from database")
        finally:
            db.close()


# Global IP ban manager instance
ip_ban_manager = IPBanManager()


# =============================================================================
# Authentication Helpers
# =============================================================================

async def authenticate_request(
    authorization: Optional[str] = None,
    x_fingerprint: Optional[str] = None,
    x_timestamp: Optional[str] = None,
    x_signature: Optional[str] = None,
    x_device_uuid: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Authenticate a request using either API key or device signature.
    
    Args:
        authorization: Authorization header (Bearer <key>)
        x_fingerprint: Device fingerprint
        x_timestamp: Request timestamp
        x_signature: Device signature
        x_device_uuid: Device UUID
        
    Returns:
        Tuple of (success, auth_info, error_message)
    """
    # Try API key first
    if authorization and verify_api_key(authorization):
        return True, {"type": "api_key"}, None
    
    # Try device signature
    if not x_fingerprint or not x_timestamp or not x_signature:
        return False, None, "Missing auth headers"
    
    # Normalize fingerprint hash
    if len(x_fingerprint) == 64 and all(c in "0123456789abcdef" for c in x_fingerprint.lower()):
        fingerprint_hash = x_fingerprint.lower()
    else:
        fingerprint_hash = hash_fingerprint(x_fingerprint)
    
    # Check if device is banned
    if is_device_banned(fingerprint_hash):
        return False, None, "Device is banned"
    
    # Validate signature
    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        return False, None, "Invalid device signature or timestamp"
    
    # Register/update device
    device_info = register_or_update_device(
        fingerprint_hash,
        x_device_uuid,
        meta={"last_auth": utc_naive_now().isoformat()}
    )
    
    return True, {
        "type": "device",
        "fingerprint_hash": fingerprint_hash,
        "device_uuid": device_info.get("device_uuid"),
    }, None


# =============================================================================
# Convenience Functions
# =============================================================================

def time_datetime_now() -> datetime:
    """Return current UTC datetime"""
    return utc_naive_now()

