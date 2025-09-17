# security.py
import os
import hmac
import hashlib
import time
from typing import Optional
from sqlalchemy.orm import Session
from db import SessionLocal, Device


# env secrets
SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
API_KEYS = set([k.strip() for k in os.getenv("CONTROLLER_API_KEYS", "").split(",") if k.strip()])
DEVICE_SIGNING_SECRET = os.getenv("DEVICE_SIGNING_SECRET", os.getenv("CONTROLLER_SHARED_SECRET", ""))  # use separate if set
DEVICE_TIMESTAMP_WINDOW = int(os.getenv("DEVICE_TIMESTAMP_WINDOW", "300"))  # seconds, default 5 minutes

# Worker HMAC
def sign_payload_bytes(payload_bytes: bytes) -> str:
    if not SHARED_SECRET:
        raise ValueError("CONTROLLER_SHARED_SECRET not set")
    mac = hmac.new(SHARED_SECRET.encode(), payload_bytes, hashlib.sha256)
    return "sha256=" + mac.hexdigest()

def verify_signature(header_signature: str, payload_bytes: bytes) -> bool:
    if not SHARED_SECRET:
        return False
    if not header_signature:
        return False
    try:
        algo, hexsig = header_signature.split("=", 1)
    except Exception:
        return False
    if algo != "sha256":
        return False
    mac = hmac.new(SHARED_SECRET.encode(), payload_bytes, hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), hexsig)

# --------------------------------------------------
# Device / API key auth helpers used by main.py
# --------------------------------------------------

def hash_fingerprint(raw_fingerprint: str) -> str:
    """Return SHA256 hex of the raw_fingerprint string."""
    return hashlib.sha256(raw_fingerprint.encode("utf-8")).hexdigest()

def verify_api_key(auth_header: Optional[str]) -> bool:
    """Expect Authorization: Bearer <key>"""
    if not auth_header:
        return False
    parts = auth_header.split()
    if len(parts) != 2:
        return False
    scheme, key = parts
    if scheme.lower() != "bearer":
        return False
    return key in API_KEYS

def find_device_by_fingerprint(session: Session, fingerprint_hash: str) -> Optional[Device]:
    return session.query(Device).filter(Device.fingerprint_hash == fingerprint_hash).order_by(Device.last_seen.desc()).first()

def find_device_by_uuid(session: Session, device_uuid: str) -> Optional[Device]:
    return session.query(Device).filter(Device.device_uuid == device_uuid).order_by(Device.last_seen.desc()).first()

def register_or_update_device(fingerprint_hash: str, device_uuid: Optional[str], meta: Optional[dict] = None) -> dict:
    """
    Store device record. If device_uuid provided and exists, link; otherwise if fingerprint exists, update last_seen and
    optionally set device_uuid.
    Returns a dict with canonical device_uuid and fingerprint_hash.
    """
    db = SessionLocal()
    try:
        dev = find_device_by_fingerprint(db, fingerprint_hash)
        if dev:
            # update last_seen and optionally device_uuid
            dev.last_seen = time_datetime_now()
            if device_uuid and not dev.device_uuid:
                dev.device_uuid = device_uuid
            if meta:
                dev.meta = meta
            db.commit()
            return {"device_uuid": dev.device_uuid, "fingerprint_hash": dev.fingerprint_hash}
        # fingerprint not found -> if device_uuid exists, try match by uuid
        if device_uuid:
            existing = find_device_by_uuid(db, device_uuid)
            if existing:
                existing.fingerprint_hash = fingerprint_hash
                existing.last_seen = time_datetime_now()
                if meta:
                    existing.meta = meta
                db.commit()
                return {"device_uuid": existing.device_uuid, "fingerprint_hash": existing.fingerprint_hash}
        # create new
        new = Device(device_uuid=device_uuid, fingerprint_hash=fingerprint_hash, last_seen=time_datetime_now(), meta=meta)
        db.add(new)
        db.commit()
        return {"device_uuid": new.device_uuid, "fingerprint_hash": new.fingerprint_hash}
    finally:
        db.close()

def time_datetime_now():
    from datetime import datetime
    return datetime.utcnow()

def validate_device_signature(fingerprint_hash: str, timestamp: str, signature_hex: str) -> bool:
    """
    Signature expected: HMAC-SHA256( DEVICE_SIGNING_SECRET, f"{fingerprint_hash}:{timestamp}" )
    timestamp is unix seconds
    """
    if not DEVICE_SIGNING_SECRET:
        return False
    try:
        ts = int(timestamp)
    except Exception:
        return False
    now = int(time.time())
    if abs(now - ts) > DEVICE_TIMESTAMP_WINDOW:
        return False
    msg = f"{fingerprint_hash}:{timestamp}".encode("utf-8")
    mac = hmac.new(DEVICE_SIGNING_SECRET.encode("utf-8"), msg, hashlib.sha256)
    expected = mac.hexdigest()
    return hmac.compare_digest(expected, signature_hex)