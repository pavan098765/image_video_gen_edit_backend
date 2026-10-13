"""
Encrypted JSON payloads between GPU workers and server_controller.

Must stay in sync with ``server_controller/src/worker_transport_crypto.py`` (encode/decode).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("worker_transport_crypto")

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore

_DERIVE_PREFIX = b"qp-worker-payload-v1\x00"


def _env_falsy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("0", "false", "no", "off")


def transport_crypto_disabled() -> bool:
    return _env_falsy("WORKER_TRANSPORT_CRYPTO")


def transport_crypto_explicitly_on() -> bool:
    v = os.getenv("WORKER_TRANSPORT_CRYPTO", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _fernet_from_explicit_key(raw: str) -> Optional["Fernet"]:
    if Fernet is None:
        logger.error("cryptography package missing; install cryptography>=41")
        return None
    key = raw.strip().encode("utf-8")
    try:
        return Fernet(key)
    except Exception as e:
        logger.error("invalid WORKER_PAYLOAD_ENCRYPTION_KEY: %s", e)
        return None


def _fernet_from_shared_secret(secret: str) -> Optional["Fernet"]:
    if Fernet is None:
        return None
    digest = hashlib.sha256(_DERIVE_PREFIX + secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def get_fernet() -> Optional["Fernet"]:
    ex = os.getenv("WORKER_PAYLOAD_ENCRYPTION_KEY", "").strip()
    if ex:
        return _fernet_from_explicit_key(ex)
    sec = os.getenv("CONTROLLER_SHARED_SECRET", "").strip()
    if not sec:
        return None
    return _fernet_from_shared_secret(sec)


def crypto_enabled() -> bool:
    if transport_crypto_disabled():
        return False
    if not transport_crypto_explicitly_on():
        return False
    return get_fernet() is not None


def encode_worker_request_body(payload: Dict[str, Any]) -> bytes:
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    f = get_fernet()
    if f is None or transport_crypto_disabled():
        return inner
    token = f.encrypt(inner)
    outer = {"enc": token.decode("ascii")}
    return json.dumps(outer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_worker_request_body(body: bytes) -> Dict[str, Any]:
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    enc = data.get("enc")
    if crypto_enabled():
        if not isinstance(enc, str) or not enc.strip():
            raise ValueError("encrypted transport required (missing enc)")
        f = get_fernet()
        assert f is not None
        try:
            inner = f.decrypt(enc.strip().encode("ascii"))
        except InvalidToken as e:
            raise ValueError("invalid encrypted payload") from e
        out = json.loads(inner.decode("utf-8"))
        if not isinstance(out, dict):
            raise ValueError("decrypted payload must be a JSON object")
        return out
    return data


def verify_controller_signature(signature_header: Optional[str], body: bytes) -> bool:
    """Verify ``X-Signature`` from controller for POST /run, /run_bulk, /download_priority."""
    secret = os.getenv("CONTROLLER_SHARED_SECRET", "").strip()
    if not secret:
        return True
    if not signature_header:
        return False
    try:
        algo, hexsig = signature_header.split("=", 1)
    except ValueError:
        return False
    if algo != "sha256":
        return False
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    if hmac.compare_digest(mac.hexdigest(), hexsig):
        return True
    try:
        data = json.loads(body.decode("utf-8"))
        alt = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        mac2 = hmac.new(secret.encode("utf-8"), alt, hashlib.sha256)
        return hmac.compare_digest(mac2.hexdigest(), hexsig)
    except Exception:
        return False
