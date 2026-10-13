"""
Encrypted JSON payloads between server_controller and GPU workers.

When ``WORKER_TRANSPORT_CRYPTO=1`` is set **and** a Fernet key is available,
request bodies use: ``{"enc": "<Fernet token>"}`` and ``X-Signature: sha256=...`` is
computed over those exact bytes (same as plaintext mode).

Env:
  WORKER_TRANSPORT_CRYPTO — must be ``1``/``true``/``yes`` to enable; ``0`` disables.
  WORKER_PAYLOAD_ENCRYPTION_KEY — optional Fernet key (``Fernet.generate_key()``).
  If unset, a Fernet key is derived from ``CONTROLLER_SHARED_SECRET`` (must match workers).

Workers use the same variables (``CONTROLLER_SHARED_SECRET`` / ``WORKER_PAYLOAD_ENCRYPTION_KEY``).
"""

from __future__ import annotations

import base64
import hashlib
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


def transport_crypto_explicitly_on() -> bool:
    v = os.getenv("WORKER_TRANSPORT_CRYPTO", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def transport_crypto_disabled() -> bool:
    return _env_falsy("WORKER_TRANSPORT_CRYPTO")


def _fernet_from_explicit_key(raw: str) -> Optional["Fernet"]:
    if Fernet is None:
        logger.error("cryptography package missing; cannot enable worker transport crypto")
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
    if get_fernet() is None:
        return False
    return True


def encode_worker_request_body(payload: Dict[str, Any]) -> bytes:
    """Serialize *payload* for HTTP body (encrypted envelope when crypto is on)."""
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    f = get_fernet()
    if f is None or transport_crypto_disabled():
        return inner
    token = f.encrypt(inner)
    outer = {"enc": token.decode("ascii")}
    return json.dumps(outer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_worker_request_body(body: bytes) -> Dict[str, Any]:
    """Parse worker→controller HTTP body (decrypt if envelope present and crypto on)."""
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("worker payload must be a JSON object")
    enc = data.get("enc")
    if crypto_enabled():
        if not isinstance(enc, str) or not enc.strip():
            raise ValueError("encrypted worker transport required (missing enc)")
        f = get_fernet()
        assert f is not None
        try:
            inner = f.decrypt(enc.strip().encode("ascii"))
        except InvalidToken as e:
            raise ValueError("invalid or tampered encrypted payload") from e
        out = json.loads(inner.decode("utf-8"))
        if not isinstance(out, dict):
            raise ValueError("decrypted worker payload must be a JSON object")
        return out
    return data
