#!/usr/bin/env python3
"""Kie.ai job client: credit check, file upload, createTask, HMAC callback, recordInfo fallback."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger("kie_client")

KIE_API = os.getenv("KIE_API_BASE", "https://api.kie.ai").rstrip("/")
KIE_UPLOAD = os.getenv("KIE_UPLOAD_BASE", "https://kieai.redpandaai.co").rstrip("/")
KIE_MIN_CREDITS = float(os.getenv("KIE_MIN_CREDITS", "1"))


def _api_key() -> str:
    return (os.getenv("KIE_API_KEY") or "").strip()


def _hmac_key() -> str:
    return (os.getenv("KIE_WEBHOOK_HMAC_KEY") or "").strip()


def _headers() -> Dict[str, str]:
    key = _api_key()
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def verify_webhook_signature(
    task_id: str,
    timestamp: str,
    signature_b64: str,
) -> bool:
    key = _hmac_key()
    if not key or not task_id or not timestamp or not signature_b64:
        return False
    message = f"{task_id}.{timestamp}".encode("utf-8")
    digest = hmac.new(key.encode("utf-8"), message, hashlib.sha256).digest()
    expected = base64.b64encode(digest)
    try:
        provided = signature_b64.strip().encode("utf-8")
        return hmac.compare_digest(expected, provided)
    except Exception:
        return False


def get_credit_balance() -> Optional[float]:
    try:
        resp = requests.get(
            f"{KIE_API}/api/v1/chat/credit",
            headers=_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        # Observed shapes: {data: {credit}} or {data: number}
        payload = data.get("data", data)
        if isinstance(payload, (int, float)):
            return float(payload)
        if isinstance(payload, dict):
            for k in ("credit", "credits", "balance", "remainCredit"):
                if payload.get(k) is not None:
                    return float(payload[k])
        return None
    except Exception:
        logger.exception("Kie credit check failed")
        return None


def ensure_credits() -> Tuple[bool, str]:
    if not _api_key():
        return False, "KIE_API_KEY is not configured"
    bal = get_credit_balance()
    if bal is None:
        return False, "Could not read Kie credit balance"
    if bal < KIE_MIN_CREDITS:
        return False, f"Kie balance too low ({bal})"
    return True, ""


def kie_hosted_file_url(body: Any) -> Optional[str]:
    """Kie file upload returns downloadUrl. Older responses used fileUrl."""
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if code not in (None, 200, "200"):
        return None
    if body.get("success") is False:
        return None
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    for source in (data, body):
        for key in ("fileUrl", "downloadUrl", "url"):
            value = source.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
    return None


def kie_upload_is_image(body: Any) -> bool:
    """A .bin object stored as octet-stream is not an image reference."""
    if not isinstance(body, dict):
        return False
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    mime = str(data.get("mimeType") or "").lower()
    name = str(data.get("fileName") or "").lower()
    if mime.startswith("image/"):
        return True
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return True
    if not mime and not name.endswith(".bin"):
        return True
    return False


def sniff_image(data: bytes) -> Optional[Tuple[str, str]]:
    """Return a unique image filename and content type from the file header."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        ext, mime = "png", "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        ext, mime = "jpg", "image/jpeg"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        ext, mime = "webp", "image/webp"
    elif data.startswith(b"GIF8"):
        ext, mime = "gif", "image/gif"
    else:
        return None
    return f"{uuid.uuid4().hex}.{ext}", mime


def is_binary_object_url(value: str) -> bool:
    """True when a URL points at a generic .bin object rather than an image."""
    if not isinstance(value, str) or not value.startswith("http"):
        return False
    return urlparse(value).path.lower().endswith(".bin")


def upload_file_from_url(file_url: str, file_name: Optional[str] = None) -> str:
    """Download the source and upload the real image bytes to Kie.

    Storage keys are ``.bin`` with ``application/octet-stream``. Kie's
    file-url-upload copies that name and type, so every image model receives
    a generic file. The bytes decide the filename and content type instead.
    """
    del file_name  # The file header decides the name. A .bin source name is wrong.
    raw = requests.get(file_url, timeout=60)
    raw.raise_for_status()
    return upload_file_bytes(raw.content, "input.bin")


def upload_file_bytes(data: bytes, file_name: str, content_type: Optional[str] = None) -> str:
    del file_name, content_type
    sniffed = sniff_image(data)
    if sniffed is None:
        raise RuntimeError("Kie image upload is not a png, jpeg, webp, or gif")
    file_name, content_type = sniffed
    files = {"file": (file_name, data, content_type)}
    form = {"uploadPath": "images", "fileName": file_name}
    headers = {"Authorization": f"Bearer {_api_key()}"}
    resp = requests.post(
        f"{KIE_UPLOAD}/api/file-stream-upload",
        headers=headers,
        files=files,
        data=form,
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    kie_url = kie_hosted_file_url(body)
    if not kie_url or not kie_upload_is_image(body):
        raise RuntimeError(f"Kie stream upload did not return an image: {body}")
    return kie_url


def prepare_recraft_image(source_url: str) -> str:
    """Rehost a remove-background or upscale source. No local image library."""
    return upload_file_from_url(source_url)


def create_task(model: str, input_payload: Dict[str, Any], callback_url: str) -> str:
    body = {
        "model": model,
        "callBackUrl": callback_url,
        "input": input_payload,
    }
    resp = requests.post(
        f"{KIE_API}/api/v1/jobs/createTask",
        headers=_headers(),
        json=body,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Kie createTask HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    payload = data.get("data") or data
    task_id = payload.get("taskId") or payload.get("task_id")
    if not task_id:
        raise RuntimeError(f"Kie createTask missing taskId: {data}")
    return str(task_id)


def record_info(task_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{KIE_API}/api/v1/jobs/recordInfo",
        headers=_headers(),
        params={"taskId": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def parse_download_url_response(body: Dict[str, Any]) -> str:
    """Read POST /api/v1/common/download-url. `data` is the URL string."""
    if not isinstance(body, dict):
        raise RuntimeError(f"Kie download-url returned {type(body).__name__}")
    code = body.get("code")
    if code not in (None, 200, "200"):
        raise RuntimeError(f"Kie download-url {code}: {body.get('msg') or body}")
    data = body.get("data")
    if isinstance(data, str) and data.strip():
        return data.strip()
    if isinstance(data, dict):
        url = data.get("downloadUrl") or data.get("url") or data.get("download_url")
        if url:
            return str(url)
    raise RuntimeError(f"Kie download-url missing data: {body}")


def download_url_for(file_url: str) -> str:
    """Mint a 20-minute download link for a KIE task result URL."""
    resp = requests.post(
        f"{KIE_API}/api/v1/common/download-url",
        headers=_headers(),
        json={"url": file_url},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_download_url_response(resp.json())


def fetch_result_bytes(result_url: str) -> bytes:
    try:
        resp = requests.get(result_url, timeout=120)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        logger.warning("Direct GET of Kie result failed, trying download-url")
    alt = download_url_for(result_url)
    resp = requests.get(alt, timeout=120)
    resp.raise_for_status()
    return resp.content


def parse_result_urls(payload: Dict[str, Any]) -> List[str]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    result_json = data.get("resultJson") if isinstance(data, dict) else None
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except json.JSONDecodeError:
            result_json = {}
    if isinstance(result_json, dict):
        urls = result_json.get("resultUrls") or result_json.get("result_urls") or []
        if isinstance(urls, list):
            return [str(u) for u in urls if u]
        if isinstance(urls, str):
            return [urls]
    urls = data.get("resultUrls") if isinstance(data, dict) else None
    if isinstance(urls, list):
        return [str(u) for u in urls if u]
    return []


def coerce_aspect_ratio(value: Any, allowed: List[str]) -> str:
    """Map an app ratio onto the ratios that model actually accepts."""
    raw = str(value or "").strip()
    if raw in allowed:
        return raw
    snapped = {
        "4:3": "3:2",
        "3:4": "2:3",
        "21:9": "16:9",
        "9:21": "9:16",
    }.get(raw)
    if snapped and snapped in allowed:
        return snapped
    if "auto" in allowed:
        return "auto"
    return allowed[0] if allowed else raw


def kie_failure_message(payload: Dict[str, Any]) -> str:
    """Prefer data.failMsg. The top-level msg is only 'Playground task failed.'"""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    fail_msg = str(data.get("failMsg") or data.get("fail_msg") or "").strip()
    fail_code = str(data.get("failCode") or data.get("fail_code") or "").strip()
    if fail_msg and fail_msg != "<error detail>":
        return f"{fail_msg} ({fail_code})" if fail_code else fail_msg
    generic = str(payload.get("msg") or payload.get("message") or "").strip()
    if generic and generic != "Playground task failed.":
        return generic
    if fail_code:
        return f"Kie task failed ({fail_code})"
    return generic or "Kie task failed"


def extract_task_id(payload: Dict[str, Any]) -> Optional[str]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        return None
    tid = data.get("taskId") or data.get("task_id") or payload.get("taskId") or payload.get("task_id")
    return str(tid) if tid else None


def is_success_code(payload: Dict[str, Any]) -> bool:
    code = payload.get("code")
    if code in (200, "200", 0, "0"):
        return True
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    state = str((data or {}).get("state") or (data or {}).get("status") or "").lower()
    return state in ("success", "succeed", "completed", "done")
