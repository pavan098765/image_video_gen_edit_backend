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
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger("kie_client")

KIE_API = os.getenv("KIE_API_BASE", "https://api.kie.ai").rstrip("/")
KIE_UPLOAD = os.getenv("KIE_UPLOAD_BASE", "https://kieai.redpandaai.co").rstrip("/")
KIE_MIN_CREDITS = float(os.getenv("KIE_MIN_CREDITS", "1"))

RECRAFT_MAX_BYTES = 5 * 1024 * 1024
RECRAFT_MIN_PX = 256
RECRAFT_MAX_PX = 4096


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


def _filename_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path) or fallback
    if "." not in name:
        name = fallback
    return name[:180]


def upload_file_from_url(file_url: str, file_name: Optional[str] = None) -> str:
    """Upload via Kie file-url-upload. Returns Kie fileUrl. Falls back to stream upload."""
    name = file_name or _filename_from_url(file_url, "input.png")
    payload = {"fileUrl": file_url, "uploadPath": "images", "fileName": name}
    try:
        resp = requests.post(
            f"{KIE_UPLOAD}/api/file-url-upload",
            headers=_headers(),
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            body = resp.json()
            kie_url = (body.get("data") or {}).get("fileUrl") or body.get("fileUrl")
            if kie_url:
                return str(kie_url)
        logger.warning("Kie file-url-upload failed HTTP %s: %s", resp.status_code, resp.text[:400])
    except Exception:
        logger.exception("Kie file-url-upload error")

    raw = requests.get(file_url, timeout=60)
    raw.raise_for_status()
    return upload_file_bytes(raw.content, name)


def upload_file_bytes(data: bytes, file_name: str) -> str:
    files = {"file": (file_name, data)}
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
    kie_url = (body.get("data") or {}).get("fileUrl") or body.get("fileUrl")
    if not kie_url:
        raise RuntimeError(f"Kie stream upload missing fileUrl: {body}")
    return str(kie_url)


def prepare_recraft_image(source_url: str) -> str:
    """Enforce Recraft 5MB / 256–4096px limits. Returns a Kie-hosted URL."""
    from PIL import Image

    resp = requests.get(source_url, timeout=60)
    resp.raise_for_status()
    data = resp.content
    img = Image.open(BytesIO(data))
    w, h = img.size
    if w < RECRAFT_MIN_PX or h < RECRAFT_MIN_PX or w > RECRAFT_MAX_PX or h > RECRAFT_MAX_PX:
        scale = 1.0
        if w < RECRAFT_MIN_PX or h < RECRAFT_MIN_PX:
            scale = max(RECRAFT_MIN_PX / max(w, 1), RECRAFT_MIN_PX / max(h, 1))
        if w * scale > RECRAFT_MAX_PX or h * scale > RECRAFT_MAX_PX:
            scale = min(RECRAFT_MAX_PX / max(w, 1), RECRAFT_MAX_PX / max(h, 1))
        nw = max(RECRAFT_MIN_PX, min(RECRAFT_MAX_PX, int(w * scale)))
        nh = max(RECRAFT_MIN_PX, min(RECRAFT_MAX_PX, int(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    if len(data) > RECRAFT_MAX_BYTES:
        buf = BytesIO()
        rgb = img.convert("RGB")
        quality = 90
        while quality >= 40:
            buf.seek(0)
            buf.truncate(0)
            rgb.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= RECRAFT_MAX_BYTES:
                data = buf.getvalue()
                break
            quality -= 10
        else:
            raise ValueError("Image exceeds Recraft 5 MB limit after compression")
    name = "recraft.jpg" if data[:3] == b"\xff\xd8\xff" else "recraft.png"
    return upload_file_bytes(data, name)


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


def download_url_for(file_url: str) -> str:
    """20-minute Kie download URL fallback when a direct GET fails."""
    resp = requests.post(
        f"{KIE_API}/api/v1/common/download-url",
        headers=_headers(),
        json={"url": file_url},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    payload = data.get("data") or data
    return str(payload.get("downloadUrl") or payload.get("url") or file_url)


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
