# asset_prompt_vlm.py
"""
Vision captioning for auto-fill prompts via the Gemini Cloudflare worker.

The controller loads the image from R2 and posts it to the worker. The worker
checks the user's Firebase token and calls Gemini. The Gemini key stays on
the worker.

Env:
  ASSET_PROMPT_VLM_ENABLED          "true" to allow the route (default "true")
  GEMINI_DIRECTOR_URL               Worker POST URL
  GEMINI_DIRECTOR_MODEL             Must be on the worker allowlist
  ASSET_PROMPT_VLM_TIMEOUT          Per-request timeout seconds (default "120")
  ASSET_PROMPT_MAX_IMAGE_BYTES      Max object size from R2 (default 4 MiB)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Tuple

import requests

logger = logging.getLogger("asset_prompt_vlm")

_DEFAULT_WORKER_URL = "https://kala-gemini-director.pavanalur3.workers.dev/v1/generate"

_INSTRUCTION = (
    "You are helping users write a single prompt for AI image or video generation. "
    "Look at the attached image and write one concise, vivid English prompt that describes "
    "the main subject, setting, lighting, mood, and style. The prompt should be usable as-is "
    "in a text-to-image or image-to-video model. "
    "Output ONLY the prompt text. No title, no quotes, no bullet list, no preamble. "
    "Aim for roughly 40-120 words unless the scene is very simple. "
    "Do not mention that you are an AI or that you see an image."
)


def is_asset_prompt_vlm_enabled() -> bool:
    return os.getenv("ASSET_PROMPT_VLM_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def get_timeout_seconds() -> float:
    try:
        return float(os.getenv("ASSET_PROMPT_VLM_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def get_max_image_bytes() -> int:
    try:
        return int(os.getenv("ASSET_PROMPT_MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
    except ValueError:
        return 4 * 1024 * 1024


def _mime(r2_key: str, mime_hint: str) -> str:
    hint = (mime_hint or "").split(";", 1)[0].strip().lower()
    if hint in ("image/jpeg", "image/png", "image/webp"):
        return hint
    ext = ""
    if "." in (r2_key or ""):
        ext = r2_key.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    return ""


def _prompt_from_gemini(body: dict) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [p.get("text") or "" for p in parts if isinstance(p, dict)]
    return "\n".join(t for t in texts if t).strip()


def describe_image_with_gemini(
    r2_key: str,
    raw: bytes,
    mime_hint: str,
    authorization: str,
) -> Tuple[bool, str]:
    """POST the image to the Gemini worker using the caller's Firebase token."""
    token = (authorization or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        return False, "Sign in required"

    mime = _mime(r2_key, mime_hint)
    if not mime:
        return False, "Unsupported or unknown image format (use png, jpg, jpeg, or webp)."

    model = (os.getenv("GEMINI_DIRECTOR_MODEL") or "gemini-3.5-flash-lite").strip()
    url = (os.getenv("GEMINI_DIRECTOR_URL") or _DEFAULT_WORKER_URL).strip()
    payload = {
        "step": "asset_prompt",
        "model": model,
        "maxOutputTokens": 1024,
        "contents": [
            {
                "parts": [
                    {"text": _INSTRUCTION},
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}},
                ]
            }
        ],
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=get_timeout_seconds(),
        )
    except requests.RequestException as exc:
        logger.warning("Gemini worker request failed: %s", exc)
        return False, "Auto-prompt service is unavailable"

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code >= 400:
        err = body.get("error") if isinstance(body, dict) else None
        return False, str(err or f"Auto-prompt failed ({resp.status_code})")

    text = _prompt_from_gemini(body if isinstance(body, dict) else {})
    if not text:
        return False, "Auto-prompt returned no text"
    if len(text) > 4000:
        text = text[:4000]
    return True, text
