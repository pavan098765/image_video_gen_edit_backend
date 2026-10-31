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


def _bearer_token(authorization: str) -> str:
    token = (authorization or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _gemini_target() -> Tuple[str, str]:
    model = (os.getenv("GEMINI_DIRECTOR_MODEL") or "gemini-3.5-flash-lite").strip()
    url = (os.getenv("GEMINI_DIRECTOR_URL") or _DEFAULT_WORKER_URL).strip()
    return model, url


def _post_gemini(payload: dict, authorization: str, max_chars: int) -> Tuple[bool, str]:
    """POST one generateContent body to the Gemini worker with the caller's Firebase token."""
    token = _bearer_token(authorization)
    if not token:
        return False, "Sign in required"
    _, url = _gemini_target()
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
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return True, text


def describe_image_with_gemini(
    r2_key: str,
    raw: bytes,
    mime_hint: str,
    authorization: str,
) -> Tuple[bool, str]:
    """POST the image to the Gemini worker using the caller's Firebase token."""
    mime = _mime(r2_key, mime_hint)
    if not mime:
        return False, "Unsupported or unknown image format (use png, jpg, jpeg, or webp)."

    model, _ = _gemini_target()
    payload = {
        "step": "asset_prompt",
        "model": model,
        "maxOutputTokens": 4096,
        "contents": [
            {
                "parts": [
                    {"text": _INSTRUCTION},
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}},
                ]
            }
        ],
    }
    return _post_gemini(payload, authorization, 4000)


def suggest_text_with_gemini(prompt_text: str, authorization: str) -> Tuple[bool, str]:
    """Ask Gemini for one prompt. Text only. Same worker and Firebase token as image auto-prompt."""
    text = (prompt_text or "").strip()
    if not text:
        return False, "Prompt text is required"
    model, _ = _gemini_target()
    payload = {
        "step": "suggest_prompt",
        "model": model,
        "maxOutputTokens": 4096,
        "contents": [{"parts": [{"text": text[:20000]}]}],
    }
    return _post_gemini(payload, authorization, 8000)


# VLC equalizer curve ids, plus style aliases that map to one of those curves.
# The gains live in the app. This list only decides which id is allowed.
_EQ_CURVE_IDS = (
    "flat",
    "classical",
    "club",
    "dance",
    "fullbass",
    "fullbasstreble",
    "fulltreble",
    "headphones",
    "largehall",
    "live",
    "party",
    "pop",
    "reggae",
    "rock",
    "ska",
    "soft",
    "softrock",
    "techno",
)
_EQ_ALIASES = (
    ("hiphop", "fullbass"),
    ("trap", "fullbass"),
    ("edm", "techno"),
    ("ballad", "soft"),
    ("rnb", "soft"),
    ("metal", "rock"),
    ("jazz", "classical"),
    ("acoustic", "live"),
    ("lofi", "soft"),
    ("ambient", "soft"),
    ("latin", "dance"),
    ("punjabi", "party"),
    ("bhangra", "party"),
    ("bollywood", "pop"),
    ("indie", "softrock"),
    ("country", "softrock"),
)
EQ_PRESET_IDS = _EQ_CURVE_IDS + tuple(alias for alias, _curve in _EQ_ALIASES)
EQ_PRESET_ALLOWLIST = frozenset(EQ_PRESET_IDS)


def normalize_eq_preset(raw: str) -> str:
    """Keep a Gemini reply only when the trimmed text is exactly one allowed id."""
    text = (raw or "").strip().lower()
    if text in EQ_PRESET_ALLOWLIST:
        return text
    return "flat"


def eq_preset_prompt(title: str, style: str, tags: str, lyrics: str) -> str:
    """Tell Gemini every allowed id and which alias to use for a style."""
    ids = "\n".join(EQ_PRESET_IDS)
    aliases = "\n".join(f"{alias}: {curve}" for alias, curve in _EQ_ALIASES)
    return (
        "Choose one equalizer preset for this song. "
        "Reply with exactly one id from the list and no other text.\n\n"
        f"Ids:\n{ids}\n\n"
        "When the style matches an alias, return that alias id:\n"
        f"{aliases}\n\n"
        f"Title: {(title or '').strip()[:200]}\n"
        f"Style: {(style or '').strip()[:1000]}\n"
        f"Tags: {(tags or '').strip()[:500]}\n"
        f"Lyrics: {(lyrics or '').strip()[:500]}\n"
    )


def choose_eq_preset(
    title: str,
    style: str,
    tags: str,
    lyrics: str,
    authorization: str,
) -> str:
    """Ask Gemini for one preset id. Anything else, including a failed call, is flat."""
    if not is_asset_prompt_vlm_enabled():
        return "flat"
    model, _url = _gemini_target()
    payload = {
        "step": "eq_preset",
        "model": model,
        "maxOutputTokens": 16,
        "contents": [
            {"parts": [{"text": eq_preset_prompt(title, style, tags, lyrics)}]}
        ],
    }
    ok, text = _post_gemini(payload, authorization, 80)
    if not ok:
        return "flat"
    return normalize_eq_preset(text)
