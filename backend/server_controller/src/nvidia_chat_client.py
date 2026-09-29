# nvidia_chat_client.py
"""
NVIDIA integrate.api.nvidia.com chat/completions (Nemotron VL style) using ``requests`` only.

Aligns with NVIDIA samples: ``/think`` system for images, ``/no_think`` for video,
``image_url`` / ``video_url`` parts with ``data:{mime};base64,...`` URLs, model
``nvidia/nemotron-nano-12b-v2-vl``.

Env:
  NVIDIA_API_KEY or NVIDIA_LLM_API     Bearer nvapi-... (``NVIDIA_API_KEY`` checked first)
  NVIDIA_CHAT_COMPLETIONS_URL            Default https://integrate.api.nvidia.com/v1/chat/completions
  NVIDIA_SYSTEM_PROMPT_IMAGE             Default /think
  NVIDIA_SYSTEM_PROMPT_VIDEO             Default /no_think
  NVIDIA_MAX_TOKENS                      Default 4096
  NVIDIA_TEMPERATURE                     Default 1
  NVIDIA_TOP_P, NVIDIA_FREQUENCY_PENALTY, NVIDIA_PRESENCE_PENALTY
  NVIDIA_STREAM                          Set "true" to use SSE streaming (default off for server)
  ASSET_PROMPT_NVIDIA_MODELS / NVIDIA_VISION_MODELS
  PROMPT_ENHANCEMENT_NVIDIA_MODELS / NVIDIA_TEXT_MODELS
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger("nvidia_chat_client")

NVIDIA_CHAT_COMPLETIONS_DEFAULT = "https://integrate.api.nvidia.com/v1/chat/completions"

# ext -> (mime, content_part_type) — matches NVIDIA sample
K_SUPPORTED_LIST: Dict[str, Tuple[str, str]] = {
    "png": ("image/png", "image_url"),
    "jpg": ("image/jpeg", "image_url"),
    "jpeg": ("image/jpeg", "image_url"),
    "webp": ("image/webp", "image_url"),
    "mp4": ("video/mp4", "video_url"),
    "webm": ("video/webm", "video_url"),
    "mov": ("video/mov", "video_url"),
}

MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
}

DEFAULT_NEMOTRON_MODEL: Tuple[str, ...] = ("nvidia/nemotron-nano-12b-v2-vl",)


@lru_cache(maxsize=1)
def get_nvidia_api_key() -> Optional[str]:
    for env_name in ("NVIDIA_API_KEY", "NVIDIA_LLM_API"):
        k = os.getenv(env_name, "").strip()
        if k:
            return k
    return None


def _chat_completions_url() -> str:
    return os.getenv("NVIDIA_CHAT_COMPLETIONS_URL", NVIDIA_CHAT_COMPLETIONS_DEFAULT).strip()


def extension_from_key_or_path(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    _, ext = os.path.splitext(base)
    return ext[1:].lower() if ext else ""


def resolve_media_extension(key_or_name: str, mime_hint: str) -> Optional[str]:
    """Return a key in K_SUPPORTED_LIST, or None."""
    ext = extension_from_key_or_path(key_or_name)
    if ext in K_SUPPORTED_LIST:
        return ext
    mime = (mime_hint or "").split(";")[0].strip().lower()
    return MIME_TO_EXT.get(mime)


def mime_for_extension(ext: str) -> str:
    return K_SUPPORTED_LIST[ext][0]


def media_part_type_for_extension(ext: str) -> str:
    return K_SUPPORTED_LIST[ext][1]


def get_system_prompt_for_media_extensions(extensions: Sequence[str]) -> str:
    """Images: /think (default). Video (mp4/webm/mov): /no_think (default)."""
    has_video = any(e in ("mp4", "webm", "mov") for e in extensions)
    if has_video:
        return os.getenv("NVIDIA_SYSTEM_PROMPT_VIDEO", "/no_think").strip() or "/no_think"
    return os.getenv("NVIDIA_SYSTEM_PROMPT_IMAGE", "/think").strip() or "/think"


def media_object_data_url(ext: str, raw: bytes) -> Dict[str, Any]:
    mime = mime_for_extension(ext)
    part_type = media_part_type_for_extension(ext)
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    url = f"data:{mime};base64,{b64}"
    return {"type": part_type, part_type: {"url": url}}


def build_messages_chat_with_media(
    *,
    system_prompt: str,
    query: str,
    media_files: List[Tuple[str, bytes]],
) -> List[Dict[str, Any]]:
    """
    ``media_files``: list of ``(extension, raw_bytes)`` e.g. ``("png", data)``.

    Text-only (no media): user ``content`` is the query string (NVIDIA sample).
    With media: user ``content`` is an array starting with text, then image_url / video_url parts.
    Only a single video is allowed (assertion in sample).
    """
    if not media_files:
        content: Any = query
    else:
        has_video = False
        for ext, _ in media_files:
            if ext not in K_SUPPORTED_LIST:
                raise ValueError(f"Unsupported media extension: {ext}")
            if media_part_type_for_extension(ext) == "video_url":
                has_video = True
        if has_video and len(media_files) != 1:
            raise ValueError("Only single video supported.")

        content = [{"type": "text", "text": query}]
        for ext, raw in media_files:
            content.append(media_object_data_url(ext, raw))

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_model_list(env_value: Optional[str], defaults: Sequence[str]) -> List[str]:
    if env_value is None or not str(env_value).strip():
        return list(defaults)
    return [p.strip() for p in str(env_value).split(",") if p.strip()]


def nvidia_models_for_asset() -> List[str]:
    raw = os.getenv("ASSET_PROMPT_NVIDIA_MODELS") or os.getenv("NVIDIA_VISION_MODELS")
    return parse_model_list(raw, DEFAULT_NEMOTRON_MODEL)


def nvidia_models_for_enhancement() -> List[str]:
    raw = os.getenv("PROMPT_ENHANCEMENT_NVIDIA_MODELS") or os.getenv("NVIDIA_TEXT_MODELS")
    return parse_model_list(raw, DEFAULT_NEMOTRON_MODEL)


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif "text" in p:
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts).strip()
    return str(content).strip()


def _float_env(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("1", "true", "yes", "on")


def _collect_streaming_text(resp: requests.Response) -> str:
    chunks: List[str] = []
    for line in resp.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8", errors="replace")
        if not s.startswith("data: "):
            continue
        data = s[6:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            c = delta.get("content")
            if c:
                chunks.append(str(c))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    return "".join(chunks).strip()


def nvidia_chat_completion_try_models_sync(
    *,
    messages: List[Dict[str, Any]],
    models: Sequence[str],
    api_key: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
    min_chars: int,
    log_prefix: str = "nvidia",
    stream: Optional[bool] = None,
) -> Tuple[bool, str]:
    last_err = "Provider error"
    url = _chat_completions_url()
    use_stream = _bool_env("NVIDIA_STREAM") if stream is None else stream

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if use_stream else "application/json",
    }

    top_p = _float_env("NVIDIA_TOP_P", "1")
    freq = _float_env("NVIDIA_FREQUENCY_PENALTY", "0")
    pres = _float_env("NVIDIA_PRESENCE_PENALTY", "0")

    for model in models:
        payload: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": freq,
            "presence_penalty": pres,
            "messages": messages,
            "stream": use_stream,
            "model": model,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=use_stream)
        except requests.Timeout:
            logger.warning("[%s] NVIDIA timeout model=%s", log_prefix, model)
            last_err = "Request timed out"
            continue
        except requests.RequestException as e:
            logger.exception("[%s] NVIDIA request failed model=%s: %s", log_prefix, model, e)
            last_err = "Request failed"
            continue

        if resp.status_code != 200:
            err_snip = resp.text[:800] if not use_stream else "(stream)"
            logger.error(
                "[%s] NVIDIA error model=%s status=%s body=%r",
                log_prefix,
                model,
                resp.status_code,
                err_snip,
            )
            last_err = f"Provider error ({resp.status_code})"
            continue

        try:
            if use_stream:
                text = _collect_streaming_text(resp)
            else:
                data = resp.json()
                msg = data["choices"][0]["message"]
                text = message_content_to_text(msg.get("content"))
                selected = data.get("model", model)
                logger.debug("[%s] upstream model field=%s", log_prefix, selected)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                "[%s] Bad NVIDIA response model=%s: %s body=%s",
                log_prefix,
                model,
                e,
                (resp.text[:300] if not use_stream else ""),
            )
            last_err = "Invalid provider response"
            continue

        if len(text) < min_chars:
            logger.warning("[%s] Output too short model=%s len=%s", log_prefix, model, len(text))
            last_err = "Generated text too short"
            continue

        logger.info("[%s] ok model=%s len=%s stream=%s", log_prefix, model, len(text), use_stream)
        return True, text

    return False, last_err


async def nvidia_chat_completion_try_models(
    *,
    messages: List[Dict[str, Any]],
    models: Sequence[str],
    api_key: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
    min_chars: int,
    log_prefix: str = "nvidia",
    stream: Optional[bool] = None,
) -> Tuple[bool, str]:
    return await asyncio.to_thread(
        nvidia_chat_completion_try_models_sync,
        messages=messages,
        models=models,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        min_chars=min_chars,
        log_prefix=log_prefix,
        stream=stream,
    )


def default_max_tokens() -> int:
    return _int_env("NVIDIA_MAX_TOKENS", 4096)


def default_temperature() -> float:
    return _float_env("NVIDIA_TEMPERATURE", "1")
