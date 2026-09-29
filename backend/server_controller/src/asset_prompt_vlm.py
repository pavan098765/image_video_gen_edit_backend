# asset_prompt_vlm.py
"""
Vision-LM captioning for auto-fill prompts via NVIDIA Nemotron VL (integrate API).

Matches NVIDIA sample style: system ``/think`` for images, user content with text + ``image_url``
using ``data:{mime};base64,...``. Model default ``nvidia/nemotron-nano-12b-v2-vl``.

Env:
  ASSET_PROMPT_VLM_ENABLED          "true" to allow the route (default "false")
  NVIDIA_API_KEY or NVIDIA_LLM_API  Bearer token
  NVIDIA_SYSTEM_PROMPT_IMAGE        Default /think
  ASSET_PROMPT_VLM_TIMEOUT          Per-request timeout seconds (default "120")
  ASSET_PROMPT_MAX_IMAGE_BYTES      Max object size from R2 (default 12 MiB)
  NVIDIA_MAX_TOKENS, NVIDIA_TEMPERATURE, NVIDIA_STREAM, etc. (see nvidia_chat_client)
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

from nvidia_chat_client import (
    build_messages_chat_with_media,
    default_max_tokens,
    default_temperature,
    get_nvidia_api_key,
    get_system_prompt_for_media_extensions,
    nvidia_chat_completion_try_models,
    nvidia_models_for_asset,
    resolve_media_extension,
)

logger = logging.getLogger("asset_prompt_vlm")

_INSTRUCTION_QUERY = """You are helping users write a single prompt for AI image or video generation.
Look at the attached image and write one concise, vivid English prompt that describes the main subject,
setting, lighting, mood, and style. The prompt should be usable as-is in a text-to-image or image-to-video model.
Rules:
- Output ONLY the prompt text. No title, no quotes, no bullet list, no preamble like "Here is".
- Aim for roughly 40–120 words unless the scene is very simple.
- Do not mention that you are an AI or that you see an image."""


def is_asset_prompt_vlm_enabled() -> bool:
    return os.getenv("ASSET_PROMPT_VLM_ENABLED", "false").lower() in ("1", "true", "yes", "on")


def get_timeout_seconds() -> float:
    try:
        return float(os.getenv("ASSET_PROMPT_VLM_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def get_max_image_bytes() -> int:
    try:
        return int(os.getenv("ASSET_PROMPT_MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
    except ValueError:
        return 12 * 1024 * 1024


async def describe_image_with_nvidia(
    r2_key: str,
    raw: bytes,
    mime_hint: str,
) -> Tuple[bool, str]:
    """
    Encode image bytes as a data URL inside Nemotron-compatible ``messages`` and call NVIDIA.
    ``mime_hint`` is Content-Type or from key; used to resolve extension when key has no suffix.
    """
    api_key = get_nvidia_api_key()
    if not api_key:
        logger.error("NVIDIA_API_KEY / NVIDIA_LLM_API is not set")
        return False, "Vision prompt service is not configured"

    ext = resolve_media_extension(r2_key, mime_hint)
    if not ext:
        return False, "Unsupported or unknown image format (use png, jpg, jpeg, or webp)."

    try:
        messages = build_messages_chat_with_media(
            system_prompt=get_system_prompt_for_media_extensions([ext]),
            query=_INSTRUCTION_QUERY,
            media_files=[(ext, raw)],
        )
    except ValueError as e:
        return False, str(e)

    models = nvidia_models_for_asset()
    timeout = get_timeout_seconds()

    ok, text = await nvidia_chat_completion_try_models(
        messages=messages,
        models=models,
        api_key=api_key,
        timeout=timeout,
        max_tokens=default_max_tokens(),
        temperature=default_temperature(),
        min_chars=8,
        log_prefix="asset_vlm",
        stream=None,
    )
    if not ok:
        return False, text

    if len(text) > 4000:
        text = text[:4000]
    return True, text
