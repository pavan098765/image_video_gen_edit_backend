# prompt_enhancement.py
"""
Prompt Enhancement Service — NVIDIA integrate API via ``requests`` (``nvidia_chat_client``), no OpenAI SDK.

Configuration via environment variables:
- PROMPT_ENHANCEMENT_ENABLED: "true" to enable (default: "false")
- NVIDIA_API_KEY or NVIDIA_LLM_API: Bearer nvapi-... (required when enabled)
- PROMPT_ENHANCEMENT_NVIDIA_MODELS / NVIDIA_TEXT_MODELS: optional (default nvidia/nemotron-nano-12b-v2-vl)
- PROMPT_ENHANCEMENT_TIMEOUT: HTTP timeout seconds (default: "120")
- NVIDIA_MAX_TOKENS, NVIDIA_TEMPERATURE, NVIDIA_STREAM: shared Nemotron payload (see nvidia_chat_client)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nvidia_chat_client import (
    build_messages_chat_with_media,
    default_max_tokens,
    default_temperature,
    get_nvidia_api_key,
    get_system_prompt_for_media_extensions,
    nvidia_chat_completion_try_models,
    nvidia_models_for_enhancement,
)

logger = logging.getLogger("prompt_enhancement")


def is_enhancement_enabled() -> bool:
    return os.getenv("PROMPT_ENHANCEMENT_ENABLED", "false").lower() == "true"


def get_enhancement_timeout() -> float:
    try:
        return float(os.getenv("PROMPT_ENHANCEMENT_TIMEOUT", "120"))
    except ValueError:
        return 120.0


IMAGE_ENHANCEMENT_SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generation models like Stable Diffusion, DALL-E, and Midjourney.

Your task is to enhance the user's prompt to create better, more detailed images while preserving their original intent.

Guidelines:
1. Add specific details about lighting, composition, and style
2. Include relevant artistic terms and techniques
3. Specify camera angles or perspectives if appropriate
4. Add atmospheric and mood descriptors
5. Include quality boosters like "highly detailed", "professional", "4K"
6. Keep the enhanced prompt concise (under 200 words)
7. Preserve the core subject and intent of the original prompt
8. Do NOT add NSFW content or change the subject matter

Respond with ONLY the enhanced prompt, no explanations or prefixes."""

VIDEO_ENHANCEMENT_SYSTEM_PROMPT = """You are an expert prompt engineer for AI video generation models.

Your task is to enhance the user's prompt to create better, more cinematic videos while preserving their original intent.

Guidelines:
1. Add camera movement descriptions (pan, zoom, tracking shot, etc.)
2. Include temporal transitions and motion details
3. Specify lighting and atmosphere
4. Add cinematography terms where appropriate
5. Include pacing and rhythm descriptors
6. Keep the enhanced prompt concise (under 150 words)
7. Preserve the core subject and intent of the original prompt
8. Do NOT add NSFW content or change the subject matter

Respond with ONLY the enhanced prompt, no explanations or prefixes."""

EDIT_ENHANCEMENT_SYSTEM_PROMPT = """You are an expert prompt engineer for AI image editing models.

Your task is to enhance the user's editing instruction to achieve better results while preserving their original intent.

Guidelines:
1. Be specific about what should change and what should stay the same
2. Add details about style, colors, or effects if relevant
3. Clarify the scope of the edit (full image vs specific area)
4. Include quality preservation instructions
5. Keep the enhanced prompt concise and actionable
6. Preserve the core editing intent
7. Do NOT add NSFW content or change the editing goal

Respond with ONLY the enhanced prompt, no explanations or prefixes."""

VIDEO_MODELS = {
    "fastwan_22_5b",
    "fastwan_21_1b",
    "cogvideox",
    "animatediff",
}

EDIT_MODELS = {
    "qwen_image_edit_2509",
    "flux_kontext_dev",
    "flux_kontext_pro",
    "sdxl_inpaint",
}


def get_system_prompt_for_model(model: str) -> str:
    model_lower = model.lower()

    if any(v in model_lower for v in VIDEO_MODELS):
        return VIDEO_ENHANCEMENT_SYSTEM_PROMPT
    if any(e in model_lower for e in EDIT_MODELS):
        return EDIT_ENHANCEMENT_SYSTEM_PROMPT
    return IMAGE_ENHANCEMENT_SYSTEM_PROMPT


async def enhance_with_nvidia(
    prompt: str,
    model: str,
    api_key: str,
    timeout: float,
) -> Tuple[bool, str]:
    instruction = get_system_prompt_for_model(model)
    combined_query = f"{instruction}\n\nUser prompt to enhance:\n{prompt}"
    messages = build_messages_chat_with_media(
        system_prompt=get_system_prompt_for_media_extensions([]),
        query=combined_query,
        media_files=[],
    )

    models = nvidia_models_for_enhancement()
    ok, text = await nvidia_chat_completion_try_models(
        messages=messages,
        models=models,
        api_key=api_key,
        timeout=timeout,
        max_tokens=default_max_tokens(),
        temperature=default_temperature(),
        min_chars=5,
        log_prefix="prompt_enhance",
        stream=None,
    )

    if not ok:
        return False, text

    if len(text) > 2000:
        text = text[:2000]

    logger.info(f"Prompt enhanced: {len(prompt)} -> {len(text)} chars")
    return True, text


async def enhance_prompt(
    prompt: str,
    model: str,
    timeout: float = 10.0,
) -> Tuple[bool, str, Optional[str]]:
    if not is_enhancement_enabled():
        logger.debug("Prompt enhancement disabled")
        return False, prompt, None

    if not prompt or len(prompt.strip()) < 3:
        logger.debug("Prompt too short for enhancement")
        return False, prompt, None

    original_prompt = prompt
    api_key = get_nvidia_api_key()
    if not api_key:
        logger.warning("NVIDIA_API_KEY / NVIDIA_LLM_API not set; prompt enhancement unavailable")
        return False, prompt, None

    t = timeout if timeout > 0 else get_enhancement_timeout()
    success, result = await enhance_with_nvidia(prompt, model, api_key, t)
    if success:
        return True, result, original_prompt

    logger.warning(f"NVIDIA enhancement failed: {result}")
    return False, prompt, None


def _style_book() -> Dict[str, Any]:
    path = Path(__file__).with_name("prompt_styles.json")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("prompt_styles.json must be an object")
    return data


def _fallback_kind(model: str, field: str) -> str:
    if field in ("lyrics",):
        return "lyrics"
    if field in ("style",):
        return "style"
    if field in ("negative", "negative_prompt", "negative_tags"):
        return "negative"
    if field in ("voice_description",):
        return "voice_description"
    name = (model or "").lower()
    if any(token in name for token in ("expand",)):
        return "expand"
    if any(token in name for token in ("edit", "inpaint", "i2i")):
        return "edit"
    if any(token in name for token in ("suno", "music", "ace_step", "minimax_music")):
        return "music"
    if any(token in name for token in ("tts", "voice", "speech")):
        return "speech"
    if any(token in name for token in ("video", "ltx", "wan", "i2v", "t2v", "h3")):
        return "video"
    return "image"


def _model_field_spec(entry: Dict[str, Any], field: str) -> Optional[Dict[str, Any]]:
    names = {
        "negative": ("negative", "negative_prompt", "negative_tags"),
        "negative_prompt": ("negative_prompt", "negative", "negative_tags"),
        "negative_tags": ("negative_tags", "negative", "negative_prompt"),
    }.get(field, (field,))
    for name in names:
        spec = entry.get(name)
        if isinstance(spec, dict):
            return spec
    if field in ("", "prompt"):
        prompt = entry.get("prompt")
        if isinstance(prompt, dict):
            return prompt
    return None


def instruction_for(model: str, field: str, invent: bool) -> str:
    book = _style_book()
    fallbacks = book.get("fallbacks") if isinstance(book.get("fallbacks"), dict) else {}
    models = book.get("models") if isinstance(book.get("models"), dict) else {}
    entry = models.get(model) if isinstance(models.get(model), dict) else {}
    spec = _model_field_spec(entry, field or "prompt")
    kind = ""
    if isinstance(spec, dict):
        if spec.get("refine") or spec.get("invent"):
            text = spec.get("invent" if invent else "refine")
            if isinstance(text, str) and text.strip():
                return text.strip()
        kind = str(spec.get("kind") or "")
    if not kind:
        kind = _fallback_kind(model, field)
    chosen = fallbacks.get(kind) if isinstance(fallbacks.get(kind), dict) else None
    if chosen is None:
        chosen = fallbacks.get("image") if isinstance(fallbacks.get("image"), dict) else {}
    text = chosen.get("invent" if invent else "refine") if isinstance(chosen, dict) else ""
    if not isinstance(text, str) or not text.strip():
        return "Write one clear prompt. No explanation."
    return text.strip()


async def suggest_prompt(
    model: str,
    field: str,
    text: str,
    caption: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    typed = (text or "").strip()
    picture = (caption or "").strip()
    invent = not typed
    instruction = instruction_for(model or "", field or "prompt", invent)
    parts = [instruction]
    if picture:
        parts.append(f"Picture description:\n{picture[:1500]}")
    if typed:
        parts.append(f"User text:\n{typed[:4000]}")
    else:
        parts.append("The box is empty. Write one new prompt.")
    parts.append("Reply with only the prompt text.")
    api_key = get_nvidia_api_key()
    if not api_key:
        raise RuntimeError("Prompt suggestions are unavailable.")
    messages = build_messages_chat_with_media(
        system_prompt=get_system_prompt_for_media_extensions([]),
        query="\n\n".join(parts),
        media_files=[],
    )
    ok, result = await nvidia_chat_completion_try_models(
        messages=messages,
        models=nvidia_models_for_enhancement(),
        api_key=api_key,
        timeout=timeout if timeout > 0 else get_enhancement_timeout(),
        max_tokens=default_max_tokens(),
        temperature=default_temperature(),
        min_chars=5,
        log_prefix="suggest_prompt",
        stream=None,
    )
    if not ok:
        raise RuntimeError(result or "Prompt suggestion failed.")
    cleaned = result.strip()
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000].rstrip()
    return cleaned


def should_enhance_for_model(model: str, mode: Optional[str] = None) -> bool:
    no_prompt_models = {"basic_utils", "real_esrgan", "upscaler", "suno_v6"}
    if model.lower() in no_prompt_models:
        return False

    no_enhancement_modes = {"bg_remove", "lama_inpaint", "upscale"}
    if mode and mode.lower() in no_enhancement_modes:
        return False

    return True
