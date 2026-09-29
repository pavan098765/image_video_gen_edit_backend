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

import logging
import os
from typing import Optional, Tuple

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


def should_enhance_for_model(model: str, mode: Optional[str] = None) -> bool:
    no_prompt_models = {"basic_utils", "real_esrgan", "upscaler"}
    if model.lower() in no_prompt_models:
        return False

    no_enhancement_modes = {"bg_remove", "lama_inpaint", "upscale"}
    if mode and mode.lower() in no_enhancement_modes:
        return False

    return True
