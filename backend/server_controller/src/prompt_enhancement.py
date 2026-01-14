# prompt_enhancement.py
"""
Prompt Enhancement Service

Provides AI-powered prompt enhancement for image/video generation.
Uses OpenAI GPT-4 or Claude to improve user prompts.

Configuration via environment variables:
- PROMPT_ENHANCEMENT_ENABLED: "true" to enable (default: "false")
- OPENAI_API_KEY: OpenAI API key for GPT-4
- ANTHROPIC_API_KEY: Anthropic API key for Claude (fallback)
- PROMPT_ENHANCEMENT_MODEL: Model to use (default: "gpt-4o-mini")

The service is designed to be non-blocking and gracefully fallback
to the original prompt if enhancement fails.
"""

import os
import logging
import asyncio
from typing import Optional, Tuple
from functools import lru_cache

import httpx

logger = logging.getLogger("prompt_enhancement")


# =============================================================================
# Configuration
# =============================================================================

def is_enhancement_enabled() -> bool:
    """Check if prompt enhancement is enabled."""
    return os.getenv("PROMPT_ENHANCEMENT_ENABLED", "false").lower() == "true"


@lru_cache(maxsize=1)
def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")


def get_enhancement_model() -> str:
    """Get the model to use for enhancement."""
    return os.getenv("PROMPT_ENHANCEMENT_MODEL", "gpt-4o-mini")


# =============================================================================
# System Prompts
# =============================================================================

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


# =============================================================================
# Model Detection
# =============================================================================

# Models that are video generators
VIDEO_MODELS = {
    "fastwan_22_5b",
    "fastwan_21_1b",
    "cogvideox",
    "animatediff",
}

# Models that are image editors
EDIT_MODELS = {
    "qwen_image_edit_2509",
    "flux_kontext_dev",
    "flux_kontext_pro",
    "sdxl_inpaint",
}


def get_system_prompt_for_model(model: str) -> str:
    """Get the appropriate system prompt based on model type."""
    model_lower = model.lower()
    
    if any(v in model_lower for v in VIDEO_MODELS):
        return VIDEO_ENHANCEMENT_SYSTEM_PROMPT
    elif any(e in model_lower for e in EDIT_MODELS):
        return EDIT_ENHANCEMENT_SYSTEM_PROMPT
    else:
        return IMAGE_ENHANCEMENT_SYSTEM_PROMPT


# =============================================================================
# OpenAI Enhancement
# =============================================================================

async def enhance_with_openai(
    prompt: str,
    model: str,
    api_key: str,
    timeout: float = 10.0,
) -> Tuple[bool, str]:
    """
    Enhance prompt using OpenAI API.
    
    Args:
        prompt: Original user prompt
        model: The generation model (to determine system prompt)
        api_key: OpenAI API key
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (success, enhanced_prompt_or_error)
    """
    enhancement_model = get_enhancement_model()
    system_prompt = get_system_prompt_for_model(model)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": enhancement_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"OpenAI API error: {response.status_code} - {error_msg}")
                return False, f"OpenAI API error: {response.status_code}"
            
            data = response.json()
            enhanced = data["choices"][0]["message"]["content"].strip()
            
            # Validate enhanced prompt is reasonable
            if len(enhanced) < 5:
                logger.warning("Enhanced prompt too short, using original")
                return False, "Enhanced prompt too short"
            
            if len(enhanced) > 2000:
                # Truncate if too long
                enhanced = enhanced[:2000]
            
            logger.info(f"Prompt enhanced: {len(prompt)} -> {len(enhanced)} chars")
            return True, enhanced
            
    except httpx.TimeoutException:
        logger.warning("OpenAI API timeout")
        return False, "Enhancement timeout"
    except Exception as e:
        logger.error(f"OpenAI enhancement error: {e}")
        return False, str(e)


# =============================================================================
# Anthropic Enhancement (Fallback)
# =============================================================================

async def enhance_with_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    timeout: float = 10.0,
) -> Tuple[bool, str]:
    """
    Enhance prompt using Anthropic Claude API (fallback).
    
    Args:
        prompt: Original user prompt
        model: The generation model (to determine system prompt)
        api_key: Anthropic API key
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (success, enhanced_prompt_or_error)
    """
    system_prompt = get_system_prompt_for_model(model)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 500,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"Anthropic API error: {response.status_code} - {error_msg}")
                return False, f"Anthropic API error: {response.status_code}"
            
            data = response.json()
            enhanced = data["content"][0]["text"].strip()
            
            # Validate enhanced prompt is reasonable
            if len(enhanced) < 5:
                logger.warning("Enhanced prompt too short, using original")
                return False, "Enhanced prompt too short"
            
            if len(enhanced) > 2000:
                enhanced = enhanced[:2000]
            
            logger.info(f"Prompt enhanced (Claude): {len(prompt)} -> {len(enhanced)} chars")
            return True, enhanced
            
    except httpx.TimeoutException:
        logger.warning("Anthropic API timeout")
        return False, "Enhancement timeout"
    except Exception as e:
        logger.error(f"Anthropic enhancement error: {e}")
        return False, str(e)


# =============================================================================
# Main Enhancement Function
# =============================================================================

async def enhance_prompt(
    prompt: str,
    model: str,
    timeout: float = 10.0,
) -> Tuple[bool, str, Optional[str]]:
    """
    Enhance a user prompt using AI.
    
    This is the main entry point for prompt enhancement.
    It tries OpenAI first, then falls back to Anthropic if available.
    If enhancement is disabled or fails, returns the original prompt.
    
    Args:
        prompt: Original user prompt
        model: The generation model being used
        timeout: Total timeout for enhancement attempts
    
    Returns:
        Tuple of (was_enhanced, final_prompt, original_prompt_if_enhanced)
        - was_enhanced: True if prompt was successfully enhanced
        - final_prompt: The enhanced prompt or original if failed
        - original_prompt_if_enhanced: Original prompt if enhanced, None otherwise
    """
    # Skip if disabled
    if not is_enhancement_enabled():
        logger.debug("Prompt enhancement disabled")
        return False, prompt, None
    
    # Skip if prompt is empty or too short
    if not prompt or len(prompt.strip()) < 3:
        logger.debug("Prompt too short for enhancement")
        return False, prompt, None
    
    original_prompt = prompt
    
    # Try OpenAI first
    openai_key = get_openai_api_key()
    if openai_key:
        success, result = await enhance_with_openai(prompt, model, openai_key, timeout)
        if success:
            return True, result, original_prompt
        logger.warning(f"OpenAI enhancement failed: {result}")
    
    # Fallback to Anthropic
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        success, result = await enhance_with_anthropic(prompt, model, anthropic_key, timeout)
        if success:
            return True, result, original_prompt
        logger.warning(f"Anthropic enhancement failed: {result}")
    
    # All methods failed or no API keys configured
    if not openai_key and not anthropic_key:
        logger.warning("No API keys configured for prompt enhancement")
    
    return False, prompt, None


# =============================================================================
# Utility Functions
# =============================================================================

def should_enhance_for_model(model: str, mode: Optional[str] = None) -> bool:
    """
    Check if prompt enhancement is appropriate for the given model/mode.
    
    Some models/modes don't benefit from prompt enhancement:
    - Background removal (no prompt needed)
    - Object removal/inpainting (prompt is about what to remove)
    - Upscaling (no prompt needed)
    
    Args:
        model: Model identifier
        mode: Operation mode (e.g., "bg_remove", "lama_inpaint")
    
    Returns:
        True if enhancement is appropriate
    """
    # Models that don't use prompts
    no_prompt_models = {"basic_utils", "real_esrgan", "upscaler"}
    if model.lower() in no_prompt_models:
        return False
    
    # Modes that don't benefit from enhancement
    no_enhancement_modes = {"bg_remove", "lama_inpaint", "upscale"}
    if mode and mode.lower() in no_enhancement_modes:
        return False
    
    return True
