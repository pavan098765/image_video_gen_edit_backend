#!/usr/bin/env python3
"""
comfyui_inference.py
====================

Inference function for the Wan 2.2 ComfyUI worker (v2.5.4 workflow).

Supports four generation modes via Any Switch (rgthree) node routing:
  - i2v: Image-to-Video (default)
  - t2v: Text-to-Video
  - t2i: Text-to-Image
  - s2v: Subject-to-Video

Longer videos are achieved via context windows (single-pass) rather than
multi-pass SVI.  Frame interpolation uses the workflow's built-in FL_RIFE
node instead of external Python RIFE.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("comfyui_inference")

# ---------------------------------------------------------------------------
# Together AI config for prompt enhancement
# ---------------------------------------------------------------------------
TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "d9d8a62fa13d36b44cbb16188734e5dc17eec3fd65c2ebc4e0d533bd9bc6c33f",
)
TOGETHER_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

_ENHANCE_SYSTEM_PROMPT = (
    "You are an expert cinematic prompt engineer specialized in enhancing prompts for AI video generation models.\n\n"
    "Your task is NOT to change the user's intent, but to expand and refine it into a highly detailed, "
    "visually descriptive, and generation-optimized prompt.\n\n"
    "Follow these principles:\n"
    "1. Preserve Core Intent - Never alter the main subject or meaning.\n"
    "2. Add Missing Visual Information - subject description, environment, lighting, camera, mood, motion.\n"
    "3. Cinematic Quality - include: ultra detailed, cinematic, photorealistic, high dynamic range.\n"
    "4. Video Motion - add camera movement, subject motion, temporal continuity.\n"
    "5. Avoid Negative Behavior - don't add unrelated objects or change identity.\n\n"
    "Output ONLY:\n\nENHANCED PROMPT:\n<final enhanced prompt>\n\n"
    "Do NOT explain reasoning. Do NOT include commentary."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/app/comfyui")
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKFLOW_JSON = os.path.join(
    WORKFLOW_DIR,
    "WAN2.2 T2V-I2V-T2I-S2V K3NK v2.5.4 SVI.json",
)

# ---------------------------------------------------------------------------
# Node ID constants (v2.5.4 workflow)
# ---------------------------------------------------------------------------

# Prompt / text encoding
WILDCARD_NODE = "35"                   # ImpactWildcardProcessor (prompt entry point)
TEXT_ENCODE_CACHED_NODE = "1913"       # WanVideoTextEncodeCached (pos+neg prompt)
CLIP_TEXT_ENCODE_NODE = "3093"         # CLIPTextEncode (Positive Prompt)
NAG_NODE = "1122"                      # WanVideoApplyNAG

# Seed
SEED_NODE = "1853"                     # Seed (rgthree) -> both samplers

# Steps / Shift / Denoise
STEPS_NODE = "1223"                    # INTConstant (Steps)
SHIFT_NODE = "1221"                    # PrimitiveFloat (Shift)
DENOISE_NODE = "3010"                  # FloatConstant (Denoise V2V)

# Duration / Frames
DURATION_NODE = "3797"                 # PrimitiveInt (Duration in frames, simple mode)
FRAMES_PER_CONTEXT_NODE = "2797"       # PrimitiveInt (Frames per context, ctx mode)
NUM_CONTEXTS_NODE = "2795"             # PrimitiveInt (Nr of prompts/contexts)
TOTAL_FRAMES_NODE = "2798"             # SimpleMath+ (Total frames = a*b)

# Context window options
CONTEXT_OPTIONS_HIGH_NODE = "2801"     # WanVideoContextOptions (high noise sampler)
CONTEXT_OPTIONS_LOW_NODE = "2802"      # WanVideoContextOptions (low noise sampler)

# Samplers
HIGH_NOISE_SAMPLER_NODE = "72"         # WanVideoSampler (high noise)
LOW_NOISE_SAMPLER_NODE = "192"         # WanVideoSampler (low noise)

# Schedulers
HIGH_SCHEDULER_NODE = "3121"           # WanVideoScheduler (high)
LOW_SCHEDULER_NODE = "3122"            # WanVideoScheduler (low)

# Image loading (I2V / S2V)
LOAD_IMAGE_NODE = "657"                # LoadImage (primary I2V input, mode=0)
LOAD_IMAGE_ALT_NODE = "3915"           # LoadImage (alternate, mode=4)

# I2V / T2V encoding
I2V_ENCODE_NODE = "659"                # WanVideoImageToVideoEncode
EMPTY_EMBEDS_NODE = "108"              # WanVideoEmptyEmbeds

# SVI Pro nodes
SVI_PRO_NODE_A = "3715"                # WanVideoSVIProEmbeds (with prev_samples)
SVI_PRO_NODE_B = "3713"                # WanVideoSVIProEmbeds (without prev_samples)

# Model loaders
I2V_HIGH_MODEL_NODE = "1173"           # WanVideoModelLoader i2v high noise (mode=0)
I2V_LOW_MODEL_NODE = "1172"            # WanVideoModelLoader i2v low noise (mode=0)
T2V_HIGH_MODEL_NODE = "220"            # WanVideoModelLoader t2v high noise (mode=4)
T2V_LOW_MODEL_NODE = "164"             # WanVideoModelLoader t2v low noise (mode=4)

# Output / Video
OUTPUT_VHS_NODE = "249"                # VHS_VideoCombine "Save Interpoled" (save_output=True)
PREVIEW_VHS_NODE = "16"                # VHS_VideoCombine "Preview" (save_output=False)

# FL_RIFE (built-in frame interpolation)
RIFE_NODE = "3996"                     # FL_RIFE (ckpt_name, multiplier, ensemble)

# Post-processing chain: 1977 (Any Switch) -> 244 (ImageUpscaleWithModelBatched) -> 246 (ImageResizeKJv2) -> 3996 (FL_RIFE) -> ... -> 249
UPSCALE_MODEL_LOADER_NODE = "245"      # UpscaleModelLoader (model_name)
UPSCALE_BATCH_NODE = "244"             # ImageUpscaleWithModelBatched (images from 1977)
RESIZE_AFTER_UPSCALE_NODE = "246"      # ImageResizeKJv2 (image from 244 when upscale on, or from 1977 when off)
UPSCALE_INPUT_SWITCH_NODE = "1977"     # Any Switch: output feeds 244; 246 can take from 244 (upscaled) or 1977 (bypass)

# ---------------------------------------------------------------------------
# Any Switch node IDs
# ---------------------------------------------------------------------------

EMBEDS_HIGH_SWITCH = "814"             # Embeds high -> high noise sampler
EMBEDS_LOW_SWITCH = "1756"             # Embeds low -> low noise sampler
HIGH_MODELS_SWITCH = "1215"            # High Models switch -> high noise sampler
LOW_MODELS_SWITCH = "1214"             # Low Models switch -> low noise sampler
WIDTH_SWITCH = "2673"                  # width switch -> ImageResizeKJv2 nodes
HEIGHT_SWITCH = "2674"                 # height switch -> ImageResizeKJv2 nodes
FRAMES_SWITCH = "2803"                 # frames switch -> I2V encode, empty embeds
IMAGE_INPUT_SWITCH = "3914"            # image input switch -> I2V clip/input switches

# ---------------------------------------------------------------------------
# Mode-specific switch configurations
#
# For each mode, maps switch_node_id -> list of input keys to REMOVE.
# Any Switch selects the first non-null connected input, so removing
# earlier inputs causes later ones to be selected.
# ---------------------------------------------------------------------------

_SWITCH_CONFIGS: Dict[str, Dict[str, List[str]]] = {
    "i2v": {
        EMBEDS_HIGH_SWITCH:  ["any_01", "any_02"],
        EMBEDS_LOW_SWITCH:   ["any_01", "any_02"],
        HIGH_MODELS_SWITCH:  [],
        LOW_MODELS_SWITCH:   [],
        WIDTH_SWITCH:        ["any_01", "any_02"],
        HEIGHT_SWITCH:       ["any_01", "any_02"],
        IMAGE_INPUT_SWITCH:  ["any_01"],
    },
    "t2v": {
        EMBEDS_HIGH_SWITCH:  ["any_01", "any_02", "any_03"],
        EMBEDS_LOW_SWITCH:   ["any_01", "any_02", "any_03"],
        HIGH_MODELS_SWITCH:  ["any_01"],
        LOW_MODELS_SWITCH:   ["any_02"],
        WIDTH_SWITCH:        ["any_01"],
        HEIGHT_SWITCH:       ["any_01"],
        IMAGE_INPUT_SWITCH:  [],
    },
    "t2i": {
        EMBEDS_HIGH_SWITCH:  ["any_01", "any_02", "any_03"],
        EMBEDS_LOW_SWITCH:   ["any_01", "any_02", "any_03"],
        HIGH_MODELS_SWITCH:  ["any_01"],
        LOW_MODELS_SWITCH:   ["any_02"],
        WIDTH_SWITCH:        ["any_01"],
        HEIGHT_SWITCH:       ["any_01"],
        IMAGE_INPUT_SWITCH:  [],
    },
    "s2v": {
        EMBEDS_HIGH_SWITCH:  ["any_01", "any_02"],
        EMBEDS_LOW_SWITCH:   ["any_01", "any_02"],
        HIGH_MODELS_SWITCH:  [],
        LOW_MODELS_SWITCH:   [],
        WIDTH_SWITCH:        ["any_01", "any_02"],
        HEIGHT_SWITCH:       ["any_01", "any_02"],
        IMAGE_INPUT_SWITCH:  ["any_01"],
    },
}

# ---------------------------------------------------------------------------
# Cached workflow
# ---------------------------------------------------------------------------

_cached_api_workflow: Optional[dict] = None


def _get_api_workflow() -> dict:
    """Load and cache the API-format workflow (converted from visual format)."""
    global _cached_api_workflow
    if _cached_api_workflow is None:
        from workflow_converter import load_and_convert
        logger.info("Converting visual workflow to API format: %s", WORKFLOW_JSON)
        _cached_api_workflow = load_and_convert(WORKFLOW_JSON)
        logger.info("Workflow loaded: %d nodes", len(_cached_api_workflow))
    return _cached_api_workflow


def _strip_optional_nodes(workflow: dict) -> None:
    """
    Remove optional utility nodes that may not be installed (e.g. easy cleanGpuUsed).
    Rewire any node that was connected to the removed node's output to use the
    removed node's input instead, so the graph remains valid.

    Also strips nodes by class_type when the extension registers only in the
    frontend (e.g. rgthree Bookmark/Label) or fails to load in API mode.

    After stripping, sanitizes any remaining input that still references a removed
    node (stripped_id -> upstream map) so ComfyUI does not see NodeNotFoundError.
    """
    stripped_to_upstream: Dict[str, List[Any]] = {}  # stripped node id -> [upstream_id, slot]

    # 1) Strip by explicit node ID first (e.g. nodes that cause NodeNotFoundError
    #    when referenced but are pass-throughs or not needed at runtime).
    STRIP_NODE_IDS = ["1905"]
    for nid_str in STRIP_NODE_IDS:
        if nid_str not in workflow:
            continue
        node = workflow[nid_str]
        inputs = node.get("inputs", {})
        upstream = None
        for _k, v in sorted(inputs.items()):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                upstream = [str(v[0]), v[1]]
                break
        if not upstream:
            workflow.pop(nid_str, None)
            logger.info("Stripped node %s (by id): no upstream connection", nid_str)
            continue
        for nid, n in list(workflow.items()):
            if nid == nid_str:
                continue
            nin = n.get("inputs", {})
            for key, val in nin.items():
                if isinstance(val, (list, tuple)) and len(val) >= 2 and str(val[0]) == nid_str:
                    nin[key] = upstream
        stripped_to_upstream[nid_str] = upstream
        workflow.pop(nid_str, None)
        logger.info("Stripped node %s (by id), rewired to %s", nid_str, upstream)

    # Node ID -> class_type for optional nodes we can strip if missing.
    # Rewires consumers to the first upstream connection. Add nodes that are
    # not installed or fail to load (e.g. EG_WXZ_QH, Image Overlay).
    OPTIONAL_NODES = {
        "252": "easy cleanGpuUsed",
        "677": "EG_WXZ_QH",
        "539": "Image Overlay",
        "3648": "Image Overlay",
    }
    for node_id, class_type in OPTIONAL_NODES.items():
        nid_str = str(node_id)
        if nid_str not in workflow:
            continue
        node = workflow[nid_str]
        if node.get("class_type") != class_type:
            continue
        inputs = node.get("inputs", {})
        upstream = None  # [source_node_id, source_slot]
        for _k, v in sorted(inputs.items()):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                upstream = [str(v[0]), v[1]]
                break
        if not upstream:
            workflow.pop(nid_str, None)
            logger.info("Stripped optional node %s (%s): no upstream connection", nid_str, class_type)
            continue
        for nid, n in list(workflow.items()):
            if nid == nid_str:
                continue
            nin = n.get("inputs", {})
            for key, val in nin.items():
                if isinstance(val, (list, tuple)) and len(val) >= 2 and str(val[0]) == nid_str:
                    nin[key] = upstream
        stripped_to_upstream[nid_str] = upstream
        workflow.pop(nid_str, None)
        logger.info("Stripped optional node %s (%s), rewired to %s", nid_str, class_type, upstream)

    # Strip by class_type: nodes that are often frontend-only or from extensions
    # that don't load in headless/API mode (rgthree JS nodes, QHNodes, Impact, KJNodes).
    STRIP_CLASS_TYPES = {
        "Bookmark (rgthree)",
        "Fast Groups Bypasser (rgthree)",
        "Fast Groups Muter (rgthree)",
        "Label (rgthree)",
        "Mute / Bypass Relay (rgthree)",
        "Mute / Bypass Repeater (rgthree)",
        "EG_WXZ_QH",
        "Image Overlay",
        "MarkdownNote",
        "PrimitiveNode",
    }
    while True:
        to_strip = [
            (nid, n) for nid, n in workflow.items()
            if n.get("class_type") in STRIP_CLASS_TYPES
        ]
        if not to_strip:
            break
        stripped_any = False
        for node_id, node in to_strip:
            if node_id not in workflow:
                continue
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            upstream = None
            conn_count = 0
            for _k, v in inputs.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    conn_count += 1
                    if upstream is None:
                        upstream = [str(v[0]), v[1]]
            # Only rewire when at most one connection (pass-through or decorative)
            if conn_count > 1:
                logger.debug(
                    "Skipping strip of %s (%s): has %d connections (node left in workflow)",
                    node_id, class_type, conn_count,
                )
                continue
            if not upstream:
                workflow.pop(node_id, None)
                logger.info("Stripped node %s (%s): no upstream connection", node_id, class_type)
                stripped_any = True
                continue
            for nid, n in list(workflow.items()):
                if nid == node_id:
                    continue
                nin = n.get("inputs", {})
                for key, val in nin.items():
                    if isinstance(val, (list, tuple)) and len(val) >= 2 and str(val[0]) == node_id:
                        nin[key] = upstream
            stripped_to_upstream[node_id] = upstream
            workflow.pop(node_id, None)
            logger.info("Stripped node %s (%s), rewired to %s", node_id, class_type, upstream)
            stripped_any = True
        if not stripped_any:
            break

    # 3) Sanitize: replace any input [ref, slot] that still points to a removed node
    for nid, n in workflow.items():
        nin = n.get("inputs", {})
        for key, val in list(nin.items()):
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                continue
            ref = str(val[0])
            if ref in workflow:
                continue
            if ref in stripped_to_upstream:
                nin[key] = stripped_to_upstream[ref]
                logger.debug("Sanitized %s[%s]: %s -> %s", nid, key, val, stripped_to_upstream[ref])


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------


def _configure_mode(workflow: dict, mode: str) -> None:
    """
    Configure Any Switch node inputs for the selected generation mode.

    For each switch, removes specified input connections so the desired
    input becomes the first non-null and gets selected by the switch.
    """
    if mode not in _SWITCH_CONFIGS:
        raise ValueError(
            f"Unknown mode: {mode!r}. Must be one of: {list(_SWITCH_CONFIGS.keys())}"
        )

    config = _SWITCH_CONFIGS[mode]

    for node_id, inputs_to_remove in config.items():
        if node_id not in workflow:
            continue
        node_inputs = workflow[node_id].get("inputs", {})
        for input_name in inputs_to_remove:
            node_inputs.pop(input_name, None)

    logger.info("Configured mode: %s (modified %d switch nodes)", mode, len(config))


# ---------------------------------------------------------------------------
# Context window configuration
# ---------------------------------------------------------------------------


def _configure_context_windows(
    workflow: dict,
    num_frames: int,
    num_contexts: int,
) -> None:
    """
    Configure frame count and context windows for video duration.

    When num_contexts == 1, uses simple duration mode (node 3797).
    When num_contexts > 1, uses context window mode (nodes 2797 + 2795)
    for single-pass longer video generation.
    """
    if num_contexts > 1:
        # Context window mode: total = frames_per_context * num_contexts
        if FRAMES_PER_CONTEXT_NODE in workflow:
            workflow[FRAMES_PER_CONTEXT_NODE]["inputs"]["value"] = num_frames
        if NUM_CONTEXTS_NODE in workflow:
            workflow[NUM_CONTEXTS_NODE]["inputs"]["value"] = num_contexts

        # Frames switch: keep any_01 (context-calculated total) as first input
        # This is already the default since any_01 is connected to Total frames node
        logger.info(
            "Context window mode: %d frames/context x %d contexts = %d total frames",
            num_frames, num_contexts, num_frames * num_contexts,
        )
    else:
        # Simple duration mode: use direct frame count
        if DURATION_NODE in workflow:
            workflow[DURATION_NODE]["inputs"]["value"] = num_frames

        # Frames switch: remove any_01 so any_02 (duration) becomes first
        if FRAMES_SWITCH in workflow:
            workflow[FRAMES_SWITCH]["inputs"].pop("any_01", None)

        logger.info("Simple duration mode: %d frames", num_frames)


# ---------------------------------------------------------------------------
# Frame interpolation (FL_RIFE) and upscale configuration
# ---------------------------------------------------------------------------


def _configure_interpolation(
    workflow: dict,
    enable: bool,
    multiplier: int,
    rife_ckpt: str,
) -> None:
    """
    Configure FL_RIFE node for frame interpolation.

    When enable=False or multiplier=1, no extra frames are generated.
    When enable=True and multiplier=2, frames are 2x interpolated (e.g. 24->48 fps).
    """
    if RIFE_NODE not in workflow:
        return
    inputs = workflow[RIFE_NODE]["inputs"]
    inputs["ckpt_name"] = rife_ckpt
    inputs["multiplier"] = 2 if (enable and multiplier >= 2) else 1
    inputs["ensemble"] = True
    logger.info(
        "Interpolation: enable=%s, multiplier=%d (FL_RIFE set to %d)",
        enable, multiplier, inputs["multiplier"],
    )


def _configure_upscale(
    workflow: dict,
    enable: bool,
    upscale_model_name: Optional[str],
) -> None:
    """
    Configure upscale path. When enable=False, bypass ImageUpscaleWithModelBatched
    by feeding ImageResizeKJv2 (246) directly from the switch (1977).
    When enable=True, feed 246 from 244 (upscaled output); set model name on 245.
    """
    default_upscale_model = "2x_NMKD-UpgifLiteV2_210k.pth"
    model_name = (upscale_model_name or default_upscale_model).strip() or default_upscale_model

    if UPSCALE_MODEL_LOADER_NODE in workflow:
        workflow[UPSCALE_MODEL_LOADER_NODE]["inputs"]["model_name"] = model_name

    if RESIZE_AFTER_UPSCALE_NODE not in workflow:
        return
    # 246 (ImageResizeKJv2) image input: from 244 (upscaled) or from 1977 (bypass upscale)
    if enable:
        workflow[RESIZE_AFTER_UPSCALE_NODE]["inputs"]["image"] = [UPSCALE_BATCH_NODE, 0]
        logger.info("Upscale: enabled, model=%s", model_name)
    else:
        workflow[RESIZE_AFTER_UPSCALE_NODE]["inputs"]["image"] = [UPSCALE_INPUT_SWITCH_NODE, 0]
        logger.info("Upscale: disabled (bypass), model=%s still loaded for pipeline", model_name)


# ---------------------------------------------------------------------------
# Parameter injection
# ---------------------------------------------------------------------------


def _inject_parameters(
    workflow: dict,
    job_id: str,
    prompt: str,
    mode: str,
    params: Dict[str, Any],
) -> Optional[str]:
    """
    Inject dynamic parameters into the workflow.

    Returns the image filename if an image was saved to ComfyUI input dir,
    or None if no image was used.
    """
    import comfyui_bridge

    negative_prompt = params.get(
        "negative_prompt",
        "bright tones, overexposed, static, blurred details, subtitles, "
        "worst quality, low quality, JPEG artifacts, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
        "disfigured, misshapen limbs, fused fingers, still photo, messy background",
    )
    steps = params.get("steps", 8)
    shift = params.get("shift", 6.9)
    seed = params.get("seed", -1)
    width = params.get("width", 832)
    height = params.get("height", 480)

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # -- Prompt --
    if WILDCARD_NODE in workflow:
        workflow[WILDCARD_NODE]["inputs"]["populated_text"] = prompt

    if TEXT_ENCODE_CACHED_NODE in workflow:
        node_inputs = workflow[TEXT_ENCODE_CACHED_NODE]["inputs"]
        if "positive_prompt" in node_inputs or isinstance(node_inputs.get("positive_prompt"), str):
            node_inputs["positive_prompt"] = prompt
        if "negative_prompt" in node_inputs or isinstance(node_inputs.get("negative_prompt"), str):
            node_inputs["negative_prompt"] = negative_prompt

    # -- Seed --
    if SEED_NODE in workflow:
        workflow[SEED_NODE]["inputs"]["seed"] = seed

    # -- Steps --
    if STEPS_NODE in workflow:
        workflow[STEPS_NODE]["inputs"]["value"] = steps

    # -- Shift --
    if SHIFT_NODE in workflow:
        workflow[SHIFT_NODE]["inputs"]["value"] = shift

    # -- Dimensions (for T2V/T2I manual mode) --
    if mode in ("t2v", "t2i"):
        # Set dimensions on sampler/encode nodes or width/height if available
        # Width/height switches route to any_02 (Crop node 2671) for T2V
        # We need to set these dimensions where the workflow expects them
        if WIDTH_SWITCH in workflow:
            workflow[WIDTH_SWITCH]["inputs"]["any_02"] = width
        if HEIGHT_SWITCH in workflow:
            workflow[HEIGHT_SWITCH]["inputs"]["any_02"] = height

    # -- Reference image (I2V / S2V) --
    image_filename = None
    if mode in ("i2v", "s2v"):
        init_image_b64 = params.get("init_image", "")
        if not init_image_b64:
            raise ValueError(f"init_image (base64) is required for mode={mode}")

        image_bytes = base64.b64decode(init_image_b64)
        image_filename = f"job_{job_id}_{uuid.uuid4().hex[:8]}.png"
        comfyui_bridge.save_image_to_input(image_bytes, image_filename)

        if LOAD_IMAGE_NODE in workflow:
            workflow[LOAD_IMAGE_NODE]["inputs"]["image"] = image_filename

    # -- Output VHS node --
    if OUTPUT_VHS_NODE in workflow:
        node_inputs = workflow[OUTPUT_VHS_NODE]["inputs"]
        node_inputs["save_output"] = True
        node_inputs["filename_prefix"] = f"wan22_{job_id}"
        if "format" in node_inputs:
            node_inputs["format"] = "video/h264-mp4"

    # Disable preview node output
    if PREVIEW_VHS_NODE in workflow:
        workflow[PREVIEW_VHS_NODE]["inputs"]["save_output"] = False

    # -- Attention mode override (SageAttention fallback) --
    for model_node_id in [I2V_HIGH_MODEL_NODE, I2V_LOW_MODEL_NODE,
                          T2V_HIGH_MODEL_NODE, T2V_LOW_MODEL_NODE]:
        if model_node_id not in workflow:
            continue
        try:
            import importlib
            importlib.import_module("sageattention")
        except ImportError:
            current_attn = workflow[model_node_id]["inputs"].get("attention_mode", "")
            if isinstance(current_attn, str) and "sage" in current_attn.lower():
                logger.warning(
                    "[%s] SageAttention not installed, overriding attention_mode to 'sdpa' on node %s",
                    job_id, model_node_id,
                )
                workflow[model_node_id]["inputs"]["attention_mode"] = "sdpa"

    logger.info(
        "[%s] Parameters injected: mode=%s, seed=%d, steps=%d, shift=%.1f, "
        "frames=%d, image=%s",
        job_id, mode, seed, steps, shift,
        params.get("num_frames", 81),
        image_filename or "none",
    )

    return image_filename


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------


def _enhance_prompt(prompt: str, job_id: str) -> str:
    """Enhance a prompt using Together AI LLM. Returns original on failure."""
    import requests as http_requests

    logger.info("[%s] ENHANCE: calling Together AI to enhance prompt", job_id)

    try:
        resp = http_requests.post(
            TOGETHER_API_URL,
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": TOGETHER_MODEL,
                "messages": [
                    {"role": "system", "content": _ENHANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        match = re.search(
            r"ENHANCED PROMPT:\s*\n?(.*?)(?:\n\s*\n(?:NEGATIVE PROMPT:|VIDEO PARAMETERS:)|$)",
            content, re.DOTALL,
        )
        if match:
            enhanced = match.group(1).strip()
        else:
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            enhanced = cleaned if cleaned else content

        if enhanced:
            logger.info("[%s] ENHANCE: success (%d chars)", job_id, len(enhanced))
            return enhanced

    except Exception as e:
        logger.warning("[%s] ENHANCE: failed: %s", job_id, e, exc_info=True)

    return prompt


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------


def _find_output_video(
    output_files: list,
    job_id: str,
    output_dir: str,
) -> str:
    """
    Find the output video from ComfyUI and copy it to the output directory.

    Looks for the output from node 249 (Save Interpoled) first, then falls
    back to any mp4/webm file in the outputs.
    """
    import comfyui_bridge

    if not output_files:
        raise RuntimeError(f"[{job_id}] ComfyUI produced no output files")

    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, f"{job_id}.mp4")

    # Prefer output from the Save Interpoled node
    for f in output_files:
        if f.get("node_id") == OUTPUT_VHS_NODE:
            path = comfyui_bridge.get_output_file_path(f)
            if os.path.exists(path):
                shutil.copy2(path, dest_path)
                logger.info("[%s] Output from node %s: %s -> %s", job_id, OUTPUT_VHS_NODE, path, dest_path)
                _safe_remove(path, job_id)
                return dest_path

    # Fallback: any video file
    for f in output_files:
        fname = f.get("filename", "")
        if fname.endswith((".mp4", ".webm")):
            path = comfyui_bridge.get_output_file_path(f)
            if os.path.exists(path):
                shutil.copy2(path, dest_path)
                logger.info("[%s] Fallback output: %s -> %s", job_id, path, dest_path)
                _safe_remove(path, job_id)
                return dest_path

    raise FileNotFoundError(f"[{job_id}] No video output found in ComfyUI outputs")


def _safe_remove(path: str, job_id: str) -> None:
    """Remove a file, logging any errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("[%s] Failed to clean up %s: %s", job_id, path, e)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def run_inference_comfyui(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
) -> str:
    """
    Run the Wan 2.2 ComfyUI workflow with injected parameters.

    Parameters
    ----------
    job_id : str
        Unique job identifier.
    prompt : str
        Text prompt for generation.
    params : dict
        Additional parameters:
        - ``mode`` (str): Generation mode - "i2v", "t2v", "t2i", "s2v"
          (default "i2v")
        - ``init_image`` (str): Base64-encoded reference image (required for
          i2v and s2v modes)
        - ``num_frames`` (int): Frames per context window or simple duration
          (default 81, ~5s at 16fps)
        - ``num_contexts`` (int): Number of context windows for longer videos
          (default 1; set > 1 for longer videos)
        - ``steps`` (int): Sampling steps (default 8)
        - ``shift`` (float): Shift value (default 6.9)
        - ``seed`` (int): Random seed (-1 for random)
        - ``width`` (int): Output width for T2V/T2I (default 832)
        - ``height`` (int): Output height for T2V/T2I (default 480)
        - ``negative_prompt`` (str): Negative prompt
        - ``enhance_prompt`` (bool): Enhance prompt via LLM (default False)
        - ``enable_interpolation`` (bool): Use FL_RIFE frame interpolation (default True)
        - ``interpolation_multiplier`` (int): 1 = no extra frames, 2 = 2x frames (default 2)
        - ``rife_ckpt_name`` (str): FL_RIFE checkpoint name, e.g. "rife49" (default "rife49")
        - ``enable_upscale`` (bool): Use workflow upscale before interpolation (default True)
        - ``upscale_model_name`` (str): Upscale model filename, e.g. "2x_NMKD-UpgifLiteV2_210k.pth" (optional)
        - ``output_dir`` (str): Output directory

    Returns
    -------
    str
        Absolute path to the output video file.
    """
    import comfyui_bridge

    start_time = time.time()
    mode = (params.get("mode") or "i2v").lower()
    if mode not in ("i2v", "t2v", "t2i", "s2v"):
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of: i2v, t2v, t2i, s2v")
    num_frames = max(1, min(513, params.get("num_frames", 81)))
    num_contexts = max(1, min(32, params.get("num_contexts", 1)))
    output_dir = params.get("output_dir", "/workspace")

    logger.info("[%s] Starting inference: mode=%s, frames=%d, contexts=%d", job_id, mode, num_frames, num_contexts)

    # Prompt enhancement
    if params.get("enhance_prompt", False):
        prompt = _enhance_prompt(prompt, job_id)

    # 1. Deep-copy the cached workflow
    base_workflow = _get_api_workflow()
    workflow = copy.deepcopy(base_workflow)

    # 2. Strip optional utility nodes (e.g. easy cleanGpuUsed) so workflow runs without them
    _strip_optional_nodes(workflow)

    # 3. Configure mode (switch routing)
    _configure_mode(workflow, mode)

    # 3. Configure context windows / duration
    _configure_context_windows(workflow, num_frames, num_contexts)

    # 4. Configure frame interpolation (FL_RIFE) and upscale
    enable_interpolation = params.get("enable_interpolation", True)
    interpolation_multiplier = max(1, min(2, params.get("interpolation_multiplier", 2)))
    rife_ckpt_name = params.get("rife_ckpt_name", "rife49")
    _configure_interpolation(workflow, enable_interpolation, interpolation_multiplier, rife_ckpt_name)

    enable_upscale = params.get("enable_upscale", True)
    upscale_model_name = params.get("upscale_model_name")
    _configure_upscale(workflow, enable_upscale, upscale_model_name)

    # 5. Inject all parameters
    image_filename = _inject_parameters(workflow, job_id, prompt, mode, params)

    logger.info("[%s] Workflow prepared (%d nodes), submitting to ComfyUI...", job_id, len(workflow))

    # 6. Submit to ComfyUI
    submit_time = time.time()
    prompt_id, output_files = comfyui_bridge.run_prompt(workflow)
    comfyui_elapsed = time.time() - submit_time

    logger.info(
        "[%s] ComfyUI completed in %.1fs, %d output files",
        job_id, comfyui_elapsed, len(output_files),
    )

    # 7. Retrieve output video
    video_path = _find_output_video(output_files, job_id, output_dir)

    # 8. Clean up input image
    if image_filename:
        input_path = os.path.join(COMFYUI_DIR, "input", image_filename)
        _safe_remove(input_path, job_id)

    elapsed = time.time() - start_time
    final_size = os.path.getsize(video_path) / 1e6 if os.path.exists(video_path) else 0
    logger.info(
        "[%s] Done: %.1fs total (ComfyUI: %.1fs) | output: %s (%.1f MB)",
        job_id, elapsed, comfyui_elapsed, video_path, final_size,
    )

    return video_path
