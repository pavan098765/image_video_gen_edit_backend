#!/usr/bin/env python3
"""
comfyui_inference.py
====================

Main inference function for the Wan 2.2 I2V ComfyUI worker.

This module:
1. Loads and caches the pre-converted API workflow
2. Deep-copies and injects dynamic parameters per job
3. Submits the workflow to the ComfyUI server via comfyui_bridge
4. Waits for completion and returns the output video path

Supports:
- Single prompt mode (all segments use the same prompt)
- Multi-prompt mode (list of prompts, one per SVI segment)
- Dynamic width/height, num_frames, steps, cfg, seed
- Reference image injection
"""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import random
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("comfyui_inference")

# ---------------------------------------------------------------------------
# RIFE availability
# ---------------------------------------------------------------------------

RIFE_AVAILABLE = False
_rife_model_cache = None

try:
    from video_postprocessing import (
        load_rife_model,
        apply_rife_interpolation,
        ssim_interpolation_rife,
        check_rife_code_available,
        RIFE_AVAILABLE as _VP_RIFE_AVAILABLE,
    )
    RIFE_AVAILABLE = _VP_RIFE_AVAILABLE
    if RIFE_AVAILABLE:
        logger.info("RIFE frame interpolation is available")
    else:
        logger.warning("RIFE code loaded but model import failed")
except ImportError as e:
    logger.warning("video_postprocessing not available: %s", e)

# Default RIFE weights directory
RIFE_WEIGHTS_DIR = os.environ.get(
    "RIFE_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.environ.get("COMFYUI_DIR", "/app/comfyui")), "rife_weights"),
)

# Together AI config for prompt generation
TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "d9d8a62fa13d36b44cbb16188734e5dc17eec3fd65c2ebc4e0d533bd9bc6c33f",
)
TOGETHER_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

_ENHANCE_SYSTEM_PROMPT_VIDEO_I2V = (
    "You are an expert cinematic prompt engineer specialized in enhancing prompts for AI image-to-video generation models.\n\n"
    "Your task is NOT to change the user's intent, but to expand and refine it into a highly detailed, visually descriptive, and generation-optimized prompt.\n\n"
    "Follow these principles:\n\n"
    "1. Preserve Core Intent\n- Never alter the main subject or meaning.\n- Only enhance clarity, detail, and visual richness.\n"
    "- The input image is the primary visual reference; the prompt should describe desired motion and changes.\n\n"
    "2. Add Missing Visual Information\nEnhance prompts with:\n- Subject description (appearance, clothing, age, materials, textures)\n"
    "- Environment and background\n- Lighting (cinematic lighting, volumetric light, rim light, neon glow, etc.)\n"
    "- Camera details (lens, depth of field, perspective, framing)\n- Mood and atmosphere\n- Color palette\n- Composition\n"
    "- Motion (for video prompts)\n- Time of day / weather\n- Style references if appropriate\n\n"
    "3. Cinematic & Photorealistic Quality Boost\nWhen suitable, include:\n- ultra detailed\n- cinematic\n- photorealistic\n"
    "- high dynamic range\n- global illumination\n- 8k / high resolution\n- shallow depth of field\n"
    "- film grain\n- realistic shadows\n\n"
    "4. Video Prompt Enhancement\nAdd:\n- camera movement (slow pan, dolly, handheld, drone, tracking shot)\n"
    "- subject motion\n- environmental motion\n- temporal continuity\n- physics realism\n- cinematic pacing\n\n"
    "5. Editing Requests\nClearly describe:\n- what remains unchanged from the input image\n- what is modified or animated\n"
    "- integration realism\n- lighting consistency\n- perspective matching\n\n"
    "6. Avoid Negative Behavior\nDo NOT:\n- add unrelated objects\n- change gender, ethnicity, or identity unless asked\n"
    "- introduce unwanted elements\n- overcomplicate simple prompts\n\n"
    "7. Output Format Rules\nAlways output:\n\nENHANCED PROMPT:\n<final enhanced prompt>\n\n"
    "Optional sections when useful:\n\nNEGATIVE PROMPT:\n<artifacts to avoid>\n\n"
    "VIDEO PARAMETERS:\n<only if video related>\n\n"
    "Do NOT explain reasoning.\nDo NOT include commentary.\nOnly output the final structured result.\n\n"
    "8. Style Intelligence\nAutomatically infer best style based on user intent.\n\n"
    "9. Model Awareness\nOptimize prompts to work well with video generation models such as:\nWAN, Hunyuan, LTX, Sora-style pipelines.\n\n"
    "10. Precision Over Length\nBe descriptive but meaningful. Avoid useless adjectives.\n\n"
    "Your goal: produce prompts that generate visually stunning, coherent, and realistic videos from images with minimal artifacts."
)


def _enhance_prompt(prompt: str, job_id: str) -> str:
    """
    Enhance a prompt using Together AI LLM for video I2V generation.

    Returns enhanced prompt, or original prompt on failure.
    """
    import requests as http_requests

    logger.info("[%s] ENHANCE: calling Together AI (%s) to enhance prompt (type=video_i2v)", job_id, TOGETHER_MODEL)
    logger.info("[%s] ENHANCE: original prompt = '%s'", job_id, prompt[:200] + ("..." if len(prompt) > 200 else ""))

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
                    {"role": "system", "content": _ENHANCE_SYSTEM_PROMPT_VIDEO_I2V},
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
        logger.info("[%s] ENHANCE: raw response (%d chars): %s", job_id, len(content), content[:300])

        match = re.search(r"ENHANCED PROMPT:\s*\n?(.*?)(?:\n\s*\n(?:NEGATIVE PROMPT:|VIDEO PARAMETERS:|EDIT INSTRUCTIONS:)|$)", content, re.DOTALL)
        if match:
            enhanced = match.group(1).strip()
        else:
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            enhanced = cleaned if cleaned else content

        if enhanced:
            logger.info("[%s] ENHANCE: enhanced prompt = '%s'", job_id, enhanced[:200] + ("..." if len(enhanced) > 200 else ""))
            return enhanced

    except Exception as e:
        logger.warning("[%s] ENHANCE: prompt enhancement failed: %s", job_id, e, exc_info=True)

    logger.info("[%s] ENHANCE: falling back to original prompt", job_id)
    return prompt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/app/comfyui")
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKFLOW_JSON = os.path.join(WORKFLOW_DIR, "wan_22_i2v_svi.json")

# ---------------------------------------------------------------------------
# Cached workflow
# ---------------------------------------------------------------------------

_cached_api_workflow: Optional[dict] = None


def _get_api_workflow() -> dict:
    """
    Load and cache the API-format workflow.

    On first call, converts the visual workflow to API format.
    Subsequent calls return the cached version.
    """
    global _cached_api_workflow
    if _cached_api_workflow is None:
        from workflow_converter import load_and_convert
        logger.info("Converting visual workflow to API format: %s", WORKFLOW_JSON)
        _cached_api_workflow = load_and_convert(WORKFLOW_JSON)
        logger.info("Workflow loaded: %d nodes", len(_cached_api_workflow))
    return _cached_api_workflow


# ---------------------------------------------------------------------------
# Node ID mappings (from workflow analysis)
# ---------------------------------------------------------------------------

# The workflow has 10 SVI segments, each with its own text encode + sampler
# Segment order: 1st segment generates from the reference image, subsequent
# segments chain from the previous segment's last frame.

# LoadImage node (reference image)
LOAD_IMAGE_NODE = "58"

# ImageResizeKJv2 (controls output width/height)
IMAGE_RESIZE_NODE = "68"

# INTConstant for num_frames per segment
NUM_FRAMES_NODE = "150"

# INTConstant for overlap frames between segments
OVERLAP_FRAMES_NODE = "188"

# INTConstant for seed
SEED_NODE = "530"

# WanVideoTextEncode nodes (one per segment, in segment order)
# Segment 1 uses the reference image, segments 2-10 chain from previous
TEXT_ENCODE_NODES = [
    "16",   # Segment 1
    "116",  # Segment 2
    "543",  # Segment 3
    "580",  # Segment 4
    "583",  # Segment 5
    "628",  # Segment 6
    "651",  # Segment 7
    "652",  # Segment 8
    "697",  # Segment 9
    "698",  # Segment 10
]

# WanVideoSampler nodes (one per segment)
SAMPLER_NODES = [
    "27",   # Segment 1
    "113",  # Segment 2
    "542",  # Segment 3
    "566",  # Segment 4
    "590",  # Segment 5
    "612",  # Segment 6
    "635",  # Segment 7
    "659",  # Segment 8
    "681",  # Segment 9
    "705",  # Segment 10
]

# WanVideoImageToVideoEncode nodes (one per segment)
I2V_ENCODE_NODES = [
    "63",   # Segment 1
    "117",  # Segment 2
    "539",  # Segment 3
    "564",  # Segment 4
    "588",  # Segment 5
    "610",  # Segment 6
    "633",  # Segment 7
    "657",  # Segment 8
    "679",  # Segment 9
    "703",  # Segment 10
]

# WanVideoDecode nodes (one per segment)
DECODE_NODES = [
    "28",   # Segment 1
    "110",  # Segment 2
    "546",  # Segment 3
    "569",  # Segment 4
    "593",  # Segment 5
    "615",  # Segment 6
    "638",  # Segment 7
    "662",  # Segment 8
    "684",  # Segment 9
    "708",  # Segment 10
]

# VHS_VideoCombine nodes
# Per-segment output nodes (10 individual) + 1 final SVI-film combined
PER_SEGMENT_VHS_NODES = [
    "30",   # Segment 1
    "120",  # Segment 2
    "547",  # Segment 3
    "581",  # Segment 4
    "604",  # Segment 5
    "626",  # Segment 6
    "649",  # Segment 7
    "673",  # Segment 8
    "695",  # Segment 9
    "719",  # Segment 10
]
FINAL_VHS_NODE = "721"  # Final SVI-film combined output

# WanVideoModelLoader node
MODEL_LOADER_NODE = "22"

# WanVideoTorchCompileSettings node
TORCH_COMPILE_NODE = "35"

# Number of segments in the SVI workflow
NUM_SEGMENTS = 10

# All per-segment node lists grouped for easy segment trimming
_PER_SEGMENT_NODE_LISTS = [
    TEXT_ENCODE_NODES,
    SAMPLER_NODES,
    I2V_ENCODE_NODES,
    DECODE_NODES,
    PER_SEGMENT_VHS_NODES,
]

# Additional nodes that are known to be part of the inter-segment chain.
# These are WanVideoClipVisionEncode, GetImageRangeFromBatch, ImageBatchMulti,
# VRAM_Debug, etc. that connect segment N-1's output to segment N's input.
# We rely on cascade-deletion to remove them automatically.


# ---------------------------------------------------------------------------
# Segment trimming
# ---------------------------------------------------------------------------


def _trim_to_segments(workflow: dict, num_segments: int) -> List[str]:
    """
    Remove nodes for segments beyond *num_segments* from the workflow.

    This also cascade-deletes any node whose inputs reference a removed node,
    and removes the final SVI-film combining node (node 721 + its upstream
    combining chain) since it expects all 10 segments.

    All active per-segment VHS_VideoCombine nodes are enabled for output.

    Parameters
    ----------
    workflow : dict
        The API-format workflow (mutated in place).
    num_segments : int
        Number of segments to keep (1-10).

    Returns
    -------
    List[str]
        Ordered list of active per-segment VHS node IDs that will produce
        output.  For ``num_segments == 10`` this returns
        ``[FINAL_VHS_NODE]`` (the combiner handles everything).
    """
    if num_segments >= NUM_SEGMENTS:
        return [FINAL_VHS_NODE]  # Nothing to trim; combiner handles output

    # 1. Collect node IDs to remove: segments (num_segments+1) .. 10
    nodes_to_remove: set = set()

    for node_list in _PER_SEGMENT_NODE_LISTS:
        for idx in range(num_segments, NUM_SEGMENTS):
            nodes_to_remove.add(node_list[idx])

    # 2. Also remove the final SVI-film combining node and everything
    #    upstream that only serves the combiner.  The final VHS node (721)
    #    takes its input from combining/batching nodes that merge all 10
    #    segment outputs.  With fewer segments these would have broken refs.
    nodes_to_remove.add(FINAL_VHS_NODE)

    # 3. Cascade-delete: repeatedly remove any node whose inputs reference
    #    a node that has already been removed.
    changed = True
    while changed:
        changed = False
        for node_id in list(workflow.keys()):
            if node_id in nodes_to_remove:
                continue
            inputs = workflow[node_id].get("inputs", {})
            for val in inputs.values():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    ref_id = val[0]
                    if ref_id in nodes_to_remove:
                        nodes_to_remove.add(node_id)
                        changed = True
                        break

    # 4. Remove the nodes from the workflow
    removed_count = 0
    for nid in nodes_to_remove:
        if nid in workflow:
            del workflow[nid]
            removed_count += 1

    logger.info(
        "Trimmed workflow to %d segment(s): removed %d nodes, %d remaining",
        num_segments, removed_count, len(workflow),
    )

    # 5. Enable save_output on ALL active per-segment VHS nodes
    active_vhs_nodes: List[str] = []
    for i in range(num_segments):
        nid = PER_SEGMENT_VHS_NODES[i]
        if nid in workflow:
            workflow[nid]["inputs"]["save_output"] = True
            workflow[nid]["inputs"]["filename_prefix"] = f"wan22_seg{i+1}"
            active_vhs_nodes.append(nid)

    return active_vhs_nodes


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def run_inference_comfyui(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
) -> str:
    """
    Run the Wan 2.2 I2V ComfyUI workflow with injected parameters.

    Parameters
    ----------
    job_id : str
        Unique job identifier.
    prompt : str
        Text prompt for video generation.  If ``params["prompts"]`` is
        provided (list of strings), each segment gets its own prompt.
    params : dict
        Additional parameters:
        - ``init_image`` (str): Base64-encoded reference image (required)
        - ``width`` (int): Output width (default 832)
        - ``height`` (int): Output height (default 480)
        - ``num_frames`` (int): Frames per segment (default 81)
        - ``overlap_frames`` (int): Overlap frames between segments (default 5)
        - ``steps`` (int): Sampling steps (default 8)
        - ``cfg`` (float): CFG scale (default 1.0)
        - ``seed`` (int): Random seed (-1 for random, default -1)
        - ``negative_prompt`` (str): Negative prompt
        - ``prompts`` (list[str]): Per-segment prompts (overrides ``prompt``)
        - ``num_segments`` (int): How many SVI segments to run (1-10,
          default 10).  Use 1-2 for fast testing.
        - ``output_dir`` (str): Output directory
        - ``enable_rife`` (bool): Enable RIFE frame interpolation (default True)
        - ``rife_target_fps`` (int): Target FPS after RIFE (default 30)
        - ``auto_prompts`` (bool): Auto-generate per-segment prompts via LLM
          when only one prompt is provided (default True)

    Returns
    -------
    str
        Absolute path to the output video file.
    """
    import comfyui_bridge
    import base64

    start_time = time.time()
    logger.info("[%s] Starting ComfyUI inference", job_id)

    # 1. Get the cached API workflow and deep-copy it
    base_workflow = _get_api_workflow()
    workflow = copy.deepcopy(base_workflow)

    # 2. Extract parameters
    width = params.get("width", 832)
    height = params.get("height", 480)
    num_frames = params.get("num_frames", 81)
    overlap_frames = params.get("overlap_frames", 5)
    steps = params.get("steps", 8)
    cfg = params.get("cfg", 1.0)
    seed = params.get("seed", -1)
    negative_prompt = params.get(
        "negative_prompt",
        "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still photo, messy background"
    )
    output_dir = params.get("output_dir", "/workspace")
    enable_rife = params.get("enable_rife", True)
    rife_target_fps = params.get("rife_target_fps", 30)
    auto_prompts = params.get("auto_prompts", True)

    # Number of SVI segments (1-10). Default 10 for production, use fewer for testing.
    active_segments = max(1, min(NUM_SEGMENTS, params.get("num_segments", NUM_SEGMENTS)))

    # Prompt enhancement (enhance base prompt before segment generation)
    if params.get("enhance_prompt", False):
        prompt = _enhance_prompt(prompt, job_id)

    # Handle prompts - single or multi-segment
    prompts: List[str] = params.get("prompts", [])
    if not prompts:
        if active_segments > 1 and auto_prompts:
            # Auto-generate per-segment prompts via LLM
            logger.info("[%s] PROMPT: auto-generating %d segment prompts via Together AI LLM...", job_id, active_segments)
            prompts = _generate_segment_prompts(prompt, active_segments, job_id)
        else:
            # Single prompt mode - use same prompt for all active segments
            logger.info("[%s] PROMPT: using same prompt for all %d segment(s)", job_id, active_segments)
            prompts = [prompt] * active_segments
    elif len(prompts) < active_segments:
        # Pad with the last prompt if fewer than needed
        logger.info("[%s] PROMPT: user provided %d prompts, padding to %d", job_id, len(prompts), active_segments)
        prompts = prompts + [prompts[-1]] * (active_segments - len(prompts))
    else:
        logger.info("[%s] PROMPT: user provided %d per-segment prompts", job_id, len(prompts))

    # Log all segment prompts
    for i, p in enumerate(prompts):
        logger.info("[%s] PROMPT seg %d: %s", job_id, i + 1, p[:120] + ("..." if len(p) > 120 else ""))

    # Generate seed if random
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    logger.info(
        "[%s] === PARAMS: %dx%d, %d frames/seg, %d segment(s), %d steps, cfg=%.1f, seed=%d ===",
        job_id, width, height, num_frames, active_segments, steps, cfg, seed,
    )
    logger.info(
        "[%s] RIFE: enabled=%s, target_fps=%d, available=%s, weights_dir=%s (exists=%s)",
        job_id, enable_rife, rife_target_fps, RIFE_AVAILABLE, RIFE_WEIGHTS_DIR,
        os.path.exists(RIFE_WEIGHTS_DIR),
    )
    logger.info(
        "[%s] Auto-prompts: enabled=%s, overlap_frames=%d",
        job_id, auto_prompts, overlap_frames,
    )

    # 3. Save reference image to ComfyUI input directory
    init_image_b64 = params.get("init_image", "")
    if not init_image_b64:
        raise ValueError("init_image (base64) is required")

    image_bytes = base64.b64decode(init_image_b64)
    image_filename = f"job_{job_id}_{uuid.uuid4().hex[:8]}.png"
    comfyui_bridge.save_image_to_input(image_bytes, image_filename)

    # 4. Trim workflow to active_segments (removes unused segment nodes)
    active_vhs_nodes = _trim_to_segments(workflow, active_segments)

    # 5. Inject parameters into workflow

    # -- LoadImage node: set reference image filename --
    if LOAD_IMAGE_NODE in workflow:
        workflow[LOAD_IMAGE_NODE]["inputs"]["image"] = image_filename

    # -- ImageResizeKJv2: set output dimensions --
    #    Use "pad" (letterbox) instead of "crop" to preserve full image content
    if IMAGE_RESIZE_NODE in workflow:
        workflow[IMAGE_RESIZE_NODE]["inputs"]["width"] = width
        workflow[IMAGE_RESIZE_NODE]["inputs"]["height"] = height
        workflow[IMAGE_RESIZE_NODE]["inputs"]["keep_proportion"] = "pad"

    # -- INTConstant: num_frames per segment --
    if NUM_FRAMES_NODE in workflow:
        workflow[NUM_FRAMES_NODE]["inputs"]["value"] = num_frames

    # -- INTConstant: overlap frames --
    if OVERLAP_FRAMES_NODE in workflow:
        workflow[OVERLAP_FRAMES_NODE]["inputs"]["value"] = overlap_frames

    # -- INTConstant: seed --
    if SEED_NODE in workflow:
        workflow[SEED_NODE]["inputs"]["value"] = seed

    # -- WanVideoTextEncode: inject prompts (only active segments) --
    for i in range(active_segments):
        node_id = TEXT_ENCODE_NODES[i]
        if node_id in workflow:
            workflow[node_id]["inputs"]["positive_prompt"] = prompts[i]
            workflow[node_id]["inputs"]["negative_prompt"] = negative_prompt

    # -- WanVideoSampler: set steps and cfg (only active segments) --
    for i in range(active_segments):
        node_id = SAMPLER_NODES[i]
        if node_id in workflow:
            workflow[node_id]["inputs"]["steps"] = steps
            workflow[node_id]["inputs"]["cfg"] = cfg

    # -- WanVideoModelLoader: force attention_mode to sdpa if sageattention is
    #    not installed (the workflow default is sageattn_compiled) --
    if MODEL_LOADER_NODE in workflow:
        try:
            import importlib
            importlib.import_module("sageattention")
            logger.info("[%s] SageAttention available, keeping workflow attention_mode", job_id)
        except ImportError:
            current_attn = workflow[MODEL_LOADER_NODE]["inputs"].get("attention_mode", "")
            if "sage" in current_attn.lower():
                logger.warning(
                    "[%s] SageAttention not installed, overriding attention_mode from '%s' to 'sdpa'",
                    job_id, current_attn,
                )
                workflow[MODEL_LOADER_NODE]["inputs"]["attention_mode"] = "sdpa"

    # -- Disable torch.compile if the GPU doesn't support fp8e4nv with Triton --
    #    The WanVideoTorchCompileSettings (node 35) feeds compile_args into the
    #    model loader.  On GPUs older than Ada Lovelace (sm_89+) this causes:
    #    "type fp8e4nv not supported in this architecture"
    #    Removing the compile_args connection disables torch.compile entirely.
    if MODEL_LOADER_NODE in workflow:
        inputs = workflow[MODEL_LOADER_NODE]["inputs"]
        if "compile_args" in inputs:
            try:
                import torch
                cap = torch.cuda.get_device_capability()
                if cap < (8, 9):
                    logger.warning(
                        "[%s] GPU compute capability %s < 8.9 (Ada), "
                        "disabling torch.compile (fp8e4nv unsupported)",
                        job_id, f"{cap[0]}.{cap[1]}",
                    )
                    del inputs["compile_args"]
                    # Also remove the TorchCompileSettings node itself
                    workflow.pop(TORCH_COMPILE_NODE, None)
                else:
                    logger.info("[%s] GPU sm_%d%d supports fp8e4nv, keeping torch.compile", job_id, *cap)
            except Exception as e:
                logger.warning("[%s] Could not check GPU capability, disabling torch.compile: %s", job_id, e)
                del inputs["compile_args"]
                workflow.pop(TORCH_COMPILE_NODE, None)

    # -- VHS_VideoCombine output handling --
    #    active_vhs_nodes: list of VHS node IDs whose output we need.
    #    For full 10-segment: [FINAL_VHS_NODE] (the combiner).
    #    For trimmed: per-segment VHS nodes that we'll concatenate later.
    active_vhs_set = set(active_vhs_nodes)

    # Set filename prefixes with job_id for active output nodes
    for node_id in active_vhs_nodes:
        if node_id in workflow:
            workflow[node_id]["inputs"]["filename_prefix"] = f"wan22_svi_{job_id}"

    # Disable save_output on per-segment VHS nodes that are NOT active
    for node_id in PER_SEGMENT_VHS_NODES:
        if node_id in workflow and node_id not in active_vhs_set:
            workflow[node_id]["inputs"]["save_output"] = False

    # Also disable save_output on FINAL_VHS_NODE if it's not in our active set
    if FINAL_VHS_NODE in workflow and FINAL_VHS_NODE not in active_vhs_set:
        workflow[FINAL_VHS_NODE]["inputs"]["save_output"] = False

    logger.info("[%s] Workflow prepared (%d nodes), active VHS nodes: %s", job_id, len(workflow), active_vhs_nodes)
    logger.info("[%s] Submitting workflow to ComfyUI...", job_id)

    # 6. Submit to ComfyUI and wait for completion
    submit_time = time.time()
    prompt_id, output_files = comfyui_bridge.run_prompt(workflow)
    comfyui_elapsed = time.time() - submit_time

    logger.info(
        "[%s] ComfyUI prompt %s completed in %.1fs, %d output files: %s",
        job_id, prompt_id, comfyui_elapsed, len(output_files),
        [f.get("filename", "?") for f in output_files[:10]],
    )

    # 7. Find the final output video (concatenate segments if needed, RIFE if enabled)
    logger.info("[%s] === POST-PROCESSING (RIFE=%s, target_fps=%d) ===", job_id, enable_rife, rife_target_fps)
    postproc_time = time.time()
    video_path = _find_output_video(
        output_files, job_id, output_dir,
        active_vhs_nodes=active_vhs_nodes,
        overlap_frames=overlap_frames,
        frame_rate=16,
        enable_rife=enable_rife,
        rife_target_fps=rife_target_fps,
    )
    postproc_elapsed = time.time() - postproc_time

    elapsed = time.time() - start_time
    final_size = os.path.getsize(video_path) / 1e6 if os.path.exists(video_path) else 0
    logger.info(
        "[%s] === DONE: %.1fs total (ComfyUI: %.1fs, post-process: %.1fs) | output: %s (%.1f MB) ===",
        job_id, elapsed, comfyui_elapsed, postproc_elapsed, video_path, final_size,
    )

    # 8. Clean up input image
    try:
        input_path = os.path.join(COMFYUI_DIR, "input", image_filename)
        if os.path.exists(input_path):
            os.remove(input_path)
    except Exception as e:
        logger.warning("[%s] Failed to clean up input image: %s", job_id, e)

    return video_path


# ---------------------------------------------------------------------------
# Auto-prompt generation for multi-segment videos
# ---------------------------------------------------------------------------


def _generate_segment_prompts(
    base_prompt: str,
    num_segments: int,
    job_id: str,
) -> List[str]:
    """
    Use Together AI LLM to expand a single base prompt into N temporally
    coherent segment prompts for multi-segment video generation.

    Falls back to repeating the base prompt if the API call fails.

    Parameters
    ----------
    base_prompt : str
        The user's original prompt.
    num_segments : int
        Number of segment prompts to generate.
    job_id : str
        Job ID for logging.

    Returns
    -------
    List[str]
        List of N prompts, one per segment.
    """
    if num_segments <= 1:
        logger.info("[%s] LLM: single segment, skipping prompt generation", job_id)
        return [base_prompt]

    import requests as http_requests

    logger.info(
        "[%s] LLM: calling Together AI (%s) to generate %d segment prompts",
        job_id, TOGETHER_MODEL, num_segments,
    )
    logger.info("[%s] LLM: base prompt = '%s'", job_id, base_prompt[:150] + ("..." if len(base_prompt) > 150 else ""))
    logger.info("[%s] LLM: API URL = %s, API key = %s...%s", job_id, TOGETHER_API_URL, TOGETHER_API_KEY[:8], TOGETHER_API_KEY[-4:])

    system_prompt = (
        "You are a video prompt expansion assistant. The user provides a single "
        "text prompt describing a video scene. Your job is to expand it into "
        f"exactly {num_segments} sequential video segment descriptions that together "
        "form a smooth, temporally coherent video.\n\n"
        "Rules:\n"
        "- Each segment should describe 3-5 seconds of video.\n"
        "- Maintain visual continuity — same scene, characters, and setting across segments.\n"
        "- Gradually evolve the action, camera angle, or lighting across segments.\n"
        "- Keep each prompt concise (1-3 sentences).\n"
        "- Output ONLY a JSON array of strings, no other text.\n"
        f"- The array must have exactly {num_segments} elements.\n"
        "- Do NOT include segment numbers or labels.\n\n"
        "Example output for 3 segments:\n"
        '["A cat sits calmly on a windowsill, sunlight streaming in.", '
        '"The cat stretches slowly, its tail curling as it watches a bird outside.", '
        '"The cat leaps gracefully from the windowsill, landing softly on the floor."]'
    )

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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("[%s] LLM: HTTP %d, response keys: %s", job_id, resp.status_code, list(data.keys()))

        content = data["choices"][0]["message"]["content"].strip()
        logger.info("[%s] LLM: raw response (%d chars): %s", job_id, len(content), content[:300])

        # Try to extract JSON array from response (handle thinking tags or extra text)
        # Look for the JSON array pattern
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            logger.info("[%s] LLM: extracted JSON array from response", job_id)
            prompts = json.loads(json_match.group())
        else:
            logger.info("[%s] LLM: parsing entire response as JSON", job_id)
            prompts = json.loads(content)

        if isinstance(prompts, list) and len(prompts) >= num_segments:
            prompts = [str(p) for p in prompts[:num_segments]]
            logger.info("[%s] LLM: ✓ generated %d segment prompts successfully", job_id, len(prompts))
            for i, p in enumerate(prompts):
                logger.info("[%s] LLM: seg %d prompt = '%s'", job_id, i + 1, p[:100])
            return prompts
        elif isinstance(prompts, list) and len(prompts) > 0:
            # Pad with last prompt if fewer than needed
            prompts = [str(p) for p in prompts]
            while len(prompts) < num_segments:
                prompts.append(prompts[-1])
            logger.warning(
                "[%s] LLM: returned %d prompts (needed %d), padded with last",
                job_id, len(prompts), num_segments,
            )
            return prompts

    except Exception as e:
        logger.warning("[%s] LLM: prompt generation failed: %s", job_id, e, exc_info=True)

    # Fallback: repeat the base prompt
    logger.info("[%s] Falling back to repeating base prompt for all %d segments", job_id, num_segments)
    return [base_prompt] * num_segments


# ---------------------------------------------------------------------------
# Two-pass RIFE pipeline
# ---------------------------------------------------------------------------


def _get_rife_model(device: str = "cuda"):
    """Load and cache the RIFE model."""
    global _rife_model_cache
    if _rife_model_cache is not None:
        logger.info("RIFE: using cached model")
        return _rife_model_cache

    if not RIFE_AVAILABLE:
        logger.warning("RIFE: not available (import failed at startup)")
        return None

    rife_path = RIFE_WEIGHTS_DIR
    if not os.path.exists(rife_path):
        logger.warning("RIFE: weights directory not found at %s", rife_path)
        return None

    # List contents of weights directory for debugging
    try:
        contents = os.listdir(rife_path)
        logger.info("RIFE: weights dir '%s' contents: %s", rife_path, contents)
    except Exception as e:
        logger.warning("RIFE: cannot list weights dir: %s", e)

    try:
        model = load_rife_model(rife_path)
        if model is not None:
            _rife_model_cache = model
            logger.info("RIFE: ✓ model loaded and cached from %s", rife_path)
        else:
            logger.warning("RIFE: load_rife_model returned None")
        return model
    except Exception as e:
        logger.warning("RIFE: failed to load model: %s", e, exc_info=True)
        return None


def _decode_video_to_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Decode a video file into a list of numpy frames [H, W, 3] uint8
    and return (frames, fps).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _encode_frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """Encode a list of RGB numpy frames [H, W, 3] uint8 to an mp4 video."""
    import cv2

    if not frames:
        raise ValueError("No frames to encode")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()

    # Re-encode with ffmpeg for proper h264 + faststart
    import subprocess
    tmp_path = output_path + ".tmp.mp4"
    os.rename(output_path, tmp_path)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "19", "-preset", "fast",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _rife_interpolate_pair(
    rife_model,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_intermediate: int = 3,
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Use RIFE to generate intermediate frames between frame_a and frame_b.

    Parameters
    ----------
    rife_model : RIFE model instance
    frame_a, frame_b : np.ndarray [H, W, 3] uint8
    num_intermediate : int
        Number of intermediate frames to insert (3 means exp=2, i.e. 2^2-1=3).
    device : str

    Returns
    -------
    List[np.ndarray]
        List of intermediate frames (NOT including frame_a and frame_b).
    """
    import torch

    # Determine exp from num_intermediate: 2^exp - 1 = num_intermediate
    # exp=1 -> 1 frame, exp=2 -> 3 frames, exp=3 -> 7 frames
    exp = max(1, int(np.log2(num_intermediate + 1)))

    # Convert frames to tensor [1, C, H, W] float32 in [0, 1]
    fa = torch.from_numpy(frame_a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    fb = torch.from_numpy(frame_b).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Stack as [2, C, H, W] — the format ssim_interpolation_rife expects
    tensor = torch.cat([fa, fb], dim=0).to(device)

    # Run RIFE interpolation
    result = ssim_interpolation_rife(
        rife_model, tensor, exp=exp, upscale_amount=1, output_device="cpu"
    )

    # result is [N, C, H, W] float tensor including start and end frames
    # Extract only the intermediate frames (exclude first and last)
    intermediate = []
    for i in range(1, result.shape[0] - 1):
        frame_np = (result[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        intermediate.append(frame_np)

    return intermediate


def _rife_stitch_and_interpolate(
    segment_paths: List[str],
    output_path: str,
    overlap_frames: int,
    frame_rate: int,
    job_id: str,
    rife_model,
    device: str = "cuda",
    target_fps: int = 30,
) -> None:
    """
    Two-pass RIFE pipeline for multi-segment video stitching.

    Pass 1: Decode segments, trim overlap, insert RIFE-interpolated transition
             frames at segment boundaries (output stays at original frame_rate).
    Pass 2: RIFE-interpolate entire concatenated video from frame_rate to ~target_fps.
    Re-encode final result as mp4 at target_fps.

    Parameters
    ----------
    segment_paths : List[str]
        Paths to per-segment mp4 files, in order.
    output_path : str
        Where to write the final stitched video.
    overlap_frames : int
        Number of frames that overlap between adjacent segments.
    frame_rate : int
        Original video frame rate (typically 16 for Wan).
    job_id : str
        For logging.
    rife_model
        Loaded RIFE model instance.
    device : str
        CUDA device.
    target_fps : int
        Desired output FPS (default 30).
    """
    import torch

    n = len(segment_paths)
    logger.info(
        "[%s] RIFE two-pass pipeline: %d segments, overlap=%d, %dfps -> %dfps",
        job_id, n, overlap_frames, frame_rate, target_fps,
    )

    # ---- Decode all segments into frame lists ----
    all_segment_frames: List[List[np.ndarray]] = []
    for i, path in enumerate(segment_paths):
        frames, detected_fps = _decode_video_to_frames(path)
        logger.info(
            "[%s] Segment %d: %d frames @ %.1f fps",
            job_id, i + 1, len(frames), detected_fps,
        )
        all_segment_frames.append(frames)

    # ---- Pass 1: Trim overlap + RIFE boundary transitions (stays at frame_rate) ----
    #
    # In SVI, segment N+1 re-generates the last `overlap_frames` of segment N
    # at its START.  So we keep ALL of segment 1, and for segments 2+ we skip
    # the first `overlap_frames` frames (the re-generated overlap region).
    # This avoids near-duplicate "frozen" frames at every boundary.
    stitched_frames: List[np.ndarray] = []

    for i, seg_frames in enumerate(all_segment_frames):
        if i == 0:
            # First segment: keep all frames
            trimmed = seg_frames
            logger.info(
                "[%s] Seg %d: keeping all %d frames (first segment)",
                job_id, i + 1, len(trimmed),
            )
        else:
            # Segments 2+: trim overlap_frames from the START
            if overlap_frames > 0 and len(seg_frames) > overlap_frames:
                trimmed = seg_frames[overlap_frames:]
                logger.info(
                    "[%s] Seg %d: trimmed first %d overlap frames, keeping %d of %d",
                    job_id, i + 1, overlap_frames, len(trimmed), len(seg_frames),
                )
            else:
                trimmed = seg_frames
                logger.info(
                    "[%s] Seg %d: no overlap trim needed (%d frames, overlap=%d)",
                    job_id, i + 1, len(seg_frames), overlap_frames,
                )

        # Insert RIFE transition frames BEFORE appending this segment (for seg 2+)
        if i > 0 and rife_model is not None:
            last_frame = stitched_frames[-1]  # last frame already in output
            first_next = trimmed[0]  # first frame of this segment after trimming

            try:
                transition_frames = _rife_interpolate_pair(
                    rife_model, last_frame, first_next,
                    num_intermediate=3,  # exp=2 -> 3 intermediate frames
                    device=device,
                )
                stitched_frames.extend(transition_frames)
                logger.info(
                    "[%s] RIFE: inserted %d transition frames between seg %d and %d",
                    job_id, len(transition_frames), i, i + 1,
                )
            except Exception as e:
                logger.warning(
                    "[%s] RIFE boundary interpolation failed between seg %d-%d: %s",
                    job_id, i, i + 1, e,
                )

        stitched_frames.extend(trimmed)

    logger.info(
        "[%s] Pass 1 complete: %d stitched frames at %d fps (%.1fs)",
        job_id, len(stitched_frames), frame_rate,
        len(stitched_frames) / frame_rate if frame_rate > 0 else 0,
    )

    # ---- Pass 2: Whole-video RIFE interpolation (frame_rate -> target_fps) ----
    if rife_model is not None and target_fps > frame_rate:
        logger.info(
            "[%s] RIFE Pass 2: interpolating %d frames from %dfps to %dfps (frame shape: %s)",
            job_id, len(stitched_frames), frame_rate, target_fps,
            stitched_frames[0].shape if stitched_frames else "N/A",
        )
        try:
            # Convert frames to tensor for apply_rife_interpolation
            # apply_rife_interpolation expects numpy [f, h, w, c] float32 in [0, 255]
            frames_np = np.stack(stitched_frames, axis=0)  # [f, h, w, c] uint8
            logger.info("[%s] RIFE Pass 2: stacked tensor shape=%s dtype=%s", job_id, frames_np.shape, frames_np.dtype)

            interpolated, actual_fps = apply_rife_interpolation(
                rife_model, frames_np, device=device,
                original_fps=frame_rate, target_fps=target_fps,
            )

            if interpolated is not None:
                logger.info(
                    "[%s] RIFE Pass 2: interpolation returned shape=%s dtype=%s, actual_fps=%.1f",
                    job_id, interpolated.shape, interpolated.dtype, actual_fps,
                )
                # interpolated is numpy [f, h, w, c] in [0, 255] or [0, 1]
                if interpolated.dtype == np.float32 or interpolated.dtype == np.float64:
                    if interpolated.max() <= 1.0:
                        interpolated = (interpolated * 255).astype(np.uint8)
                    else:
                        interpolated = interpolated.astype(np.uint8)

                stitched_frames = [interpolated[i] for i in range(interpolated.shape[0])]
                frame_rate = int(actual_fps)
                logger.info(
                    "[%s] RIFE Pass 2 ✓ complete: %d frames at %d fps (%.1fs video)",
                    job_id, len(stitched_frames), frame_rate,
                    len(stitched_frames) / frame_rate if frame_rate > 0 else 0,
                )
            else:
                logger.warning("[%s] RIFE Pass 2: returned None, keeping original %d frames at %dfps", job_id, len(stitched_frames), frame_rate)
        except Exception as e:
            logger.warning("[%s] RIFE Pass 2: whole-video interpolation failed: %s", job_id, e, exc_info=True)
            # Keep the pass-1 stitched frames at original frame_rate
    else:
        logger.info(
            "[%s] RIFE Pass 2: SKIPPED (target_fps=%d, frame_rate=%d, rife_model=%s)",
            job_id, target_fps, frame_rate, "loaded" if rife_model is not None else "None",
        )

    # ---- Encode final video ----
    _encode_frames_to_video(stitched_frames, output_path, fps=float(frame_rate))
    logger.info("[%s] Final video written: %s (%d frames at %d fps)", job_id, output_path, len(stitched_frames), frame_rate)


def _rife_single_segment(
    video_path: str,
    output_path: str,
    frame_rate: int,
    job_id: str,
    rife_model,
    device: str = "cuda",
    target_fps: int = 30,
) -> None:
    """
    Apply RIFE interpolation to a single-segment video to increase fps.
    Only runs Pass 2 (whole-video interpolation).
    """
    import torch

    logger.info(
        "[%s] RIFE single-segment: %dfps -> %dfps",
        job_id, frame_rate, target_fps,
    )

    frames, detected_fps = _decode_video_to_frames(video_path)
    logger.info("[%s] Decoded %d frames at %.1f fps", job_id, len(frames), detected_fps)

    if rife_model is not None and target_fps > frame_rate:
        try:
            frames_np = np.stack(frames, axis=0)
            interpolated, actual_fps = apply_rife_interpolation(
                rife_model, frames_np, device=device,
                original_fps=frame_rate, target_fps=target_fps,
            )
            if interpolated is not None:
                if interpolated.dtype == np.float32 or interpolated.dtype == np.float64:
                    if interpolated.max() <= 1.0:
                        interpolated = (interpolated * 255).astype(np.uint8)
                    else:
                        interpolated = interpolated.astype(np.uint8)
                frames = [interpolated[i] for i in range(interpolated.shape[0])]
                frame_rate = int(actual_fps)
                logger.info("[%s] RIFE interpolation: %d frames at %d fps", job_id, len(frames), frame_rate)
            else:
                logger.warning("[%s] RIFE returned None, keeping original", job_id)
        except Exception as e:
            logger.warning("[%s] RIFE single-segment interpolation failed: %s", job_id, e)

    _encode_frames_to_video(frames, output_path, fps=float(frame_rate))
    logger.info("[%s] Output written: %s", job_id, output_path)


# ---------------------------------------------------------------------------
# Output video assembly
# ---------------------------------------------------------------------------


def _find_output_video(
    output_files: list,
    job_id: str,
    output_dir: str,
    *,
    active_vhs_nodes: List[str],
    overlap_frames: int = 5,
    frame_rate: int = 16,
    enable_rife: bool = False,
    rife_target_fps: int = 30,
) -> str:
    """
    Find and assemble the final output video from ComfyUI's output files.

    - **Single output node** (1 segment or full 10-segment combiner):
      copies the single video to ``output_dir``.  If RIFE is enabled,
      applies single-segment RIFE interpolation (fps upscale only).
    - **Multiple output nodes** (2-9 segments, trimmed):
      collects one video per segment.  If RIFE is enabled, uses the
      two-pass RIFE pipeline (boundary transitions + fps upscale).
      Otherwise falls back to ffmpeg concatenation.
    """
    import shutil
    import subprocess
    import comfyui_bridge

    if not output_files:
        raise RuntimeError(
            f"[{job_id}] ComfyUI produced no output files. "
            f"Check ComfyUI logs for errors."
        )

    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, f"{job_id}.mp4")

    # Load RIFE model if needed
    rife_model = None
    logger.info(
        "[%s] POST-PROCESS: enable_rife=%s, RIFE_AVAILABLE=%s, rife_target_fps=%d, frame_rate=%d",
        job_id, enable_rife, RIFE_AVAILABLE, rife_target_fps, frame_rate,
    )
    if enable_rife and RIFE_AVAILABLE:
        logger.info("[%s] RIFE: loading model from %s ...", job_id, RIFE_WEIGHTS_DIR)
        rife_model = _get_rife_model()
        if rife_model is None:
            logger.warning("[%s] RIFE enabled but model not loaded, falling back to ffmpeg", job_id)
        else:
            logger.info("[%s] RIFE: model loaded successfully", job_id)
    elif enable_rife and not RIFE_AVAILABLE:
        logger.warning(
            "[%s] RIFE: requested but NOT AVAILABLE (import failed). "
            "Check that rife/ directory and video_postprocessing.py are present.",
            job_id,
        )
    else:
        logger.info("[%s] RIFE: disabled by user (enable_rife=%s)", job_id, enable_rife)

    # ---- Single output node (1 segment or full combiner) ----
    if len(active_vhs_nodes) == 1:
        logger.info("[%s] ASSEMBLY: single output node '%s'", job_id, active_vhs_nodes[0])
        src_path = _resolve_node_video(output_files, active_vhs_nodes[0], job_id)
        logger.info("[%s] ASSEMBLY: source video = %s (%.1f MB)", job_id, src_path, os.path.getsize(src_path) / 1e6)

        if rife_model is not None and rife_target_fps > frame_rate:
            logger.info("[%s] RIFE: applying single-segment interpolation %dfps -> %dfps", job_id, frame_rate, rife_target_fps)
            _rife_single_segment(
                src_path, dest_path, frame_rate, job_id,
                rife_model, target_fps=rife_target_fps,
            )
            _safe_remove(src_path, job_id)
        else:
            shutil.copy2(src_path, dest_path)
            logger.info("[%s] ASSEMBLY: copied as-is (no RIFE): %s", job_id, dest_path)
            _safe_remove(src_path, job_id)

        return dest_path

    # ---- Multiple segments: collect, stitch, clean up ----
    logger.info(
        "[%s] ASSEMBLY: collecting videos from %d segment nodes: %s",
        job_id, len(active_vhs_nodes), active_vhs_nodes,
    )
    segment_paths: List[str] = []
    for nid in active_vhs_nodes:
        try:
            p = _resolve_node_video(output_files, nid, job_id)
            sz = os.path.getsize(p) / 1e6
            logger.info("[%s] ASSEMBLY: node %s -> %s (%.1f MB)", job_id, nid, p, sz)
            segment_paths.append(p)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("[%s] Skipping segment node %s: %s", job_id, nid, e)

    if not segment_paths:
        raise RuntimeError(f"[{job_id}] No segment videos found from ComfyUI outputs")

    if len(segment_paths) == 1:
        logger.info("[%s] ASSEMBLY: only 1 segment found, treating as single-segment", job_id)
        if rife_model is not None and rife_target_fps > frame_rate:
            logger.info("[%s] RIFE: applying single-segment interpolation %dfps -> %dfps", job_id, frame_rate, rife_target_fps)
            _rife_single_segment(
                segment_paths[0], dest_path, frame_rate, job_id,
                rife_model, target_fps=rife_target_fps,
            )
            _safe_remove(segment_paths[0], job_id)
        else:
            shutil.copy2(segment_paths[0], dest_path)
            _safe_remove(segment_paths[0], job_id)
        return dest_path

    # Multi-segment assembly
    if rife_model is not None:
        logger.info(
            "[%s] ASSEMBLY: using two-pass RIFE pipeline (%d segments, overlap=%d frames, %dfps -> %dfps)",
            job_id, len(segment_paths), overlap_frames, frame_rate, rife_target_fps,
        )
        _rife_stitch_and_interpolate(
            segment_paths, dest_path, overlap_frames, frame_rate, job_id,
            rife_model, target_fps=rife_target_fps,
        )
    else:
        logger.info(
            "[%s] ASSEMBLY: using ffmpeg concatenation (no RIFE, %d segments, overlap=%d frames)",
            job_id, len(segment_paths), overlap_frames,
        )
        _concatenate_segments_ffmpeg(
            segment_paths, dest_path, overlap_frames, frame_rate, job_id,
        )

    # Clean up individual segment files
    for p in segment_paths:
        _safe_remove(p, job_id)

    return dest_path


def _resolve_node_video(
    output_files: list, node_id: str, job_id: str,
) -> str:
    """Find the video file produced by a specific ComfyUI node."""
    import comfyui_bridge

    # Look for this node's output
    for f in output_files:
        if f.get("node_id") == node_id:
            path = comfyui_bridge.get_output_file_path(f)
            if os.path.exists(path):
                return path

    # Fallback: any mp4 matching the node's prefix in output_files
    for f in output_files:
        fname = f.get("filename", "")
        if fname.endswith(".mp4"):
            path = comfyui_bridge.get_output_file_path(f)
            if os.path.exists(path):
                return path

    raise FileNotFoundError(
        f"[{job_id}] No video output found for node {node_id}"
    )


def _concatenate_segments_ffmpeg(
    segment_paths: List[str],
    output_path: str,
    overlap_frames: int,
    frame_rate: int,
    job_id: str,
) -> None:
    """
    Concatenate segment videos with ffmpeg, trimming overlap between
    adjacent segments.

    For SVI, adjacent segments share ``overlap_frames`` frames at the
    boundary. We trim those overlapping frames from the **end** of each
    segment except the last one so the final video plays smoothly.
    """
    import subprocess

    overlap_duration = overlap_frames / frame_rate if frame_rate > 0 else 0

    n = len(segment_paths)
    logger.info(
        "[%s] Concatenating %d segments (overlap=%.3fs per boundary)",
        job_id, n, overlap_duration,
    )

    if n == 1:
        import shutil
        shutil.copy2(segment_paths[0], output_path)
        return

    # Collect input args and get durations via ffprobe
    inputs = []
    for path in segment_paths:
        inputs.extend(["-i", path])

    durations = []
    for path in segment_paths:
        dur = _get_video_duration(path)
        durations.append(dur)
        logger.info("[%s] Segment %s duration: %.3fs", job_id, os.path.basename(path), dur)

    # Build ffmpeg filter_complex: trim overlap from the START of segments 2+
    # In SVI, segment N+1 re-generates the last `overlap_frames` of segment N
    # at its start.  Keep all of segment 1, skip overlap at the start of 2+.
    filter_complex_parts = []
    concat_inputs = []

    for i in range(n):
        if i > 0 and overlap_duration > 0:
            # Segments 2+: trim overlap from the START
            filter_complex_parts.append(
                f"[{i}:v]trim=start={overlap_duration:.4f},setpts=PTS-STARTPTS[v{i}]"
            )
            logger.info(
                "[%s] Seg %d: ffmpeg trim start=%.4fs (overlap removal)",
                job_id, i + 1, overlap_duration,
            )
        else:
            # First segment: keep everything
            filter_complex_parts.append(
                f"[{i}:v]setpts=PTS-STARTPTS[v{i}]"
            )
        concat_inputs.append(f"[v{i}]")

    concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"
    filter_complex_parts.append(concat_str)
    filter_complex = ";".join(filter_complex_parts)

    cmd = inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "19",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]
    cmd = ["ffmpeg"] + cmd

    logger.info("[%s] Running ffmpeg: %s", job_id, " ".join(cmd[:6]) + " ...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        logger.error("[%s] ffmpeg failed (rc=%d): %s", job_id, result.returncode, result.stderr[-1000:])
        raise RuntimeError(f"ffmpeg concatenation failed: {result.stderr[-500:]}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"[{job_id}] ffmpeg did not produce output: {output_path}")

    out_size = os.path.getsize(output_path)
    logger.info("[%s] Concatenated video: %s (%.1f MB)", job_id, output_path, out_size / 1e6)


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("Could not get duration for %s: %s, assuming 5s", path, e)
        return 5.0


def _safe_remove(path: str, job_id: str) -> None:
    """Remove a file, logging any errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("[%s] Failed to clean up %s: %s", job_id, path, e)
