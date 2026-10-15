#!/usr/bin/env python3
"""Patch the Civitai LTX-2.5 INT8 AllModes visual workflow for a /run job."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ltx_patch")

CONTROLS_NODE_ID = 408
PROMPT_NODE_ID = 376
NEGATIVE_NODE_ID = 373
FIRST_IMAGE_NODE_ID = 409
LAST_IMAGE_NODE_ID = 411
STAGE2_NODE_ID = 423
LORA_NODE_ID = 422
SAVE_NODE_ID = 75

MODE_T2V = "T2V (text only)"
MODE_I2V = "I2V (first frame)"
MODE_FLF2V = "FLF2V (first + last frame)"

MODE_ALIASES = {
    "t2v": MODE_T2V,
    "text": MODE_T2V,
    "text_to_video": MODE_T2V,
    "ltx25_int8_t2v": MODE_T2V,
    "i2v": MODE_I2V,
    "image": MODE_I2V,
    "image_to_video": MODE_I2V,
    "first_frame": MODE_I2V,
    "ltx25_int8_i2v": MODE_I2V,
    "flf2v": MODE_FLF2V,
    "flf": MODE_FLF2V,
    "first_last": MODE_FLF2V,
    "first_last_frame": MODE_FLF2V,
    "ltx25_int8_flf2v": MODE_FLF2V,
}

STAGE2_SPEED = "Speed — Sage + Sol (12–16GB target)"
STAGE2_QUALITY = "Quality — Native exact (24GB+ recommended)"
STAGE1_SAMPLER_ID = "344"
UNLOAD_NODE_ID = "407"
STAGE1_SEPARATE_ID = "367"
VIDEO_DECODE_ID = "374"
AUDIO_DECODE_ID = "358"
_SKIP_STAGE2 = frozenset({"skip", "draft", "stage1", "stage1_only", "none"})

VAE_PRESETS = {
    "auto": "Auto (VRAM-based)",
    "12gb": "Low VRAM (12GB target, experimental)",
    "16gb": "Medium Tiles (16GB recommended)",
    "24gb": "Large Tiles (24GB recommended)",
}


def _find_node(visual: dict, node_id: int) -> Optional[dict]:
    for node in visual.get("nodes") or []:
        try:
            if int(node.get("id")) == int(node_id):
                return node
        except (TypeError, ValueError):
            continue
    return None


def _set_list_widget(node: dict, index: int, value: Any) -> None:
    wv = node.get("widgets_values")
    if not isinstance(wv, list):
        wv = []
        node["widgets_values"] = wv
    while len(wv) <= index:
        wv.append(None)
    wv[index] = value
    named = node.get("widgets_values_named")
    if isinstance(named, dict):
        inputs = node.get("inputs") or []
        if 0 <= index < len(inputs) and isinstance(inputs[index], dict):
            name = inputs[index].get("name")
            if name:
                named[name] = value


def resolve_mode(model: str, params: Dict[str, Any]) -> str:
    raw = params.get("ltx_mode") or params.get("mode") or model or MODE_I2V
    key = str(raw).strip()
    if key in (MODE_T2V, MODE_I2V, MODE_FLF2V):
        return key
    return MODE_ALIASES.get(key.lower().replace("-", "_").replace(" ", "_"), MODE_I2V)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def ltx_canvas_dim(value: Any) -> int:
    """LTX25AllModesControlsV2 rejects widths and heights that are not multiples of 64."""
    try:
        original = int(value)
    except (TypeError, ValueError):
        original = 256
    n = original
    if n < 256:
        n = 256
    if n > 2048:
        n = 2048
    snapped = (n // 64) * 64
    if snapped < 256:
        snapped = 256
    if snapped != original:
        logger.info("LTX canvas %s snapped to multiple of 64 -> %s", value, snapped)
    return snapped


def wants_skip_stage2(params: Dict[str, Any]) -> bool:
    """Decode the stage-1 latent and skip the x2 refine. Output is half of final size."""
    if _truthy(params.get("skip_stage2")) or _truthy(params.get("draft")):
        return True
    raw = str(params.get("stage2_mode") or params.get("quality_mode") or "").strip().lower()
    return raw in _SKIP_STAGE2


def resolve_stage2(params: Dict[str, Any]) -> str:
    raw = str(params.get("stage2_mode") or params.get("quality_mode") or "speed").strip().lower()
    if raw in _SKIP_STAGE2:
        return STAGE2_SPEED
    if raw in ("quality", "exact", "native", STAGE2_QUALITY.lower()):
        return STAGE2_QUALITY
    return STAGE2_SPEED


def _require_class(prompt_graph: dict, node_id: str, class_type: str) -> dict:
    node = prompt_graph.get(node_id)
    if not isinstance(node, dict) or node.get("class_type") != class_type:
        raise RuntimeError(
            f"LTX API node {node_id} is not {class_type}; refusing to skip stage 2"
        )
    return node


def apply_skip_stage2(prompt_graph: dict) -> None:
    """Bypass the x2 upscaler and stage-2 sampler. Stage-1 attention path stays.

    VRAM unload still runs before Conv VAE decode. Default jobs never call this.
    """
    stage1 = _require_class(prompt_graph, STAGE1_SAMPLER_ID, "SamplerCustomAdvanced")
    unload = _require_class(prompt_graph, UNLOAD_NODE_ID, "VRAM_Debug")
    separate = _require_class(prompt_graph, STAGE1_SEPARATE_ID, "LTXVSeparateAVLatent")
    video_decode = _require_class(prompt_graph, VIDEO_DECODE_ID, "VAEDecodeTiled")
    audio_decode = _require_class(prompt_graph, AUDIO_DECODE_ID, "LTXVAudioVAEDecode")
    if stage1.get("_meta", {}).get("title") and "Stage 1" not in str(stage1["_meta"].get("title")):
        raise RuntimeError("LTX node 344 is not the stage-1 sampler; refusing to skip stage 2")
    unload_inputs = unload.setdefault("inputs", {})
    unload_inputs["any_input"] = [STAGE1_SAMPLER_ID, 0]
    separate.setdefault("inputs", {})["av_latent"] = [UNLOAD_NODE_ID, 0]
    video_decode.setdefault("inputs", {})["samples"] = [STAGE1_SEPARATE_ID, 0]
    audio_decode.setdefault("inputs", {})["samples"] = [STAGE1_SEPARATE_ID, 1]
    logger.info(
        "LTX stage 2 skipped: decode stage-1 latent (half of final_width x final_height)"
    )


def patch_prompt(visual: dict, prompt: str) -> None:
    node = _find_node(visual, PROMPT_NODE_ID)
    if node is None:
        logger.warning("PrimitiveStringMultiline prompt node %s missing", PROMPT_NODE_ID)
        return
    _set_list_widget(node, 0, prompt)


def patch_negative(visual: dict, negative: str) -> None:
    node = _find_node(visual, NEGATIVE_NODE_ID)
    if node is None:
        return
    _set_list_widget(node, 0, negative)


def patch_controls(visual: dict, mode: str, params: Dict[str, Any]) -> None:
    node = _find_node(visual, CONTROLS_NODE_ID)
    if node is None:
        logger.warning("LTX25AllModesControlsV2 node %s missing", CONTROLS_NODE_ID)
        return
    _set_list_widget(node, 0, mode)
    preset = params.get("vae_decode_preset") or params.get("gpu_profile")
    if preset:
        key = str(preset).strip().lower()
        _set_list_widget(node, 1, VAE_PRESETS.get(key, str(preset)))
    if params.get("width") is not None:
        _set_list_widget(node, 2, ltx_canvas_dim(params["width"]))
    elif params.get("final_width") is not None:
        _set_list_widget(node, 2, ltx_canvas_dim(params["final_width"]))
    if params.get("height") is not None:
        _set_list_widget(node, 3, ltx_canvas_dim(params["height"]))
    elif params.get("final_height") is not None:
        _set_list_widget(node, 3, ltx_canvas_dim(params["final_height"]))
    if params.get("fps") is not None:
        _set_list_widget(node, 5, float(params["fps"]))
    if params.get("duration") is not None or params.get("duration_seconds") is not None:
        _set_list_widget(node, 4, "Duration drives frames")
        _set_list_widget(
            node,
            6,
            float(params.get("duration") if params.get("duration") is not None else params["duration_seconds"]),
        )
    if params.get("frames") is not None or params.get("manual_frames") is not None:
        _set_list_widget(node, 4, "Manual exact frames")
        _set_list_widget(
            node,
            7,
            int(params.get("frames") if params.get("frames") is not None else params["manual_frames"]),
        )
    if params.get("seed") is not None:
        _set_list_widget(node, 8, int(params["seed"]))
        _set_list_widget(node, 9, "fixed")


def patch_stage2(visual: dict, params: Dict[str, Any]) -> None:
    node = _find_node(visual, STAGE2_NODE_ID)
    if node is None:
        return
    _set_list_widget(node, 0, resolve_stage2(params))


def patch_lora(visual: dict, params: Dict[str, Any]) -> None:
    node = _find_node(visual, LORA_NODE_ID)
    if node is None:
        return
    name = params.get("lora_name")
    if name:
        _set_list_widget(node, 0, str(name))
    if params.get("lora_strength") is not None:
        _set_list_widget(node, 1, float(params["lora_strength"]))


def patch_image_loader(visual: dict, node_id: int, filename: Optional[str]) -> None:
    node = _find_node(visual, node_id)
    if node is None:
        return
    _set_list_widget(node, 0, filename or "")


def apply_api_job_params(
    prompt_graph: dict,
    prompt: str,
    params: Dict[str, Any],
    model: str,
    first_filename: Optional[str] = None,
    last_filename: Optional[str] = None,
) -> None:
    from workflow_converter import set_api_input

    mode = resolve_mode(model, params)
    node_376 = prompt_graph.get("376")
    if prompt and isinstance(node_376, dict):
        set_api_input(node_376, "value", prompt)
    node_373 = prompt_graph.get("373")
    if params.get("negative_prompt") is not None and isinstance(node_373, dict):
        set_api_input(node_373, "text", str(params["negative_prompt"]))
    controls = prompt_graph.get("408")
    if isinstance(controls, dict):
        set_api_input(controls, "mode", mode)
        preset = params.get("vae_decode_preset") or params.get("gpu_profile")
        if preset:
            key = str(preset).strip().lower()
            set_api_input(controls, "vae_decode_preset", VAE_PRESETS.get(key, str(preset)))
        if params.get("width") is not None:
            set_api_input(controls, "final_width", ltx_canvas_dim(params["width"]))
        elif params.get("final_width") is not None:
            set_api_input(controls, "final_width", ltx_canvas_dim(params["final_width"]))
        if params.get("height") is not None:
            set_api_input(controls, "final_height", ltx_canvas_dim(params["height"]))
        elif params.get("final_height") is not None:
            set_api_input(controls, "final_height", ltx_canvas_dim(params["final_height"]))
        if params.get("fps") is not None:
            set_api_input(controls, "fps", float(params["fps"]))
        if params.get("duration") is not None or params.get("duration_seconds") is not None:
            set_api_input(controls, "timing_mode", "Duration drives frames")
            set_api_input(
                controls,
                "duration_seconds",
                float(params.get("duration") if params.get("duration") is not None else params["duration_seconds"]),
            )
        if params.get("frames") is not None or params.get("manual_frames") is not None:
            set_api_input(controls, "timing_mode", "Manual exact frames")
            set_api_input(
                controls,
                "manual_frames",
                int(params.get("frames") if params.get("frames") is not None else params["manual_frames"]),
            )
        if params.get("seed") is not None:
            set_api_input(controls, "seed", int(params["seed"]))
    stage2 = prompt_graph.get("423")
    if isinstance(stage2, dict):
        set_api_input(stage2, "stage2_mode", resolve_stage2(params))
    if wants_skip_stage2(params):
        apply_skip_stage2(prompt_graph)
    lora = prompt_graph.get("422")
    if isinstance(lora, dict):
        if params.get("lora_name"):
            set_api_input(lora, "lora_name", str(params["lora_name"]))
        if params.get("lora_strength") is not None:
            set_api_input(lora, "strength_model", float(params["lora_strength"]))
    first = prompt_graph.get("409")
    last = prompt_graph.get("411")
    if mode == MODE_T2V:
        if isinstance(first, dict):
            set_api_input(first, "image", "")
        if isinstance(last, dict):
            set_api_input(last, "image", "")
    elif mode == MODE_I2V:
        if isinstance(first, dict):
            set_api_input(first, "image", first_filename or "")
        if isinstance(last, dict):
            set_api_input(last, "image", "")
    else:
        if isinstance(first, dict):
            set_api_input(first, "image", first_filename or "")
        if isinstance(last, dict):
            set_api_input(last, "image", last_filename or "")


def apply_job_params(
    visual: dict,
    prompt: str,
    params: Dict[str, Any],
    model: str,
    first_filename: Optional[str] = None,
    last_filename: Optional[str] = None,
) -> dict:
    params = dict(params or {})
    mode = resolve_mode(model, params)
    if prompt:
        patch_prompt(visual, prompt)
    if params.get("negative_prompt") is not None:
        patch_negative(visual, str(params["negative_prompt"]))
    patch_controls(visual, mode, params)
    patch_stage2(visual, params)
    patch_lora(visual, params)
    if mode == MODE_T2V:
        patch_image_loader(visual, FIRST_IMAGE_NODE_ID, "")
        patch_image_loader(visual, LAST_IMAGE_NODE_ID, "")
    elif mode == MODE_I2V:
        patch_image_loader(visual, FIRST_IMAGE_NODE_ID, first_filename or "")
        patch_image_loader(visual, LAST_IMAGE_NODE_ID, "")
    else:
        patch_image_loader(visual, FIRST_IMAGE_NODE_ID, first_filename or "")
        patch_image_loader(visual, LAST_IMAGE_NODE_ID, last_filename or "")
    return visual
