#!/usr/bin/env python3
"""Patch the official MiniMax Music 3 visual workflow for a /run job."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("music_patch")

SUBGRAPH_TYPE = "ac99f841-a3de-4329-9564-953b81cf9e16"
SAVE_NODE_ID = 35
INT8_UNET = "minimax_music3_dit_int8_convrot.safetensors"
FP16_UNET = "minimax_music3_dit_fp16.safetensors"
INT8_CLIP = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
BF16_CLIP = "minimax_music3_text_encoder_pruned_bf16.safetensors"
DEFAULT_UNET = INT8_UNET
DEFAULT_CLIP = INT8_CLIP
DEFAULT_VAE = "minimax_music3_dav.safetensors"

# Official subgraph instance widget order (no widgets_values_named on this graph).
CAPTION_IDX = 0
LYRICS_IDX = 1
DURATION_IDX = 2
SEED_IDX = 3
UNET_IDX = 4
CLIP_IDX = 5
VAE_IDX = 6
TILED_IDX = 7


def _find_nodes_by_type(visual: dict, node_type: str) -> List[dict]:
    return [n for n in (visual.get("nodes") or []) if n.get("type") == node_type]


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
        keys = list(named.keys())
        if 0 <= index < len(keys):
            named[keys[index]] = value


def _clear_input_link(node: dict, name: str) -> None:
    for inp in node.get("inputs") or []:
        if isinstance(inp, dict) and inp.get("name") == name:
            inp["link"] = None


def _set_named(node: dict, name: str, value: Any) -> None:
    named = node.get("widgets_values_named")
    if not isinstance(named, dict):
        named = {}
        node["widgets_values_named"] = named
    named[name] = value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/app/comfyui/models"))


def _weight_exists(rel: str) -> bool:
    path = _models_dir() / rel.replace("\\", "/")
    return path.is_file() and path.stat().st_size > 0


def _detect_vram_gb() -> Optional[float]:
    try:
        from comfyui_bridge import detect_gpu
        gb, _name = detect_gpu()
        return gb
    except Exception:
        return None


def resolve_precision(params: Dict[str, Any]) -> str:
    """Job precision wins, else MUSIC3_PRECISION, else auto from VRAM."""
    raw = params.get("precision") or params.get("quality")
    if raw:
        key = str(raw).strip().lower()
        if key in ("quality", "bf16", "full", "hq"):
            return "quality"
        if key in ("fp16", "float16"):
            return "fp16"
        if key in ("int8", "fast", "lowvram", "int8_convrot"):
            return "int8"
    try:
        from comfyui_bridge import music3_precision
        env = music3_precision()
        if env != "auto":
            return env
    except Exception:
        pass
    vram = _detect_vram_gb()
    if vram is not None and vram >= 20.0:
        return "quality"
    if vram is not None and vram >= 15.0:
        return "fp16"
    return "int8"


def resolve_unet(params: Dict[str, Any]) -> str:
    unet, _clip, _tiled, _fmt = resolve_quality_stack(params)
    return unet


def _explicit_unet(params: Dict[str, Any]) -> Optional[str]:
    raw = params.get("unet_name") or params.get("dit")
    if not raw:
        return None
    key = str(raw).strip().lower()
    if key in ("fp16", "float16", FP16_UNET.lower()):
        return FP16_UNET if _weight_exists(f"diffusion_models/{FP16_UNET}") else INT8_UNET
    if key in ("int8", "int8_convrot", "convrot", INT8_UNET.lower()):
        return INT8_UNET
    if key in ("quality", "bf16", "full", "hq", "fast", "lowvram"):
        return None
    return str(raw)


def resolve_quality_stack(params: Dict[str, Any]) -> Tuple[str, str, bool, str]:
    """Return (unet, clip, tiled_decode, audio_format) for this GPU / job.

    Hugging Face Space uses full BF16 + WAV. On a 3090 we use the closest
    Comfy-Org pack that fits: pruned BF16 text encoder + FP16 DiT, dense VAE,
    WAV. Weaker cards stay on INT8 and tiled decode.
    """
    prec = resolve_precision(params)
    fp16_ok = _weight_exists(f"diffusion_models/{FP16_UNET}")
    bf16_ok = _weight_exists(f"text_encoders/{BF16_CLIP}")
    vram = _detect_vram_gb()

    unet = INT8_UNET
    clip = INT8_CLIP
    tiled = True
    fmt = "mp3"

    if prec == "quality":
        if fp16_ok:
            unet = FP16_UNET
        if bf16_ok:
            clip = BF16_CLIP
        tiled = False
        fmt = "wav"
        if not bf16_ok:
            logger.warning(
                "MUSIC3 quality requested but %s is missing; AR stays INT8. "
                "Download Comfy-Org pruned BF16 encoder for Space-like composition.",
                BF16_CLIP,
            )
        if not fp16_ok:
            logger.warning(
                "MUSIC3 quality requested but %s is missing; DiT stays INT8.",
                FP16_UNET,
            )
    elif prec == "fp16":
        if fp16_ok:
            unet = FP16_UNET
        tiled = False
        fmt = "wav"
    elif vram is not None and vram >= 15.0:
        tiled = False
        fmt = "wav"
        if fp16_ok:
            unet = FP16_UNET

    explicit = _explicit_unet(params)
    if explicit:
        unet = explicit
    if params.get("clip_name"):
        clip = str(params["clip_name"])
    if params.get("tiled_decode") is not None:
        tiled = _truthy(params.get("tiled_decode"))
    fmt_raw = params.get("audio_format") or params.get("output_format")
    if fmt_raw:
        key = str(fmt_raw).strip().lower()
        if key in ("wav", "flac", "mp3", "opus", "ogg"):
            fmt = key

    logger.info(
        "Music 3 stack precision=%s vram=%s unet=%s clip=%s tiled=%s format=%s",
        prec,
        f"{vram:.1f}GiB" if vram is not None else "unknown",
        unet,
        clip,
        tiled,
        fmt,
    )
    return unet, clip, tiled, fmt


def patch_subgraph(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    nodes = _find_nodes_by_type(visual, SUBGRAPH_TYPE)
    if not nodes:
        logger.debug("Music 3 subgraph instance %s already flattened", SUBGRAPH_TYPE)
        return
    node = nodes[0]
    caption = params.get("caption") or prompt or ""
    lyrics = params.get("lyrics") or params.get("negative_prompt") or ""
    _set_list_widget(node, CAPTION_IDX, caption)
    _set_list_widget(node, LYRICS_IDX, lyrics)
    _set_named(node, "caption", caption)
    _set_named(node, "lyrics", lyrics)
    duration = params.get("max_duration") or params.get("duration") or params.get("duration_seconds")
    if duration is not None:
        _set_list_widget(node, DURATION_IDX, float(duration))
        _set_named(node, "max_duration", float(duration))
    if params.get("seed") is not None:
        _set_list_widget(node, SEED_IDX, int(params["seed"]))
        _set_named(node, "seed", int(params["seed"]))
    unet, clip, tiled, _fmt = resolve_quality_stack(params)
    _set_list_widget(node, UNET_IDX, unet)
    _set_named(node, "unet_name", unet)
    _set_list_widget(node, CLIP_IDX, clip)
    _set_named(node, "clip_name", clip)
    if params.get("vae_name"):
        _set_list_widget(node, VAE_IDX, params["vae_name"])
        _set_named(node, "vae_name", params["vae_name"])
    else:
        _set_list_widget(node, VAE_IDX, DEFAULT_VAE)
        _set_named(node, "vae_name", DEFAULT_VAE)
    _set_list_widget(node, TILED_IDX, tiled)
    _set_named(node, "switch", tiled)


def patch_inner_nodes(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    unet, clip, tiled, fmt = resolve_quality_stack(params)
    for node in _find_nodes_by_type(visual, "UNETLoader"):
        _set_list_widget(node, 0, unet)
        _set_named(node, "unet_name", unet)
        _clear_input_link(node, "unet_name")
    for node in _find_nodes_by_type(visual, "CLIPLoader"):
        _set_list_widget(node, 0, clip)
        _set_list_widget(node, 1, "minimax")
        _set_named(node, "clip_name", clip)
        _set_named(node, "type", "minimax")
        _clear_input_link(node, "clip_name")
    vae = params.get("vae_name") or DEFAULT_VAE
    for node in _find_nodes_by_type(visual, "VAELoader"):
        _set_list_widget(node, 0, vae)
        _set_named(node, "vae_name", vae)
        _clear_input_link(node, "vae_name")

    caption = params.get("caption") or prompt or ""
    lyrics = params.get("lyrics") or ""
    duration = params.get("max_duration") or params.get("duration") or params.get("duration_seconds")
    seed = params.get("seed")
    for node in _find_nodes_by_type(visual, "MiniMaxMusic3TextEncode"):
        _set_list_widget(node, 0, caption)
        _set_list_widget(node, 1, lyrics)
        _set_named(node, "caption", caption)
        _set_named(node, "lyrics", lyrics)
        _clear_input_link(node, "caption")
        _clear_input_link(node, "lyrics")
        if seed is not None:
            _set_list_widget(node, 2, int(seed))
            _set_named(node, "seed", int(seed))
            _clear_input_link(node, "seed")
        if duration is not None:
            _set_list_widget(node, 4, float(duration))
            _set_named(node, "max_duration", float(duration))
            _clear_input_link(node, "max_duration")
    if seed is not None:
        for node in _find_nodes_by_type(visual, "KSampler"):
            _set_list_widget(node, 0, int(seed))
            _set_named(node, "seed", int(seed))
            _clear_input_link(node, "seed")
        for node in _find_nodes_by_type(visual, "SeedNode"):
            _set_list_widget(node, 0, int(seed))
    if params.get("steps") is not None:
        for node in _find_nodes_by_type(visual, "KSampler"):
            _set_list_widget(node, 2, int(params["steps"]))
    if params.get("cfg") is not None:
        for node in _find_nodes_by_type(visual, "KSampler"):
            _set_list_widget(node, 3, float(params["cfg"]))
    for node in _find_nodes_by_type(visual, "ComfySwitchNode"):
        _set_list_widget(node, 0, tiled)
        _set_named(node, "switch", tiled)
        _clear_input_link(node, "switch")
    saves = _find_nodes_by_type(visual, "SaveAudioAdvanced")
    save = _find_node(visual, SAVE_NODE_ID) or (saves[0] if saves else None)
    if save is not None:
        prefix = params.get("filename_prefix")
        if prefix:
            _set_list_widget(save, 0, str(prefix))
        _set_named(save, "format", fmt)
        _set_list_widget(save, 1, fmt)


def apply_api_job_params(prompt_graph: dict, prompt: str, params: Dict[str, Any]) -> None:
    from workflow_converter import api_nodes_by_type, set_api_input

    caption = params.get("caption") or prompt or ""
    lyrics = params.get("lyrics") or params.get("negative_prompt") or ""
    duration = params.get("max_duration") or params.get("duration") or params.get("duration_seconds")
    seed = params.get("seed")
    unet, clip, tiled, fmt = resolve_quality_stack(params)
    vae = params.get("vae_name") or DEFAULT_VAE
    for _, node in api_nodes_by_type(prompt_graph, "UNETLoader"):
        set_api_input(node, "unet_name", unet)
    for _, node in api_nodes_by_type(prompt_graph, "CLIPLoader"):
        set_api_input(node, "clip_name", clip)
        set_api_input(node, "type", "minimax")
    for _, node in api_nodes_by_type(prompt_graph, "VAELoader"):
        set_api_input(node, "vae_name", vae)
    for _, node in api_nodes_by_type(prompt_graph, "MiniMaxMusic3TextEncode"):
        set_api_input(node, "caption", caption)
        set_api_input(node, "lyrics", lyrics)
        if duration is not None:
            set_api_input(node, "max_duration", float(duration))
        if seed is not None:
            set_api_input(node, "seed", int(seed))
    if seed is not None:
        for _, node in api_nodes_by_type(prompt_graph, "SeedNode"):
            set_api_input(node, "seed", int(seed))
        for _, node in api_nodes_by_type(prompt_graph, "KSampler"):
            set_api_input(node, "seed", int(seed))
    if params.get("steps") is not None:
        for _, node in api_nodes_by_type(prompt_graph, "KSampler"):
            set_api_input(node, "steps", int(params["steps"]))
    if params.get("cfg") is not None:
        for _, node in api_nodes_by_type(prompt_graph, "KSampler"):
            set_api_input(node, "cfg", float(params["cfg"]))
    for _, node in api_nodes_by_type(prompt_graph, "ComfySwitchNode"):
        set_api_input(node, "switch", tiled)
    for _, node in api_nodes_by_type(prompt_graph, "SaveAudioAdvanced"):
        set_api_input(node, "format", fmt)
        prefix = params.get("filename_prefix")
        if prefix:
            set_api_input(node, "filename_prefix", str(prefix))


def apply_job_params(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    patch_subgraph(visual, prompt, params)
    patch_inner_nodes(visual, prompt, params)
