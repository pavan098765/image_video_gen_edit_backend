#!/usr/bin/env python3
"""Patch the official MiniMax Music 3 visual workflow for a /run job."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("music_patch")

SUBGRAPH_TYPE = "ac99f841-a3de-4329-9564-953b81cf9e16"
SAVE_NODE_ID = 35
DEFAULT_UNET = "minimax_music3_dit_int8_convrot.safetensors"
DEFAULT_CLIP = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
DEFAULT_VAE = "minimax_music3_dav.safetensors"
FP16_UNET = "minimax_music3_dit_fp16.safetensors"

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


def resolve_unet(params: Dict[str, Any]) -> str:
    raw = params.get("unet_name") or params.get("dit") or params.get("precision")
    if not raw:
        return DEFAULT_UNET
    key = str(raw).strip().lower()
    if key in ("fp16", "float16", FP16_UNET.lower()):
        return FP16_UNET
    if key in ("int8", "int8_convrot", "convrot", DEFAULT_UNET.lower()):
        return DEFAULT_UNET
    return str(raw)


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
    unet = resolve_unet(params)
    _set_list_widget(node, UNET_IDX, unet)
    _set_named(node, "unet_name", unet)
    if params.get("clip_name"):
        _set_list_widget(node, CLIP_IDX, params["clip_name"])
        _set_named(node, "clip_name", params["clip_name"])
    else:
        _set_list_widget(node, CLIP_IDX, DEFAULT_CLIP)
        _set_named(node, "clip_name", DEFAULT_CLIP)
    if params.get("vae_name"):
        _set_list_widget(node, VAE_IDX, params["vae_name"])
        _set_named(node, "vae_name", params["vae_name"])
    else:
        _set_list_widget(node, VAE_IDX, DEFAULT_VAE)
        _set_named(node, "vae_name", DEFAULT_VAE)
    tiled = params.get("tiled_decode")
    if tiled is None:
        tiled = True
    _set_list_widget(node, TILED_IDX, _truthy(tiled))
    _set_named(node, "switch", _truthy(tiled))


def patch_inner_nodes(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    unet = resolve_unet(params)
    for node in _find_nodes_by_type(visual, "UNETLoader"):
        _set_list_widget(node, 0, unet)
        _set_named(node, "unet_name", unet)
        _clear_input_link(node, "unet_name")
    clip = params.get("clip_name") or DEFAULT_CLIP
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
    tiled = True if params.get("tiled_decode") is None else _truthy(params.get("tiled_decode"))
    for node in _find_nodes_by_type(visual, "ComfySwitchNode"):
        _set_list_widget(node, 0, tiled)
        _set_named(node, "switch", tiled)
        _clear_input_link(node, "switch")
    prefix = params.get("filename_prefix")
    if prefix:
        save = _find_node(visual, SAVE_NODE_ID) or (
            _find_nodes_by_type(visual, "SaveAudioAdvanced")[0]
            if _find_nodes_by_type(visual, "SaveAudioAdvanced")
            else None
        )
        if save is not None:
            _set_list_widget(save, 0, str(prefix))


def apply_job_params(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    patch_subgraph(visual, prompt, params)
    patch_inner_nodes(visual, prompt, params)
