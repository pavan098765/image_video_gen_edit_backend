#!/usr/bin/env python3
"""Patch the Qwen Image 2.1 Turbo 8-step visual workflow for a /run job."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("qwen_patch")

SUBGRAPH_TYPE = "c291ceec-b98f-4751-9d0b-7bc288f27b30"
SAVE_NODE_ID = 461
DEFAULT_UNET = "qwen-image-2.1-turbo-Q8_0.gguf"
DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"


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
    wv = node.get("widgets_values")
    inputs = node.get("inputs") or []
    names: List[str] = []
    if isinstance(named, dict) and isinstance(wv, list) and len(named) == len(wv):
        names = list(named.keys())
    else:
        for inp in inputs:
            if isinstance(inp, dict) and inp.get("widget"):
                names.append(str(inp.get("name")))
    if isinstance(wv, list) and name in names:
        idx = names.index(name)
        while len(wv) <= idx:
            wv.append(None)
        wv[idx] = value
        return
    if isinstance(wv, list) and names:
        return
    if not isinstance(wv, list):
        node["widgets_values"] = [value]


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


def _next_ids(visual: dict) -> tuple[int, int]:
    max_node = 0
    for node in visual.get("nodes") or []:
        try:
            max_node = max(max_node, int(node.get("id") or 0))
        except (TypeError, ValueError):
            continue
    max_link = int(visual.get("last_link_id") or 0)
    for link in visual.get("links") or []:
        if isinstance(link, list) and link:
            try:
                max_link = max(max_link, int(link[0]))
            except (TypeError, ValueError):
                continue
    return max_node + 1, max_link + 1


def patch_subgraph(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    nodes = _find_nodes_by_type(visual, SUBGRAPH_TYPE)
    if not nodes:
        logger.debug("Qwen subgraph instance %s already flattened", SUBGRAPH_TYPE)
        return
    node = nodes[0]
    _set_named(node, "prompt", prompt or params.get("prompt") or "")
    if params.get("negative_prompt") is not None or params.get("negative") is not None:
        _set_named(node, "negative_prompt", params.get("negative_prompt") or params.get("negative") or "")
    if params.get("cfg") is not None:
        _set_named(node, "cfg", float(params["cfg"]))
    if params.get("width") is not None:
        _set_named(node, "width", int(params["width"]))
    if params.get("height") is not None:
        _set_named(node, "height", int(params["height"]))
    if params.get("seed") is not None:
        _set_named(node, "seed", int(params["seed"]))
    unet = params.get("unet_name") or params.get("gguf") or DEFAULT_UNET
    _set_named(node, "unet_name", unet)
    if params.get("clip_name"):
        _set_named(node, "clip_name", params["clip_name"])
    if params.get("vae_name"):
        _set_named(node, "vae_name", params["vae_name"])


def patch_resolution(visual: dict, params: Dict[str, Any]) -> None:
    nodes = _find_nodes_by_type(visual, "ResolutionSelector")
    if not nodes:
        return
    node = nodes[0]
    if params.get("aspect_ratio"):
        _set_list_widget(node, 0, str(params["aspect_ratio"]))
    mp = params.get("megapixels")
    if mp is not None:
        _set_list_widget(node, 1, float(mp))


def patch_inner_nodes(visual: dict, prompt: str, params: Dict[str, Any]) -> None:
    unet = params.get("unet_name") or params.get("gguf") or DEFAULT_UNET
    for node in _find_nodes_by_type(visual, "UnetLoaderGGUF"):
        _set_named(node, "unet_name", unet)
        _set_list_widget(node, 0, unet)
        _clear_input_link(node, "unet_name")
    for node in _find_nodes_by_type(visual, "UNETLoader"):
        _set_named(node, "unet_name", unet)
        _set_list_widget(node, 0, unet)
        _clear_input_link(node, "unet_name")
    clip = params.get("clip_name") or DEFAULT_CLIP
    for node in _find_nodes_by_type(visual, "CLIPLoader"):
        _set_named(node, "clip_name", clip)
        _set_list_widget(node, 0, clip)
        _set_named(node, "type", "qwen_image")
        _set_list_widget(node, 1, "qwen_image")
        _clear_input_link(node, "clip_name")
    vae = params.get("vae_name") or DEFAULT_VAE
    for node in _find_nodes_by_type(visual, "VAELoader"):
        _set_named(node, "vae_name", vae)
        _set_list_widget(node, 0, vae)
        _clear_input_link(node, "vae_name")
    for node in _find_nodes_by_type(visual, "TextEncodeQwenImage21"):
        _set_named(node, "prompt", prompt or params.get("prompt") or "")
        _clear_input_link(node, "prompt")
        neg = params.get("negative_prompt")
        if neg is None:
            neg = params.get("negative")
        if neg is None:
            named = node.get("widgets_values_named") if isinstance(node.get("widgets_values_named"), dict) else {}
            neg = named.get("negative_prompt") or ""
        _set_named(node, "negative_prompt", neg)
        _clear_input_link(node, "negative_prompt")
        if params.get("encode_resolution") is not None:
            _set_named(node, "resolution", int(params["encode_resolution"]))
    seed = params.get("seed")
    cfg = params.get("cfg")
    if cfg is None:
        cfg = 1
    for node in _find_nodes_by_type(visual, "SamplerCustom"):
        if seed is not None:
            _set_named(node, "noise_seed", int(seed))
            # widgets_values: [add_noise, noise_seed, control_after_generate, cfg]
            _set_list_widget(node, 1, int(seed))
            _clear_input_link(node, "noise_seed")
        _set_named(node, "cfg", float(cfg))
        _set_list_widget(node, 3, float(cfg))
        _clear_input_link(node, "cfg")
    if params.get("width") is not None or params.get("height") is not None:
        for node in _find_nodes_by_type(visual, "EmptyLatentImage"):
            if params.get("width") is not None:
                _set_named(node, "width", int(params["width"]))
                _set_list_widget(node, 0, int(params["width"]))
            if params.get("height") is not None:
                _set_named(node, "height", int(params["height"]))
                _set_list_widget(node, 1, int(params["height"]))
    prefix = params.get("filename_prefix")
    if prefix:
        save = _find_node(visual, SAVE_NODE_ID) or (
            _find_nodes_by_type(visual, "SaveImageAdvanced")[0]
            if _find_nodes_by_type(visual, "SaveImageAdvanced")
            else None
        )
        if save is not None:
            _set_named(save, "filename_prefix", str(prefix))
            _set_list_widget(save, 0, str(prefix))


def inject_guide_image(visual: dict, filename: str) -> None:
    """Wire LoadImage into TextEncodeQwenImage21.image_1 for I2V / image-edit."""
    encoders = _find_nodes_by_type(visual, "TextEncodeQwenImage21")
    if not encoders:
        logger.warning("TextEncodeQwenImage21 missing; cannot attach guide image")
        return
    encoder = encoders[0]
    vaes = _find_nodes_by_type(visual, "VAELoader")
    node_id, link_id = _next_ids(visual)
    load_node = {
        "id": node_id,
        "type": "LoadImage",
        "pos": [0, 0],
        "size": [300, 80],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [link_id], "slot_index": 0},
            {"name": "MASK", "type": "MASK", "links": None},
        ],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": [filename, "image"],
        "widgets_values_named": {"image": filename},
    }
    visual.setdefault("nodes", []).append(load_node)

    image_slot = None
    vae_slot = None
    for idx, inp in enumerate(encoder.get("inputs") or []):
        name = str(inp.get("name") or "")
        if name in ("images.image_1", "image_1"):
            image_slot = idx
            inp["link"] = link_id
        if name == "vae":
            vae_slot = idx
    visual.setdefault("links", []).append(
        [link_id, node_id, 0, encoder.get("id"), image_slot if image_slot is not None else 1, "IMAGE"]
    )
    if vaes and vae_slot is not None:
        vae_node = vaes[0]
        vae_link = link_id + 1
        vae_out = None
        for out in vae_node.get("outputs") or []:
            if out.get("name") == "VAE":
                vae_out = out
                break
        if vae_out is not None:
            links = vae_out.get("links") or []
            links.append(vae_link)
            vae_out["links"] = links
            encoder["inputs"][vae_slot]["link"] = vae_link
            visual["links"].append(
                [vae_link, vae_node.get("id"), 0, encoder.get("id"), vae_slot, "VAE"]
            )
            visual["last_link_id"] = vae_link
        else:
            visual["last_link_id"] = link_id
    else:
        visual["last_link_id"] = link_id
    visual["last_node_id"] = node_id
    logger.info("Attached guide image %s to TextEncodeQwenImage21", filename)


def apply_api_job_params(
    prompt_graph: dict,
    prompt: str,
    params: Dict[str, Any],
    guide_image: Optional[str] = None,
) -> None:
    """Patch a ComfyUI API-format prompt (no flatten)."""
    from workflow_converter import api_nodes_by_type, set_api_input

    unet = params.get("unet_name") or params.get("gguf") or DEFAULT_UNET
    clip = params.get("clip_name") or DEFAULT_CLIP
    vae = params.get("vae_name") or DEFAULT_VAE
    for _, node in api_nodes_by_type(prompt_graph, "UnetLoaderGGUF"):
        set_api_input(node, "unet_name", unet)
    for _, node in api_nodes_by_type(prompt_graph, "CLIPLoader"):
        set_api_input(node, "clip_name", clip)
        set_api_input(node, "type", "qwen_image")
    for _, node in api_nodes_by_type(prompt_graph, "VAELoader"):
        set_api_input(node, "vae_name", vae)
    text = prompt or params.get("prompt") or ""
    if guide_image and "<image1>" not in text and "<image_1>" not in text:
        text = "Keep the subject, pose, and composition of <image1>. " + text
    for _, node in api_nodes_by_type(prompt_graph, "TextEncodeQwenImage21"):
        set_api_input(node, "prompt", text)
        neg = params.get("negative_prompt")
        if neg is None:
            neg = params.get("negative")
        if neg is not None:
            set_api_input(node, "negative_prompt", neg)
        if params.get("encode_resolution") is not None:
            set_api_input(node, "resolution", int(params["encode_resolution"]))
    cfg = params.get("cfg")
    if cfg is None:
        cfg = 1
    seed = params.get("seed")
    for _, node in api_nodes_by_type(prompt_graph, "SamplerCustom"):
        set_api_input(node, "cfg", float(cfg))
        if seed is not None:
            set_api_input(node, "noise_seed", int(seed))
        set_api_input(node, "add_noise", True)
    for _, node in api_nodes_by_type(prompt_graph, "ResolutionSelector"):
        if params.get("aspect_ratio"):
            set_api_input(node, "aspect_ratio", str(params["aspect_ratio"]))
        if params.get("megapixels") is not None:
            set_api_input(node, "megapixels", float(params["megapixels"]))
    if params.get("width") is not None or params.get("height") is not None:
        for _, node in api_nodes_by_type(prompt_graph, "EmptyLatentImage"):
            if params.get("width") is not None:
                set_api_input(node, "width", int(params["width"]))
            if params.get("height") is not None:
                set_api_input(node, "height", int(params["height"]))
    prefix = params.get("filename_prefix")
    if prefix:
        for _, node in api_nodes_by_type(prompt_graph, "SaveImageAdvanced"):
            set_api_input(node, "filename_prefix", str(prefix))
    if guide_image:
        encoders = api_nodes_by_type(prompt_graph, "TextEncodeQwenImage21")
        vaes = api_nodes_by_type(prompt_graph, "VAELoader")
        if encoders:
            load_id = "qwen_guide_load"
            prompt_graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": guide_image},
            }
            enc_id, enc = encoders[0]
            inp = enc.setdefault("inputs", {})
            if not isinstance(inp, dict):
                inp = {}
                enc["inputs"] = inp
            # execute() takes autogrow ``images`` (image_1..image_16), not a
            # top-level image_1 kwarg. Visual socket name is images.image_1.
            inp.pop("image_1", None)
            inp["images.image_1"] = [load_id, 0]
            if vaes:
                inp["vae"] = [vaes[0][0], 0]
            for _, sc in api_nodes_by_type(prompt_graph, "SamplerCustom"):
                set_api_input(sc, "latent_image", [enc_id, 2])


def apply_job_params(
    visual: dict,
    prompt: str,
    params: Dict[str, Any],
    guide_image: Optional[str] = None,
) -> None:
    patch_subgraph(visual, prompt, params)
    patch_resolution(visual, params)
    patch_inner_nodes(visual, prompt, params)
    if guide_image:
        inject_guide_image(visual, guide_image)
