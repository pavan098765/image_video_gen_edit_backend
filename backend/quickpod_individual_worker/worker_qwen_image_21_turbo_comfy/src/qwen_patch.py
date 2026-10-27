#!/usr/bin/env python3
"""Patch the Qwen Image 2.1 Turbo 8-step visual workflow for a /run job."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("qwen_patch")

SUBGRAPH_TYPE = "c291ceec-b98f-4751-9d0b-7bc288f27b30"
SAVE_NODE_ID = 461
MAX_REFERENCE_IMAGES = 10
DEFAULT_UNET = "qwen-image-2.1-turbo-Q8_0.gguf"
DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"

# Official Qwen Image 2.1 native 2K canvases (multiples of 32).
ASPECT_2K: Dict[str, Tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# ResolutionSelector combo labels in ComfyUI v0.37.
ASPECT_SELECTOR: Dict[str, str] = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}


def _snap32(value: float) -> int:
    return max(32, int(round(float(value) / 32.0) * 32))


def _canonical_aspect(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower().replace(" ", "")
    aliases = {
        "square": "1:1",
        "portrait": "9:16",
        "landscape": "16:9",
        "widescreen": "16:9",
        "ultrawide": "21:9",
    }
    if lowered in aliases:
        return aliases[lowered]
    token = text.split()[0].replace("×", ":").replace("x", ":").replace("X", ":")
    match = re.match(r"^(\d+)\s*:\s*(\d+)$", token)
    if not match:
        return None
    return f"{int(match.group(1))}:{int(match.group(2))}"


def _size_tier(params: Dict[str, Any]) -> str:
    for key in ("size", "output_size", "resolution_tier", "quality"):
        raw = params.get(key)
        if raw is None:
            continue
        token = str(raw).strip().lower().replace(" ", "")
        if token in ("2k", "2048", "4mp", "4"):
            return "2k"
        if token in ("1k", "1024", "1mp", "1"):
            return "1k"
    megapixels = params.get("megapixels")
    if megapixels is not None:
        try:
            if float(megapixels) >= 3.0:
                return "2k"
        except (TypeError, ValueError):
            pass
    return "1k"


def _pixels_for_aspect(aspect: str, tier: str) -> Tuple[int, int]:
    if aspect in ASPECT_2K:
        width, height = ASPECT_2K[aspect]
        if tier == "2k":
            return width, height
        return _snap32(width / 2.0), _snap32(height / 2.0)
    parts = aspect.split(":")
    wr, hr = int(parts[0]), int(parts[1])
    area = (2048 * 2048) if tier == "2k" else (1024 * 1024)
    scale = math.sqrt(area / float(wr * hr))
    return _snap32(wr * scale), _snap32(hr * scale)


def _pixels_for_source(src_w: int, src_h: int, tier: str) -> Tuple[int, int]:
    area = (2048 * 2048) if tier == "2k" else (1024 * 1024)
    scale = math.sqrt(area / float(max(1, src_w * src_h)))
    return _snap32(src_w * scale), _snap32(src_h * scale)


def resolve_qwen_size(
    params: Dict[str, Any],
    guide_wh: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Pick canvas + encoder resolution for T2I vs image-edit."""
    tier = _size_tier(params)
    encode_res = 2048 if tier == "2k" else 1024
    if params.get("encode_resolution") is not None:
        try:
            encode_res = int(params["encode_resolution"])
        except (TypeError, ValueError):
            pass
    aspect = _canonical_aspect(params.get("aspect_ratio"))
    explicit_w = params.get("width")
    explicit_h = params.get("height")
    try:
        explicit_w = int(explicit_w) if explicit_w is not None else None
        explicit_h = int(explicit_h) if explicit_h is not None else None
    except (TypeError, ValueError):
        explicit_w, explicit_h = None, None

    if guide_wh:
        # Edits must sample at the encoder latent size (source aspect).
        # Changing EmptyLatentImage independently shifts the edit.
        src_w, src_h = int(guide_wh[0]), int(guide_wh[1])
        width, height = _pixels_for_source(src_w, src_h, tier)
        if aspect:
            logger.info(
                "Ignoring aspect_ratio=%s on edit; using source %sx%s -> %sx%s (%s)",
                aspect, src_w, src_h, width, height, tier,
            )
        return {
            "tier": tier,
            "aspect": f"{src_w}:{src_h}",
            "selector": None,
            "width": width,
            "height": height,
            "encode_resolution": encode_res,
            "megapixels": 4.0 if tier == "2k" else 1.0,
            "use_encoder_latent": True,
        }

    if aspect:
        width, height = _pixels_for_aspect(aspect, tier)
    elif explicit_w and explicit_h:
        if (explicit_w, explicit_h) in ((1024, 1024), (2048, 2048)) and tier == "2k":
            width, height = ASPECT_2K["1:1"]
            aspect = "1:1"
        elif (explicit_w, explicit_h) == (2048, 2048) and tier == "1k":
            width, height = 1024, 1024
            aspect = "1:1"
        else:
            width, height = _snap32(explicit_w), _snap32(explicit_h)
            aspect = f"{width}:{height}"
    else:
        aspect = "1:1"
        width, height = _pixels_for_aspect(aspect, tier)

    return {
        "tier": tier,
        "aspect": aspect,
        "selector": ASPECT_SELECTOR.get(aspect),
        "width": width,
        "height": height,
        "encode_resolution": encode_res,
        "megapixels": 4.0 if tier == "2k" else 1.0,
        "use_encoder_latent": False,
    }


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
    size_info = params.get("_qwen_size") if isinstance(params.get("_qwen_size"), dict) else None
    if size_info and not size_info.get("use_encoder_latent"):
        _set_named(node, "width", int(size_info["width"]))
        _set_named(node, "height", int(size_info["height"]))
    elif params.get("width") is not None and params.get("height") is not None and not (size_info or {}).get("use_encoder_latent"):
        _set_named(node, "width", int(params["width"]))
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
    size_info = params.get("_qwen_size") if isinstance(params.get("_qwen_size"), dict) else None
    if size_info:
        if size_info.get("selector"):
            _set_list_widget(node, 0, str(size_info["selector"]))
        _set_list_widget(node, 1, float(size_info["megapixels"]))
        _set_list_widget(node, 2, 32)
        return
    if params.get("aspect_ratio"):
        aspect = _canonical_aspect(params["aspect_ratio"])
        _set_list_widget(node, 0, ASPECT_SELECTOR.get(aspect or "", str(params["aspect_ratio"])))
    mp = params.get("megapixels")
    if mp is not None:
        _set_list_widget(node, 1, float(mp))
    _set_list_widget(node, 2, 32)


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
        size_info = params.get("_qwen_size") if isinstance(params.get("_qwen_size"), dict) else None
        enc_res = None
        if size_info:
            enc_res = size_info.get("encode_resolution")
        elif params.get("encode_resolution") is not None:
            enc_res = params["encode_resolution"]
        if enc_res is not None:
            _set_named(node, "resolution", int(enc_res))
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
    size_info = params.get("_qwen_size") if isinstance(params.get("_qwen_size"), dict) else None
    if size_info and not size_info.get("use_encoder_latent"):
        for node in _find_nodes_by_type(visual, "EmptyLatentImage"):
            _set_named(node, "width", int(size_info["width"]))
            _set_list_widget(node, 0, int(size_info["width"]))
            _set_named(node, "height", int(size_info["height"]))
            _set_list_widget(node, 1, int(size_info["height"]))
    elif params.get("width") is not None or params.get("height") is not None:
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


def _ensure_size(params: Dict[str, Any], guide_image: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(params.get("_qwen_size"), dict):
        return params["_qwen_size"]
    guide_wh = params.get("_guide_wh")
    parsed_wh = None
    if guide_wh and len(guide_wh) >= 2:
        try:
            parsed_wh = (int(guide_wh[0]), int(guide_wh[1]))
        except (TypeError, ValueError):
            parsed_wh = None
    size_info = resolve_qwen_size(params, guide_wh=parsed_wh)
    if guide_image:
        size_info["use_encoder_latent"] = True
    params["_qwen_size"] = size_info
    logger.info(
        "Qwen canvas tier=%s aspect=%s %sx%s encode=%s encoder_latent=%s",
        size_info.get("tier"),
        size_info.get("aspect"),
        size_info.get("width"),
        size_info.get("height"),
        size_info.get("encode_resolution"),
        size_info.get("use_encoder_latent"),
    )
    return size_info


def _node_title(node: dict) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict) and meta.get("title"):
        return str(meta["title"])
    return ""


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (str, int))
    )


def _strip_reference_nodes(prompt_graph: dict) -> None:
    """Drop reference loaders so a text-to-image queue does not require files."""
    drop = []
    for nid, node in list(prompt_graph.items()):
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        title = _node_title(node)
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if ctype == "LoadImage":
            drop.append(str(nid))
            continue
        if ctype != "ComfySwitchNode" or title == "custom_size":
            continue
        has_false = "on_false" in inputs
        has_true = "on_true" in inputs
        if title.startswith("use_reference") or (has_true and not has_false):
            drop.append(str(nid))
    dropped = set(drop)
    for nid in drop:
        prompt_graph.pop(nid, None)
    for node in prompt_graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in list(inputs):
            value = inputs[key]
            if key == "image_1" or str(key).startswith("images.image_"):
                inputs.pop(key, None)
                continue
            if _is_link(value) and str(value[0]) in dropped:
                inputs.pop(key, None)


def _set_custom_canvas(prompt_graph: dict, custom: bool) -> None:
    from workflow_converter import api_nodes_by_type, set_api_input

    switched = False
    for _, node in api_nodes_by_type(prompt_graph, "ComfySwitchNode"):
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if _node_title(node) == "custom_size" or ("on_false" in inputs and "on_true" in inputs):
            set_api_input(node, "switch", bool(custom))
            switched = True
    if switched or custom:
        return
    encoders = api_nodes_by_type(prompt_graph, "TextEncodeQwenImage21")
    if not encoders:
        return
    enc_id = encoders[0][0]
    for _, sampler in api_nodes_by_type(prompt_graph, "SamplerCustom"):
        set_api_input(sampler, "latent_image", [enc_id, 2])


def _attach_references(prompt_graph: dict, filenames: List[str]) -> None:
    from workflow_converter import api_nodes_by_type

    encoders = api_nodes_by_type(prompt_graph, "TextEncodeQwenImage21")
    vaes = api_nodes_by_type(prompt_graph, "VAELoader")
    if not encoders or not filenames:
        return
    _, encoder = encoders[0]
    inputs = encoder.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
        encoder["inputs"] = inputs
    if vaes:
        inputs["vae"] = [vaes[0][0], 0]
    for index, filename in enumerate(filenames[:MAX_REFERENCE_IMAGES], start=1):
        load_id = f"ref_load_{index}"
        prompt_graph[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
            "_meta": {"title": f"reference_{index}"},
        }
        inputs.pop("image_1", None)
        inputs[f"images.image_{index}"] = [load_id, 0]


def apply_api_job_params(
    prompt_graph: dict,
    prompt: str,
    params: Dict[str, Any],
    guide_image: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
    custom_canvas: Optional[bool] = None,
) -> None:
    """Patch a ComfyUI API-format prompt (no flatten)."""
    from workflow_converter import api_nodes_by_type, set_api_input

    refs = [str(name) for name in (reference_images or []) if name]
    if not refs and guide_image:
        refs = [str(guide_image)]
    if len(refs) > MAX_REFERENCE_IMAGES:
        logger.warning(
            "Qwen reference list truncated from %d to %d",
            len(refs),
            MAX_REFERENCE_IMAGES,
        )
        refs = refs[:MAX_REFERENCE_IMAGES]
    if custom_canvas is None:
        custom_canvas = not bool(refs)
    size_guide = refs[0] if refs and not custom_canvas else None
    size_info = _ensure_size(params, size_guide)
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
    for _, node in api_nodes_by_type(prompt_graph, "TextEncodeQwenImage21"):
        set_api_input(node, "prompt", text)
        neg = params.get("negative_prompt")
        if neg is None:
            neg = params.get("negative")
        if neg is not None:
            set_api_input(node, "negative_prompt", neg)
        set_api_input(node, "resolution", int(size_info["encode_resolution"]))
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
        if size_info.get("selector"):
            set_api_input(node, "aspect_ratio", str(size_info["selector"]))
        set_api_input(node, "megapixels", float(size_info["megapixels"]))
        set_api_input(node, "multiple", 32)
    if not size_info.get("use_encoder_latent"):
        for _, node in api_nodes_by_type(prompt_graph, "EmptyLatentImage"):
            set_api_input(node, "width", int(size_info["width"]))
            set_api_input(node, "height", int(size_info["height"]))
    prefix = params.get("filename_prefix")
    if prefix:
        for _, node in api_nodes_by_type(prompt_graph, "SaveImageAdvanced"):
            set_api_input(node, "filename_prefix", str(prefix))
    _strip_reference_nodes(prompt_graph)
    if refs:
        _attach_references(prompt_graph, refs)
    _set_custom_canvas(prompt_graph, bool(custom_canvas))
    logger.info(
        "Qwen prompt mode=%s references=%d custom_canvas=%s",
        "reference" if refs else "text",
        len(refs),
        bool(custom_canvas),
    )


def apply_job_params(
    visual: dict,
    prompt: str,
    params: Dict[str, Any],
    guide_image: Optional[str] = None,
) -> None:
    _ensure_size(params, guide_image)
    patch_subgraph(visual, prompt, params)
    patch_resolution(visual, params)
    patch_inner_nodes(visual, prompt, params)
    if guide_image:
        inject_guide_image(visual, guide_image)
