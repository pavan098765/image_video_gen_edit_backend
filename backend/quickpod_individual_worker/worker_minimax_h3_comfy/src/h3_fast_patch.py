#!/usr/bin/env python3
"""Patch the Fast MiniMax H3 T2V graph (optional first/last frame = I2V / FLF2V)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("h3_fast_patch")

PROMPT_NODE_ID = 10
DURATION_NODE_ID = 8
RESOLUTION_NODE_ID = 7
SEED_NODE_ID = 11
SAMPLER_NODE_ID = 12
SCHEDULER_NODE_ID = 13
MANUAL_SIGMAS_NODE_ID = 30
SHIFT_NODE_ID = 5
CREATE_VIDEO_NODE_ID = 18
SAVE_VIDEO_NODE_ID = 19
I2V_NODE_ID = 10
LOAD_FIRST_ID = 40
LOAD_LAST_ID = 41
SAMPLER_ADV_NODE_ID = 15
VAE_DECODE_ID = 16
VAE_DECODE_AUDIO_ID = 17
CLIP_NODE_ID = 2
VIDEO_VAE_NODE_ID = 3
DURATION_MATH_NODE_ID = 9
NOISE_NODE_ID = 11
KSAMPLER_SELECT_ID = 12
SIGMA_SHIFT_NODE_ID = 5
UPSCALE_RES_NODE_ID = 50
UPSCALE_I2V_NODE_ID = 51
UPSCALE_PARAM_NODE_ID = 52
ULTIMATE_UPSCALE_NODE_ID = 53
DEFAULT_UPSCALE_MP = 1.0
UPSCALE_MODEL = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"

MODE_ALWAYS = 0
MODE_BYPASS = 4

SIGMAS_BY_STEPS = {
    4: "0.9999166, 0.9728326, 0.9230769, 0.8, 0.0",
    6: "0.9999166, 0.9868421, 0.9638554, 0.9230769, 0.8695652, 0.8, 0.0",
    8: "1.0, 0.9882352941, 0.9729729730, 0.9523809524, 0.9230769231, 0.8780487805, 0.8, 0.6315789474, 0.0",
}


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _find_node(visual: dict, node_id: int) -> Optional[dict]:
    for node in visual.get("nodes") or []:
        try:
            if int(node.get("id")) == int(node_id):
                return node
        except (TypeError, ValueError):
            continue
    return None


def _set_named_or_list(node: dict, name: str, index: int, value: Any) -> None:
    named = node.get("widgets_values_named")
    if isinstance(named, dict):
        named[name] = value
        node["widgets_values_named"] = named
    wv = node.get("widgets_values")
    if isinstance(wv, list):
        while len(wv) <= index:
            wv.append(None)
        wv[index] = value
        node["widgets_values"] = wv
    elif isinstance(wv, dict):
        wv[name] = value
        node["widgets_values"] = wv
    elif not named:
        node["widgets_values"] = [value]


def _next_ids(visual: dict) -> tuple:
    last_n = int(visual.get("last_node_id") or 0)
    last_l = int(visual.get("last_link_id") or 0)
    for node in visual.get("nodes") or []:
        try:
            last_n = max(last_n, int(node.get("id") or 0))
        except (TypeError, ValueError):
            pass
    for link in visual.get("links") or []:
        if isinstance(link, list) and link:
            try:
                last_l = max(last_l, int(link[0]))
            except (TypeError, ValueError):
                pass
    return last_n, last_l


def _ensure_load_image(
    visual: dict,
    node_id: int,
    filename: str,
    i2v_input_index: int,
    pos_y: float,
) -> None:
    node = _find_node(visual, node_id)
    if node is None:
        node = {
            "id": node_id,
            "type": "LoadImage",
            "pos": [400, pos_y],
            "size": [320, 80],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": None, "slot_index": 1},
            ],
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": [filename],
            "widgets_values_named": {"image": filename},
        }
        visual.setdefault("nodes", []).append(node)
        visual["last_node_id"] = max(int(visual.get("last_node_id") or 0), node_id)
    else:
        _set_named_or_list(node, "image", 0, filename)

    i2v = _find_node(visual, I2V_NODE_ID)
    if i2v is None:
        return
    inputs = i2v.get("inputs") or []
    if i2v_input_index >= len(inputs):
        return
    existing = inputs[i2v_input_index].get("link")
    links = visual.setdefault("links", [])
    if existing is not None:
        for link in links:
            if isinstance(link, list) and link and int(link[0]) == int(existing):
                link[1] = node_id
                link[2] = 0
                link[3] = I2V_NODE_ID
                link[4] = i2v_input_index
                return
    _, last_l = _next_ids(visual)
    link_id = last_l + 1
    links.append([link_id, node_id, 0, I2V_NODE_ID, i2v_input_index, "IMAGE"])
    inputs[i2v_input_index]["link"] = link_id
    visual["last_link_id"] = link_id
    out_links = (node.get("outputs") or [{}])[0].setdefault("links", [])
    if isinstance(out_links, list) and link_id not in out_links:
        out_links.append(link_id)


def _named_input(node: dict, name: str) -> Optional[dict]:
    for inp in node.get("inputs") or []:
        if inp.get("name") == name:
            return inp
    return None


def _add_link(
    visual: dict,
    src: int,
    src_slot: int,
    dst: int,
    dst_slot: int,
    type_str: str,
) -> int:
    _, last_l = _next_ids(visual)
    link_id = last_l + 1
    visual.setdefault("links", []).append([link_id, src, src_slot, dst, dst_slot, type_str])
    visual["last_link_id"] = link_id
    dst_node = _find_node(visual, dst)
    if dst_node is not None:
        inputs = dst_node.get("inputs") or []
        if dst_slot < len(inputs):
            inputs[dst_slot]["link"] = link_id
    src_node = _find_node(visual, src)
    if src_node is not None:
        outs = src_node.get("outputs") or []
        if src_slot < len(outs):
            sl = outs[src_slot].setdefault("links", [])
            if sl is None:
                outs[src_slot]["links"] = [link_id]
            elif link_id not in sl:
                sl.append(link_id)
    return link_id


def _remove_link(visual: dict, link_id: Optional[int]) -> None:
    if link_id is None:
        return
    lid = int(link_id)
    kept = []
    removed = None
    for ln in visual.get("links") or []:
        if isinstance(ln, list) and ln and int(ln[0]) == lid:
            removed = ln
            continue
        kept.append(ln)
    visual["links"] = kept
    if removed is None:
        return
    src_node = _find_node(visual, int(removed[1]))
    if src_node is not None:
        outs = src_node.get("outputs") or []
        src_slot = int(removed[2])
        if src_slot < len(outs):
            sl = outs[src_slot].get("links")
            if isinstance(sl, list) and lid in sl:
                sl.remove(lid)
    dst_node = _find_node(visual, int(removed[3]))
    if dst_node is not None:
        inputs = dst_node.get("inputs") or []
        dst_slot = int(removed[4])
        if dst_slot < len(inputs) and inputs[dst_slot].get("link") == lid:
            inputs[dst_slot]["link"] = None


def _copy_incoming(
    visual: dict,
    src_node_id: int,
    src_input: str,
    dst_node_id: int,
    dst_slot: int,
    type_str: str,
) -> None:
    src = _find_node(visual, src_node_id)
    dst = _find_node(visual, dst_node_id)
    if src is None or dst is None:
        return
    inp = _named_input(src, src_input)
    if inp is None or inp.get("link") is None:
        return
    old = int(inp["link"])
    for ln in visual.get("links") or []:
        if isinstance(ln, list) and ln and int(ln[0]) == old:
            existing = (dst.get("inputs") or [{}])[dst_slot].get("link") if dst_slot < len(dst.get("inputs") or []) else None
            if existing is not None:
                return
            _add_link(visual, int(ln[1]), int(ln[2]), dst_node_id, dst_slot, type_str)
            return


def _sigmas_source(visual: dict) -> tuple:
    sampler = _find_node(visual, SAMPLER_ADV_NODE_ID)
    if sampler is None:
        return MANUAL_SIGMAS_NODE_ID, 0
    inp = _named_input(sampler, "sigmas")
    if inp is None or inp.get("link") is None:
        return MANUAL_SIGMAS_NODE_ID, 0
    lid = int(inp["link"])
    for ln in visual.get("links") or []:
        if isinstance(ln, list) and ln and int(ln[0]) == lid:
            return int(ln[1]), int(ln[2])
    return MANUAL_SIGMAS_NODE_ID, 0


def _enable_ultimate_upscale(visual: dict, params: Dict[str, Any]) -> None:
    """Second pass: 0.4 MP sample -> MMH3 Ultimate Upscale at 0.6-1.0 MP."""
    first_i2v = _find_node(visual, I2V_NODE_ID)
    first_res = _find_node(visual, RESOLUTION_NODE_ID)
    if first_i2v is None or first_res is None:
        logger.warning("Fast H3 Ultimate Upscale skipped: missing I2V/resolution nodes")
        return

    second_mp = params.get("upscale_megapixels")
    if second_mp is None:
        second_mp = params.get("megapixels_upscale")
    if second_mp is None:
        second_mp = DEFAULT_UPSCALE_MP
    try:
        second_mp = float(second_mp)
    except (TypeError, ValueError):
        second_mp = DEFAULT_UPSCALE_MP

    aspect = "16:9 (Widescreen)"
    named = first_res.get("widgets_values_named") or {}
    if isinstance(named, dict) and named.get("aspect_ratio"):
        aspect = named["aspect_ratio"]
    elif isinstance(first_res.get("widgets_values"), list) and first_res["widgets_values"]:
        aspect = first_res["widgets_values"][0]
    if params.get("aspect_ratio") is not None:
        aspect = params["aspect_ratio"]

    model_name = str(params.get("upscale_model") or UPSCALE_MODEL)

    res = _find_node(visual, UPSCALE_RES_NODE_ID)
    if res is None:
        res = {
            "id": UPSCALE_RES_NODE_ID,
            "type": "ResolutionSelector",
            "pos": [384, 480],
            "size": [434, 126],
            "flags": {},
            "order": 6,
            "mode": MODE_ALWAYS,
            "inputs": [],
            "outputs": [
                {"name": "width", "type": "INT", "links": [], "slot_index": 0},
                {"name": "height", "type": "INT", "links": [], "slot_index": 1},
            ],
            "title": "Resolution 2  -  Ultimate Upscale (0.6-1.0 MP)",
            "properties": {"Node name for S&R": "ResolutionSelector"},
            "widgets_values": [aspect, second_mp, 32],
            "widgets_values_named": {
                "aspect_ratio": aspect,
                "megapixels": second_mp,
                "multiple": 32,
            },
        }
        visual.setdefault("nodes", []).append(res)
    else:
        _set_named_or_list(res, "aspect_ratio", 0, aspect)
        _set_named_or_list(res, "megapixels", 1, second_mp)
        _set_named_or_list(res, "multiple", 2, 32)

    prompt_wv = (first_i2v.get("widgets_values_named") or {}).get("prompt")
    if prompt_wv is None and isinstance(first_i2v.get("widgets_values"), list) and first_i2v["widgets_values"]:
        prompt_wv = first_i2v["widgets_values"][0]
    prompt_wv = prompt_wv or ""

    i2v2 = _find_node(visual, UPSCALE_I2V_NODE_ID)
    if i2v2 is None:
        i2v2 = {
            "id": UPSCALE_I2V_NODE_ID,
            "type": "MiniMaxH3ImageToVideo",
            "pos": [950, 760],
            "size": [470, 260],
            "flags": {},
            "order": 13,
            "mode": MODE_ALWAYS,
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": None},
                {"name": "vae", "type": "VAE", "link": None},
                {"name": "first_frame", "shape": 7, "type": "IMAGE", "link": None},
                {"name": "last_frame", "shape": 7, "type": "IMAGE", "link": None},
                {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
                {"name": "length", "type": "INT", "widget": {"name": "length"}, "link": None},
            ],
            "outputs": [
                {"name": "positive", "type": "CONDITIONING", "links": [], "slot_index": 0},
                {"name": "LATENT", "type": "LATENT", "links": None, "slot_index": 1},
            ],
            "title": "MiniMax H3 Image to Video  (Ultimate Upscale conditioning)",
            "properties": {"Node name for S&R": "MiniMaxH3ImageToVideo"},
            "widgets_values": [prompt_wv, 1280, 704, 56],
            "widgets_values_named": {
                "prompt": prompt_wv,
                "width": 1280,
                "height": 704,
                "length": 56,
            },
        }
        visual.setdefault("nodes", []).append(i2v2)
        _add_link(visual, CLIP_NODE_ID, 0, UPSCALE_I2V_NODE_ID, 0, "CLIP")
        _add_link(visual, VIDEO_VAE_NODE_ID, 0, UPSCALE_I2V_NODE_ID, 1, "VAE")
        _add_link(visual, UPSCALE_RES_NODE_ID, 0, UPSCALE_I2V_NODE_ID, 4, "INT")
        _add_link(visual, UPSCALE_RES_NODE_ID, 1, UPSCALE_I2V_NODE_ID, 5, "INT")
        _add_link(visual, DURATION_MATH_NODE_ID, 1, UPSCALE_I2V_NODE_ID, 6, "INT")
    else:
        _set_named_or_list(i2v2, "prompt", 0, prompt_wv)

    _copy_incoming(visual, I2V_NODE_ID, "first_frame", UPSCALE_I2V_NODE_ID, 2, "IMAGE")
    _copy_incoming(visual, I2V_NODE_ID, "last_frame", UPSCALE_I2V_NODE_ID, 3, "IMAGE")

    param = _find_node(visual, UPSCALE_PARAM_NODE_ID)
    if param is None:
        param = {
            "id": UPSCALE_PARAM_NODE_ID,
            "type": "MMH3LatentUpscaleWithModelParams",
            "pos": [1550, 420],
            "size": [320, 180],
            "flags": {},
            "order": 14,
            "mode": MODE_ALWAYS,
            "inputs": [
                {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            ],
            "outputs": [
                {
                    "name": "latent_upscale_param",
                    "type": "H3_UPSCALE_PARAM",
                    "links": [],
                    "slot_index": 0,
                }
            ],
            "properties": {
                "cnr_id": "Comfyui-MMH3-UltimateUpscale",
                "Node name for S&R": "MMH3LatentUpscaleWithModelParams",
                "aux_id": "bbaudio-2025/Comfyui-MMH3-UltimateUpscale",
            },
            "widgets_values": [model_name, 1280, 704, "cuda", "fp16", True],
            "widgets_values_named": {
                "model_name": model_name,
                "width": 1280,
                "height": 704,
                "device": "cuda",
                "precision": "fp16",
                "offload_model": True,
            },
        }
        visual.setdefault("nodes", []).append(param)
        _add_link(visual, UPSCALE_RES_NODE_ID, 0, UPSCALE_PARAM_NODE_ID, 0, "INT")
        _add_link(visual, UPSCALE_RES_NODE_ID, 1, UPSCALE_PARAM_NODE_ID, 1, "INT")
    else:
        _set_named_or_list(param, "model_name", 0, model_name)

    ultimate = _find_node(visual, ULTIMATE_UPSCALE_NODE_ID)
    if ultimate is None:
        ultimate = {
            "id": ULTIMATE_UPSCALE_NODE_ID,
            "type": "MMH3UltimateUpscale",
            "pos": [2018, 360],
            "size": [280, 360],
            "flags": {},
            "order": 18,
            "mode": MODE_ALWAYS,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": None},
                {"name": "conditioning", "type": "CONDITIONING", "link": None},
                {"name": "latent", "type": "LATENT", "link": None},
                {"name": "noise", "type": "NOISE", "link": None},
                {"name": "sampler", "type": "SAMPLER", "link": None},
                {"name": "sigmas", "type": "SIGMAS", "link": None},
                {"name": "negative", "shape": 7, "type": "CONDITIONING", "link": None},
                {"name": "latent_upscale_param", "shape": 7, "type": "H3_UPSCALE_PARAM", "link": None},
                {"name": "temporal_split_param", "shape": 7, "type": "H3_TEMPORAL_PARAM", "link": None},
                {"name": "spatial_split_param", "shape": 7, "type": "H3_SPATIAL_PARAM", "link": None},
                {"name": "fun_control_param", "shape": 7, "type": "H3_FUN_CONTROL_PARAM", "link": None},
                {"name": "inpaint_param", "shape": 7, "type": "H3_INPAINT_PARAM", "link": None},
            ],
            "outputs": [
                {"name": "latent", "type": "LATENT", "links": [], "slot_index": 0},
                {"name": "segments_info", "type": "DICT", "links": None, "slot_index": 1},
                {"name": "tiles_info", "type": "DICT", "links": None, "slot_index": 2},
            ],
            "properties": {
                "cnr_id": "Comfyui-MMH3-UltimateUpscale",
                "Node name for S&R": "MMH3UltimateUpscale",
                "aux_id": "bbaudio-2025/Comfyui-MMH3-UltimateUpscale",
            },
            "widgets_values": [1.0],
            "widgets_values_named": {"cfg": 1.0},
        }
        visual.setdefault("nodes", []).append(ultimate)
        sig_src, sig_slot = _sigmas_source(visual)
        _add_link(visual, SIGMA_SHIFT_NODE_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 0, "MODEL")
        _add_link(visual, UPSCALE_I2V_NODE_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 1, "CONDITIONING")
        _add_link(visual, SAMPLER_ADV_NODE_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 2, "LATENT")
        _add_link(visual, NOISE_NODE_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 3, "NOISE")
        _add_link(visual, KSAMPLER_SELECT_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 4, "SAMPLER")
        _add_link(visual, sig_src, sig_slot, ULTIMATE_UPSCALE_NODE_ID, 5, "SIGMAS")
        _add_link(
            visual,
            UPSCALE_PARAM_NODE_ID,
            0,
            ULTIMATE_UPSCALE_NODE_ID,
            7,
            "H3_UPSCALE_PARAM",
        )

    decode = _find_node(visual, VAE_DECODE_ID)
    decode_a = _find_node(visual, VAE_DECODE_AUDIO_ID)
    for node in (decode, decode_a):
        if node is None:
            continue
        samples = _named_input(node, "samples")
        if samples is None:
            continue
        old = samples.get("link")
        already = False
        if old is not None:
            for ln in visual.get("links") or []:
                if isinstance(ln, list) and ln and int(ln[0]) == int(old):
                    if int(ln[1]) == ULTIMATE_UPSCALE_NODE_ID:
                        already = True
                    break
        if already:
            continue
        _remove_link(visual, old)
        slot = 0 if node is decode else 0
        _add_link(
            visual,
            ULTIMATE_UPSCALE_NODE_ID,
            0,
            int(node["id"]),
            slot,
            "LATENT",
        )

    visual["last_node_id"] = max(int(visual.get("last_node_id") or 0), ULTIMATE_UPSCALE_NODE_ID)
    logger.info("Fast H3 Ultimate Upscale enabled at %.2f MP (%s)", second_mp, model_name)


def _rewire_sigmas_from_scheduler(visual: dict) -> None:
    """Use BasicScheduler instead of ManualSigmas (non 4/6/8 step counts)."""
    sched = _find_node(visual, SCHEDULER_NODE_ID)
    manual = _find_node(visual, MANUAL_SIGMAS_NODE_ID)
    sampler = _find_node(visual, 15)
    if sched is None or sampler is None:
        return
    sched["mode"] = MODE_ALWAYS
    if manual is not None:
        manual["mode"] = MODE_BYPASS
    sig_inp = None
    for inp in sampler.get("inputs") or []:
        if inp.get("name") == "sigmas":
            sig_inp = inp
            break
    if sig_inp is None:
        return
    old = sig_inp.get("link")
    links = visual.setdefault("links", [])
    if old is not None:
        visual["links"] = [
            ln for ln in links
            if not (isinstance(ln, list) and ln and int(ln[0]) == int(old))
        ]
        links = visual["links"]
    _, last_l = _next_ids(visual)
    link_id = last_l + 1
    links.append([link_id, SCHEDULER_NODE_ID, 0, 15, 3, "SIGMAS"])
    sig_inp["link"] = link_id
    visual["last_link_id"] = link_id
    outs = sched.get("outputs") or []
    if outs:
        sl = outs[0].setdefault("links", [])
        if isinstance(sl, list) and link_id not in sl:
            sl.append(link_id)
    ult = _find_node(visual, ULTIMATE_UPSCALE_NODE_ID)
    if ult is not None:
        ult_sig = _named_input(ult, "sigmas")
        if ult_sig is not None:
            _remove_link(visual, ult_sig.get("link"))
            _add_link(visual, SCHEDULER_NODE_ID, 0, ULTIMATE_UPSCALE_NODE_ID, 5, "SIGMAS")


def _linked_node_id(node: dict, slot_name: str) -> Optional[str]:
    link = (node.get("inputs") or {}).get(slot_name)
    if isinstance(link, list) and link:
        return str(link[0])
    return None


def _ensure_api_load_image(
    prompt_graph: dict,
    node_id: str,
    filename: str,
    title: str,
) -> str:
    from workflow_converter import set_api_input

    existing = prompt_graph.get(node_id)
    if isinstance(existing, dict) and existing.get("class_type") == "LoadImage":
        set_api_input(existing, "image", filename)
        return node_id
    prompt_graph[node_id] = {
        "inputs": {"image": filename},
        "class_type": "LoadImage",
        "_meta": {"title": title},
    }
    return node_id


def _wire_i2v_frames(prompt_graph: dict, media_files: Dict[str, str]) -> None:
    """Attach optional first and/or last frames on MiniMaxH3ImageToVideo.

    The fast API graph has those inputs unconnected and no LoadImage. Image
    jobs inject loaders here so they stay on the fused turbo graph. An older
    graph that already links a LoadImage has that node's filename updated
    instead of leaving a placeholder image in the prompt.
    """
    from workflow_converter import api_nodes_by_type, set_api_input

    i2v_nodes = api_nodes_by_type(prompt_graph, "MiniMaxH3ImageToVideo")
    if not i2v_nodes:
        return
    first = media_files.get("first_frame") or media_files.get("init_image")
    last = media_files.get("last_frame")
    if not first and not last:
        return

    if first:
        linked = _linked_node_id(i2v_nodes[0][1], "first_frame")
        first_id = _ensure_api_load_image(
            prompt_graph,
            linked or "h3_first_image",
            first,
            "First frame",
        )
        for _, node in i2v_nodes:
            set_api_input(node, "first_frame", [first_id, 0])
    if last:
        linked = _linked_node_id(i2v_nodes[0][1], "last_frame")
        if linked and linked == _linked_node_id(i2v_nodes[0][1], "first_frame"):
            linked = None
        last_id = _ensure_api_load_image(
            prompt_graph,
            linked or "h3_last_image",
            last,
            "Last frame",
        )
        for _, node in i2v_nodes:
            set_api_input(node, "last_frame", [last_id, 0])


def apply_fast_api_job_params(
    prompt_graph: dict,
    prompt: str,
    params: Dict[str, Any],
    media_files: Optional[Dict[str, str]] = None,
) -> None:
    """Patch Fast T2V or Fast I2V API graphs by class_type."""
    from workflow_converter import api_nodes_by_type, set_api_input

    params = params or {}
    media_files = media_files or {}
    if prompt:
        for _, node in api_nodes_by_type(prompt_graph, "MiniMaxH3ImageToVideo"):
            set_api_input(node, "prompt", prompt)
    if params.get("seed") is not None:
        try:
            seed_i = int(params["seed"])
        except (TypeError, ValueError):
            seed_i = None
        if seed_i is not None:
            for _, node in api_nodes_by_type(prompt_graph, "RandomNoise"):
                set_api_input(node, "noise_seed", seed_i)
    if params.get("seconds") is not None:
        try:
            seconds = float(params["seconds"])
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            for _, node in api_nodes_by_type(prompt_graph, "PrimitiveFloat"):
                set_api_input(node, "value", seconds)
    megapixels = None
    if params.get("megapixels") is not None:
        try:
            megapixels = float(params["megapixels"])
        except (TypeError, ValueError):
            megapixels = None
    for _, node in api_nodes_by_type(prompt_graph, "ResolutionSelector"):
        if params.get("aspect_ratio") is not None:
            set_api_input(node, "aspect_ratio", params["aspect_ratio"])
        if megapixels is not None:
            set_api_input(node, "megapixels", megapixels)
    if megapixels is not None:
        for _, node in api_nodes_by_type(prompt_graph, "ImageScaleToTotalPixels"):
            set_api_input(node, "megapixels", megapixels)
    if params.get("sampler_name") is not None:
        for _, node in api_nodes_by_type(prompt_graph, "KSamplerSelect"):
            set_api_input(node, "sampler_name", params["sampler_name"])
    if params.get("fps") is not None:
        for _, node in api_nodes_by_type(prompt_graph, "CreateVideo"):
            set_api_input(node, "fps", int(params["fps"]))
    _wire_i2v_frames(prompt_graph, media_files)
    prefix = params.get("filename_prefix")
    if prefix:
        for _, node in api_nodes_by_type(prompt_graph, "SaveVideo"):
            set_api_input(node, "filename_prefix", str(prefix))
    sigmas = params.get("sigmas")
    steps = params.get("steps")
    steps_i = None
    if steps is not None:
        try:
            steps_i = int(steps)
        except (TypeError, ValueError):
            steps_i = None
    if isinstance(sigmas, str) and sigmas.strip():
        for _, node in api_nodes_by_type(prompt_graph, "ManualSigmas"):
            set_api_input(node, "sigmas", sigmas.strip())
    elif steps_i in SIGMAS_BY_STEPS:
        for _, node in api_nodes_by_type(prompt_graph, "ManualSigmas"):
            set_api_input(node, "sigmas", SIGMAS_BY_STEPS[steps_i])
    if steps_i is not None and steps_i > 0:
        for _, node in api_nodes_by_type(prompt_graph, "BasicScheduler"):
            set_api_input(node, "steps", steps_i)
    shift = api_nodes_by_type(prompt_graph, "MiniMaxH3SigmaShift")
    if shift:
        node = shift[0][1]
        if params.get("shift_video") is not None:
            set_api_input(node, "shift_video", float(params["shift_video"]))
        if params.get("shift_audio") is not None:
            set_api_input(node, "shift_audio", float(params["shift_audio"]))


def apply_fast_job_params(
    visual: dict,
    prompt: str,
    params: Dict[str, Any],
    media_files: Optional[Dict[str, str]] = None,
) -> dict:
    params = params or {}
    media_files = media_files or {}

    i2v = _find_node(visual, PROMPT_NODE_ID)
    if i2v is not None and prompt:
        _set_named_or_list(i2v, "prompt", 0, prompt)

    if params.get("seed") is not None:
        seed_node = _find_node(visual, SEED_NODE_ID)
        if seed_node is not None:
            try:
                seed_i = int(params["seed"])
            except (TypeError, ValueError):
                seed_i = None
            if seed_i is not None:
                _set_named_or_list(seed_node, "noise_seed", 0, seed_i)
                _set_named_or_list(seed_node, "control_after_generate", 1, "fixed")

    if params.get("seconds") is not None:
        dur = _find_node(visual, DURATION_NODE_ID)
        if dur is not None:
            try:
                _set_named_or_list(dur, "value", 0, float(params["seconds"]))
            except (TypeError, ValueError):
                pass

    res = _find_node(visual, RESOLUTION_NODE_ID)
    if res is not None:
        if params.get("aspect_ratio") is not None:
            _set_named_or_list(res, "aspect_ratio", 0, params["aspect_ratio"])
        if params.get("megapixels") is not None:
            try:
                _set_named_or_list(res, "megapixels", 1, float(params["megapixels"]))
            except (TypeError, ValueError):
                pass

    if params.get("sampler_name") is not None:
        samp = _find_node(visual, SAMPLER_NODE_ID)
        if samp is not None:
            _set_named_or_list(samp, "sampler_name", 0, params["sampler_name"])

    steps = params.get("steps")
    steps_i = None
    if steps is not None:
        try:
            steps_i = int(steps)
        except (TypeError, ValueError):
            steps_i = None

    sigmas = params.get("sigmas")
    if isinstance(sigmas, str) and sigmas.strip():
        man = _find_node(visual, MANUAL_SIGMAS_NODE_ID)
        if man is not None:
            man["mode"] = MODE_ALWAYS
            _set_named_or_list(man, "sigmas", 0, sigmas.strip())
        sched = _find_node(visual, SCHEDULER_NODE_ID)
        if sched is not None:
            sched["mode"] = MODE_BYPASS
    elif steps_i in SIGMAS_BY_STEPS:
        man = _find_node(visual, MANUAL_SIGMAS_NODE_ID)
        if man is not None:
            man["mode"] = MODE_ALWAYS
            _set_named_or_list(man, "sigmas", 0, SIGMAS_BY_STEPS[steps_i])
        sched = _find_node(visual, SCHEDULER_NODE_ID)
        if sched is not None:
            sched["mode"] = MODE_BYPASS
            _set_named_or_list(sched, "steps", 1, steps_i)
    elif steps_i is not None:
        sched = _find_node(visual, SCHEDULER_NODE_ID)
        if sched is not None:
            _set_named_or_list(sched, "scheduler", 0, params.get("scheduler") or "simple")
            _set_named_or_list(sched, "steps", 1, steps_i)
        _rewire_sigmas_from_scheduler(visual)

    shift = _find_node(visual, SHIFT_NODE_ID)
    if shift is not None:
        if params.get("shift_video") is not None:
            try:
                _set_named_or_list(shift, "shift_video", 0, float(params["shift_video"]))
            except (TypeError, ValueError):
                pass
        if params.get("shift_audio") is not None:
            try:
                _set_named_or_list(shift, "shift_audio", 1, float(params["shift_audio"]))
            except (TypeError, ValueError):
                pass

    if params.get("fps") is not None:
        cv = _find_node(visual, CREATE_VIDEO_NODE_ID)
        if cv is not None:
            try:
                _set_named_or_list(cv, "fps", 0, int(params["fps"]))
            except (TypeError, ValueError):
                pass

    first = media_files.get("first_frame") or media_files.get("init_image")
    last = media_files.get("last_frame")
    if first:
        _ensure_load_image(visual, LOAD_FIRST_ID, first, 2, 80)
    if last:
        _ensure_load_image(visual, LOAD_LAST_ID, last, 3, 260)

    if _truthy(params.get("upscale")):
        _enable_ultimate_upscale(visual, params)

    return visual
