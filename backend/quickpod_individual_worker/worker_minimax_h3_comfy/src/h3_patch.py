#!/usr/bin/env python3
"""Apply MiniMax H3 job params and Pixaroma group toggles on a visual workflow."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("h3_patch")

MODE_ALWAYS = 0
MODE_MUTE = 2
MODE_BYPASS = 4

# Pixaroma rectangle ids from minimaxH3SEEDHUNTERLatent_v21.json
GROUP_IDS = {
    "preview_1": "pg_mubh542l_25_1k1b",
    "preview_2": "pg_mubh8x9p_29_v0lx",
    "preview_3": "pg_mubh78tl_27_7cmg",
    "upscale": "pg_mubh34y9_23_xm6p",
    "single_pass": "pg_mubgr360_6_fqy4",
    "continuation": "pg_mubgb15a_0_rulr",
    "normal_refs": "pg_mubie224_44_u6a7",
    "ref_mods": "pg_mubgqbpo_5_2qmg",
    "rife": "pg_mubgn3ae_1_94ut",
    "first_frame": "pg_mubh11b8_19_9jwz",
    "last_frame": "pg_mubh1ikc_21_kd2s",
    "force_audio": "pg_mubgzhxp_17_x1rw",
    "low_vram": "pg_mubgw5r2_11_ykpa",
    "chunking": "pg_mubgwjs4_13_eeaz",
    "comfy_kitchen": "pg_mubgv5fe_7_8i6g",
    "sampler_shift": "pg_mubgvqgl_9_yxxj",
    "model_previewer": "pg_mubgxuul_15_trtv",
}

# Saved graph defaults: previews on, upscale off, turbo lora on, previewer off.
TOGGLE_DEFAULTS = {
    "preview_1": True,
    "preview_2": True,
    "preview_3": True,
    "upscale": False,
    "single_pass": False,
    "continuation": False,
    "ref_mods": False,
    "rife": False,
    "first_frame": False,
    "last_frame": False,
    "force_audio": False,
    "low_vram": False,
    "chunking": False,
    "comfy_kitchen": False,
    "sampler_shift": False,
    "model_previewer": False,
    "turbo_lora": True,
}

# Which PixaromaGroupSwitch action to use when disabling a group.
GROUP_OFF_MODE = {
    "continuation": MODE_MUTE,
    "normal_refs": MODE_MUTE,
}

H3_PROMPT_NODE_ID = 365
MEDIA_LOADER_NODE_ID = 416
SEED_NODE_ID = 16
TURBO_LORA_NODE_ID = 21
RIFE_NODE_ID = 165
VIDEO_SETTINGS_INSTANCE_ID = 22
SECOND_PASS_INSTANCE_ID = 135

PREVIEW_COMBINE_NODES = {"1": 18, "2": 134, "3": 144}
FINAL_COMBINE_NODE = 34


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _node_pos(node: dict) -> Optional[Tuple[float, float]]:
    pos = node.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None
    if isinstance(pos, dict):
        try:
            return float(pos.get("x", 0)), float(pos.get("y", 0))
        except (TypeError, ValueError):
            return None
    return None


def _in_group(pos: Tuple[float, float], group: dict) -> bool:
    try:
        x, y, w, h = float(group["x"]), float(group["y"]), float(group["w"]), float(group["h"])
    except (KeyError, TypeError, ValueError):
        return False
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def pixaroma_groups(visual: dict) -> Dict[str, dict]:
    groups = (visual.get("extra") or {}).get("pixaromaGroups") or []
    return {g["id"]: g for g in groups if isinstance(g, dict) and g.get("id")}


def set_group_enabled(visual: dict, group_id: str, enabled: bool) -> int:
    groups = pixaroma_groups(visual)
    group = groups.get(group_id)
    if not group:
        logger.warning("Pixaroma group %s not found", group_id)
        return 0
    off_mode = GROUP_OFF_MODE.get(
        next((k for k, v in GROUP_IDS.items() if v == group_id), ""),
        MODE_BYPASS,
    )
    target_mode = MODE_ALWAYS if enabled else off_mode
    n = 0
    for node in visual.get("nodes") or []:
        pos = _node_pos(node)
        if pos is None:
            continue
        if _in_group(pos, group):
            node["mode"] = target_mode
            n += 1
    return n


def _find_node(visual: dict, node_id: int) -> Optional[dict]:
    for node in visual.get("nodes") or []:
        try:
            if int(node.get("id")) == int(node_id):
                return node
        except (TypeError, ValueError):
            continue
    return None


def _nodes_from_subgraph(visual: dict, name: str) -> List[dict]:
    out = []
    for node in visual.get("nodes") or []:
        props = node.get("properties") or {}
        if props.get("_h3_subgraph") == name:
            out.append(node)
    return out


def _set_list_widget(node: dict, index: int, value: Any) -> None:
    wv = node.get("widgets_values")
    if isinstance(wv, list):
        while len(wv) <= index:
            wv.append(None)
        wv[index] = value
        node["widgets_values"] = wv
        return
    if isinstance(wv, dict):
        # fall through to named widgets if present
        return


def _set_named_widget(node: dict, name: str, value: Any) -> None:
    wv = node.get("widgets_values")
    if isinstance(wv, dict):
        wv[name] = value
        node["widgets_values"] = wv


def patch_prompt(visual: dict, prompt: str) -> None:
    node = _find_node(visual, H3_PROMPT_NODE_ID)
    if node is None:
        logger.warning("H3PromptIDE node %s missing", H3_PROMPT_NODE_ID)
        return
    wv = node.get("widgets_values")
    if isinstance(wv, list) and wv:
        wv[0] = prompt
    elif isinstance(wv, dict):
        wv["prompt"] = prompt
    else:
        node["widgets_values"] = [prompt]


def patch_seed(visual: dict, seed: int) -> None:
    node = _find_node(visual, SEED_NODE_ID)
    if node is None:
        return
    wv = node.get("widgets_values")
    if isinstance(wv, list) and wv:
        wv[0] = int(seed)
        if len(wv) > 1:
            wv[1] = "fixed"


def patch_turbo_lora(visual: dict, enabled: bool, strength: Optional[float] = None) -> None:
    node = _find_node(visual, TURBO_LORA_NODE_ID)
    if node is None:
        return
    wv = node.get("widgets_values")
    if not isinstance(wv, list):
        return
    for item in wv:
        if isinstance(item, dict) and "lora" in item:
            item["on"] = bool(enabled)
            if strength is not None:
                item["strength"] = float(strength)


def patch_rife_fps(visual: dict, target_fps: Any) -> None:
    node = _find_node(visual, RIFE_NODE_ID)
    if node is None:
        return
    try:
        fps = int(target_fps)
    except (TypeError, ValueError):
        return
    wv = node.get("widgets_values")
    if isinstance(wv, list) and len(wv) >= 2:
        wv[1] = fps


def patch_video_settings(visual: dict, params: Dict[str, Any]) -> None:
    vs_nodes = _nodes_from_subgraph(visual, "Video Settings")
    if not vs_nodes:
        inst = _find_node(visual, VIDEO_SETTINGS_INSTANCE_ID)
        if inst is None:
            return
        wv = inst.get("widgets_values")
        if not isinstance(wv, list):
            return
        mapping = [
            ("sampler_name", 0),
            ("scheduler", 1),
            ("steps", 2),
            ("megapixels", 3),
            ("aspect_ratio", 4),
            ("seconds", 5),
        ]
        for key, idx in mapping:
            if key in params and params[key] is not None:
                wv[idx] = params[key]
        return

    for node in vs_nodes:
        ntype = node.get("type")
        if ntype == "KSamplerSelect" and params.get("sampler_name") is not None:
            _set_list_widget(node, 0, params["sampler_name"])
        elif ntype == "BasicScheduler":
            if params.get("scheduler") is not None:
                _set_list_widget(node, 0, params["scheduler"])
            if params.get("steps") is not None:
                _set_list_widget(node, 1, params["steps"])
        elif ntype == "INTConstant" and params.get("steps") is not None:
            _set_list_widget(node, 0, params["steps"])
        elif ntype == "ResolutionSelector":
            if params.get("aspect_ratio") is not None:
                _set_list_widget(node, 0, params["aspect_ratio"])
            if params.get("megapixels") is not None:
                _set_list_widget(node, 1, params["megapixels"])
        elif ntype == "PrimitiveFloat" and params.get("seconds") is not None:
            _set_list_widget(node, 0, params["seconds"])


def patch_upscale_steps(visual: dict, steps: Any) -> None:
    try:
        steps_i = int(steps)
    except (TypeError, ValueError):
        return
    for node in _nodes_from_subgraph(visual, "2nd Pass"):
        if node.get("type") == "BasicScheduler":
            _set_list_widget(node, 1, steps_i)
    inst = _find_node(visual, SECOND_PASS_INSTANCE_ID)
    if inst is not None:
        wv = inst.get("widgets_values")
        if isinstance(wv, list) and len(wv) >= 3:
            wv[2] = steps_i


def patch_media_state(visual: dict, items: List[dict]) -> None:
    node = _find_node(visual, MEDIA_LOADER_NODE_ID)
    if node is None:
        logger.warning("MiniMaxH3MediaLoader node %s missing", MEDIA_LOADER_NODE_ID)
        return
    payload = json.dumps(items)
    wv = node.get("widgets_values")
    if isinstance(wv, list) and wv:
        wv[0] = payload
    elif isinstance(wv, dict):
        wv["media_state"] = payload
    else:
        node["widgets_values"] = [payload]


def apply_toggles(visual: dict, params: Dict[str, Any]) -> None:
    flags = dict(TOGGLE_DEFAULTS)
    for key in TOGGLE_DEFAULTS:
        if key in params:
            flags[key] = _as_bool(params.get(key), flags[key])

    if flags["continuation"]:
        flags["single_pass"] = True

    for key, group_id in GROUP_IDS.items():
        if key == "normal_refs":
            set_group_enabled(visual, group_id, not flags["continuation"])
            continue
        if key in flags:
            set_group_enabled(visual, group_id, flags[key])

    patch_turbo_lora(
        visual,
        flags["turbo_lora"],
        params.get("lora_strength"),
    )
    if params.get("target_fps") is not None:
        patch_rife_fps(visual, params.get("target_fps"))


def apply_job_params(
    visual: dict,
    prompt: str,
    params: Dict[str, Any],
    media_items: Optional[List[dict]] = None,
    apply_group_toggles: bool = True,
) -> dict:
    params = params or {}
    if prompt:
        patch_prompt(visual, prompt)
    if params.get("seed") is not None:
        try:
            patch_seed(visual, int(params["seed"]))
        except (TypeError, ValueError):
            pass
    patch_video_settings(visual, params)
    if params.get("upscale_steps") is not None:
        patch_upscale_steps(visual, params.get("upscale_steps"))
    if apply_group_toggles:
        apply_toggles(visual, params)
    else:
        patch_turbo_lora(
            visual,
            _as_bool(params.get("turbo_lora"), TOGGLE_DEFAULTS["turbo_lora"]),
            params.get("lora_strength"),
        )
        if params.get("target_fps") is not None:
            patch_rife_fps(visual, params.get("target_fps"))
    if media_items:
        patch_media_state(visual, media_items)
    return visual


def pick_output_node_id(params: Dict[str, Any]) -> int:
    params = params or {}
    if _as_bool(params.get("upscale"), False) or _as_bool(params.get("single_pass"), False):
        return FINAL_COMBINE_NODE
    preview = str(params.get("output_preview") or "1").strip()
    return PREVIEW_COMBINE_NODES.get(preview, PREVIEW_COMBINE_NODES["1"])
