#!/usr/bin/env python3
"""Run MiniMax H3 ComfyUI workflows (Fast T2V default, Seed Hunter for extras)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("h3_inference")

WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
FAST_WORKFLOW = "fastMinimaxH3_textToVideo.json"
SEED_HUNTER_WORKFLOW = "minimaxH3SEEDHUNTERLatent_v21.json"
DEFAULT_WORKFLOW = os.environ.get("H3_WORKFLOW_FILE", FAST_WORKFLOW)

SEED_HUNTER_PARAM_FLAGS = (
    "continuation",
    "rife",
    "ref_mods",
    "single_pass",
    "force_audio",
    "chunking",
    "low_vram",
    "comfy_kitchen",
    "sampler_shift",
    "model_previewer",
)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def pick_workflow_file(params: Dict[str, Any]) -> str:
    """Fast T2V/I2V/FLF2V by default; Seed Hunter when leftover extras are requested."""
    params = params or {}
    raw = str(params.get("workflow") or "").strip()
    key = raw.lower()
    if key in ("fast", "fast_minimax", "t2v_fast", FAST_WORKFLOW.lower()):
        return FAST_WORKFLOW
    if key in ("seed_hunter", "seedhunter", "hunter", "quality", SEED_HUNTER_WORKFLOW.lower()):
        return SEED_HUNTER_WORKFLOW
    if raw.endswith(".json"):
        return raw
    for flag in SEED_HUNTER_PARAM_FLAGS:
        if _truthy(params.get(flag)):
            return SEED_HUNTER_WORKFLOW
    preview = str(params.get("output_preview") or "1").strip()
    if preview in ("2", "3"):
        return SEED_HUNTER_WORKFLOW
    if params.get("continuation_video_url"):
        return SEED_HUNTER_WORKFLOW
    return DEFAULT_WORKFLOW if DEFAULT_WORKFLOW else FAST_WORKFLOW


def _ext_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext and len(ext) <= 8:
        return ext
    return fallback


def _kind_for_field(field: str) -> str:
    if "audio" in field:
        return "audio"
    if "video" in field or "continuation" in field:
        return "video"
    return "picture"


def download_media_items(params: Dict[str, Any], comfyui_bridge: Any) -> List[dict]:
    items: List[dict] = []
    mapping = [
        ("first_frame_url", "picture", "h3_first"),
        ("last_frame_url", "picture", "h3_last"),
        ("audio_url", "audio", "h3_audio"),
        ("continuation_video_url", "video", "h3_cont"),
        ("init_image", "picture", "h3_init"),
        ("init_video", "video", "h3_initvid"),
    ]
    seen = set()
    for field, kind, prefix in mapping:
        url = params.get(field)
        if not url or not isinstance(url, str) or not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        ext = _ext_from_url(url, ".png" if kind == "picture" else (".mp3" if kind == "audio" else ".mp4"))
        filename = f"{prefix}{ext}"
        logger.info("Downloading media %s -> %s", field, filename)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        comfyui_bridge.save_bytes_to_input(resp.content, filename)
        items.append({"kind": kind, "file": filename, "enabled": True})
    return items


def _pick_output(
    comfyui_bridge: Any,
    output_files: List[dict],
    prefer_node_id: Optional[int],
) -> str:
    if not output_files:
        raise RuntimeError("ComfyUI produced no output files")
    mp4s = [
        f
        for f in output_files
        if str(f.get("filename") or "").lower().endswith(".mp4")
        or str(f.get("format") or "").find("mp4") >= 0
    ]
    candidates = mp4s or output_files
    chosen = None
    if prefer_node_id is not None:
        for item in candidates:
            if str(item.get("node_id")) == str(prefer_node_id):
                chosen = item
                break
    if chosen is None:
        saved = [f for f in candidates if f.get("type") == "output"]
        chosen = saved[-1] if saved else candidates[-1]
    src = comfyui_bridge.get_output_file_path(chosen)
    out_dir = Path(WORKSPACE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(chosen["filename"]).name
    shutil.copy2(src, dest)
    logger.info("Copied output %s -> %s", src, dest)
    return str(dest)


def run_inference_h3(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    import comfyui_bridge
    from workflow_converter import convert_visual_to_api
    from workflow_flatten import flatten_subgraphs

    params = dict(params or {})
    wf_name = pick_workflow_file(params)
    path = Path(WORKFLOW_DIR) / wf_name
    if not path.is_file():
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path, encoding="utf-8") as f:
        visual = json.load(f)

    media_items = download_media_items(params, comfyui_bridge)
    prefer = None

    if wf_name == FAST_WORKFLOW or wf_name.endswith("fastMinimaxH3_textToVideo.json"):
        from h3_fast_patch import SAVE_VIDEO_NODE_ID, apply_fast_job_params

        media_files: Dict[str, str] = {}
        for item in media_items:
            kind = item.get("kind")
            fname = item.get("file")
            if not fname:
                continue
            # download_media_items prefixes first_frame / last_frame / init
            if str(fname).startswith("h3_first"):
                media_files["first_frame"] = fname
            elif str(fname).startswith("h3_last"):
                media_files["last_frame"] = fname
            elif str(fname).startswith("h3_init") and kind == "picture":
                media_files.setdefault("first_frame", fname)
        apply_fast_job_params(
            visual,
            prompt or params.get("prompt") or "",
            params,
            media_files,
        )
        prefer = SAVE_VIDEO_NODE_ID
    else:
        from h3_patch import apply_job_params, apply_toggles, pick_output_node_id

        apply_toggles(visual, params)
        visual = flatten_subgraphs(visual)
        apply_job_params(
            visual,
            prompt or params.get("prompt") or "",
            params,
            media_items,
            apply_group_toggles=False,
        )
        prefer = pick_output_node_id(params)

    api_wf = convert_visual_to_api(visual)

    logger.info(
        "[%s] Queueing MiniMax H3 workflow %s (%d nodes)",
        job_id,
        wf_name,
        len(api_wf),
    )
    timeout = int(params.get("comfy_timeout", getattr(comfyui_bridge, "COMFYUI_PROMPT_TIMEOUT", 3600)))
    _pid, output_files = comfyui_bridge.run_prompt(api_wf, timeout=timeout)
    return _pick_output(comfyui_bridge, output_files, prefer)
