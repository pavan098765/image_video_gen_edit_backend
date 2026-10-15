#!/usr/bin/env python3
"""Run the Civitai LTX-2.5 INT8 AllModes workflow and return one mp4 path."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from ltx_patch import MODE_FLF2V, MODE_I2V, SAVE_NODE_ID, apply_api_job_params, apply_job_params, resolve_mode

logger = logging.getLogger("ltx_inference")

WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEFAULT_WORKFLOW = os.environ.get(
    "LTX25_WORKFLOW_FILE",
    "LTX-2.5-INT8-AllModes-Stage2-Speed-Quality-LoRA-Civitai-v1.0_api.json",
)


def _ext_from_url(url: str, fallback: str = ".png") -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext and len(ext) <= 8:
        return ext
    return fallback


def _download_image(url: str, filename: str, comfyui_bridge: Any) -> str:
    logger.info("Downloading LTX guide %s -> %s", url, filename)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    comfyui_bridge.save_bytes_to_input(resp.content, filename)
    return filename


def _pick_output(comfyui_bridge: Any, output_files: List[dict]) -> str:
    if not output_files:
        raise RuntimeError("ComfyUI produced no output files")
    videos = [
        f
        for f in output_files
        if str(f.get("filename") or "").lower().endswith((".mp4", ".webm", ".mkv"))
        or "video" in str(f.get("format") or "").lower()
    ]
    candidates = videos or output_files
    chosen = None
    for item in candidates:
        if str(item.get("node_id")) == str(SAVE_NODE_ID):
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


def run_inference_ltx25(job_id: str, prompt: str, params: Dict[str, Any], model: str = "ltx25_int8") -> str:
    import comfyui_bridge
    from workflow_converter import convert_visual_to_api, is_api_prompt, prefer_api_filename
    from workflow_flatten import flatten_subgraphs

    params = dict(params or {})
    model = params.pop("_model_key", None) or model
    wf_name = prefer_api_filename(WORKFLOW_DIR, params.get("workflow") or DEFAULT_WORKFLOW)
    path = Path(WORKFLOW_DIR) / wf_name
    if not path.is_file():
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path, encoding="utf-8") as f:
        graph = json.load(f)

    mode = resolve_mode(model, params)
    first_name = None
    last_name = None
    first_url = (
        params.get("first_frame_url")
        or params.get("first_frame_image")
        or params.get("init_image")
        or params.get("image_url")
    )
    last_url = (
        params.get("last_frame_url")
        or params.get("last_frame_image")
        or params.get("last_image")
    )
    if mode in (MODE_I2V, MODE_FLF2V) and isinstance(first_url, str) and first_url.startswith("http"):
        first_name = _download_image(first_url, f"ltx25_first{_ext_from_url(first_url)}", comfyui_bridge)
    if mode == MODE_FLF2V and isinstance(last_url, str) and last_url.startswith("http"):
        last_name = _download_image(last_url, f"ltx25_last{_ext_from_url(last_url)}", comfyui_bridge)

    text = prompt or params.get("prompt") or ""
    if is_api_prompt(graph):
        apply_api_job_params(graph, text, params, model, first_name, last_name)
        api_wf = graph
    else:
        apply_job_params(graph, text, params, model, first_name, last_name)
        graph = flatten_subgraphs(graph)
        api_wf = convert_visual_to_api(graph)

    logger.info("[%s] Queueing LTX-2.5 INT8 workflow (%d nodes, mode=%s)", job_id, len(api_wf), mode)
    timeout = int(params.get("comfy_timeout", getattr(comfyui_bridge, "COMFYUI_PROMPT_TIMEOUT", 3600)))
    _pid, output_files = comfyui_bridge.run_prompt(api_wf, timeout=timeout)
    return _pick_output(comfyui_bridge, output_files)
