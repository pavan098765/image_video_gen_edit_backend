#!/usr/bin/env python3
"""Run the official MiniMax Music 3 workflow and return one audio path."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from music_patch import SAVE_NODE_ID, apply_api_job_params, apply_job_params

logger = logging.getLogger("music_inference")

WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEFAULT_WORKFLOW = os.environ.get(
    "MUSIC3_WORKFLOW_FILE",
    "audio_minimax_music_3_api.json",
)


def _pick_output(comfyui_bridge: Any, output_files: List[dict]) -> str:
    if not output_files:
        raise RuntimeError("ComfyUI produced no output files")
    audio = [
        f
        for f in output_files
        if str(f.get("filename") or "").lower().endswith((".flac", ".mp3", ".wav", ".ogg", ".opus"))
        or "audio" in str(f.get("format") or "").lower()
    ]
    candidates = audio or output_files
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


def run_inference_music3(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    model: str = "minimax_music3",
) -> str:
    import comfyui_bridge
    from workflow_converter import convert_visual_to_api, is_api_prompt, prefer_api_filename
    from workflow_flatten import flatten_subgraphs

    params = dict(params or {})
    params.pop("_model_key", None)
    wf_name = prefer_api_filename(WORKFLOW_DIR, params.get("workflow") or DEFAULT_WORKFLOW)
    path = Path(WORKFLOW_DIR) / wf_name
    if not path.is_file():
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path, encoding="utf-8") as f:
        graph = json.load(f)

    text = prompt or params.get("caption") or params.get("prompt") or ""
    if is_api_prompt(graph):
        apply_api_job_params(graph, text, params)
        api_wf = graph
    else:
        apply_job_params(graph, text, params)
        graph = flatten_subgraphs(graph)
        apply_job_params(graph, text, params)
        api_wf = convert_visual_to_api(graph)

    logger.info("[%s] Queueing MiniMax Music 3 workflow (%d nodes)", job_id, len(api_wf))
    timeout = int(params.get("comfy_timeout", getattr(comfyui_bridge, "COMFYUI_PROMPT_TIMEOUT", 3600)))
    _pid, output_files = comfyui_bridge.run_prompt(api_wf, timeout=timeout)
    return _pick_output(comfyui_bridge, output_files)
