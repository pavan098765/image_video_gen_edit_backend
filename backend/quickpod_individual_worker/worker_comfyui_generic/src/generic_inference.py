#!/usr/bin/env python3
"""
generic_inference.py
====================

Load a workflow from WORKFLOW_DIR, apply an API-shaped inputs_patch, run ComfyUI,
and return the first primary output file path (image/video).

Params (job ``params`` dict):
  workflow (str): Filename under WORKFLOW_DIR (required unless workflow_api inline).
  inputs_patch (dict): Nested ``{ "node_id_str": { "inputs": { ... } } }`` merged
    into the API workflow. Values override; nested dicts are shallow-merged for inputs.
  outputs_preference (list, optional): Preferred output keys to hunt in history,
    default [\"gifs\", \"images\", \"videos\"].
  prompt (str): Accepted for API compatibility; if inputs_patch omits all text
    widgets, you must set patches explicitly (no magic node id).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("generic_inference")

WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")


def _deep_merge_api_inputs(
    base: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge patch into base (API prompt dict). Shallow merge per node; inputs merged shallowly."""
    out = copy.deepcopy(base)
    for nid, node_patch in patch.items():
        if not isinstance(node_patch, dict):
            continue
        nid = str(nid)
        if nid not in out:
            out[nid] = {}
        if "class_type" in node_patch and node_patch["class_type"]:
            out[nid]["class_type"] = node_patch["class_type"]
        if "inputs" in node_patch and isinstance(node_patch["inputs"], dict):
            if "inputs" not in out[nid]:
                out[nid]["inputs"] = {}
            merged_inputs = copy.deepcopy(out[nid]["inputs"])
            for k, v in node_patch["inputs"].items():
                merged_inputs[k] = v
            out[nid]["inputs"] = merged_inputs
    return out


def load_workflow_api(workflow_filename: str) -> dict:
    path = Path(WORKFLOW_DIR) / workflow_filename
    if not path.is_file():
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unrecognized workflow JSON: {path}")

    if "nodes" in data:
        from workflow_converter import convert_visual_to_api

        return convert_visual_to_api(data)

    if not data:
        raise ValueError(f"Empty workflow JSON: {path}")

    first = next(iter(data.values()), None)
    is_api_shape = isinstance(first, dict) and "class_type" in first
    all_digit_keys = all(isinstance(k, str) and k.isdigit() for k in data.keys())
    if is_api_shape or all_digit_keys:
        return data

    from workflow_converter import convert_visual_to_api

    return convert_visual_to_api(data)


def _pick_output_path(
    comfyui_bridge: Any,
    output_files: List[dict],
    preference: Optional[List[str]] = None,
) -> str:
    if not output_files:
        raise RuntimeError("ComfyUI produced no output files in history")

    chosen = output_files[-1]
    src = comfyui_bridge.get_output_file_path(chosen)

    out_dir = Path(WORKSPACE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / chosen["filename"]
    shutil.copy2(src, dest)
    logger.info("Copied output %s -> %s", src, dest)
    return str(dest)


def run_inference_generic(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """Run a ComfyUI workflow; return path to result file under WORKSPACE_DIR."""
    import comfyui_bridge

    wf_name = params.get("workflow")
    if not wf_name:
        raise ValueError("params['workflow'] is required (filename under WORKFLOW_DIR)")

    inputs_patch = params.get("inputs_patch") or {}
    if not isinstance(inputs_patch, dict):
        raise ValueError("params['inputs_patch'] must be a dict")

    if prompt:
        logger.debug("[%s] prompt field set (use inputs_patch to drive workflow nodes)", job_id)

    api_wf = load_workflow_api(str(wf_name))
    api_wf = _deep_merge_api_inputs(api_wf, inputs_patch)

    logger.info("[%s] Queueing workflow %s (%d nodes)", job_id, wf_name, len(api_wf))

    timeout = int(params.get("comfy_timeout", getattr(comfyui_bridge, "COMFYUI_PROMPT_TIMEOUT", 3600)))
    _pid, output_files = comfyui_bridge.run_prompt(api_wf, timeout=timeout)

    return _pick_output_path(comfyui_bridge, output_files)
