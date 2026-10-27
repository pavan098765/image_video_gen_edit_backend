#!/usr/bin/env python3
"""Run the Qwen Image 2.1 Turbo GGUF workflow and return one image path."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from qwen_patch import SAVE_NODE_ID, apply_api_job_params, apply_job_params

logger = logging.getLogger("qwen_inference")

WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/app/worker/workflow")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEFAULT_WORKFLOW = os.environ.get(
    "QWEN21_WORKFLOW_FILE",
    "qwen_image_2_1_turbo_multiref_api.json",
)
MULTIREF_API_NAME = "qwen_image_2_1_turbo_multiref_api.json"
MAX_REFERENCE_IMAGES = 10


def _ext_from_url(url: str, fallback: str = ".png") -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext and len(ext) <= 8:
        return ext
    return fallback


def _image_size(path: str) -> Optional[tuple]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        logger.warning("Could not read image size from %s", path, exc_info=True)
        return None


def _source_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def collect_reference_sources(params: Dict[str, Any]) -> List[str]:
    """Ordered reference images. Numbered slots win over a single list."""
    numbered: List[str] = []
    seen: List[int] = []
    for index in range(1, MAX_REFERENCE_IMAGES + 1):
        found = ""
        for key in (f"image_{index}", f"reference_{index}", f"ref_{index}"):
            items = _source_list(params.get(key))
            if items:
                found = items[0]
                break
        if found:
            numbered.append(found)
            seen.append(index)
    if numbered:
        if seen != list(range(1, seen[-1] + 1)):
            logger.warning(
                "Reference slots %s skip a number. Later prompt tags move forward to fill the gap.",
                seen,
            )
        return numbered[:MAX_REFERENCE_IMAGES]
    for key in ("init_image", "images", "reference_images", "image_urls", "init_images"):
        items = _source_list(params.get(key))
        if items:
            return items[:MAX_REFERENCE_IMAGES]
    for key in ("first_frame_url", "image_url", "image"):
        items = _source_list(params.get(key))
        if items:
            return items[:MAX_REFERENCE_IMAGES]
    return []


def _custom_canvas(params: Dict[str, Any], has_references: bool) -> bool:
    flag = params.get("custom_size")
    if flag is not None and str(flag).strip() != "":
        return str(flag).strip().lower() in {"1", "true", "yes", "on"}
    if not has_references:
        return True
    if params.get("aspect_ratio") or params.get("wh_ratio"):
        return True
    return params.get("width") is not None and params.get("height") is not None


def _safe_token(job_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]", "", job_id or "")[:40]
    return token or "job"


def _looks_like_base64(value: str) -> bool:
    if value.startswith("data:"):
        return True
    if value.startswith(("http://", "https://", "file:")):
        return False
    if len(value) < 64 or any(ch in value for ch in " /\\"):
        return False
    return True


def _decode_image_bytes(source: str) -> bytes:
    payload = source.split(",", 1)[1] if source.startswith("data:") else source
    return base64.b64decode(payload)


def _materialize_references(job_id: str, sources: List[str], comfyui_bridge: Any) -> List[str]:
    names: List[str] = []
    token = _safe_token(job_id)
    input_dir = os.path.join(os.environ.get("COMFYUI_DIR", "/app/comfyui"), "input")
    for index, source in enumerate(sources, start=1):
        ext = _ext_from_url(source) if source.startswith(("http://", "https://")) else ".png"
        filename = f"qwen21_{token}_ref_{index}{ext}"
        if source.startswith(("http://", "https://")):
            names.append(_download_image(source, filename, comfyui_bridge))
            continue
        if _looks_like_base64(source):
            comfyui_bridge.save_bytes_to_input(_decode_image_bytes(source), filename)
            names.append(filename)
            continue
        local = source if os.path.isabs(source) else os.path.join(input_dir, Path(source).name)
        if os.path.isfile(local):
            names.append(Path(local).name if local.startswith(input_dir) else Path(source).name)
            if not local.startswith(input_dir):
                comfyui_bridge.save_bytes_to_input(Path(local).read_bytes(), filename)
                names[-1] = filename
            continue
        raise FileNotFoundError(f"Reference image {index} is not a URL, file, or base64 payload")
    return names


def _has_muted_loaders(graph: dict) -> bool:
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and node.get("type") == "LoadImage" and node.get("mode") == 2:
            return True
    return False


def _download_image(url: str, filename: str, comfyui_bridge: Any) -> str:
    logger.info("Downloading Qwen guide %s -> %s", url, filename)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    comfyui_bridge.save_bytes_to_input(resp.content, filename)
    return filename


def _pick_output(comfyui_bridge: Any, output_files: List[dict]) -> str:
    if not output_files:
        raise RuntimeError("ComfyUI produced no output files")
    images = [
        f
        for f in output_files
        if str(f.get("filename") or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        or "image" in str(f.get("format") or "").lower()
    ]
    candidates = images or output_files
    chosen = None
    preferred = {"12", "461", str(SAVE_NODE_ID)}
    for item in candidates:
        if str(item.get("node_id")) in preferred:
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


def run_inference_qwen21(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    model: str = "qwen_image_21_turbo",
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

    if not is_api_prompt(graph) and _has_muted_loaders(graph):
        api_path = Path(WORKFLOW_DIR) / MULTIREF_API_NAME
        if not api_path.is_file():
            raise FileNotFoundError(f"API workflow not found: {api_path}")
        logger.info("Using %s for execution; muted UI loaders are not queued", MULTIREF_API_NAME)
        with open(api_path, encoding="utf-8") as f:
            graph = json.load(f)

    sources = collect_reference_sources(params)
    filenames = _materialize_references(job_id, sources, comfyui_bridge)
    custom_canvas = _custom_canvas(params, bool(filenames))
    if filenames and not custom_canvas:
        input_dir = os.environ.get("COMFYUI_DIR", "/app/comfyui")
        guide_path = os.path.join(input_dir, "input", filenames[0])
        guide_wh = _image_size(guide_path)
        if guide_wh:
            params["_guide_wh"] = guide_wh
            logger.info("[%s] First reference %sx%s", job_id, guide_wh[0], guide_wh[1])
    logger.info(
        "[%s] Qwen %s with %d reference image(s), custom_canvas=%s",
        job_id,
        "reference edit" if filenames else "text to image",
        len(filenames),
        custom_canvas,
    )

    text = prompt or params.get("prompt") or ""
    if is_api_prompt(graph):
        apply_api_job_params(
            graph,
            text,
            params,
            reference_images=filenames,
            custom_canvas=custom_canvas,
        )
        api_wf = graph
    else:
        apply_job_params(graph, text, params, filenames[0] if filenames else None)
        graph = flatten_subgraphs(graph)
        apply_job_params(graph, text, params, filenames[0] if filenames else None)
        api_wf = convert_visual_to_api(graph)

    logger.info("[%s] Queueing Qwen Image 2.1 Turbo workflow (%d nodes)", job_id, len(api_wf))
    timeout = int(params.get("comfy_timeout", getattr(comfyui_bridge, "COMFYUI_PROMPT_TIMEOUT", 1800)))
    _pid, output_files = comfyui_bridge.run_prompt(api_wf, timeout=timeout)
    return _pick_output(comfyui_bridge, output_files)
