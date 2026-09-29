#!/usr/bin/env python3
"""Gate worker readiness on ComfyUI /object_info covering the Seed Hunter graph."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("validate_workflow")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKFLOW_DIR = Path(os.environ.get("WORKFLOW_DIR", str(PROJECT_ROOT / "workflow")))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")

SKIP_TYPES = {
    "Note",
    "MarkdownNote",
    "Reroute",
    "SetNode",
    "GetNode",
}

# Fast H3 injects these at runtime (not in the saved T2V JSON).
EXTRA_REQUIRED_TYPES = {
    "MMH3UltimateUpscale",
    "MMH3LatentUpscaleWithModelParams",
}


def _iter_workflow_files(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def collect_types_from_visual(data: dict) -> Set[str]:
    from workflow_flatten import collect_class_types, flatten_subgraphs

    types = collect_class_types(data)
    subgraph_ids = {
        sg.get("id") for sg in (data.get("definitions") or {}).get("subgraphs") or [] if sg.get("id")
    }
    types -= {t for t in types if t in subgraph_ids}
    try:
        flat = flatten_subgraphs(data)
        from workflow_converter import convert_visual_to_api

        api = convert_visual_to_api(flat)
        types |= {n.get("class_type") for n in api.values() if n.get("class_type")}
    except Exception:
        logger.exception("Flatten/convert during validation failed; using visual types only")
    return {t for t in types if t and t not in SKIP_TYPES}


def fetch_object_info(url: str) -> Optional[Dict]:
    try:
        import requests
    except ImportError:
        logger.error("Install requests")
        return None
    try:
        r = requests.get(f"{url}/object_info", timeout=120)
        if r.status_code == 200:
            return r.json()
        logger.error("object_info HTTP %s: %s", r.status_code, r.text[:500])
        return None
    except Exception as e:
        logger.error("object_info failed: %s", e)
        return None


def validate(
    workflow_dir: Optional[Path] = None,
    comfy_url: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    wd = Path(workflow_dir or WORKFLOW_DIR)
    url = (comfy_url or COMFYUI_URL).rstrip("/")
    files = _iter_workflow_files(wd)
    used = [p.name for p in files]
    needed: Set[str] = set(EXTRA_REQUIRED_TYPES)
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "nodes" in data:
            needed |= collect_types_from_visual(data)
        elif isinstance(data, dict):
            needed |= {
                data[k].get("class_type")
                for k in data
                if isinstance(data.get(k), dict) and data[k].get("class_type")
            }
    info = fetch_object_info(url)
    if info is None:
        return sorted(needed), used
    missing = sorted(t for t in needed if t not in info and t not in SKIP_TYPES)
    return missing, used


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    missing, used = validate()
    print("workflows:", used)
    if missing:
        print("MISSING", len(missing))
        for t in missing:
            print(" ", t)
        sys.exit(1)
    print("ok")
