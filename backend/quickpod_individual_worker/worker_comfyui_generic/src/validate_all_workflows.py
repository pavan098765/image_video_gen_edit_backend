#!/usr/bin/env python3
"""
validate_all_workflows.py
===========================

After ComfyUI is up, GET /object_info and verify every class_type used across
all workflows in WORKFLOW_DIR is registered.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKFLOW_DIR = Path(os.environ.get("WORKFLOW_DIR", str(PROJECT_ROOT / "workflow")))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def iter_workflow_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def extract_class_types(workflow_path: Path) -> Set[str]:
    with open(workflow_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "nodes" in data:
        sys.path.insert(0, str(SCRIPT_DIR))
        from workflow_converter import load_and_convert  # noqa: E402

        api = load_and_convert(workflow_path)
        return {api[nid].get("class_type") for nid in api if api[nid].get("class_type")}

    if isinstance(data, dict):
        return {
            data[nid].get("class_type")
            for nid in data
            if isinstance(data.get(nid), dict) and data[nid].get("class_type")
        }
    return set()


def collect_all_types(paths: list[Path]) -> Set[str]:
    u: Set[str] = set()
    for p in paths:
        u |= {t for t in extract_class_types(p) if t}
    return u


def fetch_object_info(url: str) -> Optional[Dict]:
    try:
        import requests
    except ImportError:
        print("Install requests.", file=sys.stderr)
        return None
    try:
        r = requests.get(f"{url}/object_info", timeout=60)
        if r.status_code == 200:
            return r.json()
        print(f"object_info HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"object_info failed: {e}", file=sys.stderr)
        return None


def validate(
    workflow_dir: Optional[Path] = None,
    comfy_url: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """
    Returns (missing_types, workflow_files_used).
    missing_types empty => success.
    """
    wd = workflow_dir or WORKFLOW_DIR
    url = (comfy_url or COMFYUI_URL).rstrip("/")
    paths = iter_workflow_files(wd)
    if not paths:
        return [], []

    types_needed = collect_all_types(paths)
    info = fetch_object_info(url)
    if info is None:
        raise RuntimeError("Could not fetch /object_info")

    available = set(info.keys())
    missing = sorted(t for t in types_needed if t and t not in available)
    return missing, [str(p.name) for p in paths]


def main() -> int:
    workflow_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKFLOW_DIR
    paths = iter_workflow_files(workflow_dir)
    if not paths:
        print("No workflow JSON files; nothing to validate.")
        return 0

    try:
        missing, used = validate(workflow_dir=workflow_dir)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"Workflows: {used}")
    print(f"Distinct node types (post-convert): {len(collect_all_types(paths))}")

    if missing:
        print(f"\n*** MISSING NODE TYPES ({len(missing)}) ***", file=sys.stderr)
        for t in missing:
            print(f"  - {t}", file=sys.stderr)
        return 1

    print("All workflow node types are available in ComfyUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
