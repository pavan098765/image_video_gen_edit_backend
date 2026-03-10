#!/usr/bin/env python3
"""
validate_workflow_nodes.py
===========================

Uses ComfyUI's /object_info API to list ALL missing node types for the workflow
in one go (no trial-and-error). Run this with ComfyUI already running.

Usage:
  # With ComfyUI running (e.g. test_local.sh in another terminal):
  cd worker_wan_22_comfyui
  python src/validate_workflow_nodes.py

  # Custom URL / workflow path:
  COMFYUI_URL=http://host:8188 python src/validate_workflow_nodes.py
  python src/validate_workflow_nodes.py path/to/workflow.json

Exit: 0 if all node types are available; 1 if any are missing (and prints full list).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", str(PROJECT_ROOT / "workflow"))
DEFAULT_WORKFLOW = Path(WORKFLOW_DIR) / "WAN2.2 T2V-I2V-T2I-S2V K3NK v2.5.4 SVI.json"
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def get_workflow_class_types(workflow_path: Path) -> set[str]:
    """Load workflow (visual or API) and return set of class_type values."""
    with open(workflow_path, encoding="utf-8") as f:
        data = json.load(f)

    # Visual format: nodes[].type
    if "nodes" in data:
        sys.path.insert(0, str(SCRIPT_DIR))
        from workflow_converter import load_and_convert  # noqa: E402
        api = load_and_convert(workflow_path)
        return {api[nid].get("class_type") for nid in api if api[nid].get("class_type")}
    # API format: top-level keys are node ids, value has class_type
    if isinstance(data, dict):
        return {data[nid].get("class_type") for nid in data if isinstance(data.get(nid), dict) and data[nid].get("class_type")}
    return set()


def get_object_info(url: str) -> dict | None:
    """GET /object_info from ComfyUI. Returns dict of node_type -> info or None on error."""
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests", file=sys.stderr)
        return None
    try:
        r = __import__("requests").get(f"{url}/object_info", timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"object_info returned {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Failed to fetch object_info: {e}", file=sys.stderr)
        return None


def main() -> int:
    workflow_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKFLOW
    if not workflow_path.is_file():
        print(f"Workflow not found: {workflow_path}", file=sys.stderr)
        return 2

    print("Validate workflow nodes against ComfyUI /object_info")
    print(f"  Workflow: {workflow_path}")
    print(f"  ComfyUI:  {COMFYUI_URL}")

    workflow_types = get_workflow_class_types(workflow_path)
    print(f"  Workflow uses {len(workflow_types)} node types.")

    object_info = get_object_info(COMFYUI_URL)
    if object_info is None:
        print("ComfyUI not reachable or /object_info failed. Start ComfyUI first (e.g. test_local.sh).", file=sys.stderr)
        return 2

    available = set(object_info.keys())
    print(f"  ComfyUI has   {len(available)} node types registered.")

    missing = sorted(workflow_types - available)
    if not missing:
        print("  All workflow node types are available.")
        return 0

    print(f"\n*** MISSING NODE TYPES ({len(missing)}) ***")
    for t in missing:
        print(f"  - {t}")
    print("\nInstall the custom node extensions that provide these types (see prep_model.py CUSTOM_NODES),")
    print("then re-run prep and rebuild. Or run: python src/scan_workflow_nodes.py to refresh workflow_node_types.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
