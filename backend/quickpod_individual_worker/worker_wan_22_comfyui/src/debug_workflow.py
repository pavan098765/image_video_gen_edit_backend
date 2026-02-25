#!/usr/bin/env python3
"""
debug_workflow.py
=================

Debug script to test the converted workflow against ComfyUI's /prompt endpoint.

Run while ComfyUI is running (i.e. test_local.sh is active in another terminal):
    cd /work_space/worker_wan_22_comfyui
    COMFYUI_DIR=./comfyui WORKFLOW_DIR=./workflow PYTHONPATH=src python3 src/debug_workflow.py

This will:
1. Convert the visual workflow to API format
2. Print node summary
3. POST it to ComfyUI and show the FULL error response
"""

import json
import os
import sys
import requests

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflow"))
WORKFLOW_JSON = os.path.join(WORKFLOW_DIR, "wan_22_i2v_svi.json")


def main():
    print("=" * 60)
    print("Debug: Workflow Conversion & Validation")
    print("=" * 60)

    # Step 1: Convert workflow
    print(f"\n--- Step 1: Convert workflow ---")
    print(f"  Source: {WORKFLOW_JSON}")

    from workflow_converter import load_and_convert
    api_workflow = load_and_convert(WORKFLOW_JSON)
    print(f"  Converted: {len(api_workflow)} nodes")

    # Step 2: Node summary
    print(f"\n--- Step 2: Node types ---")
    type_counts = {}
    for nid, node in api_workflow.items():
        ct = node.get("class_type", "UNKNOWN")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct}: {count}")

    # Step 3: Save API workflow for inspection
    api_out = os.path.join(WORKFLOW_DIR, "wan_22_i2v_svi_api_debug.json")
    with open(api_out, "w") as f:
        json.dump(api_workflow, f, indent=2)
    print(f"\n  API workflow saved to: {api_out}")

    # Step 4: Check ComfyUI is running
    print(f"\n--- Step 3: Check ComfyUI ---")
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        print(f"  ComfyUI responding: {resp.status_code}")
    except Exception as e:
        print(f"  ERROR: ComfyUI not responding at {COMFYUI_URL}: {e}")
        print(f"  Make sure test_local.sh is running in another terminal")
        sys.exit(1)

    # Step 5: Get available node types from ComfyUI
    print(f"\n--- Step 4: Check node types ---")
    try:
        resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
        if resp.status_code == 200:
            available_nodes = set(resp.json().keys())
            print(f"  ComfyUI has {len(available_nodes)} node types available")

            # Check which node types in our workflow are missing
            missing = []
            for nid, node in api_workflow.items():
                ct = node.get("class_type", "")
                if ct and ct not in available_nodes:
                    missing.append((nid, ct))

            if missing:
                print(f"\n  *** MISSING NODE TYPES ({len(missing)}): ***")
                seen = set()
                for nid, ct in missing:
                    if ct not in seen:
                        print(f"    - {ct} (e.g. node {nid})")
                        seen.add(ct)
                print(f"\n  These nodes are not installed in ComfyUI!")
                print(f"  This is likely the cause of the 400 error.")
            else:
                print(f"  All node types in workflow are available ✓")
        else:
            print(f"  Could not fetch object_info: {resp.status_code}")
    except Exception as e:
        print(f"  Error checking node types: {e}")

    # Step 6: Validate prompt via ComfyUI
    print(f"\n--- Step 5: Validate workflow via /prompt ---")
    payload = {
        "prompt": api_workflow,
        "client_id": "debug-test",
    }

    try:
        resp = requests.post(
            f"{COMFYUI_URL}/prompt",
            json=payload,
            timeout=30,
        )
        print(f"  HTTP Status: {resp.status_code}")

        try:
            data = resp.json()
            print(f"  Response keys: {list(data.keys())}")

            if resp.status_code == 200:
                print(f"  ✓ Workflow accepted! prompt_id={data.get('prompt_id')}")
                # Cancel it - we don't want it to actually run
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    try:
                        requests.post(f"{COMFYUI_URL}/queue", json={"delete": [prompt_id]}, timeout=5)
                        print(f"  (Cancelled prompt to avoid running)")
                    except Exception:
                        print(f"  (Warning: could not cancel, it may run)")
            else:
                print(f"\n  *** FULL ERROR RESPONSE: ***")
                print(json.dumps(data, indent=2)[:5000])

                if "node_errors" in data:
                    print(f"\n  --- Node Errors ---")
                    for node_id, errors in data["node_errors"].items():
                        node_type = api_workflow.get(node_id, {}).get("class_type", "?")
                        print(f"  Node {node_id} ({node_type}):")
                        if isinstance(errors, dict):
                            for k, v in errors.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"    {errors}")

                if "error" in data:
                    print(f"\n  --- Error ---")
                    print(f"  {json.dumps(data['error'], indent=2)}")

        except Exception:
            print(f"  Raw response: {resp.text[:3000]}")

    except Exception as e:
        print(f"  ERROR posting prompt: {e}")

    # Step 7: Print a few key nodes for inspection
    print(f"\n--- Step 6: Key node details ---")
    key_nodes = {
        "58": "LoadImage",
        "68": "ImageResizeKJv2",
        "16": "WanVideoTextEncode (seg1)",
        "27": "WanVideoSampler (seg1)",
        "63": "WanVideoImageToVideoEncode (seg1)",
        "721": "VHS_VideoCombine (final)",
    }
    for nid, label in key_nodes.items():
        if nid in api_workflow:
            node = api_workflow[nid]
            print(f"\n  [{nid}] {label} ({node.get('class_type')}):")
            inputs = node.get("inputs", {})
            for k, v in inputs.items():
                if isinstance(v, str) and len(v) > 80:
                    v = v[:80] + "..."
                print(f"    {k}: {v}")
        else:
            print(f"\n  [{nid}] {label}: NOT FOUND in workflow!")

    print(f"\n{'=' * 60}")
    print("Debug complete.")


if __name__ == "__main__":
    main()
