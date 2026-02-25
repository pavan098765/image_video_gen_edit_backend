#!/usr/bin/env python3
"""
merge.py
========

Reassemble split safetensors files at container startup.

Large model files (>700MB) are split during Docker build for optimal
layer caching.  This script merges them back before ComfyUI starts.

Works with files in both /app/models (ComfyUI model directories) and
/app/comfyui/models (ComfyUI's own models directory).
"""

import os
import glob
import sys
import subprocess

# Directories to scan for split safetensors
SCAN_DIRS = [
    os.environ.get("MODELS_DIR", "/app/models"),
    os.path.join(os.environ.get("COMFYUI_DIR", "/app/comfyui"), "models"),
]


def merge_parts(base_path: str, delete_parts: bool = True) -> bool:
    """
    Merge .part-* files into a single safetensors file using ``cat``.

    Parameters
    ----------
    base_path : str
        Full path to the target safetensors file (without .part-* suffix).
    delete_parts : bool
        Whether to delete part files after successful merge.

    Returns
    -------
    bool
        True if merge was performed or file already exists with correct size.
    """
    parts = sorted(glob.glob(base_path + ".part-*"))
    if not parts:
        return False

    output_path = base_path
    expected_size = sum(os.path.getsize(p) for p in parts)
    expected_size_mb = expected_size / (1024 * 1024)

    # Check if output already exists and verify its size
    if os.path.exists(output_path):
        actual_size = os.path.getsize(output_path)
        size_diff = abs(actual_size - expected_size)
        size_tolerance = expected_size * 0.01  # 1% tolerance

        if size_diff <= size_tolerance:
            print(f"  File exists and size matches: {output_path} ({actual_size / (1024**2):.2f} MB)")
            return True
        else:
            print(f"  Size mismatch for {output_path}, re-merging...")
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"  ERROR: Failed to remove incomplete file: {e}")
                sys.exit(1)

    print(f"  Found {len(parts)} parts for {os.path.basename(output_path)} ({expected_size_mb:.2f} MB total)")

    # Verify all parts exist
    for p in parts:
        if not os.path.exists(p):
            print(f"  ERROR: Missing part file: {p}")
            sys.exit(1)

    try:
        cmd = ["cat"] + parts
        with open(output_path, "wb") as outfile:
            subprocess.run(cmd, stdout=outfile, check=True, stderr=subprocess.PIPE)

        merged_size = os.path.getsize(output_path)
        if abs(merged_size - expected_size) <= expected_size * 0.01:
            print(f"  Merged successfully: {os.path.basename(output_path)} ({merged_size / (1024**2):.2f} MB)")
        else:
            print(f"  ERROR: Merge verification failed for {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to merge {output_path}: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        sys.exit(1)

    # Delete parts
    if delete_parts:
        for p in parts:
            try:
                os.remove(p)
            except Exception as e:
                print(f"  WARNING: Failed to delete {p}: {e}")

    return True


def main():
    print("Scanning for split safetensors to merge...")
    merged_any = False
    merge_errors = []

    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue

        print(f"\nScanning: {scan_dir}")

        for root, _, files in os.walk(scan_dir):
            safetensor_bases = set()
            for fn in files:
                if ".safetensors.part-" in fn:
                    base = fn.split(".part-")[0]
                    safetensor_bases.add(os.path.join(root, base))

            for base in sorted(safetensor_bases):
                try:
                    if merge_parts(base, delete_parts=True):
                        merged_any = True
                except SystemExit:
                    merge_errors.append(base)
                    print(f"  ERROR: Failed to merge: {base}")
                except Exception as e:
                    merge_errors.append(base)
                    print(f"  ERROR: Error merging {base}: {e}")

    if merge_errors:
        print(f"\nMerge completed with {len(merge_errors)} error(s):")
        for err in merge_errors:
            print(f"  - {err}")
        sys.exit(1)

    if not merged_any:
        print("No split safetensors found. Nothing to merge.")
    else:
        print("All splits merged successfully.")


if __name__ == "__main__":
    main()
