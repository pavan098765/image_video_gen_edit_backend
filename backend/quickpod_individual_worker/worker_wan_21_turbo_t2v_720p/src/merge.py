#!/usr/bin/env python3
"""
Merge .part-* files into single files under MODELS_DIR.
Handles both .safetensors.part-* and .pth.part-* for TurboDiffusion workers.
"""
import os
import glob
import sys
import subprocess

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def merge_parts(base_path: str, delete_parts: bool = True) -> bool:
    parts = sorted(glob.glob(base_path + ".part-*"))
    if not parts:
        return False
    output_path = base_path
    expected_size = sum(os.path.getsize(p) for p in parts)
    expected_size_mb = expected_size / (1024 * 1024)
    if os.path.exists(output_path):
        actual_size = os.path.getsize(output_path)
        if abs(actual_size - expected_size) <= expected_size * 0.01:
            print(f"✓ Already merged: {output_path}")
            return True
        try:
            os.remove(output_path)
        except Exception as e:
            print(f"❌ Failed to remove: {e}")
            sys.exit(1)
    print(f"🔍 Merging {len(parts)} parts -> {os.path.basename(output_path)}")
    try:
        with open(output_path, "wb") as outfile:
            subprocess.run(["cat"] + parts, stdout=outfile, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"❌ Merge failed: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        sys.exit(1)
    if abs(os.path.getsize(output_path) - expected_size) > expected_size * 0.01:
        print(f"❌ Size mismatch")
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(1)
    if delete_parts:
        for p in parts:
            try:
                os.remove(p)
            except Exception:
                pass
    return True


def main():
    print(f"🔍 Scanning for .part-* in {MODELS_DIR}")
    merged_any = False
    merge_errors = []
    bases = set()
    for root, _, files in os.walk(MODELS_DIR):
        for fn in files:
            if ".part-" in fn:
                base = fn.split(".part-")[0]
                bases.add(os.path.join(root, base))
    for base in sorted(bases):
        try:
            if merge_parts(base, delete_parts=True):
                merged_any = True
        except SystemExit:
            merge_errors.append(base)
        except Exception as e:
            merge_errors.append(base)
            print(f"❌ {base}: {e}")
    if merge_errors:
        print(f"❌ Merge errors: {len(merge_errors)}")
        sys.exit(1)
    if not merged_any:
        print("ℹ️  No split files found.")
    else:
        print("✅ Merge completed.")


if __name__ == "__main__":
    main()
