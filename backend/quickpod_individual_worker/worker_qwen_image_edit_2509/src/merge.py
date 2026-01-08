#!/usr/bin/env python3
import os
import glob
import sys
import subprocess

MODELS_DIR = "/app/models"

def merge_parts(base_path: str, delete_parts: bool = True):
    """
    Merge .part-* files into a single safetensors file using `cat`.
    base_path = full path to the safetensor (without .part-* suffix)
    """
    parts = sorted(glob.glob(base_path + ".part-*"))
    if not parts:
        return False

    output_path = base_path
    if os.path.exists(output_path):
        print(f"⏩ Skipping merge, already exists: {output_path}")
        return True

    print(f"🔍 Found {len(parts)} part files for {output_path} (total ~{len(parts) * 500 / 1024:.1f} GB)")
    
    # Verify all parts exist before starting merge
    for p in parts:
        if not os.path.exists(p):
            print(f"❌ Missing part file: {p}")
            sys.exit(1)

    try:
        import time
        merge_start = time.time()
        # Run `cat part-* > output` - using cat is fast for large files
        cmd = ["cat"] + parts
        with open(output_path, "wb") as outfile:
            subprocess.run(cmd, stdout=outfile, check=True)
        merge_time = time.time() - merge_start
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ Merged into {output_path} ({file_size_mb:.1f} MB) in {merge_time:.1f}s ({file_size_mb/merge_time:.1f} MB/s)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to merge {output_path}: {e}")
        sys.exit(1)

    # Delete parts if requested (batch delete - only log summary to reduce noise)
    if delete_parts:
        deleted_count = 0
        for p in parts:
            try:
                os.remove(p)
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Failed to delete {p}: {e}")
        if deleted_count > 0:
            print(f"🗑️ Deleted {deleted_count} part files")

    return True


def main():
    print(f"🔍 Scanning for .safetensors.part-* in {MODELS_DIR}")
    merged_any = False
    for root, _, files in os.walk(MODELS_DIR):
        safetensor_bases = set()
        for fn in files:
            if ".safetensors.part-" in fn:
                base = fn.split(".part-")[0]
                safetensor_bases.add(os.path.join(root, base))

        for base in sorted(safetensor_bases):
            if merge_parts(base, delete_parts=True):
                merged_any = True

    if not merged_any:
        print("ℹ️ No split safetensors found. Nothing to merge.")


if __name__ == "__main__":
    main()
