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

    print(f"🔍 Found {len(parts)} part files for {output_path}")
    for p in parts:
        if not os.path.exists(p):
            print(f"❌ Missing part file: {p}")
            sys.exit(1)

    try:
        # Run `cat part-* > output`
        cmd = ["cat"] + parts
        with open(output_path, "wb") as outfile:
            subprocess.run(cmd, stdout=outfile, check=True)
        print(f"✅ Merged into {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to merge {output_path}: {e}")
        sys.exit(1)

    # Delete parts if requested
    if delete_parts:
        for p in parts:
            try:
                os.remove(p)
                print(f"🗑️ Deleted {p}")
            except Exception as e:
                print(f"⚠️ Failed to delete {p}: {e}")

    return True


def main():
    print(f"🔍 Scanning for split files (.part-*) in {MODELS_DIR}")
    merged_any = False
    
    # Handle both .safetensors.part-* and .pth.part-*
    for root, _, files in os.walk(MODELS_DIR):
        split_bases = set()
        for fn in files:
            if ".part-" in fn:
                # Extract base filename (remove .part-* suffix)
                base = fn.split(".part-")[0]
                split_bases.add(os.path.join(root, base))

        for base in sorted(split_bases):
            if merge_parts(base, delete_parts=True):
                merged_any = True

    if not merged_any:
        print("ℹ️ No split files found. Nothing to merge.")


if __name__ == "__main__":
    main()
