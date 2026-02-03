#!/usr/bin/env python3
"""
merge.py - Merge split .safetensors.part-* files

Large model files are split for Docker layer caching (500MB chunks).
This script merges them back together at container startup.

Usage:
    python merge.py [--dir /path/to/models]
"""

import os
import sys
import glob
import time
import subprocess


# Default model directory (set by Dockerfile)
MODELS_DIR = os.environ.get("LTX2_VIDEO_LOCAL_PATH", "/app/models/ltx2_video")


def merge_parts(base_path: str, delete_parts: bool = True) -> bool:
    """
    Merge .part-* files into a single safetensors file using `cat`.
    
    Args:
        base_path: Full path to output file (without .part-* suffix)
        delete_parts: Whether to delete part files after merge
    
    Returns:
        bool: True if merge successful or file already exists
    """
    parts = sorted(glob.glob(base_path + ".part-*"))
    if not parts:
        return False

    output_path = base_path
    
    # Skip if already merged
    if os.path.exists(output_path):
        print(f"⏩ Skipping merge, already exists: {output_path}")
        return True

    total_size_mb = sum(os.path.getsize(p) for p in parts) / 1024 / 1024
    print(f"🔍 Found {len(parts)} part files for {os.path.basename(output_path)} (total ~{total_size_mb:.1f} MB)")
    
    # Verify all parts exist
    for p in parts:
        if not os.path.exists(p):
            print(f"❌ Missing part file: {p}")
            return False

    try:
        merge_start = time.time()
        
        # Use `cat` for fast merging on Linux
        # Fallback to Python for Windows
        try:
            cmd = ["cat"] + parts
            with open(output_path, "wb") as outfile:
                subprocess.run(cmd, stdout=outfile, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: pure Python merge
            print("   Using Python fallback for merge...")
            with open(output_path, "wb") as outfile:
                for part_path in parts:
                    with open(part_path, "rb") as infile:
                        while True:
                            chunk = infile.read(64 * 1024 * 1024)  # 64MB chunks
                            if not chunk:
                                break
                            outfile.write(chunk)
        
        merge_time = time.time() - merge_start
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        speed_mbps = file_size_mb / merge_time if merge_time > 0 else 0
        
        print(f"✅ Merged: {os.path.basename(output_path)} ({file_size_mb:.1f} MB) in {merge_time:.1f}s ({speed_mbps:.1f} MB/s)")
        
    except Exception as e:
        print(f"❌ Failed to merge {output_path}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    # Delete parts after successful merge
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


def main(directory: str = None):
    """
    Main entry point - find and merge all split safetensors files.
    
    Args:
        directory: Directory to search (defaults to common locations)
    """
    if directory is None:
        # Search in common model directories
        search_dirs = [
            MODELS_DIR,
            "/app/models",
            os.environ.get("LTX2_GEMMA_ROOT", "/app/models/gemma"),
        ]
        # Remove duplicates
        search_dirs = list(dict.fromkeys(search_dirs))
    else:
        search_dirs = [directory]
    
    print(f"🔍 Scanning for .safetensors.part-* files...")
    
    merged_any = False
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        for root, _, files in os.walk(search_dir):
            # Find all unique base names
            safetensor_bases = set()
            for fn in files:
                if ".safetensors.part-" in fn:
                    base = fn.split(".part-")[0]
                    safetensor_bases.add(os.path.join(root, base))
            
            # Merge each unique base
            for base in sorted(safetensor_bases):
                if merge_parts(base, delete_parts=True):
                    merged_any = True

    if not merged_any:
        print("ℹ️ No split safetensors found. Nothing to merge.")
    else:
        print("✅ All merges complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge split safetensors files")
    parser.add_argument("--dir", type=str, default=None, help="Directory to search")
    args = parser.parse_args()
    
    main(args.dir)
