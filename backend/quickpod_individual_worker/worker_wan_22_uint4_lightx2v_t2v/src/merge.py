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
    
    # Calculate expected total size from all parts
    expected_size = sum(os.path.getsize(p) for p in parts)
    expected_size_mb = expected_size / (1024 * 1024)
    
    # Check if output already exists and verify its size
    if os.path.exists(output_path):
        actual_size = os.path.getsize(output_path)
        actual_size_mb = actual_size / (1024 * 1024)
        
        # Allow 1% tolerance for file size differences
        size_diff = abs(actual_size - expected_size)
        size_tolerance = expected_size * 0.01
        
        if size_diff <= size_tolerance:
            print(f"✓ File already exists and size matches: {output_path} ({actual_size_mb:.2f} MB)")
            return True
        else:
            print(f"⚠️  Existing file size mismatch!")
            print(f"   Expected: {expected_size_mb:.2f} MB ({expected_size} bytes)")
            print(f"   Actual: {actual_size_mb:.2f} MB ({actual_size} bytes)")
            print(f"   Difference: {abs(actual_size - expected_size) / (1024 * 1024):.2f} MB")
            print(f"   Removing incomplete file and re-merging...")
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"❌ Failed to remove incomplete file: {e}")
                sys.exit(1)

    print(f"🔍 Found {len(parts)} part files for {output_path}")
    print(f"   Expected total size: {expected_size_mb:.2f} MB")
    
    # Verify all parts exist and log their sizes
    total_parts_size = 0
    for i, p in enumerate(parts, 1):
        if not os.path.exists(p):
            print(f"❌ Missing part file: {p}")
            sys.exit(1)
        part_size = os.path.getsize(p)
        total_parts_size += part_size
        print(f"   Part {i}/{len(parts)}: {os.path.basename(p)} ({part_size / (1024 * 1024):.2f} MB)")
    
    if total_parts_size != expected_size:
        print(f"⚠️  Warning: Sum of part sizes ({total_parts_size / (1024 * 1024):.2f} MB) != expected ({expected_size_mb:.2f} MB)")

    try:
        # Run `cat part-* > output`
        print(f"🔄 Merging {len(parts)} parts into {os.path.basename(output_path)}...")
        cmd = ["cat"] + parts
        with open(output_path, "wb") as outfile:
            result = subprocess.run(cmd, stdout=outfile, check=True, stderr=subprocess.PIPE)
        
        # Verify merged file size
        merged_size = os.path.getsize(output_path)
        merged_size_mb = merged_size / (1024 * 1024)
        
        if abs(merged_size - expected_size) <= expected_size * 0.01:  # 1% tolerance
            print(f"✅ Merged successfully: {output_path}")
            print(f"   Size: {merged_size_mb:.2f} MB ({merged_size} bytes)")
        else:
            print(f"❌ Merge verification failed!")
            print(f"   Expected: {expected_size_mb:.2f} MB ({expected_size} bytes)")
            print(f"   Actual: {merged_size_mb:.2f} MB ({merged_size} bytes)")
            print(f"   Difference: {abs(merged_size - expected_size) / (1024 * 1024):.2f} MB")
            if os.path.exists(output_path):
                os.remove(output_path)
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to merge {output_path}: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.decode()}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during merge: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        sys.exit(1)

    # Delete parts if requested
    if delete_parts:
        for p in parts:
            try:
                os.remove(p)
                print(f"🗑️  Deleted {os.path.basename(p)}")
            except Exception as e:
                print(f"⚠️  Failed to delete {p}: {e}")

    return True


def main():
    print(f"🔍 Scanning for .safetensors.part-* in {MODELS_DIR}")
    merged_any = False
    merge_errors = []
    
    for root, _, files in os.walk(MODELS_DIR):
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
                # merge_parts calls sys.exit(1) on error
                merge_errors.append(base)
                print(f"❌ Failed to merge: {base}")
            except Exception as e:
                merge_errors.append(base)
                print(f"❌ Error merging {base}: {e}")

    if merge_errors:
        print(f"\n❌ Merge completed with {len(merge_errors)} error(s)")
        for error_base in merge_errors:
            print(f"   - {error_base}")
        sys.exit(1)
    
    if not merged_any:
        print("ℹ️  No split safetensors found. Nothing to merge.")
    else:
        print(f"✅ Merge process completed successfully")


if __name__ == "__main__":
    main()
