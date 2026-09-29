#!/usr/bin/env python3
import glob
import os
import subprocess
import sys

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")


def merge_parts(base_path: str, delete_parts: bool = True) -> bool:
    parts = sorted(glob.glob(base_path + ".part-*"))
    if not parts:
        return False
    if os.path.exists(base_path):
        return True
    with open(base_path, "wb") as outfile:
        subprocess.run(["cat", *parts], stdout=outfile, check=True)
    if delete_parts:
        for part in parts:
            try:
                os.remove(part)
            except OSError:
                pass
    return True


def main() -> int:
    merge_errors = []
    for root, _, files in os.walk(MODELS_DIR):
        bases = set()
        for fn in files:
            if ".safetensors.part-" in fn:
                bases.add(os.path.join(root, fn.split(".part-")[0]))
        for base in sorted(bases):
            try:
                merge_parts(base, delete_parts=True)
            except Exception:
                merge_errors.append(base)
    if merge_errors:
        print(f"❌ merge failures: {len(merge_errors)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
