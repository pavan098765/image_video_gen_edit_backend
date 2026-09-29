#!/usr/bin/env python3
import os
import subprocess
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

QPW_ROOT = os.path.dirname(PROJECT_ROOT)
_MC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if _MC not in sys.path:
    sys.path.insert(0, _MC)
from omni_prep_bake import run_omni_full_bake  # noqa: E402


def split_large_safetensors(base_dir: str, max_size: int = 850 * 1024 * 1024):
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                if os.path.getsize(full_path) <= max_size:
                    continue
                subprocess.run(
                    ["split", "-b", "850M", full_path, f"{full_path}.part-"],
                    check=True,
                )
                os.remove(full_path)
                print(f"✅ Split: {full_path}")


def main():
    try:
        run_omni_full_bake(PROJECT_ROOT, include_qwen_stack=True)
    except Exception as e:
        print(f"❌ omni full bake failed: {e}")
        traceback.print_exc()

    split_large_safetensors(MODELS_DIR)
    subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
        cwd=SCRIPT_DIR,
        check=False,
    )


if __name__ == "__main__":
    main()
