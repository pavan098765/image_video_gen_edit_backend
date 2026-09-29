#!/usr/bin/env python3
import os
import subprocess
import sys
import traceback

from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
QPW_ROOT = os.path.dirname(PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

_MC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if _MC not in sys.path:
    sys.path.insert(0, _MC)
from omni_prep_bake import run_omni_full_bake  # noqa: E402

MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32")
MODEL_REPO = os.getenv(
    "HF_REPO_WAN_T2V_BAKED",
    "rockapaper/wan22-14b-t2v-lightning-sdnq-fp8-svd-r32",
)


def split_large_safetensors(base_dir: str, max_size: int = 700 * 1024 * 1024):
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                if os.path.getsize(full_path) <= max_size:
                    continue
                subprocess.run(
                    ["split", "-b", "700M", full_path, f"{full_path}.part-"],
                    check=True,
                )
                os.remove(full_path)
                print(f"✅ Split: {full_path}")


def main():
    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
    try:
        if not os.listdir(MODEL_LOCAL_DIR):
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=MODEL_LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        print("✅ WAN T2V baked model ready")
    except Exception as e:
        print(f"❌ WAN T2V download failed: {e}")
        traceback.print_exc()

    run_omni_full_bake(PROJECT_ROOT)

    split_large_safetensors(MODELS_DIR)
    subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
        cwd=SCRIPT_DIR,
        check=False,
    )


if __name__ == "__main__":
    main()

