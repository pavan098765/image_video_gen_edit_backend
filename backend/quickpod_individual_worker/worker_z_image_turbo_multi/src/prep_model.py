#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import traceback

from huggingface_hub import hf_hub_download, snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
QPW_ROOT = os.path.dirname(PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
Z_IMAGE_TURBO_DIR = os.path.join(MODELS_DIR, "z_image_turbo")
NUNCHAKU_DIR = os.path.join(MODELS_DIR, "nunchaku_z_image_turbo")

_MC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if _MC not in sys.path:
    sys.path.insert(0, _MC)
from omni_prep_bake import run_omni_full_bake  # noqa: E402

os.makedirs(Z_IMAGE_TURBO_DIR, exist_ok=True)
os.makedirs(NUNCHAKU_DIR, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def split_large_safetensors(base_dir: str, max_size: int = 500 * 1024 * 1024):
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                if os.path.getsize(full_path) <= max_size:
                    continue
                subprocess.run(
                    ["split", "-b", "500M", full_path, f"{full_path}.part-"],
                    check=True,
                )
                os.remove(full_path)
                print(f"✅ Split: {full_path}")


def main():
    try:
        if not os.listdir(Z_IMAGE_TURBO_DIR):
            snapshot_download(
                repo_id=os.getenv("HF_REPO_Z_IMAGE_TURBO_BAKED", "Tongyi-MAI/Z-Image-Turbo"),
                local_dir=Z_IMAGE_TURBO_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["transformer/*"],
            )
        print("✅ Z-Image Turbo base repo ready")
    except Exception as e:
        print(f"❌ Failed Z-Image Turbo download: {e}")
        traceback.print_exc()

    n_repo = os.getenv("HF_REPO_NUNCHAKU_Z", "nunchaku-tech/nunchaku-z-image-turbo")
    for filename in (
        "svdq-fp4_r128-z-image-turbo.safetensors",
        "svdq-int4_r128-z-image-turbo.safetensors",
    ):
        target_path = os.path.join(NUNCHAKU_DIR, filename)
        if os.path.exists(target_path):
            print(f"✓ Nunchaku file present: {filename}")
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=n_repo,
                filename=filename,
                resume_download=True,
            )
            shutil.copy2(downloaded, target_path)
            print(f"✅ Nunchaku transformer ready: {filename}")
        except Exception as e:
            print(f"❌ Failed Nunchaku download {filename}: {e}")
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

