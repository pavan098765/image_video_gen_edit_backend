#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # go one level up from src/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

HUNYAN15_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"

HUNYAN15_DIR = os.path.join(MODELS_DIR, "hunyan15")

os.makedirs(HUNYAN15_DIR, exist_ok=True)

# ------------------------
# 1. Download Hunyan15 model
# ------------------------
try:
    print(f"⬇️ Downloading {HUNYAN15_REPO} into {HUNYAN15_DIR}")
    snapshot_download(
        HUNYAN15_REPO,
        local_dir=HUNYAN15_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Hunyan15 repo ready at {HUNYAN15_DIR}")
except Exception as e:
    print(f"❌ Failed to download {HUNYAN15_REPO}: {e}")
    traceback.print_exc()


# ------------------------
# 2. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=500 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "500M", full_path, f"{full_path}.part-"],
                        check=True,
                    )
                    os.remove(full_path)

                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")


split_large_safetensors(MODELS_DIR)

# ------------------------
# 3. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", "generate_dockerfile.py"], check=True)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")
