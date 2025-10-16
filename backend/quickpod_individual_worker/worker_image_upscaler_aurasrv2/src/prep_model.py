#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

AURASR_REPO = "fal/AuraSR-v2"
AURASR_DIR = os.path.join(MODELS_DIR, "aurasr_v2")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(AURASR_DIR, exist_ok=True)

# ------------------------
# 1. Download AuraSR v2 model (skip model.ckpt)
# ------------------------
if not os.listdir(AURASR_DIR):
    try:
        print(f"⬇️ Downloading {AURASR_REPO} into {AURASR_DIR} (excluding model.ckpt)")
        snapshot_download(
            repo_id=AURASR_REPO,
            local_dir=AURASR_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["model.ckpt"],
        )
        print(f"✓ AuraSR v2 ready at {AURASR_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {AURASR_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ AuraSR v2 already exists at {AURASR_DIR}")

# ------------------------
# 2. Copy local Realesrgan models (optional)
# ------------------------
if os.path.exists(LOCAL_REALESRGAN_SRC):
    print(f"📦 Copying {LOCAL_REALESRGAN_SRC} -> {LOCAL_REALESRGAN_DST}")
    shutil.copytree(LOCAL_REALESRGAN_SRC, LOCAL_REALESRGAN_DST, dirs_exist_ok=True)
    print("✓ Realesrgan models copied.")
else:
    print("⚠️ No local realesrgan_models/ found, skipping.")

# ------------------------
# 3. Split large safetensors recursively
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
                        check=True
                    )
                    os.remove(full_path)
                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")

split_large_safetensors(MODELS_DIR)

# ------------------------
# 4. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", "generate_dockerfile.py"], check=True)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ AuraSR v2 prep finished successfully.")
print(f"AuraSR model at: {AURASR_DIR}")
print(f"Set ENV AURASR_LOCAL_PATH={AURASR_DIR}")
