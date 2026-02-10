#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model repository
GLM_OCR_REPO = "zai-org/GLM-OCR"
GLM_OCR_DIR = os.path.join(MODELS_DIR, "glm_ocr")

os.makedirs(GLM_OCR_DIR, exist_ok=True)

# ------------------------
# 1. Download GLM-OCR model
# ------------------------
# Check for required files to verify model is actually downloaded
REQUIRED_FILES = ["config.json"]  # At minimum, config.json should exist
model_exists = any(os.path.exists(os.path.join(GLM_OCR_DIR, f)) for f in REQUIRED_FILES) and len([f for f in os.listdir(GLM_OCR_DIR) if not f.startswith('.')]) > 0

if not model_exists:
    try:
        print(f"⬇️ Downloading {GLM_OCR_REPO} into {GLM_OCR_DIR}")
        snapshot_download(
            GLM_OCR_REPO,
            local_dir=GLM_OCR_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ GLM-OCR model ready at {GLM_OCR_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {GLM_OCR_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ GLM-OCR model already exists at {GLM_OCR_DIR}")

# ------------------------
# 2. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=600 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and not fn.startswith("."):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "600M", full_path, f"{full_path}.part-"],
                        check=True
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
    subprocess.run(["python3", "generate_dockerfile.py"], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print(f"✅ Prep finished. GLM-OCR model ready in {GLM_OCR_DIR}")
print(f"Set ENV GLM_OCR_LOCAL_PATH={GLM_OCR_DIR}")
