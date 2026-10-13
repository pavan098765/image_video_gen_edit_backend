#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------------
# 1. Download RIFE model
# ------------------------
RIFE_MODEL_DIR = os.path.join(MODELS_DIR, "rife")
if not os.path.exists(RIFE_MODEL_DIR) or not os.listdir(RIFE_MODEL_DIR):
    try:
        print(f"⬇️ Downloading RIFE model weights to {RIFE_MODEL_DIR}...")
        os.makedirs(RIFE_MODEL_DIR, exist_ok=True)
        snapshot_download(
            "AlexWortega/RIFE",
            local_dir=RIFE_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ RIFE model weights downloaded successfully to {RIFE_MODEL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download RIFE model: {e}")
        traceback.print_exc()
else:
    print(f"✓ RIFE model already exists at {RIFE_MODEL_DIR}")

# ------------------------
# 2. Download Real-ESRGAN models (2x, 4x)
# ------------------------
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")
os.makedirs(REALESRGAN_DIR, exist_ok=True)

_repo = os.getenv("REALESRGAN_HF_REPO", "rockapaper/realesrgan")
for model_filename in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
    model_file = os.path.join(REALESRGAN_DIR, model_filename)
    if not os.path.exists(model_file):
        try:
            print(f"⬇️ Downloading {_repo} / {model_filename} → {REALESRGAN_DIR} …")
            hf_hub_download(
                repo_id=_repo,
                filename=model_filename,
                local_dir=REALESRGAN_DIR,
                resume_download=True,
            )
            print(f"✓ Real-ESRGAN {model_filename} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download Real-ESRGAN {model_filename}: {e}")
            traceback.print_exc()
    else:
        print(f"✓ Real-ESRGAN already exists at {model_file}")

# ------------------------
# 3. Download RIFE code (clone from GitHub)
# ------------------------
RIFE_CODE_DIR = os.path.join(SCRIPT_DIR, "rife")
if not os.path.exists(RIFE_CODE_DIR) or not os.listdir(RIFE_CODE_DIR):
    try:
        print(f"⬇️ Cloning RIFE code to {RIFE_CODE_DIR}...")
        os.makedirs(RIFE_CODE_DIR, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/hzwer/RIFE.git", RIFE_CODE_DIR],
            check=True,
            cwd=os.path.dirname(RIFE_CODE_DIR)
        )
        print(f"✓ RIFE code cloned successfully to {RIFE_CODE_DIR}")
    except Exception as e:
        print(f"⚠️  Failed to clone RIFE code: {e}")
        print(f"   RIFE code will need to be available at runtime for interpolation to work")
        traceback.print_exc()
else:
    print(f"✓ RIFE code already exists at {RIFE_CODE_DIR}")

print(f"✅ Prep finished. Models ready in {MODELS_DIR}")

# ------------------------
# 6. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()