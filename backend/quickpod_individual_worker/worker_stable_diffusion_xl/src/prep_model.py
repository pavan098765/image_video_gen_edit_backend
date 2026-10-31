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
# Config
# ------------------------
SDXL_REPO = "HyHorX/SDXL-Lighting-8-steps"
SDXL_DIR = os.path.join(MODELS_DIR, "sdxl_lighting_8_steps")

REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(SDXL_DIR, exist_ok=True)
os.makedirs(REALESRGAN_DIR, exist_ok=True)

# ------------------------
# 1. Download SDXL Lighting 8-steps model
# ------------------------
if not os.listdir(SDXL_DIR):
    try:
        print(f"⬇️ Downloading {SDXL_REPO} into {SDXL_DIR} ...")
        snapshot_download(
            SDXL_REPO,
            local_dir=SDXL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"✓ SDXL Lighting 8-steps ready at {SDXL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {SDXL_REPO}: {e}")
        traceback.print_exc()
        exit(1)
else:
    print(f"✓ SDXL Lighting 8-steps already exists at {SDXL_DIR}")

# ------------------------
# 2. Download Real-ESRGAN models (2x, 4x) from rockapaper/realesrgan
# ------------------------
REALESRGAN_HF_REPO = "rockapaper/realesrgan"
for filename in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
    model_file = os.path.join(REALESRGAN_DIR, filename)
    if os.path.exists(model_file):
        print(f"✓ Real-ESRGAN {filename} already exists at {model_file}")
        continue
    try:
        print(f"⬇️ Downloading {filename} from {REALESRGAN_HF_REPO}...")
        hf_hub_download(
            repo_id=REALESRGAN_HF_REPO,
            filename=filename,
            local_dir=REALESRGAN_DIR,
            resume_download=True,
        )
        print(f"✓ {filename} downloaded to {REALESRGAN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        traceback.print_exc()

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

# ------------------------
# Done
# ------------------------
print("✅ SDXL Lighting 8-steps prep finished successfully.")
print(f"Model available at: {SDXL_DIR}")
print("Set environment variables:")
print(f"  SDXL_LOCAL_PATH={SDXL_DIR}")