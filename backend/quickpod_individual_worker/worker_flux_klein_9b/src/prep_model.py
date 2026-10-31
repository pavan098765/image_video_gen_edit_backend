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
MODEL_REPO = "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "flux2_klein_9b_sdnq_4bit_svd_r32")
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
os.makedirs(REALESRGAN_DIR, exist_ok=True)

# ------------------------
# 1. Download Flux2Klein SDNQ model
# ------------------------
if not os.path.exists(MODEL_LOCAL_DIR) or not os.listdir(MODEL_LOCAL_DIR):
    try:
        print(f"⬇️ Downloading {MODEL_REPO} to {MODEL_LOCAL_DIR}...")
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.environ.get("HF_TOKEN")
        )
        print(f"✓ Flux2Klein SDNQ model downloaded successfully to {MODEL_LOCAL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        traceback.print_exc()
else:
    print(f"✓ Model already exists at {MODEL_LOCAL_DIR}")

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
            if fn.endswith(".safetensors") and ".part-" not in fn:
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
    subprocess.run(["python3", "generate_dockerfile.py"], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

# ------------------------
# Done
# ------------------------
print("✅ Flux2Klein SDNQ prep finished.")
print(f"Model at: {MODEL_LOCAL_DIR}")
print("Set env vars:")
print(f"  MODELS_DIR = {MODELS_DIR}")
