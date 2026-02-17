#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # go one level up from src/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

HUNYAN15_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"

HUNYAN15_DIR = os.path.join(MODELS_DIR, "hunyan15_t2v")

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
def split_large_safetensors(base_dir, max_size=850 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "850M", full_path, f"{full_path}.part-"],
                        check=True,
                    )
                    os.remove(full_path)

                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")


split_large_safetensors(MODELS_DIR)

# ------------------------
# 3. Download RIFE model
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
# 4. Download Real-ESRGAN models (2x, 4x, 8x)
# ------------------------
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")
os.makedirs(REALESRGAN_DIR, exist_ok=True)

for upscale_factor in [2, 4, 8]:
    model_filename = f"RealESRGAN_x{upscale_factor}.pth"
    model_file = os.path.join(REALESRGAN_DIR, model_filename)
    
    if not os.path.exists(model_file):
        try:
            print(f"⬇️ Downloading Real-ESRGAN {upscale_factor}x model to {REALESRGAN_DIR}...")
            hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename=model_filename,
                local_dir=REALESRGAN_DIR,
            )
            print(f"✓ Real-ESRGAN {upscale_factor}x model downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download Real-ESRGAN {upscale_factor}x model: {e}")
            traceback.print_exc()
    else:
        print(f"✓ Real-ESRGAN {upscale_factor}x model already exists at {model_file}")

# ------------------------
# 5. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", "generate_dockerfile.py"], check=True)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")