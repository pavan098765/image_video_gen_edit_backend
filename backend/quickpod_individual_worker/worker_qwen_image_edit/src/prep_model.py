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
QWEN_REPO = "Qwen/Qwen-Image-Edit"
NUNCHAKU_REPO = "nunchaku-tech/nunchaku-qwen-image-edit"

QWEN_DIR = os.path.join(MODELS_DIR, "qwen_image_edit")
NUNCHAKU_DIR = os.path.join(MODELS_DIR, "nunchaku_qwen")
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(QWEN_DIR, exist_ok=True)
os.makedirs(NUNCHAKU_DIR, exist_ok=True)
os.makedirs(REALESRGAN_DIR, exist_ok=True)

# ------------------------
# 1. Download Qwen (skip transformer/*)
# ------------------------
if not os.listdir(QWEN_DIR):
    try:
        print(f"⬇️ Downloading {QWEN_REPO} into {QWEN_DIR} (excluding transformer/*)")
        snapshot_download(
            QWEN_REPO,
            local_dir=QWEN_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["transformer/*"],
        )
        print(f"✓ Qwen repo ready at {QWEN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {QWEN_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ Qwen repo already exists at {QWEN_DIR}")

# ------------------------
# 2. Download transformer file from Nunchaku
# ------------------------
rank = 128
steps = 8
filename = f"svdq-fp4_r{rank}-qwen-image-edit-lightningv1.0-{steps}steps.safetensors"
target_path = os.path.join(NUNCHAKU_DIR, filename)

if not os.path.exists(target_path):
    try:
        print(f"⬇️ Downloading {filename} from {NUNCHAKU_REPO}")
        downloaded_path = hf_hub_download(
            repo_id=NUNCHAKU_REPO,
            filename=filename,
            resume_download=True,
        )
        shutil.copy2(downloaded_path, target_path)
        print(f"✓ Transformer ready at {target_path}")
    except Exception as e:
        print(f"❌ Failed to download transformer: {e}")
        traceback.print_exc()
else:
    print(f"✓ Transformer already exists at {target_path}")

# ------------------------
# 3. Download Real-ESRGAN models (2x, 4x)
# ------------------------
# Download Real-ESRGAN models - try "plus" versions first, fallback to regular versions
realesrgan_models = [
    ("RealESRGAN_x2plus.pth", "RealESRGAN_x2.pth", 2),
    ("RealESRGAN_x4plus.pth", "RealESRGAN_x4.pth", 4),
]

for preferred_name, fallback_name, scale in realesrgan_models:
    model_file = os.path.join(REALESRGAN_DIR, preferred_name)
    
    if os.path.exists(model_file):
        print(f"✓ Real-ESRGAN {scale}x model already exists at {model_file}")
        continue
    
    # Try downloading preferred name first
    downloaded = False
    try:
        print(f"⬇️ Downloading Real-ESRGAN {scale}x model ({preferred_name})...")
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename=preferred_name,
            local_dir=REALESRGAN_DIR,
        )
        # Verify the file was downloaded
        if os.path.exists(model_file):
            downloaded = True
            print(f"✓ Real-ESRGAN {scale}x model downloaded successfully as {preferred_name}")
        else:
            raise FileNotFoundError(f"Downloaded file not found at expected path: {model_file}")
    except Exception as e:
        # If preferred name doesn't exist, try fallback name
        try:
            print(f"  ⚠️  {preferred_name} not found, trying {fallback_name}...")
            downloaded_path = hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename=fallback_name,
                local_dir=REALESRGAN_DIR,
            )
            # Copy/rename to expected name
            if os.path.exists(downloaded_path):
                shutil.copy2(downloaded_path, model_file)
                # Remove the fallback file if it's different
                if downloaded_path != model_file and os.path.exists(downloaded_path):
                    try:
                        os.remove(downloaded_path)
                    except:
                        pass
                if os.path.exists(model_file):
                    downloaded = True
                    print(f"✓ Real-ESRGAN {scale}x model downloaded and renamed to {preferred_name}")
                else:
                    raise FileNotFoundError(f"Failed to create {model_file} from {downloaded_path}")
            else:
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")
        except Exception as e2:
            print(f"❌ Failed to download Real-ESRGAN {scale}x model: {e2}")
            traceback.print_exc()

# ------------------------
# 4. Split large safetensors recursively
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
# 5. Generate runtime Dockerfile with per-part COPY
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
print("✅ Qwen Image Edit prep finished successfully.")
print(f"Qwen base repo at: {QWEN_DIR}")
print(f"Nunchaku transformer at: {NUNCHAKU_DIR}")
print("Set env vars:")
print(f"  QWEN_LOCAL_PATH={QWEN_DIR}")
print(f"  NUNCHAKU_QWEN_LOCAL_PATH={NUNCHAKU_DIR}")
