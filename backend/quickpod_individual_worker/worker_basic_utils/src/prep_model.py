#!/usr/bin/env python3
import os, traceback
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Config
# ------------------------
CACHE_DIR = "/app/models/basic_utils"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
REALESRGAN_HF_REPO = "rockapaper/realesrgan"
REALESRGAN_DIR = os.path.join(CACHE_DIR, "realesrgan")

# ------------------------
# Downloader helper
# ------------------------
def download_model(repo_id: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"✓ {repo_id} already exists at {target_dir}")
        return target_dir

    print(f"⬇️ Downloading {repo_id} -> {target_dir}")
    path = snapshot_download(
        repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Downloaded {repo_id} -> {path}")
    return path

# ------------------------
# Run
# ------------------------
os.makedirs(CACHE_DIR, exist_ok=True)

birefnet_dir = os.path.join(CACHE_DIR, "birefnet")
lama_dir = os.path.join(CACHE_DIR, "lama")

# Download BiRefNet, not LaMa (since lama.pt is pre-baked)
download_model(BIREFNET_REPO, birefnet_dir)
os.makedirs(lama_dir, exist_ok=True)

# Download Real-ESRGAN models (2x, 4x) from rockapaper/realesrgan
os.makedirs(REALESRGAN_DIR, exist_ok=True)
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

print("✅ All models ready:")
print("   BIREFNET_LOCAL_PATH =", birefnet_dir)
print("   LAMA_LOCAL_PATH =", os.path.join(lama_dir, "lama.pt"))
print("   REALESRGAN_LOCAL_PATH =", REALESRGAN_DIR)
