#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# ------------------------
# Config
# ------------------------
CACHE_DIR = "/app/models/basic_utils"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"

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

print("✅ All models ready:")
print("   BIREFNET_LOCAL_PATH =", birefnet_dir)
print("   LAMA_LOCAL_PATH =", os.path.join(lama_dir, "lama.pt"))
