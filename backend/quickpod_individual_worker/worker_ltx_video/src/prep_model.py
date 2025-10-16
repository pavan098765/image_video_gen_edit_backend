#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

BASE_DIR = "/models/basic_utils"
os.makedirs(BASE_DIR, exist_ok=True)

def download_rmbg():
    target_dir = os.path.join(BASE_DIR, "rmbg")
    os.makedirs(target_dir, exist_ok=True)
    print(f"⬇️ Downloading RMBG model into {target_dir} ...")

    # RMBG-1.4 (BriaAI)
    snapshot_download(
        repo_id="briaai/RMBG-1.4",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"],  # cut weight bloat
    )
    print("✅ RMBG model ready")

def download_lama():
    target_dir = os.path.join(BASE_DIR, "lama")
    os.makedirs(target_dir, exist_ok=True)
    print(f"⬇️ Downloading SimpleLama model into {target_dir} ...")

    # SimpleLama auto-download (uses HF cache normally)
    # Snapshot from the official repo:
    snapshot_download(
        repo_id="advimman/lama",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print("✅ LaMa model ready")

if __name__ == "__main__":
    download_rmbg()
    download_lama()
    print("🎉 All basic utils models prepared under /models/basic_utils")
