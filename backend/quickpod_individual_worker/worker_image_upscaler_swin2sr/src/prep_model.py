#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# ------------------------
# Config
# ------------------------
BASE_DIR_SWIN2SR = "/app/models/swin2sr"

REPOS = {
    "swin2sr_x2": "caidas/swin2SR-classical-sr-x2-64",
    "swin2sr_x4": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "swin2sr_x4_classic": "caidas/swin2SR-classical-sr-x4-64",
}

# ------------------------
# Downloader
# ------------------------
def download_swin2sr_all():
    for name, repo_id in REPOS.items():
        local_dir = os.path.join(BASE_DIR_SWIN2SR, name)
        os.makedirs(local_dir, exist_ok=True)

        if os.listdir(local_dir):
            print(f"✓ {name} already exists at {local_dir}")
            continue

        print(f"⬇️ Downloading {repo_id} into {local_dir} ...")
        dl_kwargs = {
            "repo_id": repo_id,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "ignore_patterns": ["pytorch_model.bin"],
        }
        tok = os.getenv("HF_TOKEN", "").strip()
        if tok:
            dl_kwargs["token"] = tok
        snapshot_download(**dl_kwargs)
        print(f"✅ {name} ready at {local_dir}")


if __name__ == "__main__":
    download_swin2sr_all()
