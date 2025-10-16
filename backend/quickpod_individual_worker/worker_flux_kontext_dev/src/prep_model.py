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
FLUX_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
NUNCHAKU_REPO = "nunchaku-tech/nunchaku-flux.1-kontext-dev"

FLUX_DIR = os.path.join(MODELS_DIR, "flux_kontext_dev")
NUNCHAKU_DIR = os.path.join(MODELS_DIR, "nunchaku_flux")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(FLUX_DIR, exist_ok=True)
os.makedirs(NUNCHAKU_DIR, exist_ok=True)

# ------------------------
# 1. Download Flux-Kontext (skip transformer + big safetensors)
# ------------------------
try:
    print(f"⬇️ Downloading {FLUX_REPO} into {FLUX_DIR} (excluding transformer/* & large safetensors)")
    snapshot_download(
        FLUX_REPO,
        local_dir=FLUX_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        token="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",
        ignore_patterns=[
            "transformer/*",
            "ae.safetensors",
            "flux1-kontext-dev.safetensors"
        ],
    )
    print(f"✓ Flux-Kontext model ready at {FLUX_DIR}")
except Exception as e:
    print(f"❌ Failed to download {FLUX_REPO}: {e}")
    traceback.print_exc()

# ------------------------
# 2. Download transformer weight from Nunchaku
# ------------------------
filename = "svdq-fp4_r32-flux.1-kontext-dev.safetensors"
target_path = os.path.join(NUNCHAKU_DIR, filename)

if not os.path.exists(target_path):
    try:
        print(f"⬇️ Downloading transformer {filename} from {NUNCHAKU_REPO}")
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
# 3. Copy local Realesrgan models (optional)
# ------------------------
if os.path.exists(LOCAL_REALESRGAN_SRC):
    print(f"📦 Copying {LOCAL_REALESRGAN_SRC} -> {LOCAL_REALESRGAN_DST}")
    shutil.copytree(LOCAL_REALESRGAN_SRC, LOCAL_REALESRGAN_DST, dirs_exist_ok=True)
    print("✓ Realesrgan models copied.")
else:
    print("⚠️ No local realesrgan_models/ found, skipping.")

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
print("✅ Flux-Kontext Dev prep finished.")
print(f"Flux model at: {FLUX_DIR}")
print(f"Nunchaku transformer at: {NUNCHAKU_DIR}")
print("Set env vars:")
print(f"  FLUX_KONTEXT_LOCAL_PATH = {FLUX_DIR}")
print(f"  NUNCHAKU_LOCAL_PATH     = {NUNCHAKU_DIR}")
