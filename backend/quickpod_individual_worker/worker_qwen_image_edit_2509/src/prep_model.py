#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # go one level up from src/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"
NUNCHAKU_REPO = "nunchaku-tech/nunchaku-qwen-image-edit-2509"

QWEN_DIR = os.path.join(MODELS_DIR, "qwen_image_edit_2509")
NUNCHAKU_DIR = os.path.join(MODELS_DIR, "nunchaku_qwen_2509")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(QWEN_DIR, exist_ok=True)
os.makedirs(NUNCHAKU_DIR, exist_ok=True)

# ------------------------
# 1. Download Qwen (skip transformer/*)
# ------------------------
try:
    print(f"⬇️ Downloading {QWEN_REPO} into {QWEN_DIR} (excluding transformer/*)")
    snapshot_download(
        QWEN_REPO,
        local_dir=QWEN_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["transformer/*"],  # skip heavy transformer weights
    )
    print(f"✓ Qwen repo ready at {QWEN_DIR}")
except Exception as e:
    print(f"❌ Failed to download {QWEN_REPO}: {e}")
    traceback.print_exc()


# ------------------------
# 2. Download transformer file from Nunchaku
# ------------------------
rank = 128
steps = 8
filename = (
    f"svdq-fp4_r{rank}-qwen-image-edit-2509-lightningv2.0-{steps}steps.safetensors"
)
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
# 3. Copy local Realesrgan models
# ------------------------
if os.path.exists(LOCAL_REALESRGAN_SRC):
    print(f"📦 Copying {LOCAL_REALESRGAN_SRC} -> {LOCAL_REALESRGAN_DST}")
    shutil.copytree(LOCAL_REALESRGAN_SRC, LOCAL_REALESRGAN_DST, dirs_exist_ok=True)
    # os.remove(LOCAL_REALESRGAN_SRC)  # remove source to avoid confusion
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
                        check=True,
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

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")
