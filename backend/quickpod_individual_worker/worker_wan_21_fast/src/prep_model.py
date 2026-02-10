#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WAN_DIR = os.path.join(MODELS_DIR, "wan_21_fast")
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
LORA_DIR = os.path.join(WAN_DIR, "lora")

os.makedirs(WAN_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# ------------------------
# 1. Download WAN 2.1 14B model
# ------------------------
# Check for required model subdirectories to verify model is actually downloaded
REQUIRED_SUBDIRS = ["image_encoder", "vae", "transformer"]
model_exists = all(os.path.exists(os.path.join(WAN_DIR, subdir)) for subdir in REQUIRED_SUBDIRS)

if not model_exists:
    try:
        print(f"⬇️ Downloading {WAN_REPO} into {WAN_DIR}")
        snapshot_download(
            WAN_REPO,
            local_dir=WAN_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ WAN 2.1 14B model ready at {WAN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {WAN_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ WAN 2.1 14B model already exists at {WAN_DIR}")

# ------------------------
# 2. Download CausVid LoRA
# ------------------------
lora_path = os.path.join(LORA_DIR, LORA_FILENAME)
if not os.path.exists(lora_path):
    try:
        print(f"⬇️ Downloading LoRA {LORA_REPO_ID}/{LORA_FILENAME} into {LORA_DIR}")
        hf_hub_download(
            repo_id=LORA_REPO_ID,
            filename=LORA_FILENAME,
            local_dir=LORA_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"✓ CausVid LoRA ready at {lora_path}")
    except Exception as e:
        print(f"❌ Failed to download LoRA: {e}")
        traceback.print_exc()
else:
    print(f"✓ CausVid LoRA already exists at {lora_path}")

# ------------------------
# 3. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=1200 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and not fn.startswith("."):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "1200M", full_path, f"{full_path}.part-"],
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

print(f"✅ Prep finished. Wan 2.1 14B Fast model ready in {WAN_DIR}")
print(f"Set ENV WAN_21_LOCAL_PATH={WAN_DIR}")
print(f"Set ENV WAN_21_LORA_LOCAL_PATH={LORA_DIR}")
