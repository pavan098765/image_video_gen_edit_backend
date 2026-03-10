#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

WAN_REPO = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
WAN_DIR = os.path.join(MODELS_DIR, "wan_21_1b")

os.makedirs(WAN_DIR, exist_ok=True)

# ------------------------
# 1. Download WAN model
# ------------------------
if not os.listdir(WAN_DIR):
    try:
        print(f"⬇️ Downloading {WAN_REPO} into {WAN_DIR}")
        snapshot_download(
            WAN_REPO,
            local_dir=WAN_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ WAN model ready at {WAN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {WAN_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ WAN model already exists at {WAN_DIR}")

# ------------------------
# 2. Download RIFE model
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
# 3. Download Real-ESRGAN models (2x, 4x, 8x)
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
# 4. Split large safetensors with 700MB chunks
# ------------------------
def split_large_safetensors(base_dir, max_size=700 * 1024 * 1024):
    """
    Split safetensors files larger than max_size into chunks.
    max_size: 700MB (700 * 1024 * 1024 bytes)
    """
    chunk_size_mb = max_size / (1024 * 1024)
    print(f"🔍 Scanning for .safetensors > {chunk_size_mb:.0f}MB in {base_dir}")
    split_count = 0
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB) into {chunk_size_mb:.0f}MB chunks")
                    num_parts = (size + max_size - 1) // max_size
                    print(f"   Will create {num_parts} parts")
                    subprocess.run(
                        ["split", "-b", f"{int(chunk_size_mb)}M", full_path, f"{full_path}.part-"],
                        check=True
                    )
                    os.remove(full_path)
                    import glob
                    parts = glob.glob(f"{full_path}.part-*")
                    split_count += len(parts)
                    print(f"✅ Split into {len(parts)} parts, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")
    return split_count

total_splits = split_large_safetensors(MODELS_DIR, max_size=700 * 1024 * 1024)
print(f"📊 Total split files created: {total_splits}")

# ------------------------
# 5. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print(f"✅ Prep finished. WAN 2.1 1.3B model ready in {WAN_DIR}")
print(f"Set ENV FASTWAN_1B_LOCAL_PATH={WAN_DIR}")
