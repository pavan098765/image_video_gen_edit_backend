#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------------
# Config
# ------------------------
SDXL_REPO = "John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"
SDXL_DIR = os.path.join(MODELS_DIR, "sdxl_realism_illustrious")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(SDXL_DIR, exist_ok=True)

# ------------------------
# 1. Download SDXL Realism Illustrious repo
# ------------------------
if not os.listdir(SDXL_DIR):
    try:
        print(f"⬇️ Downloading {SDXL_REPO} into {SDXL_DIR} ...")
        snapshot_download(
            SDXL_REPO,
            local_dir=SDXL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token="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",
        )
        print(f"✓ SDXL Realism Illustrious ready at {SDXL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {SDXL_REPO}: {e}")
        traceback.print_exc()
        exit(1)
else:
    print(f"✓ SDXL Realism Illustrious already exists at {SDXL_DIR}")

# ------------------------
# 2. Check for embeddings
# ------------------------
emb_dir = os.path.join(SDXL_DIR, "embeddings")
pos = os.path.join(emb_dir, "Stable_Yogis_Illustrious_Positives.safetensors")
neg = os.path.join(emb_dir, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

print("🔍 Checking embeddings...")
print("  POS:", "✓ found" if os.path.exists(pos) else "❌ missing")
print("  NEG:", "✓ found" if os.path.exists(neg) else "❌ missing")

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
print("✅ SDXL Realism Illustrious prep finished successfully.")
print(f"Model available at: {SDXL_DIR}")
print(f"Embedding directory: {emb_dir}")
print("Set environment variables:")
print(f"  SDXL_LOCAL_PATH={SDXL_DIR}")
print(f"  SDXL_EMBEDDINGS_PATH={emb_dir}")
