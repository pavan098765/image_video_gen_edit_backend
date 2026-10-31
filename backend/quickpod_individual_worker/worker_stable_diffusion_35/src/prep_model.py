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
SD35_REPO = "second-state/stable-diffusion-3.5-large-GGUF"
SD35_DIR = os.path.join(MODELS_DIR, "sd35")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(SD35_DIR, exist_ok=True)

# ------------------------
# Required files
# ------------------------
REQUIRED = {
    "DIFFUSION_MODEL": "sd3.5_large-Q8_0.gguf",
    "CLIP_G_PATH": "clip_g-Q8_0.gguf",
    "CLIP_L_PATH": "clip_l.safetensors",
    "T5XXL_PATH": "t5xxl-Q8_0.gguf",
}

# ------------------------
# 1. Download snapshot (only required files)
# ------------------------
try:
    allow = list(REQUIRED.values())
    print(f"⬇️ Downloading {SD35_REPO} (only {allow}) ...")
    sd35_cache = snapshot_download(
        repo_id=SD35_REPO,
        allow_patterns=allow,
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"✓ SD 3.5 snapshot ready at {sd35_cache}")
except Exception as e:
    print("❌ Failed to download SD3.5:", e)
    traceback.print_exc()
    exit(1)

# ------------------------
# 2. Copy required files
# ------------------------
for key, filename in REQUIRED.items():
    found_file = None
    for root, _, files in os.walk(sd35_cache):
        if filename in files:
            found_file = os.path.join(root, filename)
            break
    if not found_file:
        print(f"❌ Missing required file for {key}: {filename}")
        continue

    dest_path = os.path.join(SD35_DIR, filename)
    shutil.copy2(found_file, dest_path)
    print(f"✓ {key}: copied {found_file} -> {dest_path}")

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
            if fn.endswith(".safetensors") or fn.endswith(".gguf"):
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
print("✅ SD3.5 prep finished successfully.")
print(f"All model components are in: {SD35_DIR}")
print("Set environment variable:")
print(f"  SD35_LOCAL_PATH={SD35_DIR}")
