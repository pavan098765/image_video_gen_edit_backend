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
SD15_REPO = "second-state/stable-diffusion-v1-5-GGUF"
VAE_REPO = "stabilityai/sd-vae-ft-mse-original"

SD15_DIR = os.path.join(MODELS_DIR, "sd15")
VAE_DIR = os.path.join(MODELS_DIR, "sd15_vae")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(SD15_DIR, exist_ok=True)
os.makedirs(VAE_DIR, exist_ok=True)

# ------------------------
# 1. Download SD15 GGUF (Q8_0 file only)
# ------------------------
try:
    print(f"⬇️ Downloading {SD15_REPO} (Q8_0 only) ...")
    sd15_cache = snapshot_download(
        repo_id=SD15_REPO,
        allow_patterns=["stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"],
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),

    )
    print(f"✓ SD15 GGUF snapshot at {sd15_cache}")

    found_file = None
    for root, _, files in os.walk(sd15_cache):
        for f in files:
            if f == "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf":
                found_file = os.path.join(root, f)
                break
        if found_file:
            break

    if not found_file:
        raise FileNotFoundError("Q8_0 GGUF file not found in snapshot!")

    target_path = os.path.join(SD15_DIR, "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf")
    shutil.copy2(found_file, target_path)
    print(f"✓ Copied GGUF model to {target_path}")

except Exception as e:
    print("❌ SD15 GGUF download failed:", e)
    traceback.print_exc()

# ------------------------
# 2. Download VAE (safetensors only)
# ------------------------
try:
    print(f"⬇️ Downloading {VAE_REPO} (safetensors only) ...")
    vae_cache = snapshot_download(
        repo_id=VAE_REPO,
        allow_patterns=["*.safetensors"],
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"✓ VAE snapshot at {vae_cache}")

    found_file = None
    for root, _, files in os.walk(vae_cache):
        for f in files:
            if f.endswith(".safetensors"):
                found_file = os.path.join(root, f)
                break
        if found_file:
            break

    if not found_file:
        raise FileNotFoundError("No .safetensors VAE file found!")

    target_path = os.path.join(VAE_DIR, "vae-ft-mse-840000-ema-pruned.safetensors")
    shutil.copy2(found_file, target_path)
    print(f"✓ Copied VAE model to {target_path}")

except Exception as e:
    print("❌ VAE download failed:", e)
    traceback.print_exc()

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
print("✅ SD15 prep finished successfully.")
print(f"Diffusion model in: {SD15_DIR}")
print(f"VAE in: {VAE_DIR}")
print("Set env vars:")
print(f"  SD15_LOCAL_PATH={SD15_DIR}")
print(f"  SD15_VAE_PATH={VAE_DIR}")
