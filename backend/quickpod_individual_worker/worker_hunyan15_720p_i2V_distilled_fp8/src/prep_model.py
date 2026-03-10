#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # go one level up from src/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

HUNYAN15_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"
FP8_TRANSFORMER_REPO = "Comfy-Org/HunyuanVideo_1.5_repackaged"
FP8_TRANSFORMER_FILE = "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_cfg_distilled_fp8_scaled.safetensors"

HUNYAN15_DIR = os.path.join(MODELS_DIR, "hunyan15_720p_fp8")

os.makedirs(HUNYAN15_DIR, exist_ok=True)

# ------------------------
# 1. Download Hunyan15 model (all files except transformer)
# ------------------------
try:
    print(f"⬇️ Downloading {HUNYAN15_REPO} into {HUNYAN15_DIR}")
    snapshot_download(
        HUNYAN15_REPO,
        local_dir=HUNYAN15_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Hunyan15 repo ready at {HUNYAN15_DIR}")
except Exception as e:
    print(f"❌ Failed to download {HUNYAN15_REPO}: {e}")
    traceback.print_exc()

# ------------------------
# 2. Download FP8 transformer file and replace the original
# ------------------------
try:
    transformer_dir = os.path.join(HUNYAN15_DIR, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    
    # Remove original transformer files if they exist
    if os.path.exists(transformer_dir):
        for f in os.listdir(transformer_dir):
            if f.endswith(".safetensors"):
                old_file = os.path.join(transformer_dir, f)
                print(f"🗑️  Removing original transformer file: {f}")
                os.remove(old_file)
    
    # Download FP8 transformer file
    # The file will be downloaded to: transformer_dir/split_files/diffusion_models/filename
    # We need to move it to transformer_dir/diffusion_pytorch_model.safetensors
    fp8_file_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    print(f"⬇️ Downloading FP8 transformer from {FP8_TRANSFORMER_REPO}")
    print(f"   File: {FP8_TRANSFORMER_FILE}")
    print(f"   Destination: {fp8_file_path}")
    
    # Download to a temp location first
    temp_download_dir = os.path.join(transformer_dir, "temp_download")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    hf_hub_download(
        repo_id=FP8_TRANSFORMER_REPO,
        filename=FP8_TRANSFORMER_FILE,
        local_dir=temp_download_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    # Find the downloaded file (it will be in the nested directory structure)
    downloaded_file = os.path.join(temp_download_dir, FP8_TRANSFORMER_FILE)
    if not os.path.exists(downloaded_file):
        # Try alternative path structure
        alt_path = os.path.join(temp_download_dir, FP8_TRANSFORMER_FILE.split("/")[-1])
        if os.path.exists(alt_path):
            downloaded_file = alt_path
        else:
            # Search for the file
            for root, dirs, files in os.walk(temp_download_dir):
                for f in files:
                    if f.endswith(".safetensors") and "fp8" in f.lower():
                        downloaded_file = os.path.join(root, f)
                        break
    
    if os.path.exists(downloaded_file):
        # Move to final location
        if os.path.exists(fp8_file_path):
            os.remove(fp8_file_path)
        shutil.move(downloaded_file, fp8_file_path)
        print(f"✓ FP8 transformer moved to: {fp8_file_path}")
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_download_dir)
        except Exception:
            pass
    else:
        raise FileNotFoundError(f"Downloaded FP8 transformer file not found. Expected at: {downloaded_file}")
    
    print(f"✓ FP8 transformer downloaded and ready at {fp8_file_path}")
except Exception as e:
    print(f"❌ Failed to download FP8 transformer: {e}")
    traceback.print_exc()


# ------------------------
# 2. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=850 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "850M", full_path, f"{full_path}.part-"],
                        check=True,
                    )
                    os.remove(full_path)

                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")


split_large_safetensors(MODELS_DIR)

# ------------------------
# 3. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", "generate_dockerfile.py"], check=True)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")