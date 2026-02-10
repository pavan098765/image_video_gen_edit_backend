#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")

LOCAL_REALESRGAN_SRC = os.path.join(PROJECT_ROOT, "realesrgan_models")
LOCAL_REALESRGAN_DST = REALESRGAN_DIR

os.makedirs(REALESRGAN_DIR, exist_ok=True)

# ------------------------
# Model URLs (from huggingface_app.py)
# ------------------------
MODELS_TO_DOWNLOAD = {
    "realesr-general-x4v3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "repo": None,  # Direct download
    },
    "GFPGANv1.2.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
        "repo": None,
    },
    "GFPGANv1.3.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "repo": None,
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "repo": None,
    },
    "RestoreFormer.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "repo": None,
    },
    "CodeFormer.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
        "repo": None,
    },
}

# ------------------------
# Download models
# ------------------------
print("⬇️ Downloading GFPGAN and RealESRGAN models...")

for filename, info in MODELS_TO_DOWNLOAD.items():
    target_path = os.path.join(REALESRGAN_DIR, filename)
    
    if os.path.exists(target_path):
        print(f"  ✓ {filename} already exists")
        continue
    
    try:
        if info["repo"]:
            # Download from HuggingFace Hub
            print(f"  ⬇️ Downloading {filename} from HuggingFace Hub...")
            downloaded_path = hf_hub_download(
                repo_id=info["repo"],
                filename=filename,
                resume_download=True,
            )
            shutil.copy2(downloaded_path, target_path)
        else:
            # Download directly from URL
            print(f"  ⬇️ Downloading {filename} from {info['url']}...")
            import urllib.request
            urllib.request.urlretrieve(info["url"], target_path)
        
        file_size_mb = os.path.getsize(target_path) / 1024 / 1024
        print(f"  ✓ {filename} downloaded ({file_size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ❌ Failed to download {filename}: {e}")
        traceback.print_exc()

print("✓ Model downloads completed\n")

# ------------------------
# Copy local Realesrgan models (optional)
# ------------------------
if os.path.exists(LOCAL_REALESRGAN_SRC):
    print(f"📦 Copying {LOCAL_REALESRGAN_SRC} -> {LOCAL_REALESRGAN_DST}")
    shutil.copytree(LOCAL_REALESRGAN_SRC, LOCAL_REALESRGAN_DST, dirs_exist_ok=True)
    print("✓ Realesrgan models copied.")
else:
    print("⚠️ No local realesrgan_models/ found, skipping.")

# ------------------------
# Split large model files recursively
# ------------------------
def split_large_files(base_dir, max_size=500 * 1024 * 1024, extensions=(".pth", ".safetensors")):
    print(f"🔍 Scanning for large files > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if any(fn.endswith(ext) for ext in extensions):
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


split_large_files(MODELS_DIR)

# ------------------------
# Generate runtime Dockerfile
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
        check=True,
        cwd=SCRIPT_DIR
    )
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")
