#!/usr/bin/env python3
"""
Prep model for FlashVSR Video Upscaler worker.
1. Clone FlashVSR-Pro
2. Build flashvsr-pro:latest Docker image
3. Download FlashVSR-v1.1 models
4. Generate Dockerfile
"""
import os
import subprocess
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FLASHVSR_PRO_DIR = os.path.join(PROJECT_ROOT, "FlashVSR-Pro")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "FlashVSR-v1.1")

os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Clone FlashVSR-Pro
if not os.path.exists(FLASHVSR_PRO_DIR) or not os.listdir(FLASHVSR_PRO_DIR):
    try:
        print(f"Cloning FlashVSR-Pro to {FLASHVSR_PRO_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/LujiaJin/FlashVSR-Pro.git", FLASHVSR_PRO_DIR],
            check=True,
        )
        # Git LFS optional (models come from huggingface_hub, not from the repo)
        try:
            subprocess.run(["git", "lfs", "install"], cwd=FLASHVSR_PRO_DIR, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Note: git-lfs not installed (optional; models are downloaded from HuggingFace)")
        print("FlashVSR-Pro cloned successfully.")
    except Exception as e:
        print(f"Failed to clone FlashVSR-Pro: {e}")
        traceback.print_exc()
        raise
else:
    print(f"FlashVSR-Pro already exists at {FLASHVSR_PRO_DIR}")

# 1b. Patch Dockerfile to limit Block-Sparse-Attention compile parallelism (avoid OOM during nvcc)
_dockerfile_path = os.path.join(FLASHVSR_PRO_DIR, "Dockerfile")
with open(_dockerfile_path, "r", encoding="utf-8") as f:
    _dockerfile = f.read()
if "MAX_JOBS" not in _dockerfile:
    _dockerfile = _dockerfile.replace(
        "ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n# Apply optimizations",
        "ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n# Limit parallel CUDA compilation (nvcc OOM)\nENV MAX_JOBS=1\n# Apply optimizations",
    )
    with open(_dockerfile_path, "w", encoding="utf-8") as f:
        f.write(_dockerfile)
    print("Patched Dockerfile: MAX_JOBS=1 for Block-Sparse-Attention build")

# 2. Build base Docker image (tagged for your registry; push manually if needed)
BASE_IMAGE = "pavanalur3/flashvsr-pro:latest"
print(f"Building {BASE_IMAGE} Docker image...")
try:
    subprocess.run(
        ["docker", "build", "-t", BASE_IMAGE, "."],
        cwd=FLASHVSR_PRO_DIR,
        check=True,
    )
    print(f"{BASE_IMAGE} built successfully.")
except Exception as e:
    print(f"Failed to build Docker image: {e}")
    traceback.print_exc()
    raise

# 3. Download FlashVSR-v1.1 models
try:
    from huggingface_hub import snapshot_download

    print(f"Downloading FlashVSR-v1.1 models to {MODELS_DIR}...")
    snapshot_download(
        "JunhaoZhuang/FlashVSR-v1.1",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("FlashVSR-v1.1 models downloaded successfully.")
except Exception as e:
    print(f"Failed to download models: {e}")
    traceback.print_exc()
    raise

# 4. Generate Dockerfile
try:
    print("Generating Dockerfile via generate_dockerfile.py...")
    subprocess.run(
        ["python3", "generate_dockerfile.py"],
        cwd=SCRIPT_DIR,
        check=True,
    )
    print("Dockerfile generated successfully.")
except Exception as e:
    print(f"Failed to generate Dockerfile: {e}")
    traceback.print_exc()
    raise

print("Prep finished. Models ready in ./models and Dockerfile created.")
