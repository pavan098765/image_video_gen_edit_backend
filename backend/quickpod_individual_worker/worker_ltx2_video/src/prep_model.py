#!/usr/bin/env python3
"""
prep_model.py - Download and prepare LTX-2 models for Docker image

This script:
1. Downloads required model files from HuggingFace Lightricks/LTX-2
2. Splits large safetensors files for Docker layer caching
3. Calls generate_dockerfile.py to create the Dockerfile

Model: ltx-2-19b-dev-fp4.safetensors (20 GB)
- Works safely on both RTX 4090 (24GB) and RTX 5090 (32GB)
- No OOM risk

Usage:
    python prep_model.py                    # Full: download + split + dockerfile
    python prep_model.py --dockerfile-only  # Skip download, just regenerate Dockerfile
    python prep_model.py --skip-download    # Same as --dockerfile-only
"""

import os
import sys
import subprocess
import traceback
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Prepare LTX-2 models for Docker image")
parser.add_argument("--dockerfile-only", "--skip-download", action="store_true",
                    help="Skip model download, just regenerate Dockerfile (use if models already downloaded)")
args = parser.parse_args()

SKIP_DOWNLOAD = args.dockerfile_only

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # go one level up from src/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# HuggingFace repositories
LTX2_REPO = "Lightricks/LTX-2"
GEMMA_REPO = "google/gemma-3-12b-it"

# Model directories
LTX2_DIR = os.path.join(MODELS_DIR, "ltx2_video")
GEMMA_DIR = os.path.join(MODELS_DIR, "gemma")

# We ONLY use fp4 model - works safely on both RTX 4090 (24GB) and RTX 5090 (32GB)
# No risk of OOM with this checkpoint
CHECKPOINT_FILE = "ltx-2-19b-dev-fp4.safetensors"  # 20 GB
CHECKPOINT_SIZE_GB = 20.0

# Additional required files
SPATIAL_UPSAMPLER_FILE = "ltx-2-spatial-upscaler-x2-1.0.safetensors"  # 996 MB
DISTILLED_LORA_FILE = "ltx-2-19b-distilled-lora-384.safetensors"  # 7.67 GB

# Diffusers-compatible model directories to download
# NOTE: We skip "text_encoder" (100GB!) because:
# - ltx-pipelines loads Gemma via gemma_root parameter
# - We download Gemma directly from google/gemma-3-12b-it (~24GB) instead
# - The text_encoder folder is just a redundant diffusers-format copy of Gemma
DIFFUSERS_SUBDIRS = [
    "scheduler",
    "tokenizer", 
    # "text_encoder",  # SKIP - redundant, use gemma_root instead (saves 100GB!)
    "vae",
    "connectors",
]

os.makedirs(LTX2_DIR, exist_ok=True)
os.makedirs(GEMMA_DIR, exist_ok=True)

print("=" * 60)
print("LTX-2 Model Preparation")
print(f"Target: {CHECKPOINT_FILE} ({CHECKPOINT_SIZE_GB} GB)")
print(f"Works on: RTX 4090 (24GB), RTX 5090 (32GB) - No OOM risk")
if SKIP_DOWNLOAD:
    print("Mode: --dockerfile-only (skipping downloads)")
print("=" * 60)
print()

if SKIP_DOWNLOAD:
    print("⏩ Skipping model downloads (--dockerfile-only mode)")
    print("   Using existing model files in models/ directory")
    print()
else:
    # Import huggingface_hub only if we need to download
    from huggingface_hub import hf_hub_download, snapshot_download

# ------------------------
# 1. Download main checkpoint (FP4 - 20 GB)
# ------------------------
checkpoint_path = os.path.join(LTX2_DIR, CHECKPOINT_FILE)
if not SKIP_DOWNLOAD:
    if not os.path.exists(checkpoint_path):
        try:
            print(f"⬇️ Downloading {CHECKPOINT_FILE} from {LTX2_REPO}")
            print(f"   This is ~{CHECKPOINT_SIZE_GB} GB, may take a while...")
            downloaded_path = hf_hub_download(
                repo_id=LTX2_REPO,
                filename=CHECKPOINT_FILE,
                local_dir=LTX2_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ Checkpoint ready at {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to download checkpoint: {e}")
            traceback.print_exc()
    else:
        size_gb = os.path.getsize(checkpoint_path) / 1024**3
        print(f"✓ Checkpoint already exists at {checkpoint_path} ({size_gb:.2f} GB)")

# ------------------------
# 2. Download spatial upsampler (996 MB)
# ------------------------
upsampler_path = os.path.join(LTX2_DIR, SPATIAL_UPSAMPLER_FILE)
if not SKIP_DOWNLOAD:
    if not os.path.exists(upsampler_path):
        try:
            print(f"⬇️ Downloading {SPATIAL_UPSAMPLER_FILE} from {LTX2_REPO}")
            hf_hub_download(
                repo_id=LTX2_REPO,
                filename=SPATIAL_UPSAMPLER_FILE,
                local_dir=LTX2_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ Spatial upsampler ready at {upsampler_path}")
        except Exception as e:
            print(f"❌ Failed to download upsampler: {e}")
            traceback.print_exc()
    else:
        print(f"✓ Spatial upsampler already exists at {upsampler_path}")

# ------------------------
# 3. Download distilled LoRA (7.67 GB) - OPTIONAL
# ------------------------
# NOTE: LoRA is NOT required for inference!
# - "distilled" pipeline (default) does NOT use LoRA
# - Other pipelines (two_stage, ic_lora, keyframe) can work without it
# - LoRA is a model distillation technique for speedup, NOT for poses/canny
# - Skip this to save 7.67 GB if you only use the "distilled" pipeline
# ------------------------
DOWNLOAD_LORA = os.environ.get("LTX2_DOWNLOAD_LORA", "0") == "1"  # Set to "1" to download

lora_path = os.path.join(LTX2_DIR, DISTILLED_LORA_FILE)
if not SKIP_DOWNLOAD:
    if DOWNLOAD_LORA:
        if not os.path.exists(lora_path):
            try:
                print(f"⬇️ Downloading {DISTILLED_LORA_FILE} from {LTX2_REPO} (7.67 GB)")
                print("   Note: This is optional - only needed for two_stage/ic_lora pipelines")
                hf_hub_download(
                    repo_id=LTX2_REPO,
                    filename=DISTILLED_LORA_FILE,
                    local_dir=LTX2_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"✓ Distilled LoRA ready at {lora_path}")
            except Exception as e:
                print(f"⚠️ Distilled LoRA download failed: {e}")
        else:
            print(f"✓ Distilled LoRA already exists at {lora_path}")
    else:
        print(f"⏩ Skipping LoRA download (set LTX2_DOWNLOAD_LORA=1 to download)")
        print("   LoRA is optional - not needed for 'distilled' pipeline (default)")

# ------------------------
# 4. Download diffusers-compatible components
# ------------------------
if not SKIP_DOWNLOAD:
    print(f"\n⬇️ Downloading diffusers components from {LTX2_REPO}")
    for subdir in DIFFUSERS_SUBDIRS:
        subdir_path = os.path.join(LTX2_DIR, subdir)
        if os.path.exists(subdir_path) and os.listdir(subdir_path):
            print(f"✓ {subdir}/ already exists")
            continue
        
        try:
            print(f"   Downloading {subdir}/...")
            snapshot_download(
                repo_id=LTX2_REPO,
                local_dir=LTX2_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[f"{subdir}/*"],
            )
            print(f"✓ {subdir}/ ready")
        except Exception as e:
            print(f"⚠️ Failed to download {subdir}/: {e}")

# ------------------------
# 5. Download Gemma 3 text encoder
# ------------------------
if not SKIP_DOWNLOAD:
    if not os.path.exists(GEMMA_DIR) or not os.listdir(GEMMA_DIR):
        try:
            print(f"\n⬇️ Downloading Gemma 3 text encoder from {GEMMA_REPO}")
            print("   This is required for text-to-video generation...")
            snapshot_download(
                repo_id=GEMMA_REPO,
                local_dir=GEMMA_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                token="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"
            )
            print(f"✓ Gemma text encoder ready at {GEMMA_DIR}")
        except Exception as e:
            print(f"❌ Failed to download Gemma: {e}")
            traceback.print_exc()
    else:
        print(f"\n✓ Gemma text encoder already exists at {GEMMA_DIR}")


# ------------------------
# 6. Split large safetensors recursively (for Docker layer caching)
# ------------------------
# Use 800MB chunks: 20GB file → ~25 parts → stays under Docker's 127 layer limit
CHUNK_SIZE_MB = 800
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

def split_large_safetensors(base_dir, max_size=CHUNK_SIZE_BYTES):
    """Split safetensors > 800MB into parts for Docker layer caching."""
    print(f"\n🔍 Scanning for .safetensors > {max_size/1024/1024:.0f}MB in {base_dir}")
    
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                
                # Skip if already split
                if ".part-" in fn:
                    continue
                
                size = os.path.getsize(full_path)
                if size > max_size:
                    num_parts = (size // max_size) + 1
                    print(f"⚡ Splitting {fn} ({size/1024/1024/1024:.1f} GB) into ~{num_parts} parts of {CHUNK_SIZE_MB}MB")
                    try:
                        subprocess.run(
                            ["split", "-b", f"{CHUNK_SIZE_MB}M", full_path, f"{full_path}.part-"],
                            check=True,
                        )
                        os.remove(full_path)
                        print(f"   ✅ Split done, original removed")
                    except Exception as e:
                        print(f"   ❌ Split failed: {e}")
                else:
                    print(f"✔ {fn} ({size/1024/1024:.1f} MB) kept as-is")


split_large_safetensors(MODELS_DIR)

# ------------------------
# 7. Generate Dockerfile via generate_dockerfile.py
# ------------------------
print("\n" + "=" * 60)
print("Generating Dockerfile...")
print("=" * 60)

try:
    generate_script = os.path.join(SCRIPT_DIR, "generate_dockerfile.py")
    subprocess.run(["python3", generate_script], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

# ------------------------
# Summary
# ------------------------
print("\n" + "=" * 60)
print("✅ Prep finished!")
print("=" * 60)
print(f"Models directory: {MODELS_DIR}")
print(f"Dockerfile: {os.path.join(PROJECT_ROOT, 'Dockerfile')}")
print()
print("To build the Docker image:")
print(f"  cd {PROJECT_ROOT}")
print("  docker build -t ltx2-video-worker .")
print()
print("Model configuration:")
print(f"  Checkpoint: {CHECKPOINT_FILE} (20 GB)")
print("  VRAM: Works on RTX 4090 (24GB) and RTX 5090 (32GB)")
print("  No OOM risk")
print()
if not DOWNLOAD_LORA:
    print("⚠️  LoRA NOT downloaded (optional, saves 7.67 GB)")
    print("   - Not needed for 'distilled' pipeline (default)")
    print("   - Set LTX2_DOWNLOAD_LORA=1 to download for other pipelines")
else:
    print("✓ LoRA downloaded (optional, improves quality for two_stage/ic_lora)")