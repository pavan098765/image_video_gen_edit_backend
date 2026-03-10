#!/usr/bin/env python3
"""
generate_dockerfile.py - Generate Dockerfile for LTX-2 Video Worker

Called by prep_model.py after models are downloaded.
Generates Dockerfile with:
- Proper COPY commands for model files
- Split files (.part-*) with 800MB chunks (~25 parts for 20GB model)
- Environment variables for LTX-2
- Worker setup

Model: ltx-2-19b-dev-fp4.safetensors (20 GB)
Works on: RTX 5090 (32GB) - No OOM risk
"""

import os

# ------------------------
# Resolve project root (one level above src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Base image with CUDA, PyTorch, and common dependencies pre-installed
BASE_IMAGE = "pavanalur3/deps-base-extended:py311-cu129"

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WORKER_DIR = os.path.join(PROJECT_ROOT, "src")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")


def write_header(f):
    """Write Dockerfile header with base image."""
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("# --- LTX-2 Video Worker ---\n\n")


# Match chunk size from prep_model.py (800MB)
CHUNK_SIZE_MB = 800
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

def copy_non_chunk_files(f):
    """
    Copy everything under models/ except:
    - .part-* files (handled separately for layer caching)
    - safetensors > 800MB (should be split)
    - .cache or hidden dirs
    """
    if not os.path.exists(MODELS_DIR):
        f.write("# WARNING: models/ directory not found - run prep_model.py first\n")
        return
    
    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs like .cache, .git, etc.
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fn in files:
            # Skip part files
            if ".part-" in fn:
                continue
            
            # Skip large safetensors (should be split)
            if fn.endswith(".safetensors"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > CHUNK_SIZE_BYTES:
                    continue

            abs_path = os.path.join(root, fn)
            rel_src = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
            f.write(f"COPY {rel_src} {dest_dir}/\n")


def copy_chunks(f):
    """
    Copy .part-* files (800MB chunks, ~25 parts for 20GB model).
    
    Each part gets its own COPY for better layer caching.
    """
    if not os.path.exists(MODELS_DIR):
        return
    
    parts = []
    for root, _, files in os.walk(MODELS_DIR):
        for fn in files:
            if ".part-" in fn:
                parts.append(os.path.join(root, fn))
    parts.sort()
    
    for p in parts:
        rel_src = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
        rel_to_models = os.path.relpath(os.path.dirname(p), MODELS_DIR).replace("\\", "/")
        dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
        f.write(f"COPY {rel_src} {dest_dir}/\n")


def write_rest(f):
    """Write worker code and final setup."""
    f.write("\n# Worker code (last to leverage cache)\n")
    f.write("COPY src /app/worker/src\n\n")

    # Fix torchvision C++ extensions (reinstall to ensure proper CUDA bindings)
    # Then upgrade transformers for Gemma 3 support (requires >=4.49.0)
    f.write("# Reinstall torchvision to fix C++ extensions, upgrade transformers for Gemma 3\n")
    f.write("RUN pip install --no-cache-dir --force-reinstall torchvision \\\n")
    f.write("    && pip install --no-cache-dir --upgrade \\\n")
    f.write('    "transformers>=4.49.0" \\\n')
    f.write('    "accelerate>=1.2.0" \\\n')
    f.write('    "sentencepiece>=0.2.0"\n\n')
    
    # Install LTX-2 specific packages from git (ltx-core and ltx-pipelines are in subdirectories)
    f.write("# Install LTX-2 packages from Lightricks GitHub\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends git \\\n")
    f.write('    && pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core" \\\n')
    f.write('    && pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines" \\\n')
    f.write("    && pip install --no-cache-dir imageio-ffmpeg av \\\n")
    f.write("    && apt-get purge -y git \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    # Environment variables for LTX-2
    f.write('# LTX-2 environment variables\n')
    f.write('ENV LTX2_VIDEO_LOCAL_PATH="/app/models/ltx2_video"\n')
    f.write('ENV LTX2_GEMMA_ROOT="/app/models/gemma"\n')
    f.write('ENV LTX2_DEFAULT_PIPELINE="distilled"\n')
    f.write('ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    # Worker setup
    f.write("# Worker setup\n")
    f.write("RUN mkdir -p /app/worker/outputs && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n\n")


def main():
    """Generate the Dockerfile."""
    print(f"📝 Generating Dockerfile for LTX-2 Video Worker")
    print(f"   Base image: {BASE_IMAGE}")
    print(f"   Model: ltx-2-19b-dev-fp4.safetensors (20 GB)")
    print(f"   Works on: RTX 5090 (32GB)")
    print()
    
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files (configs, tokenizer, etc.)\n")
        copy_non_chunk_files(f)
        f.write("\n# Part files (800MB chunks, ~25 parts for 20GB model)\n")
        copy_chunks(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE}")
    print()
    print("To build the Docker image:")
    print(f"  cd {PROJECT_ROOT}")
    print("  docker build -t ltx2-video-worker .")


if __name__ == "__main__":
    main()
