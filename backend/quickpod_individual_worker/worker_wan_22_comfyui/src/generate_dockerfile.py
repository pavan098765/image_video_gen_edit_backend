#!/usr/bin/env python3
"""
generate_dockerfile.py
======================

Generates the Dockerfile for the Wan 2.2 I2V ComfyUI worker.

The Dockerfile is structured to maximize Docker layer caching:
1. Base image with CUDA + Python
2. ComfyUI code + custom nodes (rarely changes)
3. Non-chunk model files (configs, small weights)
4. Per-chunk safetensors COPY layers (each ~700MB, cached independently)
5. pip install requirements (ComfyUI + custom nodes + FastAPI deps)
6. Worker source code (changes most frequently, last for cache)

Run after prep_model.py has downloaded everything.
"""

from __future__ import annotations

import os
import stat

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_IMAGE = "pavanalur3/deps-base-new:py311-cu128"
COMFYUI_DIR = os.path.join(PROJECT_ROOT, "comfyui")
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")
WORKER_DIR = os.path.join(PROJECT_ROOT, "src")
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "workflow")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")


def write_header(f):
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("# Environment variables\n")
    f.write('ENV COMFYUI_DIR="/app/comfyui"\n')
    f.write('ENV MODELS_DIR="/app/comfyui/models"\n')
    f.write('ENV WORKFLOW_DIR="/app/worker/workflow"\n')
    f.write('ENV WORKSPACE_DIR="/workspace"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')


def copy_comfyui(f) -> int:
    """
    Copy ComfyUI code (excluding models/ and __pycache__).

    Returns the number of COPY instructions written.
    """
    f.write("# ComfyUI application code\n")

    # Copy main ComfyUI files (exclude models/, output/, temp/, input/)
    # We copy in a way that leverages layer caching
    count = 0

    # Copy top-level files
    for fn in sorted(os.listdir(COMFYUI_DIR)):
        full = os.path.join(COMFYUI_DIR, fn)
        rel = os.path.relpath(full, PROJECT_ROOT).replace("\\", "/")

        if fn.startswith("."):
            continue
        if fn in ("models", "output", "temp", "input", "__pycache__", ".git"):
            continue

        if os.path.isfile(full):
            f.write(f"COPY {rel} /app/comfyui/\n")
            count += 1
        elif os.path.isdir(full):
            f.write(f"COPY {rel}/ /app/comfyui/{fn}/\n")
            count += 1

    f.write("\n")
    return count


def copy_non_chunk_model_files(f) -> int:
    """
    Copy model files < 700MB (configs, tokenizer, small weights).
    Excludes .part-* files (handled separately).
    """
    f.write("# Non-chunk model files (small configs, weights < 700MB)\n")
    count = 0

    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fn in sorted(files):
            if ".part-" in fn:
                continue
            if fn.startswith("."):
                continue
            if fn.endswith(".safetensors"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 700 * 1024 * 1024:
                    continue

            abs_path = os.path.join(root, fn)
            rel_src = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = (
                f"/app/comfyui/models/{rel_to_models}"
                if rel_to_models != "."
                else "/app/comfyui/models"
            )
            f.write(f"COPY {rel_src} {dest_dir}/\n")
            count += 1

    f.write("\n")
    return count


def copy_chunks_one_per_line(f) -> int:
    """
    Each .part-* gets its own COPY layer for optimal caching.
    """
    f.write("# Per-part COPYs (one COPY per chunk => one layer per chunk)\n")
    parts = []

    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for fn in sorted(files):
            if ".part-" in fn:
                parts.append(os.path.join(root, fn))

    parts.sort()

    for p in parts:
        rel_src = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
        rel_to_models = os.path.relpath(os.path.dirname(p), MODELS_DIR).replace("\\", "/")
        dest_dir = (
            f"/app/comfyui/models/{rel_to_models}"
            if rel_to_models != "."
            else "/app/comfyui/models"
        )
        f.write(f"COPY {rel_src} {dest_dir}/\n")

    f.write("\n")
    return len(parts)


def write_install_deps(f):
    """Write dependency installation instructions."""
    f.write("# Install system dependencies\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends \\\n")
    f.write("    git openssh-client build-essential ffmpeg \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/*\n\n")

    f.write("# Install ComfyUI requirements\n")
    f.write("RUN pip install --no-cache-dir -r /app/comfyui/requirements.txt\n\n")

    f.write("# Install custom node requirements\n")
    # Each custom node may have a requirements.txt
    custom_nodes_dir = os.path.join(COMFYUI_DIR, "custom_nodes")
    if os.path.isdir(custom_nodes_dir):
        for node_name in sorted(os.listdir(custom_nodes_dir)):
            node_dir = os.path.join(custom_nodes_dir, node_name)
            req_file = os.path.join(node_dir, "requirements.txt")
            if os.path.isfile(req_file):
                rel_req = f"/app/comfyui/custom_nodes/{node_name}/requirements.txt"
                f.write(
                    f"RUN pip install --no-cache-dir -r {rel_req} "
                    f"|| echo 'Warning: some deps for {node_name} may have failed'\n"
                )

    f.write("\n# Install worker requirements\n")
    f.write("COPY requirements.txt /app/worker/requirements.txt\n")
    f.write("RUN pip install --no-cache-dir -r /app/worker/requirements.txt\n\n")

    # Install flash attention
    f.write("# Install flash-attention (for ComfyUI optimizations)\n")
    f.write("RUN pip install --no-cache-dir ")
    f.write("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl ")
    f.write("|| echo 'Flash attention install failed, will use fallback'\n\n")

    # Install sageattention for WanVideoWrapper
    f.write("# Install sageattention (used by WanVideoWrapper for efficient attention)\n")
    f.write("RUN pip install --no-cache-dir sageattention || echo 'sageattention install failed'\n\n")


def copy_rife_weights(f) -> int:
    """Copy RIFE model weights if they exist."""
    rife_dir = os.path.join(PROJECT_ROOT, "rife_weights")
    if not os.path.isdir(rife_dir):
        f.write("# RIFE weights not found (RIFE interpolation will be disabled)\n\n")
        return 0

    f.write("# RIFE model weights (frame interpolation)\n")
    rel_src = os.path.relpath(rife_dir, PROJECT_ROOT).replace("\\", "/")
    f.write(f"COPY {rel_src}/ /app/rife_weights/\n")
    f.write('ENV RIFE_WEIGHTS_DIR="/app/rife_weights"\n\n')
    return 1


def write_worker_code(f):
    """Write worker code copy and final setup."""
    f.write("# Copy workflow files\n")
    f.write("COPY workflow/ /app/worker/workflow/\n\n")

    f.write("# Worker source code (last for best cache)\n")
    f.write("COPY src/ /app/worker/src/\n\n")

    f.write("# Create required directories\n")
    f.write("RUN mkdir -p /app/comfyui/output /app/comfyui/temp /app/comfyui/input /workspace\n\n")

    f.write("# Copy and set permissions for startup script\n")
    f.write("COPY src/start.sh /app/start.sh\n")
    f.write("RUN chmod +x /app/start.sh\n\n")

    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n\n")

    f.write('CMD ["/app/start.sh"]\n')


def main():
    print(f"Generating Dockerfile at {OUT_DOCKERFILE}")
    print(f"  Base image: {BASE_IMAGE}")
    print(f"  ComfyUI: {COMFYUI_DIR}")
    print(f"  Models: {MODELS_DIR}")

    with open(OUT_DOCKERFILE, "w", newline="\n") as f:
        write_header(f)

        # ComfyUI code
        comfyui_count = copy_comfyui(f)

        # Non-chunk model files
        non_chunk_count = copy_non_chunk_model_files(f)

        # Chunk model files (one per layer)
        chunk_count = copy_chunks_one_per_line(f)

        # RIFE weights
        rife_count = copy_rife_weights(f)

        # Install dependencies
        write_install_deps(f)

        # Worker code and final setup
        write_worker_code(f)

    print(f"\nDockerfile written: {OUT_DOCKERFILE}")
    print(f"  ComfyUI copies:   {comfyui_count}")
    print(f"  Non-chunk files:   {non_chunk_count}")
    print(f"  Chunk files:       {chunk_count}")
    print(f"  Estimated layers:  {comfyui_count + non_chunk_count + chunk_count + 8}")


if __name__ == "__main__":
    main()
