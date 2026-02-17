#!/usr/bin/env python3
import os

# ------------------------
# Resolve project root (one level above src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_IMAGE = "pavanalur3/deps-base-new:py311-cu128"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WORKER_DIR = os.path.join(PROJECT_ROOT, "src")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")


def write_header(f):
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")


def copy_non_chunk_files(f):
    """
    Copy everything under models/ except:
      - .part-* files (handled separately)
      - safetensors > 700MB (should be split)
      - .cache or hidden dirs
    """
    count = 0
    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs like .cache, .git, etc.
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fn in files:
            if ".part-" in fn:
                continue
            if fn.endswith(".safetensors"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 700 * 1024 * 1024:
                    continue

            abs_path = os.path.join(root, fn)
            rel_src = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
            f.write(f"COPY {rel_src} {dest_dir}/\n")
            count += 1
    return count


def copy_chunks_one_per_line(f):
    """
    Each .part-* gets its own COPY layer (helps with caching and parallel upload).
    Returns count of chunk layers.
    """
    parts = []
    for root, _, files in os.walk(MODELS_DIR):
        # Skip hidden dirs
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue
        for fn in files:
            if ".part-" in fn:
                parts.append(os.path.join(root, fn))
    parts.sort()

    for p in parts:
        rel_src = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
        rel_to_models = os.path.relpath(os.path.dirname(p), MODELS_DIR).replace("\\", "/")
        dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
        f.write(f"COPY {rel_src} {dest_dir}/\n")
    
    return len(parts)


def write_rest(f):
    f.write("\n# Copy requirements.txt first for dependency installation\n")
    f.write("COPY requirements.txt /app/requirements.txt\n\n")
    
    # Install system dependencies and Python packages
    f.write("# Install system dependencies and Python packages\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client build-essential \\\n")
    f.write("    && pip install --no-cache-dir -r /app/requirements.txt \\\n")
    f.write("    && pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    f.write("# Worker code (last to leverage cache)\n")
    f.write("COPY src /app/worker/src\n\n")

    # Environment variables for models and runtime
    f.write('ENV MODEL_LOCAL_DIR="/app/models/fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32"\n')
    f.write('ENV MODELS_DIR="/app/models"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    # Worker setup
    f.write("RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n")


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files (configs, tokenizer, ckpt, etc.)\n")
        non_chunk_count = copy_non_chunk_files(f)
        f.write("\n# Per-part COPYs (one COPY per chunk => one layer per chunk)\n")
        chunk_count = copy_chunks_one_per_line(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE}")
    print(f"   Non-chunk files: {non_chunk_count}")
    print(f"   Chunk files: {chunk_count}")
    print(f"   Total layers: {non_chunk_count + chunk_count + 5} (estimated)")


if __name__ == "__main__":
    main()
