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
    f.write("# --- SDXL Lighting 8-steps build ---\n\n")


def copy_non_chunk_files(f):
    """
    Copy everything under models/ except:
      - .part-* files (handled separately)
      - safetensors > 500MB (should be split)
      - .cache or hidden dirs
    """
    for root, dirs, files in os.walk(MODELS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip hidden dirs

        for fn in files:
            if ".part-" in fn:
                continue
            if fn.endswith(".safetensors"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 500 * 1024 * 1024:
                    continue

            abs_path = os.path.join(root, fn)
            rel_src = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
            f.write(f"COPY {rel_src} {dest_dir}/\n")


def copy_chunks_one_per_line(f):
    """
    Each .part-* gets its own COPY layer (helps with caching & parallel upload).
    """
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
    f.write("\n# Worker code (last to leverage cache)\n")
    f.write("COPY src /app/worker/src\n\n")

    # Install minimal dependencies and cleanup
    f.write("RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0\n")
    f.write(
        "RUN pip install --no-cache-dir "
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
        "flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl\n"
    )
    f.write("RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    # Environment variables
    f.write('ENV SDXL_LOCAL_PATH="/app/models/sdxl_lighting_8_steps"\n')
    f.write('ENV REALESRGAN_LOCAL_PATH="/app/models/realesrgan"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    # Worker setup
    f.write("RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n")


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files (.safetensors, configs, etc.)\n")
        copy_non_chunk_files(f)
        f.write("\n# Per-part COPYs (one COPY per chunk => one layer per chunk)\n")
        copy_chunks_one_per_line(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE} for SDXL Lighting 8-steps + Realesrgan setup.")


if __name__ == "__main__":
    main()
