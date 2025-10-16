#!/usr/bin/env python3
import os

# ------------------------
# Resolve project root (one level above src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_IMAGE = "pavanalur3/deps-base-extended:py311-cu129"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WORKER_DIR = os.path.join(PROJECT_ROOT, "src")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")


def write_header(f):
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("# --- SDXL Outpaint (controlnet / vae / pipeline) build ---\n\n")


def copy_grouped_files(f):
    """
    Groups files by their directory to minimize COPY lines.
    Skips .part-* chunks (handled separately) and very large safetensors.
    """
    grouped = {}

    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Collect eligible files for this directory
        rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
        dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
        rel_files = []

        for fn in files:
            if ".part-" in fn:
                continue

            full = os.path.join(root, fn)
            if fn.endswith(".safetensors"):
                try:
                    if os.path.getsize(full) > 500 * 1024 * 1024:
                        continue
                except OSError:
                    continue

            rel_src = os.path.relpath(full, PROJECT_ROOT).replace("\\", "/")
            rel_files.append(rel_src)

        if rel_files:
            grouped[dest_dir] = grouped.get(dest_dir, []) + rel_files

    # Write grouped COPY lines
    for dest_dir, rel_files in grouped.items():
        # Docker COPY command length can get too long, so chunk if >100 files
        chunk_size = 100
        for i in range(0, len(rel_files), chunk_size):
            batch = rel_files[i : i + chunk_size]
            joined = " \\\n    ".join(batch)
            f.write(f"COPY {joined} {dest_dir}/\n")


def copy_chunks_one_per_line(f):
    """Keep each .part-* file as a separate COPY line (to avoid cache busting large chunks)."""
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

    f.write("# Install Python dependencies\n")
    # Install specific versions (will override base image versions)
    f.write("RUN pip install --no-cache-dir --upgrade \\\n\
        diffusers==0.32.2 \\\n\
        transformers==4.45.0 \\\n\
        accelerate==0.33.0\n")
    f.write("RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    # Environment variables
    f.write('ENV CTRLNET_LOCAL_PATH="/app/models/sdxl_outpaint/controlnet"\n')
    f.write('ENV VAE_LOCAL_PATH="/app/models/sdxl_outpaint/vae"\n')
    f.write('ENV PIPELINE_LOCAL_PATH="/app/models/sdxl_outpaint/pipeline"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    f.write("RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n")


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Grouped COPY for non-chunk model files (configs, tokenizer, etc.)\n")
        copy_grouped_files(f)
        f.write("\n# One COPY per chunked part file (to optimize caching)\n")
        copy_chunks_one_per_line(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE} with optimized COPY grouping.")


if __name__ == "__main__":
    main()
