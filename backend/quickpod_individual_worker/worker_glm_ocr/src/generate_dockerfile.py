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


def copy_non_chunk_files(f):
    """
    Copy everything under models/ except:
      - .part-* files (handled separately)
      - safetensors > 600MB (should be split)
      - .cache or hidden dirs
    """
    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip hidden dirs like .cache, .git, etc.
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fn in files:
            if ".part-" in fn:
                continue
            if fn.endswith(".safetensors"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 600 * 1024 * 1024:
                    continue

            abs_path = os.path.join(root, fn)
            rel_src = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
            f.write(f"COPY {rel_src} {dest_dir}/\n")


def copy_chunks_one_per_line(f):
    """
    Each .part-* gets its own COPY layer (helps with caching and parallel upload).
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

    # Install system dependencies and Python packages
    f.write("# Install system dependencies and Python packages\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client build-essential \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    # Install transformers from git (>=5.1.0 required for GLM-OCR)
    f.write("# Install transformers>=5.1.0 from git (required for GLM-OCR)\n")
    f.write("RUN pip install --no-cache-dir 'git+https://github.com/huggingface/transformers.git'\n\n")

    # Export model environment paths
    f.write('ENV GLM_OCR_LOCAL_PATH="/app/models/glm_ocr"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    # Worker setup
    f.write("RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n")


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files (configs, tokenizer, ckpt, etc.)\n")
        copy_non_chunk_files(f)
        f.write("\n# Per-part COPYs (one COPY per chunk => one layer per chunk)\n")
        copy_chunks_one_per_line(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE} for GLM-OCR setup.")


if __name__ == "__main__":
    main()
