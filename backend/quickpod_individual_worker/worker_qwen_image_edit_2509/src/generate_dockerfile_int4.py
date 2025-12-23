#!/usr/bin/env python3
import os

# ------------------------
# Resolve project root (one level above src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_IMAGE = "pavanalur3/deps-base-extended:py311-cu129-int4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WORKER_DIR = os.path.join(PROJECT_ROOT, "src")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile_int4")


def write_header(f):
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("# --- Qwen 2509 + Nunchaku build (RTX 4090 Compatible) ---\n\n")


def copy_non_chunk_files(f):
    """
    Copy everything under models/ except:
      - .part-* files (handled separately)
      - safetensors > 500MB (should be split)
      - hidden dirs/files
    """
    for root, dirs, files in os.walk(MODELS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

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

            dest_dir = (
                f"/app/models/{rel_to_models}"
                if rel_to_models != "."
                else "/app/models"
            )
            f.write(f"COPY {rel_src} {dest_dir}/\n")


def copy_chunks_one_per_line(f):
    """
    Each .part-* gets its own COPY layer.
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

        dest_dir = (
            f"/app/models/{rel_to_models}"
            if rel_to_models != "."
            else "/app/models"
        )
        f.write(f"COPY {rel_src} {dest_dir}/\n")


def write_rest(f):
    f.write("\n# Worker code (last to leverage cache)\n")
    f.write("COPY src /app/worker/src\n\n")

    # Diffusers (kept as-is)
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends git \\\n")
    f.write("    && pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git \\\n")
    f.write("    && apt-get purge -y git \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n\n")

    # --------------------
    # Install runtime Python deps INTO VENV (CRITICAL)
    # --------------------
    f.write("RUN /opt/venv/bin/pip install --no-cache-dir \\\n")
    f.write("    tenacity \\\n")
    f.write("    realesrgan \\\n")
    f.write("    simple-lama-inpainting \\\n")
    f.write("    aura-sr \\\n")
    f.write("    gfpgan==1.3.8 \\\n")
    f.write("    gguf==0.17.1 \\\n")
    f.write("    opencv-python \\\n")
    f.write("    requests \\\n")
    f.write("    fastapi \\\n")
    f.write("    uvicorn \\\n")
    f.write("    uvloop \\\n")
    f.write("    && rm -rf /root/.cache /tmp/*\n\n")

    # Environment variables
    f.write('ENV QWEN_2509_LOCAL_PATH="/app/models/qwen_image_edit_2509"\n')
    f.write('ENV NUNCHAKU_QWEN_2509_LOCAL_PATH="/app/models/nunchaku_qwen_2509"\n')
    f.write('ENV REALESRGAN_LOCAL_PATH="/app/models/realesrgan"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')

    # Worker setup
    f.write("RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n\n")

    # IMPORTANT: do NOT rely on PATH ambiguity in QuickPod
    f.write('CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]\n')


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files (configs, tokenizer, ckpt, etc.)\n")
        copy_non_chunk_files(f)
        f.write("\n# Per-part COPYs (one COPY per chunk => one layer per chunk)\n")
        copy_chunks_one_per_line(f)
        write_rest(f)

    print(f"✅ Wrote {OUT_DOCKERFILE} (INT4, QuickPod-safe, venv-pinned).")


if __name__ == "__main__":
    main()
