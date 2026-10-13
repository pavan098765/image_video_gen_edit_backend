#!/usr/bin/env python3
"""
generate_dockerfile.py
======================

Generate Dockerfile for worker_comfyui_generic: ComfyUI + custom_nodes from prep,
optional models under comfyui/models, strict pip installs for custom node requirements.

Optional env:
  DOCKER_BASE_IMAGE  (default: pavanalur3/deps-base-new:py311-cu128)
"""

from __future__ import annotations

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_IMAGE = os.environ.get(
    "DOCKER_BASE_IMAGE",
    "pavanalur3/deps-base-new:py311-cu128",
)
COMFYUI_DIR = os.path.join(PROJECT_ROOT, "comfyui")
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")
OVERRIDES_FILE = os.path.join(PROJECT_ROOT, "overrides.json")
MODEL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "model_config.json")
REGISTRY_FILE = os.path.join(PROJECT_ROOT, "model_registry.json")


def write_header(f) -> None:
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("# Environment variables\n")
    f.write('ENV COMFYUI_DIR="/app/comfyui"\n')
    f.write('ENV MODELS_DIR="/app/comfyui/models"\n')
    f.write('ENV WORKFLOW_DIR="/app/worker/workflow"\n')
    f.write('ENV WORKSPACE_DIR="/workspace"\n')
    f.write('ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"\n\n')


def copy_comfyui(f) -> int:
    f.write("# ComfyUI application code\n")
    count = 0
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
    f.write("# Non-chunk model files (< 700MB)\n")
    count = 0
    if not os.path.isdir(MODELS_DIR):
        f.write("# (no models directory)\n\n")
        return 0
    for root, dirs, files in os.walk(MODELS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if ".part-" in fn or fn.startswith("."):
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
    f.write("# Chunk model parts (.part-*)\n")
    if not os.path.isdir(MODELS_DIR):
        f.write("\n")
        return 0
    parts: list[str] = []
    for root, dirs, files in os.walk(MODELS_DIR):
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


def write_install_deps(f) -> None:
    f.write("# System packages\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends \\\n")
    f.write("    git openssh-client build-essential ffmpeg \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/*\n\n")

    f.write("# ComfyUI Python deps\n")
    f.write("RUN pip install --no-cache-dir -r /app/comfyui/requirements.txt\n\n")

    f.write("# Custom node requirements (build fails if install fails)\n")
    custom_nodes_dir = os.path.join(COMFYUI_DIR, "custom_nodes")
    if os.path.isdir(custom_nodes_dir):
        for node_name in sorted(os.listdir(custom_nodes_dir)):
            node_dir = os.path.join(custom_nodes_dir, node_name)
            req_file = os.path.join(node_dir, "requirements.txt")
            if os.path.isfile(req_file):
                rel_req = f"/app/comfyui/custom_nodes/{node_name}/requirements.txt"
                f.write(
                    f"RUN pip install --no-cache-dir -r {rel_req}\n"
                )

    f.write("\n# Worker (FastAPI)\n")
    f.write("COPY requirements.txt /app/worker/requirements.txt\n")
    f.write("RUN pip install --no-cache-dir -r /app/worker/requirements.txt\n\n")


def copy_optional_overrides(f) -> None:
    if os.path.isfile(OVERRIDES_FILE):
        rel = os.path.relpath(OVERRIDES_FILE, PROJECT_ROOT).replace("\\", "/")
        f.write(f"COPY {rel} /app/worker/overrides.json\n\n")


def copy_optional_model_files(f) -> None:
    if os.path.isfile(MODEL_CONFIG_FILE):
        rel = os.path.relpath(MODEL_CONFIG_FILE, PROJECT_ROOT).replace("\\", "/")
        f.write(f"COPY {rel} /app/worker/model_config.json\n\n")
    if os.path.isfile(REGISTRY_FILE):
        rel = os.path.relpath(REGISTRY_FILE, PROJECT_ROOT).replace("\\", "/")
        f.write(f"COPY {rel} /app/worker/model_registry.json\n\n")


def write_worker_code(f) -> None:
    f.write("# Workflows and source (last layers)\n")
    f.write("COPY workflow/ /app/worker/workflow/\n\n")
    copy_optional_overrides(f)
    copy_optional_model_files(f)
    f.write("COPY src/ /app/worker/src/\n\n")
    f.write(
        "RUN mkdir -p /app/comfyui/output /app/comfyui/temp /app/comfyui/input /workspace\n\n"
    )
    f.write("COPY src/start.sh /app/start.sh\n")
    f.write("RUN chmod +x /app/start.sh\n\n")
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n\n")
    f.write('CMD ["/app/start.sh"]\n')


def main() -> None:
    if not os.path.isdir(COMFYUI_DIR):
        raise SystemExit(f"ComfyUI dir missing; run prep_model.py first: {COMFYUI_DIR}")

    print(f"Writing {OUT_DOCKERFILE}")
    with open(OUT_DOCKERFILE, "w", newline="\n") as f:
        write_header(f)
        n1 = copy_comfyui(f)
        n2 = copy_non_chunk_model_files(f)
        n3 = copy_chunks_one_per_line(f)
        write_install_deps(f)
        write_worker_code(f)

    print(f"  ComfyUI copy entries: {n1}, model files: {n2 + n3}")


if __name__ == "__main__":
    main()
