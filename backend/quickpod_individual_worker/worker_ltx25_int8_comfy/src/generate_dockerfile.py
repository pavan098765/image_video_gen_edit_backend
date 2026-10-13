#!/usr/bin/env python3
"""Emit Dockerfile that clones pinned ComfyUI + custom nodes and copies ltx25_smart_controls."""

from __future__ import annotations

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCK_PATH = PROJECT_ROOT / "nodes.lock.json"
OUT_DOCKERFILE = PROJECT_ROOT / "Dockerfile"
BASE_IMAGE = os.environ.get("DOCKER_BASE_IMAGE", "pavanalur3/deps-base-new:py311-cu128")


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    comfy = lock["comfyui"]
    nodes = lock["custom_nodes"]
    local_nodes = lock.get("local_nodes") or []
    lines = [
        f"FROM {BASE_IMAGE}",
        "",
        "WORKDIR /app",
        "",
        'ENV COMFYUI_DIR="/app/comfyui"',
        'ENV MODELS_DIR="/app/comfyui/models"',
        'ENV WORKFLOW_DIR="/app/worker/workflow"',
        'ENV WORKSPACE_DIR="/workspace"',
        'ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"',
        "ENV GIT_TERMINAL_PROMPT=0",
        "",
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*",
        "",
        "RUN mkdir -p /app/comfyui /app/worker /workspace",
        "",
        f"RUN git clone --depth 1 --branch {comfy['ref']} {comfy['repo']} /app/comfyui "
        "&& pip install --no-cache-dir -r /app/comfyui/requirements.txt",
        "",
        "RUN mkdir -p /app/comfyui/custom_nodes /app/comfyui/models /app/comfyui/input "
        "/app/comfyui/output /app/comfyui/temp /app/comfyui/user/default/workflows",
        "",
        "# Stage 1 SageAttention + Sol-Attn (Speed path). Quality Stage 2 uses native attention.",
        "RUN pip install --no-cache-dir triton "
        "&& (pip install --no-cache-dir sageattention "
        "|| (git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention "
        "&& pip install --no-cache-dir /tmp/SageAttention && rm -rf /tmp/SageAttention)) "
        "|| echo 'WARNING: SageAttention install failed; Stage 2 Quality may still run'",
        "",
    ]
    for node in nodes:
        name = node["name"]
        repo = node["repo"]
        commit = node["commit"]
        dest = f"/app/comfyui/custom_nodes/{name}"
        lines.append(
            f"RUN git clone {repo} {dest} "
            f"&& git -C {dest} checkout {commit} "
            f"&& git -C {dest} submodule update --init --recursive "
            f"&& if [ -f {dest}/requirements.txt ]; then pip install --no-cache-dir -r {dest}/requirements.txt; fi"
        )
        lines.append("")
    for node in local_nodes:
        name = node["name"]
        src = node["src"]
        dest = f"/app/comfyui/custom_nodes/{name}"
        lines.append(f"COPY {src} {dest}")
        lines.append(
            f"RUN if [ -f {dest}/requirements.txt ]; then pip install --no-cache-dir -r {dest}/requirements.txt; fi"
        )
        lines.append("")
    lines.extend(
        [
            "COPY requirements.txt /app/worker/requirements.txt",
            "RUN pip install --no-cache-dir -r /app/worker/requirements.txt",
            "",
            "COPY nodes.lock.json /app/worker/nodes.lock.json",
            "COPY models_manifest.json /app/worker/models_manifest.json",
            "COPY workflow/ /app/worker/workflow/",
            "COPY src/ /app/worker/src/",
            "COPY static/ /app/worker/static/",
            "",
            "RUN mkdir -p /app/comfyui/output /app/comfyui/temp /app/comfyui/input /workspace "
            "&& cp /app/worker/workflow/*.json /app/comfyui/user/default/workflows/ "
            "&& chmod +x /app/worker/src/start.sh",
            "",
            "WORKDIR /app/worker",
            "EXPOSE 8000 8188",
            'CMD ["/app/worker/src/start.sh"]',
            "",
        ]
    )
    OUT_DOCKERFILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DOCKERFILE}")


if __name__ == "__main__":
    main()
