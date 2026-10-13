#!/usr/bin/env python3
"""
prep_model.py
=============

1. Clone ComfyUI into ./comfyui
2. Run install_workflow_nodes.py for workflow/*.json (+ overrides.json if present)
3. Sync ``model_config.json`` from workflow paths (merge from ``model_registry.json`` when present)
4. Download weights: **model_config.json** first, then **model_registry.json** fallback
5. Generate Dockerfile via generate_dockerfile.py

``model_config.json`` is written/updated by prep with one entry per scanned model path.
Set ``hf_repo`` / ``hf_path`` there for exact HF locations; optional ``model_registry.json``
fills gaps for shared catalogs.

Env:
  SKIP_MODEL_DOWNLOAD=1         Skip HF downloads only (still refreshes model_config.json)
  SKIP_RUNTIME_MODEL_DOWNLOAD=1 Worker skips download on startup (debug)
  HF_TOKEN                      Optional token for gated repos
  MODEL_REGISTRY_PATH           Override registry path
  MODEL_CONFIG_PATH             Override model_config path
  PREP_FORCE_COMFYUI_RECLONE=1  Delete broken comfyui/ tree even if models/ has files
  NODE_INSTALL_SUMMARY_PATH     Optional path for custom node install report JSON
  SKIP_NODE_INSTALL_SUMMARY=1   Omit node_install_summary.json after install_workflow_nodes
  SKIP_UNLISTED_AUTOMATION=1    Only overrides.json for types missing from Manager map
  UNLISTED_NODE_HINTS_PATH      Optional JSON path for extra unlisted node -> git URL rules

Usage:
    cd worker_comfyui_generic
    python src/prep_model.py

Run from **project root** or from ``src/`` (paths are resolved from this file).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMFYUI_DIR = os.path.join(PROJECT_ROOT, "comfyui")
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "workflow")
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")

COMFYUI_REPO = os.environ.get(
    "COMFYUI_REPO",
    "https://github.com/comfyanonymous/ComfyUI.git",
)


def _comfy_has_models_payload(cui_dir: str) -> bool:
    """True if comfyui/models contains any file (user may have copied weights)."""
    models_root = os.path.join(cui_dir, "models")
    if not os.path.isdir(models_root):
        return False
    for _root, _dirs, files in os.walk(models_root):
        if files:
            return True
    return False


def clone_comfyui() -> None:
    main_py = os.path.join(COMFYUI_DIR, "main.py")
    if os.path.isfile(main_py):
        print(f"ComfyUI already present: {COMFYUI_DIR}")
        return

    if os.path.isdir(COMFYUI_DIR):
        if _comfy_has_models_payload(COMFYUI_DIR):
            if os.environ.get("PREP_FORCE_COMFYUI_RECLONE", "").strip() != "1":
                raise SystemExit(
                    f"{COMFYUI_DIR} exists without main.py but has files under models/.\n"
                    "Move models out, then: rm -rf {0}\n"
                    "Or re-run with PREP_FORCE_COMFYUI_RECLONE=1 (deletes entire {0}).".format(
                        COMFYUI_DIR
                    )
                )
        print(
            f"WARNING: Removing incomplete {COMFYUI_DIR} for a fresh ComfyUI clone.",
            file=sys.stderr,
        )
        shutil.rmtree(COMFYUI_DIR)

    parent = os.path.dirname(COMFYUI_DIR)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"Cloning ComfyUI into {COMFYUI_DIR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", COMFYUI_REPO, COMFYUI_DIR],
        check=True,
    )


def ensure_layout() -> None:
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(COMFYUI_DIR, "output"), exist_ok=True)
    os.makedirs(os.path.join(COMFYUI_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(COMFYUI_DIR, "temp"), exist_ok=True)


def run_install_workflow_nodes() -> None:
    env = os.environ.copy()
    env["COMFYUI_DIR"] = COMFYUI_DIR
    overrides = os.path.join(PROJECT_ROOT, "overrides.json")
    if os.path.isfile(overrides):
        env["OVERRIDES_PATH"] = overrides

    install_script = os.path.join(SCRIPT_DIR, "install_workflow_nodes.py")
    cmd = [
        sys.executable,
        install_script,
        "--workflow-dir",
        WORKFLOW_DIR,
        "--comfyui-dir",
        COMFYUI_DIR,
    ]
    if os.path.isfile(overrides):
        cmd.extend(["--overrides", overrides])

    sp = os.environ.get("NODE_INSTALL_SUMMARY_PATH", "").strip()
    if sp:
        cmd.extend(["--summary-out", sp])
    if os.environ.get("SKIP_NODE_INSTALL_SUMMARY", "").strip() == "1":
        cmd.append("--no-install-summary")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_generate_dockerfile() -> None:
    subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_sync_model_config() -> None:
    from download_workflow_models import sync_model_config_from_workflows

    sync_model_config_from_workflows(
        workflow_dir=Path(WORKFLOW_DIR),
        project_root=Path(PROJECT_ROOT),
    )


def run_download_models() -> None:
    from download_workflow_models import run_download

    if os.environ.get("SKIP_MODEL_DOWNLOAD", "").strip() == "1":
        return

    _, missing = run_download(
        workflow_dir=Path(WORKFLOW_DIR),
        models_dir=Path(MODELS_DIR),
    )
    if missing:
        stub = os.path.join(PROJECT_ROOT, "model_registry.stub.json")
        print(
            "\nIncomplete HF mappings (model_config.json preferred, registry fallback). Missing:\n  "
            + "\n  ".join(missing[:40])
            + ("\n  ..." if len(missing) > 40 else ""),
            file=sys.stderr,
        )
        print(
            f"\nHint: edit model_config.json, or python src/download_workflow_models.py --write-stub {stub}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    ensure_layout()
    clone_comfyui()

    workflows = [
        f for f in os.listdir(WORKFLOW_DIR)
        if f.endswith(".json") and os.path.isfile(os.path.join(WORKFLOW_DIR, f))
    ]
    if workflows:
        run_install_workflow_nodes()
        run_sync_model_config()
        run_download_models()
    else:
        print("No workflow JSON in workflow/ — skipping custom node install.")

    run_generate_dockerfile()
    print("\nprep_model.py finished. Build: docker build -t comfyui-generic-worker .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
