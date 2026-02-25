#!/usr/bin/env python3
"""
prep_model.py
=============

Prepares the Docker build context for the Wan 2.2 I2V ComfyUI worker:

1. Parses the workflow JSON to identify required models and custom nodes
2. Clones ComfyUI and required custom nodes
3. Downloads model files from HuggingFace to correct ComfyUI directories
4. Splits large safetensors for Docker layer optimization
5. Generates the Dockerfile via generate_dockerfile.py

Run this LOCALLY before building the Docker image:
    cd worker_wan_22_comfyui
    python src/prep_model.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMFYUI_DIR = os.path.join(PROJECT_ROOT, "comfyui")
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "workflow")

# HuggingFace token (set via env or hardcode for private repos)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# Each model entry:
#   (hf_repo, hf_filename, local_subdir, local_filename)
#
# local_subdir is relative to COMFYUI_DIR/models/
# If hf_filename is None, use snapshot_download for the whole repo

MODEL_FILES = [
    # 1. Main diffusion model (Wan 2.1 I2V 14B FP8)
    #    ~14.3 GB - largest file, will be split
    {
        "hf_repo": "Kijai/WanVideo_comfy",
        "hf_filename": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "local_dir": "diffusion_models/Wan",
        "local_filename": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "description": "Wan 2.1 I2V 14B FP8 diffusion model",
    },
    # 2. T5 text encoder (FP8 quantized)
    #    ~4.9 GB
    {
        "hf_repo": "Kijai/WanVideo_comfy",
        "hf_filename": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "local_dir": "text_encoders",
        "local_filename": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "description": "UMT5-XXL text encoder (FP8)",
    },
    # 3. CLIP Vision model
    #    ~3.9 GB
    {
        "hf_repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "hf_filename": "split_files/clip_vision/clip_vision_h.safetensors",
        "local_dir": "clip_vision",
        "local_filename": "clip_vision_h.safetensors",
        "description": "CLIP Vision H model",
    },
    # 4. VAE (BF16)
    #    ~335 MB
    {
        "hf_repo": "Kijai/WanVideo_comfy",
        "hf_filename": "Wan2_1_VAE_bf16.safetensors",
        "local_dir": "vae",
        "local_filename": "Wan2_1_VAE_bf16.safetensors",
        "description": "Wan 2.1 VAE (BF16)",
    },
    # 5. CausVid LoRA (LightX2V distillation)
    #    ~1.2 GB
    #    NOTE: local_filename must match the workflow reference:
    #    "WAN/CausVid/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
    {
        "hf_repo": "Kijai/WanVideo_comfy",
        "hf_filename": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "local_dir": "loras/WAN/CausVid",
        "local_filename": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "description": "CausVid LoRA (LightX2V distillation, rank64 BF16)",
    },
    # 6. SVI Film LoRA (Stable Video Infinity)
    #    ~1.2 GB
    #    NOTE: workflow references "svi-film-opt-10212025.safetensors" but the
    #    actual HF file is "svi-film-opt-10212025_lora_rank_128_fp16.safetensors".
    #    The workflow_converter._normalize_paths() corrects this at runtime.
    {
        "hf_repo": "Kijai/WanVideo_comfy",
        "hf_filename": "LoRAs/Stable-Video-Infinity/svi-film-opt-10212025_lora_rank_128_fp16.safetensors",
        "local_dir": "loras/WAN/Stable Video Infinity",
        "local_filename": "svi-film-opt-10212025_lora_rank_128_fp16.safetensors",
        "description": "SVI Film LoRA (Stable Video Infinity)",
    },
]

# RIFE model weights (for frame interpolation post-processing)
RIFE_MODEL_DIR = os.path.join(MODELS_DIR, "..", "rife_weights")  # Store outside ComfyUI models dir

# ---------------------------------------------------------------------------
# Custom node definitions
# ---------------------------------------------------------------------------

CUSTOM_NODES = [
    {
        "name": "ComfyUI-WanVideoWrapper",
        "url": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        "description": "WanVideo nodes (model loader, sampler, text encode, etc.)",
    },
    {
        "name": "ComfyUI-KJNodes",
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "description": "SetNode, GetNode, INTConstant, SimpleMath, ImageResizeKJv2",
    },
    {
        "name": "ComfyUI-VideoHelperSuite",
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "description": "VHS_VideoCombine (video output), VRAM_Debug",
    },
    {
        "name": "ComfyUI_essentials",
        "url": "https://github.com/cubiq/ComfyUI_essentials.git",
        "description": "GetImageSizeAndCount, GetImageRangeFromBatch",
    },
]


# ============================================================================
# 1. Clone ComfyUI
# ============================================================================

def clone_comfyui():
    """Clone ComfyUI repo if not present."""
    if os.path.isdir(COMFYUI_DIR) and os.path.isfile(os.path.join(COMFYUI_DIR, "main.py")):
        print(f"ComfyUI already cloned at {COMFYUI_DIR}")
        # Pull latest changes
        try:
            subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=COMFYUI_DIR,
                check=True,
                capture_output=True,
            )
            print("  Updated to latest commit")
        except subprocess.CalledProcessError:
            print("  Warning: git pull failed, using existing version")
        return

    print(f"Cloning ComfyUI to {COMFYUI_DIR}...")
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFYUI_DIR,
        ],
        check=True,
    )
    print("  ComfyUI cloned successfully")


# ============================================================================
# 2. Install custom nodes
# ============================================================================

def install_custom_nodes():
    """Clone required custom node repositories."""
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)

    for node in CUSTOM_NODES:
        node_dir = os.path.join(CUSTOM_NODES_DIR, node["name"])

        if os.path.isdir(node_dir):
            print(f"Custom node already exists: {node['name']}")
            # Pull latest
            try:
                subprocess.run(
                    ["git", "pull", "--ff-only"],
                    cwd=node_dir,
                    check=True,
                    capture_output=True,
                )
                print(f"  Updated {node['name']}")
            except subprocess.CalledProcessError:
                print(f"  Warning: git pull failed for {node['name']}")
            continue

        print(f"Cloning {node['name']}...")
        print(f"  {node['description']}")
        subprocess.run(
            ["git", "clone", node["url"], node_dir],
            check=True,
        )
        print(f"  Installed {node['name']}")


# ============================================================================
# 3. Download model files
# ============================================================================

def download_models():
    """Download all required model files from HuggingFace."""
    for model in MODEL_FILES:
        dest_dir = os.path.join(MODELS_DIR, model["local_dir"])
        dest_path = os.path.join(dest_dir, model["local_filename"])

        # Check if already downloaded (or split parts exist)
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"Model exists: {model['local_filename']} ({size_mb:.1f} MB)")
            continue

        # Check if split parts already exist
        parts = glob.glob(dest_path + ".part-*")
        if parts:
            print(f"Split parts exist for {model['local_filename']} ({len(parts)} parts)")
            continue

        os.makedirs(dest_dir, exist_ok=True)

        print(f"Downloading {model['description']}...")
        print(f"  From: {model['hf_repo']}/{model['hf_filename']}")
        print(f"  To:   {dest_path}")

        try:
            downloaded_path = hf_hub_download(
                repo_id=model["hf_repo"],
                filename=model["hf_filename"],
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=HF_TOKEN,
            )

            # hf_hub_download may place files in a subdirectory matching the
            # hf_filename path.  Move to the expected location if needed.
            if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
                shutil.move(downloaded_path, dest_path)
                # Clean up any empty subdirectories left behind
                dl_parent = os.path.dirname(downloaded_path)
                while dl_parent != dest_dir:
                    try:
                        os.rmdir(dl_parent)
                    except OSError:
                        break
                    dl_parent = os.path.dirname(dl_parent)

            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  Downloaded: {size_mb:.1f} MB")

        except Exception as e:
            print(f"  ERROR downloading {model['local_filename']}: {e}")
            traceback.print_exc()
            sys.exit(1)


# ============================================================================
# 3b. Download RIFE model weights
# ============================================================================

def download_rife_weights():
    """Download RIFE model weights from HuggingFace (AlexWortega/RIFE)."""
    rife_dir = os.path.normpath(RIFE_MODEL_DIR)
    flownet_path = os.path.join(rife_dir, "flownet.pkl")

    if os.path.exists(flownet_path):
        size_mb = os.path.getsize(flownet_path) / (1024 * 1024)
        print(f"RIFE model already exists: {flownet_path} ({size_mb:.1f} MB)")
        return

    os.makedirs(rife_dir, exist_ok=True)
    print(f"Downloading RIFE model weights to {rife_dir}...")
    try:
        snapshot_download(
            "AlexWortega/RIFE",
            local_dir=rife_dir,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        print(f"  RIFE model weights downloaded successfully")
    except Exception as e:
        print(f"  WARNING: Failed to download RIFE model weights: {e}")
        print(f"  RIFE frame interpolation will not be available")


# ============================================================================
# 4. Split large safetensors
# ============================================================================

def split_large_safetensors(base_dir: str, max_size: int = 700 * 1024 * 1024) -> int:
    """
    Split safetensors files larger than max_size into chunks.

    Parameters
    ----------
    base_dir : str
        Directory to scan recursively.
    max_size : int
        Maximum file size before splitting (default: 700 MB).

    Returns
    -------
    int
        Number of split parts created.
    """
    chunk_size_mb = max_size / (1024 * 1024)
    print(f"\nScanning for .safetensors > {chunk_size_mb:.0f} MB in {base_dir}")
    split_count = 0

    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    num_parts = (size + max_size - 1) // max_size
                    print(f"  Splitting {fn} ({size / (1024**3):.2f} GB) -> {num_parts} parts")
                    subprocess.run(
                        [
                            "split", "-b", f"{int(chunk_size_mb)}M",
                            full_path, f"{full_path}.part-",
                        ],
                        check=True,
                    )
                    os.remove(full_path)
                    parts = glob.glob(f"{full_path}.part-*")
                    split_count += len(parts)
                    print(f"    Split into {len(parts)} parts, original removed")
                else:
                    print(f"  Kept as-is: {fn} ({size / (1024**2):.1f} MB)")

    return split_count


def verify_split_files(base_dir: str) -> bool:
    """Verify that all split parts exist and are valid."""
    issues = []

    for root, _, files in os.walk(base_dir):
        part_files = [f for f in files if ".safetensors.part-" in f]
        if not part_files:
            continue

        bases = {}
        for pf in part_files:
            base = pf.split(".part-")[0]
            if base not in bases:
                bases[base] = []
            bases[base].append(os.path.join(root, pf))

        for base, parts in bases.items():
            parts_sorted = sorted(parts)
            total_size = sum(os.path.getsize(p) for p in parts_sorted)
            base_path = os.path.join(root, base)

            if os.path.exists(base_path):
                issues.append(f"Original file still exists: {base_path}")

            print(f"  Verified {len(parts_sorted)} parts for {base} "
                  f"(total: {total_size / (1024**3):.2f} GB)")

    if issues:
        print(f"\n  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print("  All split files verified")
    return True


# ============================================================================
# 5. Install pip dependencies (for local testing)
# ============================================================================

def install_all_pip_deps():
    """Install all pip dependencies for local testing (no Docker)."""

    # ComfyUI requirements
    comfyui_reqs = os.path.join(COMFYUI_DIR, "requirements.txt")
    if os.path.isfile(comfyui_reqs):
        print("Installing ComfyUI requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", comfyui_reqs],
            check=False,
        )

    # Custom node requirements
    for node in CUSTOM_NODES:
        node_dir = os.path.join(CUSTOM_NODES_DIR, node["name"])
        reqs = os.path.join(node_dir, "requirements.txt")
        if os.path.isfile(reqs):
            print(f"Installing requirements for {node['name']}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", reqs],
                check=False,
            )

    # Worker requirements
    worker_reqs = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.isfile(worker_reqs):
        print("Installing worker requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", worker_reqs],
            check=False,
        )

    print("All pip dependencies installed")


# ============================================================================
# 6. Create requirements.txt
# ============================================================================

def create_requirements():
    """Create requirements.txt for the worker."""
    reqs_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    requirements = [
        "fastapi==0.115.0",
        "uvicorn[standard]==0.30.6",
        "httpx==0.27.0",
        "requests==2.32.3",
        "tenacity==8.5.0",
        "pydantic==2.9.1",
        "Pillow>=10.0.0",
        # Video post-processing (RIFE frame interpolation)
        "opencv-python>=4.8.0",
        "spandrel>=0.4.0",
        "imageio-ffmpeg>=0.5.1",
        "safetensors>=0.4.0",
    ]

    with open(reqs_path, "w") as f:
        f.write("\n".join(requirements) + "\n")

    print(f"Created requirements.txt with {len(requirements)} packages")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Wan 2.2 I2V ComfyUI worker")
    parser.add_argument(
        "--no-split", action="store_true",
        help="Skip splitting large safetensors (for local testing, not Docker)",
    )
    parser.add_argument(
        "--no-dockerfile", action="store_true",
        help="Skip Dockerfile generation (for local testing)",
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Install pip dependencies for ComfyUI, custom nodes, and worker (for local testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Wan 2.2 I2V ComfyUI Worker - Model Preparation")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ComfyUI dir:  {COMFYUI_DIR}")
    print(f"Models dir:   {MODELS_DIR}")
    if args.no_split:
        print("  ** Split disabled (local testing mode) **")
    print()

    # Step 1: Clone ComfyUI
    print("\n--- Step 1: Clone ComfyUI ---")
    clone_comfyui()

    # Step 2: Install custom nodes
    print("\n--- Step 2: Install Custom Nodes ---")
    install_custom_nodes()

    # Step 3: Download models
    print("\n--- Step 3: Download Model Files ---")
    download_models()

    # Step 3b: Download RIFE weights
    print("\n--- Step 3b: Download RIFE Model Weights ---")
    download_rife_weights()

    # Step 4: Create requirements.txt
    print("\n--- Step 4: Create Requirements ---")
    create_requirements()

    # Step 5: Install pip deps (for local testing)
    if args.install_deps:
        print("\n--- Step 5: Install Dependencies ---")
        install_all_pip_deps()
    else:
        print("\n--- Step 5: Skipping pip install (use --install-deps to enable) ---")

    if not args.no_split:
        # Step 6: Brief pause for filesystem sync
        print("\nWaiting for filesystem sync...")
        time.sleep(5)

        # Step 7: Split large safetensors
        print("\n--- Step 6: Split Large Safetensors ---")
        total_splits = split_large_safetensors(MODELS_DIR, max_size=700 * 1024 * 1024)
        print(f"Total split parts created: {total_splits}")

        # Step 8: Verify splits
        print("\n--- Step 7: Verify Split Files ---")
        verify_split_files(MODELS_DIR)
    else:
        print("\n--- Step 6/7: Skipping split (--no-split) ---")

    if not args.no_dockerfile:
        # Step 9: Generate Dockerfile
        print("\n--- Step 8: Generate Dockerfile ---")
        try:
            subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
                check=True,
                cwd=SCRIPT_DIR,
            )
            print("Dockerfile generated successfully")
        except Exception as e:
            print(f"ERROR generating Dockerfile: {e}")
            traceback.print_exc()
    else:
        print("\n--- Step 8: Skipping Dockerfile generation (--no-dockerfile) ---")

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print(f"  ComfyUI:      {COMFYUI_DIR}")
    print(f"  Models:        {MODELS_DIR}")
    print(f"  Custom nodes:  {CUSTOM_NODES_DIR}")
    print(f"  Workflow:      {WORKFLOW_DIR}")
    print()

    if args.no_split:
        print("Local testing mode - next steps:")
        print("  1. Run: bash src/test_local.sh")
        print("     OR manually:")
        print(f"     export COMFYUI_DIR={COMFYUI_DIR}")
        print(f"     export WORKFLOW_DIR={WORKFLOW_DIR}")
        print(f"     export PYTHONPATH={os.path.join(PROJECT_ROOT, 'src')}:$PYTHONPATH")
        print(f"     cd {PROJECT_ROOT}")
        print("     uvicorn src.app:app --host 0.0.0.0 --port 8000")
    else:
        print("Docker mode - next steps:")
        print("  1. Review the generated Dockerfile")
        print("  2. Build: docker build -t wan22-comfyui-worker .")
        print("  3. Run:   docker run --gpus all -p 8000:8000 wan22-comfyui-worker")
    print("=" * 60)


if __name__ == "__main__":
    main()
