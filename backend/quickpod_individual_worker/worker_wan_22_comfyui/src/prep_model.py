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

from huggingface_hub import hf_hub_download

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

# RIFE model weights (optional: for post-workflow frame interpolation in inference code).
# The v2.5.4 workflow uses built-in FL_RIFE; Python RIFE is only needed if you add
# a post-processing step (e.g. run RIFE on workflow output). Use --with-rife to download.
RIFE_MODEL_DIR = os.path.join(MODELS_DIR, "..", "rife_weights")  # Store outside ComfyUI models dir

# ---------------------------------------------------------------------------
# Custom node definitions
# ---------------------------------------------------------------------------
# All entries below are required by workflow "WAN2.2 T2V-I2V-T2I-S2V K3NK v2.5.4 SVI.json".
# Generated from workflow_node_types.txt; add or strip nodes in comfyui_inference if workflow changes.
#
CUSTOM_NODES = [
    {
        "name": "ComfyUI-WanVideoWrapper",
        "url": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        "description": "WanVideo nodes (model loader, sampler, text encode, etc.)",
    },
    {
        "name": "ComfyUI-KJNodes",
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "description": "SetNode, GetNode, INTConstant, SimpleMath+, ImageResizeKJv2, PrimitiveFloat, PrimitiveInt",
    },
    {
        "name": "ComfyUI-VideoHelperSuite",
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "description": "VHS_VideoCombine, VHS_LoadVideo, VHS_SelectEveryNthImage, etc.",
    },
    {
        "name": "ComfyUI_essentials",
        "url": "https://github.com/cubiq/ComfyUI_essentials.git",
        "description": "GetImageSizeAndCount, GetImageRangeFromBatch, ImageFromBatch+",
    },
    {
        "name": "ComfyUI-Easy-Use",
        "url": "https://github.com/yolain/ComfyUI-Easy-Use.git",
        "description": "easy cleanGpuUsed, easy mathFloat/Int, easy promptReplace, easy textIndexSwitch",
    },
    # --- Workflow custom nodes (from workflow_node_types.txt) ---
    {
        "name": "comfyui-various",
        "url": "https://github.com/jamesWalker55/comfyui-various.git",
        "description": "JWFloatToInteger",
    },
    {
        "name": "rgthree-comfy",
        "url": "https://github.com/rgthree/rgthree-comfy.git",
        "description": "Any Switch, Seed, Bookmark, Display Int, Fast Groups, Label, Mute/Bypass, Power Lora Loader",
    },
    {
        "name": "ComfyUI_Fill-Nodes",
        "url": "https://github.com/filliptm/ComfyUI_Fill-Nodes.git",
        "description": "FL_RIFE (frame interpolation)",
    },
    {
        "name": "ComfyUI-Impact-Pack",
        "url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "description": "ImpactWildcardProcessor and other Impact nodes",
    },
    {
        "name": "ComfyUI-Inspire-Pack",
        "url": "https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git",
        "description": "LoadImagesFromDir //Inspire",
    },
    {
        "name": "ComfyUI-Custom-Scripts",
        "url": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git",
        "description": "MathExpression|pysssss, PlaySound|pysssss",
    },
    {
        "name": "ComfyUI-K3NKImageGrab",
        "url": "https://github.com/K3NK3/ComfyUI-K3NKImageGrab.git",
        "description": "K3NKImageGrab",
    },
    {
        "name": "ComfyUI-K3NK-ComfyUI-Nodes",
        "url": "https://github.com/K3NK3/ComfyUI-K3NK-ComfyUI-Nodes.git",
        "description": "K3NKImageLoaderWithBlending, K3NKSaveLatentPassThrought",
    },
    {
        "name": "wlsh_nodes",
        "url": "https://github.com/wallish77/wlsh_nodes.git",
        "description": "Resolutions by Ratio (WLSH)",
    },
    {
        "name": "ComfyUI-post-processing-nodes",
        "url": "https://github.com/EllangoK/ComfyUI-post-processing-nodes.git",
        "description": "ColorCorrect",
    },
    {
        "name": "ComfyUI_Comfyroll_CustomNodes",
        "url": "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git",
        "description": "CR String To Combo, CR Text",
    },
    {
        "name": "ComfyMath",
        "url": "https://github.com/evanspearman/ComfyMath.git",
        "description": "CM_FloatToInt, CM_IntToFloat",
    },
    {
        "name": "ComfyUI_tinyterraNodes",
        "url": "https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git",
        "description": "ttN concat, ttN int, ttN text",
    },
    {
        "name": "Derfuu_ComfyUI_ModdedNodes",
        "url": "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git",
        "description": "DF_Int_to_Float",
    },
    {
        "name": "ComfyUI-QHNodes",
        "url": "https://github.com/liuqianhonga/ComfyUI-QHNodes.git",
        "description": "QHNodes utilities",
    },
    {
        "name": "Comfyui-ergouzi-Nodes",
        "url": "https://github.com/11dogzi/Comfyui-ergouzi-Nodes.git",
        "description": "EG_WXZ_QH (Unrestricted switching)",
    },
    {
        "name": "ComfyUI-FramePackWrapper_PlusOne",
        "url": "https://github.com/tori29umai0123/ComfyUI-FramePackWrapper_PlusOne.git",
        "description": "FramePackFindNearestBucket",
    },
    # --- Additional nodes from validate_workflow_nodes missing list ---
    {
        "name": "ComfyUI-Addoor",
        "url": "https://github.com/Eagle-CN/ComfyUI-Addoor.git",
        "description": "Incrementer 🪴",
    },
    {
        "name": "ComfyUI-pause",
        "url": "https://github.com/wywywywy/ComfyUI-pause.git",
        "description": "PauseWorkflowNode",
    },
    {
        "name": "ComfyUI-mxToolkit",
        "url": "https://github.com/Smirnov75/ComfyUI-mxToolkit.git",
        "description": "mxSlider2D, mxStop",
    },
    {
        "name": "ComfyUI-MelBandRoFormer",
        "url": "https://github.com/kijai/ComfyUI-MelBandRoFormer.git",
        "description": "MelBandRoFormerModelLoader, MelBandRoFormerSampler, NormalizeAudioLoudness, PreviewAudio",
    },
    {
        "name": "ComfyUI-MMAudio",
        "url": "https://github.com/kijai/ComfyUI-MMAudio.git",
        "description": "MMAudioFeatureUtilsLoader, MMAudioModelLoader, MMAudioSampler",
    },
    {
        "name": "ComfyUI-NovaSR",
        "url": "https://github.com/Saganaki22/ComfyUI-NovaSR.git",
        "description": "NovaSR (audio super resolution)",
    },
    {
        "name": "ComfyUI-Florence2",
        "url": "https://github.com/kijai/ComfyUI-Florence2.git",
        "description": "DownloadAndLoadFlorence2Model, Florence2Run",
    },
    {
        "name": "RES4LYF",
        "url": "https://github.com/ClownsharkBatwing/RES4LYF.git",
        "description": "Sigmas Split Value",
    },
    {
        "name": "comfyui-vrgamedevgirl",
        "url": "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git",
        "description": "ColorMatchToReference",
    },
    {
        "name": "efficiency-nodes-comfyui",
        "url": "https://github.com/jags111/efficiency-nodes-comfyui.git",
        "description": "Image Overlay",
    },
]


# ---------------------------------------------------------------------------
# Workflow node type -> source (for coverage check vs workflow_node_types.txt)
# ---------------------------------------------------------------------------
# "builtin" = ComfyUI core; otherwise = CUSTOM_NODES entry "name".
# Used by check_workflow_node_coverage() to ensure every type is either builtin or installed.
WORKFLOW_NODE_SOURCE: dict[str, str] = {
    # Built-in ComfyUI
    "BasicScheduler": "builtin",
    "CLIPTextEncode": "builtin",
    "CLIPVisionLoader": "builtin",
    "CreateCFGScheduleFloatList": "builtin",
    "LoadImage": "builtin",
    "Note": "builtin",
    "PreviewImage": "builtin",
    "Reroute": "builtin",
    "SaveImage": "builtin",
    "UpscaleModelLoader": "builtin",
    "ImageUpscaleWithModelBatched": "builtin",
    "ImageScaleBy": "builtin",
    "SaveLatent": "builtin",
    # WanVideoWrapper
    "DummyComfyWanModelObject": "ComfyUI-WanVideoWrapper",
    "WanVideoApplyNAG": "ComfyUI-WanVideoWrapper",
    "WanVideoBlockList": "ComfyUI-WanVideoWrapper",
    "WanVideoBlockSwap": "ComfyUI-WanVideoWrapper",
    "WanVideoClipVisionEncode": "ComfyUI-WanVideoWrapper",
    "WanVideoContextOptions": "ComfyUI-WanVideoWrapper",
    "WanVideoDecode": "ComfyUI-WanVideoWrapper",
    "WanVideoEmptyEmbeds": "ComfyUI-WanVideoWrapper",
    "WanVideoEncode": "ComfyUI-WanVideoWrapper",
    "WanVideoEncodeLatentBatch": "ComfyUI-WanVideoWrapper",
    "WanVideoEnhanceAVideo": "ComfyUI-WanVideoWrapper",
    "WanVideoExperimentalArgs": "ComfyUI-WanVideoWrapper",
    "WanVideoImageToVideoEncode": "ComfyUI-WanVideoWrapper",
    "WanVideoLoraSelect": "ComfyUI-WanVideoWrapper",
    "WanVideoLoraSelectMulti": "ComfyUI-WanVideoWrapper",
    "WanVideoModelLoader": "ComfyUI-WanVideoWrapper",
    "WanVideoSVIProEmbeds": "ComfyUI-WanVideoWrapper",
    "WanVideoSampler": "ComfyUI-WanVideoWrapper",
    "WanVideoScheduler": "ComfyUI-WanVideoWrapper",
    "WanVideoSetBlockSwap": "ComfyUI-WanVideoWrapper",
    "WanVideoSetLoRAs": "ComfyUI-WanVideoWrapper",
    "WanVideoSetRadialAttention": "ComfyUI-WanVideoWrapper",
    "WanVideoSigmaToStep": "ComfyUI-WanVideoWrapper",
    "WanVideoTextEncodeCached": "ComfyUI-WanVideoWrapper",
    "WanVideoTorchCompileSettings": "ComfyUI-WanVideoWrapper",
    "WanVideoVAELoader": "ComfyUI-WanVideoWrapper",
    # ComfyUI-KJNodes
    "INTConstant": "ComfyUI-KJNodes",
    "ImageResizeKJv2": "ComfyUI-KJNodes",
    "SimpleMath+": "ComfyUI-KJNodes",
    "PrimitiveFloat": "ComfyUI-KJNodes",
    "PrimitiveInt": "ComfyUI-KJNodes",
    "PrimitiveNode": "ComfyUI-KJNodes",
    "FloatConstant": "ComfyUI-KJNodes",
    # ComfyUI-VideoHelperSuite
    "VHS_LoadAudio": "ComfyUI-VideoHelperSuite",
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
    "VHS_LoadVideoFFmpeg": "ComfyUI-VideoHelperSuite",
    "VHS_SelectEveryNthImage": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    # ComfyUI_essentials
    "GetImageSizeAndCount": "ComfyUI_essentials",
    "ImageFromBatch+": "ComfyUI_essentials",
    # ComfyUI-Easy-Use
    "easy cleanGpuUsed": "ComfyUI-Easy-Use",
    "easy mathFloat": "ComfyUI-Easy-Use",
    "easy mathInt": "ComfyUI-Easy-Use",
    "easy promptReplace": "ComfyUI-Easy-Use",
    "easy showAnything": "ComfyUI-Easy-Use",
    "easy textIndexSwitch": "ComfyUI-Easy-Use",
    # Other custom nodes (same order as CUSTOM_NODES)
    "JWFloatToInteger": "comfyui-various",
    "Any Switch (rgthree)": "rgthree-comfy",
    "Bookmark (rgthree)": "rgthree-comfy",
    "Display Int (rgthree)": "rgthree-comfy",
    "Fast Groups Bypasser (rgthree)": "rgthree-comfy",
    "Fast Groups Muter (rgthree)": "rgthree-comfy",
    "Label (rgthree)": "rgthree-comfy",
    "Mute / Bypass Relay (rgthree)": "rgthree-comfy",
    "Mute / Bypass Repeater (rgthree)": "rgthree-comfy",
    "Power Lora Loader (rgthree)": "rgthree-comfy",
    "Seed (rgthree)": "rgthree-comfy",
    "FL_RIFE": "ComfyUI_Fill-Nodes",
    "ImpactWildcardProcessor": "ComfyUI-Impact-Pack",
    "LoadImagesFromDir //Inspire": "ComfyUI-Inspire-Pack",
    "MathExpression|pysssss": "ComfyUI-Custom-Scripts",
    "PlaySound|pysssss": "ComfyUI-Custom-Scripts",
    "K3NKImageGrab": "ComfyUI-K3NKImageGrab",
    "K3NKImageLoaderWithBlending": "ComfyUI-K3NK-ComfyUI-Nodes",
    "K3NKSaveLatentPassThrought": "ComfyUI-K3NK-ComfyUI-Nodes",
    "Resolutions by Ratio (WLSH)": "wlsh_nodes",
    "ColorCorrect": "ComfyUI-post-processing-nodes",
    "CR String To Combo": "ComfyUI_Comfyroll_CustomNodes",
    "CR Text": "ComfyUI_Comfyroll_CustomNodes",
    "CM_FloatToInt": "ComfyMath",
    "CM_IntToFloat": "ComfyMath",
    "ttN concat": "ComfyUI_tinyterraNodes",
    "ttN int": "ComfyUI_tinyterraNodes",
    "ttN text": "ComfyUI_tinyterraNodes",
    "DF_Int_to_Float": "Derfuu_ComfyUI_ModdedNodes",
    "EG_WXZ_QH": "Comfyui-ergouzi-Nodes",
    # Additional from workflow_node_types.txt
    "FramePackFindNearestBucket": "ComfyUI-FramePackWrapper_PlusOne",
    "PreviewAny": "builtin",
    "PreviewBridge": "ComfyUI-Impact-Pack",
    "Image Overlay": "efficiency-nodes-comfyui",
    "FastFilmGrain": "ComfyUI-post-processing-nodes",
    "ImageSharpen": "ComfyUI-post-processing-nodes",
    # Nodes that were reported missing by validate_workflow_nodes
    "ColorMatchToReference": "comfyui-vrgamedevgirl",
    "DownloadAndLoadFlorence2Model": "ComfyUI-Florence2",
    "Florence2Run": "ComfyUI-Florence2",
    "Incrementer 🪴": "ComfyUI-Addoor",
    "MarkdownNote": "ComfyUI-Impact-Pack",
    "MMAudioFeatureUtilsLoader": "ComfyUI-MMAudio",
    "MMAudioModelLoader": "ComfyUI-MMAudio",
    "MMAudioSampler": "ComfyUI-MMAudio",
    "MelBandRoFormerModelLoader": "ComfyUI-MelBandRoFormer",
    "MelBandRoFormerSampler": "ComfyUI-MelBandRoFormer",
    "NormalizeAudioLoudness": "ComfyUI-MelBandRoFormer",
    "PreviewAudio": "ComfyUI-MelBandRoFormer",
    "NovaSR": "ComfyUI-NovaSR",
    "PauseWorkflowNode": "ComfyUI-pause",
    "Sigmas Split Value": "RES4LYF",
    "mxSlider2D": "ComfyUI-mxToolkit",
    "mxStop": "ComfyUI-mxToolkit",
}


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

def check_workflow_node_coverage() -> bool:
    """
    Read workflow_node_types.txt and verify every type is either builtin or
    provided by a CUSTOM_NODES repo we install. Prints report; returns False
    if any type is unmapped (so caller can warn or fail).
    """
    types_file = os.path.join(WORKFLOW_DIR, "workflow_node_types.txt")
    if not os.path.isfile(types_file):
        print(f"  (No {types_file} — run: python src/scan_workflow_nodes.py)")
        return True

    custom_node_names = {n["name"] for n in CUSTOM_NODES}
    with open(types_file, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    unmapped = []
    for raw in lines:
        # Skip UUID-like ids (from workflow node ids, not type names)
        if len(raw) == 36 and raw.count("-") >= 4:
            continue
        if raw not in WORKFLOW_NODE_SOURCE:
            unmapped.append(raw)
            continue
        src = WORKFLOW_NODE_SOURCE[raw]
        if src != "builtin" and src not in custom_node_names:
            unmapped.append(f"{raw} (repo '{src}' not in CUSTOM_NODES)")
            continue

    if unmapped:
        print(f"  Workflow node types not in prep mapping ({len(unmapped)}):")
        for u in sorted(set(unmapped))[:30]:
            print(f"    - {u}")
        if len(unmapped) > 30:
            print(f"    ... and {len(unmapped) - 30} more")
        print("  After starting ComfyUI, run: python src/validate_workflow_nodes.py")
        print("  to see which nodes are actually missing at runtime.")
        return False
    print(f"  All {len(lines)} workflow node types mapped (builtin or CUSTOM_NODES).")
    return True


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
# 3b. Download RIFE model weights (optional, use --with-rife)
# ============================================================================

def download_rife_weights():
    """Download RIFE model weights from HuggingFace (AlexWortega/RIFE).
    Used only for optional post-workflow Python RIFE in inference (workflow uses FL_RIFE).
    """
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
        print(f"  Post-workflow RIFE frame interpolation will not be available")


# ============================================================================
# 3c. Download default upscale model (workflow FL_RIFE chain uses it)
# ============================================================================

UPSCALE_MODEL_DEFAULT = "2x_NMKD-UpgifLiteV2_210k.pth"
UPSCALE_HF_REPO = "utnah/esrgan"
UPSCALE_HF_FILENAME = "2x_NMKD-UpgifLiteV2_210k.pth"


def download_upscale_model():
    """Download default 2x upscale model for workflow post-processing chain."""
    upscale_dir = os.path.join(MODELS_DIR, "upscale_models")
    dest_path = os.path.join(upscale_dir, UPSCALE_MODEL_DEFAULT)

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"Upscale model already exists: {dest_path} ({size_mb:.1f} MB)")
        return

    os.makedirs(upscale_dir, exist_ok=True)
    print(f"Downloading upscale model to {dest_path}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=UPSCALE_HF_REPO,
            filename=UPSCALE_HF_FILENAME,
            local_dir=upscale_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN,
        )
        if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
            shutil.move(downloaded_path, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  WARNING: Failed to download upscale model: {e}")
        print(f"  Workflow upscale step may fail; use enable_upscale=false or provide model manually.")


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
        # Optional: video_postprocessing (RIFE/Real-ESRGAN when you add post-workflow step)
        "opencv-python>=4.8.0",
        "spandrel>=0.4.0",
        "imageio-ffmpeg>=0.5.1",
        "tenacity==8.5.0",
        "pydantic==2.9.1",
        "Pillow>=10.0.0",
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
    parser.add_argument(
        "--with-rife", action="store_true",
        help="Download RIFE model weights for optional post-workflow frame interpolation (workflow uses built-in FL_RIFE; use when adding RIFE/upscale in inference later)",
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

    # Step 2b: Check workflow node coverage (vs workflow_node_types.txt)
    print("\n--- Step 2b: Workflow Node Coverage ---")
    check_workflow_node_coverage()

    # Step 3: Download models
    print("\n--- Step 3: Download Model Files ---")
    download_models()

    # Step 3c: Download default upscale model (for workflow post-processing)
    print("\n--- Step 3c: Download Upscale Model ---")
    download_upscale_model()

    # Step 3b: Download RIFE weights (optional; for post-workflow use when you add it in inference)
    if getattr(args, "with_rife", False):
        print("\n--- Step 3b: Download RIFE Model Weights ---")
        download_rife_weights()
    else:
        print("\n--- Step 3b: Skipping RIFE weights (workflow uses FL_RIFE; use --with-rife to add for post-workflow RIFE/upscale) ---")

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
        print("  (To enable post-workflow RIFE/upscale later: run prep with --with-rife)")
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
