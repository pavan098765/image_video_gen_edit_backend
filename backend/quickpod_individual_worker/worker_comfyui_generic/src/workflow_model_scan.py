#!/usr/bin/env python3
"""
workflow_model_scan.py
======================

Extract model-like asset paths referenced in ComfyUI workflows by converting
visual JSON to API format (so filename normalization matches inference) and
walking string-valued inputs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Filename extensions commonly used for weights on disk
MODEL_FILE_EXTENSIONS = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".gguf",
)

# Explicit (class_type, input_name) -> folder under ComfyUI models/
CLASS_INPUT_TO_MODELS_SUBDIR: Dict[Tuple[str, str], str] = {
    ("CheckpointLoaderSimple", "ckpt_name"): "checkpoints",
    ("CheckpointLoader", "ckpt_name"): "checkpoints",
    ("VAELoader", "vae_name"): "vae",
    ("CLIPLoader", "clip_name"): "clip",
    ("DualCLIPLoader", "clip_name1"): "clip",
    ("DualCLIPLoader", "clip_name2"): "clip",
    ("TripleCLIPLoader", "clip_name1"): "clip",
    ("TripleCLIPLoader", "clip_name2"): "clip",
    ("TripleCLIPLoader", "clip_name3"): "clip",
    ("LoraLoader", "lora_name"): "loras",
    ("LoraLoaderModelOnly", "lora_name"): "loras",
    ("ControlNetLoader", "control_net_name"): "controlnet",
    ("DiffControlNetLoader", "control_net_name"): "controlnet",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
    ("CLIPVisionLoader", "clip_name"): "clip_vision",
    ("StyleModelLoader", "style_model_name"): "style_models",
    ("UNETLoader", "unet_name"): "diffusion_models",
    ("CLIPTextEncode", "clip"): "clip",
    ("GLIGENLoader", "gligen_name"): "gligen",
    ("HypernetworkLoader", "hypernetwork_name"): "hypernetworks",
    ("PhotomakerLoader", "photomaker_model_name"): "photomaker",
    ("WanVideoModelLoader", "model"): "diffusion_models",
    ("WanVideoVAELoader", "vae_name"): "vae",
    ("WanVideoTorchCompileSettings", "model"): "diffusion_models",
    ("SaveImage", "filename_prefix"): "",
    ("VHS_LoadVideo", "video"): "",
    ("VHS_LoadVideoFFmpeg", "video"): "",
    ("FL_RIFE", "ckpt_name"): "rife",
    ("FL_RIFE", "model"): "rife",
}


def _is_model_candidate_string(s: str) -> bool:
    t = s.strip()
    if len(t) < 3 or len(t) > 1024:
        return False
    if "\n" in t or t.startswith("http://") or t.startswith("https://"):
        return False
    if not any(t.lower().endswith(ext) for ext in MODEL_FILE_EXTENSIONS):
        return False
    return True


def infer_models_subdir(class_type: str, input_key: str) -> str:
    key = (class_type, input_key)
    if key in CLASS_INPUT_TO_MODELS_SUBDIR:
        sub = CLASS_INPUT_TO_MODELS_SUBDIR[key]
        return sub
    ik = input_key.lower()
    ct = class_type.lower()
    if "lora" in ik:
        return "loras"
    if "vae" in ik and "tile" not in ik:
        return "vae"
    if "clip" in ik and "vision" in ik:
        return "clip_vision"
    if "clip" in ik:
        return "clip"
    if "control_net" in ik or "controlnet" in ik:
        return "controlnet"
    if "upscale" in ik or "upscale_" in ct:
        return "upscale_models"
    if "unet" in ik:
        return "diffusion_models"
    if "ckpt" in ik or "checkpoint" in ik:
        return "checkpoints"
    if "style_model" in ik:
        return "style_models"
    if "embedding" in ik or "embed" in ik:
        return "embeddings"
    if "hyper" in ik:
        return "hypernetworks"
    if "gligen" in ik:
        return "gligen"
    if "t5" in ik or "umt5" in ik or "text_encoder" in ik:
        return "text_encoders"
    if "vision" in ik:
        return "clip_vision"
    if ik in ("model", "model_name", "weight_dtype"):
        if "wan" in ct or "video" in ct or "unet" in ct:
            return "diffusion_models"
    if "rife" in ct or "fl_rife" in ct:
        return "rife"
    if "audio" in ct and ("model" in ik or "ckpt" in ik):
        return "checkpoints"
    return "checkpoints"


def models_relpath(class_type: str, input_key: str, value: str) -> str:
    """Relative path under ComfyUI models/ (posix, no leading slash)."""
    sub = infer_models_subdir(class_type, input_key)
    if not sub:
        return ""
    val = value.strip().replace("\\", "/").lstrip("/")
    parts = val.split("/")
    basename = parts[-1]
    if not any(basename.lower().endswith(ext) for ext in MODEL_FILE_EXTENSIONS):
        return ""
    if class_type == "WanVideoModelLoader" and "/" not in val:
        val = f"Wan/{val}"
    return f"{sub}/{val}" if val else ""


@dataclass(frozen=True)
class ModelRef:
    """Path relative to ComfyUI models/ directory."""

    relpath: str
    class_type: str
    input_name: str


def scan_workflow_file(workflow_path: Path) -> List[ModelRef]:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from workflow_converter import load_and_convert

    api = load_and_convert(workflow_path)
    return scan_api_workflow(api)


def scan_api_workflow(api: dict) -> List[ModelRef]:
    out: List[ModelRef] = []
    for _nid, node in api.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type") or ""
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for ik, val in inputs.items():
            if isinstance(val, str) and _is_model_candidate_string(val):
                rel = models_relpath(ct, ik, val)
                if rel:
                    out.append(ModelRef(relpath=rel, class_type=ct, input_name=ik))
    return out


def collect_unique_refs(workflow_paths: Iterable[Path]) -> Dict[str, ModelRef]:
    """relpath -> one representative ModelRef (dedup)."""
    by_path: Dict[str, ModelRef] = {}
    for p in workflow_paths:
        if not p.is_file():
            continue
        for ref in scan_workflow_file(p):
            by_path.setdefault(ref.relpath, ref)
    return by_path


def list_workflow_jsons(workflow_dir: Path) -> List[Path]:
    if not workflow_dir.is_dir():
        return []
    return sorted(workflow_dir.glob("*.json"))
