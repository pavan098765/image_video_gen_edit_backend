# ------------------------------------------------------------------
# Qwen Image Edit 2511 Rapid AIO Inference
# - Extracted from huggingface_app.py
# - Supports adapters and LoRA selection via params
# - Clean inference API without Gradio/UI code
# ------------------------------------------------------------------

import os
import re
import gc
import math
import base64
import traceback
import random
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import torch
import numpy as np

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file

# --- Auto-merge split safetensors if merge.py exists ---
# NOTE: Models are always in parts for Docker image size management
# Merge happens every time on startup before model loading
try:
    import time
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_start = time.time()
    merge_main()
    merge_time = time.time() - merge_start
    print(f"✅ Merge completed in {merge_time:.1f}s")
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel, QwenDoubleStreamAttnProcessor2_0
try:
    from qwenimage.qwen_fa2_processor import QwenDoubleStreamAttnProcessorFA2
    FA2_AVAILABLE = True
except ImportError:
    FA2_AVAILABLE = False
    print("⚠️ Flash Attention 2 processor not available (flash-attn package may not be installed)")

# -------------------------
# Configuration
# -------------------------
AIO_REPO_ID = "Pr0f3ssi0n4ln00b/Phr00t-Qwen-Rapid-AIO"
DEFAULT_AIO_VERSION = "v19"
QWEN_REPO = "Qwen/Qwen-Image-Edit-2511"

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEED = np.iinfo(np.int32).max

# -------------------------
# GPU Memory Detection
# -------------------------
def get_gpu_memory() -> float:
    """Get GPU memory in GB."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        return 0.0
    except Exception:
        return 0.0

# -------------------------
# Version normalization
# -------------------------
_VER_RE = re.compile(r"^v\d+$")
_DIGITS_RE = re.compile(r"^\d+$")


def _normalize_version(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _VER_RE.fullmatch(s):
        return s
    # forgiving: allow "21" -> "v21"
    if _DIGITS_RE.fullmatch(s):
        return f"v{s}"
    return None


# -------------------------
# Adapter specifications
# -------------------------
NONE_LORA = "None"

ADAPTER_SPECS = {
    "Consistance": {
        "type": "single",
        "repo": "Pr0f3ssi0n4ln00b/QIE_2511_Consistency_Lora",
        "weights": "qe2511_consis_alpha_patched.safetensors",
        "adapter_name": "Consistency",
        "strength": 0.6,
        "target_long_edge": 1024,
    },
    "Semirealistic-photo-detailer": {
        "type": "single",
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
        "adapter_name": "semirealistic",
        "strength": 1.0,
        "target_long_edge": 1024,
    },
    "AnyPose": {
        "type": "package",
        "requires_two_images": True,
        "image2_label": "Upload Pose Reference (Image 2)",
        "parts": [
            {
                "repo": "lilylilith/AnyPose",
                "weights": "2511-AnyPose-base-000006250.safetensors",
                "adapter_name": "anypose-base",
                "strength": 0.7,
            },
            {
                "repo": "lilylilith/AnyPose",
                "weights": "2511-AnyPose-helper-00006000.safetensors",
                "adapter_name": "anypose-helper",
                "strength": 0.7,
            },
        ],
    },
    "Any2Real_2601": {
        "type": "single",
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601_A_final_patched.safetensors",
        "adapter_name": "photoreal",
        "strength": 1.0,
    },
    "Hyperrealistic-Portrait": {
        "type": "single",
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "HRPortrait",
        "strength": 1.0,
    },
    "Ultrarealistic-Portrait": {
        "type": "single",
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "URPortrait",
        "strength": 1.0,
    },
    "BFS-Best-FaceSwap-merge": {
        "type": "single",
        "requires_two_images": True,
        "image2_label": "Upload Head/Face Donor (Image 2)",
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors",
        "adapter_name": "BFS-Best-Faceswap-merge",
        "strength": 1.1,
        "needs_alpha_fix": True,
    },
    "F2P": {
        "type": "single",
        "repo": "DiffSynth-Studio/Qwen-Image-Edit-F2P",
        "weights": "edit_0928_lora_step40000.safetensors",
        "adapter_name": "F2P",
        "strength": 1.0,
    },
    "Multiple-Angles": {
        "type": "single",
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
        "strength": 1.0,
    },
    "Light-Restoration": {
        "type": "single",
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
        "adapter_name": "light-restoration",
        "strength": 1.0,
    },
    "Relight": {
        "type": "single",
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
        "adapter_name": "relight",
        "strength": 1.0,
    },
    "Edit-Skin": {
        "type": "single",
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
        "adapter_name": "edit-skin",
        "strength": 1.0,
    },
    "Next-Scene": {
        "type": "single",
        "repo": "lovis93/next-scene-qwen-image-lora-2509",
        "weights": "next-scene_lora-v2-3000.safetensors",
        "adapter_name": "next-scene",
        "strength": 1.0,
    },
    "Flat-Log": {
        "type": "single",
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
        "adapter_name": "flat-log",
        "strength": 1.0,
    },
}

# Track what is currently loaded in memory (adapter_name values)
LOADED_ADAPTERS = set()

# Track the currently ACTIVE adapter configuration to avoid redundant set_adapters() calls
# Format: (tuple(sorted adapter_names), tuple(sorted adapter_weights)) or None
_active_adapter_config = None

# -------------------------
# Lazy pipeline global
# -------------------------
_pipeline = None
_aio_version = None


def get_aio_version() -> str:
    """Get AIO version from env or default."""
    global _aio_version
    if _aio_version is not None:
        return _aio_version
    
    _AIO_ENV_RAW = os.environ.get("AIO_VERSION", "")
    _AIO_ENV_NORM = _normalize_version(_AIO_ENV_RAW)
    _aio_version = _AIO_ENV_NORM or DEFAULT_AIO_VERSION
    return _aio_version

def get_pipeline(version: Optional[str] = None) -> QwenImageEditPlusPipeline:
    """Lazy-initialize the pipeline only once to prevent duplicate GPU memory usage."""
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline

    import time
    total_start = time.time()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    try:
        vram_gb = get_gpu_memory()
    except Exception:
        vram_gb = 8.0
    
    print(f"🔍 Detected GPU VRAM: {vram_gb:.1f} GB")

    aio_version = version or get_aio_version()
    
    AIO_TRANSFORMER_LOCAL_PATH = os.environ.get("AIO_TRANSFORMER_LOCAL_PATH", "/app/models/aio_transformers")
    qwen_local_path = os.environ.get("QWEN_2511_LOCAL_PATH", "/app/models/qwen_image_edit_2511")
    transformer_local_path = os.path.join(AIO_TRANSFORMER_LOCAL_PATH, aio_version, "transformer")
    
    print(f"🔧 Loading AIO transformer (version={aio_version})")
    transformer_start = time.time()
    
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": "cpu",
    }
    
    if os.path.exists(transformer_local_path) and os.listdir(transformer_local_path):
        print(f"  Using local path: {transformer_local_path}")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_local_path,
            **load_kwargs
        )
    else:
        sub = f"{aio_version}/transformer"
        print(f"  Downloading from {AIO_REPO_ID} / {sub}")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            AIO_REPO_ID,
            subfolder=sub,
            **load_kwargs
        )
    
    transformer_time = time.time() - transformer_start
    print(f"✅ Transformer loaded to CPU in {transformer_time:.1f}s")

    print(f"🔧 Building QwenImageEditPlusPipeline")
    pipeline_start = time.time()
    
    pipeline_load_kwargs = {
        "transformer": transformer,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    
    if os.path.exists(qwen_local_path) and os.listdir(qwen_local_path):
        print(f"  Using local Qwen path: {qwen_local_path}")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            qwen_local_path,
            **pipeline_load_kwargs
        )
    else:
        print(f"  Downloading from {QWEN_REPO}")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_REPO,
            **pipeline_load_kwargs
        )

    pipeline_time = time.time() - pipeline_start
    print(f"✅ Pipeline built on CPU in {pipeline_time:.1f}s")

    # Set attention processor BEFORE offload hooks
    attention_processor_set = False
    if FA2_AVAILABLE:
        try:
            pipeline.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA2())
            print("✅ Flash Attention 2 processor set.")
            attention_processor_set = True
        except Exception as e:
            print(f"⚠️ FA2 failed: {e}")
    
    if not attention_processor_set:
        try:
            pipeline.transformer.set_attn_processor(QwenDoubleStreamAttnProcessor2_0())
            print("✅ Using QwenDoubleStreamAttnProcessor2_0.")
            attention_processor_set = True
        except Exception as e:
            print(f"⚠️ Could not set Processor2_0: {e}.")

    # -------------------------
    # Verify all components are on CPU before applying offload.
    # When a pre-instantiated transformer is passed to from_pretrained,
    # diffusers may not register it in self.components properly, causing
    # enable_model_cpu_offload to miss it and leave it on GPU.
    # We explicitly move everything to CPU first to guarantee a clean state.
    # -------------------------
    print("🔧 Ensuring all components are on CPU before applying offload hooks...")
    for name, component in pipeline.components.items():
        if isinstance(component, torch.nn.Module):
            try:
                component.to("cpu")
                print(f"  ✅ {name} → CPU")
            except Exception as e:
                print(f"  ⚠️ Could not move {name} to CPU: {e}")
    
    # Also explicitly move transformer since it may not be in pipeline.components
    # if it was passed as a pre-instantiated object
    try:
        pipeline.transformer.to("cpu")
        print(f"  ✅ transformer (explicit) → CPU")
    except Exception as e:
        print(f"  ⚠️ Could not move transformer to CPU: {e}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Verify GPU memory is now near-zero before applying offload
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free = free_bytes / 1024**3
            print(f"💾 GPU state before offload setup: {allocated:.2f}GB allocated, {free:.2f}GB free of {total_bytes/1024**3:.1f}GB")
            if allocated > 2.0:
                print(f"⚠️ WARNING: {allocated:.2f}GB still on GPU before offload hooks — offload may not fully work")
        except Exception:
            pass

    if vram_gb >= 48:
        print("🧠 >=48GB → enable_model_cpu_offload")
        pipeline.enable_model_cpu_offload()
    elif vram_gb >= 31:
        # RTX 5090: ~31.7GB reported
        # Transformer (20GB) + activations (~4-6GB) = ~24-26GB peak, fits fine
        # with component-level offload keeping text_encoder and VAE on CPU
        print("🧠 RTX 5090 → enable_model_cpu_offload (component-level)")
        # pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        
        # -------------------------
        # Verify offload hooks were actually attached to the transformer.
        # The transformer is the critical component — if hooks are missing here
        # it will sit on GPU consuming 20GB permanently.
        # -------------------------
        # from accelerate.hooks import AlignDevicesHook
        # transformer_has_hook = hasattr(pipeline.transformer, '_hf_hook') or \
        #                        any(hasattr(m, '_hf_hook') for m in pipeline.transformer.modules())
        
        # if not transformer_has_hook:
        #     print("⚠️ WARNING: enable_model_cpu_offload did not attach hooks to transformer!")
        #     print("   Falling back to enable_sequential_cpu_offload for reliable offloading.")
        #     # Reset pipeline state and try sequential offload instead
        #     # First ensure transformer is back on CPU
        #     pipeline.transformer.to("cpu")
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     pipeline.enable_sequential_cpu_offload()
        #     print("✅ Fallback: enable_sequential_cpu_offload applied")
        # else:
        #     print("✅ Verified: offload hooks attached to transformer")

    elif vram_gb >= 23:
        print("🧠 RTX 4090 → enable_sequential_cpu_offload")
        pipeline.enable_sequential_cpu_offload()
    else:
        print(f"🧠 {vram_gb:.1f}GB → enable_sequential_cpu_offload")
        pipeline.enable_sequential_cpu_offload()

    _pipeline = pipeline
    total_time = time.time() - total_start
    print(
        f"✅ Pipeline ready in {total_time:.1f}s "
        f"(transformer: {transformer_time:.1f}s, pipeline: {pipeline_time:.1f}s)"
    )
    
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free = free_bytes / 1024**3
            print(
                f"💾 GPU after pipeline init: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {free:.2f}GB truly free"
            )
            # After offload setup, GPU should be near-empty (model on CPU)
            # If more than 2GB is allocated here, offload hooks are not working
            if allocated > 2.0:
                print(
                    f"🚨 CRITICAL: {allocated:.2f}GB allocated after offload setup. "
                    f"Model is NOT properly offloaded. Inference will OOM."
                )
        except Exception:
            pass
    
    return _pipeline


# -------------------------
# Helper functions
# -------------------------
def _load_image_from_base64(img_b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")


def _ensure_image_list(maybe_one_or_many) -> List[Image.Image]:
    """Accept single base64, list of base64s, or PIL.Image(s); return list of PIL.Images."""
    if maybe_one_or_many is None:
        return []
    if isinstance(maybe_one_or_many, Image.Image):
        return [maybe_one_or_many]
    if isinstance(maybe_one_or_many, (list, tuple)):
        return [
            (_load_image_from_base64(item) if isinstance(item, str) else item)
            for item in maybe_one_or_many
        ]
    if isinstance(maybe_one_or_many, str):
        return [_load_image_from_base64(maybe_one_or_many)]
    raise ValueError("Unsupported image input type")


def _get_dynamic_size_for_images(images: List[Image.Image], max_dim: int = 1152) -> Tuple[int, int]:
    """Keep aspect ratio, clamp to max_dim (scale down only), round to multiples of 64.
    Same logic as worker_qwen_image_edit_2509: avoids upscaling small images and uses 64 alignment."""
    if not images:
        raise ValueError("No images provided")
    w, h = images[0].size
    aspect = w / h

    # First, scale down if needed (never scale up)
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(round(max_dim / aspect))
        else:
            new_h = max_dim
            new_w = int(round(max_dim * aspect))
    else:
        new_w, new_h = w, h

    # Round to multiples of 64 while preserving aspect ratio
    if new_w >= new_h:
        new_w = (new_w // 64) * 64
        new_h = int(round(new_w / aspect))
        new_h = (new_h // 64) * 64
        new_w = int(round(new_h * aspect))
        new_w = (new_w // 64) * 64
    else:
        new_h = (new_h // 64) * 64
        new_w = int(round(new_h * aspect))
        new_w = (new_w // 64) * 64
        new_h = int(round(new_w / aspect))
        new_h = (new_h // 64) * 64

    return new_w, new_h


def _resize_images_to(images: List[Image.Image], size: Tuple[int, int]) -> List[Image.Image]:
    """Resize images to the specified size while preserving aspect ratio."""
    w, h = size
    return [img.resize((w, h), Image.LANCZOS) for img in images]


def get_target_long_edge_for_lora(lora_adapter: str) -> int:
    spec = ADAPTER_SPECS.get(lora_adapter, {})
    return int(spec.get("target_long_edge", 1024))


def lora_requires_two_images(lora_adapter: str) -> bool:
    return bool(ADAPTER_SPECS.get(lora_adapter, {}).get("requires_two_images", False))


def build_labeled_images(images: List[Image.Image]) -> Dict[str, Image.Image]:
    """
    Creates labels image_1, image_2, image_3... from a list of images.
    Supports up to 3 images (Qwen Image Edit 2511 limitation).
    """
    labeled: Dict[str, Image.Image] = {}
    
    # Limit to 3 images (Qwen Image Edit 2511 supports up to 3)
    max_images = min(len(images), 3)
    
    for idx in range(1, max_images + 1):
        if images[idx - 1] is not None:
            labeled[f"image_{idx}"] = images[idx - 1]
    
    return labeled


# -------------------------
# LoRA loading helpers
# -------------------------
def _inject_missing_alpha_keys(state_dict: dict) -> dict:
    """
    Diffusers' Qwen LoRA converter expects '<module>.alpha' keys.
    BFS safetensors omits them. We inject alpha = rank (neutral scaling).
    """
    bases = {}

    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_down.weight") and v.ndim >= 1:
            base = k[: -len(".lora_down.weight")]
            rank = int(v.shape[0])
            bases[base] = rank

    for base, rank in bases.items():
        alpha_tensor = torch.tensor(float(rank), dtype=torch.float32)

        full_alpha = f"{base}.alpha"
        if full_alpha not in state_dict:
            state_dict[full_alpha] = alpha_tensor

        if base.startswith("diffusion_model."):
            stripped_base = base[len("diffusion_model.") :]
            stripped_alpha = f"{stripped_base}.alpha"
            if stripped_alpha not in state_dict:
                state_dict[stripped_alpha] = alpha_tensor

    return state_dict


def _filter_to_diffusers_lora_keys(state_dict: dict) -> tuple[dict, dict]:
    """Return (filtered_state_dict, stats)."""
    keep_suffixes = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_mid.weight",
        ".alpha",
        ".lora_alpha",
    )

    dropped_patch = 0
    dropped_other = 0
    kept = 0
    normalized_alpha = 0

    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            dropped_other += 1
            continue

        if k.endswith(".diff") or k.endswith(".diff_b"):
            dropped_patch += 1
            continue

        if not k.endswith(keep_suffixes):
            dropped_other += 1
            continue

        if k.endswith(".lora_alpha"):
            base = k[: -len(".lora_alpha")]
            k2 = f"{base}.alpha"
            out[k2] = v.float() if v.dtype != torch.float32 else v
            normalized_alpha += 1
            kept += 1
            continue

        out[k] = v
        kept += 1

    stats = {
        "kept": kept,
        "dropped_patch": dropped_patch,
        "dropped_other": dropped_other,
        "normalized_alpha": normalized_alpha,
    }
    return out, stats


def _duplicate_stripped_prefix_keys(state_dict: dict, prefix: str = "diffusion_model.") -> dict:
    """Ensure both prefixed and unprefixed variants exist for LoRA-related keys."""
    out = dict(state_dict)
    for k, v in list(state_dict.items()):
        if not k.startswith(prefix):
            continue
        stripped = k[len(prefix) :]
        if stripped not in out:
            out[stripped] = v
    return out


def _load_lora_weights_with_fallback(
    pipe: QwenImageEditPlusPipeline,
    repo: str,
    weight_name: str,
    adapter_name: str,
    needs_alpha_fix: bool = False,
    display_name: Optional[str] = None,
):
    """
    Load LoRA weights with local file check first, then HuggingFace Hub fallback.
    BFS fallback: download safetensors, inject missing alpha keys, then load from dict.
    
    Args:
        display_name: The display name from ADAPTER_SPECS keys (e.g., "Consistance")
                      Used to find local files. If None, uses adapter_name.
    """
    # Check for local LoRA file first (from prep_model.py)
    # prep_model.py uses display_name (ADAPTER_SPECS key) for directory names
    LORA_ADAPTERS_LOCAL_PATH = os.environ.get("LORA_ADAPTERS_LOCAL_PATH", "/app/models/lora_adapters")
    
    # Use display_name if provided, otherwise try adapter_name
    # prep_model.py creates dirs like: lora_adapters/Consistance/, lora_adapters/Semirealistic-photo-detailer/
    dir_name = (display_name or adapter_name).replace("/", "_")
    adapter_dir = os.path.join(LORA_ADAPTERS_LOCAL_PATH, dir_name)
    local_lora_path = os.path.join(adapter_dir, weight_name)
    
    # Try local file first
    if os.path.exists(local_lora_path):
        print(f"  📁 Using local LoRA file: {local_lora_path}")
        try:
            # Load from local safetensors file
            sd = safetensors_load_file(local_lora_path)
            
            # If needs alpha fix, process it
            if needs_alpha_fix:
                # 1) Inject required `<module>.alpha` keys
                sd = _inject_missing_alpha_keys(sd)
                
                # 2) Keep only LoRA + alpha keys; drop ComfyUI patch/delta keys
                sd, stats = _filter_to_diffusers_lora_keys(sd)
                
                # 3) Duplicate stripped keys
                sd = _duplicate_stripped_prefix_keys(sd)
                
                print(
                    f"🧹 LoRA dict cleanup stats: "
                    f"kept={stats['kept']} dropped_patch={stats['dropped_patch']} "
                    f"dropped_other={stats['dropped_other']} normalized_alpha={stats['normalized_alpha']}"
                )
            
            # Load from state dict
            pipe.load_lora_weights(sd, adapter_name=adapter_name)
            return
        except Exception as local_error:
            print(f"  ⚠️ Failed to load from local file: {local_error}, falling back to HuggingFace Hub")
    
    # Try HuggingFace Hub (will use cache if available)
    try:
        pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=adapter_name)
        return
    except (KeyError, ValueError) as e:
        if not needs_alpha_fix:
            raise

        print(
            f"⚠️ LoRA load failed (will try safe dict fallback). "
            f"Adapter={adapter_name!r} file={weight_name!r} error={type(e).__name__}: {e}"
        )

        # Download from HuggingFace Hub as last resort
        local_path = hf_hub_download(repo_id=repo, filename=weight_name)
        sd = safetensors_load_file(local_path)

        # 1) Inject required `<module>.alpha` keys
        sd = _inject_missing_alpha_keys(sd)

        # 2) Keep only LoRA + alpha keys; drop ComfyUI patch/delta keys
        sd, stats = _filter_to_diffusers_lora_keys(sd)

        # 3) Duplicate stripped keys
        sd = _duplicate_stripped_prefix_keys(sd)

        print(
            f"🧹 LoRA dict cleanup stats: "
            f"kept={stats['kept']} dropped_patch={stats['dropped_patch']} "
            f"dropped_other={stats['dropped_other']} normalized_alpha={stats['normalized_alpha']}"
        )

        pipe.load_lora_weights(sd, adapter_name=adapter_name)
        return


def _ensure_loaded_and_get_active_adapters(
    pipe: QwenImageEditPlusPipeline, selected_lora: str
) -> Tuple[List[str], List[float]]:
    spec = ADAPTER_SPECS.get(selected_lora)
    if not spec:
        raise ValueError(f"Configuration not found for: {selected_lora}")

    adapter_names = []
    adapter_weights = []

    if spec.get("type") == "package":
        parts = spec.get("parts", [])
        if not parts:
            raise ValueError(f"Package spec has no parts: {selected_lora}")

        for part in parts:
            repo = part["repo"]
            weights = part["weights"]
            adapter_name = part["adapter_name"]
            strength = float(part.get("strength", 1.0))
            needs_alpha_fix = bool(part.get("needs_alpha_fix", False))

            if adapter_name not in LOADED_ADAPTERS:
                print(f"--- Loading Adapter Part: {selected_lora} / {adapter_name} ---")
                try:
                    _load_lora_weights_with_fallback(
                        pipe=pipe,
                        repo=repo,
                        weight_name=weights,
                        adapter_name=adapter_name,
                        needs_alpha_fix=needs_alpha_fix,
                        display_name=selected_lora,  # Use display name for local file lookup
                    )
                    LOADED_ADAPTERS.add(adapter_name)
                except Exception as e:
                    raise ValueError(f"Failed to load adapter part {selected_lora}/{adapter_name}: {e}")
            else:
                print(f"--- Adapter part already loaded: {selected_lora} / {adapter_name} ---")

            adapter_names.append(adapter_name)
            adapter_weights.append(strength)

    else:
        repo = spec["repo"]
        weights = spec["weights"]
        adapter_name = spec["adapter_name"]
        strength = float(spec.get("strength", 1.0))
        needs_alpha_fix = bool(spec.get("needs_alpha_fix", False))

        if adapter_name not in LOADED_ADAPTERS:
            print(f"--- Loading Adapter: {selected_lora} ---")
            try:
                _load_lora_weights_with_fallback(
                    pipe=pipe,
                    repo=repo,
                    weight_name=weights,
                    adapter_name=adapter_name,
                    needs_alpha_fix=needs_alpha_fix,
                    display_name=selected_lora,  # Use display name for local file lookup
                )
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise ValueError(f"Failed to load adapter {selected_lora}: {e}")
        else:
            print(f"--- Adapter {selected_lora} is already loaded. ---")

        adapter_names = [adapter_name]
        adapter_weights = [strength]

    return adapter_names, adapter_weights


# -------------------------
# Main inference function
# -------------------------
def run_inference_qwen_image_edit_2511_rapid_aio(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    default_out = "/app/worker/outputs"
    output_dir = os.path.abspath(params.get("output_dir", default_out))
    os.makedirs(output_dir, exist_ok=True)

    aio_version = params.get("aio_version")
    if aio_version:
        aio_version = _normalize_version(aio_version) or DEFAULT_AIO_VERSION
    pipeline = get_pipeline(version=aio_version)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            # CORRECT free memory: actual CUDA free, not total-reserved
            # mem_get_info() returns (free, total) from CUDA driver perspective,
            # accounting for ALL processes and the CUDA context itself —
            # not just what PyTorch knows about.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free = free_bytes / 1024**3
            print(
                f"💾 GPU memory before inference: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {free:.2f}GB truly free "
                f"(of {total_bytes/1024**3:.1f}GB total)"
            )
            if free < 2.0:
                raise RuntimeError(
                    f"Insufficient GPU memory before inference: only {free:.2f}GB free. "
                    f"Another process may be holding GPU memory. Run nvidia-smi to check."
                )
        except RuntimeError:
            raise
        except Exception:
            pass

    global _active_adapter_config
    lora_adapter = params.get("lora_adapter", NONE_LORA)

    if lora_adapter == NONE_LORA:
        desired_config = ((), ())
    else:
        adapter_names, adapter_weights = _ensure_loaded_and_get_active_adapters(pipeline, lora_adapter)
        desired_config = (tuple(adapter_names), tuple(adapter_weights))

    if desired_config != _active_adapter_config:
        with torch.enable_grad():
            if lora_adapter == NONE_LORA:
                try:
                    pipeline.set_adapters([], adapter_weights=[])
                except Exception:
                    if LOADED_ADAPTERS:
                        pipeline.set_adapters(list(LOADED_ADAPTERS), adapter_weights=[0.0] * len(LOADED_ADAPTERS))
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"💾 GPU memory after LoRA loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    except Exception:
                        pass

        _active_adapter_config = desired_config
        print(f"🔄 Adapter config changed → set_adapters({lora_adapter})")
    else:
        print(f"⚡ Adapter config unchanged ({lora_adapter}) → skipping set_adapters()")

    seed = params.get("seed")
    randomize_seed = params.get("randomize_seed", True)
    if randomize_seed or seed is None:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    negative_prompt = params.get(
        "negative_prompt",
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )

    init_image_input = params.get("init_image") or params.get("images")
    if not init_image_input:
        raise ValueError("No input images provided (init_image required)")
    
    image_list = _ensure_image_list(init_image_input)
    if not image_list:
        raise ValueError("No valid images found in init_image")
    
    image_list = [img.convert("RGB") for img in image_list]
    
    if lora_requires_two_images(lora_adapter):
        if len(image_list) < 2:
            raise ValueError(
                f"LoRA adapter '{lora_adapter}' requires two images. "
                f"Please provide init_image as an array with at least 2 images."
            )
    
    if len(image_list) > 3:
        print(f"⚠️ {len(image_list)} images provided, using first 3.")
        image_list = image_list[:3]
    
    max_dim = get_target_long_edge_for_lora(lora_adapter)
    width, height = _get_dynamic_size_for_images(image_list, max_dim=max_dim)
    image_list = _resize_images_to(image_list, (width, height))
    
    labeled = build_labeled_images(image_list)
    pipe_images = list(labeled.values())
    if len(pipe_images) == 1:
        pipe_images = pipe_images[0]

    num_inference_steps = int(params.get("num_inference_steps", 4))
    guidance_scale = float(params.get("guidance_scale", 1.0))

    print(
        f"🧪 Running pipeline job {job_id} | "
        f"lora={lora_adapter!r} seed={seed} steps={num_inference_steps} "
        f"guidance={guidance_scale} size={width}x{height}"
    )

    result = None
    image = None
    try:
        # Final cache flush immediately before pipeline call
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # CRITICAL: inference_mode must wrap the pipeline call.
        # Without it, PyTorch builds a full autograd graph for every operation,
        # keeping ALL intermediate activation tensors alive for potential
        # gradient computation. For this transformer this 2-3x the memory
        # footprint during the forward pass and will cause OOM.
        # torch.enable_grad() around set_adapters() above is a separate concern
        # and does not affect this — inference_mode here is not redundant.
        with torch.inference_mode():
            result = pipeline(
                image=pipe_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=guidance_scale,
            )
            image = result.images[0]

    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU OOM during inference: {e}")
        print("🧹 Performing aggressive GPU cleanup...")
        if result is not None:
            del result
        if image is not None:
            del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        raise RuntimeError(
            f"GPU Out of Memory during inference. "
            f"Try reducing image size, num_inference_steps, or using a smaller LoRA adapter. "
            f"Original error: {e}"
        )
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        if result is not None:
            del result
        if image is not None:
            del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        raise
    finally:
        if result is not None:
            del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free_bytes, _ = torch.cuda.mem_get_info()
                free = free_bytes / 1024**3
                print(
                    f"💾 GPU memory after inference: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved, {free:.2f}GB truly free"
                )
            except Exception:
                pass

    if image is None:
        raise RuntimeError("Inference failed: no image was generated")
    
    out_path = os.path.join(output_dir, f"qwen_2511_{job_id}.png")
    image.save(out_path)
    print(f"✅ Saved base image: {out_path}")

    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = "RealESRGAN_x2plus" if scale_factor <= 2 else "RealESRGAN_x4plus"
        try:
            from upscaler import upscale_image
            upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)
            up_path = os.path.join(output_dir, f"qwen_2511_{job_id}_upscaled.png")
            upscaled.save(up_path)
            print(f"✅ Saved upscaled image: {up_path}")
            return os.path.abspath(up_path)
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
            print(traceback.format_exc())
            return os.path.abspath(out_path)

    return os.path.abspath(out_path)

# -------------------------
# CLI test mode
# -------------------------
if __name__ == "__main__":
    import httpx
    job_id = "localtest"
    prompt = "Change her body orientation so she is facing the camera directly."
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"

    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    img_b64 = base64.b64encode(img_bytes).decode()

    params = {
        "init_image": img_b64,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "lora_adapter": NONE_LORA,
        "scale_image": True,
        "scale_factor": 2,
        "output_dir": "/app/worker/outputs",
    }

    out = run_inference_qwen_image_edit_2511_rapid_aio(job_id, prompt, params)
    print("✅ Final output path:", out)
