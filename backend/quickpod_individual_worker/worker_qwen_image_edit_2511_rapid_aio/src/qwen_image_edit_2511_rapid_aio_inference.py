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
    },
    "Semirealistic-photo-detailer": {
        "type": "single",
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
        "adapter_name": "semirealistic",
        "strength": 1.0,
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
    
    # Clear GPU cache before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # -------------------------
    # OOM Protection: Detect VRAM FIRST to determine loading strategy
    # -------------------------
    try:
        vram_gb = get_gpu_memory()
    except Exception:
        vram_gb = 8.0  # Default to low VRAM if detection fails
    
    print(f"🔍 Detected GPU VRAM: {vram_gb:.1f} GB")
    
    # Determine loading strategy based on VRAM
    # For 32GB and below, load to CPU first, then enable offloading
    # For 48GB+, can load directly to GPU
    use_cpu_first = vram_gb < 48
    
    # Use provided version or get from env/default
    aio_version = version or get_aio_version()
    
    # Check for local transformer path first (from prep_model.py)
    AIO_TRANSFORMER_LOCAL_PATH = os.environ.get("AIO_TRANSFORMER_LOCAL_PATH", "/app/models/aio_transformers")
    qwen_local_path = os.environ.get("QWEN_2511_LOCAL_PATH", "/app/models/qwen_image_edit_2511")
    
    transformer_local_path = os.path.join(AIO_TRANSFORMER_LOCAL_PATH, aio_version, "transformer")
    
    print(f"🔧 Loading AIO transformer (version={aio_version})")
    if use_cpu_first:
        print("  📦 Loading to CPU first (will offload to GPU after)")
    
    transformer_start = time.time()
    
    # Load transformer with memory-efficient settings
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    
    # For 32GB and below, load to CPU first to avoid OOM during loading
    if use_cpu_first:
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
    
    # Try local path first, fallback to HuggingFace Hub
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
    print(f"✅ Transformer loaded in {transformer_time:.1f}s")

    print(f"🔧 Building QwenImageEditPlusPipeline")
    pipeline_start = time.time()
    
    # Load pipeline - for CPU-first strategy, load to CPU then move components
    pipeline_load_kwargs = {
        "transformer": transformer,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    
    # Try local Qwen path first, fallback to HuggingFace Hub
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
    
    # For CPU-first strategy, don't move to GPU yet - offloading will handle it
    if not use_cpu_first and torch.cuda.is_available():
        pipeline = pipeline.to(device)
    
    pipeline_time = time.time() - pipeline_start
    print(f"✅ Pipeline built in {pipeline_time:.1f}s")

    # Apply Flash Attention 2 (FA2) - Compatible with RTX 5090
    # Priority: FA2 > Processor2_0 (PyTorch native)
    # FA2 is preferred because it's faster and more memory-efficient than PyTorch native attention
    attention_processor_set = False
    
    # Try Flash Attention 2 first (compatible with RTX 5090)
    if FA2_AVAILABLE:
        try:
            pipeline.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA2())
            print("✅ Flash Attention 2 Processor set successfully (compatible with RTX 5090).")
            attention_processor_set = True
        except (RuntimeError, ImportError, Exception) as e:
            error_msg = str(e)
            if "no kernel image" in error_msg or "CUDA error" in error_msg or "cuda" in error_msg.lower():
                print(f"⚠️ Flash Attention 2 not compatible with this GPU architecture")
                print(f"   Error: {error_msg[:200]}...")
            else:
                print(f"⚠️ Warning: Could not set FA2 processor: {e}")
    
    # Fall back to Processor2_0 (PyTorch native attention) if FA2 failed
    if not attention_processor_set:
        try:
            pipeline.transformer.set_attn_processor(QwenDoubleStreamAttnProcessor2_0())
            print("✅ Using QwenDoubleStreamAttnProcessor2_0 (PyTorch native scaled_dot_product_attention)")
            attention_processor_set = True
        except Exception as fallback_error:
            print(f"⚠️ Could not set Processor2_0 fallback: {fallback_error}")
            print("   Pipeline will use default attention processor")

    # -------------------------
    # OOM Protection: Enable CPU Offloading Based on VRAM
    # -------------------------
    # VRAM thresholds for adaptive offloading:
    # - 48GB+ (A100, H100): Simple CPU offload (plenty of VRAM)
    # - 32GB (RTX 5090): Aggressive sequential offload with memory-efficient attention
    # - 24GB (RTX 4090): Maximum aggressive offload (1 block, no pin memory)
    # - <24GB: Most aggressive sequential offload
    
    if vram_gb >= 48:
        print("🧠 Very High VRAM (>=48GB) → enabling simple CPU offload")
        pipeline.enable_model_cpu_offload()
    elif vram_gb >= 32:
        # RTX 5090 (32GB): Use aggressive sequential offload with memory-efficient attention
        print("🧠 High VRAM (32GB RTX 5090) → enabling aggressive sequential offload with memory-efficient attention")
        
        # CRITICAL: Enable sequential CPU offload BEFORE any GPU operations
        # This moves models from CPU to GPU in a memory-efficient way
        pipeline.enable_sequential_cpu_offload()
        
        # Try to enable transformer offloading if available
        try:
            if hasattr(transformer, 'set_offload'):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=2)
                if hasattr(pipeline, '_exclude_from_cpu_offload'):
                    pipeline._exclude_from_cpu_offload.append("transformer")
                print("✅ Transformer offloading enabled (2 blocks on GPU)")
        except Exception as e:
            print(f"ℹ️ Transformer offloading not available: {e}")
        
        # Enable memory-efficient attention
        # NOTE: XFormers may use incompatible Flash Attention kernels on RTX 5090
        # Use attention slicing instead for better compatibility
        try:
            pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled (compatible with RTX 5090)")
        except Exception as e:
            print(f"ℹ️ Attention slicing not available: {e}")
            # Try xformers as last resort, but catch CUDA errors
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except (RuntimeError, Exception) as xformers_err:
                error_msg = str(xformers_err)
                if "CUDA error" in error_msg or "invalid argument" in error_msg:
                    print(f"⚠️ XFormers not compatible with this GPU: {error_msg[:200]}")
                    print("   Continuing without memory-efficient attention optimizations")
                else:
                    print(f"ℹ️ XFormers not available: {xformers_err}")
                
    elif vram_gb >= 24:
        # RTX 4090 (24GB): Maximum aggressive offload
        print("🧠 24GB VRAM (RTX 4090) → enabling MAXIMUM aggressive sequential offload (1 block, no pin memory)")
        
        # CRITICAL: Enable sequential CPU offload FIRST
        pipeline.enable_sequential_cpu_offload()
        
        try:
            if hasattr(transformer, 'set_offload'):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                if hasattr(pipeline, '_exclude_from_cpu_offload'):
                    pipeline._exclude_from_cpu_offload.append("transformer")
                print("✅ Transformer offloading enabled (1 block on GPU)")
        except Exception as e:
            print(f"ℹ️ Transformer offloading not available: {e}")
        
        # Enable memory-efficient attention
        # NOTE: XFormers may use incompatible Flash Attention kernels on RTX 5090
        # Use attention slicing instead for better compatibility
        try:
            pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled (compatible with RTX 5090)")
        except Exception as e:
            print(f"ℹ️ Attention slicing not available: {e}")
            # Try xformers as last resort, but catch CUDA errors
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except (RuntimeError, Exception) as xformers_err:
                error_msg = str(xformers_err)
                if "CUDA error" in error_msg or "invalid argument" in error_msg:
                    print(f"⚠️ XFormers not compatible with this GPU: {error_msg[:200]}")
                    print("   Continuing without memory-efficient attention optimizations")
                else:
                    print(f"ℹ️ XFormers not available: {xformers_err}")
                
    elif vram_gb >= 20:
        # 20-24GB: Aggressive sequential offload
        print("🧠 Medium VRAM (20-24GB) → enabling aggressive sequential offload")
        
        # CRITICAL: Enable sequential CPU offload FIRST
        pipeline.enable_sequential_cpu_offload()
        
        try:
            if hasattr(transformer, 'set_offload'):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                if hasattr(pipeline, '_exclude_from_cpu_offload'):
                    pipeline._exclude_from_cpu_offload.append("transformer")
        except Exception:
            pass
        
        try:
            pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled")
        except Exception:
            pass
    else:
        # <20GB: Most aggressive sequential offload
        print("🧠 Low VRAM (<20GB) → enabling most aggressive sequential offload")
        
        # CRITICAL: Enable sequential CPU offload FIRST
        pipeline.enable_sequential_cpu_offload()
        
        try:
            if hasattr(transformer, 'set_offload'):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                if hasattr(pipeline, '_exclude_from_cpu_offload'):
                    pipeline._exclude_from_cpu_offload.append("transformer")
        except Exception:
            pass
        
        try:
            pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled")
        except Exception:
            pass

    _pipeline = pipeline
    total_time = time.time() - total_start
    print(f"✅ Pipeline initialized successfully in {total_time:.1f}s (transformer: {transformer_time:.1f}s, pipeline: {pipeline_time:.1f}s)")
    
    # Log memory state after loading
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU memory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
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


def _round8(x: int) -> int:
    return max(8, (int(x) // 8) * 8)


def compute_dimensions(image: Image.Image, long_edge: int) -> tuple[int, int]:
    """Compute dimensions preserving aspect ratio, rounded to multiples of 8."""
    w, h = image.size
    aspect = w / h
    
    # Determine which edge is longer
    if w >= h:
        # Width is longer - round it first
        new_w = _round8(long_edge)
        # Compute height from rounded width to maintain exact aspect ratio
        new_h = int(round(new_w / aspect))
        # Round height to multiple of 8
        new_h = _round8(new_h)
        # Recompute width from rounded height to ensure aspect ratio is maintained
        new_w = int(round(new_h * aspect))
        new_w = _round8(new_w)
    else:
        # Height is longer - round it first
        new_h = _round8(long_edge)
        # Compute width from rounded height to maintain exact aspect ratio
        new_w = int(round(new_h * aspect))
        # Round width to multiple of 8
        new_w = _round8(new_w)
        # Recompute height from rounded width to ensure aspect ratio is maintained
        new_h = int(round(new_w / aspect))
        new_h = _round8(new_h)
    
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
    """
    Run inference for Qwen Image Edit 2511 Rapid AIO.
    
    Args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for image editing
        params:
            - output_dir (str): output directory
            - init_image (base64 string OR array of base64 strings): input image(s)
                - Single string: one image
                - Array: multiple images (up to 3 supported by Qwen Image Edit 2511)
                - For 2-image LoRAs: ensure init_image is array with at least 2 images
            - lora_adapter (str): LoRA adapter name (from ADAPTER_SPECS keys or "None")
            - aio_version (str, optional): AIO version (default: v19)
            - num_inference_steps (int, default: 4)
            - guidance_scale (float, default: 1.0)
            - seed (int, optional): random seed (if not provided, uses random)
            - randomize_seed (bool, default: True): whether to randomize seed
            - negative_prompt (str, optional): negative prompt
    
    Returns:
        absolute path to final image
    """
    # Consistent global output path
    default_out = "/app/worker/outputs"
    output_dir = os.path.abspath(params.get("output_dir", default_out))
    os.makedirs(output_dir, exist_ok=True)

    # Get pipeline (with optional version override)
    aio_version = params.get("aio_version")
    if aio_version:
        aio_version = _normalize_version(aio_version) or DEFAULT_AIO_VERSION
    pipeline = get_pipeline(version=aio_version)

    # CRITICAL: Aggressive GPU memory cleanup before inference
    # This maximizes available VRAM, especially important for 32GB cards
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    # Log memory state before inference
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = get_gpu_memory() - reserved
            print(f"💾 GPU memory before inference: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free")
            
            # Warn if memory is getting tight
            if free < 4.0:
                print(f"⚠️ WARNING: Only {free:.2f} GB free GPU memory. Inference may be at risk of OOM.")
        except Exception:
            pass

    # Handle LoRA adapter
    # OPTIMIZATION: Track active adapter config and skip set_adapters() if unchanged.
    # set_adapters() walks every LoRA layer, toggling requires_grad — expensive and also
    # triggers "requires_grad on inference tensor" errors after torch.inference_mode().
    # By skipping redundant calls we avoid both the overhead AND the error.
    global _active_adapter_config
    lora_adapter = params.get("lora_adapter", NONE_LORA)

    if lora_adapter == NONE_LORA:
        desired_config = ((), ())  # no adapters active
    else:
        # Ensure LoRA weights are loaded (cheap no-op if already in LOADED_ADAPTERS)
        adapter_names, adapter_weights = _ensure_loaded_and_get_active_adapters(pipeline, lora_adapter)
        # Build a hashable key from (names, weights) to compare with current state
        desired_config = (tuple(adapter_names), tuple(adapter_weights))

    if desired_config != _active_adapter_config:
        # Adapter config changed — must call set_adapters().
        # Use torch.enable_grad() to temporarily exit inference mode so that
        # requires_grad can be toggled on inference tensors created by the previous job.
        with torch.enable_grad():
            if lora_adapter == NONE_LORA:
                try:
                    pipeline.set_adapters([], adapter_weights=[])
                except Exception:
                    if LOADED_ADAPTERS:
                        pipeline.set_adapters(list(LOADED_ADAPTERS), adapter_weights=[0.0] * len(LOADED_ADAPTERS))
            else:
                # Clear GPU memory before loading LoRA (can be memory-intensive)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

                # Clear GPU memory after loading LoRA
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Log memory after LoRA loading
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"💾 GPU memory after LoRA loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                    except Exception:
                        pass

        _active_adapter_config = desired_config
        print(f"🔄 Adapter config changed → set_adapters({lora_adapter})")
    else:
        print(f"⚡ Adapter config unchanged ({lora_adapter}) → skipping set_adapters()")

    # Handle seed
    seed = params.get("seed")
    randomize_seed = params.get("randomize_seed", True)
    if randomize_seed or seed is None:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    # Handle negative prompt
    negative_prompt = params.get(
        "negative_prompt",
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )

    # Normalize images - init_image can be single base64 string or array of base64 strings
    init_image_input = params.get("init_image") or params.get("images")
    if not init_image_input:
        raise ValueError("No input images provided (init_image required)")
    
    # Convert to list of PIL Images
    image_list = _ensure_image_list(init_image_input)
    if not image_list:
        raise ValueError("No valid images found in init_image")
    
    # Convert all to RGB
    image_list = [img.convert("RGB") for img in image_list]
    
    # Enforce 2-image LoRA behavior - check if init_image is array with at least 2 images
    if lora_requires_two_images(lora_adapter):
        if len(image_list) < 2:
            raise ValueError(
                f"LoRA adapter '{lora_adapter}' requires two images. "
                f"Please provide init_image as an array with at least 2 images."
            )
    
    # Limit to 3 images (Qwen Image Edit 2511 supports up to 3)
    if len(image_list) > 3:
        print(f"⚠️ Warning: {len(image_list)} images provided, but Qwen Image Edit 2511 supports up to 3. Using first 3 images.")
        image_list = image_list[:3]
    
    # Resolution derived from first image
    target_long_edge = get_target_long_edge_for_lora(lora_adapter)
    width, height = compute_dimensions(image_list[0], target_long_edge)
    
    # CRITICAL: Resize images to computed dimensions to prevent cropping
    # This ensures the output matches the full input image aspect ratio
    image_list = _resize_images_to(image_list, (width, height))
    
    # Label images as image_1, image_2, image_3...
    labeled = build_labeled_images(image_list)

    # Pass to pipeline in labeled order
    pipe_images = list(labeled.values())
    if len(pipe_images) == 1:
        pipe_images = pipe_images[0]

    # Prepare pipeline inputs
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
        # CRITICAL: Handle OOM errors gracefully
        print(f"❌ GPU OOM detected during inference: {e}")
        print("🧹 Performing aggressive GPU cleanup...")
        
        # Aggressive cleanup
        if result is not None:
            del result
        if image is not None:
            del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        
        # Re-raise with helpful message
        raise RuntimeError(
            f"GPU Out of Memory during inference. "
            f"Try reducing image size, num_inference_steps, or using a smaller LoRA adapter. "
            f"Original error: {e}"
        )
    except Exception as e:
        # Handle any other exceptions during inference
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
        # CRITICAL: Aggressive GPU memory cleanup after inference
        # This prevents OOM on back-to-back jobs, especially on 32GB cards
        if result is not None:
            del result
        # Note: Don't delete image here - we need it for saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        
        # Log memory state after inference
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU memory after inference: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            except Exception:
                pass

    # Save the image (only if inference succeeded)
    if image is None:
        raise RuntimeError("Inference failed: no image was generated")
    
    # Handle upscaling if requested
    scale_image = params.get("scale_image", False)
    scale_factor = params.get("scale_factor", 2.0)
    
    if scale_image and scale_factor:
        try:
            # Lazy import to avoid loading RealESRGAN if not needed
            from upscaler import upscale_image
            
            # Determine model based on scale factor
            if scale_factor <= 2.0:
                model_name = "RealESRGAN_x2plus"
            else:
                model_name = "RealESRGAN_x4plus"
            
            print(f"🔍 Upscaling image with {model_name} (scale_factor={scale_factor})...")
            
            # Upscale the image
            image = upscale_image(
                input_image=image,
                model_name=model_name,
                scale_factor=float(scale_factor),
                remove_metadata=True
            )
            
            print(f"✅ Image upscaled successfully to {image.size[0]}x{image.size[1]}")
        except ImportError as e:
            print(f"⚠️ Upscaler not available: {e}. Saving original image without upscaling.")
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}. Saving original image without upscaling.")
            # Continue with original image if upscaling fails
    
    out_path = os.path.join(output_dir, f"qwen_2511_{job_id}.png")
    image.save(out_path)
    print(f"✅ Saved output image: {out_path}")

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
        "output_dir": "/app/worker/outputs",
    }

    out = run_inference_qwen_image_edit_2511_rapid_aio(job_id, prompt, params)
    print("✅ Final output path:", out)
