# qwen_image_edit_2509_inference.py
# ------------------------------------------------------------------
# Qwen Image Edit 2509 Inference
# - Lazy pipeline loading (prevents OOM during FastAPI startup)
# - Consistent /app/worker/outputs path
# - Supports base64/multiple init images + optional upscaling
# ------------------------------------------------------------------

import os
import math
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple
from PIL import Image
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from upscaler import upscale_image

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -------------------------
# Scheduler (Nunchaku reference config)
# -------------------------
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# -------------------------
# Model path setup
# -------------------------
NUM_INFERENCE_STEPS = 8     # lower steps for speed; set to 8 for higher quality
RANK = 128                  
precision = get_precision()

model_filename = f"svdq-{precision}_r{RANK}-qwen-image-edit-2509-lightningv2.0-{NUM_INFERENCE_STEPS}steps.safetensors"
model_path = os.path.join(
    os.environ.get("NUNCHAKU_QWEN_2509_LOCAL_PATH", "/app/models/nunchaku_qwen_2509"),
    model_filename,
)
qwen_repo_path = os.environ.get("QWEN_2509_LOCAL_PATH", "Qwen/Qwen-Image-Edit-2509")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Transformer safetensor not found: {model_path}")

# -------------------------
# Lazy pipeline global (will load on first use)
# -------------------------
_pipeline = None


def get_pipeline():
    """Lazy-initialize the pipeline only once to prevent duplicate GPU memory usage."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"🔧 Loading Nunchaku transformer from {model_path}")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

    print(f"🔧 Building QwenImageEditPlusPipeline from {qwen_repo_path}")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_repo_path,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )

    try:
        vram = get_gpu_memory()
    except Exception:
        vram = 8
    
    print(f"🔍 Detected GPU VRAM: {vram:.1f} GB")

    # VRAM thresholds:
    # - 48GB+ (5090, A100): High VRAM mode with simple CPU offload
    # - 24-48GB (4090, 3090): Medium VRAM with more aggressive offloading
    # - <24GB: Low VRAM with sequential offload
    if vram >= 40:
        print("🧠 Very High VRAM (>=40GB) → enabling simple CPU offload")
        pipeline.enable_model_cpu_offload()
    elif vram >= 20:
        # 24GB cards (4090) need more aggressive memory management
        print("🧠 Medium VRAM (20-40GB) → enabling aggressive sequential offload")
        # More blocks on GPU for 24GB, but still use sequential offload
        num_blocks = 2 if vram >= 24 else 1
        transformer.set_offload(True, use_pin_memory=True, num_blocks_on_gpu=num_blocks)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
        
        # Also enable memory efficient attention if available
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory efficient attention enabled")
        except Exception as e:
            print(f"ℹ️ XFormers not available: {e}")
    else:
        print("🧠 Low VRAM (<20GB) → enabling most aggressive sequential offload")
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()

    _pipeline = pipeline
    print("✅ Pipeline initialized successfully")
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
    """Keep aspect ratio, clamp to max_dim, round to multiples of 64."""
    if not images:
        raise ValueError("No images provided")
    w, h = images[0].size
    aspect = w / h
    if max(w, h) > max_dim:
        if w > h:
            w, h = max_dim, int(max_dim / aspect)
        else:
            h, w = max_dim, int(max_dim * aspect)
    w = (w // 64) * 64
    h = (h // 64) * 64
    return w, h


def _resize_images_to(images: List[Image.Image], size: Tuple[int, int]) -> List[Image.Image]:
    w, h = size
    return [img.resize((w, h), Image.LANCZOS) for img in images]


# -------------------------
# Main inference function
# -------------------------
def run_inference_qwen_image_edit_2509(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for image editing
        params:
        - output_dir (str)
        - init_image OR images (base64/list/PIL)
        - num_inference_steps (int)
        - cfg_scale (float)
        - negative_prompt (str)
        - scale_image (bool)
        - scale_factor (int)
        - scale_model_name (str)
    
    output: absolute path to final image (upscaled if requested)
    """
    # Consistent global output path
    default_out = "/app/worker/outputs"
    output_dir = os.path.abspath(params.get("output_dir", default_out))
    os.makedirs(output_dir, exist_ok=True)

    # Normalize images
    images = _ensure_image_list(params.get("init_image") or params.get("images"))
    if not images:
        raise ValueError("No input images provided")

    width, height = _get_dynamic_size_for_images(images)
    images = _resize_images_to(images, (width, height))

    # Prepare pipeline and inputs
    pipeline = get_pipeline()
    
    # CRITICAL: Clear GPU memory before inference to maximize available VRAM
    # This is especially important for 24GB cards (4090) running int4
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Log memory state before inference
    try:
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"🔍 GPU memory before inference: {free_mem / 1024**3:.2f} GB free")
    except Exception:
        pass

    inputs = {
        "image": images if len(images) > 1 else images[0],
        "prompt": prompt,
        "true_cfg_scale": float(params.get("cfg_scale", 1.0)),
        "num_inference_steps": int(params.get("num_inference_steps", NUM_INFERENCE_STEPS)),
    }

    # Only include width/height if *no init image* is provided
    if not params.get("init_image") and not params.get("images"):
        inputs["width"] = width
        inputs["height"] = height

    print(f"🧪 Running pipeline job {job_id} | {len(images)} images, steps={inputs['num_inference_steps']}")
    
    # Run inference with automatic mixed precision for memory efficiency
    try:
        with torch.inference_mode():
            result = pipeline(**inputs)
        image = result.images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        # This prevents OOM on back-to-back jobs, especially on 24GB cards
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"qwen_{job_id}.png")
    image.save(out_path)
    print(f"✅ Saved base image: {out_path}")

    # Optional upscaling
    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        try:
            upscaled = upscale_image(out_path,  model_name=model_name, scale_factor=scale_factor)
            up_path = os.path.join(output_dir, f"qwen_{job_id}_upscaled.png")
            upscaled.save(up_path)
            print(f"✅ Saved upscaled image: {up_path}")
            return os.path.abspath(up_path)
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
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
        "cfg_scale": 1.1,
        "num_inference_steps": 4,
        "scale_image": False,
        "output_dir": "/app/worker/outputs",
    }

    out = run_inference_qwen_image_edit_2509(job_id, prompt, params)
    print("✅ Final output path:", out)
