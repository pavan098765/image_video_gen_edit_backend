# z_image_turbo.py
# ✅ Z-Image Turbo Lazy Inference with GPU memory optimizations
# - Based on official Nunchaku z-image-turbo example
# - Keeps lazy loading, caching, offload, and upscaler integration

import gc
import os
import math
from typing import Dict, Any
from PIL import Image
import torch

from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import is_turing
from upscaler import upscale_image

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -------------------------
# Paths / Model config
# -------------------------
Z_IMAGE_TURBO_LOCAL_PATH = os.environ.get("Z_IMAGE_TURBO_LOCAL_PATH")
NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH = os.environ.get("NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH")

if not Z_IMAGE_TURBO_LOCAL_PATH or not os.path.exists(Z_IMAGE_TURBO_LOCAL_PATH):
    raise RuntimeError("❌ Z-Image Turbo repo not found. Ensure it is baked and available at Z_IMAGE_TURBO_LOCAL_PATH.")
if not NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH or not os.path.exists(NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH):
    raise RuntimeError("❌ Nunchaku Z-Image Turbo repo not found. Ensure it is baked and available at NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH.")

# Fixed rank/steps for Turbo model
NUM_INFERENCE_STEPS = 8  # This actually results in 8 DiT forwards
RANK = 128  # Use 32 for faster sampling; 256 (INT4 only) for best quality

# Nunchaku DiT weights: int4 on <25GB VRAM, fp4 on >=25GB VRAM.
# Override with NUNCHAKU_FORCE_PRECISION=int4|fp4 if needed.
_VRAM_INT4_CUTOFF_BYTES = int(
    float(os.environ.get("NUNCHAKU_VRAM_INT4_CUTOFF_GB", "25")) * (1024**3)
)


def _select_nunchaku_precision() -> str:
    o = os.environ.get("NUNCHAKU_FORCE_PRECISION", "").strip().lower()
    if o in ("int4", "fp4"):
        return o
    if torch.cuda.is_available():
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
            if total < _VRAM_INT4_CUTOFF_BYTES:
                return "int4"
            return "fp4"
        except Exception:
            return "int4"
    return "int4"


precision = _select_nunchaku_precision()
dtype = torch.float16 if is_turing() else torch.bfloat16  # float16 for Turing (20-series) GPUs

transformer_file = f"svdq-{precision}_r{RANK}-z-image-turbo.safetensors"
TRANSFORMER_PATH = os.path.join(NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH, transformer_file)

if not os.path.exists(TRANSFORMER_PATH):
    raise FileNotFoundError(f"❌ Transformer file not found at {TRANSFORMER_PATH}")

print(f"✅ Z-Image Turbo config: precision={precision}, rank={RANK}, dtype={dtype}")
print(f"✅ Transformer path: {TRANSFORMER_PATH}")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "transformer": None,
    "pipeline": None,
}


# -------------------------
# Lazy Load Functions
# -------------------------
def get_transformer():
    """Lazily load Nunchaku Z-Image Turbo Transformer."""
    global _model_cache
    if _model_cache["transformer"] is not None:
        return _model_cache["transformer"]

    # Pre-load GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🧠 Loading Nunchaku Z-Image Turbo Transformer model...")
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH, torch_dtype=dtype)
    _model_cache["transformer"] = transformer
    print("✅ Transformer loaded and cached.")
    return transformer


def get_pipeline():
    """Lazily load Z-Image Turbo pipeline with proper offload."""
    global _model_cache
    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"]

    # Pre-load GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 Loading Z-Image Turbo pipeline...")
    transformer = get_transformer()

    pipeline = ZImagePipeline.from_pretrained(
        Z_IMAGE_TURBO_LOCAL_PATH,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    # Enable sequential CPU offload for low VRAM environments
    # Or use .to("cuda") for full GPU mode if you have enough VRAM
    print("⚙️ Enabling sequential CPU offload...")
    pipeline.enable_sequential_cpu_offload()

    _model_cache["pipeline"] = pipeline
    print("✅ Pipeline loaded and cached.")
    return pipeline


def unload_models():
    """Manually clear all cached models and free VRAM."""
    global _model_cache
    _model_cache["transformer"] = None
    _model_cache["pipeline"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Cleared Z-Image Turbo models and freed GPU memory.")


# -------------------------
# Helpers
# -------------------------
def _normalize_dimension(dim: int, default: int = 1024) -> int:
    """Ensure dimension is a positive multiple of 64."""
    if dim <= 0:
        dim = default
    # Round to nearest multiple of 64
    return ((dim + 32) // 64) * 64


def _get_dimensions(params: Dict[str, Any], default_size: int = 1024) -> tuple[int, int]:
    """
    Get width and height from params, ensuring they are valid multiples of 64.
    Falls back to default_size x default_size if not provided.
    """
    width = params.get("width", default_size)
    height = params.get("height", default_size)
    
    # Normalize to multiples of 64
    width = _normalize_dimension(width, default_size)
    height = _normalize_dimension(height, default_size)
    
    return width, height


# -------------------------
# Main Inference Function
# -------------------------
def run_inference_z_image_turbo(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Z-Image Turbo pipeline using the official Nunchaku config.
    Returns local path to PNG output (or upscaled PNG).
    """
    print(f"🧩 Starting Z-Image Turbo inference job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    inference_steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    width, height = _get_dimensions(params, default_size=1024)
    seed = params.get("seed", None)

    pipeline = get_pipeline()

    print(f"⚙️ Running Z-Image Turbo | Steps={inference_steps}, Size={width}x{height}")
    print(f"🪶 Prompt: {prompt}")

    # Build generator with optional seed
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
        print(f"🎲 Using seed: {seed}")

    try:
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=inference_steps,
                guidance_scale=0.0,  # Guidance should be 0 for Turbo models
                generator=generator,
            ).images[0]
    finally:
        # Post-inference GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    out_path = os.path.join(output_dir, f"z_image_turbo_{job_id}.png")
    output.save(out_path)
    print(f"🖼️ Saved generated image to {out_path}")

    # -------------------------
    # Optional Upscaling
    # -------------------------
    scale_image = params.get("scale_image", False)
    if scale_image:
        scale_factor = params.get("scale_factor", 2)
        model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

        try:
            upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)
            upscaled_path = os.path.join(output_dir, f"z_image_turbo_{job_id}_upscaled.png")
            upscaled.save(upscaled_path)
            print(f"🚀 Upscaled image saved to {upscaled_path}")
            return os.path.abspath(upscaled_path)
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}, returning original")
            return os.path.abspath(out_path)

    return os.path.abspath(out_path)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "z_image_test"
    prompt = "a young military male cooking in the kitchen for therapy"

    params = {
        "output_dir": "./outputs",
        "num_inference_steps": 8,
        "width": 1024,
        "height": 1024,
        "seed": 12345,
        "scale_image": False,
    }

    out = run_inference_z_image_turbo(job_id, prompt, params)
    print("✅ Output image saved at:", out)
