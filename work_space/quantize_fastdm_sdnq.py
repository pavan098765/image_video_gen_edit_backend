#!/usr/bin/env python3
"""
One-time SDNQ quantization script for FastDM Wan2.2 T2V Lightning model.

This script quantizes the 126GB FastDM model to SDNQ int8 (int8) + SVD rank 32,
reducing it to ~20-28 GB on disk while providing near-lossless quality.

Usage:
    python quantize_fastdm_sdnq.py

    # Optional: specify quantization settings via env vars
    SDNQ_WEIGHTS_DTYPE=int8 SDNQ_SVD_RANK=32 python quantize_fastdm_sdnq.py

Requirements:
    pip install sdnq diffusers accelerate transformers safetensors

The script:
1. Downloads FastDM model (if not already present)
2. Loads pipeline on CPU (low_cpu_mem_usage=True)
3. Quantizes transformer, transformer_2, and text_encoder with SDNQ int8 + SVD
4. Keeps VAE in bfloat16 (small, quantization hurts quality)
5. Saves quantized model using save_sdnq_model (is_pipeline=True)

SDNQ quantization settings:
    - weights_dtype: int8 (8-bit floating point quantization, near-lossless)
    - svd_rank: 32 (SVDQuant residual rank for quality preservation)
    - svd_steps: 8 (iterations for SVD lowrank estimation)
    - use_svd: True (enables SVDQuant on top of SDNQ quantization)
    - quantization_device: cuda (SVD computation requires GPU)
    - return_device: cpu (quantized weights stored on CPU)

Note: This is a one-time operation. After quantization, the model can be loaded
directly using diffusers from_pretrained with SDNQConfig auto-detection.
The int8 quantized model should fit in 32GB VRAM with CPU offloading.
"""

import os
import sys
import gc
import time
import torch

# ============================================================================
# Add local SDNQ source to path (if not installed via pip)
# ============================================================================
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"✓ Added local SDNQ source: {SDNQ_SRC_DIR}")

from diffusers import WanImageToVideoPipeline
from huggingface_hub import snapshot_download

# Import SDNQ (registers quantizer with diffusers/transformers automatically)
from sdnq import sdnq_post_load_quant, SDNQConfig
from sdnq.loader import save_sdnq_model

# ============================================================================
# Configuration
# ============================================================================
MODEL_REPO = "FastDM/Wan2.2-T2V-A14B-Merge-Lightning-V1.1-Diffusers"
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning")
QUANTIZED_OUTPUT_DIR = os.path.join(
    MODELS_DIR, "fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32"
)

DTYPE = torch.bfloat16

# SDNQ quantization settings
WEIGHTS_DTYPE = os.environ.get("SDNQ_WEIGHTS_DTYPE", "fp8") #uint4, int8, fp8
SVD_RANK = int(os.environ.get("SDNQ_SVD_RANK", "32"))
SVD_STEPS = int(os.environ.get("SDNQ_SVD_STEPS", "8"))
USE_SVD = True  # SVD correction improves quality for all quantization types
QUANTIZATION_DEVICE = "cuda"  # SVD requires GPU
RETURN_DEVICE = "cpu"  # Store quantized weights on CPU

# Modules that should NOT be quantized (keep in full precision)
MODULES_TO_NOT_CONVERT = [
    "correction_coefs",
    "prediction_coefs",
    "lm_head",
    "embedding_projection",
]


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def get_model_size_gb(model):
    """Calculate approximate model size in GB from parameters."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024**3)


def get_dir_size_gb(directory):
    """Calculate directory size in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total += os.path.getsize(filepath)
    return total / (1024**3)


def download_model():
    """Download the FastDM model locally if not already present."""
    if os.path.exists(MODEL_LOCAL_DIR):
        config_file = os.path.join(MODEL_LOCAL_DIR, "config.json")
        if os.path.exists(config_file):
            print(f"✓ Model already exists at {MODEL_LOCAL_DIR}")
            return MODEL_LOCAL_DIR, 0.0
        else:
            print(
                f"⚠️ Model directory exists but appears incomplete, re-downloading..."
            )

    print(f"⬇️ Downloading {MODEL_REPO} to {MODEL_LOCAL_DIR}...")
    print(f"   ⚠️  WARNING: This is a 126GB download and will take significant time!")
    os.makedirs(MODELS_DIR, exist_ok=True)

    start_time = time.time()
    try:
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
            token="hf_SOmwbGayIEoFqgofeaGmwCgUWNDNzLqnRly",
        )
        elapsed_time = time.time() - start_time
        print(f"✓ Model downloaded successfully to {MODEL_LOCAL_DIR}")
        print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
        return MODEL_LOCAL_DIR, elapsed_time
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


def quantize_component(component, component_name):
    """Quantize a single pipeline component with SDNQ."""
    print(f"\n   🔧 Quantizing {component_name}...")
    print(f"      Settings: {WEIGHTS_DTYPE} + SVD rank {SVD_RANK}")

    pre_size = get_model_size_gb(component)
    print(f"      Original size: {pre_size:.2f} GB (in-memory)")

    start_time = time.time()

    # sdnq_post_load_quant quantizes each layer:
    #   1. Moves layer weight to CUDA (for SVD computation)
    #   2. Computes SVD low-rank decomposition (rank=SVD_RANK)
    #   3. Quantizes residual to WEIGHTS_DTYPE (e.g. int8)
    #   4. Moves quantized weight back to CPU
    # Peak VRAM usage: one layer weight at a time (~100-500 MB)
    component = sdnq_post_load_quant(
        component,
        weights_dtype=WEIGHTS_DTYPE,
        svd_rank=SVD_RANK,
        svd_steps=SVD_STEPS,
        use_svd=USE_SVD,
        torch_dtype=DTYPE,
        quantization_device=QUANTIZATION_DEVICE,
        return_device=RETURN_DEVICE,
        modules_to_not_convert=MODULES_TO_NOT_CONVERT.copy(),
        add_skip_keys=True,
    )

    elapsed = time.time() - start_time
    post_size = get_model_size_gb(component)
    reduction = (1 - post_size / pre_size) * 100 if pre_size > 0 else 0

    print(f"      ✓ Quantized: {pre_size:.2f} GB → {post_size:.2f} GB ({reduction:.1f}% reduction)")
    print(f"      ⏱️  Time: {format_time(elapsed)}")

    # Free CUDA cache after each component
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return component


def quantize_model(model_path):
    """Quantize the FastDM model using SDNQ."""
    print("\n" + "=" * 60)
    print("Quantizing FastDM Model with SDNQ")
    print("=" * 60)
    print(f"   Quantization: {WEIGHTS_DTYPE} + SVD rank {SVD_RANK}")
    print(f"   SVD steps: {SVD_STEPS}")
    print(f"   Quantization device: {QUANTIZATION_DEVICE}")

    # ================================================================
    # Verify CUDA is available (required for SVD computation)
    # ================================================================
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for SDNQ quantization (SVD computation needs GPU). "
            "Please run this script on a machine with a CUDA-capable GPU."
        )
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ================================================================
    # Load pipeline on CPU
    # ================================================================
    print(f"\n📦 Loading pipeline from {model_path}...")
    print(f"   ⚠️  Loading 126GB model on CPU (requires ~130GB system RAM)")
    load_start_time = time.time()

    # Load on CPU with lazy loading. Do NOT call enable_model_cpu_offload()
    # as it installs inference hooks we don't need for quantization.
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )

    # Fix tied weights for text encoder before quantization
    # FastDM model's text_encoder uses UMT5EncoderModel where encoder.embed_tokens.weight
    # should be tied to shared.weight, but diffusers.from_pretrained() doesn't always handle this.
    # This ensures correct weights before quantization.
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if (
            hasattr(te, "shared")
            and hasattr(te, "encoder")
            and hasattr(te.encoder, "embed_tokens")
        ):
            shared_ptr = te.shared.weight.data_ptr()
            embed_ptr = te.encoder.embed_tokens.weight.data_ptr()
            if shared_ptr != embed_ptr:
                te.encoder.embed_tokens.weight = te.shared.weight
                print(
                    "   ✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight"
                )
            else:
                print("   ✓ Text encoder weights already tied correctly")
        # Also call tie_weights() as a safety net
        if hasattr(te, "tie_weights"):
            te.tie_weights()

    print(f"   ✓ Pipeline loaded on CPU")
    print(f"   ⏱️  Loading time: {format_time(time.time() - load_start_time)}")

    # ================================================================
    # Quantize components one at a time
    # ================================================================
    print(f"\n🔧 Quantizing pipeline components with SDNQ...")
    quantize_start_time = time.time()

    # Quantize transformer (largest component, ~56GB → ~10-13GB with int8)
    pipe.transformer = quantize_component(pipe.transformer, "transformer")
    gc.collect()

    # Quantize transformer_2 (second largest, ~56GB → ~10-13GB with int8)
    pipe.transformer_2 = quantize_component(pipe.transformer_2, "transformer_2")
    gc.collect()

    # Quantize text_encoder (~8GB → ~2-3GB with int8)
    pipe.text_encoder = quantize_component(pipe.text_encoder, "text_encoder")
    gc.collect()

    # VAE stays in bfloat16 (small, quantization hurts decode quality)
    print(f"\n   ✓ VAE kept in bfloat16 (not quantized)")

    quantize_time = time.time() - quantize_start_time
    print(f"\n✓ All components quantized ({format_time(quantize_time)})")

    # ================================================================
    # Save quantized model using SDNQ's save function
    # ================================================================
    print(f"\n💾 Saving quantized model to {QUANTIZED_OUTPUT_DIR}...")
    save_start_time = time.time()

    os.makedirs(QUANTIZED_OUTPUT_DIR, exist_ok=True)

    # Create an SDNQConfig that matches our quantization settings
    # This will be saved alongside the model for auto-detection on load
    sdnq_config = SDNQConfig(
        weights_dtype=WEIGHTS_DTYPE,
        svd_rank=SVD_RANK,
        svd_steps=SVD_STEPS,
        use_svd=USE_SVD,
        add_skip_keys=False,  # Already applied during quantization
        modules_to_not_convert=MODULES_TO_NOT_CONVERT.copy(),
    )

    # save_sdnq_model handles per-component quantization_config saving
    save_sdnq_model(
        pipe,
        QUANTIZED_OUTPUT_DIR,
        max_shard_size="5GB",
        is_pipeline=True,
        sdnq_config=sdnq_config,
    )

    save_time = time.time() - save_start_time
    print(f"   ✓ Quantized model saved")
    print(f"   ⏱️  Save time: {format_time(save_time)}")

    # ================================================================
    # Report size comparison
    # ================================================================
    try:
        original_size = get_dir_size_gb(MODEL_LOCAL_DIR)
        quantized_size = get_dir_size_gb(QUANTIZED_OUTPUT_DIR)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"\n📊 Size Comparison:")
        print(f"   Original:  {original_size:.2f} GB")
        print(f"   Quantized: {quantized_size:.2f} GB")
        print(f"   Reduction: {reduction:.1f}%")
    except Exception as e:
        print(f"   ⚠️  Could not calculate size reduction: {e}")

    return QUANTIZED_OUTPUT_DIR


def main():
    """Main quantization workflow."""
    total_start_time = time.time()

    print("\n" + "=" * 60)
    print("FastDM Model SDNQ Quantization Script")
    print("=" * 60)
    print(f"Model:        {MODEL_REPO}")
    print(f"Quantization: SDNQ {WEIGHTS_DTYPE} + SVD rank {SVD_RANK}")
    print(f"Output:       {QUANTIZED_OUTPUT_DIR}")
    print("=" * 60)

    try:
        # Download model if needed
        model_path, download_time = download_model()

        # Quantize model
        quantized_path = quantize_model(model_path)

        total_time = time.time() - total_start_time

        print("\n" + "=" * 60)
        print("✅ SDNQ Quantization completed successfully!")
        print("=" * 60)
        print(f"📦 Quantized model saved to: {quantized_path}")
        print(f"⏱️  Total time: {format_time(total_time)}")
        print(f"\n💡 Next steps:")
        print(f"   1. Load with diffusers (auto-detects SDNQ config):")
        print(f"      from sdnq import SDNQConfig  # registers SDNQ quantizer")
        print(f'      pipe = WanImageToVideoPipeline.from_pretrained("{quantized_path}")')
        print(f"   2. Or upload to HuggingFace Hub:")
        print(f"      python upload_quantized_fastdm.py")
        print(f"   3. The quantized model should fit in 32GB VRAM with CPU offloading")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Error during SDNQ quantization:")
        print(f"   {str(e)}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
