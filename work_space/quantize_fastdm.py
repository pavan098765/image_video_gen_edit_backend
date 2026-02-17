#!/usr/bin/env python3
"""
One-time quantization script for FastDM Wan2.2 I2V Lightning model.

This script quantizes the 126GB FastDM model to INT8 (~32GB) using optimum.quanto,
making it usable on RTX 5090 (24GB VRAM) with CPU offloading.

Usage:
    python quantize_fastdm.py

Requirements:
    pip install optimum[quanto] diffusers accelerate transformers

The script:
1. Downloads FastDM model (if not already present)
2. Loads with CPU offloading to prevent OOM on 24GB VRAM
3. Quantizes transformer, transformer_2, and text_encoder to INT8
4. Keeps VAE in bfloat16 (small, quantization hurts quality)
5. Saves quantized model to models/fastdm_wan22_i2v_lightning_int8/

Note: This is a one-time operation. After quantization, use the quantized model
in your inference scripts.
"""

import os
import time
import torch
from diffusers import WanImageToVideoPipeline
from huggingface_hub import snapshot_download
from optimum.quanto import quantize, freeze, qint8

# Configuration
MODEL_REPO = "FastDM/Wan2.2-I2V-A14B-Merge-Lightning-V1.0-Diffusers"
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "fastdm_wan22_i2v_lightning")
QUANTIZED_OUTPUT_DIR = os.path.join(
    MODELS_DIR, "fastdm_wan22_i2v_lightning_int8"
)

DTYPE = torch.bfloat16


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


def download_model():
    """Download the FastDM model locally if not already present."""
    if os.path.exists(MODEL_LOCAL_DIR):
        config_file = os.path.join(MODEL_LOCAL_DIR, "config.json")
        if os.path.exists(config_file):
            print(f"✓ Model already exists at {MODEL_LOCAL_DIR}")
            return MODEL_LOCAL_DIR, 0.0
        else:
            print(f"⚠️ Model directory exists but appears incomplete, re-downloading...")

    print(f"⬇️ Downloading {MODEL_REPO} to {MODEL_LOCAL_DIR}...")
    print(f"   ⚠️  WARNING: This is a 126GB download and will take significant time!")
    os.makedirs(MODELS_DIR, exist_ok=True)

    start_time = time.time()
    try:
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
        )
        elapsed_time = time.time() - start_time
        print(f"✓ Model downloaded successfully to {MODEL_LOCAL_DIR}")
        print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
        return MODEL_LOCAL_DIR, elapsed_time
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


def quantize_model(model_path):
    """Quantize the FastDM model to INT8."""
    print("\n" + "=" * 60)
    print("Quantizing FastDM Model to INT8")
    print("=" * 60)

    print(f"\n📦 Loading pipeline from {model_path}...")
    print(f"   ⚠️  Loading 126GB model on CPU (low_cpu_mem_usage=True)")
    load_start_time = time.time()

    # Load pipeline on CPU with lazy loading to minimize peak memory
    # Note: We do NOT call enable_model_cpu_offload() here because:
    # 1. The model is already on CPU from low_cpu_mem_usage=True
    # 2. CPU offload hooks are for inference, not quantization
    # 3. Hooks could interfere with save_pretrained()
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
    # Quantize components
    # ================================================================
    print(f"\n🔧 Quantizing model components to INT8...")
    quantize_start_time = time.time()

    # Quantize transformer (largest component, ~56GB -> ~14GB)
    print(f"   Quantizing transformer...")
    quantize(pipe.transformer, weights=qint8)
    freeze(pipe.transformer)
    print(f"   ✓ Transformer quantized to INT8")

    # Quantize transformer_2 (second largest, ~56GB -> ~14GB)
    print(f"   Quantizing transformer_2...")
    quantize(pipe.transformer_2, weights=qint8)
    freeze(pipe.transformer_2)
    print(f"   ✓ Transformer_2 quantized to INT8")

    # Quantize text_encoder (~8GB -> ~2GB)
    print(f"   Quantizing text_encoder...")
    quantize(pipe.text_encoder, weights=qint8)
    freeze(pipe.text_encoder)
    print(f"   ✓ Text encoder quantized to INT8")

    # VAE stays in bfloat16 (small, quantization hurts quality)
    print(f"   ✓ VAE kept in bfloat16 (not quantized)")

    quantize_time = time.time() - quantize_start_time
    print(f"\n✓ Quantization completed ({format_time(quantize_time)})")

    # ================================================================
    # Save quantized model
    # ================================================================
    print(f"\n💾 Saving quantized model to {QUANTIZED_OUTPUT_DIR}...")
    save_start_time = time.time()

    os.makedirs(QUANTIZED_OUTPUT_DIR, exist_ok=True)
    pipe.save_pretrained(QUANTIZED_OUTPUT_DIR)
    print(f"✓ Quantized model saved successfully")
    print(f"   ⏱️  Save time: {format_time(time.time() - save_start_time)}")

    # Calculate size reduction
    try:
        original_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(MODEL_LOCAL_DIR)
            for filename in filenames
        ) / (1024**3)

        quantized_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(QUANTIZED_OUTPUT_DIR)
            for filename in filenames
        ) / (1024**3)

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
    print("FastDM Model Quantization Script")
    print("=" * 60)
    print(f"Model: {MODEL_REPO}")
    print(f"Quantization: INT8 (optimum.quanto)")
    print(f"Output: {QUANTIZED_OUTPUT_DIR}")
    print("=" * 60)

    try:
        # Download model if needed
        model_path, download_time = download_model()

        # Quantize model
        quantized_path = quantize_model(model_path)

        total_time = time.time() - total_start_time

        print("\n" + "=" * 60)
        print("✅ Quantization completed successfully!")
        print("=" * 60)
        print(f"📦 Quantized model saved to: {quantized_path}")
        print(f"⏱️  Total time: {format_time(total_time)}")
        print("\n💡 Next steps:")
        print(f"   1. Update your inference script to load from: {quantized_path}")
        print(f"   2. The quantized model should fit in 24GB VRAM with CPU offloading")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Error during quantization:")
        print(f"   {str(e)}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
