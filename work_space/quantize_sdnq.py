#!/usr/bin/env python3
"""
Generic SDNQ quantization script for any diffusers model.

This script quantizes any diffusers pipeline model to SDNQ (uint4, int8, or fp8) + SVD,
reducing model size while providing near-lossless quality.

Usage:
    python quantize_sdnq.py --repo "FastDM/Wan2.2-T2V-A14B-Merge-Lightning-V1.1-Diffusers" --dtype fp8
    
    # With custom output directory:
    python quantize_sdnq.py --repo "model/repo" --dtype int8 --output-dir "./models/my_quantized_model"
    
    # With custom SVD settings:
    python quantize_sdnq.py --repo "model/repo" --dtype fp8 --svd-rank 64 --svd-steps 16
    
    # With HuggingFace token:
    python quantize_sdnq.py --repo "model/repo" --dtype fp8 --token hf_xxxxx

Requirements:
    pip install sdnq diffusers accelerate transformers safetensors

SDNQ quantization settings:
    - weights_dtype: uint4, int8, or fp8 (default: fp8)
    - svd_rank: SVDQuant residual rank for quality preservation (default: 32)
    - svd_steps: iterations for SVD lowrank estimation (default: 8)
    - use_svd: True (enables SVDQuant on top of SDNQ quantization)
    - quantization_device: cuda (SVD computation requires GPU)
    - return_device: cpu (quantized weights stored on CPU)

Note: This is a one-time operation. After quantization, the model can be loaded
directly using diffusers from_pretrained with SDNQConfig auto-detection.
"""

import os
import sys
import gc
import time
import argparse
import torch

# ============================================================================
# Add local SDNQ source to path (if not installed via pip)
# ============================================================================
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"✓ Added local SDNQ source: {SDNQ_SRC_DIR}")

from diffusers import DiffusionPipeline, AutoPipelineForImageToVideo, AutoPipelineForTextToVideo
from huggingface_hub import snapshot_download

# Import SDNQ (registers quantizer with diffusers/transformers automatically)
from sdnq import sdnq_post_load_quant, SDNQConfig
from sdnq.loader import save_sdnq_model

# ============================================================================
# Default Configuration
# ============================================================================
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
DTYPE = torch.bfloat16

# Default SDNQ quantization settings
DEFAULT_WEIGHTS_DTYPE = "fp8"  # uint4, int8, fp8
DEFAULT_SVD_RANK = 32
DEFAULT_SVD_STEPS = 8
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


def download_model(model_repo, token=None):
    """Download the model locally if not already present."""
    # Create local dir name from repo name
    repo_name = model_repo.split("/")[-1]
    model_local_dir = os.path.join(MODELS_DIR, repo_name)
    
    if os.path.exists(model_local_dir):
        config_file = os.path.join(model_local_dir, "config.json")
        if os.path.exists(config_file):
            print(f"✓ Model already exists at {model_local_dir}")
            return model_local_dir, 0.0
        else:
            print(f"⚠️ Model directory exists but appears incomplete, re-downloading...")

    print(f"⬇️ Downloading {model_repo} to {model_local_dir}...")
    print(f"   ⚠️  WARNING: This may be a large download and will take significant time!")
    os.makedirs(MODELS_DIR, exist_ok=True)

    start_time = time.time()
    try:
        snapshot_download(
            model_repo,
            local_dir=model_local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        elapsed_time = time.time() - start_time
        print(f"✓ Model downloaded successfully to {model_local_dir}")
        print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
        return model_local_dir, elapsed_time
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


def quantize_component(component, component_name, weights_dtype, svd_rank, svd_steps):
    """Quantize a single pipeline component with SDNQ."""
    print(f"\n   🔧 Quantizing {component_name}...")
    print(f"      Settings: {weights_dtype} + SVD rank {svd_rank}")

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
        weights_dtype=weights_dtype,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
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


def load_pipeline(model_path):
    """Load pipeline from local path, auto-detecting pipeline type."""
    print(f"\n📦 Loading pipeline from {model_path}...")
    load_start_time = time.time()

    # Try to auto-detect pipeline type
    # Start with most generic approach
    pipe = None
    last_error = None
    
    # Try AutoPipelineForImageToVideo first
    try:
        pipe = AutoPipelineForImageToVideo.from_pretrained(
            model_path, torch_dtype=DTYPE, low_cpu_mem_usage=True
        )
        print(f"   ✓ Loaded as ImageToVideo pipeline")
    except Exception as e1:
        last_error = e1
        # Try AutoPipelineForTextToVideo
        try:
            pipe = AutoPipelineForTextToVideo.from_pretrained(
                model_path, torch_dtype=DTYPE, low_cpu_mem_usage=True
            )
            print(f"   ✓ Loaded as TextToVideo pipeline")
        except Exception as e2:
            last_error = e2
            # Try generic DiffusionPipeline as fallback
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=DTYPE, low_cpu_mem_usage=True
                )
                print(f"   ✓ Loaded as generic DiffusionPipeline")
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load pipeline. Tried AutoPipelineForImageToVideo, AutoPipelineForTextToVideo, and DiffusionPipeline.\n"
                    f"Last error: {e3}\n"
                    f"Please ensure the model is a valid diffusers pipeline."
                )

    # Fix tied weights for text encoder if present
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
                print("   ✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight")
        if hasattr(te, "tie_weights"):
            te.tie_weights()

    print(f"   ⏱️  Loading time: {format_time(time.time() - load_start_time)}")
    return pipe


def merge_lora(pipe, lora_repo, lora_weights, lora_scale=1.0):
    """Load a LoRA adapter, fuse it into the base model, and unload the adapter."""
    print(f"\n🔗 Merging LoRA into base model...")
    print(f"   Repo:    {lora_repo}")
    print(f"   Weights: {lora_weights}")
    print(f"   Scale:   {lora_scale}")

    merge_start = time.time()

    # Load LoRA weights into the pipeline (uses PEFT under the hood)
    print(f"   Loading LoRA weights...")
    pipe.load_lora_weights(lora_repo, weight_name=lora_weights)
    print(f"   ✓ LoRA weights loaded")

    # Fuse LoRA into base model weights (permanently merges with given scale)
    print(f"   Fusing LoRA (scale={lora_scale})...")
    pipe.fuse_lora(lora_scale=[lora_scale])
    print(f"   ✓ LoRA fused into base weights")

    # Unload PEFT adapter wrappers (fused weights stay, frees adapter memory)
    pipe.unload_lora_weights()
    print(f"   ✓ LoRA adapter wrappers unloaded")

    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merge_time = time.time() - merge_start
    print(f"   ⏱️  LoRA merge time: {format_time(merge_time)}")
    return pipe


def quantize_model(model_path, output_dir, weights_dtype, svd_rank, svd_steps, lora_repo=None, lora_weights=None, lora_scale=1.0):
    """Quantize the model using SDNQ."""
    print("\n" + "=" * 60)
    print("Quantizing Model with SDNQ")
    print("=" * 60)
    print(f"   Quantization: {weights_dtype} + SVD rank {svd_rank}")
    print(f"   SVD steps: {svd_steps}")
    print(f"   Quantization device: {QUANTIZATION_DEVICE}")

    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for SDNQ quantization (SVD computation needs GPU). "
            "Please run this script on a machine with a CUDA-capable GPU."
        )
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load pipeline
    pipe = load_pipeline(model_path)

    # Merge LoRA if requested (before quantization)
    if lora_repo and lora_weights:
        pipe = merge_lora(pipe, lora_repo, lora_weights, lora_scale)

    # Quantize components
    print(f"\n🔧 Quantizing pipeline components with SDNQ...")
    quantize_start_time = time.time()

    # Quantize transformer if present
    if hasattr(pipe, "transformer"):
        pipe.transformer = quantize_component(
            pipe.transformer, "transformer", weights_dtype, svd_rank, svd_steps
        )
        gc.collect()

    # Quantize transformer_2 if present
    if hasattr(pipe, "transformer_2"):
        pipe.transformer_2 = quantize_component(
            pipe.transformer_2, "transformer_2", weights_dtype, svd_rank, svd_steps
        )
        gc.collect()

    # Quantize text_encoder if present
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder = quantize_component(
            pipe.text_encoder, "text_encoder", weights_dtype, svd_rank, svd_steps
        )
        gc.collect()

    # VAE stays in bfloat16 (small, quantization hurts decode quality)
    if hasattr(pipe, "vae"):
        print(f"\n   ✓ VAE kept in bfloat16 (not quantized)")

    quantize_time = time.time() - quantize_start_time
    print(f"\n✓ All components quantized ({format_time(quantize_time)})")

    # Save quantized model
    print(f"\n💾 Saving quantized model to {output_dir}...")
    save_start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # Create SDNQConfig
    sdnq_config = SDNQConfig(
        weights_dtype=weights_dtype,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=USE_SVD,
        add_skip_keys=False,
        modules_to_not_convert=MODULES_TO_NOT_CONVERT.copy(),
    )

    # Save model
    save_sdnq_model(
        pipe,
        output_dir,
        max_shard_size="5GB",
        is_pipeline=True,
        sdnq_config=sdnq_config,
    )

    save_time = time.time() - save_start_time
    print(f"   ✓ Quantized model saved")
    print(f"   ⏱️  Save time: {format_time(save_time)}")

    # Report size comparison
    try:
        original_size = get_dir_size_gb(model_path)
        quantized_size = get_dir_size_gb(output_dir)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"\n📊 Size Comparison:")
        print(f"   Original:  {original_size:.2f} GB")
        print(f"   Quantized: {quantized_size:.2f} GB")
        print(f"   Reduction: {reduction:.1f}%")
    except Exception as e:
        print(f"   ⚠️  Could not calculate size reduction: {e}")

    return output_dir


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantize any diffusers model with SDNQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace model repository (e.g., 'FastDM/Wan2.2-T2V-A14B-Merge-Lightning-V1.1-Diffusers')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint4", "int8", "fp8"],
        default=DEFAULT_WEIGHTS_DTYPE,
        help=f"Quantization dtype: uint4, int8, or fp8 (default: {DEFAULT_WEIGHTS_DTYPE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for quantized model (default: auto-generated from repo name and dtype)",
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=DEFAULT_SVD_RANK,
        help=f"SVD rank for quality preservation (default: {DEFAULT_SVD_RANK})",
    )
    parser.add_argument(
        "--svd-steps",
        type=int,
        default=DEFAULT_SVD_STEPS,
        help=f"SVD steps for lowrank estimation (default: {DEFAULT_SVD_STEPS})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--lora-repo",
        type=str,
        default=None,
        help="HuggingFace repo for LoRA adapter to merge before quantization (optional)",
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="Weight filename within the LoRA repo (required if --lora-repo is set)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA merge strength/scale (default: 1.0)",
    )
    return parser.parse_args()


def main():
    """Main quantization workflow."""
    args = parse_args()

    # Validate LoRA args: if one is set, both must be set
    if args.lora_repo and not args.lora_weights:
        print("❌ --lora-weights is required when --lora-repo is set")
        exit(1)
    if args.lora_weights and not args.lora_repo:
        print("❌ --lora-repo is required when --lora-weights is set")
        exit(1)

    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    
    # Generate output dir if not provided
    if args.output_dir is None:
        repo_name = args.repo.split("/")[-1]
        if args.lora_repo:
            lora_name = args.lora_weights.rsplit(".", 1)[0]  # strip extension
            output_dir = os.path.join(MODELS_DIR, f"{repo_name}_lora_{lora_name}_sdnq_{args.dtype}_svd_r{args.svd_rank}")
        else:
            output_dir = os.path.join(MODELS_DIR, f"{repo_name}_sdnq_{args.dtype}_svd_r{args.svd_rank}")
    else:
        output_dir = args.output_dir

    total_start_time = time.time()

    print("\n" + "=" * 60)
    print("SDNQ Model Quantization Script")
    print("=" * 60)
    print(f"Model:        {args.repo}")
    if args.lora_repo:
        print(f"LoRA:         {args.lora_repo} / {args.lora_weights} (scale={args.lora_scale})")
    print(f"Quantization: SDNQ {args.dtype} + SVD rank {args.svd_rank}")
    print(f"Output:       {output_dir}")
    print("=" * 60)

    try:
        # Download model if needed
        model_path, download_time = download_model(args.repo, token=token)

        # Quantize model (with optional LoRA merge)
        quantized_path = quantize_model(
            model_path, output_dir, args.dtype, args.svd_rank, args.svd_steps,
            lora_repo=args.lora_repo, lora_weights=args.lora_weights, lora_scale=args.lora_scale,
        )

        total_time = time.time() - total_start_time

        print("\n" + "=" * 60)
        print("✅ SDNQ Quantization completed successfully!")
        print("=" * 60)
        print(f"📦 Quantized model saved to: {quantized_path}")
        print(f"⏱️  Total time: {format_time(total_time)}")
        print(f"\n💡 Next steps:")
        print(f"   1. Load with diffusers (auto-detects SDNQ config):")
        print(f"      from sdnq import SDNQConfig  # registers SDNQ quantizer")
        print(f'      pipe = AutoPipelineForImageToVideo.from_pretrained("{quantized_path}")')
        print(f"   2. Or upload to HuggingFace Hub:")
        print(f'      python upload_quantized.py --repo-id "your-username/your-repo" --model-dir "{quantized_path}" --token your_token')
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
