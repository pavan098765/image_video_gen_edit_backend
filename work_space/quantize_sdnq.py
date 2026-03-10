#!/usr/bin/env python3
"""
SDNQ quantization script for Wan diffusers models (I2V, T2V).

Quantizes Wan pipeline models to SDNQ (uint4, int8, or fp8) + SVD,
reducing model size while preserving quality.

Usage:
    # Local model, no LoRA:
    python quantize_sdnq.py --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" --dtype fp8 --svd-rank 64 --svd-steps 32

    # Local model + merge pre-converted diffusers LoRA (run convert_wan_lora_to_diffusers.py first):
    python quantize_sdnq.py \
        --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" \
        --dtype fp8 --svd-rank 64 --svd-steps 32 \
        --lora-t1 "/work_space/models/loras/DR34ML4Y_I2V_14B_HIGH_V2_diffusers.safetensors" \
        --lora-t2 "/work_space/models/loras/DR34ML4Y_I2V_14B_LOW_V2_diffusers.safetensors" \
        --lora-strength 0.8

    # From HuggingFace repo:
    python quantize_sdnq.py --repo "Wan-AI/Wan2.2-I2V-A14B-Diffusers" --dtype fp8 --svd-rank 64 --svd-steps 16


    All four usage patterns now work:
    Local files (as before):
    python3 quantize_sdnq.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1 /work_space/models/loras/HIGH_diffusers.safetensors \
        --lora-t2 /work_space/models/loras/LOW_diffusers.safetensors \
        --lora-strength 0.8
    
    HF model + HF LoRAs:
    python3 quantize_sdnq.py --repo Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype fp8 \
        --lora-t1-repo profpeng/nsfwv2 --lora-t1-file DR34ML4Y_I2V_14B_HIGH_V2.safetensors \
        --lora-t2-repo profpeng/nsfwv2 --lora-t2-file DR34ML4Y_I2V_14B_LOW_V2.safetensors \
        --lora-strength 0.8
    
    HF model + local LoRAs (mixed):
    python3 quantize_sdnq.py --repo Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype fp8 \
        --lora-t1 /work_space/models/loras/HIGH_diffusers.safetensors \
        --lora-t2 /work_space/models/loras/LOW_diffusers.safetensors
    
    Single LoRA applied to both transformers (omit --lora-t2*):
    python3 quantize_sdnq.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1 /work_space/models/loras/HIGH_diffusers.safetensors

Requirements:
    pip install sdnq diffusers accelerate transformers safetensors
    # For Civitai LoRA download: pip install requests

    Civitai LoRA (use model version ID from the Civitai download URL):
    python3 quantize_sdnq.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1-civitai 123456 --lora-t2-civitai 789012 \
        --civitai-token YOUR_CIVITAI_API_KEY

Notes:
    - LoRAs must be pre-converted to diffusers format using convert_wan_lora_to_diffusers.py
    - --lora-t1 targets transformer (high lighting), --lora-t2 targets transformer_2 (low lighting)
    - Use --lora-t1 only (omit --lora-t2) to apply one LoRA to both transformers
    - LoRA is permanently baked into model weights before quantization
"""

import gc
import os
import sys
import time
import argparse
import torch
from safetensors.torch import load_file

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ============================================================================
# Add local SDNQ source to path (if not installed via pip)
# ============================================================================
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"Added local SDNQ source: {SDNQ_SRC_DIR}")

from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

WanImageToVideoPipeline = None
WanPipeline = None
try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    try:
        from diffusers.pipelines.wan import WanImageToVideoPipeline
    except ImportError:
        pass
try:
    from diffusers import WanPipeline
except ImportError:
    try:
        from diffusers.pipelines.wan import WanPipeline
    except ImportError:
        pass

if WanImageToVideoPipeline is None:
    raise ImportError(
        "WanImageToVideoPipeline not found. Upgrade: pip install -U diffusers"
    )

from sdnq import sdnq_post_load_quant, SDNQConfig
from sdnq.loader import save_sdnq_model

# ============================================================================
# Configuration
# ============================================================================
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
DTYPE = torch.bfloat16

DEFAULT_WEIGHTS_DTYPE = "fp8"
DEFAULT_SVD_RANK = 32
DEFAULT_SVD_STEPS = 8
USE_SVD = True
QUANTIZATION_DEVICE = "cuda"
RETURN_DEVICE = "cpu"

MODULES_TO_NOT_CONVERT = [
    "correction_coefs",
    "prediction_coefs",
    "lm_head",
    "embedding_projection",
]

# ---------------------------------------------------------------------------
# ComfyUI -> diffusers key remapping (same as convert_wan_lora_to_diffusers.py)
# ---------------------------------------------------------------------------
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

LORA_SUFFIXES = [
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight", ".alpha",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    else:
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), seconds % 60
        return f"{h}h {m}m {s:.2f}s"


def get_model_size_gb(model):
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 3)


def get_dir_size_gb(directory):
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


# ---------------------------------------------------------------------------
# LoRA key remapping
# ---------------------------------------------------------------------------
def remap_lora_state_dict(state_dict: dict) -> dict:
    """Remap ComfyUI LoRA keys to diffusers format."""
    PREFIX = "diffusion_model."
    new_sd = {}
    for key, val in state_dict.items():
        k = key[len(PREFIX):] if key.startswith(PREFIX) else key
        lora_sfx = ""
        for sfx in LORA_SUFFIXES:
            if k.endswith(sfx):
                lora_sfx = sfx
                k = k[: -len(sfx)]
                break
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            k = k.replace(old, new)
        new_sd[k + lora_sfx] = val
    return new_sd


# ---------------------------------------------------------------------------
# LoRA injection (proven approach — same as inference script)
# ---------------------------------------------------------------------------
def inject_lora(model: torch.nn.Module, lora_path: str, strength: float, label: str = ""):
    """
    Permanently bake LoRA delta weights into model parameters.

    Supports both ComfyUI format (auto-remapped) and pre-converted diffusers format.
    Uses bfloat16 matmul to minimise RAM usage. Frees tensors immediately after use.
    """
    tag = f" ({label})" if label else ""
    print(f"   Injecting LoRA{tag}: {os.path.basename(lora_path)} strength={strength}")

    try:
        raw_sd = load_file(lora_path, device="cpu")
    except Exception as e:
        print(f"   ERROR: Could not load LoRA: {e}")
        return

    is_comfy = any(k.startswith("diffusion_model.") for k in raw_sd)
    if is_comfy:
        print(f"   ComfyUI format detected — remapping {len(raw_sd)} keys...")
        sd = remap_lora_state_dict(raw_sd)
        del raw_sd
        gc.collect()
        print(f"   Remapped to {len(sd)} diffusers keys")
    else:
        sd = raw_sd
        print(f"   Diffusers format — {len(sd)} keys (no remapping needed)")

    # Build pairs: base_key -> {down, up, alpha?}
    pairs: dict = {}
    for k, v in sd.items():
        if k.endswith(".lora_down.weight") or k.endswith(".lora_A.weight"):
            pairs.setdefault(k[: k.rfind(".lora_")], {})["down"] = v
        elif k.endswith(".lora_up.weight") or k.endswith(".lora_B.weight"):
            pairs.setdefault(k[: k.rfind(".lora_")], {})["up"] = v
        elif k.endswith(".alpha"):
            pairs.setdefault(k[: -len(".alpha")], {})["alpha"] = v

    del sd
    gc.collect()

    model_params = dict(model.named_parameters())
    total = len(pairs)
    applied = skipped_missing = skipped_incomplete = 0

    for i, (base_key, parts) in enumerate(pairs.items()):
        if i % 50 == 0:
            print(f"   Baking... {i}/{total}", end="\r", flush=True)

        if "down" not in parts or "up" not in parts:
            skipped_incomplete += 1
            continue

        weight_key = base_key + ".weight"
        if weight_key not in model_params:
            skipped_missing += 1
            continue

        param = model_params[weight_key]
        dtype = param.dtype  # bfloat16

        down = parts["down"].to(dtype)
        up   = parts["up"].to(dtype)
        rank = down.shape[0]
        scale = (parts["alpha"].float().item() / rank) if "alpha" in parts else 1.0

        if down.dim() == 4: down = down.squeeze(3).squeeze(2)
        if up.dim()   == 4: up   = up.squeeze(3).squeeze(2)

        delta = (up @ down) * (scale * strength)

        if param.shape != delta.shape:
            try:
                delta = delta.view(param.shape)
            except Exception:
                del down, up, delta
                skipped_missing += 1
                continue

        with torch.no_grad():
            param.data.add_(delta)

        del down, up, delta
        applied += 1

    del pairs, model_params
    gc.collect()

    print()
    print(f"   Result: {applied}/{total} layers baked | "
          f"{skipped_missing} not in model | {skipped_incomplete} incomplete")
    if applied == 0:
        print(f"   ERROR: No weights updated — check LoRA format/compatibility")
    elif applied < total * 0.8:
        print(f"   WARNING: Only {applied}/{total} layers applied")
    else:
        print(f"   LoRA permanently baked into model weights")


def merge_loras_into_pipe(pipe, lora_t1_path, lora_t2_path, strength):
    """
    Bake LoRA weights into transformer and transformer_2 before quantization.

    Uses manual injection (proven to work) instead of diffusers PEFT fuse_lora()
    which silently fails on ComfyUI-format LoRAs and has issues with dual transformers.
    """
    print(f"\nBaking LoRAs into model weights (strength={strength})...")

    if lora_t1_path and hasattr(pipe, "transformer") and pipe.transformer is not None:
        inject_lora(pipe.transformer, lora_t1_path, strength, "transformer")
        gc.collect()

    # If lora_t2_path not specified, apply lora_t1 to transformer_2 as well
    effective_t2_path = lora_t2_path or lora_t1_path
    if effective_t2_path and hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        inject_lora(pipe.transformer_2, effective_t2_path, strength, "transformer_2")
        gc.collect()

    print("LoRA baking complete")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_model(model_repo, token=None):
    repo_name = model_repo.split("/")[-1]
    model_local_dir = os.path.join(MODELS_DIR, repo_name)

    if os.path.exists(model_local_dir):
        if os.path.exists(os.path.join(model_local_dir, "model_index.json")):
            print(f"Model already exists at {model_local_dir}")
            return model_local_dir, 0.0
        print(f"Model directory exists but appears incomplete, re-downloading...")

    print(f"Downloading {model_repo} to {model_local_dir}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    start = time.time()
    try:
        snapshot_download(model_repo, local_dir=model_local_dir,
                          local_dir_use_symlinks=False, token=token)
        elapsed = time.time() - start
        print(f"Model downloaded in {format_time(elapsed)}")
        return model_local_dir, elapsed
    except Exception as e:
        print(f"Download failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def load_pipeline(model_path):
    print(f"\nLoading pipeline from {model_path}...")
    start = time.time()
    pipe = None

    for PipeClass, name in [
        (WanImageToVideoPipeline, "WanImageToVideoPipeline"),
        (WanPipeline, "WanPipeline"),
        (DiffusionPipeline, "DiffusionPipeline"),
    ]:
        if PipeClass is None:
            continue
        try:
            pipe = PipeClass.from_pretrained(model_path, torch_dtype=DTYPE, low_cpu_mem_usage=True)
            print(f"   Loaded as {name}")
            break
        except Exception:
            continue

    if pipe is None:
        raise RuntimeError(f"Could not load pipeline from {model_path}")

    # Fix UMT5 weight tying
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if hasattr(te, "shared") and hasattr(te, "encoder") and hasattr(te.encoder, "embed_tokens"):
            if te.shared.weight.data_ptr() != te.encoder.embed_tokens.weight.data_ptr():
                te.encoder.embed_tokens.weight = te.shared.weight
                print("   Fixed: re-tied text encoder shared weights")
        if hasattr(te, "tie_weights"):
            te.tie_weights()

    print(f"   Pipeline loaded in {format_time(time.time() - start)}")
    return pipe


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_component(component, component_name, weights_dtype, svd_rank, svd_steps):
    print(f"\n   Quantizing {component_name} ({weights_dtype} + SVD rank {svd_rank})...")
    pre_size = get_model_size_gb(component)
    print(f"   Original size: {pre_size:.2f} GB")

    start = time.time()
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

    elapsed = time.time() - start
    post_size = get_model_size_gb(component)
    reduction = (1 - post_size / pre_size) * 100 if pre_size > 0 else 0
    print(f"   Quantized: {pre_size:.2f} GB -> {post_size:.2f} GB ({reduction:.1f}% reduction) in {format_time(elapsed)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return component


def quantize_model(model_path, output_dir, weights_dtype, svd_rank, svd_steps,
                   lora_t1_path=None, lora_t2_path=None, lora_strength=1.0):
    print("\n" + "=" * 60)
    print("Quantizing with SDNQ")
    print("=" * 60)
    print(f"   Quantization: {weights_dtype} + SVD rank {svd_rank}, steps {svd_steps}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SDNQ quantization (SVD needs GPU)")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_pipeline(model_path)

    # Bake LoRAs permanently BEFORE quantization
    if lora_t1_path or lora_t2_path:
        merge_loras_into_pipe(pipe, lora_t1_path, lora_t2_path, lora_strength)

    # Quantize
    print(f"\nQuantizing components...")
    q_start = time.time()

    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer = quantize_component(
            pipe.transformer, "transformer", weights_dtype, svd_rank, svd_steps)
        gc.collect()

    if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        pipe.transformer_2 = quantize_component(
            pipe.transformer_2, "transformer_2", weights_dtype, svd_rank, svd_steps)
        gc.collect()

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = quantize_component(
            pipe.text_encoder, "text_encoder", weights_dtype, svd_rank, svd_steps)
        gc.collect()

    if hasattr(pipe, "vae"):
        print(f"\n   VAE kept in bfloat16 (not quantized)")

    print(f"\nAll components quantized in {format_time(time.time() - q_start)}")

    # Save
    print(f"\nSaving quantized model to {output_dir}...")
    save_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    sdnq_config = SDNQConfig(
        weights_dtype=weights_dtype,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=USE_SVD,
        add_skip_keys=False,
        modules_to_not_convert=MODULES_TO_NOT_CONVERT.copy(),
    )
    save_sdnq_model(pipe, output_dir, max_shard_size="5GB",
                    is_pipeline=True, sdnq_config=sdnq_config)

    print(f"   Saved in {format_time(time.time() - save_start)}")

    try:
        orig_size = get_dir_size_gb(model_path)
        quant_size = get_dir_size_gb(output_dir)
        print(f"\nSize: {orig_size:.2f} GB -> {quant_size:.2f} GB "
              f"({(1 - quant_size/orig_size)*100:.1f}% reduction)")
    except Exception:
        pass

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize Wan diffusers model with SDNQ, with optional LoRA baking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model source
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", type=str, help="HuggingFace model repo ID")
    src.add_argument("--model-dir", type=str, help="Local model directory path")

    # Quantization
    parser.add_argument("--dtype", choices=["uint4", "int8", "fp8"],
                        default=DEFAULT_WEIGHTS_DTYPE)
    parser.add_argument("--svd-rank", type=int, default=DEFAULT_SVD_RANK)
    parser.add_argument("--svd-steps", type=int, default=DEFAULT_SVD_STEPS)
    parser.add_argument("--output-dir", type=str, default=None)

    # LoRA for transformer (high lighting)
    # Use --lora-t1 for local file OR --lora-t1-repo + --lora-t1-file to download from HF.
    # If no t2 source is given, t1 is also applied to transformer_2.
    parser.add_argument("--lora-t1", type=str, default=None,
                        help="Local path to LoRA for transformer.")
    parser.add_argument("--lora-t1-repo", type=str, default=None,
                        help="HuggingFace repo for transformer LoRA (e.g. profpeng/nsfwv2).")
    parser.add_argument("--lora-t1-file", type=str, default=None,
                        help="Filename to download from --lora-t1-repo.")

    # LoRA for transformer_2 (low lighting)
    # Use --lora-t2 for local file OR --lora-t2-repo + --lora-t2-file to download from HF.
    parser.add_argument("--lora-t2", type=str, default=None,
                        help="Local path to LoRA for transformer_2. Falls back to t1 if not set.")
    parser.add_argument("--lora-t2-repo", type=str, default=None,
                        help="HuggingFace repo for transformer_2 LoRA.")
    parser.add_argument("--lora-t2-file", type=str, default=None,
                        help="Filename to download from --lora-t2-repo.")

    # Civitai LoRA (model version ID from civitai.com download URL)
    parser.add_argument("--lora-t1-civitai", type=str, default=None,
                        help="Civitai model version ID for transformer LoRA (from download URL).")
    parser.add_argument("--lora-t2-civitai", type=str, default=None,
                        help="Civitai model version ID for transformer_2 LoRA.")
    parser.add_argument("--civitai-token", type=str, default=None,
                        help="Civitai API token (or set CIVITAI_API_KEY env var). Required for gated/NSFW models.")

    parser.add_argument("--lora-strength", type=float, default=1.0,
                        help="LoRA bake strength applied to both transformers (default: 1.0)")

    # Auth
    parser.add_argument("--token", type=str, default=None)

    return parser.parse_args()



def download_lora_from_civitai(model_id: str, save_dir: str, api_token: str, label: str) -> str | None:
    """
    Download a LoRA from Civitai by model version ID.
    URL format: https://civitai.com/api/download/models/{model_id}
    """
    if not REQUESTS_AVAILABLE:
        print(f"   {label}: Civitai download requires 'pip install requests'")
        return None
    if not api_token:
        print(f"   {label}: Civitai requires --civitai-token or CIVITAI_API_KEY env var")
        return None

    os.makedirs(save_dir, exist_ok=True)

    # Use Content-Disposition filename if available, else fallback
    url = f"https://civitai.com/api/download/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    print(f"   {label}: downloading from Civitai (model {model_id})...")
    dest = None
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        # Get filename from Content-Disposition header if present
        content_disp = response.headers.get("content-disposition", "")
        filename = None
        if "filename=" in content_disp:
            part = content_disp.split("filename=")[-1].strip().strip('"\'')
            if part:
                filename = part
        if not filename or not filename.endswith((".safetensors", ".pt", ".bin")):
            filename = f"civitai_{model_id}.safetensors"

        dest = os.path.join(save_dir, os.path.basename(filename))

        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 ** 2)
            print(f"   {label}: already downloaded at {dest} ({size_mb:.0f} MB)")
            return dest

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (5 * 1024 * 1024) < 8192:
                        pct = 100 * downloaded / total_size
                        print(f"   {label}: downloading... {pct:.0f}%", end="\r", flush=True)
        print()
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"   {label}: saved to {dest} ({size_mb:.0f} MB)")
        return dest
    except requests.RequestException as e:
        print(f"   {label}: Civitai download failed: {e}")
        try:
            if dest and os.path.exists(dest):
                os.remove(dest)
        except (NameError, OSError):
            pass
        return None


def resolve_lora_path(local_path, hf_repo, hf_file, civitai_id, save_dir, token,
                     civitai_token, label):
    """Resolve LoRA to a local path — local, HuggingFace, or Civitai."""
    if local_path and os.path.exists(local_path):
        print(f"   {label}: local file found at {local_path}")
        return local_path
    if local_path and not os.path.exists(local_path):
        print(f"   {label}: local file not found at {local_path}")

    if hf_repo and hf_file:
        os.makedirs(save_dir, exist_ok=True)
        dest = os.path.join(save_dir, os.path.basename(hf_file))
        if os.path.exists(dest):
            print(f"   {label}: already downloaded at {dest}")
            return dest
        print(f"   {label}: downloading {hf_repo}/{hf_file}...")
        from huggingface_hub import hf_hub_download
        import shutil
        downloaded = hf_hub_download(
            repo_id=hf_repo, filename=hf_file,
            local_dir=save_dir, local_dir_use_symlinks=False, token=token,
        )
        if os.path.abspath(downloaded) != os.path.abspath(dest):
            shutil.move(downloaded, dest)
        print(f"   {label}: saved to {dest}")
        return dest

    if civitai_id:
        return download_lora_from_civitai(civitai_id, save_dir, civitai_token, label)

    if local_path:
        print(f"   {label}: no HF/Civitai fallback specified — skipping")
    return None


def main():
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    civitai_token = args.civitai_token or os.getenv("CIVITAI_API_KEY")

    # Validate HF LoRA args: repo requires file and vice versa
    for repo, file, name in [
        (args.lora_t1_repo, args.lora_t1_file, "t1"),
        (args.lora_t2_repo, args.lora_t2_file, "t2"),
    ]:
        if repo and not file:
            print(f"ERROR: --lora-{name}-repo requires --lora-{name}-file"); exit(1)
        if file and not repo:
            print(f"ERROR: --lora-{name}-file requires --lora-{name}-repo"); exit(1)

    # Resolve model path
    if args.model_dir:
        model_path = os.path.abspath(args.model_dir)
        if not os.path.isdir(model_path):
            print(f"ERROR: Model directory not found: {model_path}"); exit(1)
        model_name = os.path.basename(model_path.rstrip(os.sep))
    else:
        model_path, _ = download_model(args.repo, token=token)
        model_name = args.repo.split("/")[-1]

    # Resolve LoRA paths (local, HF, or Civitai)
    lora_save_dir = os.path.join(MODELS_DIR, "loras")
    print("\nResolving LoRA sources...")
    lora_t1_path = resolve_lora_path(
        args.lora_t1, args.lora_t1_repo, args.lora_t1_file,
        args.lora_t1_civitai, lora_save_dir, token, civitai_token, "LoRA t1")
    lora_t2_path = resolve_lora_path(
        args.lora_t2, args.lora_t2_repo, args.lora_t2_file,
        args.lora_t2_civitai, lora_save_dir, token, civitai_token, "LoRA t2")

    # Auto output dir
    if args.output_dir is None:
        lora_tag = ""
        effective_t1 = lora_t1_path or lora_t2_path
        if effective_t1:
            lora_name = os.path.splitext(os.path.basename(effective_t1))[0]
            lora_tag = f"_lora_{lora_name}"
        output_dir = os.path.join(
            MODELS_DIR, f"{model_name}{lora_tag}_sdnq_{args.dtype}_svd_r{args.svd_rank}")
    else:
        output_dir = args.output_dir

    has_lora = lora_t1_path or lora_t2_path
    print("\n" + "=" * 60)
    print("SDNQ Quantization")
    print("=" * 60)
    print(f"   Model     : {args.model_dir or args.repo}")
    if has_lora:
        print(f"   LoRA t1   : {lora_t1_path or '(none)'}")
        print(f"   LoRA t2   : {lora_t2_path or '(same as t1)'}")
        print(f"   LoRA str  : {args.lora_strength}")
    print(f"   Quantize  : {args.dtype} + SVD rank {args.svd_rank}, steps {args.svd_steps}")
    print(f"   Output    : {output_dir}")
    print("=" * 60)

    total_start = time.time()
    try:
        quantized_path = quantize_model(
            model_path, output_dir, args.dtype, args.svd_rank, args.svd_steps,
            lora_t1_path=lora_t1_path,
            lora_t2_path=lora_t2_path,
            lora_strength=args.lora_strength,
        )
        print("\n" + "=" * 60)
        print("Quantization complete!")
        print("=" * 60)
        print(f"   Output  : {quantized_path}")
        print(f"   Total   : {format_time(time.time() - total_start)}")
        print(f"\nTo use:")
        print(f"   from sdnq import SDNQConfig")
        print(f"   from diffusers import WanImageToVideoPipeline")
        print(f'   pipe = WanImageToVideoPipeline.from_pretrained("{quantized_path}")')
        print(f"\nTo upload:")
        print(f'   python upload_quantized.py --repo-id "user/repo" --model-dir "{quantized_path}"')
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
