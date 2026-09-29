#!/usr/bin/env python3
"""
SDNQ quantization script for Wan diffusers models (I2V, T2V).

Quantizes Wan pipeline models to SDNQ (uint4, int8, or fp8) + SVD,
reducing model size while preserving quality.

Usage:
    # Local model, no LoRA:
    python quantize_sdnq_multi.py --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" --dtype fp8 --svd-rank 64 --svd-steps 32

    # Multiple LoRAs with individual strengths on transformer (t1) and transformer_2 (t2):
    python quantize_sdnq_multi.py \
        --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" \
        --dtype fp8 --svd-rank 64 --svd-steps 32 \
        --lora-t1 "/path/to/lora_a.safetensors:0.8" \
        --lora-t1 "/path/to/lora_b.safetensors:0.5" \
        --lora-t2 "/path/to/lora_c.safetensors:1.0" \
        --lora-t2 "/path/to/lora_d.safetensors:0.3"

    # LoRA without explicit strength falls back to --lora-strength (default 1.0):
    python quantize_sdnq_multi.py \
        --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" \
        --dtype fp8 \
        --lora-strength 0.7 \
        --lora-t1 "/path/to/lora_a.safetensors" \
        --lora-t1 "/path/to/lora_b.safetensors:0.4"

    # From HuggingFace repo:
    python quantize_sdnq_multi.py --repo "Wan-AI/Wan2.2-I2V-A14B-Diffusers" --dtype fp8 --svd-rank 64 --svd-steps 16

    # HF model + HF LoRAs (multi, with strength):
    python3 quantize_sdnq_multi.py --repo Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype fp8 \
        --lora-t1-repo "profpeng/nsfwv2:DR34ML4Y_I2V_14B_HIGH_V2.safetensors:0.8" \
        --lora-t1-repo "profpeng/other:OTHER_LORA.safetensors:0.5" \
        --lora-t2-repo "profpeng/nsfwv2:DR34ML4Y_I2V_14B_LOW_V2.safetensors:0.8"

    # Civitai LoRAs (multi):
    python3 quantize_sdnq_multi.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1-civitai "123456:0.8" \
        --lora-t1-civitai "999999:0.5" \
        --lora-t2-civitai "789012:1.0" \
        --civitai-token YOUR_CIVITAI_API_KEY

    # Mixed sources (local + HF + Civitai all at once):
    python3 quantize_sdnq_multi.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1 "/path/to/local.safetensors:0.8" \
        --lora-t1-repo "profpeng/nsfwv2:OTHER.safetensors:0.5" \
        --lora-t1-civitai "123456:0.3" \
        --civitai-token YOUR_CIVITAI_API_KEY

    # Single LoRA applied to both transformers (omit all --lora-t2* args):
    python3 quantize_sdnq_multi.py --model-dir /path/to/model --dtype fp8 \
        --lora-t1 "/path/to/lora.safetensors:0.8"

LoRA argument formats:
    --lora-t1  PATH[:STRENGTH]              Local file, optional per-LoRA strength
    --lora-t2  PATH[:STRENGTH]              Local file for transformer_2
    --lora-t1-repo  REPO:FILE[:STRENGTH]    HuggingFace repo:filename, optional strength
    --lora-t2-repo  REPO:FILE[:STRENGTH]    HuggingFace for transformer_2
    --lora-t1-civitai  MODEL_VERSION_ID[:STRENGTH]  Civitai model version ID
    --lora-t2-civitai  MODEL_VERSION_ID[:STRENGTH]  Civitai for transformer_2

    If STRENGTH is omitted from any spec, --lora-strength (default 1.0) is used.
    All --lora-t1* args are repeatable (append multiple times for multiple LoRAs).
    If no --lora-t2* args are given, all t1 LoRAs are also applied to transformer_2.

Requirements:
    pip install sdnq diffusers accelerate transformers safetensors
    # For Civitai LoRA download: pip install requests
"""

import gc
import os
import re
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
# LoRA suffix patterns — ALL known variants across all formats
#
# FIX: Added .default. variants for PEFT/SVI LoRAs.
# Old code only had .lora_A.weight which missed .lora_A.default.weight,
# causing pairs dict to be empty → "0/0 layers baked".
# ---------------------------------------------------------------------------
LORA_SUFFIXES = [
    # PEFT with adapter name (.default. segment) — SVI LoRAs
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    ".lora_down.default.weight",
    ".lora_up.default.weight",
    # Standard diffusers / ComfyUI / Kohya
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    # Alpha scalar
    ".alpha",
]

# ---------------------------------------------------------------------------
# ComfyUI → diffusers key remapping
# Used for full ComfyUI format (diffusion_model. prefix)
# ---------------------------------------------------------------------------
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0":   "condition_embedder.time_embedder.linear_1",
    "time_embedding.2":   "condition_embedder.time_embedder.linear_2",
    "text_embedding.0":   "condition_embedder.text_embedder.linear_1",
    "text_embedding.2":   "condition_embedder.text_embedder.linear_2",
    "time_projection.1":  "condition_embedder.time_proj",
    "head.modulation":    "scale_shift_table",
    "head.head":          "proj_out",
    "modulation":         "scale_shift_table",
    "ffn.0":              "ffn.net.0.proj",
    "ffn.2":              "ffn.net.2",
    "norm2":              "norm__placeholder",
    "norm3":              "norm2",
    "norm__placeholder":  "norm3",
    "self_attn.q":        "attn1.to_q",
    "self_attn.k":        "attn1.to_k",
    "self_attn.v":        "attn1.to_v",
    "self_attn.o":        "attn1.to_out.0",
    "self_attn.norm_q":   "attn1.norm_q",
    "self_attn.norm_k":   "attn1.norm_k",
    "cross_attn.q":       "attn2.to_q",
    "cross_attn.k":       "attn2.to_k",
    "cross_attn.v":       "attn2.to_v",
    "cross_attn.o":       "attn2.to_out.0",
    "cross_attn.norm_q":  "attn2.norm_q",
    "cross_attn.norm_k":  "attn2.norm_k",
    "attn2.to_k_img":     "attn2.add_k_proj",
    "attn2.to_v_img":     "attn2.add_v_proj",
    "attn2.norm_k_img":   "attn2.norm_added_k",
}

# ---------------------------------------------------------------------------
# Component path remapping for PEFT-ComfyUI LoRAs (SVI style)
#
# FIX: These LoRAs have blocks. prefix (so old code classified them as
# "diffusers" format) but use ComfyUI-style internal paths. Without this
# remapping the weight key lookup would miss even if suffixes matched.
# Verified: 400/400 layers match after applying this remap.
# ---------------------------------------------------------------------------
COMPONENT_REMAP = {
    "cross_attn.k":      "attn2.to_k",
    "cross_attn.o":      "attn2.to_out.0",
    "cross_attn.q":      "attn2.to_q",
    "cross_attn.v":      "attn2.to_v",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "self_attn.k":       "attn1.to_k",
    "self_attn.o":       "attn1.to_out.0",
    "self_attn.q":       "attn1.to_q",
    "self_attn.v":       "attn1.to_v",
    "self_attn.norm_q":  "attn1.norm_q",
    "self_attn.norm_k":  "attn1.norm_k",
    "ffn.0":             "ffn.net.0.proj",
    "ffn.2":             "ffn.net.2",
}

# Module-level prefixes to strip from PEFT-wrapped LoRAs (longest first)
STRIP_PREFIXES = [
    "base_model.model.transformer_2.",
    "base_model.model.transformer.",
    "base_model.model.",
    "transformer_2.",
    "transformer.",
    "diffusion_model.",
]


# ============================================================================
# LoRA spec parsing
# ============================================================================

def parse_strength(value: str, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_local_lora_spec(spec: str, default_strength: float) -> tuple:
    parts = spec.rsplit(":", 1)
    if len(parts) == 2:
        try:
            strength = float(parts[1])
            return parts[0], strength
        except ValueError:
            pass
    return spec, default_strength


def parse_hf_lora_spec(spec: str, default_strength: float) -> tuple:
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], default_strength
    elif len(parts) == 3:
        try:
            strength = float(parts[2])
            return parts[0], parts[1], strength
        except ValueError:
            return parts[0], ":".join(parts[1:]), default_strength
    else:
        raise ValueError(
            f"--lora-t1-repo / --lora-t2-repo expects REPO:FILE or REPO:FILE:STRENGTH, got: {spec!r}"
        )


def parse_civitai_lora_spec(spec: str, default_strength: float) -> tuple:
    parts = spec.split(":", 1)
    if len(parts) == 2:
        try:
            strength = float(parts[1])
            return parts[0], strength
        except ValueError:
            pass
    return spec, default_strength


# ============================================================================
# Helpers
# ============================================================================

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


# ============================================================================
# LoRA injection — FIXED
#
# Changes vs original:
#   1. LORA_SUFFIXES extended to include .default. variants (Bug 1 fix)
#   2. COMPONENT_REMAP applied to blocks.N.* keys (Bug 2 fix)
#   3. STRIP_PREFIXES handles PEFT-wrapped (base_model.model.) LoRAs
#   4. Both fixes are backwards-compatible: old formats work unchanged
# ============================================================================

def inject_lora(model: torch.nn.Module, lora_path: str, strength: float, label: str = ""):
    """
    Permanently bake LoRA delta weights into model parameters.

    Supports all known Wan LoRA formats:
      - Standard diffusers    (.lora_A.weight / .lora_down.weight)
      - PEFT with adapter name (.lora_A.default.weight)  ← SVI LoRAs
      - PEFT-ComfyUI paths    (cross_attn.k → attn2.to_k etc.)
      - Full ComfyUI          (diffusion_model. prefix)
      - PEFT-wrapped          (base_model.model. prefix)
    """
    tag = f" ({label})" if label else ""
    print(f"   Injecting LoRA{tag}: {os.path.basename(lora_path)} strength={strength}")

    try:
        raw_sd = load_file(lora_path, device="cpu")
    except Exception as e:
        print(f"   ERROR: Could not load LoRA: {e}")
        return

    all_keys = list(raw_sd.keys())
    is_full_comfy = any(k.startswith("diffusion_model.") for k in all_keys)

    if is_full_comfy:
        # Full ComfyUI format: strip prefix then apply full rename dict
        print(f"   Full ComfyUI format detected — remapping {len(all_keys)} keys...")
        sd = {}
        for key, val in raw_sd.items():
            k = key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key
            sfx = ""
            for s in LORA_SUFFIXES:
                if k.endswith(s):
                    sfx = s
                    k = k[:-len(s)]
                    break
            for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
                k = k.replace(old, new)
            sd[k + sfx] = val
        del raw_sd
        gc.collect()
        print(f"   Remapped to {len(sd)} diffusers keys")
    else:
        sd = raw_sd
        print(f"   Diffusers/PEFT format — {len(all_keys)} keys (no full remap needed)")

    # Show sample keys so output is visible and debuggable
    for sk in list(sd.keys())[:3]:
        print(f"   sample key: {sk}  {list(sd[sk].shape)}")

    # Build pairs: resolved_base_key -> {down, up, alpha?}
    pairs = {}
    skipped_no_sfx = 0

    for key, val in sd.items():
        k = key

        # Step 1: Strip module-level prefix (PEFT-wrapped / diffusers-native with module name)
        for pfx in STRIP_PREFIXES:
            if k.startswith(pfx):
                k = k[len(pfx):]
                break

        # Step 2: Peel LoRA suffix — LORA_SUFFIXES now includes .default. variants
        sfx = ""
        for s in LORA_SUFFIXES:
            if k.endswith(s):
                sfx = s
                k = k[:-len(s)]
                break

        if not sfx:
            skipped_no_sfx += 1
            continue

        # Step 3: Remap ComfyUI-style component paths inside block keys
        # e.g. blocks.0.cross_attn.k → blocks.0.attn2.to_k
        # e.g. blocks.0.ffn.0        → blocks.0.ffn.net.0.proj
        m = re.match(r"^(blocks\.\d+\.)(.+)$", k)
        if m:
            component = COMPONENT_REMAP.get(m.group(2), m.group(2))
            k = m.group(1) + component

        # Step 4: Categorise into pairs
        if sfx in (".lora_A.weight", ".lora_A.default.weight",
                   ".lora_down.weight", ".lora_down.default.weight"):
            pairs.setdefault(k, {})["down"] = val
        elif sfx in (".lora_B.weight", ".lora_B.default.weight",
                     ".lora_up.weight", ".lora_up.default.weight"):
            pairs.setdefault(k, {})["up"] = val
        elif sfx == ".alpha":
            pairs.setdefault(k, {})["alpha"] = val

    del sd
    gc.collect()

    if skipped_no_sfx > 0:
        print(f"   Skipped {skipped_no_sfx} keys with unrecognised suffixes")

    model_params = dict(model.named_parameters())
    total = len([p for p in pairs.values() if "down" in p and "up" in p])
    applied = skipped_missing = skipped_incomplete = 0

    print(f"   Baking {total} LoRA layers into model weights...")
    for i, (base_key, parts) in enumerate(pairs.items()):
        if i % 50 == 0 and i > 0:
            print(f"   Baking... {i}/{len(pairs)}", end="\r", flush=True)

        if "down" not in parts or "up" not in parts:
            skipped_incomplete += 1
            continue

        weight_key = base_key + ".weight"
        if weight_key not in model_params:
            skipped_missing += 1
            continue

        param = model_params[weight_key]
        dtype = param.dtype

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
    print(f"   Result : {applied}/{total} layers baked"
          f" | {skipped_missing} not in model"
          f" | {skipped_incomplete} incomplete")

    if applied == 0:
        print(f"   ERROR  : No weights updated — check LoRA format/compatibility")
    elif applied < total * 0.8:
        print(f"   WARNING: Only {applied}/{total} layers applied ({100*applied//total}%)")
    else:
        print(f"   ✓ LoRA baked successfully ({100*applied//total}% of layers)")


# ---------------------------------------------------------------------------
# Multi-LoRA injection
# ---------------------------------------------------------------------------
def inject_lora_list(model: torch.nn.Module, lora_list: list, label: str):
    if not lora_list:
        return
    print(f"\n   Baking {len(lora_list)} LoRA(s) into {label}...")
    for idx, (path, strength) in enumerate(lora_list, 1):
        print(f"   [{idx}/{len(lora_list)}] ", end="")
        inject_lora(model, path, strength, label=label)
        gc.collect()
    print(f"   All {len(lora_list)} LoRA(s) baked into {label}.")


def merge_loras_into_pipe(pipe, lora_t1_list: list, lora_t2_list: list):
    total = len(lora_t1_list) + len(lora_t2_list)
    print(f"\nBaking LoRAs into model weights "
          f"({len(lora_t1_list)} for transformer, "
          f"{len(lora_t2_list) or f'{len(lora_t1_list)} (mirrored from t1)'} for transformer_2)...")

    if lora_t1_list and hasattr(pipe, "transformer") and pipe.transformer is not None:
        inject_lora_list(pipe.transformer, lora_t1_list, "transformer")
        gc.collect()

    effective_t2_list = lora_t2_list if lora_t2_list else lora_t1_list
    if effective_t2_list and hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        inject_lora_list(pipe.transformer_2, effective_t2_list, "transformer_2")
        gc.collect()

    print(f"\nLoRA baking complete ({total} injection(s) total).")


# ---------------------------------------------------------------------------
# Download helpers
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


def download_lora_from_civitai(model_id: str, save_dir: str, api_token: str, label: str):
    if not REQUESTS_AVAILABLE:
        print(f"   {label}: Civitai download requires 'pip install requests'")
        return None
    if not api_token:
        print(f"   {label}: Civitai requires --civitai-token or CIVITAI_API_KEY env var")
        return None

    os.makedirs(save_dir, exist_ok=True)
    url = f"https://civitai.com/api/download/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    print(f"   {label}: downloading from Civitai (model {model_id})...")
    dest = None
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

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


def resolve_hf_lora(repo: str, file: str, save_dir: str, token: str, label: str):
    os.makedirs(save_dir, exist_ok=True)
    dest = os.path.join(save_dir, os.path.basename(file))
    if os.path.exists(dest):
        print(f"   {label}: already downloaded at {dest}")
        return dest
    print(f"   {label}: downloading {repo}/{file}...")
    from huggingface_hub import hf_hub_download
    import shutil
    downloaded = hf_hub_download(
        repo_id=repo, filename=file,
        local_dir=save_dir, local_dir_use_symlinks=False, token=token,
    )
    if os.path.abspath(downloaded) != os.path.abspath(dest):
        shutil.move(downloaded, dest)
    print(f"   {label}: saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Resolve all LoRA specs into (path, strength) lists
# ---------------------------------------------------------------------------
def resolve_lora_list(
    local_specs: list,
    hf_specs: list,
    civitai_specs: list,
    save_dir: str,
    token,
    civitai_token,
    default_strength: float,
    label: str,
) -> list:
    result = []

    for spec in (local_specs or []):
        path, strength = parse_local_lora_spec(spec, default_strength)
        if os.path.exists(path):
            result.append((path, strength))
        else:
            print(f"   WARNING [{label}]: local LoRA not found — skipping: {path}")

    for spec in (hf_specs or []):
        repo, file, strength = parse_hf_lora_spec(spec, default_strength)
        path = resolve_hf_lora(repo, file, save_dir, token, label)
        if path:
            result.append((path, strength))

    for spec in (civitai_specs or []):
        model_id, strength = parse_civitai_lora_spec(spec, default_strength)
        path = download_lora_from_civitai(model_id, save_dir, civitai_token, label)
        if path:
            result.append((path, strength))

    return result


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
                   lora_t1_list=None, lora_t2_list=None):
    print("\n" + "=" * 60)
    print("Quantizing with SDNQ")
    print("=" * 60)
    print(f"   Quantization: {weights_dtype} + SVD rank {svd_rank}, steps {svd_steps}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SDNQ quantization (SVD needs GPU)")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_pipeline(model_path)

    # Bake LoRAs permanently BEFORE quantization
    lora_t1_list = lora_t1_list or []
    lora_t2_list = lora_t2_list or []
    if lora_t1_list or lora_t2_list:
        merge_loras_into_pipe(pipe, lora_t1_list, lora_t2_list)

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
        description="Quantize Wan diffusers model with SDNQ, with optional multi-LoRA baking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", type=str, help="HuggingFace model repo ID")
    src.add_argument("--model-dir", type=str, help="Local model directory path")

    parser.add_argument("--dtype", choices=["uint4", "int8", "fp8"],
                        default=DEFAULT_WEIGHTS_DTYPE)
    parser.add_argument("--svd-rank", type=int, default=DEFAULT_SVD_RANK)
    parser.add_argument("--svd-steps", type=int, default=DEFAULT_SVD_STEPS)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument(
        "--lora-strength", type=float, default=1.0,
        help="Default strength for any LoRA that does not specify its own (default: 1.0)",
    )

    # transformer (t1) LoRAs
    parser.add_argument("--lora-t1", action="append", default=[], metavar="PATH[:STRENGTH]",
                        help="Local LoRA for transformer. Repeatable.")
    parser.add_argument("--lora-t1-repo", action="append", default=[], metavar="REPO:FILE[:STRENGTH]",
                        help="HF repo LoRA for transformer. Repeatable.")
    parser.add_argument("--lora-t1-civitai", action="append", default=[], metavar="MODEL_ID[:STRENGTH]",
                        help="Civitai model version ID for transformer LoRA. Repeatable.")

    # transformer_2 (t2) LoRAs
    parser.add_argument("--lora-t2", action="append", default=[], metavar="PATH[:STRENGTH]",
                        help="Local LoRA for transformer_2. Falls back to t1 list if omitted entirely.")
    parser.add_argument("--lora-t2-repo", action="append", default=[], metavar="REPO:FILE[:STRENGTH]",
                        help="HF repo LoRA for transformer_2. Repeatable.")
    parser.add_argument("--lora-t2-civitai", action="append", default=[], metavar="MODEL_ID[:STRENGTH]",
                        help="Civitai model version ID for transformer_2 LoRA. Repeatable.")

    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--civitai-token", type=str, default=None,
                        help="Civitai API token (or set CIVITAI_API_KEY env var).")

    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    civitai_token = args.civitai_token or os.getenv("CIVITAI_API_KEY")

    # Resolve model path
    if args.model_dir:
        model_path = os.path.abspath(args.model_dir)
        if not os.path.isdir(model_path):
            print(f"ERROR: Model directory not found: {model_path}"); exit(1)
        model_name = os.path.basename(model_path.rstrip(os.sep))
    else:
        model_path, _ = download_model(args.repo, token=token)
        model_name = args.repo.split("/")[-1]

    lora_save_dir = os.path.join(MODELS_DIR, "loras")
    default_strength = args.lora_strength

    # Resolve all LoRA specs into (path, strength) lists
    print("\nResolving LoRA sources...")
    lora_t1_list = resolve_lora_list(
        local_specs=args.lora_t1,
        hf_specs=args.lora_t1_repo,
        civitai_specs=args.lora_t1_civitai,
        save_dir=lora_save_dir,
        token=token,
        civitai_token=civitai_token,
        default_strength=default_strength,
        label="t1",
    )
    lora_t2_list = resolve_lora_list(
        local_specs=args.lora_t2,
        hf_specs=args.lora_t2_repo,
        civitai_specs=args.lora_t2_civitai,
        save_dir=lora_save_dir,
        token=token,
        civitai_token=civitai_token,
        default_strength=default_strength,
        label="t2",
    )

    # Auto output dir
    if args.output_dir is None:
        lora_tag = ""
        all_loras = lora_t1_list or lora_t2_list
        if all_loras:
            first_name = os.path.splitext(os.path.basename(all_loras[0][0]))[0]
            count_tag = f"_x{len(lora_t1_list or lora_t2_list)}" if len(all_loras) > 1 else ""
            lora_tag = f"_lora_{first_name}{count_tag}"
        output_dir = os.path.join(
            MODELS_DIR, f"{model_name}{lora_tag}_sdnq_{args.dtype}_svd_r{args.svd_rank}")
    else:
        output_dir = args.output_dir

    # Summary
    has_lora = bool(lora_t1_list or lora_t2_list)
    effective_t2 = lora_t2_list if lora_t2_list else lora_t1_list

    print("\n" + "=" * 60)
    print("SDNQ Quantization")
    print("=" * 60)
    print(f"   Model     : {args.model_dir or args.repo}")
    if has_lora:
        if lora_t1_list:
            print(f"   LoRA t1   : {len(lora_t1_list)} LoRA(s)")
            for path, strength in lora_t1_list:
                print(f"               {os.path.basename(path)}  strength={strength}")
        if effective_t2:
            mirror_note = " (mirrored from t1)" if not lora_t2_list else ""
            print(f"   LoRA t2   : {len(effective_t2)} LoRA(s){mirror_note}")
            for path, strength in effective_t2:
                print(f"               {os.path.basename(path)}  strength={strength}")
    print(f"   Quantize  : {args.dtype} + SVD rank {args.svd_rank}, steps {args.svd_steps}")
    print(f"   Output    : {output_dir}")
    print("=" * 60)

    total_start = time.time()
    try:
        quantized_path = quantize_model(
            model_path, output_dir, args.dtype, args.svd_rank, args.svd_steps,
            lora_t1_list=lora_t1_list,
            lora_t2_list=lora_t2_list,
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