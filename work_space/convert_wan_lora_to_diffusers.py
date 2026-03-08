#!/usr/bin/env python3
"""
convert_wan_lora_to_diffusers.py

Downloads Wan LoRA safetensors from HuggingFace, remaps keys to diffusers
format, saves locally, and optionally uploads to your HuggingFace repository.

Supported input formats:
  - ComfyUI:       keys start with "diffusion_model."
                   e.g. diffusion_model.blocks.0.cross_attn.k.lora_A.weight
  - Kohya/flat:    keys start with "lora_unet_" (underscores as path separators)
                   e.g. lora_unet_blocks_0_cross_attn_k.lora_down.weight
  - PEFT-ComfyUI:  keys start with "blocks." BUT use ComfyUI-style internal paths
                   AND PEFT adapter-name suffixes (.lora_A.default.weight)
                   e.g. blocks.0.cross_attn.k.lora_A.default.weight  ← SVI LoRAs
  - Diffusers:     keys already fully correct
                   e.g. blocks.0.attn2.to_k.lora_A.weight

Usage examples:

  # Convert local files only (no download, no upload):
  python3 convert_wan_lora_to_diffusers.py --local /path/to/lora.safetensors

  # Convert multiple local files:
  python3 convert_wan_lora_to_diffusers.py \
      --local /path/a.safetensors /path/b.safetensors

  # Download single file, convert, no upload:
  python3 convert_wan_lora_to_diffusers.py \
      --hf-repo profpeng/nsfwv2 \
      --hf-files DR34ML4Y_I2V_14B_HIGH_V2.safetensors

  # Download multiple files, convert, no upload:
  python3 convert_wan_lora_to_diffusers.py \
      --hf-repo profpeng/nsfwv2 \
      --hf-files DR34ML4Y_I2V_14B_HIGH_V2.safetensors DR34ML4Y_I2V_14B_LOW_V2.safetensors

  # Filenames with spaces must be quoted:
  python3 convert_wan_lora_to_diffusers.py \
      --hf-repo lkzd7/WAN2.2_LoraSet_NSFW \
      --hf-files "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_high_noise.safetensors" \
                 "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_low_noise.safetensors"

  # Download, convert, and upload to your HF repo:
  python3 convert_wan_lora_to_diffusers.py \
      --hf-repo profpeng/nsfwv2 \
      --hf-files DR34ML4Y_I2V_14B_HIGH_V2.safetensors DR34ML4Y_I2V_14B_LOW_V2.safetensors \
      --upload-repo your-username/your-loras-repo \
      --token YOUR_HF_TOKEN

  # Use HF_TOKEN env var instead of --token:
  HF_TOKEN=xxx python3 convert_wan_lora_to_diffusers.py ...

  # Download from Civitai (model version ID from download URL):
  python3 convert_wan_lora_to_diffusers.py \
      --civitai-files 123456 789012 \
      --civitai-token YOUR_CIVITAI_API_KEY

  # Civitai with CIVITAI_API_KEY env var:
  CIVITAI_API_KEY=xxx python3 convert_wan_lora_to_diffusers.py --civitai-files 123456
"""

import os
import re
import sys
import getpass
import argparse

from huggingface_hub import HfApi, hf_hub_download, login, upload_file
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from safetensors.torch import load_file, save_file

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default local directory for downloaded LoRAs
# ---------------------------------------------------------------------------
DEFAULT_LORA_DIR = "/work_space/models/loras"

# Suffix appended to converted files before .safetensors
OUTPUT_SUFFIX = "_diffusers"

# ---------------------------------------------------------------------------
# LoRA suffix patterns — ALL known variants across all formats
# ---------------------------------------------------------------------------
LORA_SUFFIXES = [
    # PEFT with adapter name (.default. segment) — SVI LoRAs
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    ".lora_down.default.weight",
    ".lora_up.default.weight",
    # Standard diffusers / ComfyUI
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    # Alpha scalar
    ".alpha",
]

# Output suffix normalisation — PEFT adapter-name variants → standard names
# This ensures the output file always uses .lora_A.weight / .lora_B.weight
SUFFIX_NORMALISE = {
    ".lora_A.default.weight":    ".lora_A.weight",
    ".lora_B.default.weight":    ".lora_B.weight",
    ".lora_down.default.weight": ".lora_down.weight",
    ".lora_up.default.weight":   ".lora_up.weight",
}

# ---------------------------------------------------------------------------
# ComfyUI / PEFT-ComfyUI component path → diffusers path
# Used by both remap_comfyui() and remap_peft_comfy()
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
    # Two-pass norm swap
    "norm2":              "norm__placeholder",
    "norm3":              "norm2",
    "norm__placeholder":  "norm3",
    # Self attention
    "self_attn.q":        "attn1.to_q",
    "self_attn.k":        "attn1.to_k",
    "self_attn.v":        "attn1.to_v",
    "self_attn.o":        "attn1.to_out.0",
    "self_attn.norm_q":   "attn1.norm_q",
    "self_attn.norm_k":   "attn1.norm_k",
    # Cross attention
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
# Kohya/flat format component map
# Keys start with "lora_unet_" with underscores as path separators
# ---------------------------------------------------------------------------
KOHYA_COMPONENT_MAP = {
    "cross_attn_q":      "attn2.to_q",
    "cross_attn_k":      "attn2.to_k",
    "cross_attn_v":      "attn2.to_v",
    "cross_attn_o":      "attn2.to_out.0",
    "cross_attn_norm_q": "attn2.norm_q",
    "cross_attn_norm_k": "attn2.norm_k",
    "self_attn_q":       "attn1.to_q",
    "self_attn_k":       "attn1.to_k",
    "self_attn_v":       "attn1.to_v",
    "self_attn_o":       "attn1.to_out.0",
    "self_attn_norm_q":  "attn1.norm_q",
    "self_attn_norm_k":  "attn1.norm_k",
    "ffn_0":             "ffn.net.0.proj",
    "ffn_2":             "ffn.net.2",
}

# PEFT-ComfyUI detection: these substrings appear in keys that need remapping
# even though they start with "blocks." (not "diffusion_model.")
_PEFT_COMFY_MARKERS = (
    ".cross_attn.", ".self_attn.", ".ffn.0.", ".ffn.2.",
    "cross_attn.k", "cross_attn.q", "cross_attn.v", "cross_attn.o",
    "self_attn.k",  "self_attn.q",  "self_attn.v",  "self_attn.o",
)
_PEFT_ADAPTER_SUFFIX_MARKERS = (".default.",)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def detect_format(state_dict: dict) -> str:
    """
    Detect LoRA key format. Returns one of:
      "comfyui"     — diffusion_model. prefix
      "kohya"       — lora_unet_ prefix
      "peft_comfy"  — blocks. prefix + ComfyUI-style paths + .default. suffix (SVI LoRAs)
      "diffusers"   — already fully correct
    """
    keys = list(state_dict.keys())
    sample = keys[0]

    if sample.startswith("diffusion_model."):
        return "comfyui"

    if sample.startswith("lora_unet_"):
        return "kohya"

    if sample.startswith("blocks."):
        # Check if any key has .default. suffix (PEFT adapter-name segment)
        has_default_suffix = any(
            any(m in k for m in _PEFT_ADAPTER_SUFFIX_MARKERS) for k in keys[:20]
        )
        # Check if any key has ComfyUI-style internal path
        has_comfy_paths = any(
            any(m in k for m in _PEFT_COMFY_MARKERS) for k in keys[:20]
        )
        if has_default_suffix or has_comfy_paths:
            return "peft_comfy"

    return "diffusers"


# ---------------------------------------------------------------------------
# Remapping functions
# ---------------------------------------------------------------------------
def _peel_suffix(key: str) -> tuple:
    """Strip the longest matching LoRA suffix from key. Returns (base, suffix)."""
    for sfx in LORA_SUFFIXES:
        if key.endswith(sfx):
            return key[: -len(sfx)], sfx
    return key, ""


def remap_comfyui(state_dict: dict) -> dict:
    """
    Remap ComfyUI LoRA keys to diffusers format.
    Input:  diffusion_model.blocks.0.cross_attn.k.lora_A.weight
    Output: blocks.0.attn2.to_k.lora_A.weight
    """
    PREFIX = "diffusion_model."
    new_sd = {}
    for key, val in state_dict.items():
        k = key[len(PREFIX):] if key.startswith(PREFIX) else key
        k, sfx = _peel_suffix(k)
        sfx = SUFFIX_NORMALISE.get(sfx, sfx)
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            k = k.replace(old, new)
        new_sd[k + sfx] = val
    return new_sd


def remap_peft_comfy(state_dict: dict) -> dict:
    """
    Remap PEFT-ComfyUI LoRA keys to diffusers format.

    This format is produced when PEFT saves a LoRA that was trained on a
    ComfyUI-structured model. Keys start with "blocks." (no module prefix)
    but use ComfyUI-style internal paths AND PEFT adapter-name suffixes.

    Input:  blocks.0.cross_attn.k.lora_A.default.weight
    Output: blocks.0.attn2.to_k.lora_A.weight

    Two transformations applied:
      1. Suffix normalisation: .lora_A.default.weight → .lora_A.weight
      2. Component remapping:  cross_attn.k → attn2.to_k  (via TRANSFORMER_KEYS_RENAME_DICT)
    """
    new_sd = {}
    for key, val in state_dict.items():
        k, sfx = _peel_suffix(key)

        # 1. Normalise suffix (strip .default. adapter-name segment)
        sfx = SUFFIX_NORMALISE.get(sfx, sfx)

        # 2. Remap component paths using the same dict as ComfyUI
        #    Only the component part needs remapping; the "blocks.N." prefix is correct.
        m = re.match(r"^(blocks\.\d+\.)(.+)$", k)
        if m:
            block_prefix = m.group(1)
            component = m.group(2)
            for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
                component = component.replace(old, new)
            k = block_prefix + component

        new_sd[k + sfx] = val
    return new_sd


def remap_kohya(state_dict: dict) -> dict:
    """
    Remap Kohya/flat LoRA keys to diffusers format.
    Input:  lora_unet_blocks_0_cross_attn_k.lora_down.weight
    Output: blocks.0.attn2.to_k.lora_down.weight
    """
    PREFIX = "lora_unet_"
    new_sd = {}
    for key, val in state_dict.items():
        k = key[len(PREFIX):] if key.startswith(PREFIX) else key
        k, sfx = _peel_suffix(k)
        sfx = SUFFIX_NORMALISE.get(sfx, sfx)

        m = re.match(r"^blocks_(\d+)_(.+)$", k)
        if m:
            block_idx = m.group(1)
            component = m.group(2)
            diffusers_component = KOHYA_COMPONENT_MAP.get(
                component, component.replace("_", "."))
            new_key = f"blocks.{block_idx}.{diffusers_component}"
        else:
            new_key = k.replace("_", ".")

        new_sd[new_key + sfx] = val
    return new_sd


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------
def convert_file(input_path: str, output_dir: str = None) -> str:
    """
    Load a LoRA safetensors file, auto-detect format, remap if needed,
    save converted file. Returns the output file path.

    Supported formats:
      - ComfyUI     (diffusion_model. prefix)
      - Kohya/flat  (lora_unet_ prefix)
      - PEFT-ComfyUI (blocks. prefix + ComfyUI paths + .default. suffix) ← NEW
      - Diffusers   (already correct, saved as-is)
    """
    filename_base = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]

    if output_dir:
        out_path = os.path.join(output_dir, filename_base + OUTPUT_SUFFIX + ext)
    else:
        out_path = os.path.splitext(input_path)[0] + OUTPUT_SUFFIX + ext

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"  Already converted, skipping: {os.path.basename(out_path)} ({size_mb:.0f} MB)")
        return out_path

    print(f"  Loading:  {os.path.basename(input_path)}")
    sd = load_file(input_path, device="cpu")

    fmt = detect_format(sd)

    if fmt == "comfyui":
        print(f"  Format:   ComfyUI ({len(sd)} keys) — remapping paths + suffixes...")
        new_sd = remap_comfyui(sd)
    elif fmt == "kohya":
        print(f"  Format:   Kohya/flat ({len(sd)} keys) — remapping paths + suffixes...")
        new_sd = remap_kohya(sd)
    elif fmt == "peft_comfy":
        print(f"  Format:   PEFT-ComfyUI ({len(sd)} keys) — remapping paths + normalising .default. suffixes...")
        print(f"            (SVI-style LoRA: blocks. prefix with ComfyUI internal paths)")
        new_sd = remap_peft_comfy(sd)
    else:
        print(f"  Format:   Diffusers ({len(sd)} keys) — already correct, saving as-is")
        new_sd = sd

    # Verify conversion produced sensible output
    sample_out = list(new_sd.keys())[:3]
    print(f"  Sample output keys: {sample_out}")

    print(f"  Saving:   {os.path.basename(out_path)}")
    save_file(new_sd, out_path)

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  Done:     {os.path.basename(out_path)} ({size_mb:.0f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# HuggingFace auth
# ---------------------------------------------------------------------------
def get_hf_token(token_arg: str = None):
    if token_arg:
        return token_arg
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        api = HfApi()
        user = api.whoami()
        print(f"  Already logged in as: {user['name']}")
        return None
    except Exception:
        pass
    print("\nHuggingFace token required.")
    print("Get yours at: https://huggingface.co/settings/tokens")
    token = getpass.getpass("  Enter HuggingFace token: ").strip()
    return token if token else None


def ensure_hf_login(token: str = None) -> bool:
    token = get_hf_token(token)
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"  Login failed: {e}")
            return False
    try:
        api = HfApi()
        user = api.whoami()
        print(f"  Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"  Not logged in: {e}")
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_lora(repo_id: str, filename: str, save_dir: str, token: str = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    dest = os.path.join(save_dir, os.path.basename(filename))

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"  Already downloaded: {os.path.basename(dest)} ({size_mb:.0f} MB)")
        return dest

    print(f"  Downloading: {repo_id}/{filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id, filename=filename,
        local_dir=save_dir, local_dir_use_symlinks=False, token=token,
    )
    if os.path.abspath(downloaded) != os.path.abspath(dest):
        import shutil
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(downloaded, dest)

    size_mb = os.path.getsize(dest) / (1024 ** 2)
    print(f"  Saved:       {dest} ({size_mb:.0f} MB)")
    return dest


def download_lora_from_civitai(model_id: str, save_dir: str, api_token: str):
    if not REQUESTS_AVAILABLE:
        print(f"  Civitai download requires 'pip install requests'")
        return None
    if not api_token:
        print(f"  Civitai requires --civitai-token or CIVITAI_API_KEY env var")
        return None

    os.makedirs(save_dir, exist_ok=True)
    url = f"https://civitai.com/api/download/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    print(f"  Downloading from Civitai (model {model_id})...")
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
            print(f"  Already downloaded: {os.path.basename(dest)} ({size_mb:.0f} MB)")
            return dest

        total_size = int(response.headers.get("content-length", 0))
        downloaded_bytes = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_size > 0 and downloaded_bytes % (5 * 1024 * 1024) < 8192:
                        pct = 100 * downloaded_bytes / total_size
                        print(f"  Downloading... {pct:.0f}%", end="\r", flush=True)
        print()
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"  Saved:       {dest} ({size_mb:.0f} MB)")
        return dest
    except requests.RequestException as e:
        print(f"  Civitai download failed: {e}")
        try:
            if dest and os.path.exists(dest):
                os.remove(dest)
        except (NameError, OSError):
            pass
        return None


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def ensure_repo_exists(repo_id: str, private: bool = False):
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"  Repo exists: {repo_id}")
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        is_404 = isinstance(e, RepositoryNotFoundError) or (
            isinstance(e, HfHubHTTPError) and getattr(e, "status_code", None) == 404
        )
        if is_404:
            print(f"  Creating repo: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            print(f"  Repo created.")
        else:
            raise


def upload_lora(local_path: str, repo_id: str, repo_filename: str = None, token: str = None):
    if repo_filename is None:
        repo_filename = os.path.basename(local_path)
    size_mb = os.path.getsize(local_path) / (1024 ** 2)
    print(f"  Uploading: {os.path.basename(local_path)} ({size_mb:.0f} MB) -> {repo_id}/{repo_filename}")
    upload_file(
        path_or_fileobj=local_path, path_in_repo=repo_filename,
        repo_id=repo_id, repo_type="model",
        commit_message=f"Add converted diffusers LoRA: {repo_filename}",
        token=token,
    )
    print(f"  Uploaded:  https://huggingface.co/{repo_id}/blob/main/{repo_filename}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args):
    print("\n" + "=" * 60)
    print("Wan LoRA Converter → Diffusers Format")
    print("Supports: ComfyUI | Kohya/flat | PEFT-ComfyUI (SVI) | Diffusers")
    print("=" * 60)

    hf_token = None
    if args.upload_repo:
        print("\nAuthenticating with HuggingFace...")
        if not ensure_hf_login(args.token):
            print("Authentication failed — aborting.")
            sys.exit(1)
        hf_token = get_hf_token(args.token)
        ensure_repo_exists(args.upload_repo, private=args.private)

    input_files = []

    if args.local:
        for p in args.local:
            if not os.path.exists(p):
                print(f"\nWARNING: Local file not found, skipping: {p}")
                continue
            input_files.append(p)

    if args.hf_repo and args.hf_files:
        print(f"\nDownloading from {args.hf_repo}...")
        dl_token = get_hf_token(args.token) if args.token or os.getenv("HF_TOKEN") else None
        for filename in args.hf_files:
            local_path = download_lora(
                repo_id=args.hf_repo, filename=filename,
                save_dir=args.save_dir, token=dl_token,
            )
            input_files.append(local_path)

    if args.civitai_files:
        print(f"\nDownloading from Civitai...")
        civitai_token = args.civitai_token or os.getenv("CIVITAI_API_KEY")
        for model_id in args.civitai_files:
            local_path = download_lora_from_civitai(
                model_id=model_id, save_dir=args.save_dir, api_token=civitai_token,
            )
            if local_path:
                input_files.append(local_path)

    if not input_files:
        print("\nNo input files to process.")
        sys.exit(1)

    print(f"\nConverting {len(input_files)} file(s)...")
    converted_files = []
    for input_path in input_files:
        print(f"\n[{os.path.basename(input_path)}]")
        out_path = convert_file(input_path, output_dir=os.path.dirname(input_path))
        converted_files.append(out_path)

    if args.upload_repo:
        print(f"\nUploading {len(converted_files)} file(s) to {args.upload_repo}...")
        for out_path in converted_files:
            print(f"\n[{os.path.basename(out_path)}]")
            upload_lora(local_path=out_path, repo_id=args.upload_repo, token=hf_token)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Converted {len(converted_files)} file(s):")
    for p in converted_files:
        print(f"    {p}")
    if args.upload_repo:
        print(f"\n  Uploaded to: https://huggingface.co/{args.upload_repo}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Wan LoRAs (ComfyUI/Kohya/PEFT-ComfyUI/Diffusers) to diffusers format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--local", nargs="+", metavar="PATH",
                        help="One or more local .safetensors files to convert.")
    parser.add_argument("--hf-repo", metavar="REPO_ID",
                        help="HuggingFace repo to download LoRA(s) from.")
    parser.add_argument("--hf-files", nargs="+", metavar="FILENAME",
                        help="Filenames to download from --hf-repo.")
    parser.add_argument("--save-dir", metavar="DIR", default=DEFAULT_LORA_DIR,
                        help=f"Directory to save downloaded LoRAs (default: {DEFAULT_LORA_DIR}).")
    parser.add_argument("--civitai-files", nargs="+", metavar="MODEL_ID",
                        help="Civitai model version IDs to download.")
    parser.add_argument("--civitai-token", metavar="TOKEN",
                        help="Civitai API token (or set CIVITAI_API_KEY env var).")
    parser.add_argument("--upload-repo", metavar="REPO_ID",
                        help="HuggingFace repo to upload converted files to.")
    parser.add_argument("--private", action="store_true",
                        help="Make the upload repo private if creating it.")
    parser.add_argument("--token", metavar="TOKEN",
                        help="HuggingFace token (or set HF_TOKEN env var).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    has_local    = bool(args.local)
    has_hf       = bool(args.hf_repo and args.hf_files)
    has_civitai  = bool(args.civitai_files)

    if not has_local and not has_hf and not has_civitai:
        print("Error: provide --local files, --hf-repo + --hf-files, or --civitai-files.")
        sys.exit(1)
    if args.hf_repo and not args.hf_files:
        print("Error: --hf-repo requires --hf-files.")
        sys.exit(1)
    if has_civitai and not (args.civitai_token or os.getenv("CIVITAI_API_KEY")):
        print("Error: --civitai-files requires --civitai-token or CIVITAI_API_KEY env var.")
        sys.exit(1)

    run(args)