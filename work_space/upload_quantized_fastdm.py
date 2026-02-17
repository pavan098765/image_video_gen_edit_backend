#!/usr/bin/env python3
"""
Upload quantized FastDM model to HuggingFace Hub.

This script uploads locally quantized models (INT8 or SDNQ) to your HuggingFace account,
allowing you to download them directly on other machines without re-quantizing.

Usage:
    # Upload INT8 model (default):
    python upload_quantized_fastdm.py --model-type int8
    
    # Upload SDNQ INT8+SVD model:
    python upload_quantized_fastdm.py --model-type sdnq_int8
    
    # Upload SDNQ FP8+SVD model:
    python upload_quantized_fastdm.py --model-type sdnq_fp8
    
    # With token as environment variable:
    HF_TOKEN=your_token_here python upload_quantized_fastdm.py --model-type sdnq_fp8
    
    # With token as command-line argument:
    python upload_quantized_fastdm.py --model-type sdnq_int8 --token your_token_here

Requirements:
    pip install huggingface_hub[hf_transfer]

Model Types:
    - int8:      INT8 quantized model (optimum.quanto, from quantize_fastdm.py)
    - sdnq_int8: SDNQ INT8+SVD r32 quantized model (from quantize_fastdm_sdnq.py with int8)
    - sdnq_fp8:  SDNQ FP8 (float8_e4m3fn)+SVD r32 quantized model (from quantize_fastdm_sdnq.py with fp8)

Login Options:
    1. Set HF_TOKEN environment variable
    2. Pass --token argument
    3. Script will prompt for token if neither provided
    4. Or use existing login (huggingface-cli login)

Before running:
    1. Make sure you've run quantize_fastdm.py (for INT8) or quantize_fastdm_sdnq.py (for SDNQ)
    2. Create a new repository on HuggingFace (or use existing one)
    3. Set REPO_ID_INT8, REPO_ID_SDNQ_INT8, or REPO_ID_SDNQ_FP8 below to your repository name
"""

import os
import sys
import getpass
import argparse
from huggingface_hub import HfApi, upload_folder, login
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# ============================================================================
# Configuration
# ============================================================================
# Set these to your HuggingFace repositories (repo names should NOT include "fastdm")
# Format: "your-username/your-repo-name"
REPO_ID_INT8 = "rockapaper/wan22-14b-i2v-lightning-int8"
REPO_ID_SDNQ_INT8 = "rockapaper/fastdm-wan22-14b-i2v-lightning-sdnq-int8-svd-r32"
REPO_ID_SDNQ_FP8 = "rockapaper/wan22-14b-t2v-lightning-sdnq-fp8-svd-r32"

os.environ["HF_TOKEN"] = "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"
# Local paths to quantized models
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
QUANTIZED_MODEL_DIR_INT8 = os.path.join(MODELS_DIR, "fastdm_wan22_i2v_lightning_int8")
QUANTIZED_MODEL_DIR_SDNQ_INT8 = os.path.join(
    MODELS_DIR, "fastdm_wan22_i2v_lightning_sdnq_int8_svd_r32"
)
QUANTIZED_MODEL_DIR_SDNQ_FP8 = os.path.join(
    MODELS_DIR, "fastdm_wan22_i2v_lightning_sdnq_fp8_svd_r32"
)

# Upload options
REPO_TYPE = "model"  # "model" or "dataset"
PRIVATE = False  # Set to True for private repository


def get_model_config(model_type):
    """Get configuration for the specified model type."""
    if model_type == "int8":
        return {
            "dir": QUANTIZED_MODEL_DIR_INT8,
            "repo_id": REPO_ID_INT8,
            "commit_message": "Upload INT8 quantized Wan2.2 I2V Lightning model (optimum.quanto)",
            "script": "quantize_fastdm.py",
            "name": "INT8 (optimum.quanto)",
        }
    elif model_type == "sdnq_int8":
        return {
            "dir": QUANTIZED_MODEL_DIR_SDNQ_INT8,
            "repo_id": REPO_ID_SDNQ_INT8,
            "commit_message": "Upload SDNQ INT8+SVD r32 quantized Wan2.2 I2V Lightning model",
            "script": "quantize_fastdm_sdnq.py",
            "name": "SDNQ INT8+SVD r32",
        }
    elif model_type == "sdnq_fp8":
        return {
            "dir": QUANTIZED_MODEL_DIR_SDNQ_FP8,
            "repo_id": REPO_ID_SDNQ_FP8,
            "commit_message": "Upload SDNQ FP8+SVD r32 quantized Wan2.2 I2V Lightning model",
            "script": "quantize_fastdm_sdnq.py",
            "name": "SDNQ FP8+SVD r32",
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'int8', 'sdnq_int8', or 'sdnq_fp8'.")


def check_quantized_model(model_type):
    """Verify quantized model exists."""
    config = get_model_config(model_type)
    model_dir = config["dir"]

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Quantized model not found at {model_dir}\n"
            f"Please run {config['script']} first to create the quantized model."
        )

    # Check for config.json (INT8) or quantization_config.json (SDNQ)
    config_file = os.path.join(model_dir, "config.json")
    sdnq_config_file = os.path.join(model_dir, "quantization_config.json")

    if not os.path.exists(config_file) and not os.path.exists(sdnq_config_file):
        raise FileNotFoundError(
            f"Quantized model appears incomplete. Missing config files at {model_dir}"
        )

    print(f"✓ {config['name']} quantized model found at {model_dir}")

    # Calculate size
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
            file_count += 1

    size_gb = total_size / (1024**3)
    print(f"   Total size: {size_gb:.2f} GB ({file_count} files)")
    return size_gb, config


def get_hf_token():
    """Get HuggingFace token from various sources."""
    # 1. Check command-line argument
    if "--token" in sys.argv:
        idx = sys.argv.index("--token")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    
    # 2. Check environment variable
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # 3. Check if already logged in (has token cached)
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Already logged in as: {user['name']}")
        return None  # Already logged in, no token needed
    except Exception:
        pass
    
    # 4. Prompt user for token
    print("\n🔐 HuggingFace Authentication Required")
    print("   You can get your token from: https://huggingface.co/settings/tokens")
    print("   Or set HF_TOKEN environment variable or use --token argument")
    token = getpass.getpass("   Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return None
    
    return token


def check_hf_login():
    """Check if user is logged in to HuggingFace, login if needed."""
    # Try to get token
    token = get_hf_token()
    
    # If token provided, login with it
    if token:
        try:
            print("🔐 Logging in to HuggingFace Hub...")
            login(token=token)
            print("✓ Login successful")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False
    
    # Verify login
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"❌ Not logged in to HuggingFace Hub: {e}")
        print(f"\nPlease provide a valid token:")
        print(f"   1. Set HF_TOKEN environment variable")
        print(f"   2. Use --token argument")
        print(f"   3. Or run: huggingface-cli login")
        return False


def create_repo_if_needed(repo_id):
    """Create repository if it doesn't exist."""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
        print(f"✓ Repository {repo_id} already exists")
        return False
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        # Repository doesn't exist, create it
        # RepositoryNotFoundError is used in newer huggingface_hub versions
        # HfHubHTTPError with status_code 404 is used in older versions
        is_404 = False
        if isinstance(e, RepositoryNotFoundError):
            is_404 = True
        elif isinstance(e, HfHubHTTPError) and hasattr(e, 'status_code') and e.status_code == 404:
            is_404 = True
        
        if is_404:
            print(f"\n📦 Creating repository {repo_id}...")
            api.create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                private=PRIVATE,
                exist_ok=False,
            )
            print(f"✓ Repository created successfully")
            return True
        else:
            raise


def upload_model(model_type):
    """Upload quantized model to HuggingFace Hub."""
    print("\n" + "=" * 60)
    print(f"Uploading Quantized Model to HuggingFace Hub")
    print("=" * 60)

    # Check prerequisites and get config
    size_gb, config = check_quantized_model(model_type)
    
    print(f"Model Type:   {config['name']}")
    print(f"Repository:   {config['repo_id']}")
    print(f"Source:       {config['dir']}")
    print(f"Private:       {PRIVATE}")
    print("=" * 60)
    if not check_hf_login():
        return False

    # Create repository if needed
    create_repo_if_needed(config["repo_id"])

    # Upload model
    print(f"\n⬆️  Uploading model files to {config['repo_id']}...")
    print(f"   This may take a while for ~{size_gb:.2f} GB...")

    try:
        upload_folder(
            folder_path=config["dir"],
            repo_id=config["repo_id"],
            repo_type=REPO_TYPE,
            commit_message=config["commit_message"],
            ignore_patterns=[".git*", "__pycache__", "*.pyc"],
        )
        print(f"\n✅ Upload completed successfully!")
        print(f"\n📦 Model available at: https://huggingface.co/{config['repo_id']}")
        print(f"\n💡 To use this model, update your inference script:")
        if model_type == "int8":
            print(f"   from diffusers import WanImageToVideoPipeline")
            print(f"   pipe = WanImageToVideoPipeline.from_pretrained('{config['repo_id']}', torch_dtype=torch.bfloat16)")
        else:  # sdnq_int8 or sdnq_fp8
            print(f"   from sdnq import SDNQConfig  # registers SDNQ quantizer")
            print(f"   from diffusers import WanImageToVideoPipeline")
            print(f"   pipe = WanImageToVideoPipeline.from_pretrained('{config['repo_id']}')")
        return True
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload quantized FastDM model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["int8", "sdnq_int8", "sdnq_fp8"],
        default="int8",
        help="Model type to upload: 'int8' (optimum.quanto), 'sdnq_int8' (SDNQ INT8+SVD), or 'sdnq_fp8' (SDNQ FP8+SVD). Default: int8",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (or use HF_TOKEN env var or interactive prompt)",
    )
    return parser.parse_args()


def main():
    """Main upload workflow."""
    args = parse_args()
    model_type = args.model_type

    # Check if REPO_ID is configured
    config = get_model_config(model_type)
    if config["repo_id"].startswith("YOUR_USERNAME/"):
        print("\n" + "=" * 60)
        print("⚠️  Configuration Required")
        print("=" * 60)
        repo_var_map = {
            "int8": "REPO_ID_INT8",
            "sdnq_int8": "REPO_ID_SDNQ_INT8",
            "sdnq_fp8": "REPO_ID_SDNQ_FP8",
        }
        repo_var = repo_var_map.get(model_type, "REPO_ID")
        print(f"Please set {repo_var} in this script to your HuggingFace repository.")
        print("Format: 'your-username/your-repo-name' (should NOT include 'fastdm')")
        print("\nExample:")
        if model_type == "int8":
            print("  REPO_ID_INT8 = 'rockpaper/wan22-14b-i2v-lightning-sdnq-int8-svd-r32'")
        elif model_type == "sdnq_int8":
            print("  REPO_ID_SDNQ_INT8 = 'rockpaper/wan22-14b-i2v-lightning-sdnq-int8-svd-r32'")
        else:  # sdnq_fp8
            print("  REPO_ID_SDNQ_FP8 = 'rockpaper/wan22-14b-i2v-lightning-sdnq-fp8-svd-r32'")
        print("=" * 60)
        return

    # Add token to sys.argv if provided via --token argument
    # (so get_hf_token() can find it)
    if args.token:
        sys.argv.extend(["--token", args.token])

    success = upload_model(model_type)
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload Process Complete")
        print("=" * 60)
        print(f"Your {config['name']} quantized model is now available at:")
        print(f"  https://huggingface.co/{config['repo_id']}")
        print("\nYou can now download it directly on other machines using:")
        if model_type == "int8":
            print(f"  from diffusers import WanImageToVideoPipeline")
            print(f"  pipe = WanImageToVideoPipeline.from_pretrained('{config['repo_id']}', torch_dtype=torch.bfloat16)")
        else:  # sdnq_int8 or sdnq_fp8
            print(f"  from sdnq import SDNQConfig  # registers SDNQ quantizer")
            print(f"  from diffusers import WanImageToVideoPipeline")
            print(f"  pipe = WanImageToVideoPipeline.from_pretrained('{config['repo_id']}')")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Upload Failed")
        print("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()
