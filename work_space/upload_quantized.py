#!/usr/bin/env python3
"""
Upload quantized model to HuggingFace Hub.

This script uploads a locally quantized model directory to your HuggingFace account,
allowing you to download it directly on other machines without re-quantizing.

Usage:
    # Basic usage (token from environment or prompt):
    python upload_quantized.py --repo-id "your-username/your-repo-name" --model-dir "./models/my_quantized_model"
    
    # With token as command-line argument:
    python upload_quantized.py --repo-id "your-username/your-repo-name" --model-dir "./models/my_quantized_model" --token your_token_here
    
    # With token as environment variable:
    HF_TOKEN=your_token_here python upload_quantized.py --repo-id "your-username/your-repo" --model-dir "./models/my_quantized_model"

Requirements:
    pip install huggingface_hub[hf_transfer]

Login Options:
    1. Set HF_TOKEN environment variable
    2. Pass --token argument
    3. Script will prompt for token if neither provided
    4. Or use existing login (huggingface-cli login)

Before running:
    1. Make sure you've run quantize_sdnq.py to create the quantized model
    2. Create a new repository on HuggingFace (or use existing one)
    3. Provide the repository name in format: "your-username/your-repo-name"
"""

import os
import sys
import getpass
import argparse
from huggingface_hub import HfApi, upload_folder, login
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# ============================================================================
# Default Configuration
# ============================================================================
REPO_TYPE = "model"  # "model" or "dataset"
PRIVATE = False  # Set to True for private repository


def get_hf_token(token_arg=None):
    """Get HuggingFace token from various sources."""
    # 1. Check command-line argument
    if token_arg:
        return token_arg
    
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


def check_hf_login(token=None):
    """Check if user is logged in to HuggingFace, login if needed."""
    # Try to get token
    token = get_hf_token(token)
    
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


def check_quantized_model(model_dir):
    """Verify quantized model exists and is valid."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Quantized model not found at {model_dir}\n"
            f"Please run quantize_sdnq.py first to create the quantized model."
        )

    # # Check for config.json or quantization_config.json
    # config_file = os.path.join(model_dir, "config.json")
    # sdnq_config_file = os.path.join(model_dir, "quantization_config.json")

    # if not os.path.exists(config_file) and not os.path.exists(sdnq_config_file):
    #     raise FileNotFoundError(
    #         f"Quantized model appears incomplete. Missing config files at {model_dir}"
    #     )

    # Calculate size
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
            file_count += 1

    size_gb = total_size / (1024**3)
    print(f"✓ Quantized model found at {model_dir}")
    print(f"   Total size: {size_gb:.2f} GB ({file_count} files)")
    return size_gb


def create_repo_if_needed(repo_id):
    """Create repository if it doesn't exist."""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
        print(f"✓ Repository {repo_id} already exists")
        return False
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        # Repository doesn't exist, create it
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


def upload_model(repo_id, model_dir, token=None):
    """Upload quantized model to HuggingFace Hub."""
    print("\n" + "=" * 60)
    print("Uploading Quantized Model to HuggingFace Hub")
    print("=" * 60)
    
    # Check model exists
    size_gb = check_quantized_model(model_dir)
    
    print(f"Repository:   {repo_id}")
    print(f"Source:       {model_dir}")
    print(f"Private:      {PRIVATE}")
    print("=" * 60)
    
    if not check_hf_login(token):
        return False

    # Create repository if needed
    create_repo_if_needed(repo_id)

    # Upload model
    print(f"\n⬆️  Uploading model files to {repo_id}...")
    print(f"   This may take a while for ~{size_gb:.2f} GB...")

    try:
        upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"Upload quantized model from {os.path.basename(model_dir)}",
            ignore_patterns=[".git*", "__pycache__", "*.pyc"],
        )
        print(f"\n✅ Upload completed successfully!")
        print(f"\n📦 Model available at: https://huggingface.co/{repo_id}")
        print(f"\n💡 To use this model:")
        print(f"   from sdnq import SDNQConfig  # registers SDNQ quantizer")
        print(f"   from diffusers import AutoPipelineForImageToVideo")
        print(f"   pipe = AutoPipelineForImageToVideo.from_pretrained('{repo_id}')")
        return True
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload quantized model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'your-username/your-repo-name')",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to quantized model directory (output from quantize_sdnq.py)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or use HF_TOKEN env var or interactive prompt)",
    )
    return parser.parse_args()


def main():
    """Main upload workflow."""
    args = parse_args()
    
    # Validate repo_id format
    if "/" not in args.repo_id:
        print("❌ Error: --repo-id must be in format 'username/repo-name'")
        print(f"   Got: {args.repo_id}")
        exit(1)
    
    # Validate model_dir exists
    if not os.path.isdir(args.model_dir):
        print(f"❌ Error: Model directory does not exist: {args.model_dir}")
        exit(1)

    success = upload_model(args.repo_id, args.model_dir, token=args.token)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload Process Complete")
        print("=" * 60)
        print(f"Your quantized model is now available at:")
        print(f"  https://huggingface.co/{args.repo_id}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Upload Failed")
        print("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()
