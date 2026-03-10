#!/usr/bin/env python3
"""
Prep model for WAN 2.2 TurboDiffusion I2V 720p worker.
Downloads VAE, umT5 text encoder, and TurboWan2.2 I2V 720p quantized checkpoints (high + low).
Splits any file > 700MB into .part-* chunks for Docker layer size limits.
The generated Dockerfile will clone the TurboDiffusion repo and install from source in the image (no clone in prep_model).
"""
import os
import sys
import glob
import subprocess
import traceback

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("pip install huggingface_hub required")
    sys.exit(1)

# ------------------------
# Paths
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
MAX_CHUNK_BYTES = 700 * 1024 * 1024  # 700MB

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# VAE and text encoder (Wan-AI/Wan2.1-T2V-1.3B) - shared for Wan2.1 and Wan2.2
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
VAE_FILENAME = "Wan2.1_VAE.pth"
TEXT_ENCODER_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"

# I2V 720p quantized checkpoints
I2V_REPO = "TurboDiffusion/TurboWan2.2-I2V-A14B-720P"
HIGH_NOISE_FILENAME = "TurboWan2.2-I2V-A14B-high-720P-quant.pth"
LOW_NOISE_FILENAME = "TurboWan2.2-I2V-A14B-low-720P-quant.pth"


def download_file(repo_id: str, filename: str, dest_dir: str) -> str:
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"✓ Already exists: {path}")
        return path
    print(f"⬇️ Downloading {repo_id}/{filename} -> {dest_dir}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, local_dir_use_symlinks=False)
    return path


def split_large_files(base_dir: str, max_size: int = MAX_CHUNK_BYTES):
    """Split files larger than max_size into .part-* chunks (for .pth, .safetensors, etc.)."""
    chunk_mb = max_size // (1024 * 1024)
    print(f"🔍 Scanning for files > {chunk_mb}MB in {base_dir}")
    split_count = 0
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if ".part-" in fn:
                continue
            full_path = os.path.join(root, fn)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue
            if size > max_size:
                print(f"⚡ Splitting {full_path} ({size / 1024 / 1024:.1f} MB) into {chunk_mb}MB chunks")
                try:
                    subprocess.run(
                        ["split", "-b", f"{chunk_mb}M", full_path, full_path + ".part-"],
                        check=True,
                        capture_output=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"⚠️  split failed (use Linux or WSL): {e}")
                    continue
                os.remove(full_path)
                parts = sorted(glob.glob(full_path + ".part-*"))
                split_count += len(parts)
                print(f"✅ Split into {len(parts)} parts")
    return split_count


def main():
    print("=== WAN 2.2 TurboDiffusion I2V 720p - prep_model ===\n")

    # 1. VAE and text encoder
    download_file(VAE_REPO, VAE_FILENAME, CHECKPOINTS_DIR)
    download_file(VAE_REPO, TEXT_ENCODER_FILENAME, CHECKPOINTS_DIR)

    # 2. I2V 720p quantized high + low
    download_file(I2V_REPO, HIGH_NOISE_FILENAME, CHECKPOINTS_DIR)
    download_file(I2V_REPO, LOW_NOISE_FILENAME, CHECKPOINTS_DIR)

    # 3. Optional: RIFE and Real-ESRGAN (same as lightx2v workers)
    rife_dir = os.path.join(MODELS_DIR, "rife")
    if not os.path.exists(rife_dir) or not os.listdir(rife_dir):
        try:
            print("⬇️ Downloading RIFE model weights...")
            os.makedirs(rife_dir, exist_ok=True)
            snapshot_download("AlexWortega/RIFE", local_dir=rife_dir, local_dir_use_symlinks=False, resume_download=True)
            print("✓ RIFE downloaded")
        except Exception as e:
            print(f"⚠️  RIFE download failed: {e}")

    realesrgan_dir = os.path.join(MODELS_DIR, "realesrgan")
    os.makedirs(realesrgan_dir, exist_ok=True)
    for factor in [2, 4, 8]:
        fn = f"RealESRGAN_x{factor}.pth"
        path = os.path.join(realesrgan_dir, fn)
        if not os.path.exists(path):
            try:
                hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename=fn, local_dir=realesrgan_dir)
                print(f"✓ {fn} downloaded")
            except Exception as e:
                print(f"⚠️  {fn} failed: {e}")

    # 4. Split large files (> 700MB)
    total_splits = split_large_files(MODELS_DIR)
    print(f"📊 Total split files: {total_splits}")

    # 5. Generate Dockerfile
    try:
        print("Generating Dockerfile (image build will clone TurboDiffusion from git)...")
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
            check=True,
            cwd=SCRIPT_DIR,
        )
        print("✅ Dockerfile generated")
    except Exception as e:
        print(f"❌ generate_dockerfile failed: {e}")
        traceback.print_exc()

    print("\n✅ Prep finished. WAN 2.2 Turbo I2V 720p ready.")


if __name__ == "__main__":
    main()
