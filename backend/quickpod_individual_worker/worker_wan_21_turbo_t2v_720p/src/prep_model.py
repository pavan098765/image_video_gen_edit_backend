#!/usr/bin/env python3
"""
Prep model for WAN 2.1 TurboDiffusion T2V 720p worker.
Downloads VAE, umT5, and TurboWan2.1-T2V-14B-720P-quant.pth. Splits large files into .part-*.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
MAX_CHUNK_BYTES = 700 * 1024 * 1024

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
VAE_FILENAME = "Wan2.1_VAE.pth"
TEXT_ENCODER_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"
T2V_REPO = "TurboDiffusion/TurboWan2.1-T2V-14B-720P"
DIT_FILENAME = "TurboWan2.1-T2V-14B-720P-quant.pth"


def download_file(repo_id: str, filename: str, dest_dir: str) -> str:
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"✓ Already exists: {path}")
        return path
    print(f"⬇️ Downloading {repo_id}/{filename} -> {dest_dir}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, local_dir_use_symlinks=False)
    return path


def split_large_files(base_dir: str, max_size: int = MAX_CHUNK_BYTES):
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
                print(f"⚡ Splitting {full_path} ({size / 1024 / 1024:.1f} MB)")
                try:
                    subprocess.run(
                        ["split", "-b", f"{chunk_mb}M", full_path, full_path + ".part-"],
                        check=True,
                        capture_output=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"⚠️  split failed: {e}")
                    continue
                os.remove(full_path)
                split_count += len(glob.glob(full_path + ".part-*"))
                print(f"✅ Split done")
    return split_count


def main():
    print("=== WAN 2.1 TurboDiffusion T2V 720p - prep_model ===\n")
    download_file(VAE_REPO, VAE_FILENAME, CHECKPOINTS_DIR)
    download_file(VAE_REPO, TEXT_ENCODER_FILENAME, CHECKPOINTS_DIR)
    download_file(T2V_REPO, DIT_FILENAME, CHECKPOINTS_DIR)

    rife_dir = os.path.join(MODELS_DIR, "rife")
    if not os.path.exists(rife_dir) or not os.listdir(rife_dir):
        try:
            os.makedirs(rife_dir, exist_ok=True)
            snapshot_download("AlexWortega/RIFE", local_dir=rife_dir, local_dir_use_symlinks=False, resume_download=True)
        except Exception as e:
            print(f"⚠️  RIFE: {e}")

    realesrgan_dir = os.path.join(MODELS_DIR, "realesrgan")
    os.makedirs(realesrgan_dir, exist_ok=True)
    for factor in [2, 4, 8]:
        fn = f"RealESRGAN_x{factor}.pth"
        if not os.path.exists(os.path.join(realesrgan_dir, fn)):
            try:
                hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename=fn, local_dir=realesrgan_dir)
            except Exception as e:
                print(f"⚠️  {fn}: {e}")

    total_splits = split_large_files(MODELS_DIR)
    print(f"📊 Total split files: {total_splits}")

    try:
        print("Generating Dockerfile (image build will clone TurboDiffusion from git)...")
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "generate_dockerfile.py")], check=True, cwd=SCRIPT_DIR)
        print("✅ Dockerfile generated")
    except Exception as e:
        print(f"❌ generate_dockerfile: {e}")
        traceback.print_exc()
    print("\n✅ Prep finished. WAN 2.1 Turbo T2V 720p ready.")


if __name__ == "__main__":
    main()
