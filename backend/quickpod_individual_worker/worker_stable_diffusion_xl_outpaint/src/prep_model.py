#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------------
# Config
# ------------------------
TARGET_DIR = os.path.join(MODELS_DIR, "sdxl_outpaint")

REPOS = {
    "controlnet": "xinsir/controlnet-union-sdxl-1.0",   # small, copy whole
    "vae": "madebyollin/sdxl-vae-fp16-fix",             # small, copy whole
    "pipeline": "SG161222/RealVisXL_V5.0_Lightning",    # large, split later
}

REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")
REALESRGAN_HF_REPO = "rockapaper/realesrgan"

os.makedirs(TARGET_DIR, exist_ok=True)

# ------------------------
# Helper: copy all from source to destination
# ------------------------
def copy_all(src, dst):
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(target_dir, f)
            if not os.path.exists(d):
                shutil.copy2(s, d)

# ------------------------
# 1. Download repositories
# ------------------------
for name, repo in REPOS.items():
    try:
        target_path = os.path.join(TARGET_DIR, name)
        os.makedirs(target_path, exist_ok=True)

        print(f"⬇️ Downloading {repo} into {target_path} ...")

        ignore = []
        if name == "pipeline":
            ignore = [
                # text_encoder
                "text_encoder/model.safetensors",
                # text_encoder_2
                "text_encoder_2/model.safetensors",
                # unet fp32 shards
                "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
                "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
                # vae fp32
                "vae/diffusion_pytorch_model.safetensors",
                # root fp32
                "RealVisXL_V5.0_Lightning_fp32.safetensors",
            ]

        snapshot_download(
            repo_id=repo,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),
            ignore_patterns=ignore,
        )
        print(f"✓ {name} ready at {target_path}")
    except Exception as e:
        print(f"❌ Failed to download {repo}: {e}")
        traceback.print_exc()

# ------------------------
# 2. Download Real-ESRGAN models (2x, 4x) from rockapaper/realesrgan
# ------------------------
os.makedirs(REALESRGAN_DIR, exist_ok=True)
for filename in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
    model_file = os.path.join(REALESRGAN_DIR, filename)
    if os.path.exists(model_file):
        print(f"✓ Real-ESRGAN {filename} already exists at {model_file}")
        continue
    try:
        print(f"⬇️ Downloading {filename} from {REALESRGAN_HF_REPO}...")
        hf_hub_download(
            repo_id=REALESRGAN_HF_REPO,
            filename=filename,
            local_dir=REALESRGAN_DIR,
            resume_download=True,
        )
        print(f"✓ {filename} downloaded to {REALESRGAN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        traceback.print_exc()

# ------------------------
# 3. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=500 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024:.0f}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "500M", full_path, f"{full_path}.part-"],
                        check=True
                    )
                    os.remove(full_path)
                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")

split_large_safetensors(TARGET_DIR)

# ------------------------
# 4. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", "generate_dockerfile.py"], check=True)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

# ------------------------
# Done
# ------------------------
print("\n✅ Final structure prepared at:", TARGET_DIR)
print("Set environment variables for inference:")
print(f"  CTRLNET_LOCAL_PATH={os.path.join(TARGET_DIR, 'controlnet')}")
print(f"  VAE_LOCAL_PATH={os.path.join(TARGET_DIR, 'vae')}")
print(f"  PIPELINE_LOCAL_PATH={os.path.join(TARGET_DIR, 'pipeline')}")
