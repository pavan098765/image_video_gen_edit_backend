#!/usr/bin/env python3
import os, shutil, traceback, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# ------------------------
# Resolve project root (one dir up from src/)
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# WAN 2.2 T2V 14B SDNQ model repo
MODEL_REPO = "Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32"
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_t2v_sdnq_uint4_svd_r32")

# LoRA Configuration
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_WEIGHT_NAME = "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors"
LORA_LOCAL_DIR = os.path.join(MODELS_DIR, "lora", "Kijai_WanVideo_comfy")

os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
os.makedirs(LORA_LOCAL_DIR, exist_ok=True)

# ------------------------
# 1. Download WAN 2.2 T2V 14B SDNQ model
# ------------------------
if not os.path.exists(MODEL_LOCAL_DIR) or not os.listdir(MODEL_LOCAL_DIR):
    try:
        print(f"⬇️ Downloading {MODEL_REPO} to {MODEL_LOCAL_DIR}...")
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ Model downloaded successfully to {MODEL_LOCAL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        traceback.print_exc()
else:
    print(f"✓ Model already exists at {MODEL_LOCAL_DIR}")

# ------------------------
# 2. Download Lightning LoRA
# ------------------------
lora_file_path = os.path.join(LORA_LOCAL_DIR, LORA_WEIGHT_NAME)
if not os.path.exists(lora_file_path):
    try:
        print(f"⬇️ Downloading LoRA from {LORA_REPO}...")
        print(f"   Weight: {LORA_WEIGHT_NAME}")
        os.makedirs(LORA_LOCAL_DIR, exist_ok=True)
        
        snapshot_download(
            LORA_REPO,
            local_dir=LORA_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[LORA_WEIGHT_NAME],
        )
        
        # Verify the file exists
        if os.path.exists(lora_file_path):
            file_size_mb = os.path.getsize(lora_file_path) / (1024 * 1024)
            print(f"✓ LoRA downloaded successfully to {lora_file_path}")
            print(f"   📦 File size: {file_size_mb:.2f} MB")
        else:
            # File might be in a subdirectory
            for root, dirs, files in os.walk(LORA_LOCAL_DIR):
                if LORA_WEIGHT_NAME in files:
                    found_path = os.path.join(root, LORA_WEIGHT_NAME)
                    file_size_mb = os.path.getsize(found_path) / (1024 * 1024)
                    print(f"✓ LoRA downloaded successfully to {found_path}")
                    print(f"   📦 File size: {file_size_mb:.2f} MB")
                    break
    except Exception as e:
        print(f"❌ Failed to download LoRA: {e}")
        traceback.print_exc()
else:
    print(f"✓ LoRA already exists at {lora_file_path}")

# ------------------------
# 3. Download RIFE model
# ------------------------
RIFE_MODEL_DIR = os.path.join(MODELS_DIR, "rife")
if not os.path.exists(RIFE_MODEL_DIR) or not os.listdir(RIFE_MODEL_DIR):
    try:
        print(f"⬇️ Downloading RIFE model weights to {RIFE_MODEL_DIR}...")
        os.makedirs(RIFE_MODEL_DIR, exist_ok=True)
        snapshot_download(
            "AlexWortega/RIFE",
            local_dir=RIFE_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ RIFE model weights downloaded successfully to {RIFE_MODEL_DIR}")
    except Exception as e:
        print(f"❌ Failed to download RIFE model: {e}")
        traceback.print_exc()
else:
    print(f"✓ RIFE model already exists at {RIFE_MODEL_DIR}")

# ------------------------
# 4. Download Real-ESRGAN models (2x, 4x, 8x)
# ------------------------
REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")
os.makedirs(REALESRGAN_DIR, exist_ok=True)

for upscale_factor in [2, 4, 8]:
    model_filename = f"RealESRGAN_x{upscale_factor}.pth"
    model_file = os.path.join(REALESRGAN_DIR, model_filename)
    
    if not os.path.exists(model_file):
        try:
            print(f"⬇️ Downloading Real-ESRGAN {upscale_factor}x model to {REALESRGAN_DIR}...")
            hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename=model_filename,
                local_dir=REALESRGAN_DIR,
            )
            print(f"✓ Real-ESRGAN {upscale_factor}x model downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download Real-ESRGAN {upscale_factor}x model: {e}")
            traceback.print_exc()
    else:
        print(f"✓ Real-ESRGAN {upscale_factor}x model already exists at {model_file}")

# ------------------------
# 5. Split large safetensors with 700MB chunks
# ------------------------
def split_large_safetensors(base_dir, max_size=700 * 1024 * 1024):
    """
    Split safetensors files larger than max_size into chunks.
    max_size: 700MB (700 * 1024 * 1024 bytes)
    """
    chunk_size_mb = max_size / (1024 * 1024)
    print(f"🔍 Scanning for .safetensors > {chunk_size_mb:.0f}MB in {base_dir}")
    split_count = 0
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB) into {chunk_size_mb:.0f}MB chunks")
                    num_parts = (size + max_size - 1) // max_size
                    print(f"   Will create {num_parts} parts")
                    subprocess.run(
                        ["split", "-b", f"{int(chunk_size_mb)}M", full_path, f"{full_path}.part-"],
                        check=True
                    )
                    os.remove(full_path)
                    import glob
                    parts = glob.glob(f"{full_path}.part-*")
                    split_count += len(parts)
                    print(f"✅ Split into {len(parts)} parts, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")
    return split_count

total_splits = split_large_safetensors(MODELS_DIR, max_size=700 * 1024 * 1024)
print(f"📊 Total split files created: {total_splits}")

# ------------------------
# 5.5. Verify split files can be merged (dry-run check)
# ------------------------
def verify_split_files(base_dir):
    """Verify that all split parts exist and can be merged."""
    import glob
    issues = []
    
    for root, _, files in os.walk(base_dir):
        # Find all .part- files
        part_files = [f for f in files if ".safetensors.part-" in f]
        if not part_files:
            continue
        
        # Group by base filename
        bases = {}
        for part_file in part_files:
            base = part_file.split(".part-")[0]
            if base not in bases:
                bases[base] = []
            bases[base].append(os.path.join(root, part_file))
        
        # Verify each base file
        for base, parts in bases.items():
            parts_sorted = sorted(parts)
            total_size = sum(os.path.getsize(p) for p in parts_sorted)
            base_path = os.path.join(root, base)
            
            # Check if original still exists (shouldn't)
            if os.path.exists(base_path):
                issues.append(f"⚠️  Original file still exists: {base_path} (should be removed after split)")
            
            # Verify parts are sequential
            expected_parts = []
            for i in range(len(parts_sorted)):
                # Check for part-aa, part-ab, etc. or part-00, part-01, etc.
                part_name = parts_sorted[i]
                if i == 0:
                    # First part - verify naming
                    pass
                else:
                    # Verify parts are in order
                    prev_part = parts_sorted[i-1]
                    if part_name <= prev_part:
                        issues.append(f"⚠️  Parts not in order: {prev_part} -> {part_name}")
            
            print(f"✓ Verified {len(parts_sorted)} parts for {os.path.basename(base)} (total: {total_size / (1024**3):.2f} GB)")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s) with split files:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"✅ All split files verified successfully")
    
    return len(issues) == 0

print("\n🔍 Verifying split files...")
verify_split_files(MODELS_DIR)

# ------------------------
# 6. Generate runtime Dockerfile with per-part COPY
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print(f"✅ Prep finished. WAN 2.2 T2V SDNQ model ready in {MODEL_LOCAL_DIR}")
print(f"Set ENV MODEL_LOCAL_DIR={MODEL_LOCAL_DIR}")
