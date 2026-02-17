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
QWEN_REPO = "Qwen/Qwen-Image-Edit-2511"
AIO_REPO_ID = "Pr0f3ssi0n4ln00b/Phr00t-Qwen-Rapid-AIO"
DEFAULT_AIO_VERSION = "v19"

QWEN_DIR = os.path.join(MODELS_DIR, "qwen_image_edit_2511")
AIO_DIR = os.path.join(MODELS_DIR, "aio_transformers")

REALESRGAN_DIR = os.path.join(MODELS_DIR, "realesrgan")

os.makedirs(QWEN_DIR, exist_ok=True)
os.makedirs(AIO_DIR, exist_ok=True)

# ------------------------
# 1. Download Qwen base model (skip transformer/*)
# ------------------------
if not os.listdir(QWEN_DIR):
    try:
        print(f"⬇️ Downloading {QWEN_REPO} into {QWEN_DIR} (excluding transformer/*)")
        snapshot_download(
            QWEN_REPO,
            local_dir=QWEN_DIR,
            local_dir_use_symlinks=False,
            token="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",
            resume_download=True,
            ignore_patterns=["transformer/*"],  # skip heavy transformer weights (from AIO repo)
        )
        print(f"✓ Qwen repo ready at {QWEN_DIR}")
    except Exception as e:
        print(f"❌ Failed to download {QWEN_REPO}: {e}")
        traceback.print_exc()
else:
    print(f"✓ Qwen repo already exists at {QWEN_DIR}")

# ------------------------
# 2. Download transformer from Rapid AIO (default v19)
# ------------------------
aio_version = os.environ.get("AIO_VERSION", DEFAULT_AIO_VERSION)
# Normalize version (allow "19" -> "v19")
if not aio_version.startswith("v"):
    aio_version = f"v{aio_version}"

aio_subfolder = f"{aio_version}/transformer"
# Target path: models/aio_transformers/v19/transformer/
aio_version_dir = os.path.join(AIO_DIR, aio_version)
aio_target_dir = os.path.join(aio_version_dir, "transformer")

os.makedirs(aio_target_dir, exist_ok=True)

# Check if files are in wrong location (v19/v19/transformer/) and fix them first
wrong_path = os.path.join(aio_version_dir, aio_version, "transformer")
if os.path.exists(wrong_path) and os.listdir(wrong_path):
    print(f"⚠️ Found files in wrong location: {wrong_path}")
    print(f"📦 Moving files to correct location: {aio_target_dir}")
    # Move files from wrong location to correct location
    for file in os.listdir(wrong_path):
        src = os.path.join(wrong_path, file)
        dst = os.path.join(aio_target_dir, file)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"  ✓ Moved {file}")
    # Remove empty nested directories
    try:
        os.rmdir(wrong_path)
        os.rmdir(os.path.join(aio_version_dir, aio_version))
        print(f"  ✓ Cleaned up nested directories")
    except OSError as e:
        print(f"  ℹ️ Could not remove nested dirs: {e}")
    print(f"✓ AIO transformer moved to correct location: {aio_target_dir}")

if not os.listdir(aio_target_dir):
    try:
        print(f"⬇️ Downloading AIO transformer: {AIO_REPO_ID} / {aio_subfolder}")
        # CRITICAL: Download to AIO_DIR (parent) not aio_version_dir
        # Repo structure is: v19/transformer/*
        # If we download to models/aio_transformers/v19/, it creates v19/v19/transformer/*
        # If we download to models/aio_transformers/, it creates v19/transformer/* ✓
        snapshot_download(
            repo_id=AIO_REPO_ID,
            local_dir=AIO_DIR,  # Download to parent: models/aio_transformers/
            local_dir_use_symlinks=False,
            resume_download=True,
            token="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",
            allow_patterns=[f"{aio_version}/transformer/*"],
        )
        
        # Verify the files are in the correct location
        if os.path.exists(aio_target_dir) and os.listdir(aio_target_dir):
            print(f"✓ AIO transformer ready at {aio_target_dir}")
        else:
            raise FileNotFoundError(
                f"Transformer files not found at expected location: {aio_target_dir}. "
                f"Please check the download."
            )
    except Exception as e:
        print(f"❌ Failed to download AIO transformer: {e}")
        traceback.print_exc()
else:
    print(f"✓ AIO transformer already exists at {aio_target_dir}")

# ------------------------
# 3. Download LoRA adapters (all from ADAPTER_SPECS)
# ------------------------
ADAPTER_SPECS = {
    "Consistance": {
        "repo": "Pr0f3ssi0n4ln00b/QIE_2511_Consistency_Lora",
        "weights": "qe2511_consis_alpha_patched.safetensors",
    }, # Example prompt: transfer face expression from picture 2 to picture 1. Keep the same structure and color in picture 1. Avoid picture 2 character to be existed in picture 1.
    "Semirealistic-photo-detailer": {
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
    }, # input image: transform the background of the image to semi-realistic image
    "AnyPose": {
        "repo": "lilylilith/AnyPose",
        "weights": [
            "2511-AnyPose-base-000006250.safetensors",
            "2511-AnyPose-helper-00006000.safetensors",
        ],
    }, # input image: make the person in image 1 do the exact same pose of the person in image 2. Changing the style and background of the image of the person in image 1 is undesirable, so don't do it. The new pose should be pixel accurate to the pose we are trying to copy. The position of the arms and head and legs should be the same as the pose we are trying to copy. Change the field of view and angle to match exactly image 2. Head tilt and eye gaze pose should match the person in image 2.
    "Any2Real_2601": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601_A_final_patched.safetensors",
    }, # change the picture 1 to realistic photograph, [description of your image]
    "Hyperrealistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
    }, # "Transform into a hyper-realistic face portrait."
    "Ultrarealistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
    }, # "ultra-realistic portrait"
    "BFS-Best-FaceSwap-merge": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors",
    }, # send body first (image 1), then face (image 2).
    "F2P": {
        "repo": "DiffSynth-Studio/Qwen-Image-Edit-F2P",
        "weights": "edit_0928_lora_step40000.safetensors",
    }, # Given a cropped facial image as input, it generates full portrait images of the same person.
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
    }, #  You can control the camera to move up, down, left, and right, as well as rotate it to the left and right. You can also look down or up. The camera can be changed to a wide-angle or close-up shot.
    "Light-Restoration": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
    }, # remove the shadows and relight the image. PROMPT: "移除光影,使用柔和光线（无明显光斑和阴影）对图片进行重新照明"
    "Relight": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
    }, # There are two ways to use it: 1. Use the prompts in the dataset to change the image directly. 2. Use the trigger word: " 重新照明 "。 （For example:重新照明,使用窗帘透光（柔和漫射）的光线对图片进行重新照明） Meaning: Relighting, using the soft, diffuse light from the curtains to relight the image.
    "Edit-Skin": {
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
    },  # make the subjects skin details more prominent and natural
    "Next-Scene": {
        "repo": "lovis93/next-scene-qwen-image-lora-2509",
        "weights": "next-scene_lora-v2-3000.safetensors",
    },
    "Flat-Log": {
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
    },
}

LORA_DIR = os.path.join(MODELS_DIR, "lora_adapters")
os.makedirs(LORA_DIR, exist_ok=True)

print("\n⬇️ Downloading LoRA adapters...")
for adapter_name, spec in ADAPTER_SPECS.items():
    repo = spec["repo"]
    weights = spec["weights"]
    
    # Handle both single weight and list of weights
    if isinstance(weights, str):
        weights = [weights]
    
    adapter_dir = os.path.join(LORA_DIR, adapter_name.replace("/", "_"))
    os.makedirs(adapter_dir, exist_ok=True)
    
    for weight_file in weights:
        target_path = os.path.join(adapter_dir, weight_file)
        
        if os.path.exists(target_path):
            print(f"  ✓ {adapter_name}/{weight_file} already exists")
            continue
        
        try:
            print(f"  ⬇️ Downloading {adapter_name}/{weight_file} from {repo}")
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=weight_file,
                resume_download=True,
            )
            shutil.copy2(downloaded_path, target_path)
            print(f"  ✓ {adapter_name}/{weight_file} downloaded")
        except Exception as e:
            print(f"  ❌ Failed to download {adapter_name}/{weight_file}: {e}")
            traceback.print_exc()

print("✓ LoRA adapters download completed\n")

# ------------------------
# 4. Download Real-ESRGAN models (2x, 4x)
# ------------------------
os.makedirs(REALESRGAN_DIR, exist_ok=True)

# Download Real-ESRGAN models - try "plus" versions first, fallback to regular versions
realesrgan_models = [
    ("RealESRGAN_x2plus.pth", "RealESRGAN_x2.pth", 2),
    ("RealESRGAN_x4plus.pth", "RealESRGAN_x4.pth", 4),
]

for preferred_name, fallback_name, scale in realesrgan_models:
    model_file = os.path.join(REALESRGAN_DIR, preferred_name)
    
    if os.path.exists(model_file):
        print(f"✓ Real-ESRGAN {scale}x model already exists at {model_file}")
        continue
    
    # Try downloading preferred name first
    downloaded = False
    try:
        print(f"⬇️ Downloading Real-ESRGAN {scale}x model ({preferred_name})...")
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename=preferred_name,
            local_dir=REALESRGAN_DIR,
        )
        # Verify the file was downloaded
        if os.path.exists(model_file):
            downloaded = True
            print(f"✓ Real-ESRGAN {scale}x model downloaded successfully as {preferred_name}")
        else:
            raise FileNotFoundError(f"Downloaded file not found at expected path: {model_file}")
    except Exception as e:
        # If preferred name doesn't exist, try fallback name
        try:
            print(f"  ⚠️  {preferred_name} not found, trying {fallback_name}...")
            downloaded_path = hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename=fallback_name,
                local_dir=REALESRGAN_DIR,
            )
            # Copy/rename to expected name
            if os.path.exists(downloaded_path):
                shutil.copy2(downloaded_path, model_file)
                # Remove the fallback file if it's different
                if downloaded_path != model_file and os.path.exists(downloaded_path):
                    try:
                        os.remove(downloaded_path)
                    except:
                        pass
                if os.path.exists(model_file):
                    downloaded = True
                    print(f"✓ Real-ESRGAN {scale}x model downloaded and renamed to {preferred_name}")
                else:
                    raise FileNotFoundError(f"Failed to create {model_file} from {downloaded_path}")
            else:
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")
        except Exception as e2:
            print(f"❌ Failed to download Real-ESRGAN {scale}x model: {e2}")
            traceback.print_exc()

# ------------------------
# 5. Split large safetensors recursively
# ------------------------
def split_large_safetensors(base_dir, max_size=850 * 1024 * 1024):
    print(f"🔍 Scanning for .safetensors > {max_size/1024/1024}MB in {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                full_path = os.path.join(root, fn)
                size = os.path.getsize(full_path)
                if size > max_size:
                    print(f"⚡ Splitting {full_path} ({size/1024/1024:.1f} MB)")
                    subprocess.run(
                        ["split", "-b", "850M", full_path, f"{full_path}.part-"],
                        check=True,
                    )
                    os.remove(full_path)
                    print("✅ Split done, original removed.")
                else:
                    print(f"✔ {full_path} ({size/1024/1024:.1f} MB) kept as-is")


split_large_safetensors(MODELS_DIR)

# ------------------------
# 6. Generate runtime Dockerfile
# ------------------------
try:
    print("⚙️ Generating Dockerfile via generate_dockerfile.py ...")
    subprocess.run(["python3", os.path.join(SCRIPT_DIR, "generate_dockerfile.py")], check=True, cwd=SCRIPT_DIR)
    print("✅ Dockerfile generated successfully.")
except Exception as e:
    print(f"❌ Failed to generate Dockerfile: {e}")
    traceback.print_exc()

print("✅ Prep finished. Models ready in ./models and Dockerfile created.")
