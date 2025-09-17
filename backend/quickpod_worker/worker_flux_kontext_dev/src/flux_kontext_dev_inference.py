# flux_kontext_diffusers.py
# Usage: python flux_kontext_diffusers.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), seed, CFG, steps

import os
os.environ["HF_HOME"] = "/models_storage/all_models/flux_kontext_dev"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/flux_kontext_dev"
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import torch
torch.cuda.empty_cache()

from diffusers import FluxKontextPipeline
from huggingface_hub import hf_hub_download
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

# -------------------------
# Paths / Model config
# -------------------------
MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "nunchaku-tech/nunchaku-flux.1-kontext-dev",
        f"svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
    ),
}

def download_models():
    local_paths = {}
    for key, (repo_id, filename) in MODEL_FILES.items():
        print(f"[Download] {key} → {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=MODEL_DIR,
            resume_download=True
        )
        local_paths[key] = path
    return local_paths

MODEL_PATHS = download_models()
TRANSFORMER_PATH = MODEL_PATHS["DIFFUSION_MODEL"]

# -------------------------
# Initialize Diffusers pipeline
# -------------------------
transformer = NunchakuFluxTransformer2dModel.from_pretrained(TRANSFORMER_PATH)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
).to("cuda")

# ### LoRA Related Code ###
# # """
# # first convert the LoRA to nunchaku format using the command below, then load it in the code.
# # python -m nunchaku.lora.flux.convert \
# #     --quant-path mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors \
# #     --lora-path /models_storage/all_models/nunchaku-flux_kontext_nsfw_lora/background_remover_v0_flux_kontext.safetensors \
# #     --lora-format diffusers \
# #     --output-root /models_storage/all_models/nunchaku-flux_kontext_nsfw_lora \
# #     --lora-name background_remover_v0_flux_kontext_nunchaku
# # """


# transformer.update_lora_params(
#     lora_path
#     # "nunchaku-tech/nunchaku-test-models/relight-kontext-lora-single-caption_comfy.safetensors"
#     # "linoyts/relight-kontext-lora-single-caption/relight-kontext-lora-single-caption.safetensors"
# )  # Path to your LoRA safetensors, can also be a remote HuggingFace path
# transformer.set_lora_strength(1)  # Your LoRA strength here
# ### End of LoRA Related Code ###

# -------------------------
# LoRA Setup (optional)
# -------------------------
def maybe_apply_lora(transformer, params: Dict[str, Any]):
    """
    Enable LoRA only if params contain lora_path (and optionally lora_strength).
    """
    apply_lora = params.get("apply_lora", False)
    if not apply_lora:
        print("[LoRA] Skipped (apply_lora is False)")
        return
    lora_path = "/models_storage/all_models/nunchaku-flux_kontext_nsfw_lora/clothes_remover_v0_flux_kontext.safetensors"

    if lora_path:
        print(f"[LoRA] Loading LoRA from {lora_path}")
        transformer.update_lora_params(lora_path)
        lora_strength = params.get("lora_strength", 1.0)
        transformer.set_lora_strength(float(lora_strength))
        print(f"[LoRA] Strength set to {lora_strength}")
    else:
        print("[LoRA] Skipped (no lora_path in params)")


# -------------------------
# Helpers
# -------------------------
def _load_image(img_b64: str) -> Image.Image:
    decoded_bytes = base64.b64decode(img_b64)
    return Image.open(BytesIO(decoded_bytes)).convert("RGB")

def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
    """Return width/height close to input, within max_dim, multiple-of-64."""
    w, h = img.size
    aspect = w / h
    if w > h:
        new_w = min(w, max_dim)
        new_h = int(new_w / aspect)
    else:
        new_h = min(h, max_dim)
        new_w = int(new_h * aspect)
    # Round to multiple-of-64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h

# -------------------------
# Main inference
# -------------------------
def run_inference_flux_kontext_dev(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Flux-Kontext inference (image editing) with dynamic resolution using Diffusers.
    Returns the local path to the saved PNG file.
    """
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    init_images = params.get("images")
    if init_images is None:
        raise ValueError("init_images (base64) is required")
    image = _load_image(init_images)

    # Decide output resolution (close to input, multiple-of-64)
    width, height = _get_dynamic_size(image, max_dim=1152)

    # Inference parameters
    steps = params.get("sample_steps", 28)
    cfg = params.get("cfg_scale", 1.5)

    # Run Diffusers pipeline
    output = pipeline(
        image=image,          # keep original image as input
        prompt=prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        width=width,
        height=height
    ).images[0]

    out_path = os.path.join(output_dir, f"flux_kontext_{job_id}.png")
    output.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "bikini_dynamic"
    prompt = "Completely remove the subjects clothes. Her clothes are now completely removed to expose her (small breasts) and clean shaven vagina. Maintain identical camera angle, framing, and perspective and make sure to leave her facial features identical. The rest of the image remains the same, only remove her clothes to expose her naked body, mnmk"
    input_path = "/quickpod/quickpod_workers/worker_flux_kontext_dev/input_image.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 2.5,
        "output_dir": "./outputs",
        "images": img_b64,
        "apply_lora": True,
        "lora_strength": 1.0
    }

    out = run_inference_flux_kontext_dev(job_id, prompt, params)
    print("✅ Output image saved at:", out)
