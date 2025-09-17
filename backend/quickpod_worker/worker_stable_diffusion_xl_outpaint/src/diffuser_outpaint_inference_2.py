# diffuser_outpaint.py
"""
Diffusers-based SDXL outpainting wrapper.
Formatted like simple_lama_inpaint.py but retains all original functionality.
"""

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image, ImageDraw
import torch

# ----------------------------
# Environment Setup
# ----------------------------
os.environ["XFORMERS_OFFLOAD_CPU"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
)

# Force HuggingFace model caches into /models_storage
CACHE_DIR = "/models_storage/all_models"
for var in ["HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"]:
    os.environ[var] = CACHE_DIR

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# ----------------------------
# Globals
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_OUTPUT_DIR = "./outputs_outpaint"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Lazy pipeline init (global cache)
PIPE = None


# ----------------------------
# Helpers
# ----------------------------
def _safe_b64decode(data: Union[str, bytes]) -> bytes:
    if isinstance(data, str):
        raw = data.encode()
    elif isinstance(data, bytes):
        raw = data
    else:
        raise ValueError("Base64 input must be str or bytes")

    missing_padding = len(raw) % 4
    if missing_padding:
        raw += b"=" * (4 - missing_padding)

    return base64.b64decode(raw)


def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load an image from path, PIL, bytes, or base64."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")

    s = str(img)
    if os.path.exists(s):
        return Image.open(s).convert("RGB")

    try:
        decoded = _safe_b64decode(s)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def _prepare_image_and_mask(
    image: Image.Image,
    width: int,
    height: int,
    overlap_percentage: int,
    resize_option: str,
    custom_resize_percentage: int,
    alignment: str,
    overlap_left: bool,
    overlap_right: bool,
    overlap_top: bool,
    overlap_bottom: bool,
):
    """Resize, position, and create a mask for outpainting."""
    target_size = (width, height)

    # Scale image to fit target
    scale_factor = min(width / image.width, height / image.height)
    source = image.resize(
        (int(image.width * scale_factor), int(image.height * scale_factor)),
        Image.LANCZOS,
    )

    # Resize option
    resize_map = {"Full": 100, "50%": 50, "33%": 33, "25%": 25}
    resize_percentage = resize_map.get(resize_option, custom_resize_percentage)
    resize_factor = resize_percentage / 100
    new_w = max(int(source.width * resize_factor), 64)
    new_h = max(int(source.height * resize_factor), 64)
    source = source.resize((new_w, new_h), Image.LANCZOS)

    # Overlaps
    overlap_x = max(int(new_w * (overlap_percentage / 100)), 1)
    overlap_y = max(int(new_h * (overlap_percentage / 100)), 1)

    # Alignment margins
    margins = {
        "Middle": ((width - new_w) // 2, (height - new_h) // 2),
        "Left": (0, (height - new_h) // 2),
        "Right": (width - new_w, (height - new_h) // 2),
        "Top": ((width - new_w) // 2, 0),
        "Bottom": ((width - new_w) // 2, height - new_h),
    }
    margin_x, margin_y = margins.get(alignment, margins["Middle"])
    margin_x = max(0, min(margin_x, width - new_w))
    margin_y = max(0, min(margin_y, height - new_h))

    # Background + mask
    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    white_patch = 2
    left = margin_x + (overlap_x if overlap_left else white_patch)
    right = margin_x + new_w - (overlap_x if overlap_right else white_patch)
    top = margin_y + (overlap_y if overlap_top else white_patch)
    bottom = margin_y + new_h - (overlap_y if overlap_bottom else white_patch)
    mask_draw.rectangle([left, top, right, bottom], fill=0)

    return background, mask


def _init_pipeline():
    """Load SDXL + ControlNet pipeline once."""
    global PIPE
    if PIPE is not None:
        return PIPE

    config_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", "config_promax.json")
    config = ControlNetModel_Union.load_config(config_file)
    controlnet = ControlNetModel_Union.from_config(config)

    model_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors")
    state_dict = load_state_dict(model_file)
    controlnet, *_ = ControlNetModel_Union._load_pretrained_model(
        controlnet, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", list(state_dict.keys())
    )
    controlnet.to(device=DEVICE, dtype=DTYPE)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE).to(DEVICE)

    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=DTYPE,
        vae=vae,
        controlnet=controlnet,
        variant="fp16",
    ).to(DEVICE)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    PIPE = pipe
    return PIPE


# ----------------------------
# Core Function
# ----------------------------
def run_inference_diffuser_outpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Diffusers SDXL outpainting.
    Required params:
        init_image (base64, path, or PIL)
        width, height
        overlap_percentage
        resize_option ("Full", "50%", "33%", "25%", or "Custom")
        custom_resize_percentage
        alignment ("Middle", "Left", "Right", "Top", "Bottom")
        overlap_left/right/top/bottom (bools)
        num_inference_steps
        output_dir (optional)
    """
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required")

    pipe = _init_pipeline()
    background, mask = _prepare_image_and_mask(
        init_image,
        params["width"],
        params["height"],
        params["overlap_percentage"],
        params["resize_option"],
        params.get("custom_resize_percentage", 100),
        params["alignment"],
        params["overlap_left"],
        params["overlap_right"],
        params["overlap_top"],
        params["overlap_bottom"],
    )

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    if prompt is None or prompt.strip() == "":
        final_prompt = ""
    else:
        final_prompt = f"{prompt}, high quality, 4k"
        
    prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = pipe.encode_prompt(final_prompt, DEVICE, True)

    result = None
    for out in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pooled_embeds,
        negative_pooled_prompt_embeds=neg_pooled,
        image=cnet_image,
        num_inference_steps=params.get("num_inference_steps", 8),
    ):
        result = out

    if result is None:
        raise RuntimeError("Generation failed, no image returned")

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"diffuser_outpaint_{job_id}.png")
    result.save(out_path)
    return os.path.abspath(out_path)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    input_path = "./input_image.jpg"
    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    test_params = {
        "init_image": img_b64,
        "width": 1280,
        "height": 720,
        "overlap_percentage": 10,
        "resize_option": "Full",
        "custom_resize_percentage": 50,
        "alignment": "Middle",
        "overlap_left": True,
        "overlap_right": True,
        "overlap_top": True,
        "overlap_bottom": True,
        "num_inference_steps": 8,
        "output_dir": "./outputs_outpaint",
    }

    print("Using device:", DEVICE)
    result_path = run_inference_diffuser_outpaint("test", "", test_params)
    print("✅ Saved:", result_path)
