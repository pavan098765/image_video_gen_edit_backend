import os
import io
import base64
from typing import Any, Dict
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from PIL import Image, ImageDraw
from upscaler import upscale_image

# Custom local imports
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


# --- Default Local Model Paths ---
CTRLNET_LOCAL = os.environ.get(
    "CTRLNET_LOCAL_PATH",
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/controlnet",
)
VAE_LOCAL = os.environ.get(
    "VAE_LOCAL_PATH",
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/vae",
)
PIPELINE_LOCAL = os.environ.get(
    "PIPELINE_LOCAL_PATH",
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/pipeline",
)


# ==========================================================
# Utility Functions
# ==========================================================

def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def _parse_ratio(aspect_ratio: str):
    """Parse strings like '16:9' or '1:1' -> (w_ratio, h_ratio)."""
    try:
        a, b = aspect_ratio.split(":")
        return max(1, int(a)), max(1, int(b))
    except Exception:
        return 1, 1


def compute_target_canvas(
    image,
    aspect_ratio: str = "9:16",
    alignment: str = "Middle",
    max_dim: int = 1152,
    expansion_factor: float = 1.25,
):
    """
    Smartly compute target canvas dimensions and paste offsets
    that preserve input image scale (no distortion) and expand
    based on alignment + aspect ratio.
    """
    orig_w, orig_h = image.size
    w_ratio, h_ratio = _parse_ratio(aspect_ratio)
    target_ratio = w_ratio / h_ratio
    orig_ratio = orig_w / orig_h if orig_h > 0 else 1.0

    expand_width_preferred = target_ratio > orig_ratio
    if alignment in ("Left", "Right"):
        expand_width_preferred = True
    elif alignment in ("Top", "Bottom"):
        expand_width_preferred = False

    if expand_width_preferred:
        cand_h = min(orig_h, max_dim)
        cand_w = int(round(cand_h * target_ratio))
        if cand_w <= orig_w:
            cand_w = max(int(orig_w * expansion_factor), orig_w, cand_w)
    else:
        cand_w = min(orig_w, max_dim)
        cand_h = int(round(cand_w / target_ratio))
        if cand_h <= orig_h:
            cand_h = max(int(orig_h * expansion_factor), orig_h, cand_h)

    long_side = max(cand_w, cand_h)
    if long_side > max_dim:
        scale_down = max_dim / long_side
        cand_w = int(round(cand_w * scale_down))
        cand_h = int(round(cand_h * scale_down))

    cand_w = max(cand_w, orig_w)
    cand_h = max(cand_h, orig_h)

    def round64(x):
        return max(64, (int(x) // 64) * 64)

    target_w = round64(cand_w)
    target_h = round64(cand_h)

    if target_w < orig_w:
        target_w = round64(orig_w)
    if target_h < orig_h:
        target_h = round64(orig_h)

    max_source_w, max_source_h = target_w, target_h
    scale_factor = min(1.0, min(max_source_w / orig_w, max_source_h / orig_h))
    final_image_w = int(round(orig_w * scale_factor))
    final_image_h = int(round(orig_h * scale_factor))

    if alignment == "Left":
        paste_x = 0
        paste_y = (target_h - final_image_h) // 2
    elif alignment == "Right":
        paste_x = target_w - final_image_w
        paste_y = (target_h - final_image_h) // 2
    elif alignment == "Top":
        paste_x = (target_w - final_image_w) // 2
        paste_y = 0
    elif alignment == "Bottom":
        paste_x = (target_w - final_image_w) // 2
        paste_y = target_h - final_image_h
    elif alignment == "All Sides":
        paste_x = (target_w - final_image_w) // 2
        paste_y = (target_h - final_image_h) // 2
    else:
        paste_x = (target_w - final_image_w) // 2
        paste_y = (target_h - final_image_h) // 2

    return {
        "target_w": target_w,
        "target_h": target_h,
        "final_image_w": final_image_w,
        "final_image_h": final_image_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
    }


def prepare_image_and_mask(
    image,
    width,
    height,
    overlap_percentage,
    resize_option,
    custom_resize_percentage,
    alignment,
    overlap_left,
    overlap_right,
    overlap_top,
    overlap_bottom,
    pre_resized_source=None,
    paste_x=None,
    paste_y=None,
):
    """
    Create base background + mask with optional pre-resized source image and
    explicit paste offsets (used by our smart canvas calculator).
    """
    target_size = (width, height)
    if pre_resized_source is not None:
        source = pre_resized_source
        new_width, new_height = source.size
        margin_x = paste_x or (width - new_width) // 2
        margin_y = paste_y or (height - new_height) // 2
    else:
        # Fallback: legacy scaling logic
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        source = image.resize((new_width, new_height), Image.LANCZOS)
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    overlap_x = max(int(source.width * (overlap_percentage / 100)), 1)
    overlap_y = max(int(source.height * (overlap_percentage / 100)), 1)
    white_gaps_patch = 2

    left_overlap = margin_x + (overlap_x if overlap_left else white_gaps_patch)
    right_overlap = margin_x + source.width - (overlap_x if overlap_right else white_gaps_patch)
    top_overlap = margin_y + (overlap_y if overlap_top else white_gaps_patch)
    bottom_overlap = margin_y + source.height - (overlap_y if overlap_bottom else white_gaps_patch)

    mask_draw.rectangle(
        [(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0
    )
    return background, mask


# ==========================================================
# ENTRYPOINT FUNCTION
# ==========================================================

def run_inference_sdxl_outpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Entry point: Outpainting with auto aspect ratio + alignment expansion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ControlNet ---
    controlnet_config_path = os.path.join(CTRLNET_LOCAL, "config_promax.json")
    controlnet_weights_path = os.path.join(
        CTRLNET_LOCAL, "diffusion_pytorch_model_promax.safetensors"
    )

    config = ControlNetModel_Union.load_config(controlnet_config_path)
    controlnet_model = ControlNetModel_Union.from_config(config)
    state_dict = load_state_dict(controlnet_weights_path)
    loaded_keys = list(state_dict.keys())
    controlnet_model, *_ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model,
        state_dict,
        controlnet_weights_path,
        CTRLNET_LOCAL,
        loaded_keys,
    )
    controlnet_model = controlnet_model.to(device=device, dtype=dtype)

    # --- Load VAE and Pipeline ---
    vae = AutoencoderKL.from_pretrained(VAE_LOCAL, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        PIPELINE_LOCAL,
        torch_dtype=dtype,
        vae=vae,
        controlnet=controlnet_model,
        variant="fp16",
    ).to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    # --- Load Input Image ---
    init_image = params.get("init_image") or params.get("images")
    if isinstance(init_image, str):
        if os.path.exists(init_image):
            image = Image.open(init_image)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(init_image)))
    elif isinstance(init_image, Image.Image):
        image = init_image
    elif isinstance(init_image, list) and isinstance(init_image[0], Image.Image):
        image = init_image[0]
    else:
        raise ValueError("Invalid input image format in params['init_image'] or ['images'].")

    # --- Compute Smart Canvas ---
    aspect_ratio = params.get("aspect_ratio", "9:16")
    alignment = params.get("alignment", "Middle")
    max_dim = int(params.get("max_dim", 1152))
    expansion_factor = float(params.get("expansion_factor", 1.25))

    canvas = compute_target_canvas(image, aspect_ratio, alignment, max_dim, expansion_factor)
    resized_source = image.resize(
        (canvas["final_image_w"], canvas["final_image_h"]), Image.LANCZOS
    )

    background, mask = prepare_image_and_mask(
        image=image,
        width=canvas["target_w"],
        height=canvas["target_h"],
        overlap_percentage=params.get("overlap_percentage", 1),
        resize_option=params.get("resize_option", "Full"),
        custom_resize_percentage=params.get("custom_resize_percentage", 100),
        alignment=alignment,
        overlap_left=params.get("overlap_left", False),
        overlap_right=params.get("overlap_right", False),
        overlap_top=params.get("overlap_top", False),
        overlap_bottom=params.get("overlap_bottom", False),
        pre_resized_source=resized_source,
        paste_x=canvas["paste_x"],
        paste_y=canvas["paste_y"],
    )

    # --- Diffusion Outpaint Inference ---
    if not can_expand(background.width, background.height, canvas["target_w"], canvas["target_h"], alignment):
        alignment = "Middle"

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt}, high quality, 4k"

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)

    generated_image = None
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=int(params.get("num_inference_steps", 12)),
    ):
        generated_image = image

    if generated_image is None:
        raise RuntimeError("Generation failed: No image returned.")

    output_path = os.path.join(output_dir, f"{job_id}_final.png")
    generated_image.save(output_path)

    # --- Optional Upscaling ---
    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        try:
            upscaled = upscale_image(
                output_path, model_name=model_name, scale_factor=scale_factor
            )
            up_path = os.path.join(output_dir, f"qwen_{job_id}_upscaled.png")
            upscaled.save(up_path)
            print(f"✅ Saved upscaled image: {up_path}")
            return os.path.abspath(up_path)
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
            return os.path.abspath(output_path)

    return os.path.abspath(output_path)


# ==========================================================
# LOCAL TEST MODE
# ==========================================================

if __name__ == "__main__":
    print("⚙️  Running test inference via entrypoint...")
    test_params = {
        "output_dir": "./outputs_test",
        "init_image": "input_image.jpg",
        "aspect_ratio": "16:9",
        "alignment": "Right",
        "num_inference_steps": 10,
        "scale_image": True,
        "scale_factor": 2,
    }
    result_path = run_inference_sdxl_outpaint(
        "job1234", "Expand the image naturally", test_params
    )
    print(f"✅ Test complete. Output at: {result_path}")
