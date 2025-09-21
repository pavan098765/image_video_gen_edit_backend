# sd_outpaint_inference.py
import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"

import torch
import base64
from io import BytesIO
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, Union

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


# -----------------------
# Helpers
# -----------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from PIL, file path, bytes, or base64 string."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, str):
        if os.path.exists(img):  # file path
            return Image.open(img).convert("RGB")
        try:  # base64
            decoded = base64.b64decode(img)
            return Image.open(BytesIO(decoded)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to decode image: {e}")
    raise ValueError(f"Unsupported input type for image: {type(img)}")


def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


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
):
    target_size = (width, height)

    # Resize to fit target
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize percentage
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:
        resize_percentage = custom_resize_percentage

    resize_factor = resize_percentage / 100
    new_width = max(int(source.width * resize_factor), 64)
    new_height = max(int(source.height * resize_factor), 64)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate margins
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height
    else:
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create background & paste
    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create mask
    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    overlap_x = max(int(new_width * (overlap_percentage / 100)), 1)
    overlap_y = max(int(new_height * (overlap_percentage / 100)), 1)
    white_gaps_patch = 2

    # Default overlaps
    left_overlap = margin_x + (overlap_x if overlap_left else white_gaps_patch)
    right_overlap = margin_x + new_width - (overlap_x if overlap_right else white_gaps_patch)
    top_overlap = margin_y + (overlap_y if overlap_top else white_gaps_patch)
    bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else white_gaps_patch)

    # Adjust overlaps for alignment edges
    if alignment == "Left":
        left_overlap = margin_x + (overlap_x if overlap_left else 0)
    elif alignment == "Right":
        right_overlap = margin_x + new_width - (overlap_x if overlap_right else 0)
    elif alignment == "Top":
        top_overlap = margin_y + (overlap_y if overlap_top else 0)
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else 0)

    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)
    return background, mask


def run_inference_diffuser_outpaint(
    pipe,
    image,
    width,
    height,
    overlap_percentage,
    num_inference_steps,
    resize_option,
    custom_resize_percentage,
    prompt_input,
    alignment,
    overlap_left,
    overlap_right,
    overlap_top,
    overlap_bottom,
):
    background, mask = prepare_image_and_mask(
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
    )

    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input}, high quality, 4k"

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
        num_inference_steps=num_inference_steps,
    ):
        generated_image = image

    return cnet_image, generated_image


# -----------------------
# Unified API function
# -----------------------
def run_inference_sdxl_outpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SDXL outpainting and save output.
    Accepts input_image as path, PIL.Image, bytes, or base64.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load controlnet
    config_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", filename="config_promax.json")
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    model_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", filename="diffusion_pytorch_model_promax.safetensors")
    state_dict = load_state_dict(model_file)
    loaded_keys = list(state_dict.keys())
    result_tuple = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
    )
    model = result_tuple[0].to(device=device, dtype=model_dtype)

    # Load VAE & pipeline
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=model_dtype).to(device)
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=model_dtype,
        vae=vae,
        controlnet=model,
        variant="fp16",
    ).to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    # Load input (path, PIL, bytes, or base64)
    input_image = _load_image(params.get("input_image"))
    if input_image is None:
        raise ValueError("Missing or invalid input_image")

    # Run inference
    preview_params = {
        "width": params.get("width", 1280),
        "height": params.get("height", 720),
        "overlap_percentage": params.get("overlap_percentage", 10),
        "resize_option": params.get("resize_option", "Full"),
        "custom_resize_percentage": params.get("custom_resize_percentage", 50),
        "alignment": params.get("alignment", "Middle"),
        "overlap_left": params.get("overlap_left", True),
        "overlap_right": params.get("overlap_right", True),
        "overlap_top": params.get("overlap_top", True),
        "overlap_bottom": params.get("overlap_bottom", True),
    }
    inference_params = {
        "num_inference_steps": params.get("num_inference_steps", 8),
        "prompt_input": prompt,
    }

    control_image, final_image = run_inference_diffuser_outpaint(pipe, input_image, **preview_params, **inference_params)

    # Save output
    output_dir = params.get("output_dir", "/quickpod")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sdxl_outpaint_{job_id}.png")

    if final_image:
        final_image.save(out_path)
        return out_path
    else:
        raise RuntimeError("Outpaint generation failed; no image returned")
