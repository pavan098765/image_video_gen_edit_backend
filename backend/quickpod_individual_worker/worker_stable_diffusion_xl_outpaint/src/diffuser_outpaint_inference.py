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
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/controlnet"
)
VAE_LOCAL = os.environ.get(
    "VAE_LOCAL_PATH",
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/vae"
)
PIPELINE_LOCAL = os.environ.get(
    "PIPELINE_LOCAL_PATH",
    "/quickpod/app/worker_stable_diffusion_xl_outpaint/models/sdxl_outpaint/pipeline"
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


def prepare_image_and_mask(
    image, width, height, overlap_percentage, resize_option,
    custom_resize_percentage, alignment, overlap_left, overlap_right,
    overlap_top, overlap_bottom
):
    target_size = (width, height)

    # Scale image to fit target
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Resize option handling
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

    overlap_x = max(int(new_width * (overlap_percentage / 100)), 1)
    overlap_y = max(int(new_height * (overlap_percentage / 100)), 1)

    # Alignment-based margins
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x, margin_y = 0, (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x, margin_y = target_size[0] - new_width, (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x, margin_y = (target_size[0] - new_width) // 2, 0
    elif alignment == "Bottom":
        margin_x, margin_y = (target_size[0] - new_width) // 2, target_size[1] - new_height
    else:
        margin_x, margin_y = (target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create base image + mask
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Overlap calculations
    white_gaps_patch = 2
    left_overlap = margin_x + (overlap_x if overlap_left else white_gaps_patch)
    right_overlap = margin_x + new_width - (overlap_x if overlap_right else white_gaps_patch)
    top_overlap = margin_y + (overlap_y if overlap_top else white_gaps_patch)
    bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else white_gaps_patch)

    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)
    return background, mask


def run_inference_diffuser_outpaint(
    pipe, image, width, height, overlap_percentage, num_inference_steps,
    resize_option, custom_resize_percentage, prompt_input, alignment,
    overlap_left, overlap_right, overlap_top, overlap_bottom
):
    background, mask = prepare_image_and_mask(
        image, width, height, overlap_percentage, resize_option,
        custom_resize_percentage, alignment, overlap_left, overlap_right,
        overlap_top, overlap_bottom
    )

    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input}, high quality, 4k"

    (
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds
    ) = pipe.encode_prompt(final_prompt, "cuda", True)

    generated_image = None
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        generated_image = image

    return cnet_image, generated_image


# ==========================================================
# ENTRYPOINT FUNCTION
# ==========================================================

def run_inference_sdxl_outpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for image editing
        params:
        - output_dir (str)
        - init_image OR images (base64/list/PIL)
        - num_inference_steps (int)
        - cfg_scale (float)
        - negative_prompt (str)
        - scale_image (bool)
        - scale_factor (int)
        - scale_model_name (str)
    output: absolute path to final image (upscaled if requested)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ControlNet ---
    controlnet_config_path = os.path.join(CTRLNET_LOCAL, "config_promax.json")
    controlnet_weights_path = os.path.join(CTRLNET_LOCAL, "diffusion_pytorch_model_promax.safetensors")

    config = ControlNetModel_Union.load_config(controlnet_config_path)
    controlnet_model = ControlNetModel_Union.from_config(config)
    state_dict = load_state_dict(controlnet_weights_path)
    loaded_keys = list(state_dict.keys())
    controlnet_model, *_ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, controlnet_weights_path, CTRLNET_LOCAL, loaded_keys
    )
    controlnet_model = controlnet_model.to(device=device, dtype=dtype)

    # --- Load VAE and Pipeline ---
    vae = AutoencoderKL.from_pretrained(VAE_LOCAL, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        PIPELINE_LOCAL, torch_dtype=dtype, vae=vae, controlnet=controlnet_model, variant="fp16"
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

    # --- Run Inference ---
    _, final_image = run_inference_diffuser_outpaint(
        pipe,
        image=image,
        width=params.get("width", 1080),
        height=params.get("height", 1920),
        overlap_percentage=params.get("overlap_percentage", 1),
        num_inference_steps=params.get("num_inference_steps", 12),
        resize_option=params.get("resize_option", "Full"),
        custom_resize_percentage=params.get("custom_resize_percentage", 100),
        prompt_input=prompt,
        alignment=params.get("alignment", "Middle"),
        overlap_left=params.get("overlap_left", False),
        overlap_right=params.get("overlap_right", False),
        overlap_top=params.get("overlap_top", False),
        overlap_bottom=params.get("overlap_bottom", False),
    )

    if final_image is None:
        raise RuntimeError("Generation failed: No image returned.")

    output_path = os.path.join(output_dir, f"{job_id}_final.png")
    final_image.save(output_path)

    # Optional upscaling
    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        try:
            upscaled = upscale_image(output_path,  model_name=model_name, scale_factor=scale_factor)
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
        "width": 1080,
        "height": 1920,
        "overlap_percentage": 1,
        "resize_option": "Full",
        "custom_resize_percentage": 100,
        "alignment": "Middle",
        "overlap_left": False,
        "overlap_right": False,
        "overlap_top": False,
        "overlap_bottom": False,
        "init_image": "input_image.jpg",
        "num_inference_steps": 10,
        "scale_image": True,
        "scale_factor": 2,
    }
    result_path = run_inference_sdxl_outpaint("job1234", "Expand the image naturally", test_params)
    print(f"✅ Test complete. Output at: {result_path}")
