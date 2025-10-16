import os
import sys
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append("/quickpod")

# Custom local imports from the original project structure
# from controlnet_union import ControlNetModel_Union
# from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from quickpod_worker.worker_stable_diffusion_xl_outpaint.src.controlnet_union import ControlNetModel_Union
from quickpod_worker.worker_stable_diffusion_xl_outpaint.src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# from ...src.controlnet_union import ControlNetModel_Union
# from ...src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


## --- Core Functionalities ---

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """
    Prepares the source image and mask for the outpainting process.
    It resizes, positions, and creates a mask based on user-defined parameters.
    """
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
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
    else: # Default to Middle if alignment is unknown
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2
    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Creates a preview image showing the placement and the masked area."""
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    preview = background.copy().convert('RGBA')
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask)
    preview = Image.alpha_composite(preview, red_mask)
    
    return preview

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
    overlap_bottom
):
    """
    Generates an outpainted image using the provided pipeline and parameters.
    """
    background, mask = prepare_image_and_mask(
        image, width, height, overlap_percentage, 
        resize_option, custom_resize_percentage, alignment,
        overlap_left, overlap_right, overlap_top, overlap_bottom
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

    # The pipe returns a generator, we are interested in the final result
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


if __name__ == "__main__":
    # --- 1. Model and Pipeline Setup ---
    print("🚀 Initializing models and pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json",
    )
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
    )
    state_dict = load_state_dict(model_file)
    loaded_keys = list(state_dict.keys()) # <--- ADD THIS LINE

    result_tuple  = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
    )
    model = result_tuple[0]
    model.to(device=device, dtype=model_dtype)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=model_dtype
    ).to(device)

    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=model_dtype,
        vae=vae,
        controlnet=model,
        variant="fp16",
    ).to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    print("✅ Models and pipeline loaded successfully.")

    # --- 2. Define Generation Parameters ---
    input_image_path = "/quickpod/quickpod_workers/worker_stable_diffusion_xl/src/input_image.jpg" # <--- IMPORTANT: Change this to your image path
    
    # Try to open the image, handle error if it doesn't exist
    try:
        input_image = Image.open(input_image_path)
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_image_path}' was not found.")
        print("Please update the 'input_image_path' variable with the correct path to your image.")
        exit()
        
    preview_params = {
    "width": 1280,
    "height": 720,
    "overlap_percentage": 10,
    "resize_option": "Full",
    "custom_resize_percentage": 50, # Only used if resize_option is "Custom"
    "alignment": "Middle",
    "overlap_left": True,
    "overlap_right": True,
    "overlap_top": True,
    "overlap_bottom": True
    }
        
        # Parameters for the diffusion model inference
    inference_params = {
        "num_inference_steps": 8,
        "prompt_input": "",
    }

    # --- 3. Generate a Preview (Optional) ---
    print("\n🎨 Generating preview of mask and alignment...")
    # Pass only the preview parameters
    preview = preview_image_and_mask(input_image, **preview_params)
    preview.save("output_preview.png")
    print("✅ Preview saved to 'output_preview.png'")

    # --- 4. Run Inference ---
    print("\n🧠 Starting image generation...")
    # Combine the dictionaries to pass all required parameters to infer
    control_image, final_image = run_inference_diffuser_outpaint(pipe, input_image, **preview_params, **inference_params)
    
    # --- 5. Save Results ---
    if final_image:
        final_image.save("output_generated.png")
        control_image.save("output_control_image.png")
        print("\n🎉 Generation complete!")
        print("✅ Final image saved to 'output_generated.png'")
        print("✅ Control image (with mask) saved to 'output_control_image.png'")
    else:
        print("❌ Generation failed. No image was returned.")