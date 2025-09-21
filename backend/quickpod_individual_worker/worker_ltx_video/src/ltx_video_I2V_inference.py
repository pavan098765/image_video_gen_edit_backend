
import base64
import datetime
import gc
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Dict
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
from PIL import Image

MODEL_PATH = "../all_models/ltx_video/" 

pipe = LTXConditionPipeline.from_pretrained("jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE", torch_dtype=torch.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

def run_inference_ltx_video_i2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run ltx video Img2Img inference and save the result locally.
    Returns the path to the saved MP4 file.
    """
    mode = params.get("mode", "Ltx_Video_I2V-2B")
    negative_prompt = params.get("negative_prompt", None)
    expected_width = params.get("width", 512)
    expected_height = params.get("height", 512)
    init_images=params.get("init_image", None)  
    decoded_bytes = base64.b64decode(init_images)

    if image is not None and not isinstance(image, Image.Image):
        image = Image.open(BytesIO(decoded_bytes)).convert("RGB")
        
    video = load_video(export_to_video([image])) # compress the image using video compression as the model was trained on videos
    output_dir = params.get("output_dir", "/quickpod")
    num_frames=params.get("num_frames", 96),
    downscale_factor = 2 / 3
    num_frames = 96

    condition1 = LTXVideoCondition(video=video, frame_index=0)

    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the video
    latents = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(0),
    output_type="latent",
    ).frames

    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(
        latents=latents,
        output_type="latent"
    ).frames

    # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=num_frames,
        denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
    ).frames[0]

    # Part 4. Downscale the video to the expected resolution
    video = [frame.resize((expected_width, expected_height)) for frame in video]
    output_dir = params.get("output_dir", "/quickpod")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(output_dir) / f"ltx_video_I2v_{job_id}_{timestamp}.mp4")
    export_to_video(video, out_path, fps=24)

    gc.collect()
    
    return out_path
    
