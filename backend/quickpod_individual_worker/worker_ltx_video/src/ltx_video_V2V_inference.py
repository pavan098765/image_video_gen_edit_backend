# ltx_video_V2V_inference.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import datetime
import gc
import os
from pathlib import Path
from typing import Any, Dict
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
from PIL import Image

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

MODEL_PATH = os.environ.get("LTX_VIDEO_V2V_LOCAL_PATH", "../all_models/ltx_video/")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "pipe": None,
    "pipe_upsample": None,
}


def get_ltx_v2v_pipeline():
    """
    Lazily load LTX Video V2V pipeline.
    Cached globally for reuse.
    """
    global _model_cache
    if _model_cache["pipe"] is not None:
        return _model_cache["pipe"], _model_cache["pipe_upsample"]

    print("🧠 Loading LTX Video V2V pipeline...")
    pipe = LTXConditionPipeline.from_pretrained(
        "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE", 
        torch_dtype=torch.bfloat16
    )
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.8", 
        vae=pipe.vae, 
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe_upsample.to("cuda")
    pipe.vae.enable_tiling()

    _model_cache["pipe"] = pipe
    _model_cache["pipe_upsample"] = pipe_upsample
    print("✅ LTX Video V2V pipeline loaded and cached.")
    return pipe, pipe_upsample


def unload_ltx_v2v_pipeline():
    """
    Optional cleanup: unload LTX V2V models and free VRAM/CPU cache.
    """
    global _model_cache
    _model_cache["pipe"] = None
    _model_cache["pipe_upsample"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 LTX Video V2V model cache cleared and GPU memory freed.")

def round_to_nearest_resolution_acceptable_by_vae(height, width, pipe):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

def run_inference_ltx_video_v2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run ltx video vid2vid inference and save the result locally.
    Returns the path to the saved MP4 file.
    """
    print(f"🧩 Starting LTX Video V2V job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Get cached pipeline
    pipe, pipe_upsample = get_ltx_v2v_pipeline()
    
    mode = params.get("mode", "Ltx_Video_V2V-2B")
    negative_prompt = params.get("negative_prompt", None)
    expected_width = params.get("width", 512)
    expected_height = params.get("height", 512)
    video_param = params.get("init_video", None)  # PIL Image or path
    video = load_video(video_param)[:21]  # Use only the first 21 frames as conditioning
    output_dir = params.get("output_dir", "/quickpod")
    downscale_factor = 2 / 3
    num_frames = 96

    condition1 = LTXVideoCondition(video=video, frame_index=0)

    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width, pipe)
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
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
        video_output = pipe(
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
    video_output = [frame.resize((expected_width, expected_height)) for frame in video_output]

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    output_dir = params.get("output_dir", "/quickpod")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(output_dir) / f"ltx_video_v2v_{job_id}_{timestamp}.mp4")

    export_to_video(video_output, out_path, fps=24)

    print(f"[LTX V2V] ✅ Video saved at {out_path}")
    return out_path
    
