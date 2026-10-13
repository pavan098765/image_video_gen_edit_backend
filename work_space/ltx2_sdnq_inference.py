import time

import torch
from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import load_image

model_id = "dg845/LTX-2.3-Distilled-Diffusers"
device = "cuda"
dtype = torch.bfloat16
seed = 42

frame_rate = 24.0
width = 768
height = 512
num_inference_steps = 30

prompt = (
    "Completely remove all the subjects clothes. Her clothes are now completely removed to expose her big breasts and vagina. Maintain identical camera angle, framing, skintone and perspective and make sure to leave her facial features identical. The rest of the image remains the same, only remove her clothes to expose her naked body."
)
negative_prompt = None

image = load_image(
    "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/input_image.jpg"
)

pipe = LTX2ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device).manual_seed(seed)

output_path = "ltx2_3_i2v_distilled.mp4"
t_e2e_start = time.perf_counter()

t0 = time.perf_counter()
video_latent, audio_latent = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=240,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,  # Disable all guidance for distilled inference
    stg_scale=0.0,
    modality_scale=1.0,
    guidance_rescale=0.0,
    audio_guidance_scale=1.0,
    audio_stg_scale=0.0,
    audio_modality_scale=1.0,
    audio_guidance_rescale=0.0,
    spatio_temporal_guidance_blocks=None,
    use_cross_timestep=True,
    generator=generator,
    output_type="latent",
    return_dict=False,
)
t_stage1 = time.perf_counter() - t0

t0 = time.perf_counter()
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "dg845/LTX-2.3-Spatial-Upsampler-Diffusers",
    subfolder="latent_upsampler",
    torch_dtype=dtype,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
t_upsampler_setup = time.perf_counter() - t0

t0 = time.perf_counter()
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]
t_upsample = time.perf_counter() - t0

t0 = time.perf_counter()
video, audio = pipe(
    image=image,
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width * 2,
    height=height * 2,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,  # Disable all guidance for distilled inference
    stg_scale=0.0,
    modality_scale=1.0,
    guidance_rescale=0.0,
    audio_guidance_scale=1.0,
    audio_stg_scale=0.0,
    audio_modality_scale=1.0,
    audio_guidance_rescale=0.0,
    spatio_temporal_guidance_blocks=None,
    use_cross_timestep=True,
    generator=generator,
    output_type="np",
    return_dict=False,
)
t_stage2 = time.perf_counter() - t0

t0 = time.perf_counter()
encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path=output_path,
)
t_encode = time.perf_counter() - t0

t_e2e = time.perf_counter() - t_e2e_start
print(f"Output file: {output_path}")
print(
    "Inference timing (s): "
    f"stage1={t_stage1:.2f}, upsampler_load={t_upsampler_setup:.2f}, "
    f"latent_upsample={t_upsample:.2f}, stage2={t_stage2:.2f}, "
    f"encode={t_encode:.2f}, end_to_end={t_e2e:.2f}"
)