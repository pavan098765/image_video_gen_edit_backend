# Download the weights from the links below:

# Download sd3.5_large from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
# Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
# Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
# Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors


from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    model_path="../models/sd3.5_large.safetensors",
    clip_l_path="../models/clip_l.safetensors",
    clip_g_path="../models/clip_g.safetensors",
    t5xxl_path="../models/t5xxl_fp16.safetensors",
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat holding a sign says 'Stable diffusion 3.5 Large'",
      height=1024,
      width=1024,
      cfg_scale=4.5,
      sample_method="euler",
)