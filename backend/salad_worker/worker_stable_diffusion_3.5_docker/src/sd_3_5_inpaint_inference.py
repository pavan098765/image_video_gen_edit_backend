# Download the weights from the links below:

# Download sd3.5_large from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
# Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
# Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
# Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors

from stable_diffusion_cpp import StableDiffusion

# Note: Inpainting with a base model gives poor results. A model fine-tuned for inpainting is recommended.
stable_diffusion = StableDiffusion(model_path="../models/v1-5-pruned-emaonly.safetensors")

output = stable_diffusion.generate_image(
      prompt="blue eyes",
      init_image="../input.png",
      mask_image="../mask.png", # A grayscale image where 0 is masked and 255 is unmasked
      strength=0.4,
)