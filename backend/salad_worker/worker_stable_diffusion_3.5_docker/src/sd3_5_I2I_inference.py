# Download the weights from the links below:

# Download sd3.5_large from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
# Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
# Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
# Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors


from stable_diffusion_cpp import StableDiffusion
# from PIL import Image

INPUT_IMAGE = "../input.png"
# INPUT_IMAGE = Image.open("../input.png") # or alternatively, pass as PIL Image

stable_diffusion = StableDiffusion(model_path="../models/v1-5-pruned-emaonly.safetensors")

output = stable_diffusion.generate_image(
      prompt="blue eyes",
      init_image=INPUT_IMAGE, # Note: The input image will be automatically resized to the match the width and height arguments (default: 512x512)
      strength=0.4,
)