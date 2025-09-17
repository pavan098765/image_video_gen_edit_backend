# Download the weights from the links below:

# Preconverted gguf model from https://huggingface.co/silveroxides/Chroma-GGUFF
# Otherwise, download chroma's safetensors from https://huggingface.co/lodestones/Chroma
# Download vae from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
# Download clip_l from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
# Download t5xxl from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors



from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    diffusion_model_path="../models/chroma-unlocked-v40-Q4_0.gguf", # In place of model_path
    t5xxl_path="../models/t5xxl_fp16.safetensors",
    vae_path="../models/ae.safetensors",
    vae_decode_only=True, # Can be True if we are not generating image to image
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat holding a sign says 'chroma.cpp'",
      sample_steps=4,
      cfg_scale=4.0, # a cfg_scale of 4 is recommended for Chroma
      sample_method="euler", # euler is recommended for FLUX
)