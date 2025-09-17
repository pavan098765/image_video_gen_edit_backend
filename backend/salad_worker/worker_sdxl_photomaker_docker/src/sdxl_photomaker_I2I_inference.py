# You can use PhotoMaker (https://github.com/TencentARC/PhotoMaker) to personalize generated images with your own ID.

# NOTE, currently PhotoMaker ONLY works with SDXL (any SDXL model files will work). The VAE in SDXL encounters NaN issues. You can find a fixed VAE here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors

# Download PhotoMaker model file (in safetensor format) https://huggingface.co/bssrdf/PhotoMaker. The official release of the model file (in .bin format) does not work with stablediffusion.cpp.

# In prompt, make sure you have a class word followed by the trigger word "img" (hard-coded for now). The class word could be one of "man, woman, girl, boy". If input ID images contain asian faces, add Asian before the class word.



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