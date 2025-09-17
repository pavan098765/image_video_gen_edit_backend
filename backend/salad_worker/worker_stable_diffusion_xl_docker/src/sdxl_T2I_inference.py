# sdxl_infer.py
# Usage: python sdxl_infer.py
# Requirements: pip install stable-diffusion-cpp-python (built with appropriate backend)
# Replace paths with your local model and VAE (use the SDXL VAE FP16 fix if running fp16).
# The VAE in SDXL encounters NaN issues. You can find a fixed VAE here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors

from stable_diffusion_cpp import StableDiffusion
from PIL import Image
import os

MODEL_PATH = "../models/sd_xl_base_1.0.safetensors"      # or sd_xl_base_1.0.gguf
VAE_PATH   = "../models/sdxl_vae-fp16-fix.safetensors"  # SDXL VAE fix recommended for fp16
OUT_DIR    = "./outputs_sdxl"
os.makedirs(OUT_DIR, exist_ok=True)

def progress_cb(step, steps, t):
    print(f"[SDXL] step {step}/{steps} time {t:.2f}s")

sd = StableDiffusion(
    model_path=MODEL_PATH,
    vae_path=VAE_PATH,
    # wtype: choose "f16" for best quality (if you have memory), otherwise try "q8_0" for quantized.
    wtype="f16",   # change to "q8_0" or "q5_1" if you need lower memory
    vae_decode_only=True,        
    # Optional: enable flash attn or other options at build; chosen backend depends on how you pip-installed the package
)

prompt = "a cinematic portrait of an old temple at sunrise, ultra-detailed, 8k"
images = sd.generate_image(
    prompt=prompt,
    width=1024,
    height=1024,
    sample_steps=28,
    sample_method="dpm2",         # or "euler_a", "euler" 
    seed=-1, # Uncomment to set a specific seed (use -1 for a random seed)
    cfg_scale=1.0,      # a cfg_scale of 1 is recommended for FLUX
    progress_callback=progress_cb
)

out_path = os.path.join(OUT_DIR, "sdxl_output.png")
images[0].save(out_path)
print("Saved", out_path)
