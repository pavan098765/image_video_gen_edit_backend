# flux_kontext_infer.py
# Usage: python flux_kontext_infer.py
# Download the weights from the links below:

# Preconverted gguf model from https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF
# Download vae from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
# Download clip_l from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
# Download t5xxl from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors


from stable_diffusion_cpp import StableDiffusion
import os

DIFFUSION_MODEL = "../all_models/flux1-kontext-dev-unet.gguf"   # or .safetensors (.gguf recommended if quantized)
CLIP_L_PATH      = "../all_models/clip_l.safetensors"          # or corresponding CLIP model
T5XXL_PATH       = "../all_models/t5xxl_fp16.safetensors"
VAE_PATH         = "../all_models/flux_vae.safetensors"
OUT_DIR          = "./outputs_flux_kontext"
os.makedirs(OUT_DIR, exist_ok=True)

def progress_cb(step, steps, t):
    print(f"[Flux-Kontext] {step}/{steps} ({t:.2f}s)")

flux_kontext = StableDiffusion(
    diffusion_model_path=DIFFUSION_MODEL,
    clip_l_path=CLIP_L_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_path=VAE_PATH,
    # Recommended starting quant for Flux Kontext: q5_1 or q8_0 (q5_1 is a good balance)
    wtype="q5_1",
    vae_decode_only=False,
)

prompt = "Edit: convert the character's outfit to a renaissance painterly dress while keeping face identity"
# For Kontext style workflows you may need to supply an init image or special edit params per flux docs.
images = flux_kontext.generate_image(
    prompt=prompt,
    images=["input.png"],       # Kontext needs an init image for editing
    width=1024,
    height=1024,
    sample_steps=28,
    seed=-1, # Uncomment to set a specific seed (use -1 for a random seed)
    cfg_scale=1.0,      # a cfg_scale of 1 is recommended for FLUX
    sample_method="euler",      # euler is recommended for FLUX
    progress_callback=progress_cb
)

images[0].save(os.path.join(OUT_DIR, "flux_kontext_output.png"))
print("Saved", os.path.join(OUT_DIR, "flux_kontext_output.png"))
