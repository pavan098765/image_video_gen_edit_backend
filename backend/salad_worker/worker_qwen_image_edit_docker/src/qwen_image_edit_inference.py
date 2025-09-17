# qwen_image_edit_inference.py
import os
import datetime
import random
import torch
import gc
from pathlib import Path
from PIL import Image
import numpy as np

# ---- import ComfyUI nodes (assumes setup already done once in container build) ----
from nodes import (
    CLIPLoader, VAEEncode, VAEDecode, VAELoader, KSamplerAdvanced,
    ImageScale, LoraLoaderModelOnly, LoadImage, EmptySD3LatentImage
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_images import ImageStitch
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_flux import FluxKontextImageScale

# ---- utilities ----
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ---- main function ----
def run_inference_and_save(job_id: str, prompt: str, params: dict) -> str:
    """
    Runs Qwen Image Edit inference and saves PNG to disk.
    Args:
        job_id: str - unique job ID
        prompt: str - positive prompt
        params: dict - includes negative_prompt, init_image, width, height, steps, cfg, etc.
    Returns:
        output_path: str - path to saved PNG
    """
    positive_prompt = prompt
    negative_prompt = params.get("negative_prompt", "")
    init_image_path = params.get("init_image")
    init_image_path_2 = params.get("init_image2")
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    steps = int(params.get("steps", 20))
    cfg = float(params.get("cfg", 2.5))
    seed = int(params.get("seed", 0)) or random.randint(0, 2**32 - 1)
    sampler_name = params.get("sampler", "euler")
    scheduler = params.get("scheduler", "simple")

    output_dir = params.get("output_dir", "qwen_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        clip_loader = CLIPLoader()
        unet_loader = UnetLoaderGGUF()
        vae_loader = VAELoader()
        vae_encode = VAEEncode()
        vae_decode = VAEDecode()
        ksampler = KSamplerAdvanced()
        load_image = LoadImage()
        load_image2 = LoadImage()
        image_stitch = ImageStitch()
        flux_kontext_scale = FluxKontextImageScale()
        empty_latent_image = EmptySD3LatentImage()
        text_encode_qwen = TextEncodeQwenImageEdit()
        model_sampling = ModelSamplingSD3()

        # ---- load encoders ----
        clip = clip_loader.load_clip(
            params["text_encoder_path"], type="qwen_image", device="default"
        )[0]
        vae = vae_loader.load_vae(params["vae_path"])[0]

        # ---- prepare input ----
        if init_image_path and os.path.exists(init_image_path):
            img_tensor = load_image.load_image(init_image_path)[0]

            if init_image_path_2 and os.path.exists(init_image_path_2):
                img2_tensor = load_image2.load_image(init_image_path_2)[0]
                img_tensor = image_stitch.stitch(img_tensor, "right", True, 0, "white", img2_tensor)[0]

            img_tensor = flux_kontext_scale.scale(img_tensor)[0]
            pos_enc = text_encode_qwen.encode(clip, positive_prompt, vae=vae, image=img_tensor)[0]
            neg_enc = text_encode_qwen.encode(clip, negative_prompt, vae=vae, image=img_tensor)[0]
            latent = vae_encode.encode(vae, img_tensor)[0]
        else:
            pos_enc = text_encode_qwen.encode(clip, positive_prompt)[0]
            neg_enc = text_encode_qwen.encode(clip, negative_prompt)[0]
            latent = empty_latent_image.generate(width, height, 1)[0]

        # ---- load unet ----
        model = unet_loader.load_unet(params["unet_path"])[0]
        model = model_sampling.patch(model, cfg, 1)[0]

        # ---- sample ----
        latent_out = ksampler.sample(
            model=model,
            add_noise="enable",
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=pos_enc,
            negative=neg_enc,
            latent_image=latent,
            start_at_step=0,
            end_at_step=1000,
            return_with_leftover_noise="disable",
        )[0]

        # ---- decode ----
        decoded = vae_decode.decode(vae, latent_out)[0]

        # ---- save ----
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(output_dir) / f"{job_id}_{timestamp}.png")
        tensor2pil(decoded[0]).save(out_path)

    clear_memory()
    return out_path
