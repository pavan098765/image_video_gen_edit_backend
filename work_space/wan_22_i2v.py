import os
import sys

import torch
import numpy as np
import diffusers
from diffusers.utils import export_to_video, load_image

# SDNQ repo layout is sdnq/src/sdnq/ — add src/ so `import sdnq` resolves to the package, not the repo root folder.
_WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
_SDNQ_SRC = os.path.join(_WORKSPACE_DIR, "sdnq", "src")
if os.path.isdir(_SDNQ_SRC) and _SDNQ_SRC not in sys.path:
    sys.path.insert(0, _SDNQ_SRC)


from sdnq import SDNQConfig  # noqa: F401 — register sdnq with diffusers/transformers

# _DEFAULT_HF_REPO = "Disty0/Wan2.2-I2V-A14B-SDNQ-uint4-svd-r32" # "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
_DEFAULT_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = diffusers.WanImageToVideoPipeline.from_pretrained(_DEFAULT_HF_REPO, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()


image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
generator = torch.Generator(device=device.type).manual_seed(0)
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=1.0,
    num_inference_steps=6,
    generator=generator,
).frames[0]
export_to_video(output, "i2v_output.mp4", fps=16)
