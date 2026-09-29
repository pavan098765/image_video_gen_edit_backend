# Dependency lock and model placement

This document records the exact environment used to validate the companion
nodes and matching workflow. Tested versions are evidence, not universal
minimum requirements.

## Runtime

| Component | Validated value |
|---|---|
| ComfyUI | `0.31.0`, commit `2a19bbf0140743553d396d2ac49a2c73439195cc` |
| ComfyUI frontend | `1.48.7` |
| OS / GPU | Windows 11 / NVIDIA RTX 5080 16GB, compute capability 12.0 |
| Python | `3.13.12` |
| PyTorch | `2.12.1+cu130` |
| SageAttention | `2.2.0+cu130torch2.10.0andhigher.post6` on the validated host |
| Triton provider | `triton-windows 3.7.1.post27` on the validated host |

## External custom nodes

These projects are **not bundled** in this ZIP.

| Required nodes | Authoritative source | Pin |
|---|---|---|
| `LTX2MemoryEfficientSageAttentionPatch`, `VRAM_Debug`, `PreviewAny` | `https://github.com/kijai/ComfyUI-KJNodes` / `https://registry.comfy.org/nodes/comfyui-kjnodes` | CNR package `comfyui-kjnodes` version `1.5.0` |
| `SolAttnPatch` | `https://github.com/kijai/ComfyUI-SolAttn_triton` | commit `842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc` |

`LTX2MemoryEfficientSageAttentionPatch` also requires a SageAttention build
compatible with the recipient's Python, PyTorch, CUDA, operating system, and
GPU architecture. `SolAttnPatch` requires a provider exposing the Python
module `triton`. Only the Windows combination above was validated; this package
does not prescribe an untested Linux binary or version.

## Required LTX-2.5 files

Accept the official access terms before downloading. All links are pinned to
Hugging Face revision `28dac7acdc1f78a70e98687db261a949754f8941`.

| Component | Exact filename | Destination under `ComfyUI/` | Bytes |
|---|---|---|---:|
| Distilled INT8 ConvRot transformer | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` | 21,504,034,224 |
| Gemma4 INT8 ConvRot text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` | 15,372,971,786 |
| Conv video VAE | `ltx-2.5-video-vae-conv-bf16.safetensors` | `models/vae/` | 1,452,269,922 |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 364,866,540 |
| x2 latent upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | 995,778,752 |

Repository: `https://huggingface.co/Lightricks/LTX-2.5`

Pinned download paths:

- `https://huggingface.co/Lightricks/LTX-2.5/resolve/28dac7acdc1f78a70e98687db261a949754f8941/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
- `https://huggingface.co/Lightricks/LTX-2.5/resolve/28dac7acdc1f78a70e98687db261a949754f8941/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
- `https://huggingface.co/Lightricks/LTX-2.5/resolve/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-conv-bf16.safetensors`
- `https://huggingface.co/Lightricks/LTX-2.5/resolve/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-audio-vae-bf16.safetensors`
- `https://huggingface.co/Lightricks/LTX-2.5/resolve/28dac7acdc1f78a70e98687db261a949754f8941/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`

The validated local transformer SHA-256 was
`C4279EEFF115CBEACA494BD2183E7D768C38FE85A184DC6AFBB7159157C44334`.

## Optional LoRA

Place an ordinary LTX transformer LoRA in `ComfyUI/models/loras/`, refresh the
model list, then select it in `LTX-2.5 Optional INT8 LoRA`. IC-LoRAs require a
separate guide workflow and are not supported by this optional slot. A selected
adapter can increase VRAM usage and runtime; `None` is the portable default.
