FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# --- SDXL Outpaint (controlnet / vae / pipeline) build ---

# Non-chunk model files (configs, tokenizer, weights, etc.)
COPY models/sdxl_outpaint/controlnet/config.json /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/README.md /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/config_promax.json /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/.gitattributes /app/models/sdxl_outpaint/controlnet/

# Combine all image formats into one COPY command (small files)
COPY models/sdxl_outpaint/controlnet/images /app/models/sdxl_outpaint/controlnet/images/

# VAE files
COPY models/sdxl_outpaint/vae/config.json /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/README.md /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/.gitattributes /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/diffusion_pytorch_model.safetensors /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/diffusion_pytorch_model.bin /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/sdxl_vae.safetensors /app/models/sdxl_outpaint/vae/
COPY models/sdxl_outpaint/vae/sdxl.vae.safetensors /app/models/sdxl_outpaint/vae/

# Combine all VAE image formats
COPY models/sdxl_outpaint/vae/images /app/models/sdxl_outpaint/vae/images/

# Pipeline configs
COPY models/sdxl_outpaint/pipeline/.gitattributes /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/README.md /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/model_index.json /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/text_encoder/config.json /app/models/sdxl_outpaint/pipeline/text_encoder/
COPY models/sdxl_outpaint/pipeline/text_encoder/model.fp16.safetensors /app/models/sdxl_outpaint/pipeline/text_encoder/
COPY models/sdxl_outpaint/pipeline/scheduler/scheduler_config.json /app/models/sdxl_outpaint/pipeline/scheduler/
COPY models/sdxl_outpaint/pipeline/text_encoder_2/config.json /app/models/sdxl_outpaint/pipeline/text_encoder_2/
COPY models/sdxl_outpaint/pipeline/tokenizer/* /app/models/sdxl_outpaint/pipeline/tokenizer/
COPY models/sdxl_outpaint/pipeline/tokenizer_2/* /app/models/sdxl_outpaint/pipeline/tokenizer_2/
COPY models/sdxl_outpaint/pipeline/unet/config.json /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.safetensors.index.json /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/vae/config.json /app/models/sdxl_outpaint/pipeline/vae/
COPY models/sdxl_outpaint/pipeline/vae/diffusion_pytorch_model.fp16.safetensors /app/models/sdxl_outpaint/pipeline/vae/

# RealESRGAN
COPY models/realesrgan/*.pth /app/models/realesrgan/


# Per-part COPYs (one COPY per chunk => one layer per chunk)
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model.safetensors.part-aa /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model.safetensors.part-ab /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model.safetensors.part-ac /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model.safetensors.part-ad /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model.safetensors.part-ae /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model_promax.safetensors.part-aa /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model_promax.safetensors.part-ab /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model_promax.safetensors.part-ac /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model_promax.safetensors.part-ad /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/controlnet/diffusion_pytorch_model_promax.safetensors.part-ae /app/models/sdxl_outpaint/controlnet/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-aa /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ab /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ac /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ad /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ae /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-af /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ag /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ah /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ai /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-aj /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-ak /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-al /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-am /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/RealVisXL_V5.0_Lightning_fp16.safetensors.part-an /app/models/sdxl_outpaint/pipeline/
COPY models/sdxl_outpaint/pipeline/text_encoder_2/model.fp16.safetensors.part-aa /app/models/sdxl_outpaint/pipeline/text_encoder_2/
COPY models/sdxl_outpaint/pipeline/text_encoder_2/model.fp16.safetensors.part-ab /app/models/sdxl_outpaint/pipeline/text_encoder_2/
COPY models/sdxl_outpaint/pipeline/text_encoder_2/model.fp16.safetensors.part-ac /app/models/sdxl_outpaint/pipeline/text_encoder_2/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-aa /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ab /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ac /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ad /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ae /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-af /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ag /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ah /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-ai /app/models/sdxl_outpaint/pipeline/unet/
COPY models/sdxl_outpaint/pipeline/unet/diffusion_pytorch_model.fp16.safetensors.part-aj /app/models/sdxl_outpaint/pipeline/unet/

# Worker code (last to leverage cache)
COPY src /app/worker/src

RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV CTRLNET_LOCAL_PATH="/app/models/sdxl_outpaint/controlnet"
ENV VAE_LOCAL_PATH="/app/models/sdxl_outpaint/vae"
ENV PIPELINE_LOCAL_PATH="/app/models/sdxl_outpaint/pipeline"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
