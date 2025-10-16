FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Non-chunk model files (configs, tokenizer, ckpt, etc.)
COPY models/sdxl_realism_illustrious/model_index.json /app/models/sdxl_realism_illustrious/
COPY models/sdxl_realism_illustrious/README.md /app/models/sdxl_realism_illustrious/
COPY models/sdxl_realism_illustrious/.gitattributes /app/models/sdxl_realism_illustrious/
COPY models/sdxl_realism_illustrious/text_encoder/config.json /app/models/sdxl_realism_illustrious/text_encoder/
COPY models/sdxl_realism_illustrious/text_encoder/model.safetensors /app/models/sdxl_realism_illustrious/text_encoder/
COPY models/sdxl_realism_illustrious/text_encoder_2/config.json /app/models/sdxl_realism_illustrious/text_encoder_2/
COPY models/sdxl_realism_illustrious/scheduler/scheduler_config.json /app/models/sdxl_realism_illustrious/scheduler/
COPY models/sdxl_realism_illustrious/tokenizer/special_tokens_map.json /app/models/sdxl_realism_illustrious/tokenizer/
COPY models/sdxl_realism_illustrious/tokenizer/merges.txt /app/models/sdxl_realism_illustrious/tokenizer/
COPY models/sdxl_realism_illustrious/tokenizer/tokenizer_config.json /app/models/sdxl_realism_illustrious/tokenizer/
COPY models/sdxl_realism_illustrious/tokenizer/vocab.json /app/models/sdxl_realism_illustrious/tokenizer/
COPY models/sdxl_realism_illustrious/tokenizer_2/merges.txt /app/models/sdxl_realism_illustrious/tokenizer_2/
COPY models/sdxl_realism_illustrious/tokenizer_2/special_tokens_map.json /app/models/sdxl_realism_illustrious/tokenizer_2/
COPY models/sdxl_realism_illustrious/tokenizer_2/tokenizer_config.json /app/models/sdxl_realism_illustrious/tokenizer_2/
COPY models/sdxl_realism_illustrious/tokenizer_2/vocab.json /app/models/sdxl_realism_illustrious/tokenizer_2/
COPY models/sdxl_realism_illustrious/unet/config.json /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/vae/config.json /app/models/sdxl_realism_illustrious/vae/
COPY models/sdxl_realism_illustrious/vae/diffusion_pytorch_model.safetensors /app/models/sdxl_realism_illustrious/vae/
COPY models/realesrgan/RealESRGAN_x2plus.pth /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x4plus.pth /app/models/realesrgan/

# Per-part COPYs (one COPY per chunk => one layer per chunk)
COPY models/sdxl_realism_illustrious/text_encoder_2/model.safetensors.part-aa /app/models/sdxl_realism_illustrious/text_encoder_2/
COPY models/sdxl_realism_illustrious/text_encoder_2/model.safetensors.part-ab /app/models/sdxl_realism_illustrious/text_encoder_2/
COPY models/sdxl_realism_illustrious/text_encoder_2/model.safetensors.part-ac /app/models/sdxl_realism_illustrious/text_encoder_2/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-aa /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ab /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ac /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ad /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ae /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-af /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ag /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ah /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-ai /app/models/sdxl_realism_illustrious/unet/
COPY models/sdxl_realism_illustrious/unet/diffusion_pytorch_model.safetensors.part-aj /app/models/sdxl_realism_illustrious/unet/

# Worker code (last to leverage cache)
COPY src /app/worker/src

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git \
    && apt-get purge -y git \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV QWEN_2509_LOCAL_PATH="/app/models/qwen_image_edit_2509"

ENV NUNCHAKU_QWEN_2509_LOCAL_PATH="/app/models/nunchaku_qwen_2509"

ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
