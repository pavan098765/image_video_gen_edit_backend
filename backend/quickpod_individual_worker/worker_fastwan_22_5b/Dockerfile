FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Non-chunk model files (configs, tokenizer, ckpt, etc.)
COPY models/wan_22_5b/model_index.json /app/models/wan_22_5b/
COPY models/wan_22_5b/README.md /app/models/wan_22_5b/
COPY models/wan_22_5b/.gitattributes /app/models/wan_22_5b/
COPY models/wan_22_5b/assets/logo.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/assets/moe_arch.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/assets/moe_2.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/assets/vae.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/assets/comp_effic.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/assets/performance.png /app/models/wan_22_5b/assets/
COPY models/wan_22_5b/examples/i2v_input.JPG /app/models/wan_22_5b/examples/
COPY models/wan_22_5b/scheduler/scheduler_config.json /app/models/wan_22_5b/scheduler/
COPY models/wan_22_5b/text_encoder/config.json /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model.safetensors.index.json /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/tokenizer/special_tokens_map.json /app/models/wan_22_5b/tokenizer/
COPY models/wan_22_5b/tokenizer/tokenizer_config.json /app/models/wan_22_5b/tokenizer/
COPY models/wan_22_5b/tokenizer/tokenizer.json /app/models/wan_22_5b/tokenizer/
COPY models/wan_22_5b/tokenizer/spiece.model /app/models/wan_22_5b/tokenizer/
COPY models/wan_22_5b/transformer/config.json /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/vae/config.json /app/models/wan_22_5b/vae/
COPY models/realesrgan/out_upscaled.png /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x2plus.pth /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x4plus.pth /app/models/realesrgan/

# Per-part COPYs (one COPY per chunk => one layer per chunk)
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-aa /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ab /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ac /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ad /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ae /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-af /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ag /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ah /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-ai /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00001-of-00003.safetensors.part-aj /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-aa /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ab /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ac /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ad /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ae /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-af /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ag /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ah /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-ai /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00002-of-00003.safetensors.part-aj /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00003-of-00003.safetensors.part-aa /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00003-of-00003.safetensors.part-ab /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/text_encoder/model-00003-of-00003.safetensors.part-ac /app/models/wan_22_5b/text_encoder/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-aa /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ab /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ac /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ad /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ae /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-af /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ag /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ah /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ai /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-aj /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ak /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-al /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-am /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-an /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ao /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ap /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-aq /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-ar /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-as /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/transformer/diffusion_pytorch_model.safetensors.part-at /app/models/wan_22_5b/transformer/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-aa /app/models/wan_22_5b/vae/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-ab /app/models/wan_22_5b/vae/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-ac /app/models/wan_22_5b/vae/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-ad /app/models/wan_22_5b/vae/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-ae /app/models/wan_22_5b/vae/
COPY models/wan_22_5b/vae/diffusion_pytorch_model.safetensors.part-af /app/models/wan_22_5b/vae/

# Worker code (last to leverage cache)
COPY src /app/worker/src

# --- Install SageAttention 2.x from source ---
    # RUN apt-get update && apt-get install -y git build-essential python3-dev && \
    #     git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    #     cd /tmp/SageAttention && \
    #     export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 && \
    #     python setup.py install && \
    #     cd / && rm -rf /tmp/SageAttention
RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV FASTWAN_5B_LOCAL_PATH="/app/models/wan_22_5b"
ENV REALESRGAN_LOCAL_PATH="/app/models/realesrgan"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
