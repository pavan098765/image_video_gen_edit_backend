FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Non-chunk model files (configs, tokenizer, ckpt, etc.)
COPY models/wan_21_1b/README.md /app/models/wan_21_1b/
COPY models/wan_21_1b/.gitattributes /app/models/wan_21_1b/
COPY models/wan_21_1b/model_index.json /app/models/wan_21_1b/
COPY models/wan_21_1b/assets/comp_effic.png /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/logo.png /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/data_for_diff_stage.jpg /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/t2v_res.jpg /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/i2v_res.png /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/vben_vs_sota.png /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/video_dit_arch.jpg /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/vben_1.3b_vs_sota.png /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/assets/video_vae_res.jpg /app/models/wan_21_1b/assets/
COPY models/wan_21_1b/examples/i2v_input.JPG /app/models/wan_21_1b/examples/
COPY models/wan_21_1b/scheduler/scheduler_config.json /app/models/wan_21_1b/scheduler/
COPY models/wan_21_1b/text_encoder/config.json /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model.safetensors.index.json /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/tokenizer/special_tokens_map.json /app/models/wan_21_1b/tokenizer/
COPY models/wan_21_1b/tokenizer/tokenizer_config.json /app/models/wan_21_1b/tokenizer/
COPY models/wan_21_1b/tokenizer/spiece.model /app/models/wan_21_1b/tokenizer/
COPY models/wan_21_1b/tokenizer/tokenizer.json /app/models/wan_21_1b/tokenizer/
COPY models/wan_21_1b/transformer/config.json /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/vae/config.json /app/models/wan_21_1b/vae/
COPY models/wan_21_1b/vae/diffusion_pytorch_model.safetensors /app/models/wan_21_1b/vae/
COPY models/realesrgan/out_upscaled.png /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x2plus.pth /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x4plus.pth /app/models/realesrgan/

# Per-part COPYs (one COPY per chunk => one layer per chunk)
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-aa /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ab /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ac /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ad /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ae /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-af /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ag /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ah /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-ai /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00001-of-00005.safetensors.part-aj /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-aa /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ab /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ac /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ad /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ae /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-af /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ag /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ah /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-ai /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00002-of-00005.safetensors.part-aj /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-aa /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ab /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ac /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ad /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ae /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-af /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ag /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ah /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-ai /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00003-of-00005.safetensors.part-aj /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-aa /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ab /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ac /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ad /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ae /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-af /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ag /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ah /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-ai /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00004-of-00005.safetensors.part-aj /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-aa /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-ab /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-ac /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-ad /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-ae /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/text_encoder/model-00005-of-00005.safetensors.part-af /app/models/wan_21_1b/text_encoder/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-aa /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-ab /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-ac /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-ad /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-ae /app/models/wan_21_1b/transformer/
COPY models/wan_21_1b/transformer/diffusion_pytorch_model.safetensors.part-af /app/models/wan_21_1b/transformer/

# Worker code (last to leverage cache)
COPY src /app/worker/src

# Install Python dependencies
# # --- Install SageAttention 2.x from source ---
#     RUN apt-get update && apt-get install -y git build-essential python3-dev && \
#         git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
#         cd /tmp/SageAttention && \
#         export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 && \
#         python setup.py install && \
#         cd / && rm -rf /tmp/SageAttention
RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV FASTWAN_1B_LOCAL_PATH="/app/models/wan_21_1b"
ENV REALESRGAN_LOCAL_PATH="/app/models/realesrgan"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
