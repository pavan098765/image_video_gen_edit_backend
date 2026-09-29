FROM pavanalur3/deps-base-new:py311-cu128
WORKDIR /app
ENV OMNI_SLIM_INFERENCE_SRC=1
ENV OMNI_MULTI_IMAGE_ID=worker_ltx2_3_multi
# Omni runtime app/router for this specific multi image
COPY worker_ltx2_3_multi/src /app/worker/src
# Shared omni runtime framework
COPY worker_multi_common/src /app/worker_multi_common/src

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg build-essential \
    && git clone --depth 1 https://github.com/Disty0/sdnq.git /opt/sdnq \
    && pip install --no-cache-dir -e /opt/sdnq \
    && pip install --no-cache-dir -r /app/worker/src/requirements.txt \
    && pip install --no-cache-dir "huggingface_hub[hf_transfer]" hf_xet \
    && mkdir -p /app/worker_multi_common \
    && [ -f /app/worker_multi_common/omni_manifest.json ] || printf '{}\n' > /app/worker_multi_common/omni_manifest.json \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV PYTHONPATH=/app/worker/src:/app/worker_multi_common/src:${PYTHONPATH}
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
# Encrypted JSON + HMAC with controller (set CONTROLLER_SHARED_SECRET at runtime).
ENV WORKER_TRANSPORT_CRYPTO=1
ENV MODELS_DIR=/app/models
# diffusers v0.36.x omni primary models; LTX (v038) overrides via write_flavor_env ltx2_3.
ENV OMNI_POD_DIFFUSERS_GROUP=v036
ENV QPW_ROOT=/app/worker/src/inference
ENV OMNI_RUNTIME_MANIFEST_PATH=/app/worker_multi_common/omni_manifest.json
ENV OMNI_RUNTIME_MANIFEST_SYNC_PATH=true
ENV QWEN_2511_OFFLINE=1
ENV REALESRGAN_LOCAL_PATH=/app/models/realesrgan
ENV AURASR_LOCAL_PATH=/app/models/aurasr_v2
ENV SWIN2SR_X2_LOCAL_PATH=/app/models/swin2sr/swin2sr_x2
ENV SWIN2SR_X4_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4
ENV SWIN2SR_X4_CLASSIC_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4_classic
ENV SWIN2SR_HF_CACHE_DIR=/app/models/swin2sr_hf_cache
ENV BIREFNET_LOCAL_PATH=/app/models/basic_utils/birefnet
ENV LAMA_LOCAL_PATH=/app/models/basic_utils/lama/lama.pt
ENV GLM_OCR_LOCAL_PATH=/app/models/glm_ocr
ENV OMNI_POD_DIFFUSERS_GROUP=v038
ENV WORKSPACE_DIR=/workspace
ENV LTX2_3_MODELS_ROOT=/app/models/ltx2_3
ENV LTX2_3_DISTILLED_LOCAL=/app/models/ltx2_3/distilled
ENV LTX2_3_SPATIAL_UPSAMPLER_LOCAL=/app/models/ltx2_3/spatial_upsampler
ENV LTX2_3_LORA_CACHE=/app/models/ltx2_3/lora_cache
ENV LTX2_3_OFFLOAD_MODE=model
ENV LTX2_3_STAGE2_OFFLOAD=sequential
ENV CIVITAI_LORA_URL="https://civitai.com/models/1811313/dr34ml4y-all-in-one-nsfw-wanltx2"
ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511
ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers
ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters
ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo
ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo
ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps
ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet
ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae
ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline
ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15
ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v

WORKDIR /app/worker
EXPOSE 8000
