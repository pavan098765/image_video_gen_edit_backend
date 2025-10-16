FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# --- Qwen 2509 + Nunchaku build ---

# Non-chunk model files (configs, tokenizer, ckpt, etc.)
COPY models/realesrgan/RealESRGAN_x2plus.pth /app/models/realesrgan/
COPY models/realesrgan/RealESRGAN_x4plus.pth /app/models/realesrgan/

# Per-part COPYs (one COPY per chunk => one layer per chunk)

# Worker code (last to leverage cache)
COPY src /app/worker/src

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git \
    && apt-get purge -y git \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV QWEN_2509_LOCAL_PATH="/app/models/qwen_image_edit_2509"
ENV NUNCHAKU_QWEN_2509_LOCAL_PATH="/app/models/nunchaku_qwen_2509"
ENV REALESRGAN_LOCAL_PATH="/app/models/realesrgan"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
