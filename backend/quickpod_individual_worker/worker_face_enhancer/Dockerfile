# =============================
# Builder stage (copies pre-baked models)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129 AS builder

WORKDIR /app

# Copy full pre-baked realesrgan/gfpgan models into builder stage
COPY realesrgan_models_full /app/models/realesrgan

# =============================
# Runtime stage (lightweight)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Copy baked models from builder
COPY --from=builder /app/models /app/models

# Copy worker code *after* models so model layer is cached when only code changes
COPY . /app/worker

WORKDIR /app/worker

# Environment variables for baked model paths
ENV REALESRGAN_X4_PLUS_PATH="/app/models/realesrgan/RealESRGAN_x4plus.pth" \
    REALESRGAN_X4_GENERAL_WDN_PATH="/app/models/realesrgan/realesr-general-wdn-x4v3.pth" \
    REALESRGAN_X4_GENERAL_PATH="/app/models/realesrgan/realesr-general-x4v3.pth" \
    REALESRGAN_X4_PLUS_ANIME_PATH="/app/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" \
    REALESRGAN_X2_PATH="/app/models/realesrgan/RealESRGAN_x2plus.pth" \
    GFPGAN_MODEL_PATH="/app/models/realesrgan/GFPGANv1.4.pth" \
    PYTHONPATH="/app/worker/src:${PYTHONPATH}"

# Ensure log file exists
RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

EXPOSE 8000
