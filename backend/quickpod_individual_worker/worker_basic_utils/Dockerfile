# =============================
# Builder stage (bakes models)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129 AS builder

WORKDIR /app

# Copy prep script + pre-baked models in one go
COPY src/prep_model.py /app/prep_model.py
COPY realesrgan_models /app/models/realesrgan
COPY lama.pt /app/models/basic_utils/lama/lama.pt

# Run prep_model.py to download and bake all models into this stage
RUN mkdir -p /app/models && \
    python3 /app/prep_model.py && \
    rm -rf /root/.cache /tmp/*

# =============================
# Runtime stage (lightweight)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Copy baked models from builder (no runtime downloads)
COPY --from=builder /app/models /app/models

# Copy code *after* models so model layer is cached when only code changes
COPY . /app/worker

# Set env vars so inference loads baked model paths
ENV RMBG_LOCAL_PATH="/app/models/basic_utils/rmbg" \
    LAMA_LOCAL_PATH="/app/models/basic_utils/lama/lama.pt" \
    LAMA_MODEL="/app/models/basic_utils/lama/lama.pt" \
    PYTHONPATH="/app/worker/src:${PYTHONPATH}"

# Ensure log file exists
RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
