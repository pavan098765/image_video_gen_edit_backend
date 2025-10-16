# =============================
# Builder stage (bakes models)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129 AS builder

WORKDIR /app

# Copy prep script first (so baking layer can be cached)
COPY src/prep_model.py /app/prep_model.py

# Run prep_model.py to download and bake Swin2SR models into this stage
RUN mkdir -p /app/models && \
    python3 /app/prep_model.py && \
    rm -rf /root/.cache /tmp/*

# =============================
# Runtime stage (lightweight)
# =============================
FROM pavanalur3/deps-base-extended:py311-cu129

WORKDIR /app

# Copy baked models from builder (no runtime downloads needed)
COPY --from=builder /app/models /app/models

RUN pip install aiofiles aiohttp

# Copy worker code *after* models so model layer is cached when only code changes
COPY . /app/worker

# Env vars for baked model paths
ENV SWIN2SR_X2_LOCAL_PATH="/app/models/swin2sr/swin2sr_x2" \
    SWIN2SR_X4_LOCAL_PATH="/app/models/swin2sr/swin2sr_x4" \
    SWIN2SR_X4_CLASSIC_LOCAL_PATH="/app/models/swin2sr/swin2sr_x4_classic" \
    PYTHONPATH="/app/worker/src:${PYTHONPATH}"

# Ensure log file exists
RUN mkdir -p /app/worker && touch /app/worker/uvicorn.log

WORKDIR /app/worker
EXPOSE 8000
