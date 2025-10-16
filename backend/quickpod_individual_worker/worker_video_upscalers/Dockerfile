FROM pavanalur3/deps-base:py311-cu129

WORKDIR /app

# Copy only prep_model.py first so model download can be cached separately
COPY src/prep_model.py /app/prep_model.py

# Download models (heavy, cached unless prep_model.py changes)
RUN mkdir -p /app/models && \
    python3 /app/prep_model.py

# Now copy the rest of the worker code (FastAPI + inference logic)
# Code changes here won’t bust the model cache
COPY . /app/worker

# Env vars for baked model paths
ENV CTRLNET_LOCAL_PATH="/app/models/sdxl_outpaint/controlnet" \
    VAE_LOCAL_PATH="/app/models/sdxl_outpaint/vae" \
    PIPELINE_LOCAL_PATH="/app/models/sdxl_outpaint/pipeline"

# Add src folder to PYTHONPATH so imports like `from fastwan_1b_inference` work
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"

# Expose FastAPI port
WORKDIR /app/worker
EXPOSE 8000

# Start worker
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port 8000 --log-level debug --access-log | tee /app/worker/uvicorn.log"]
