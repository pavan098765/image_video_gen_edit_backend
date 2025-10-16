# BASE DEPS DOCKERFILE IMG P-3.11, CUDA 12.9
# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools needed only for wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip curl ca-certificates build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Create venv so all Python deps are isolated
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Workdir for building
WORKDIR /deps

# Copy wheels directory (already downloaded on host)
COPY wheels /deps/wheels

# Install PyTorch first (largest wheel, pinned to CUDA 12.9)
RUN sh -c "pip install --no-cache-dir /deps/wheels/torch-*.whl" \
 && rm -rf /root/.cache/pip /tmp/*

# Install Nvidia-specific deps (cuDNN, NCCL, etc.)
RUN sh -c "pip install --no-cache-dir /deps/wheels/nvidia_*.whl" \
 && rm -rf /root/.cache/pip /tmp/*

# Install all remaining wheels
RUN sh -c "pip install --no-cache-dir /deps/wheels/*.whl" \
 && rm -rf /root/.cache/pip /tmp/*

# Install Real-ESRGAN from source
COPY realesrgan /deps/realesrgan
RUN pip install --no-cache-dir ./realesrgan \
 && rm -rf /root/.cache/pip /tmp/* /deps/realesrgan/.git

# Install extra PyPI deps not bundled as wheels
RUN pip install --no-cache-dir \
    simple-lama-inpainting \
    aura-sr \
    gfpgan==1.3.8 \
    gguf==0.17.1 \
    opencv-python \
    tenacity \
    pillow \
    fastapi \
    uvicorn \
    uvloop \
    requests \
 && rm -rf /root/.cache/pip /tmp/*

# Clean up wheels after build
RUN rm -rf /deps/wheels


# =========================
# Stage 2: Final Runtime
# =========================
FROM python:3.11-slim

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Minimal runtime deps (no build-essential, cmake, ninja, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Cleanup: strip Python metadata, tests/docs, and shared lib debug symbols
RUN find /opt/venv/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + && \
    find /opt/venv/lib/python3.11/site-packages -type d -name "docs" -exec rm -rf {} + && \
    find /opt/venv -name "*.so" -exec strip --strip-unneeded {} + || true

# Workdir (blank, this is base only)
WORKDIR /app

# Default envs for HF + PyTorch
ENV XFORMERS_OFFLOAD_CPU="1" \
    PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

# Base deps image has no entrypoint
