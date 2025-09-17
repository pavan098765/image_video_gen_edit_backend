# Base image with CUDA 12.8 + cuDNN + Python
FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl aria2 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel \

# Copy requirements
COPY requirements.txt /workspace/

# Install PyTorch + Torchvision + Xformers with CUDA 12.8 wheels explicitly
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt

# Clone ComfyUI + custom nodes
RUN git clone --branch ComfyUI_v0.3.51 https://github.com/Isi-dev/ComfyUI /workspace/ComfyUI \
    && cd /workspace/ComfyUI/custom_nodes \
    && git clone https://github.com/Isi-dev/ComfyUI_Img2PaintingAssistant \
    && git clone --branch forQwen https://github.com/Isi-dev/ComfyUI_GGUF.git \
    && git clone https://github.com/Isi-dev/ComfyUI_KJNodes.git

# Install ComfyUI GGUF requirements
RUN pip install --no-cache-dir -r /workspace/ComfyUI/custom_nodes/ComfyUI_GGUF/requirements.txt

# Copy your code
COPY src /workspace/src
COPY worker_entrypoint.sh /workspace/worker_entrypoint.sh
COPY models /workspace/models

# Permissions
RUN chmod +x /workspace/worker_entrypoint.sh

# Default env vars
ENV PORT=8080
EXPOSE 8080

# Start FastAPI app
ENTRYPOINT ["/workspace/worker_entrypoint.sh"]
