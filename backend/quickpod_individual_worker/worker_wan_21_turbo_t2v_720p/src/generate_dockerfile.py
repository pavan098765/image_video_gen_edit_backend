#!/usr/bin/env python3
"""
Generate Dockerfile for WAN 2.1 TurboDiffusion T2V 720p worker.
COPYs models/. Clones TurboDiffusion repo and installs from source (with submodules);
installs SpargeAttn (SageSLA).
Uses nvidia/cuda:12.8.0-devel-ubuntu22.04 so the image has CUDA toolkit (nvcc) for
building TurboDiffusion and SpargeAttn; works on any pod (4090, 5090/Blackwell).
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")
MAX_FILE_BYTES = 700 * 1024 * 1024


def write_header(f):
    f.write(f"FROM {BASE_IMAGE}\n\n")
    f.write("ENV DEBIAN_FRONTEND=noninteractive\n\n")
    f.write("# Python 3.11 + build deps + gcc-11 (avoids CUDA 12.8 + glibc toolchain issues)\n")
    f.write("RUN apt-get update && apt-get install -y --no-install-recommends \\\n")
    f.write("    software-properties-common \\\n")
    f.write("    && add-apt-repository -y ppa:deadsnakes/ppa \\\n")
    f.write("    && apt-get update \\\n")
    f.write("    && apt-get install -y --no-install-recommends \\\n")
    f.write("    python3.11 python3.11-venv python3.11-dev \\\n")
    f.write("    git build-essential ninja-build gcc-11 g++-11 \\\n")
    f.write("    wget curl ca-certificates \\\n")
    f.write("    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \\\n")
    f.write("    && rm -rf /var/lib/apt/lists/*\n\n")
    f.write("ENV CUDA_HOME=/usr/local/cuda \\\n")
    f.write("    CC=gcc-11 CXX=g++-11 \\\n")
    f.write("    PATH=\"/usr/local/cuda/bin:$PATH\" \\\n")
    f.write("    TORCH_CUDA_ARCH_LIST=\"8.9;9.0;12.0\"\n\n")
    f.write("RUN python3.11 -m venv /opt/venv\n")
    f.write("ENV PATH=\"/opt/venv/bin:$PATH\"\n\n")
    f.write("RUN pip install --no-cache-dir --upgrade pip wheel setuptools\n\n")
    f.write("WORKDIR /app\n\n")


def copy_non_chunk_files(f):
    count = 0
    for root, dirs, files in os.walk(MODELS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if ".part-" in fn:
                continue
            full = os.path.join(root, fn)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel_src = os.path.relpath(full, PROJECT_ROOT).replace("\\", "/")
            rel_to_models = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
            f.write(f"COPY {rel_src} {dest_dir}/\n")
            count += 1
    return count


def copy_chunks(f):
    parts = []
    for root, _, files in os.walk(MODELS_DIR):
        if any(p.startswith(".") for p in root.split(os.sep)):
            continue
        for fn in files:
            if ".part-" in fn:
                parts.append(os.path.join(root, fn))
    parts.sort()
    for p in parts:
        rel_src = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
        rel_to_models = os.path.relpath(os.path.dirname(p), MODELS_DIR).replace("\\", "/")
        dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
        f.write(f"COPY {rel_src} {dest_dir}/\n")
    return len(parts)


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        write_header(f)
        f.write("# Non-chunk model files\n")
        n = copy_non_chunk_files(f)
        f.write("\n# Split .part-* chunks\n")
        c = copy_chunks(f)
        f.write("\nCOPY requirements.txt /app/requirements.txt\n\n")
        f.write("# PyTorch 2.8.0 CUDA 12 (for 4090 + 5090); TurboDiffusion README recommends torch 2.8.0\n")
        f.write("RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \\\n")
        f.write("    --index-url https://download.pytorch.org/whl/cu128\n\n")
        f.write("# Worker deps; TurboDiffusion installed from clone below\n")
        f.write("RUN pip install --no-cache-dir -r /app/requirements.txt\n\n")
        f.write("# Precompiled flash-attn wheel (cu12, torch 2.8, cp311) - no source build\n")
        f.write("RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl\n\n")
        f.write("# TurboDiffusion + SpargeAttn from source (CUDA_HOME + gcc-11 set above)\n")
        f.write("RUN git clone https://github.com/thu-ml/TurboDiffusion.git /app/TurboDiffusion \\\n")
        f.write("    && cd /app/TurboDiffusion && git submodule update --init --recursive \\\n")
        f.write("    && pip install --no-cache-dir psutil ninja packaging \\\n")
        f.write("    && MAX_JOBS=2 pip install --no-cache-dir -e /app/TurboDiffusion --no-build-isolation \\\n")
        f.write("    && MAX_JOBS=2 pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation \\\n")
        f.write("    && rm -rf /root/.cache /tmp/*\n\n")
        f.write("COPY src /app/worker/src\n\n")
        f.write('ENV MODELS_DIR="/app/models" \\\n')
        f.write('    PYTHONPATH="/app/TurboDiffusion/turbodiffusion:/app/worker/src:${PYTHONPATH}" \\\n')
        f.write('    TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"\n\n')
        f.write("RUN mkdir -p /app/worker /workspace\n")
        f.write("WORKDIR /app/worker\n")
        f.write("EXPOSE 8000\n")
    print(f"Wrote {OUT_DOCKERFILE} (non-chunk: {n}, chunks: {c})")


if __name__ == "__main__":
    main()
