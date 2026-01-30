# Modular Base Dependencies Dockerfile

A unified, flexible base Docker image for ML inference workers that supports multiple CUDA versions and easy package updates.

## Features

- ✅ **Multi-CUDA Support**: Build for CUDA 12.8, 12.9, or 13.0
- ✅ **Flexible PyTorch**: Use pre-built wheels OR PyPI (default)
- ✅ **Modular Structure**: Easy to update individual package groups
- ✅ **Optional Components**: Nunchaku, Real-ESRGAN (conditional)
- ✅ **Optimized Size**: Multi-stage build, stripped binaries
- ✅ **QuickPod Compatible**: No entrypoint, ready for worker customization

## Quick Start

### Build for CUDA 12.9 (Default - Most Common)

```bash
# Using PyPI (recommended - most flexible)
docker build -t deps-base:py311-cu129 -f Dockerfile.base .

# Or use the helper script
./build_base.sh 12.9 129
```

### Build for CUDA 12.8 (Legacy Machines)

```bash
docker build \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg CUDA_SHORT=128 \
    -t deps-base:py311-cu128 \
    -f Dockerfile.base .

# Or use helper script
./build_base.sh 12.8 128
```

### Build for CUDA 13.0 (Future-Proof)

```bash
docker build \
    --build-arg CUDA_VERSION=13.0 \
    --build-arg CUDA_SHORT=130 \
    -t deps-base:py311-cu130 \
    -f Dockerfile.base .

# Or use helper script
./build_base.sh 13.0 130
```

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.9` | CUDA version (12.8, 12.9, 13.0) |
| `CUDA_SHORT` | `129` | Short CUDA code for PyPI (128, 129, 130) |
| `PYTHON_VERSION` | `3.11` | Python version |
| `PYTORCH_VERSION` | `2.7.1` | PyTorch version |
| `USE_WHEELS` | `false` | Use local wheels/ directory if true |
| `INSTALL_NUNCHAKU` | `false` | Install Nunchaku for INT4/FP4 quantization |

## Advanced Usage

### Using Pre-Built Wheels

If you have PyTorch wheels in a `wheels/` directory:

```bash
docker build \
    --build-arg USE_WHEELS=true \
    -t deps-base:py311-cu129 \
    -f Dockerfile.base .
```

### Including Nunchaku (for INT4/FP4 models)

```bash
docker build \
    --build-arg INSTALL_NUNCHAKU=true \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg CUDA_SHORT=128 \
    -t deps-base:py311-cu128-nunchaku \
    -f Dockerfile.base .
```

### Including Real-ESRGAN from Source

Place Real-ESRGAN source in `realesrgan/` directory:

```bash
# Clone Real-ESRGAN if needed
git clone https://github.com/xinntao/Real-ESRGAN.git realesrgan

# Build (will auto-detect and install)
docker build -t deps-base:py311-cu129 -f Dockerfile.base .
```

## Package Update Guide

The Dockerfile is organized into clear sections for easy updates:

### 1. Core ML/AI Dependencies (Lines ~80-90)
```dockerfile
RUN pip install --no-cache-dir \
    numpy \
    accelerate==1.10.1 \
    diffusers==0.35.1 \
    transformers==4.56.2 \
    ...
```

**To update**: Change version numbers or add new packages here.

### 2. Web Framework (Lines ~95-105)
```dockerfile
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    uvloop \
    ...
```

**To update**: Add/remove web framework packages.

### 3. Image Processing (Lines ~110-120)
```dockerfile
RUN pip install --no-cache-dir \
    pillow \
    opencv-python \
    imageio-ffmpeg \
    ...
```

**To update**: Add/remove image processing libraries.

### 4. Specialized ML Packages (Lines ~125-135)
```dockerfile
RUN pip install --no-cache-dir \
    simple-lama-inpainting \
    aura-sr \
    gfpgan==1.3.8 \
    ...
```

**To update**: Add worker-specific ML packages.

## Comparison with Previous Approaches

### Approach 1: `Dockerfile` (Wheels-Based)
- ❌ Hard-coded CUDA 12.9
- ❌ Requires pre-downloaded wheels
- ✅ Smaller image (if wheels optimized)
- ✅ Faster builds (if wheels cached)

### Approach 2: `Dockerfile - Copy` (Extends Base)
- ❌ Depends on external base image
- ❌ Less control over base
- ✅ Minimal additions
- ✅ Fast worker builds

### Approach 3: `Dockerfile_int4` (PyPI-Based)
- ❌ Hard-coded CUDA 12.8
- ✅ Flexible (PyPI)
- ✅ No wheel management
- ✅ Easy to update

### New Approach: `Dockerfile.base` (Unified)
- ✅ **Configurable CUDA version** (12.8, 12.9, 13.0)
- ✅ **Flexible PyTorch** (wheels OR PyPI)
- ✅ **Modular structure** (easy updates)
- ✅ **Optional components** (Nunchaku, Real-ESRGAN)
- ✅ **Single source of truth**

## Recommended Workflow

1. **Build base images for each CUDA version you need:**
   ```bash
   ./build_base.sh 12.8 128  # For legacy machines
   ./build_base.sh 12.9 129  # For most machines
   ```

2. **Tag and push to registry:**
   ```bash
   docker tag deps-base:py311-cu129 your-registry/deps-base:py311-cu129
   docker push your-registry/deps-base:py311-cu129
   ```

3. **Workers extend from base:**
   ```dockerfile
   FROM your-registry/deps-base:py311-cu129
   # Add worker-specific code and models
   ```

## Troubleshooting

### PyTorch Installation Fails

- **Issue**: CUDA version mismatch
- **Fix**: Ensure `CUDA_SHORT` matches your CUDA version:
  - CUDA 12.8 → `CUDA_SHORT=128`
  - CUDA 12.9 → `CUDA_SHORT=129`
  - CUDA 13.0 → `CUDA_SHORT=130`

### Nunchaku Not Installing

- **Issue**: Wheel not available for CUDA version
- **Fix**: Check [Nunchaku releases](https://github.com/nunchaku-tech/nunchaku/releases) for available wheels

### Image Too Large

- **Solution**: The Dockerfile already strips tests/docs. For further reduction:
  - Use `python:3.11-slim` (already used)
  - Remove optional packages you don't need
  - Use multi-stage builds (already implemented)

## Version Compatibility Matrix

| CUDA Version | PyTorch Index | NVIDIA Libraries | Nunchaku Support |
|--------------|----------------|------------------|------------------|
| 12.8         | `cu128`        | `cu12`           | ✅ Yes (v1.2.1)  |
| 12.9         | `cu129`        | `cu12`           | ⚠️ Check releases |
| 13.0         | `cu130`        | `cu12`           | ⚠️ Check releases |

## Maintenance

To update packages:

1. Edit the relevant section in `Dockerfile.base`
2. Rebuild for each CUDA version you support:
   ```bash
   ./build_base.sh 12.8 128
   ./build_base.sh 12.9 129
   ```
3. Test with a worker image
4. Push updated base images

---

**Last Updated**: 2025-01-27  
**Maintainer**: Your Team
