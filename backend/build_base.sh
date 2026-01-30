#!/bin/bash
# ============================================================
# Build Script for Modular Base Docker Image
# ============================================================

set -e

# Default values
CUDA_VERSION="${1:-12.9}"
CUDA_SHORT="${2:-129}"
PYTORCH_VERSION="${3:-2.7.1}"
IMAGE_TAG="${4:-deps-base:py311-cu${CUDA_SHORT}}"
USE_WHEELS="${5:-false}"
INSTALL_NUNCHAKU="${6:-false}"

echo "=========================================="
echo "Building Base Dependencies Image"
echo "=========================================="
echo "CUDA Version:      ${CUDA_VERSION}"
echo "CUDA Short:        ${CUDA_SHORT}"
echo "PyTorch Version:  ${PYTORCH_VERSION}"
echo "Image Tag:        ${IMAGE_TAG}"
echo "Use Wheels:       ${USE_WHEELS}"
echo "Install Nunchaku: ${INSTALL_NUNCHAKU}"
echo "=========================================="
echo ""

# Build arguments
BUILD_ARGS=(
    "--build-arg" "CUDA_VERSION=${CUDA_VERSION}"
    "--build-arg" "CUDA_SHORT=${CUDA_SHORT}"
    "--build-arg" "PYTORCH_VERSION=${PYTORCH_VERSION}"
    "--build-arg" "USE_WHEELS=${USE_WHEELS}"
    "--build-arg" "INSTALL_NUNCHAKU=${INSTALL_NUNCHAKU}"
)

# If using wheels, ensure wheels directory exists
if [ "$USE_WHEELS" = "true" ]; then
    if [ ! -d "wheels" ]; then
        echo "ERROR: wheels/ directory not found!"
        echo "Set USE_WHEELS=false or create wheels/ directory with PyTorch wheels"
        exit 1
    fi
    BUILD_ARGS+=("--build-arg" "USE_WHEELS=true")
fi

# If installing Real-ESRGAN from source, ensure directory exists
if [ -d "realesrgan" ]; then
    echo "Found realesrgan/ directory, will install from source"
fi

# Build the image
docker build \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_TAG}" \
    -f Dockerfile.base \
    .

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Image: ${IMAGE_TAG}"
echo ""
echo "To use this image:"
echo "  FROM ${IMAGE_TAG}"
echo ""
