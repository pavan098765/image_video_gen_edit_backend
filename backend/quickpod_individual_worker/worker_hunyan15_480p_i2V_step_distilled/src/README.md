# Hunyan15 480p Image-to-Video Worker

Production-ready Image-to-Video generation worker using Hunyan15 480p step distilled model.

## Model

**Model:** `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled`

This model works safely on:
- **RTX 4090** (24GB VRAM) - No OOM risk
- **RTX 5090** (32GB VRAM) - No OOM risk

The Hunyan15 480p i2v step distilled model provides fast Image-to-Video generation with optimized inference steps (12 steps).

## Features

### Video Generation
- **Image-to-Video (I2V)**: Animate images with text guidance
- Optimized for 480p resolution
- Step distilled inference (12 steps)
- Memory-efficient with model CPU offload
- VAE tiling for memory optimization

### Worker Features
- FastAPI-based REST API
- Background model preloading
- Job queue with status tracking
- Controller integration with heartbeat
- GPU memory management
- Health monitoring
- OOM recovery

## Quick Start

### 1. Build Docker Image

```bash
cd backend/quickpod_individual_worker/worker_hunyan15_480p_i2V_step_distilled
docker build -t hunyan15-i2v-worker .
```

### 2. Run Container

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e CONTROLLER_URL="http://your-controller:3000" \
  -e CONTROLLER_SHARED_SECRET="your-secret" \
  hunyan15-i2v-worker
```

## API Endpoints

### POST /run
Submit an image-to-video generation job.

```json
{
  "job_id": "unique-job-id",
  "prompt": "A cat wearing sunglasses sits on a surfboard on the beach",
  "model": "hunyan15_i2v",
  "mode": "image-to-video",
  "params": {
    "num_frames": 121,
    "num_inference_steps": 12,
    "guidance_scale": 7.5,
    "fps": 24,
    "seed": 42,
    "image_input": "base64_encoded_image_or_url"
  },
  "upload_url": "https://your-presigned-upload-url",
  "result_key": "videos/output.mp4"
}
```

### GET /health
Health check endpoint (always responds quickly).

Returns:
```json
{
  "status": "ok",
  "worker_id": "worker-uuid",
  "queue_size": 0,
  "model_loaded": true,
  "ready": true,
  "processing": false,
  "healthy": true,
  "gpu_info": {
    "available": true,
    "vram_gb": 24.0,
    "name": "NVIDIA RTX 4090"
  }
}
```

### GET /status
Detailed status including all jobs.

### GET /models
List available models and modes.

## Image-to-Video Generation

Hunyan15 specializes in Image-to-Video generation with step distillation for fast inference.

```json
{
  "job_id": "job-123",
  "prompt": "Summer beach vacation style, white cat on surfboard",
  "model": "hunyan15_i2v",
  "mode": "image-to-video",
  "params": {
    "image_input": "<base64-image-or-url>",
    "num_frames": 121,
    "num_inference_steps": 12,
    "guidance_scale": 7.5,
    "fps": 24,
    "seed": 42
  },
  "upload_url": "https://your-upload-url",
  "result_key": "videos/output.mp4"
}
```

**Parameters:**
- `image_input` (required): Base64 encoded image, image URL, or file path
- `prompt` (required): Text description of desired animation
- `num_frames` (int): Number of video frames (default: 121)
- `num_inference_steps` (int): Denoising steps (default: 12, recommended: 12)
- `guidance_scale` (float): Classifier-free guidance (default: 7.5)
- `fps` (int): Output fps (default: 24)
- `seed` (int): Random seed for reproducibility

## Model Details

### Hunyan15 1.5 Architecture
- **Resolution:** 480p output
- **Frame Count:** Up to 121 frames (~5 seconds at 24fps)
- **Input:** Single image + text prompt
- **Distillation:** Step-distilled for 12 inference steps
- **Memory:** ~18-20 GB during peak inference

### Performance (RTX 4090)
- **Generation Time:** 30-60 seconds per 121-frame video
- **Memory Usage:** Peak ~20 GB
- **Throughput:** 1 job per 1-2 minutes with full cleanup

### Official Example
```python
import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline, attention_backend
from diffusers.utils import export_to_video, load_image

dtype = torch.bfloat16
device = "cuda:0"

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(1)
image = load_image("image.jpg")

with attention_backend("_flash_3_hub"):
    video = pipe(
        prompt="Description of animation",
        image=image,
        generator=generator,
        num_frames=121,
        num_inference_steps=12,
    ).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROLLER_URL` | | Controller server URL |
| `CONTROLLER_SHARED_SECRET` | | HMAC signing secret |
| `CONTROLLER_API_KEY` | | API key for controller |
| `HUNYAN15_REPO_ID` | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled` | Model repo |
| `HUNYAN15_MODEL_LOCAL_PATH` | `/app/models/hunyan15` | Local model cache |
| `MODEL_KEY` | `Hunyan15-480p-i2v` | Worker model identifier |
| `WORKSPACE_DIR` | `/workspace` | Output directory |
| `HEARTBEAT_INTERVAL` | `20` | Heartbeat interval (seconds) |
| `MAX_CONCURRENT_JOBS` | `1` | Max concurrent jobs |

## Memory Optimization

The Hunyan15 i2v step distilled model is optimized for RTX 4090 (24GB):

1. **Efficient Architecture:** Requires ~18-20 GB during peak inference
2. **CPU Offload:** Model layers moved to CPU when idle
3. **VAE Tiling:** Reduces VAE memory footprint
4. **Step Distillation:** Only 12 inference steps vs 40+ for standard model

Memory profile:
- Model loading: ~8 GB
- Active inference: ~20 GB peak
- After cleanup: ~2 GB

## File Structure

```
worker_hunyan15_480p_i2v_step_distilled/
├── src/
│   ├── app.py                         # FastAPI application
│   ├── hunyan15_video_inference.py    # Hunyan15 i2v inference
│   ├── (legacy files removed)
│   ├── prep_model.py                  # Download models
│   ├── merge.py                       # Merge split safetensors
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # This file
└── Dockerfile                         # Docker configuration
```

## Development

### Local Testing

```bash
cd backend/quickpod_individual_worker/worker_hunyan15_480p_i2V_step_distilled/src
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Test Inference

```python
from hunyan15_video_inference import run_inference_hunyan15_i2v
from PIL import Image

# Create test image
image = Image.new('RGB', (512, 512), color='blue')
image.save('/tmp/test.jpg')

# Run inference
output = run_inference_hunyan15_i2v(
    job_id="test-001",
    prompt="A beautiful sunset over the ocean",
    image_input="/tmp/test.jpg",
    params={
        "num_frames": 121,
        "num_inference_steps": 12,
        "guidance_scale": 7.5,
        "fps": 24,
        "seed": 42
    }
)
print(f"Video saved to: {output}")
```

## API Integration Example

```python
import requests
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Submit job
job_data = {
    "job_id": "job-001",
    "prompt": "A white cat wearing sunglasses on the beach",
    "model": "hunyan15_i2v",
    "params": {
        "num_frames": 121,
        "num_inference_steps": 12,
        "guidance_scale": 7.5,
        "fps": 24,
        "image_input": image_b64
    },
    "upload_url": "https://upload.example.com/video.mp4",
    "result_key": "videos/output.mp4"
}

response = requests.post(
    "http://localhost:8000/run",
    json=job_data,
    timeout=600
)
print(response.json())
```

## Troubleshooting

### Out of Memory (OOM)
- Ensure `MAX_CONCURRENT_JOBS=1`
- Check GPU usage: `nvidia-smi`
- Reduce `num_frames` for testing
- Verify other processes aren't using GPU memory

### Slow Generation
- 12 steps is the minimum for step distilled model
- Check GPU utilization: `nvidia-smi`
- Ensure adequate system RAM (32GB+ recommended)

### Model Loading Issues
- Verify internet connection for HuggingFace download
- Check disk space: ~8 GB needed
- Verify cache directory permissions
- Check logs: `docker logs <container-id>`

## License

Hunyan15 model is available from HuggingFace community models.
See: https://huggingface.co/hunyuanvideo-community
