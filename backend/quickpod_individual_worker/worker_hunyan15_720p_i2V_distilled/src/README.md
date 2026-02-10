# Hunyan15 720p I2V Distilled Worker

Production-ready video generation worker using Hunyan15 720p Image-to-Video Distilled model.

## Model

**Model:** Hunyan15 720p I2V Distilled from HuggingFace

This model works safely on:
- **RTX 4090** (24GB VRAM) - No OOM risk
- **RTX 5090** (32GB VRAM) - No OOM risk

The FP4 quantized model provides excellent quality while ensuring safe operation on consumer GPUs.

## Features

### Video Generation Modes
- **Text-to-Video (T2V)**: Generate video from text prompts
- **Image-to-Video (I2V)**: Animate images with text guidance
- **Video-to-Video (V2V)**: Transform existing videos
- **Keyframe Interpolation**: Generate smooth transitions between keyframes

### Worker Features
- FastAPI-based REST API
- Background model preloading
- Job queue with status tracking
- Controller integration with heartbeat
- GPU memory management
- Health monitoring
- OOM recovery

## Quick Start

### 1. Prepare Models

Run the prep script to download models and generate Dockerfile:

```bash
cd backend/quickpod_individual_worker/worker_hunyan15_720p_i2V_distilled/src
python prep_model.py
```

This will:
1. Download Hunyan15 720p I2V Distilled model from HuggingFace
2. Split large files for Docker layer caching
3. Generate the Dockerfile

### 2. Build Docker Image

```bash
cd backend/quickpod_individual_worker/worker_hunyan15_720p_i2V_distilled
docker build -t hunyan15-720p-i2v-worker .
```

### 3. Run Container

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e CONTROLLER_URL="http://your-controller:3000" \
  -e CONTROLLER_SHARED_SECRET="your-secret" \
  hunyan15-720p-i2v-worker
```

## API Endpoints

### POST /run
Submit a video generation job.

```json
{
  "job_id": "unique-job-id",
  "prompt": "A cinematic drone shot of mountains at sunrise",
  "model": "hunyan15_720p_i2v_distilled",
  "mode": "text-to-video",
  "params": {
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "fps": 25,
    "num_inference_steps": 12,  # Step distilled model uses 12 steps
    "pipeline_type": "distilled"
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
  "healthy": true
}
```

### GET /status
Detailed status including all jobs.

### GET /models
List available models and modes.

## Generation Modes

### Text-to-Video
```json
{
  "mode": "text-to-video",
  "params": {
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "pipeline_type": "distilled"
  }
}
```

### Image-to-Video
```json
{
  "mode": "image-to-video",
  "params": {
    "init_image": "<base64-encoded-image>",
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "conditioning_strength": 1.0
  }
}
```

### Video-to-Video
```json
{
  "mode": "video-to-video",
  "params": {
    "init_video": "<base64-encoded-video>",
    "conditioning_strength": 0.8
  }
}
```

### Keyframe Interpolation
```json
{
  "mode": "keyframe-interpolation",
  "params": {
    "keyframes": [
      {"image": "<base64>", "frame_index": 0, "strength": 1.0},
      {"image": "<base64>", "frame_index": 120, "strength": 1.0}
    ],
    "num_frames": 121
  }
}
```

## Pipeline Types

| Pipeline | Steps | Speed | Quality | Use Case |
|----------|-------|-------|---------|----------|
| `distilled` | 8+4 | Fastest | Good | Real-time, preview |
| `two_stage` | 40+20 | Slower | Best | Production |
| `ic_lora` | 40 | Medium | Good | Video-to-video |
| `keyframe` | 40 | Medium | Good | Keyframe interpolation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROLLER_URL` | | Controller server URL |
| `CONTROLLER_SHARED_SECRET` | | HMAC signing secret |
| `CONTROLLER_API_KEY` | | API key for controller |
| `HUNYAN15_720P_LOCAL_PATH` | `/app/models/hunyan15_720p` | Model directory |
| `HUNYAN15_DEFAULT_PIPELINE` | `i2v` | Default pipeline type |
| `WORKSPACE_DIR` | `/workspace` | Output directory |
| `HEARTBEAT_INTERVAL` | `20` | Heartbeat interval (seconds) |
| `MAX_CONCURRENT_JOBS` | `1` | Max concurrent jobs |
| `ENABLE_AUTOTEST` | `1` | Enable auto-test on startup |

## Memory Optimization

The Hunyan15 720p I2V Distilled model is used because:

1. **20 GB checkpoint** fits comfortably in 24GB VRAM (RTX 4090)
2. **No OOM risk** on both RTX 4090 and RTX 5090
3. **Fast inference** with distilled pipeline (8 steps)
4. **Good quality** despite quantization

Additional optimizations:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory allocation
- xFormers memory-efficient attention (when available)
- Aggressive GPU cleanup between jobs
- Single worker thread to prevent OOM from concurrent jobs

## File Structure

```
worker_hunyan15_720p_i2V_distilled/
├── src/
│   ├── app.py                    # FastAPI application
│   ├── hunyan15_video_inference.py   # Inference logic
│   ├── prep_model.py             # Download models & generate Dockerfile
│   ├── generate_dockerfile.py    # Dockerfile generator
│   ├── merge.py                  # Merge split safetensors
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # This file
├── models/                       # Downloaded models (after prep)
│   ├── hunyan15_720p/
│   │   └── (model files from HuggingFace)
│   │   └── ... (diffusers components)
│   └── gemma/
│       └── ... (Gemma 3 text encoder)
└── Dockerfile                    # Generated by prep_model.py
```

## Development

### Local Testing

```bash
cd backend/quickpod_individual_worker/worker_hunyan15_720p_i2V_distilled/src
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Test Inference

```python
from hunyan15_video_inference import run_inference_hunyan15_i2v

output = run_inference_hunyan15_i2v(
    job_id="test-001",
    prompt="A cinematic drone shot of mountains at sunrise",
    params={
        "mode": "text-to-video",
        "height": 512,
        "width": 768,
        "duration": 3.0,
        "pipeline_type": "distilled",
        "output_dir": "./outputs",
    }
)
print(f"Video saved to: {output}")
```

## License

Hunyan15 model is licensed under its respective terms. See the HuggingFace model page for details.
