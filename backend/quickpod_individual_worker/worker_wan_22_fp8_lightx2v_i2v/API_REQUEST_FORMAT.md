# WAN 2.2 FP8 LightX2V I2V — Client Request Format

Submit jobs to the **controller** `POST /submit` (not directly to the worker). The controller forwards the run to the wan_22_fp8_lightx2v_i2v worker.

---

## Request: `POST /submit`

### Top-level payload

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | `"wan_22_fp8_lightx2v_i2v"` |
| `mode` | string | Yes | `"i2v"` |
| `prompt` | string | Yes | Video description / motion prompt |
| `params` | object | Yes | See **params** below |
| `input_image_r2_keys` | string[] | No* | R2 keys for uploaded init image |
| `upload_url` | string | Yes | Presigned URL for result upload |
| `result_key` | string | Yes | R2 key for output video |

\* For I2V you must provide an init image via `input_image_r2_keys` (one key) or `params.init_image` (presigned URL). The controller resolves R2 keys to presigned URLs.

---

## `params` (wan_22_fp8_lightx2v_i2v)

### Image (required — one of)

| Key | Type | Description |
|-----|------|-------------|
| `init_image` | string | Presigned R2 download URL for the input image. Used when controller provides URL directly. |
| (via `input_image_r2_keys`) | — | Controller turns R2 key into URL and passes as `init_image`. |

### Long video (duration)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `duration` | number | 5 | Total video duration in seconds. Each clip ≈ 5 sec. If 10, generates 2 clips and stitches. Examples: 5 (1 clip), 10 (2 clips), 15 (3 clips), 20 (4 clips). |
| `extend_prompt` | boolean | `true` | If true and `duration` > 5, use LLM (Together API) to expand prompt into N segment prompts (one per clip). Requires `TOGETHER_API_KEY`. |

### Generation

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `num_frames` | number | 81 | Frames per clip (81 ≈ 5 sec at 16fps). |
| `fps` | number | 16 | Output frame rate. |
| `num_inference_steps` | number | 8 | Inference steps (Lightning model: 4–8). |
| `guidance_scale` | number | 1.0 | CFG scale. |
| `seed` | number | random | Random seed for reproducibility. |
| `resize_image` | boolean | `true` | Resize init image to 480×832 area (mod 16). |
| `negative_prompt` | string | default | Override negative prompt. Multi-clip uses extended negative (fog/haze, face drift suppression). |

### Post-processing

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `enable_rife` | boolean | `false` | Enable RIFE frame interpolation. |
| `rife_target_fps` | number | 30 | Target FPS after interpolation. |
| `enable_upscaling` | boolean | `false` | Enable Real-ESRGAN upscaling. |
| `enable_upscale` | boolean | `false` | Alias for `enable_upscaling`. |
| `upscale_factor` | number | 2 | Upscale factor (2, 4, or 8). |
| `postprocess_interpolate_first` | boolean | `false` | If true: interpolate then upscale. If false (default): upscale then interpolate. |

### Other

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `enhance_prompt` | boolean | `false` | Use Together API to enhance prompt (different from `extend_prompt`). |
| `output_dir` | string | `/workspace` | Output directory. |

---

## Example payloads

### 1) Single 5-second clip (default)

```json
{
  "job_id": "job-123",
  "prompt": "Animate this image with cinematic motion, smooth camera pan",
  "model": "wan_22_fp8_lightx2v_i2v",
  "mode": "i2v",
  "params": {
    "init_image": "https://...",
    "num_frames": 81,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "fps": 16
  },
  "upload_url": "https://...",
  "result_key": "job-123.mp4"
}
```

### 2) 10-second video (2 clips, prompt extension)

```json
{
  "job_id": "job-456",
  "prompt": "A woman walking through a garden, flowers swaying in the breeze",
  "model": "wan_22_fp8_lightx2v_i2v",
  "mode": "i2v",
  "params": {
    "init_image": "https://...",
    "duration": 10,
    "extend_prompt": true,
    "enable_rife": true,
    "rife_target_fps": 30,
    "enable_upscaling": true,
    "upscale_factor": 2
  },
  "upload_url": "https://...",
  "result_key": "job-456.mp4"
}
```

### 3) 15-second video with R2 image key

```json
{
  "job_id": "job-789",
  "prompt": "Cinematic reveal of a landscape at sunset",
  "model": "wan_22_fp8_lightx2v_i2v",
  "mode": "i2v",
  "params": {
    "duration": 15,
    "extend_prompt": true,
    "enable_rife": true,
    "enable_upscaling": true
  },
  "input_image_r2_keys": ["path/to/init-image.jpg"],
  "upload_url": "https://...",
  "result_key": "job-789.mp4"
}
```

---

## Summary

- **Endpoint:** controller `POST /submit` (or worker `POST /run` for direct calls).
- **Image:** `params.init_image` (presigned URL) or `input_image_r2_keys` (controller resolves to URL).
- **Duration:** `duration` (seconds). Each clip ≈ 5 sec. 10 sec → 2 clips, 15 sec → 3 clips, etc.
- **Prompt extension:** When `duration` > 5 and `extend_prompt` is true, LLM expands prompt into N segment prompts. Requires `TOGETHER_API_KEY`.
- **Post-processing:** `enable_rife`, `enable_upscaling` for interpolation and upscaling. Uses streaming to avoid OOM on long videos.
