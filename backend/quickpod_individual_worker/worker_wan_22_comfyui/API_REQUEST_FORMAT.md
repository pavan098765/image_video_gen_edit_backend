# Wan 2.2 ComfyUI Worker — Request Format

Submit jobs to the **controller** `POST /submit` (or directly to the worker `POST /run`). The worker runs the v2.5.4 ComfyUI workflow with the given mode and params.

---

## Request: `POST /run` (worker) or via controller `POST /submit`

### Top-level payload

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | Yes | Unique job identifier |
| `prompt` | string | Yes | Text prompt for generation |
| `model` | string | Yes | `"wan_22_comfyui"` |
| `mode` | string | No | `i2v`, `t2v`, `t2i`, `s2v` (default `i2v`) |
| `params` | object | Yes | See **params** below |
| `upload_url` | string | Yes | Presigned URL for result upload |
| `result_key` | string | Yes | R2/result key for the output |

---

## `params` (wan_22_comfyui)

All of these go inside the `params` object.

### Generation

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `mode` | string | `"i2v"` | `i2v` (image-to-video), `t2v` (text-to-video), `t2i` (text-to-image), `s2v` (subject-to-video) |
| `init_image` | string | — | Base64-encoded reference image (required for `i2v` and `s2v`) |
| `prompt` | string | — | Override top-level prompt if needed |
| `negative_prompt` | string | (built-in) | Negative prompt |
| `enhance_prompt` | boolean | `false` | Use LLM to enhance prompt before generation |

### Frames and context

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `num_frames` | int | `81` | Frames per context (~5s at 16fps) |
| `num_contexts` | int | `1` | Number of context windows; use >1 for longer videos (single-pass) |

### Sampling

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `steps` | int | `8` | Sampling steps |
| `shift` | float | `6.9` | Shift value |
| `seed` | int | `-1` | Random seed (-1 = random) |

### Dimensions (T2V / T2I)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `width` | int | `832` | Output width |
| `height` | int | `480` | Output height |

### Frame interpolation (workflow FL_RIFE)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `enable_interpolation` | boolean | `true` | Use FL_RIFE frame interpolation |
| `interpolation_multiplier` | int | `2` | `1` = no extra frames, `2` = 2x frames |
| `rife_ckpt_name` | string | `"rife49"` | FL_RIFE checkpoint name |

### Upscale (workflow ImageUpscaleWithModelBatched)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `enable_upscale` | boolean | `true` | Use workflow upscale before interpolation |
| `upscale_model_name` | string | `"2x_NMKD-UpgifLiteV2_210k.pth"` | Upscale model filename (in ComfyUI `models/upscale_models`) |

### Other

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `output_dir` | string | `/workspace` | Output directory for the generated video |

---

## Example payloads

### I2V with interpolation and upscale (default)

```json
{
  "job_id": "job-001",
  "prompt": "A cat walking on a beach at sunset.",
  "model": "wan_22_comfyui",
  "mode": "i2v",
  "params": {
    "init_image": "<base64 image>",
    "num_frames": 81,
    "num_contexts": 1,
    "enable_interpolation": true,
    "interpolation_multiplier": 2,
    "enable_upscale": true
  },
  "upload_url": "https://...",
  "result_key": "results/job-001.mp4"
}
```

### T2V, no interpolation, no upscale

```json
{
  "job_id": "job-002",
  "prompt": "A drone flying over mountains.",
  "model": "wan_22_comfyui",
  "mode": "t2v",
  "params": {
    "num_frames": 81,
    "width": 832,
    "height": 480,
    "enable_interpolation": false,
    "enable_upscale": false
  },
  "upload_url": "https://...",
  "result_key": "results/job-002.mp4"
}
```

### Longer video (context windows)

```json
{
  "job_id": "job-003",
  "prompt": "Ocean waves at sunset.",
  "model": "wan_22_comfyui",
  "mode": "i2v",
  "params": {
    "init_image": "<base64>",
    "num_frames": 81,
    "num_contexts": 3,
    "enable_interpolation": true,
    "enable_upscale": true
  },
  "upload_url": "https://...",
  "result_key": "results/job-003.mp4"
}
```

---

## Summary

- **Modes:** `i2v`, `t2v`, `t2i`, `s2v`. For `i2v`/`s2v`, `init_image` (base64) is required.
- **Longer videos:** Set `num_contexts` > 1 (e.g. 3 for ~15s).
- **Interpolation:** Workflow uses built-in FL_RIFE; control with `enable_interpolation` and `interpolation_multiplier`.
- **Upscale:** Workflow upscale is before interpolation; set `enable_upscale` to `false` to bypass.
