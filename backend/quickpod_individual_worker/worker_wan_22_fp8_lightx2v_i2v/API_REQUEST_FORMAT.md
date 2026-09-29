# WAN 2.2 FP8 LightX2V I2V — Client Request Format

Submit jobs to the **controller** `POST /submit` (not directly to the worker). The controller forwards the run to the wan_22_fp8_lightx2v_i2v worker.

### Inference backend (omni multi-worker)

On the worker image, set **`WAN_I2V_INFERENCE_MODULE`** to choose the Python module:

| Value | Behavior |
|-------|----------|
| `inference.py` | **V1** legacy: multi-clip long video when multiple segment prompts / duration chaining; ignores `last_image`. |
| `inference_sdnq_space_parity.py` | **V2** SDNQ: **single-pass** only (duration → one `num_frames`, up to ~10s @ 16 fps); Space-style schedulers, dual CFG, optional `last_image`, etc. |
| `inference_hub_space_torchao.py` | **V3** Hub + torchao + AoTI (~70 GB class); requires extra pip deps. |

Details: [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md).

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
| `last_image` | string | Optional presigned URL for **end** frame (V2/V3). Alias: `last_frame_image`. Ignored by V1. |

### Long video (duration)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `duration` | number | — | **V1:** Total seconds; multi-clip ≈5s segments + stitch when using segment prompts / controller behavior. **V2:** If `num_frames` omitted, maps to one shot: `num_frames = clamp(round(duration×16), 8, 160)` (single `pipe()`). |
| `duration_seconds` | number | — | Same as `duration` for V2 resolution (either key accepted). |
| `extend_prompt` | boolean | `true` | V1 / controller: LLM segment prompts when duration > 5. Not used by V2 single-pass path. |

### Generation

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `num_frames` | number | 81 | If set, overrides duration-derived frame count (V2). V1: frames **per clip**. |
| `fps` | number | 16 | Output / base frame rate. |
| `num_inference_steps` | number | 8 | Inference steps (Lightning model: 4–8). |
| `guidance_scale` | number | 1.0 | CFG (high-noise stage). |
| `guidance_scale_2` | number | 1.0 | CFG second stage (**V2/V3**); passed to pipeline when supported. |
| `seed` | number | random | Random seed for reproducibility. |
| `resize_image` | boolean | `true` | Resize init image. |
| `resize_mode` | string | `area480` (V2) | `area480` = max-area ~480×832 (mod VAE). `wan_space` = Space crop/aspect rules (**V2**). |
| `scheduler` / `scheduler_name` | string | `UniPCMultistep` | **V2/V3** scheduler key (see INFERENCE_BACKENDS). |
| `flow_shift` | number | 3.0 | **V2/V3** flow shift for compatible schedulers. |
| `video_quality` | number | 6 | **V2/V3** `export_to_video` quality 1–10 when supported. |
| `negative_prompt` | string | default | Override negative prompt. V1 multi-clip may use extended negative. **V2** uses default + override only (no extended multi-clip string). |

### Post-processing

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `enable_rife` | boolean | `false` | Enable RIFE frame interpolation. |
| `rife_target_fps` | number | 30 | Target FPS after interpolation. |
| `frame_multiplier` | number | — | **V2:** 16 / 32 / 64 / 128 = target output FPS (Space “fluidity”). If `> 16`, enables RIFE via **video_postprocessing** and sets target fps; overrides `rife_target_fps` when set. |
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
- **Image:** `params.init_image` (presigned URL) or `input_image_r2_keys` (controller resolves to URL). Optional **`last_image`** for V2/V3.
- **Worker backend:** `WAN_I2V_INFERENCE_MODULE` selects V1 / V2 / V3 (see [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md)).
- **V1 duration:** multi-clip ≈5s segments when using extended workflows; **V2 duration:** single trajectory up to ~10s from one `num_frames` (8–160 @ 16 fps).
- **Prompt extension:** V1/controller: when `duration` > 5 and `extend_prompt` is true, LLM may expand prompts. Requires `TOGETHER_API_KEY` where used.
- **Post-processing:** `enable_rife`, `frame_multiplier` (V2), `enable_upscaling`; streaming RIFE/upscale for long outputs.
