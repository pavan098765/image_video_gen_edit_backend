# FlashVSR Video Upscaler — Client Request Format

Submit jobs to the **controller** `POST /submit` (not directly to the worker). The controller forwards the run to the flashvsr_video_upscaler worker with the params you send.

---

## Request: `POST /submit`

### Top-level payload

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | `"flashvsr_video_upscaler"` |
| `prompt` | string | No | Can be `""` for video upscaling |
| `params` | object | Yes | See **params** below |
| `input_video_r2_key` | string | No* | R2 key for uploaded video (preferred over base64) |
| `fcm_token` | string | No | FCM token for push notifications |
| `variations` | number | No | Default `1` |

\* For video input you must send **either** `input_video_r2_key` **or** put the video in `params.input_video` (base64 or URL).

---

## `params` (flashvsr_video_upscaler)

All of these go inside the `params` object.

### Video (required — one of)

| Key | Type | Description |
|-----|------|-------------|
| `input_video` | string | Base64 video, URL, or file path. Prefer R2 flow and omit when using `input_video_r2_key`. |

### FlashVSR options

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `mode` | string | `"tiny"` | `full` (high quality, high VRAM), `tiny` (balanced), `tiny-long` (for long videos) |
| `scale` | number | `2.0` | Upscale factor (2.0 or 4.0 recommended) |
| `keep_audio` | boolean | `false` | Preserve audio from input video |
| `tile_dit` | boolean | `false` | Enable tiled DiT (reduces VRAM) |
| `tile_vae` | boolean | `false` | Enable tiled VAE decoding (full mode) |
| `tile_size` | number | `256` | Tile size for tiled inference |
| `overlap` | number | `24` | Overlap between tiles |
| `color_fix` | boolean | `false` | Apply color correction |
| `quality` | number | `10` | Output video quality (0–10) |
| `fps` | number | (input) | Force output FPS |
| `seed` | number | `0` | Random seed |
| `output_dir` | string | `/workspace` | Output directory |

---

## Example payloads

### 1) R2 video (preferred)

```json
{
  "model": "flashvsr_video_upscaler",
  "prompt": "",
  "params": {
    "mode": "tiny",
    "scale": 2.0,
    "keep_audio": true
  },
  "input_video_r2_key": "image-video-generator-api/input-media/your-video-id.bin",
  "variations": 1
}
```

### 2) URL video

```json
{
  "model": "flashvsr_video_upscaler",
  "prompt": "",
  "params": {
    "input_video": "https://example.com/video.mp4",
    "mode": "tiny",
    "scale": 4.0,
    "tile_dit": true
  },
  "variations": 1
}
```

### 3) Base64 video (fallback)

```json
{
  "model": "flashvsr_video_upscaler",
  "prompt": "",
  "params": {
    "input_video": "data:video/mp4;base64,AAAA...",
    "mode": "tiny-long",
    "scale": 2.0,
    "keep_audio": true
  },
  "variations": 1
}
```

---

## Summary

- **Endpoint:** controller `POST /submit`.
- **Video:** either `input_video_r2_key` or `params.input_video` (URL/base64).
- **mode:** `full`, `tiny`, or `tiny-long`.
- **scale:** 2.0 or 4.0 (4.0 optimal per FlashVSR).
- **tile_dit** / **tile_vae:** Use for lower VRAM (e.g. 8GB+ GPUs).
