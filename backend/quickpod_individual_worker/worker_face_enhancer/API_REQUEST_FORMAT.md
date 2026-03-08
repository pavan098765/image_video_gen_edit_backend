# Face Enhancer — Client Request Format

Submit jobs to the **controller** `POST /submit` (not directly to the worker). The controller forwards the run to the face_enhancer worker with the params you send.

---

## Request: `POST /submit`

### Top-level payload

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | `"face_enhancer"` |
| `prompt` | string | No | Can be `""` for face_enhancer |
| `params` | object | Yes | See **params** below |
| `fcm_token` | string | No | FCM token for push notifications |
| `enhance_prompt` | boolean | No | Default `false` |
| `variations` | number | No | 1–4, default `1` |
| `input_image_r2_keys` | string[] | No* | R2 keys for uploaded images (preferred over base64) |
| `is_bulk` | boolean | No | Default `false` |

\* For image input you must send **either** `input_image_r2_keys` (one key) **or** put the image in `params.init_image` / `params.image` (base64).

---

## `params` (face_enhancer)

All of these go inside the `params` object.

### Image (required — one of)

| Key | Type | Description |
|-----|------|-------------|
| `init_image` | string | Base64 image **or** URL (if controller gave a presigned URL). Prefer R2 flow and omit this when using `input_image_r2_keys`. |
| `image` | string | Same as `init_image` (worker accepts either). |

When using R2: upload image first, then send `input_image_r2_keys: ["your-r2-key"]` and **do not** put image data in `params`. The controller will turn the key into a URL and the worker gets it as `init_image`.

### Scale and output

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `scale` | number | `2` | Output scale factor (e.g. 2, 2.5, 4). |
| `output_dir` | string | worker default | Output directory (optional). |

### Face vs general upscale

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `face_enhance` | boolean or string | `true` | `true` = GFPGAN + RealESRGAN (face enhancement). `false` = general upscale only (RealESRGAN, no GFPGAN). Strings `"true"`/`"false"` accepted. |

### GFPGAN model (used only when `face_enhance` is true)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `mode` | string | — | GFPGAN variant. Overrides `version` if set. |
| `version` | string | `"v1.4"` | Same as `mode`; used when `mode` is not set. |

**Allowed values (mode / version):**

- `"v1.2"`
- `"v1.3"`
- `"v1.4"`
- `"RestoreFormer"`
- `"CodeFormer"`
- `"RealESR-General-x4v3"`

### Background upsampler (RealESRGAN)

| Key | Type | Default | Description |
|-----|------|--------|-------------|
| `bg_model` | string | `"realesr-general-x4v3"` | RealESRGAN model for upscaling (and for non-face regions when face_enhance is true). |

**Allowed values:**

- `"realesr-general-x4v3"`
- `"RealESRGAN_x2plus"`
- `"RealESRGAN_x4plus_anime_6B"`

---

## Example payloads

### 1) Face enhancement (GFPGAN + RealESRGAN), R2 image

```json
{
  "model": "face_enhancer",
  "prompt": "",
  "params": {
    "version": "v1.4",
    "scale": 2,
    "face_enhance": true,
    "bg_model": "realesr-general-x4v3",
    "output_dir": "./outputs"
  },
  "input_image_r2_keys": ["image-vide-generator-api/input-media/your-file-id.bin"],
  "fcm_token": "your-fcm-token",
  "enhance_prompt": false,
  "variations": 1
}
```

### 2) General upscale only (no face enhancement)

```json
{
  "model": "face_enhancer",
  "prompt": "",
  "params": {
    "scale": 4,
    "face_enhance": false,
    "bg_model": "realesr-general-x4v3",
    "output_dir": "./outputs"
  },
  "input_image_r2_keys": ["image-vide-generator-api/input-media/your-file-id.bin"],
  "variations": 1
}
```

### 3) Face enhancement with base64 image (fallback)

```json
{
  "model": "face_enhancer",
  "prompt": "",
  "params": {
    "init_image": "data:image/png;base64,iVBORw0KGgo...",
    "version": "v1.4",
    "scale": 2,
    "face_enhance": true,
    "bg_model": "realesr-general-x4v3"
  },
  "variations": 1
}
```

### 4) Different GFPGAN mode via `mode`

```json
{
  "model": "face_enhancer",
  "prompt": "",
  "params": {
    "mode": "CodeFormer",
    "scale": 2,
    "face_enhance": true,
    "bg_model": "RealESRGAN_x2plus"
  },
  "input_image_r2_keys": ["image-vide-generator-api/input-media/your-file-id.bin"],
  "variations": 1
}
```

---

## Summary

- **Endpoint:** controller `POST /submit`.
- **Image:** either `input_image_r2_keys` (one key) or `params.init_image` / `params.image` (base64/URL).
- **Always in params:** `scale` (and `face_enhance` if you want general upscale).
- **Face enhancement:** `face_enhance: true` (default), and optionally `mode` or `version` for GFPGAN variant.
- **General upscale only:** `face_enhance: false`; only `bg_model` and `scale` are used.
- **Optional:** `bg_model`, `mode`/`version`, `output_dir`.
