# Face Enhancer (GFPGAN) - API Request Format

## Endpoint
`POST /run`

## Request Structure

```json
{
  "job_id": "unique-job-id-string",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    // See params section below
  },
  "upload_url": "https://r2-bucket-url/upload-endpoint",
  "result_key": "output-filename.png"
}
```

## Parameters (`params` object)

### Required Parameters

- **`init_image`** (string, base64): 
  - Input image as base64-encoded string
  - Example: `"init_image": "iVBORw0KGgoAAAANSUhEUgAA..."`

### Optional Parameters

#### GFPGAN Version Selection

- **`version`** (string, optional, default: `"v1.4"`):
  - GFPGAN version/model to use for face enhancement
  - **Available versions:**
    - `"v1.2"` - GFPGAN v1.2
    - `"v1.3"` - GFPGAN v1.3
    - `"v1.4"` - GFPGAN v1.4 (default, recommended)
    - `"RestoreFormer"` - RestoreFormer architecture
    - `"CodeFormer"` - CodeFormer architecture
    - `"RealESR-General-x4v3"` - RealESRGAN General x4v3 architecture
  - Example: `"version": "v1.4"`

#### Output Configuration

- **`scale`** (float, optional, default: `2.0`):
  - Final rescaling factor for the output image
  - GFPGAN internally upscales by 2x, then this scale is applied
  - Example: `scale: 2.0` = 2x final output (4x total from original)
  - Example: `scale: 4.0` = 4x final output (8x total from original)
  - Range: Any positive float
  - Example: `"scale": 2.0`

- **`output_dir`** (string, optional, default: `"/app/worker/outputs"`):
  - Output directory for saved images
  - Usually not needed (uses default)
  - Example: `"output_dir": "/app/worker/outputs"`

## Example Requests

### 1. Basic Face Enhancement (Default v1.4)

```json
{
  "job_id": "face-001",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "v1.4",
    "scale": 2.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-001.png"
}
```

### 2. Face Enhancement with v1.3

```json
{
  "job_id": "face-002",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "v1.3",
    "scale": 2.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-002.png"
}
```

### 3. RestoreFormer Architecture

```json
{
  "job_id": "face-003",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "RestoreFormer",
    "scale": 2.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-003.png"
}
```

### 4. CodeFormer Architecture

```json
{
  "job_id": "face-004",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "CodeFormer",
    "scale": 2.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-004.png"
}
```

### 5. RealESR-General-x4v3 Architecture

```json
{
  "job_id": "face-005",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "RealESR-General-x4v3",
    "scale": 2.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-005.png"
}
```

### 6. Custom Scale (4x Final Output)

```json
{
  "job_id": "face-006",
  "prompt": null,
  "model": "face_enhancer",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "version": "v1.4",
    "scale": 4.0
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "face-006.png"
}
```

## Important Notes

### Image Format
- All images must be base64-encoded strings
- Images are automatically converted to RGB format
- **Automatic preprocessing:**
  - Images with height < 300px are automatically resized to 2x (using LANCZOS4 interpolation)
  - RGBA images are handled (alpha channel preserved)
  - Grayscale images are converted to RGB

### GFPGAN Versions
- **v1.4** (default) - Recommended for most use cases
- **v1.2, v1.3** - Older versions, may have different quality characteristics
- **RestoreFormer** - Uses RestoreFormer architecture
- **CodeFormer** - Uses CodeFormer architecture
- **RealESR-General-x4v3** - Uses RealESRGAN General architecture

### Scale Parameter
- GFPGAN internally upscales by **2x** first
- Then the `scale` parameter is applied as a final rescaling
- **Total upscale = 2 × scale**
  - `scale: 2.0` → 4x total upscale
  - `scale: 4.0` → 8x total upscale
  - `scale: 1.0` → 2x total upscale
- Uses `INTER_AREA` interpolation if scale < 2, `INTER_LANCZOS4` if scale >= 2

### Background Upsampling
- All versions use **RealESRGAN realesr-general-x4v3** as the background upsampler
- This ensures high-quality upscaling of the entire image, not just faces

### Output Format
- Output images are saved as PNG format
- All metadata (EXIF/ICC) is stripped for clean output
- Images are converted from BGR (OpenCV) to RGB (PIL) for correct color display

## Response Format

The endpoint returns:

```json
{
  "status": "accepted",
  "job_id": "face-001",
  "queue_position": 0,
  "worker_id": "worker-uuid"
}
```

The actual result will be uploaded to the provided `upload_url` and the controller will be notified with `result_key`.
