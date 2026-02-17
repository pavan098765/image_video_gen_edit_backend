# Qwen Image Edit 2511 Rapid AIO - API Request Format

## Endpoint
`POST /run`

## Request Structure

```json
{
  "job_id": "unique-job-id-string",
  "prompt": "text prompt for image editing",
  "model": "qwen_image_edit_2511_rapid_aio",
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

- **`init_image`** (string, base64) OR **`init_image`** (array of base64 strings): 
  - Primary input image(s) as base64-encoded strings
  - **Single image**: `"init_image": "iVBORw0KGgoAAAANSUhEUgAA..."`
  - **Multiple images (array)**: `"init_image": ["base64_img1", "base64_img2", "base64_img3"]`
  - **Qwen Image Edit 2511 supports up to 3 images** in the array
  - **For 2-image LoRAs**: Use array format with at least 2 images
    - Example: `"init_image": ["base64_img1", "base64_img2"]` for AnyPose or BFS-Best-FaceSwap-merge
  - Images are automatically labeled as `image_1`, `image_2`, `image_3` based on array position

### Optional Parameters

#### LoRA Adapter Selection

- **`lora_adapter`** (string, optional, default: `"None"`):
  - LoRA adapter name from the list below
  - Use `"None"` to disable LoRA adapters
  - **Available LoRA Adapters:**
    - `"None"` - No LoRA (base model only)
    - `"Consistance"` - Consistency enhancement
    - `"Semirealistic-photo-detailer"` - Semi-realistic photo detailing
    - `"AnyPose"` - **2-image LoRA** - Pose transfer (requires `init_image_2`)
    - `"Any2Real_2601"` - Convert to realistic photograph
    - `"Hyperrealistic-Portrait"` - Hyper-realistic portrait enhancement
    - `"Ultrarealistic-Portrait"` - Ultra-realistic glamour portrait
    - `"BFS-Best-FaceSwap-merge"` - **2-image LoRA** - Face swap merged version (requires `init_image_2`)
    - `"F2P"` - F2P style transfer
    - `"Multiple-Angles"` - Camera angle changes
    - `"Light-Restoration"` - Remove shadows and relight
    - `"Relight"` - Relighting with golden-hour filter
    - `"Edit-Skin"` - Skin detail enhancement
    - `"Next-Scene"` - Scene transition effects
    - `"Flat-Log"` - Flat log color grading

#### AIO Version

- **`aio_version`** (string, optional, default: `"v19"`):
  - AIO transformer version to use
  - Available versions: `"v14.1"`, `"v18.1"`, `"v19"`, `"v20"`, `"v21"`, `"v22"`, `"v23"`
  - Can also use numeric format: `"19"` will be normalized to `"v19"`
  - Example: `"aio_version": "v21"`

#### Inference Parameters

- **`num_inference_steps`** (integer, optional, default: `4`):
  - Number of inference steps
  - Higher = better quality but slower
  - Range: 1-50
  - Example: `"num_inference_steps": 8`

- **`guidance_scale`** (float, optional, default: `1.0`):
  - Guidance scale (CFG scale)
  - Higher = more adherence to prompt
  - Range: 1.0-10.0
  - Example: `"guidance_scale": 1.5`

- **`seed`** (integer, optional):
  - Random seed for reproducibility
  - If not provided and `randomize_seed` is true, uses random seed
  - Example: `"seed": 42`

- **`randomize_seed`** (boolean, optional, default: `true`):
  - Whether to randomize the seed
  - If `false` and `seed` is provided, uses the provided seed
  - If `true`, ignores `seed` and uses random
  - Example: `"randomize_seed": false`

- **`negative_prompt`** (string, optional):
  - Negative prompt to avoid certain features
  - Default: `"worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"`
  - Example: `"negative_prompt": "blurry, low quality, distorted"`

#### Output Configuration

- **`output_dir`** (string, optional, default: `"/app/worker/outputs"`):
  - Output directory for saved images
  - Usually not needed (uses default)
  - Example: `"output_dir": "/app/worker/outputs"`

## Example Requests

### 1. Basic Single Image Edit (No LoRA)

```json
{
  "job_id": "edit-001",
  "prompt": "Change her body orientation so she is facing the camera directly",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "lora_adapter": "None",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "randomize_seed": true
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-001.png"
}
```

### 2. Single Image with LoRA Adapter

```json
{
  "job_id": "edit-002",
  "prompt": "Transform the image into an ultra-realistic photorealistic portrait",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "lora_adapter": "Hyperrealistic-Portrait",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42,
    "randomize_seed": false
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-002.png"
}
```

### 3. Two-Image LoRA: AnyPose (Pose Transfer)

```json
{
  "job_id": "edit-003",
  "prompt": "Make the person in image 1 do the exact same pose of the person in image 2",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": ["iVBORw0KGgoAAAANSUhEUgAA...", "iVBORw0KGgoAAAANSUhEUgAA..."],
    "lora_adapter": "AnyPose",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "randomize_seed": true
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-003.png"
}
```

### 4. Two-Image LoRA: BFS-Best-FaceSwap-merge

```json
{
  "job_id": "edit-004",
  "prompt": "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": ["iVBORw0KGgoAAAANSUhEUgAA...", "iVBORw0KGgoAAAANSUhEUgAA..."],
    "lora_adapter": "BFS-Best-FaceSwap-merge",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "randomize_seed": true
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-004.png"
}
```

### 5. Three Images (Maximum Supported)

```json
{
  "job_id": "edit-006",
  "prompt": "Combine elements from all three images",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": ["iVBORw0KGgoAAAANSUhEUgAA...", "iVBORw0KGgoAAAANSUhEUgAA...", "iVBORw0KGgoAAAANSUhEUgAA..."],
    "lora_adapter": "None",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "randomize_seed": true
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-006.png"
}
```

### 6. Custom AIO Version with Advanced Settings

```json
{
  "job_id": "edit-007",
  "prompt": "Transform into realistic photograph",
  "model": "qwen_image_edit_2511_rapid_aio",
  "mode": null,
  "params": {
    "init_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "lora_adapter": "Any2Real_2601",
    "aio_version": "v21",
    "num_inference_steps": 12,
    "guidance_scale": 1.5,
    "seed": 12345,
    "randomize_seed": false,
    "negative_prompt": "cartoon, anime, illustration, drawing"
  },
  "upload_url": "https://r2.example.com/upload",
  "result_key": "edit-007.png"
}
```

## Important Notes

### 2-Image LoRAs
The following LoRAs **REQUIRE** `init_image` to be an **array with at least 2 images**:
- `"AnyPose"` - Pose reference image (image_1 = base, image_2 = pose reference)
- `"BFS-Best-FaceSwap-merge"` - Face swap merged version (image_1 = base, image_2 = face/head donor)

**Example for 2-image LoRA:**
```json
"init_image": ["base64_image_1", "base64_image_2"]
```

If you use these LoRAs with a single image (string instead of array), the request will fail with an error.

### Image Format
- All images must be base64-encoded strings
- `init_image` can be:
  - **Single image**: base64 string → `"init_image": "base64_string"`
  - **Multiple images**: array of base64 strings → `"init_image": ["base64_1", "base64_2", "base64_3"]`
- **Qwen Image Edit 2511 supports up to 3 images** in the array
- Images are automatically converted to RGB format
- Images are labeled as `image_1`, `image_2`, `image_3` based on array position
- Resolution is automatically computed based on the LoRA adapter (most use 1024px)

### LoRA Adapter Names
- Use **exact** names as listed above (case-sensitive)
- Use `"None"` (capital N) to disable LoRA adapters
- Invalid adapter names will cause an error

### AIO Version
- Default is `"v19"` if not specified
- You can use numeric format (`"19"`) or version format (`"v19"`)
- Available versions: v14.1, v18.1, v19, v20, v21, v22, v23
