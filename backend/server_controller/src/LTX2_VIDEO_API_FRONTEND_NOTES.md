# LTX-2 Video API - Frontend Implementation Notes

## Overview

LTX-2 Video Generation supports:
- **Text-to-Video (T2V)**: Generate video from text prompts
- **Image-to-Video (I2V)**: Animate images with text guidance
- **Video-to-Video (V2V)**: Transform existing videos
- **Keyframe Interpolation**: Generate smooth transitions between keyframes

## API Endpoint

Use the existing `/submit` endpoint with `model: "ltx2_video"`:

```
POST /submit
Authorization: Bearer <firebase_id_token>
Content-Type: application/json
```

## Request Schema

### Single Video Request

```json
{
  "model": "ltx2_video",
  "prompt": "A cinematic drone shot of mountains at sunrise, smooth camera movement",
  "params": {
    "mode": "text-to-video",
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "fps": 25,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "pipeline_type": "distilled",
    "seed": 42
  },
  "enhance_prompt": false,
  "variations": 1
}
```

### Bulk Video Request (VIP Only)

```json
{
  "model": "ltx2_video",
  "prompt": "Animate this scene with gentle motion",
  "params": {
    "mode": "image-to-video",
    "duration": 5.0,
    "conditioning_strength": 1.0
  },
  "is_bulk": true,
  "bulk_images": [
    "<base64_encoded_image_1>",
    "<base64_encoded_image_2>",
    "<base64_encoded_image_3>"
  ],
  "variations": 1
}
```

## Generation Modes

### 1. Text-to-Video (T2V)

Generate video purely from text description.

```json
{
  "model": "ltx2_video",
  "prompt": "A butterfly emerging from a cocoon, macro photography, cinematic lighting",
  "params": {
    "mode": "text-to-video",
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "fps": 25,
    "pipeline_type": "distilled",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": -1,
    "negative_prompt": "blurry, jittery, distorted, low quality"
  }
}
```

### 2. Image-to-Video (I2V)

Animate an input image with text guidance.

```json
{
  "model": "ltx2_video",
  "prompt": "Gentle wind blowing through the scene, subtle motion",
  "params": {
    "mode": "image-to-video",
    "init_image": "<base64_encoded_image>",
    "height": 512,
    "width": 768,
    "duration": 5.0,
    "fps": 25,
    "conditioning_strength": 1.0,
    "pipeline_type": "two_stage",
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "stg_scale": 1.0
  }
}
```

### 3. Video-to-Video (V2V)

Transform an existing video with text guidance.

```json
{
  "model": "ltx2_video",
  "prompt": "Transform into anime style, vibrant colors",
  "params": {
    "mode": "video-to-video",
    "init_video": "<base64_encoded_video>",
    "conditioning_strength": 0.8,
    "fps": 25,
    "num_inference_steps": 40,
    "guidance_scale": 3.0
  }
}
```

### 4. Keyframe Interpolation

Generate video interpolating between keyframe images.

```json
{
  "model": "ltx2_video",
  "prompt": "Smooth transition between scenes",
  "params": {
    "mode": "keyframe-interpolation",
    "keyframes": [
      {
        "image": "<base64_start_frame>",
        "frame_index": 0,
        "strength": 1.0
      },
      {
        "image": "<base64_end_frame>",
        "frame_index": 120,
        "strength": 1.0
      }
    ],
    "num_frames": 121,
    "fps": 25,
    "num_inference_steps": 40
  }
}
```

## Parameters Reference

### Common Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mode` | string | `"text-to-video"` | See modes | Generation mode |
| `height` | int | 512 | 256-1024 | Output height (must be divisible by 32) |
| `width` | int | 768 | 256-1024 | Output width (must be divisible by 32) |
| `duration` | float | 5.0 | 1.0-10.0 | Video duration in seconds |
| `fps` | int | 25 | 8-30 | Frames per second |
| `seed` | int | -1 | -1 to 2^31 | Random seed (-1 for random) |
| `negative_prompt` | string | "" | - | What to avoid in generation |

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline_type` | string | `"distilled"` | Pipeline type (see table below) |
| `num_inference_steps` | int | 8 | Denoising steps (8 for distilled, 40 for others) |
| `guidance_scale` | float | 1.0 | CFG scale (1.0 for distilled, 3.0 for others) |
| `stg_scale` | float | 0.0 | Spatio-temporal guidance (0.0 for distilled, 1.0 for others) |

### Pipeline Types

| Pipeline | Steps | Speed | Quality | Use Case |
|----------|-------|-------|---------|----------|
| `distilled` | 8+4 | **Fastest** | Good | Real-time, preview, mobile |
| `two_stage` | 40+20 | Slower | **Best** | Production, high quality |
| `ic_lora` | 40 | Medium | Good | Video-to-video |
| `keyframe` | 40 | Medium | Good | Keyframe interpolation |

### Mode-Specific Parameters

#### Image-to-Video
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_image` | string | - | Base64-encoded input image |
| `conditioning_strength` | float | 1.0 | How much to follow the input image (0.0-1.0) |

#### Video-to-Video
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_video` | string | - | Base64-encoded input video |
| `conditioning_strength` | float | 0.8 | How much to follow the input video (0.0-1.0) |

#### Keyframe Interpolation
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyframes` | array | - | Array of keyframe objects |
| `keyframes[].image` | string | - | Base64-encoded keyframe image |
| `keyframes[].frame_index` | int | - | Frame position (0 = start, num_frames-1 = end) |
| `keyframes[].strength` | float | 1.0 | Conditioning strength for this keyframe |
| `num_frames` | int | 121 | Total frames (must satisfy (n-1) % 8 == 0) |

## Response Format

### Single Job Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "model": "ltx2_video",
  "credit_cost": 30,
  "credits_remaining": 970
}
```

### Batch Response

```json
{
  "batch_id": "batch-550e8400-e29b-41d4-a716-446655440000",
  "job_ids": [
    "job-1-uuid",
    "job-2-uuid",
    "job-3-uuid"
  ],
  "status": "queued",
  "total_jobs": 3,
  "credit_cost": 90
}
```

### Job Status Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result_url": "https://genedit.mooo.com/download_url/550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-02-02T12:00:00Z",
  "completed_at": "2026-02-02T12:01:30Z"
}
```

## Credit Costs

| Mode | Base Cost | Notes |
|------|-----------|-------|
| Text-to-Video | 30 | Default for 5s video |
| Image-to-Video | 30 | Same as T2V |
| Video-to-Video | 40 | Higher due to processing |
| Keyframe Interpolation | 35 | Between T2V and V2V |

**Note**: Costs are configured server-side in the tool_config. VIP users may get discounts.

## Best Practices

### 1. Prompt Writing

**Good prompts**:
- Keep under 200 words
- Be specific about motion/action
- Describe camera movement if needed
- Include style/mood descriptors

```
"A cinematic drone shot slowly ascending over misty mountains at golden hour. 
The camera moves smoothly upward, revealing snow-capped peaks emerging through 
layers of clouds. Warm sunlight filters through, casting long shadows."
```

**Avoid**:
- Very long prompts (>200 words)
- Conflicting instructions
- Static scene descriptions for T2V

### 2. Resolution Recommendations

| Device | Resolution | Duration | Pipeline |
|--------|------------|----------|----------|
| Mobile | 512x768 | 3-5s | distilled |
| Tablet | 768x768 | 5s | distilled |
| Web | 768x1024 | 5s | two_stage |
| Production | 1024x1024 | 5s | two_stage |

### 3. Seed Usage

```javascript
// For reproducibility
const params = {
  seed: 12345,  // Fixed seed = same output
  // ... other params
};

// For variety
const params = {
  seed: -1,  // Random seed each time
  // ... other params
};
```

### 4. Error Handling

```javascript
async function generateVideo(payload) {
  try {
    const response = await fetch('/submit', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${firebaseToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    
    if (response.status === 402) {
      // Insufficient credits
      showCreditPurchaseDialog();
      return;
    }
    
    if (response.status === 403) {
      // VIP required for bulk
      showVipUpgradeDialog();
      return;
    }
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Generation failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Video generation failed:', error);
    throw error;
  }
}
```

### 5. Polling for Status

```javascript
async function pollJobStatus(jobId, interval = 3000, maxAttempts = 120) {
  for (let i = 0; i < maxAttempts; i++) {
    const response = await fetch(`/job_status?job_id=${jobId}`, {
      headers: { 'Authorization': `Bearer ${firebaseToken}` }
    });
    
    const status = await response.json();
    
    if (status.status === 'completed') {
      return status.result_url;
    }
    
    if (status.status === 'failed') {
      throw new Error(status.error || 'Job failed');
    }
    
    // Still processing
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  throw new Error('Job timed out');
}
```

## TypeScript Interfaces

```typescript
// Mode types
type VideoMode = 
  | 'text-to-video' 
  | 'image-to-video' 
  | 'video-to-video' 
  | 'keyframe-interpolation';

// Pipeline types
type PipelineType = 'distilled' | 'two_stage' | 'ic_lora' | 'keyframe';

// Keyframe object
interface Keyframe {
  image: string;        // Base64-encoded image
  frame_index: number;  // 0 = start, num_frames-1 = end
  strength?: number;    // 0.0-1.0, default 1.0
}

// Video generation params
interface LTX2VideoParams {
  mode: VideoMode;
  height?: number;           // 256-1024, default 512
  width?: number;            // 256-1024, default 768
  duration?: number;         // 1.0-10.0 seconds, default 5.0
  fps?: number;              // 8-30, default 25
  
  // Input media
  init_image?: string;       // Base64 for I2V
  init_video?: string;       // Base64 for V2V
  keyframes?: Keyframe[];    // For keyframe interpolation
  
  // Generation
  pipeline_type?: PipelineType;     // default 'distilled'
  num_inference_steps?: number;     // default 8 (distilled) or 40
  guidance_scale?: number;          // default 1.0 (distilled) or 3.0
  stg_scale?: number;               // default 0.0 (distilled) or 1.0
  conditioning_strength?: number;   // 0.0-1.0, default 1.0
  
  // Control
  seed?: number;                    // -1 for random
  negative_prompt?: string;
  num_frames?: number;              // For keyframe mode
}

// Submit request
interface VideoSubmitRequest {
  model: 'ltx2_video';
  prompt: string;
  params: LTX2VideoParams;
  enhance_prompt?: boolean;
  variations?: number;           // 1-4
  is_bulk?: boolean;
  bulk_images?: string[];        // Max 10
}

// Submit response
interface VideoSubmitResponse {
  job_id: string;
  batch_id?: string;
  job_ids?: string[];
  status: string;
  credit_cost: number;
  credits_remaining: number;
}

// Job status
interface VideoJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  result_url?: string;
  error?: string;
  created_at: string;
  completed_at?: string;
}
```

## Example Implementations

### React Hook

```typescript
import { useState, useCallback } from 'react';

export function useVideoGeneration() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState<string>('');
  
  const generateVideo = useCallback(async (
    prompt: string,
    mode: VideoMode,
    options?: Partial<LTX2VideoParams>
  ) => {
    setIsGenerating(true);
    setProgress('Submitting...');
    
    try {
      // Submit job
      const submitResponse = await submitVideoJob({
        model: 'ltx2_video',
        prompt,
        params: {
          mode,
          pipeline_type: 'distilled',  // Fast by default
          ...options,
        },
      });
      
      setProgress('Generating...');
      
      // Poll for completion
      const resultUrl = await pollJobStatus(submitResponse.job_id);
      
      setProgress('Complete!');
      return resultUrl;
      
    } catch (error) {
      setProgress(`Error: ${error.message}`);
      throw error;
    } finally {
      setIsGenerating(false);
    }
  }, []);
  
  return { generateVideo, isGenerating, progress };
}
```

### Flutter Example

```dart
class VideoGenerationService {
  final String baseUrl;
  final String authToken;
  
  Future<String> generateTextToVideo({
    required String prompt,
    int width = 768,
    int height = 512,
    double duration = 5.0,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/submit'),
      headers: {
        'Authorization': 'Bearer $authToken',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'model': 'ltx2_video',
        'prompt': prompt,
        'params': {
          'mode': 'text-to-video',
          'width': width,
          'height': height,
          'duration': duration,
          'pipeline_type': 'distilled',
        },
      }),
    );
    
    if (response.statusCode != 200) {
      throw Exception('Failed to submit job');
    }
    
    final jobId = jsonDecode(response.body)['job_id'];
    return await _pollForResult(jobId);
  }
  
  Future<String> _pollForResult(String jobId) async {
    while (true) {
      await Future.delayed(Duration(seconds: 3));
      
      final response = await http.get(
        Uri.parse('$baseUrl/job_status?job_id=$jobId'),
        headers: {'Authorization': 'Bearer $authToken'},
      );
      
      final status = jsonDecode(response.body);
      
      if (status['status'] == 'completed') {
        return status['result_url'];
      }
      
      if (status['status'] == 'failed') {
        throw Exception(status['error'] ?? 'Generation failed');
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| 402 Insufficient credits | Not enough credits | Purchase credits |
| 403 VIP required | Bulk edit without VIP | Upgrade to VIP or use single mode |
| Timeout | Video generation takes too long | Reduce duration, use distilled pipeline |
| OOM error | Video too large | Reduce resolution or duration |
| Blurry output | Low quality settings | Use two_stage pipeline, higher steps |

### Recommended Settings by Use Case

| Use Case | Pipeline | Steps | Resolution | Duration |
|----------|----------|-------|------------|----------|
| Quick preview | distilled | 8 | 512x768 | 3s |
| Social media | distilled | 8 | 768x768 | 5s |
| Marketing | two_stage | 40 | 768x1024 | 5s |
| Film/Production | two_stage | 40 | 1024x1024 | 5-10s |

## Changelog

- **v2.0.0** (2026-02-02): Initial LTX-2 integration
  - Text-to-Video support
  - Image-to-Video support
  - Video-to-Video support
  - Keyframe interpolation
  - Bulk video generation (VIP)
  - FP4 model for 4090/5090 compatibility
