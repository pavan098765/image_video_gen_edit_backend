---
title: FLUX.1 [dev] | Runware Docs
url: https://runware.ai/docs/models/flux-1-dev
description: High-Fidelity Guidance-Distilled Architecture
---
# FLUX.1 [dev]

![FLUX.1 [dev]](https://assets.runware.ai/covers/flux-1-dev.jpg)

FLUX.1 [dev] is a 12-billion-parameter, guidance-distilled image-generation architecture from Black Forest Labs. It combines strong prompt adherence, detailed visual synthesis, and broad stylistic range, serving as a foundation for compatible checkpoints, fine-tunes, and adapters.

- **Status**: live
- Base Architecture
- Try in Playground
- **Creator**: Black Forest Labs
- **Release Date**: August 1, 2024
- **Capabilities**: Text to Image, Image to Image

## Compatibility & Validation

When `strength` or `inputs.maskImage` is provided, `inputs.seedImage` is required.

---

`width` and `height` must be used together.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/flux-1-dev#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/flux-1-dev#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/flux-1-dev#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/flux-1-dev#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/flux-1-dev#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/flux-1-dev#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/flux-1-dev#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/flux-1-dev#request-uploadendpoint)

- **Type**: `string`
- **Format**: `uri`

Specifies a URL where the generated content will be automatically uploaded using the HTTP PUT method. The raw binary data of the media file is sent directly as the request body. For secure uploads to cloud storage, use presigned URLs that include temporary authentication credentials.

**Common use cases:**

- **Cloud storage**: Upload directly to S3 buckets, Google Cloud Storage, or Azure Blob Storage using presigned URLs.
- **CDN integration**: Upload to content delivery networks for immediate distribution.

```text
// S3 presigned URL for secure upload
https://your-bucket.s3.amazonaws.com/generated/content.mp4?X-Amz-Signature=abc123&X-Amz-Expires=3600

// Google Cloud Storage presigned URL
https://storage.googleapis.com/your-bucket/content.jpg?X-Goog-Signature=xyz789

// Custom storage endpoint
https://storage.example.com/uploads/generated-image.jpg
```

The content data will be sent as the request body to the specified URL when generation is complete.

### [safety](https://runware.ai/docs/models/flux-1-dev#request-safety)

- **Path**: `safety.checkContent`
- **Type**: `object (1 property)`

Content safety checking configuration for image generation.

#### [checkContent](https://runware.ai/docs/models/flux-1-dev#request-safety-checkcontent)

- **Path**: `safety.checkContent`
- **Type**: `boolean`

Enable or disable content safety checking. Increases total generation time.

### [ttl](https://runware.ai/docs/models/flux-1-dev#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/flux-1-dev#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/flux-1-dev#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `20`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [acceleratorOptions](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions)

- **Path**: `acceleratorOptions.cacheEndStep`
- **Type**: `object (10 properties)`

Advanced caching mechanisms to speed up generation.

#### [cacheEndStep](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-cacheendstep)

- **Path**: `acceleratorOptions.cacheEndStep`
- **Type**: `integer`
- **Min**: `1`

Absolute step number to end caching. Must be greater than `cacheStartStep` and less than or equal to `steps`.

#### [cacheEndStepPercentage](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-cacheendsteppercentage)

- **Path**: `acceleratorOptions.cacheEndStepPercentage`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`

Percentage of steps to end caching. Alternative to `cacheEndStep`. Must be greater than `cacheStartStepPercentage`.

#### [cacheMaxConsecutiveSteps](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-cachemaxconsecutivesteps)

- **Path**: `acceleratorOptions.cacheMaxConsecutiveSteps`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `5`
- **Default**: `3`

Limits the maximum number of consecutive steps that can use cached computations before forcing a fresh computation.

#### [cacheStartStep](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-cachestartstep)

- **Path**: `acceleratorOptions.cacheStartStep`
- **Type**: `integer`
- **Min**: `0`

Absolute step number to start caching. Must be less than `cacheEndStep`.

#### [cacheStartStepPercentage](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-cachestartsteppercentage)

- **Path**: `acceleratorOptions.cacheStartStepPercentage`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `99`

Percentage of steps to start caching. Alternative to `cacheStartStep`. Must be less than `cacheEndStepPercentage`.

#### [teaCache](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-teacache)

- **Path**: `acceleratorOptions.teaCache`
- **Type**: `boolean`

TeaCache acceleration for transformer-based models. Estimates step differences to skip redundant computations.

#### [teaCacheDistance](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-teacachedistance)

- **Path**: `acceleratorOptions.teaCacheDistance`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.5`

Controls the aggressiveness of the TeaCache feature. Lower values prioritize quality, higher values prioritize speed.

#### [dbCache](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-dbcache)

- **Path**: `acceleratorOptions.dbCache`
- **Type**: `boolean`

DB Cache (CacheDiT) acceleration. Caches and reuses intermediate transformer block outputs to skip redundant computations.

#### [dbCacheThreshold](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-dbcachethreshold)

- **Path**: `acceleratorOptions.dbCacheThreshold`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.25`

Controls the sensitivity threshold for DB Cache. Lower values reuse cached blocks more aggressively, higher values prioritize quality.

#### [dbCacheSkipInterval](https://runware.ai/docs/models/flux-1-dev#request-acceleratoroptions-dbcacheskipinterval)

- **Path**: `acceleratorOptions.dbCacheSkipInterval`
- **Type**: `integer`
- **Min**: `1`
- **Default**: `5`

Controls how many steps to skip between cache refreshes. Higher values skip more steps for faster generation at the cost of quality.

### [routing](https://runware.ai/docs/models/flux-1-dev#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/flux-1-dev#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/flux-1-dev#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/flux-1-dev#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/flux-1-dev#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/flux-1-dev#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/flux-1-dev#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words, such as a style or a feature. A model that cannot meet them is not chosen. Output size is set with width and height rather than described here.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [seedImage](https://runware.ai/docs/models/flux-1-dev#request-inputs-seedimage)

- **Path**: `inputs.seedImage`
- **Type**: `string`

Image used as a starting point for the generation (UUID, URL, Data URI, or Base64).

**Learn more** (3 resources):

- [Image To Image: Seed Image The Foundation](https://runware.ai/docs/learn/image-to-image#seed-image-the-foundation) (learn)
- [Image Inpainting: Seed And Mask Image The Foundation](https://runware.ai/docs/learn/image-inpainting#seed-and-mask-image-the-foundation) (learn)
- [Image Outpainting: Seed Image The Starting Point](https://runware.ai/docs/learn/image-outpainting#seed-image-the-starting-point) (learn)

### [maskImage](https://runware.ai/docs/models/flux-1-dev#request-inputs-maskimage)

- **Path**: `inputs.maskImage`
- **Type**: `string`

Image used to specify which areas of the input image should be edited (UUID, URL, Data URI, or Base64). White marks the area to edit, black is preserved.

**Learn more** (1 resource):

- [Image Inpainting: Seed And Mask Image The Foundation](https://runware.ai/docs/learn/image-inpainting#seed-and-mask-image-the-foundation) (learn)

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/flux-1-dev#request-model)

- **Type**: `string`
- **Required**: true

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/flux-1-dev#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `2`
- **Max**: `3000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [negativePrompt](https://runware.ai/docs/models/flux-1-dev#request-negativeprompt)

- **Type**: `string`
- **Min**: `2`
- **Max**: `3000`

Prompt to guide what to exclude from generation. Ignored when guidance is disabled (CFGScale ≤ 1).

**Learn more** (1 resource):

- [Prompts: Negative Prompts](https://runware.ai/docs/learn/prompts#negative-prompts) (learn)

### [width](https://runware.ai/docs/models/flux-1-dev#request-width)

- **Type**: `integer`
- **Required**: true
- **Min**: `128`
- **Max**: `2048`
- **Step**: `64`

Width of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [height](https://runware.ai/docs/models/flux-1-dev#request-height)

- **Type**: `integer`
- **Required**: true
- **Min**: `128`
- **Max**: `2048`
- **Step**: `64`

Height of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [seed](https://runware.ai/docs/models/flux-1-dev#request-seed)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `9223372036854776000`

Random seed for reproducible generation. When not provided, a random seed is generated in the unsigned 32-bit range.

**Learn more** (1 resource):

- [Seed](https://runware.ai/docs/learn/seed) (learn)

### [steps](https://runware.ai/docs/models/flux-1-dev#request-steps)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`

Total number of denoising steps. Higher values generally produce more detailed results but take longer.

**Learn more** (1 resource):

- [Steps](https://runware.ai/docs/learn/steps) (learn)

### [scheduler](https://runware.ai/docs/models/flux-1-dev#request-scheduler)

- **Type**: `string`

Scheduler to use for the diffusion process.

**Allowed values**: `DDIM` `DDIMScheduler` `DDPMScheduler` `DEISMultistepScheduler` `Default` `DPM++` `DPM++ 2M` `DPM++ 2M Beta` `DPM++ 2M Exponential` `DPM++ 2M Karras` `DPM++ 2M SDE` `DPM++ 2M SDE Beta` `DPM++ 2M SDE Exponential` `DPM++ 2M SDE Karras` `DPM++ 2M SDE Uniform` `DPM++ 2M Uniform` `DPM++ 3M` `DPM++ 3M Beta` `DPM++ 3M Exponential` `DPM++ 3M Karras` `DPM++ 3M SDE Uniform` `DPM++ 3M Uniform` `DPM++ Beta` `DPM++ Exponential` `DPM++ Karras` `DPM++ SDE` `DPM++ SDE Beta` `DPM++ SDE Exponential` `DPM++ SDE Karras` `DPM++ Uniform` `DPM++ Uniform Beta` `DPM++ Uniform Exponential` `DPM++ Uniform Karras` `DPMSolverMultistepInverse` `DPMSolverMultistepScheduler` `DPMSolverSinglestepScheduler` `EDMDPMSolverMultistepScheduler` `EDMEulerScheduler` `Euler` `Euler a` `Euler Beta` `Euler Exponential` `Euler Karras` `EulerAncestralDiscreteScheduler` `EulerDiscreteScheduler` `FlowMatchEulerDiscreteScheduler` `Heun` `HeunDiscreteScheduler` `Heun Karras` `IPNDMScheduler` `IPNDM Uniform` `IPNDM Uniform Beta` `IPNDM Uniform Exponential` `IPNDM Uniform Karras` `KDPM2AncestralDiscreteScheduler` `KDPM2DiscreteScheduler` `LCM` `LCMScheduler` `LMS` `LMSDiscreteScheduler` `LMS Karras` `PNDMScheduler` `TCDScheduler` `UniPC` `UniPC 2M` `UniPC 2M Karras` `UniPC 2M Uniform` `UniPC 3M` `UniPC 3M Karras` `UniPC 3M Uniform` `UniPC Karras` `UniPC Uniform` `UniPC Uniform Beta` `UniPC Uniform Exponential` `UniPC Uniform Karras`

**Learn more** (2 resources):

- [Schedulers](https://runware.ai/docs/learn/schedulers) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [CFGScale](https://runware.ai/docs/models/flux-1-dev#request-cfgscale)

- **Type**: `float`
- **Min**: `0`
- **Max**: `50`
- **Step**: `0.01`

Guidance scale representing how closely the output will resemble the prompt. Higher values produce results more aligned with the prompt.

**Learn more** (1 resource):

- [Cfg Scale](https://runware.ai/docs/learn/cfg-scale) (learn)

### [strength](https://runware.ai/docs/models/flux-1-dev#request-strength)

- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.8`

Strength of the transformation. Lower values result in more influence from the original input.

**Learn more** (3 resources):

- [Image To Image: Strength The Transformation Intensity](https://runware.ai/docs/learn/image-to-image#strength-the-transformation-intensity) (learn)
- [Image Inpainting: Strength Controlling Transformation Intensity](https://runware.ai/docs/learn/image-inpainting#strength-controlling-transformation-intensity) (learn)
- [Image Outpainting: Strength Understanding The Critical Threshold](https://runware.ai/docs/learn/image-outpainting#strength-understanding-the-critical-threshold) (learn)

### [maskMargin](https://runware.ai/docs/models/flux-1-dev#request-maskmargin)

- **Type**: `integer`
- **Min**: `32`
- **Max**: `128`

Extra context pixels around the masked region during inpainting. The model zooms into the masked area with these additional pixels for better integration.

**Learn more** (1 resource):

- [Image Inpainting: Mask Margin Enhancing Detail](https://runware.ai/docs/learn/image-inpainting#mask-margin-enhancing-detail) (learn)

### [promptWeighting](https://runware.ai/docs/models/flux-1-dev#request-promptweighting)

- **Type**: `string`

Defines the syntax to be used for prompt weighting.

Prompt weighting allows you to adjust how strongly different parts of your prompt influence the generated image. Choose between `compel` notation with advanced weighting operations or `sdEmbeds` for simple emphasis adjustments.

**Compel syntax**:

Adds 0.2 seconds to image inference time and incurs additional costs.

When `compel` syntax is selected, you can use the following notation in prompts:

**Weighting**

Syntax: `+` `-` `(word)0.9`

Increase or decrease the attention given to specific words or phrases.

Examples:

- Single words: `small+ dog, pixar style`
- Multiple words: `small dog, (pixar style)-`
- Multiple symbols for more effect: `small+++ dog, pixar style`
- Nested weighting: `(small+ dog)++, pixar style`
- Explicit weight percentage: `small dog, (pixar)1.2 style`

**Blend**

Syntax: `.blend()`

Merge multiple conditioning prompts.

Example: `("small dog", "robot").blend(1, 0.8)`

**Conjunction**

Syntax: `.and()`

Break a prompt into multiple clauses and pass them separately.

Example: `("small dog", "pixar style").and()`

**SdEmbeds syntax**:

When `sdEmbeds` syntax is selected, you can use the following notation in prompts:

**Weighting**

Syntax: `(text)` `(text:number)` `[text]`

Use parentheses `()` to increase attention, square brackets `[]` to decrease it. Add a number after the text to specify a custom multiplier.

Examples:

- Single words: `(small) dog, pixar style`
- Multiple words: `small dog, [pixar style]`
- Higher emphasis: `(small:2.5) dog, pixar style`
- Combined emphasis: `(small dog:1.5), pixar style`

**Allowed values**: `sdEmbeds`

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [outpaint](https://runware.ai/docs/models/flux-1-dev#request-outpaint)

- **Path**: `outpaint.top`
- **Type**: `object (5 properties)`

Pixel extensions for each boundary direction of the source image. At least one direction is required.

**Learn more** (1 resource):

- [Image Outpainting: Outpaint Defining The Expansion](https://runware.ai/docs/learn/image-outpainting#outpaint-defining-the-expansion) (learn)

#### [top](https://runware.ai/docs/models/flux-1-dev#request-outpaint-top)

- **Path**: `outpaint.top`
- **Type**: `integer`
- **Min**: `0`
- **Step**: `64`

Number of pixels to extend to the top.

#### [bottom](https://runware.ai/docs/models/flux-1-dev#request-outpaint-bottom)

- **Path**: `outpaint.bottom`
- **Type**: `integer`
- **Min**: `0`
- **Step**: `64`

Number of pixels to extend to the bottom.

#### [left](https://runware.ai/docs/models/flux-1-dev#request-outpaint-left)

- **Path**: `outpaint.left`
- **Type**: `integer`
- **Min**: `0`
- **Step**: `64`

Number of pixels to extend to the left.

#### [right](https://runware.ai/docs/models/flux-1-dev#request-outpaint-right)

- **Path**: `outpaint.right`
- **Type**: `integer`
- **Min**: `0`
- **Step**: `64`

Number of pixels to extend to the right.

#### [blur](https://runware.ai/docs/models/flux-1-dev#request-outpaint-blur)

- **Path**: `outpaint.blur`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `32`

Amount of blur to apply to the mask edge.

**Learn more** (1 resource):

- [Image Outpainting: Outpaint Defining The Expansion](https://runware.ai/docs/learn/image-outpainting#outpaint-defining-the-expansion) (learn)

### [lora](https://runware.ai/docs/models/flux-1-dev#request-lora)

- **Path**: `lora.model`
- **Type**: `array of objects (2 properties)`
- **Min items**: `1`

With LoRA (Low-Rank Adaptation), you can adapt a model to specific styles or features by emphasizing particular aspects of the data. This technique enhances the quality and relevance of generated content and can be especially useful when the output needs to adhere to a specific artistic style or follow particular guidelines.

Multiple LoRA models can be used simultaneously to achieve different adaptation goals.

**Examples**:

```json
"lora": [
  {
    "model": "<lora-model-air>",
    "weight": 0.8
  }
]
```

**Learn more** (1 resource):

- [Loras](https://runware.ai/docs/learn/loras) (learn)

#### [model](https://runware.ai/docs/models/flux-1-dev#request-lora-model)

- **Path**: `lora.model`
- **Type**: `string`
- **Required**: true

LoRA model identifier.

#### [weight](https://runware.ai/docs/models/flux-1-dev#request-lora-weight)

- **Path**: `lora.weight`
- **Type**: `float`
- **Min**: `-4`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Strength of the LoRA influence. A value of 0 means no influence. Higher values increase the influence, and negative values can be used to steer away from the LoRA's style.

### [controlNet](https://runware.ai/docs/models/flux-1-dev#request-controlnet)

- **Path**: `controlNet.model`
- **Type**: `array of objects (8 properties)`
- **Min items**: `1`

With ControlNet, you can provide a guide image to help the model generate images that align with the desired structure. This guide image can be generated with our ControlNet preprocessing tool, extracting guidance information from an input image. The guide image can be in the form of an edge map, a pose, a depth estimation or any other type of control image that guides the generation process via the ControlNet model.

Multiple ControlNet models can be used at the same time to provide different types of guidance information to the model.

**Examples**:

```json
"controlNet": [
  {
    "model": "<controlnet-model-air>",
    "guideImage": "c64351d5-4c59-42f7-95e1-eace013eddab",
    "weight": 0.7,
    "startStep": 0,
    "endStep": 20,
    "controlMode": "controlnet"
  }
]
```

**Learn more** (2 resources):

- [Game Assets Canny: Controlnet And Edge Detection](https://runware.ai/docs/models/flux-1-dev/guides/game-assets-canny#controlnet-and-edge-detection) (learn)
- [Controlnet](https://runware.ai/docs/learn/controlnet) (learn)

#### [model](https://runware.ai/docs/models/flux-1-dev#request-controlnet-model)

- **Path**: `controlNet.model`
- **Type**: `string`
- **Required**: true

ControlNet model identifier.

**Allowed values**: `runware:25@1` `runware:26@1` `runware:27@1` `runware:28@1` `runware:29@1` `runware:30@1` `runware:31@1` `runware:98@1`

#### [weight](https://runware.ai/docs/models/flux-1-dev#request-controlnet-weight)

- **Path**: `controlNet.weight`
- **Type**: `float`
- **Min**: `-4`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Strength of the ControlNet influence. A value of 0 means no influence. Higher values increase the influence, and negative values can be used to steer away from the guide image.

#### [guideImage](https://runware.ai/docs/models/flux-1-dev#request-controlnet-guideimage)

- **Path**: `controlNet.guideImage`
- **Type**: `string`
- **Required**: true

Reference image for ControlNet guidance (UUID, URL, Data URI, or Base64).

#### [controlMode](https://runware.ai/docs/models/flux-1-dev#request-controlnet-controlmode)

- **Path**: `controlNet.controlMode`
- **Type**: `string`
- **Default**: `balanced`

ControlNet guidance mode.

**Allowed values**:

- `balanced` Equal weight between ControlNet and prompt.
- `controlnet` Prioritize ControlNet guidance.
- `prompt` Prioritize prompt guidance.

#### [endStep](https://runware.ai/docs/models/flux-1-dev#request-controlnet-endstep)

- **Path**: `controlNet.endStep`
- **Type**: `integer`
- **Min**: `1`

Absolute step number to end ControlNet influence. Must be greater than `startStep` and less than or equal to `steps`.

#### [endStepPercentage](https://runware.ai/docs/models/flux-1-dev#request-controlnet-endsteppercentage)

- **Path**: `controlNet.endStepPercentage`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`

Percentage of steps to end ControlNet influence. Must be greater than `startStepPercentage`.

#### [startStep](https://runware.ai/docs/models/flux-1-dev#request-controlnet-startstep)

- **Path**: `controlNet.startStep`
- **Type**: `integer`
- **Min**: `0`

Absolute step number to start ControlNet influence. Must be less than `endStep`.

#### [startStepPercentage](https://runware.ai/docs/models/flux-1-dev#request-controlnet-startsteppercentage)

- **Path**: `controlNet.startStepPercentage`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `99`

Percentage of steps to start ControlNet influence. Must be less than `endStepPercentage`.

### [ipAdapters](https://runware.ai/docs/models/flux-1-dev#request-ipadapters)

- **Path**: `ipAdapters.model`
- **Type**: `array of objects (3 properties)`
- **Min items**: `1`

IP-Adapters enable image-prompted generation, allowing you to use reference images to guide the style and content of your generations. Multiple IP Adapters can be used simultaneously.

**Examples**:

```json
"ipAdapters": [
  {
    "model": "<ip-adapter-model-air>",
    "guideImages": ["c64351d5-4c59-42f7-95e1-eace013eddab"],
    "weight": 0.75
  },
  {
    "model": "<ip-adapter-model-air>",
    "guideImages": ["d7e8f9a0-2b5c-4e7f-a1d3-9c8b7a6e5d4f"],
    "weight": 0.5
  }
]
```

**Learn more** (1 resource):

- [Ip Adapters](https://runware.ai/docs/learn/ip-adapters) (learn)

#### [model](https://runware.ai/docs/models/flux-1-dev#request-ipadapters-model)

- **Path**: `ipAdapters.model`
- **Type**: `string`
- **Required**: true

We make use of the [AIR system](https://runware.ai/docs/models) to identify IP-Adapter models. This identifier is a unique string that represents a specific model.

**Supported models list**:

| AIR ID | Model Name |
| --- | --- |
| runware:55@1 | IP Adapter SDXL |
| runware:55@2 | IP Adapter SDXL Plus |
| runware:55@3 | IP Adapter SDXL Plus Face |
| runware:55@4 | IP Adapter SDXL Vit-H |
| runware:55@5 | IP Adapter SD 1.5 |
| runware:55@6 | IP Adapter SD 1.5 Plus |
| runware:55@7 | IP Adapter SD 1.5 Light |
| runware:55@8 | IP Adapter SD 1.5 Plus Face |
| runware:55@10 | IP Adapter SD 1.5 Vit-G |

**Allowed values**: `runware:56@1` `runware:56@2` `runware:56@3` `runware:56@4`

#### [weight](https://runware.ai/docs/models/flux-1-dev#request-ipadapters-weight)

- **Path**: `ipAdapters.weight`
- **Type**: `float`
- **Min**: `-4`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Strength of the IP-Adapter influence. A value of 0 means no influence. Higher values increase the influence, and negative values can be used to steer away from the reference.

#### [guideImages](https://runware.ai/docs/models/flux-1-dev#request-ipadapters-guideimages)

- **Path**: `ipAdapters.guideImages`
- **Type**: `array of strings`
- **Required**: true
- **Min items**: `1`

Images to guide the IP-Adapter (UUID, URL, Data URI, or Base64).

### [trueCFGScale](https://runware.ai/docs/models/flux-1-dev#request-truecfgscale)

- **Type**: `float`

True Classifier-Free Guidance scale. Higher values increase prompt adherence at the cost of quality.

**Features**

Standalone addons and post-processing features.

### [ultralytics](https://runware.ai/docs/models/flux-1-dev#request-ultralytics)

- **Path**: `ultralytics.CFGScale`
- **Type**: `object (9 properties)`

Configuration object for Ultralytics face enhancement during generation. This feature uses face detection and inpainting to improve facial details in the same generation step, without requiring post-processing.

> [!NOTE]
> Face enhancement is available for Stable Diffusion 1.X, SDXL, and FLUX models. The system automatically detects faces and applies targeted refinement to improve quality while maintaining consistency with the overall generation.

#### [CFGScale](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-cfgscale)

- **Path**: `ultralytics.CFGScale`
- **Type**: `float`
- **Min**: `0`
- **Max**: `50`
- **Step**: `0.1`
- **Default**: `8`

Face refinement guidance scale.

#### [confidence](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-confidence)

- **Path**: `ultralytics.confidence`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.9`

Confidence threshold for detection.

#### [inpaintSize](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-inpaintsize)

- **Path**: `ultralytics.inpaintSize`
- **Type**: `integer`
- **Min**: `128`
- **Max**: `2048`
- **Default**: `1024`

Image size (in pixels) to use for each inpainting region. YOLO detects faces, crops the region, and scales it to this size before running diffusion. Set so most faces land in the 2–4× range of their original pixel size. Going beyond 8× may degrade identity resemblance.

#### [maskBlur](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-maskblur)

- **Path**: `ultralytics.maskBlur`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `100`
- **Default**: `5`

Mask feathering amount. Higher values create softer transitions between the enhanced face region and surrounding areas.

#### [maskPadding](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-maskpadding)

- **Path**: `ultralytics.maskPadding`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `20`
- **Default**: `5`

Padding around detected face in pixels. Expands the refinement area to include surrounding context like hair and neck.

#### [negativePrompt](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-negativeprompt)

- **Path**: `ultralytics.negativePrompt`
- **Type**: `string`

Negative prompt for detection.

#### [positivePrompt](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-positiveprompt)

- **Path**: `ultralytics.positivePrompt`
- **Type**: `string`

Positive prompt for detection.

#### [steps](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-steps)

- **Path**: `ultralytics.steps`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`
- **Default**: `20`

Number of face refinement steps.

#### [strength](https://runware.ai/docs/models/flux-1-dev#request-ultralytics-strength)

- **Path**: `ultralytics.strength`
- **Type**: `float`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.3`

Refinement strength. Lower values preserve more of the original, higher values allow more aggressive reconstruction.

### [layerDiffuse](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-layerdiffuse)

- **Path**: `advancedFeatures.layerDiffuse`
- **Type**: `boolean`

Enables LayerDiffuse technology, which allows for the direct generation of images with transparency (alpha channels).

When enabled, this feature applies the necessary LwwwwwwoRA and VAE components to produce high-quality transparent images without requiring post-processing background removal.

This is particularly useful for creating product images, overlays, composites, and other content that requires transparency. The output must be in a format that supports transparency, such as PNG.

Note: This feature is only available for the FLUX model architecture. It automatically applies the equivalent of:

```json

  "lora": [{ "model": "runware:120@2" }],
  "vae": "runware:120@4"
```

**Examples**:

```json
"outputFormat": "png",
"advancedFeatures": {
  "layerDiffuse": true
}
```

**Learn more** (1 resource):

- [Introducing Layerdiffuse Generate Images With Built In Transparency In One Step](https://runware.ai/blog/introducing-layerdiffuse-generate-images-with-built-in-transparency-in-one-step) (article)

### [watermark](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark)

- **Path**: `advancedFeatures.watermark`
- **Type**: `object (6 properties)`

Configuration object for adding watermarks to generated videos. Watermarks can be applied using either text or image content with customizable positioning and appearance. You must provide either `text` or `image` content for the watermark, but not both.

**Text watermark**:

```json
"advancedFeatures": {
  "watermark": {
    "text": "© 2025 Company",
    "displayPosition": "bottom-right",
    "opacity": 0.6,
    "fontColor": "#ffffff",
    "bgColor": "#000000"
  }
}
```

**Image watermark**:

```json
"advancedFeatures": {
  "watermark": {
    "image": "c64351d5-4c59-42f7-95e1-eace013eddab",
    "displayPosition": "top-left",
    "opacity": 0.6
  }
}
```

**Tiled watermark**:

```json
"advancedFeatures": {
  "watermark": {
    "text": "PREVIEW",
    "tiled": true,
    "opacity": 0.4,
    "fontColor": "#cccccc"
  }
}
```

#### [text](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-text)

- **Path**: `advancedFeatures.watermark.text`
- **Type**: `string`
- **Min**: `2`
- **Max**: `32`

Watermark text.

#### [image](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-image)

- **Path**: `advancedFeatures.watermark.image`
- **Type**: `string`

Watermark image (UUID, URL, Data URI, or Base64).

#### [displayPosition](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-displayposition)

- **Path**: `advancedFeatures.watermark.displayPosition`
- **Type**: `string`

Watermark position.

**Allowed values**: `top-left` `top-center` `top-right` `center-left` `center-center` `center-right` `bottom-left` `bottom-center` `bottom-right` `tiled`

#### [opacity](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-opacity)

- **Path**: `advancedFeatures.watermark.opacity`
- **Type**: `float`
- **Min**: `0.1`
- **Max**: `1`
- **Step**: `0.01`

Watermark opacity.

#### [fontColor](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-fontcolor)

- **Path**: `advancedFeatures.watermark.fontColor`
- **Type**: `string`

Text color in hex format.

#### [bgColor](https://runware.ai/docs/models/flux-1-dev#request-advancedfeatures-watermark-bgcolor)

- **Path**: `advancedFeatures.watermark.bgColor`
- **Type**: `string`

Background color in hex format.

### [hiresFix](https://runware.ai/docs/models/flux-1-dev#request-hiresfix)

- **Type**: `boolean | object`

Two-stage generation for improved resolution and detail. The model generates at a lower resolution first, then upscales and refines the result in a second pass. Can be enabled with `true` for default settings, or configured as an object for fine-grained control over the upscaling model, steps, and strength.

When using the object form, the `model` parameter is required. Available upscaling models:

| Model | Name | Upscale Factor |
| --- | --- | --- |
| `runware:504@1` | RealESRGAN\_x4plus | 4x |
| `runware:realesrgan@anime-6b` | RealESRGAN\_x4plus\_anime\_6B | 4x |
| `runware:esrgan@animesharp` | 4x-AnimeSharp | 4x |
| `runware:esrgan@ultrasharp` | 4x-UltraSharp | 4x |

**Simple (boolean)**:

```json
"hiresFix": true
```

**Configured (object)**:

```json
"hiresFix": {
  "model": "runware:esrgan@ultrasharp",
  "steps": 15,
  "strength": 0.6
}
```

**Allowed values**: `true`

### [pulid](https://runware.ai/docs/models/flux-1-dev#request-pulid)

- **Path**: `pulid.CFGstartStep`
- **Type**: `object (5 properties)`

PuLID (Pure and Lightning ID Customization) enables fast and high-quality identity customization for text-to-image generation. This object allows you to configure settings for transferring facial characteristics from a reference image to generated images with high fidelity.

> [!NOTE]
> `CFGstartStep` and `CFGstartStepPercentage` are mutually exclusive, they both control when CFG guidance begins, but as an absolute step number or a percentage respectively. Use one or the other, not both.

**Examples**:

```json
"puLID": {
  "images": ["59a2edc2-45e6-429f-be5f-7ded59b92046"],
  "idWeight": 1,
  "trueCFGScale": 1.5,
  "CFGStartStep": 3
}
```

#### [CFGstartStep](https://runware.ai/docs/models/flux-1-dev#request-pulid-cfgstartstep)

- **Path**: `pulid.CFGstartStep`
- **Type**: `integer`
- **Min**: `0`

Absolute step number to start identity influence. Must be less than `steps`.

#### [CFGstartStepPercentage](https://runware.ai/docs/models/flux-1-dev#request-pulid-cfgstartsteppercentage)

- **Path**: `pulid.CFGstartStepPercentage`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `99`

Percentage of steps to start identity influence.

#### [idWeight](https://runware.ai/docs/models/flux-1-dev#request-pulid-idweight)

- **Path**: `pulid.idWeight`
- **Type**: `float`
- **Min**: `0`
- **Max**: `3`
- **Step**: `0.01`
- **Default**: `1`

Identity preservation strength. Higher values create closer resemblance to the reference face.

#### [images](https://runware.ai/docs/models/flux-1-dev#request-pulid-images)

- **Path**: `pulid.images`
- **Type**: `array of strings`
- **Required**: true
- **Min items**: `items: 1`

Reference images for identity customization (UUID, URL, Data URI, or Base64).

#### [trueCFGScale](https://runware.ai/docs/models/flux-1-dev#request-pulid-truecfgscale)

- **Path**: `pulid.trueCFGScale`
- **Type**: `float`
- **Min**: `0`
- **Max**: `10`
- **Step**: `0.1`

Guidance scale for identity embedding.

## Response Parameters

### [taskType](https://runware.ai/docs/models/flux-1-dev#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/flux-1-dev#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/flux-1-dev#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/flux-1-dev#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/flux-1-dev#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/flux-1-dev#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [seed](https://runware.ai/docs/models/flux-1-dev#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [NSFWContent](https://runware.ai/docs/models/flux-1-dev#response-nsfwcontent)

- **Type**: `boolean`

Flag indicating if NSFW content was detected.

### [cost](https://runware.ai/docs/models/flux-1-dev#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.