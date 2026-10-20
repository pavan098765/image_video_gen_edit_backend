---
title: FLUX.2 [klein] 9B | Runware Docs
url: https://runware.ai/docs/models/bfl-flux-2-klein-9b
description: Ultra-fast image generation and editing with sub-second latency
---
# FLUX.2 [klein] 9B

![FLUX.2 [klein] 9B](https://assets.runware.ai/covers/bfl-flux-2-klein-9b.jpg)

FLUX.2 [klein] 9B is a 4-step distilled image generation and editing model designed for sub-second inference without sacrificing visual quality. It unifies text-to-image and advanced editing workflows in a single model, making it suitable for interactive applications, real-time previews, and latency-critical production use.

- **Status**: live
- **ID**: `runware:400@2`
- Try in Playground
- **Creator**: Black Forest Labs
- **Release Date**: January 23, 2026
- **Zero Data Retention**: Yes
- **Capabilities**: Text to Image, Image to Image, Edit, Checkpoint

## Pricing

$0.00078 from, per run · ~4s per request

**Measured runs**

- 1024x1024 · 4 steps: `$0.00078`

**4 example generations**

- [text-to-image](https://runware.ai/docs/models/bfl-flux-2-klein-9b/examples#example-resonator-guitar-diner-background-swap) · 1024×1024: `$0.00078` (~2s)
- [reference-to-image](https://runware.ai/docs/models/bfl-flux-2-klein-9b/examples#example-archaeological-fieldwear-fashion-lookbook) · 720p: `$0.00169` (~7s)
- [text-to-image](https://runware.ai/docs/models/bfl-flux-2-klein-9b/examples#example-municipal-snowplow-ceremony-invitation) · 1024×1024: `$0.00169` (~2s)
- [reference-to-image](https://runware.ai/docs/models/bfl-flux-2-klein-9b/examples#example-community-electronics-repair-social-ad) · 720p: `$0.00338` (~6s)

billed by compute time, so a request costs what it took · times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

`acceleration` cannot be used with `acceleratorOptions`.

---

`width` and `height` must be used together.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-uploadendpoint)

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

### [safety](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-safety)

- **Path**: `safety.checkContent`
- **Type**: `object (1 property)`

Content safety checking configuration for image generation.

#### [checkContent](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-safety-checkcontent)

- **Path**: `safety.checkContent`
- **Type**: `boolean`

Enable or disable content safety checking. Increases total generation time.

### [ttl](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `20`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [acceleration](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleration)

- **Type**: `string`
- **Default**: `high`

Optimization level.

**Allowed values**: `none` `low` `medium` `high`

### [acceleratorOptions](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions)

- **Path**: `acceleratorOptions.cacheEndStep`
- **Type**: `object (10 properties)`

Advanced caching mechanisms to speed up generation.

#### [cacheEndStep](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-cacheendstep)

- **Path**: `acceleratorOptions.cacheEndStep`
- **Type**: `integer`
- **Min**: `1`

Absolute step number to end caching. Must be greater than `cacheStartStep` and less than or equal to `steps`.

#### [cacheEndStepPercentage](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-cacheendsteppercentage)

- **Path**: `acceleratorOptions.cacheEndStepPercentage`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`

Percentage of steps to end caching. Alternative to `cacheEndStep`. Must be greater than `cacheStartStepPercentage`.

#### [cacheMaxConsecutiveSteps](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-cachemaxconsecutivesteps)

- **Path**: `acceleratorOptions.cacheMaxConsecutiveSteps`
- **Type**: `integer`
- **Min**: `1`
- **Max**: `5`
- **Default**: `3`

Limits the maximum number of consecutive steps that can use cached computations before forcing a fresh computation.

#### [cacheStartStep](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-cachestartstep)

- **Path**: `acceleratorOptions.cacheStartStep`
- **Type**: `integer`
- **Min**: `0`

Absolute step number to start caching. Must be less than `cacheEndStep`.

#### [cacheStartStepPercentage](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-cachestartsteppercentage)

- **Path**: `acceleratorOptions.cacheStartStepPercentage`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `99`

Percentage of steps to start caching. Alternative to `cacheStartStep`. Must be less than `cacheEndStepPercentage`.

#### [teaCache](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-teacache)

- **Path**: `acceleratorOptions.teaCache`
- **Type**: `boolean`

TeaCache acceleration for transformer-based models. Estimates step differences to skip redundant computations.

#### [teaCacheDistance](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-teacachedistance)

- **Path**: `acceleratorOptions.teaCacheDistance`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.5`

Controls the aggressiveness of the TeaCache feature. Lower values prioritize quality, higher values prioritize speed.

#### [dbCache](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-dbcache)

- **Path**: `acceleratorOptions.dbCache`
- **Type**: `boolean`

DB Cache (CacheDiT) acceleration. Caches and reuses intermediate transformer block outputs to skip redundant computations.

#### [dbCacheThreshold](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-dbcachethreshold)

- **Path**: `acceleratorOptions.dbCacheThreshold`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.25`

Controls the sensitivity threshold for DB Cache. Lower values reuse cached blocks more aggressively, higher values prioritize quality.

#### [dbCacheSkipInterval](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-acceleratoroptions-dbcacheskipinterval)

- **Path**: `acceleratorOptions.dbCacheSkipInterval`
- **Type**: `integer`
- **Min**: `1`
- **Default**: `5`

Controls how many steps to skip between cache refreshes. Higher values skip more steps for faster generation at the cost of quality.

### [routing](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [referenceImages](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-inputs-referenceimages)

- **Path**: `inputs.referenceImages`
- **Type**: `array of strings`
- **Min items**: `1`
- **Max items**: `4`
- **Aspect ratio (each)**: `1:8 to 8:1`

List of reference images (UUID, URL, Data URI, or Base64).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `runware:400@2`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `1`
- **Max**: `10000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [negativePrompt](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-negativeprompt)

- **Type**: `string`
- **Min**: `1`
- **Max**: `10000`

Prompt to guide what to exclude from generation. Ignored when guidance is disabled (CFGScale ≤ 1).

**Learn more** (1 resource):

- [Prompts: Negative Prompts](https://runware.ai/docs/learn/prompts#negative-prompts) (learn)

### [width](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-width)

- **Type**: `integer`
- **Required**: true
- **Min**: `128`
- **Max**: `2048`
- **Step**: `16`

Width of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [height](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-height)

- **Type**: `integer`
- **Required**: true
- **Min**: `128`
- **Max**: `2048`
- **Step**: `16`

Height of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [seed](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-seed)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `9223372036854776000`

Random seed for reproducible generation. When not provided, a random seed is generated in the unsigned 32-bit range.

**Learn more** (1 resource):

- [Seed](https://runware.ai/docs/learn/seed) (learn)

### [steps](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-steps)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `50`
- **Default**: `4`

Total number of denoising steps. Higher values generally produce more detailed results but take longer.

**Learn more** (1 resource):

- [Steps](https://runware.ai/docs/learn/steps) (learn)

### [scheduler](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-scheduler)

- **Type**: `string`

Scheduler to use for the diffusion process.

**Allowed values**: `DDIM` `DDIMScheduler` `DDPMScheduler` `DEISMultistepScheduler` `Default` `DPM++` `DPM++ 2M` `DPM++ 2M Beta` `DPM++ 2M Exponential` `DPM++ 2M Karras` `DPM++ 2M SDE` `DPM++ 2M SDE Beta` `DPM++ 2M SDE Exponential` `DPM++ 2M SDE Karras` `DPM++ 2M SDE Uniform` `DPM++ 2M Uniform` `DPM++ 3M` `DPM++ 3M Beta` `DPM++ 3M Exponential` `DPM++ 3M Karras` `DPM++ 3M SDE Uniform` `DPM++ 3M Uniform` `DPM++ Beta` `DPM++ Exponential` `DPM++ Karras` `DPM++ SDE` `DPM++ SDE Beta` `DPM++ SDE Exponential` `DPM++ SDE Karras` `DPM++ Uniform` `DPM++ Uniform Beta` `DPM++ Uniform Exponential` `DPM++ Uniform Karras` `DPMSolverMultistepInverse` `DPMSolverMultistepScheduler` `DPMSolverSinglestepScheduler` `EDMDPMSolverMultistepScheduler` `EDMEulerScheduler` `Euler` `Euler a` `Euler Beta` `Euler Exponential` `Euler Karras` `EulerAncestralDiscreteScheduler` `EulerDiscreteScheduler` `FlowMatchEulerDiscreteScheduler` `Heun` `HeunDiscreteScheduler` `Heun Karras` `IPNDMScheduler` `IPNDM Uniform` `IPNDM Uniform Beta` `IPNDM Uniform Exponential` `IPNDM Uniform Karras` `KDPM2AncestralDiscreteScheduler` `KDPM2DiscreteScheduler` `LCM` `LCMScheduler` `LMS` `LMSDiscreteScheduler` `LMS Karras` `PNDMScheduler` `TCDScheduler` `UniPC` `UniPC 2M` `UniPC 2M Karras` `UniPC 2M Uniform` `UniPC 3M` `UniPC 3M Karras` `UniPC 3M Uniform` `UniPC Karras` `UniPC Uniform` `UniPC Uniform Beta` `UniPC Uniform Exponential` `UniPC Uniform Karras`

**Learn more** (2 resources):

- [Schedulers](https://runware.ai/docs/learn/schedulers) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [CFGScale](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-cfgscale)

- **Type**: `float`
- **Min**: `1`
- **Max**: `20`
- **Step**: `0.01`
- **Default**: `3.5`

Guidance scale representing how closely the output will resemble the prompt. Higher values produce results more aligned with the prompt.

**Learn more** (1 resource):

- [Cfg Scale](https://runware.ai/docs/learn/cfg-scale) (learn)

### [lora](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-lora)

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

#### [model](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-lora-model)

- **Path**: `lora.model`
- **Type**: `string`
- **Required**: true

LoRA model identifier.

#### [weight](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-lora-weight)

- **Path**: `lora.weight`
- **Type**: `float`
- **Min**: `-4`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Strength of the LoRA influence. A value of 0 means no influence. Higher values increase the influence, and negative values can be used to steer away from the LoRA's style.

**Features**

Standalone addons and post-processing features.

### [watermark](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark)

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

#### [text](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-text)

- **Path**: `advancedFeatures.watermark.text`
- **Type**: `string`
- **Min**: `2`
- **Max**: `32`

Watermark text.

#### [image](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-image)

- **Path**: `advancedFeatures.watermark.image`
- **Type**: `string`

Watermark image (UUID, URL, Data URI, or Base64).

#### [displayPosition](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-displayposition)

- **Path**: `advancedFeatures.watermark.displayPosition`
- **Type**: `string`

Watermark position.

**Allowed values**: `top-left` `top-center` `top-right` `center-left` `center-center` `center-right` `bottom-left` `bottom-center` `bottom-right` `tiled`

#### [opacity](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-opacity)

- **Path**: `advancedFeatures.watermark.opacity`
- **Type**: `float`
- **Min**: `0.1`
- **Max**: `1`
- **Step**: `0.01`

Watermark opacity.

#### [fontColor](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-fontcolor)

- **Path**: `advancedFeatures.watermark.fontColor`
- **Type**: `string`

Text color in hex format.

#### [bgColor](https://runware.ai/docs/models/bfl-flux-2-klein-9b#request-advancedfeatures-watermark-bgcolor)

- **Path**: `advancedFeatures.watermark.bgColor`
- **Type**: `string`

Background color in hex format.

## Response Parameters

### [taskType](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [seed](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [NSFWContent](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-nsfwcontent)

- **Type**: `boolean`

Flag indicating if NSFW content was detected.

### [cost](https://runware.ai/docs/models/bfl-flux-2-klein-9b#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-image (text-to-image)

![Resonator Guitar Diner Background Swap](https://assets.runware.ai/examples/bfl-flux-2-klein-9b/1d9a4575-23df-4f00-8ea8-39fa45d2a967.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "e25674e5-13fd-483c-bd66-5dd57a18e5fe",
  "model": "runware:400@2",
  "positivePrompt": "Create a polished, photorealistic scene-replacement image for a musical instrument dealer’s editorial catalog. The hero subject is one vintage nickel-plated resonator guitar, full instrument visible, upright on a discreet black floor stand beside an oxblood-red vinyl booth inside a nearly empty 1950s roadside diner. Preserve crisp, believable product detail: circular spun-aluminum resonator cone, perforated cover plate, warm mahogany neck, six strings, aged metal patina and subtle handling marks. Compose for a square crop with the guitar slightly left of center and ample environmental space around its silhouette. Through broad diner windows, show a quiet desert highway at deep blue dusk. Amber pendant lights and a small turquoise neon glow outside create elegant reflections across the chrome body without obscuring its construction. Checkerboard terrazzo floor, brushed-steel counter, coffee urn and neatly aligned stools recede softly into the background. Natural perspective, realistic contact shadow, coherent reflections, shallow but not extreme depth of field, premium medium-format commercial photography, restrained teal, rust, cream and amber palette, contemplative after-hours mood. No people, no readable signage, no logos, no typography, no border, no mockup.",
  "width": 1024,
  "height": 1024
}
```

---

### text-to-image (text-to-image)

![Municipal Snowplow Ceremony Invitation](https://assets.runware.ai/examples/bfl-flux-2-klein-9b/4e4b0e08-202b-415c-a160-a03cb5d6987b.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "6408e7de-15ce-4730-be0a-dd7dc095232f",
  "model": "runware:400@2",
  "positivePrompt": "Create a finished square invitation design for a municipal public works snowplow naming ceremony. Full-bleed flat artwork, not a photographed card or mockup. Feature a large safety-orange snowplow seen from a dramatic three-quarter overhead angle, pushing a crisp diagonal wave of white snow across an icy cyan ground. The cleared path creates a strong graphic space for the event details. Add small bundled-up families holding ballot cards near the bottom edge, simplified into charming geometric cut-paper figures. Art direction: sophisticated mid-century civic poster blended with contemporary screen printing, bold flat shapes, subtle paper grain, slightly imperfect ink registration, navy blue shadows, safety orange, icy cyan, warm cream and tiny red accents. Bright clear winter-morning lighting, upbeat community mood, strong diagonal composition, generous spacing and excellent visual hierarchy. Include clean, highly legible navy and orange lettering with this exact copy: “YOU’RE INVITED” at the top; large headline “NAME THE FLEET”; subheading “Snowplow Naming Ceremony”; details “SATURDAY, NOVEMBER 14 • 10:00 AM”; “NORTH WORKS YARD • 2400 QUARRY ROAD”; footer line “HOT COCOA • TRUCK TOURS • FAMILY VOTING”; and small organizer credit “HARBOR POINT PUBLIC WORKS”. Make the typography feel intentionally designed and integrated into the illustration. One polished, print-ready invitation front, no envelope, no table, no hands, no border, no logos, no additional text.",
  "width": 1024,
  "height": 1024
}
```

---

### reference-to-image (reference-to-image)

![Community Electronics Repair Social Ad](https://assets.runware.ai/examples/bfl-flux-2-klein-9b/d8bd6736-7dad-422e-954e-b7b69f4b3317.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "f58de727-8e73-4976-ad93-f80d05959637",
  "model": "runware:400@2",
  "positivePrompt": "Create a finished 1280x704 landscape social media campaign image for a free community electronics repair clinic. Combine both references with clear roles: preserve the technician’s recognizable face, cobalt utility jumpsuit, burnt-orange cassette deck, visible circuit board, and soldering tool from reference image 1; place her inside the exact civic-library repair room from reference image 2, retaining its checkerboard terrazzo, mustard shelving, maple benches, tool pegboards, gridded windows, and warm afternoon light. Show her at the left-center workbench actively repairing the opened cassette deck while two diverse adult neighbors watch and learn, with a disassembled table radio and neatly sorted screws nearby. Dynamic eye-level three-quarter composition, inviting human interaction, confident competence, authentic hands and electronics, generous environmental context across the wide frame. Art direction: bold contemporary editorial photography inspired by late-1990s electronics catalogs and civic-service campaigns, crisp direct flash balanced with golden window light, saturated cobalt, burnt orange, mustard, cream, and dark brown, subtle halftone texture in the shadows, tactile analog character, energetic but trustworthy mood. Designed to stop the scroll without looking like a stock photo. Standalone image only, no headline, no typography, no logos, no watermark, no social media interface.",
  "negativePrompt": "Text, captions, signage, logos, brand marks, watermark, app interface, website mockup, borders, illegible electronics, melted components, extra fingers, malformed hands, duplicated people, floating tools, science-fiction devices, neon lighting, sterile corporate office, glossy luxury advertising, empty lifeless composition.",
  "width": 1280,
  "height": 704,
  "inputs": {
    "referenceImages": [
      "https://assets.runware.ai/assets/inputs/946b7f90-e7c2-4081-94a8-ec3a3885603b.jpg",
      "https://assets.runware.ai/assets/inputs/79295c04-c94c-4936-93f8-7b2fe110ca62.jpg"
    ]
  }
}
```

---

### reference-to-image (reference-to-image)

![Archaeological Fieldwear Fashion Lookbook](https://assets.runware.ai/examples/bfl-flux-2-klein-9b/1be20e11-43c7-4c95-99ae-dc94c507322d.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "64a77e87-0c55-4f1b-978d-b5e330b91f3b",
  "model": "runware:400@2",
  "positivePrompt": "Create a finished landscape fashion lookbook photograph for a premium archaeological fieldwear catalog. Use the exact model from reference image 2, preserving their recognizable face, warm brown complexion, short dark curls, lean proportions, and composed presence. Dress them in the complete outfit from reference image 1, faithfully retaining the sand ripstop fabric, cropped utility-jacket silhouette, cobalt reinforced shoulder panels, brass snaps, bellows pockets, high-waisted cargo trousers, articulated knees, ochre woven belt, canvas gaiters, and dusty brown field boots.\n\nPlace the model at an active but orderly desert excavation site at late afternoon, standing beside a clean gridded trench cut into pale ochre earth. Include restrained, authentic field details in the middle distance: survey stakes, taut grid string, a folding wood sieve, labeled specimen crates without readable text, and a canvas shade canopy. No ancient monuments, fantasy ruins, or adventure-film styling. The clothing remains the clear subject.\n\nCompose as a sophisticated 1280x704 double-page-style catalog hero image without page borders or typography: full-body model positioned in the right third, boots fully visible, relaxed three-quarter stance with one hand lightly touching a jacket pocket; generous negative space and layered excavation geometry on the left. Eye-level camera, 50mm editorial perspective, crisp garment construction and accurate fabric texture, subtle wind moving the jacket hem. Art direction inspired by late-1970s European workwear catalogs and contemporary minimalist fashion photography. Sun-baked ochre, sand, cobalt blue, cream, and muted tobacco palette. Hard desert sunlight softened by dusty atmosphere, long graphic shadows, natural skin texture, understated confidence, refined analog color response, fine film grain, realistic commercial fashion photography.",
  "width": 1280,
  "height": 704,
  "inputs": {
    "referenceImages": [
      "https://assets.runware.ai/assets/inputs/4b106346-51a4-4cd6-9260-c418c2fdebd6.jpg",
      "https://assets.runware.ai/assets/inputs/daede459-ae56-4706-a462-ce837eea7fba.jpg"
    ]
  }
}
```