---
title: Krea 2 Turbo | Runware Docs
url: https://runware.ai/docs/models/krea-2-turbo-oss
description: Distilled open-weight Krea 2 architecture
---
# Krea 2 Turbo

![Krea 2 Turbo](https://assets.runware.ai/covers/krea-oss-2-turbo.jpg)

Krea 2 Turbo is the distilled variant of the open-weight Krea 2 architecture, generating in a few steps without classifier-free guidance. LoRAs uploaded for it also run on Krea 2 Raw.

- **Status**: live
- Base Architecture
- Try in Playground
- **Creator**: Krea
- **Capabilities**: Text to Image, Image to Image

## Compatibility & Validation

`width` and `height` must be used together.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/krea-2-turbo-oss#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/krea-2-turbo-oss#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/krea-2-turbo-oss#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/krea-2-turbo-oss#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/krea-2-turbo-oss#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/krea-2-turbo-oss#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/krea-2-turbo-oss#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/krea-2-turbo-oss#request-uploadendpoint)

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

### [safety](https://runware.ai/docs/models/krea-2-turbo-oss#request-safety)

- **Path**: `safety.checkContent`
- **Type**: `object (1 property)`

Content safety checking configuration for image generation.

#### [checkContent](https://runware.ai/docs/models/krea-2-turbo-oss#request-safety-checkcontent)

- **Path**: `safety.checkContent`
- **Type**: `boolean`

Enable or disable content safety checking. Increases total generation time.

### [ttl](https://runware.ai/docs/models/krea-2-turbo-oss#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/krea-2-turbo-oss#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/krea-2-turbo-oss#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `20`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [acceleration](https://runware.ai/docs/models/krea-2-turbo-oss#request-acceleration)

- **Type**: `string`
- **Default**: `medium`

Optimization level.

**Allowed values**: `none` `low` `medium` `high`

### [routing](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/krea-2-turbo-oss#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words, such as a style or a feature. A model that cannot meet them is not chosen. Output size is set with width and height rather than described here.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [referenceImages](https://runware.ai/docs/models/krea-2-turbo-oss#request-inputs-referenceimages)

- **Path**: `inputs.referenceImages`
- **Type**: `array of strings or objects`
- **Min items**: `1`
- **Max items**: `4`

Style reference images for guiding generation. Each item can be a plain image input (UUID, URL, DataURI, or Base64) or an object with an image and a per-image strength.

**Examples**:

**Shorthand format:** When you don't need to specify a custom strength, you can pass a plain image input directly. It will use the default influence strength.

```json
"referenceImages": [
  "aac49721-1964-481a-ae78-8a4e29b91402"
]
```

**Object format:** When you need to control the exact influence of the reference image, use an object with `image` and `strength`. Positive values guide the output toward the reference; negative values guide away from it.

```json
"referenceImages": [
  {
    "image": "aac49721-1964-481a-ae78-8a4e29b91402",
    "strength": 0.7
  }
]
```

**Multiple images:** You can provide multiple reference images and even mix both formats.

```json
"referenceImages": [
  "aac49721-1964-481a-ae78-8a4e29b91402",
  {
    "image": "3ad204c3-a9de-4963-8a1a-c3911e3afafe",
    "strength": 1.2
  }
]
```

**Format 1: string[]**:

- **Type**: `string`

Image input (UUID, URL, Data URI, or Base64).

**Format 2: object[]**:

#### [image](https://runware.ai/docs/models/krea-2-turbo-oss#request-inputs-referenceimages-format-2-image)

- **Path**: `inputs.referenceImages.image`
- **Type**: `string`
- **Required**: true

Image input (UUID, URL, Data URI, or Base64).

#### [strength](https://runware.ai/docs/models/krea-2-turbo-oss#request-inputs-referenceimages-format-2-strength)

- **Path**: `inputs.referenceImages.strength`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Default**: `0.5`

Influence strength of the reference image on the generated output.

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/krea-2-turbo-oss#request-model)

- **Type**: `string`
- **Required**: true

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/krea-2-turbo-oss#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `2`
- **Max**: `10000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [width](https://runware.ai/docs/models/krea-2-turbo-oss#request-width)

- **Type**: `integer`
- **Required**: true
- **Min**: `64`
- **Max**: `2048`
- **Step**: `16`

Width of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [height](https://runware.ai/docs/models/krea-2-turbo-oss#request-height)

- **Type**: `integer`
- **Required**: true
- **Min**: `64`
- **Max**: `2048`
- **Step**: `16`

Height of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [seed](https://runware.ai/docs/models/krea-2-turbo-oss#request-seed)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `9223372036854776000`

Random seed for reproducible generation. When not provided, a random seed is generated in the unsigned 32-bit range.

**Learn more** (1 resource):

- [Seed](https://runware.ai/docs/learn/seed) (learn)

### [steps](https://runware.ai/docs/models/krea-2-turbo-oss#request-steps)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `100`
- **Default**: `8`

Total number of denoising steps. Higher values generally produce more detailed results but take longer.

**Learn more** (1 resource):

- [Steps](https://runware.ai/docs/learn/steps) (learn)

### [lora](https://runware.ai/docs/models/krea-2-turbo-oss#request-lora)

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

#### [model](https://runware.ai/docs/models/krea-2-turbo-oss#request-lora-model)

- **Path**: `lora.model`
- **Type**: `string`
- **Required**: true

LoRA model identifier.

#### [weight](https://runware.ai/docs/models/krea-2-turbo-oss#request-lora-weight)

- **Path**: `lora.weight`
- **Type**: `float`
- **Min**: `-4`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Strength of the LoRA influence. A value of 0 means no influence. Higher values increase the influence, and negative values can be used to steer away from the LoRA's style.

**Features**

Standalone addons and post-processing features.

### [watermark](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark)

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

#### [text](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-text)

- **Path**: `advancedFeatures.watermark.text`
- **Type**: `string`
- **Min**: `2`
- **Max**: `32`

Watermark text.

#### [image](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-image)

- **Path**: `advancedFeatures.watermark.image`
- **Type**: `string`

Watermark image (UUID, URL, Data URI, or Base64).

#### [displayPosition](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-displayposition)

- **Path**: `advancedFeatures.watermark.displayPosition`
- **Type**: `string`

Watermark position.

**Allowed values**: `top-left` `top-center` `top-right` `center-left` `center-center` `center-right` `bottom-left` `bottom-center` `bottom-right` `tiled`

#### [opacity](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-opacity)

- **Path**: `advancedFeatures.watermark.opacity`
- **Type**: `float`
- **Min**: `0.1`
- **Max**: `1`
- **Step**: `0.01`

Watermark opacity.

#### [fontColor](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-fontcolor)

- **Path**: `advancedFeatures.watermark.fontColor`
- **Type**: `string`

Text color in hex format.

#### [bgColor](https://runware.ai/docs/models/krea-2-turbo-oss#request-advancedfeatures-watermark-bgcolor)

- **Path**: `advancedFeatures.watermark.bgColor`
- **Type**: `string`

Background color in hex format.

## Response Parameters

### [taskType](https://runware.ai/docs/models/krea-2-turbo-oss#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/krea-2-turbo-oss#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/krea-2-turbo-oss#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/krea-2-turbo-oss#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/krea-2-turbo-oss#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/krea-2-turbo-oss#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [seed](https://runware.ai/docs/models/krea-2-turbo-oss#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [NSFWContent](https://runware.ai/docs/models/krea-2-turbo-oss#response-nsfwcontent)

- **Type**: `boolean`

Flag indicating if NSFW content was detected.

### [cost](https://runware.ai/docs/models/krea-2-turbo-oss#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.