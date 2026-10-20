---
title: P-Image Upscale | Runware Docs
url: https://runware.ai/docs/models/prunaai-p-image-upscale
description: AI-powered image upscaling up to 8 megapixels with detail and realism enhancement
---
# P-Image Upscale

![P-Image Upscale](https://assets.runware.ai/covers/prunaai-p-image-upscale.jpg)

P-Image Upscale is an image upscaling model from Pruna that increases resolution using AI with a configurable target from 1 to 8 megapixels. It includes optional detail enhancement and realism enhancement controls for improving output quality, particularly effective on AI-generated images. It supports JPEG and PNG output formats with both synchronous and asynchronous processing.

- **Status**: live
- **ID**: `prunaai:p-image@upscale`
- Try in Playground
- **Creator**: Pruna AI
- **Release Date**: April 2, 2026
- **Capabilities**: Image to Image, Upscale, Checkpoint

## Pricing

$0.005 from, per run · ~18s per request

**Rates** (per run)

- 1–4 MP: `$0.005`
- 5–8 MP: `$0.01`
- 9-16 MP: `$0.02`
- 17-32 MP: `$0.04`
- 33-64 MP: `$0.06`
- 65-128 MP: `$0.12`

**4 example generations**

- [upscale](https://runware.ai/docs/models/prunaai-p-image-upscale/examples#example-lichen-green-microscope-colorway-master): `$0.02` (~19s)
- [upscale](https://runware.ai/docs/models/prunaai-p-image-upscale/examples#example-deep-sea-exhibit-wall-panorama): `$0.02` (~14s)
- [upscale](https://runware.ai/docs/models/prunaai-p-image-upscale/examples#example-historic-glasshouse-cleanup-master): `$0.02` (~26s)
- [upscale](https://runware.ai/docs/models/prunaai-p-image-upscale/examples#example-falconry-hood-auction-catalog-image): `$0.02` (~16s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

`upscaleFactor` cannot be used with `targetMegapixels`.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/prunaai-p-image-upscale#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `upscale`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/prunaai-p-image-upscale#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/prunaai-p-image-upscale#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/prunaai-p-image-upscale#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/prunaai-p-image-upscale#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/prunaai-p-image-upscale#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/prunaai-p-image-upscale#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/prunaai-p-image-upscale#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/prunaai-p-image-upscale#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/prunaai-p-image-upscale#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [routing](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/prunaai-p-image-upscale#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [image](https://runware.ai/docs/models/prunaai-p-image-upscale#request-inputs-image)

- **Path**: `inputs.image`
- **Type**: `string`
- **Required**: true

Image input (UUID, URL, Data URI, or Base64).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/prunaai-p-image-upscale#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `prunaai:p-image@upscale`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [upscaleFactor](https://runware.ai/docs/models/prunaai-p-image-upscale#request-upscalefactor)

- **Type**: `integer`
- **Min**: `2`
- **Max**: `16`

Factor by which to upscale the generated image. A value of 2 doubles width and height.

### [targetMegapixels](https://runware.ai/docs/models/prunaai-p-image-upscale#request-targetmegapixels)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `128`

Target resolution in megapixels. Output is capped at 128 MP.

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [enhanceDetails](https://runware.ai/docs/models/prunaai-p-image-upscale#request-settings-enhancedetails)

- **Path**: `settings.enhanceDetails`
- **Type**: `boolean`

Enhances fine textures and small details. May increase contrast and introduce minor deviations from the original.

### [realism](https://runware.ai/docs/models/prunaai-p-image-upscale#request-settings-realism)

- **Path**: `settings.realism`
- **Type**: `boolean`

Improves realism in the upscaled image. May deviate more from the original. Recommended for AI-generated images.

## Response Parameters

### [taskType](https://runware.ai/docs/models/prunaai-p-image-upscale#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `upscale`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/prunaai-p-image-upscale#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/prunaai-p-image-upscale#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/prunaai-p-image-upscale#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/prunaai-p-image-upscale#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/prunaai-p-image-upscale#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [cost](https://runware.ai/docs/models/prunaai-p-image-upscale#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### upscale (upscale)

![Lichen Green Microscope Colorway Master](https://assets.runware.ai/examples/prunaai-p-image-upscale/bb52dd09-b229-4826-ba06-3b488ba29fe5.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "81198a45-d528-466b-93eb-dc6cd4fc8835",
  "model": "prunaai:p-image@upscale",
  "upscaleFactor": 4,
  "settings": {
    "enhanceDetails": true,
    "realism": true
  },
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/175b4f54-0c3e-46f2-a515-c1d7ecd2324d.jpg"
  }
}
```

---

### upscale (upscale)

![Deep-Sea Exhibit Wall Panorama](https://assets.runware.ai/examples/prunaai-p-image-upscale/45f14eea-7e30-4b2e-815e-58058bae846d.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "5e827228-a2ab-4147-ae07-2cee1979096c",
  "model": "prunaai:p-image@upscale",
  "upscaleFactor": 4,
  "settings": {
    "enhanceDetails": true,
    "realism": true
  },
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/d8232d9b-bb1b-45d1-92ad-54f7463e1e9c.jpg"
  }
}
```

---

### upscale (upscale)

![Historic Glasshouse Cleanup Master](https://assets.runware.ai/examples/prunaai-p-image-upscale/16bfbca7-1a14-41c6-8b1f-d57493ae0d62.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "3a63ed90-70f9-4699-a934-501df51f2259",
  "model": "prunaai:p-image@upscale",
  "upscaleFactor": 4,
  "settings": {
    "enhanceDetails": true,
    "realism": true
  },
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/4d9bb60a-b0c5-47a7-a9e4-9dc47f94683f.jpg"
  }
}
```

---

### upscale (upscale)

![Falconry Hood Auction Catalog Image](https://assets.runware.ai/examples/prunaai-p-image-upscale/4bdbfaf3-d1bf-44fd-8b33-dd951e7e35fb.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "32075ece-52c5-4247-81c1-0985b1de9807",
  "model": "prunaai:p-image@upscale",
  "upscaleFactor": 4,
  "settings": {
    "enhanceDetails": true,
    "realism": true
  },
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/505b6898-8d32-45d9-83b2-b2f25d9799e1.jpg"
  }
}
```