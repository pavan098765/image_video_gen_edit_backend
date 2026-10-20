---
title: Real-ESRGAN | Runware Docs
url: https://runware.ai/docs/models/real-esrgan
description: Practical GAN-based upscaler for real-world images
---
# Real-ESRGAN

![Real-ESRGAN](https://assets.runware.ai/covers/real-esrgan.jpg)

Real-ESRGAN is a deep learning-based image super-resolution model that enhances low-resolution images by inferring high-frequency detail and preserving texture quality. It uses generative adversarial techniques to upscale images while reducing noise and maintaining natural appearance.

- **Status**: live
- **ID**: `runware:504@1`
- Try in Playground
- **Release Date**: July 22, 2021
- **Zero Data Retention**: Yes
- **Capabilities**: Image to Image, Upscale, Checkpoint

## Pricing

$0.0006 from, per run · ~9s per request

**Measured runs**

- upscaleFactor 2: `$0.0006` (~4s)
- upscaleFactor 4: `$0.0013` (~7s)

**4 example generations**

- [upscale](https://runware.ai/docs/models/real-esrgan/examples#example-prosthetic-running-blade-catalog-photo): `$0.0013` (~7s)
- [upscale](https://runware.ai/docs/models/real-esrgan/examples#example-desert-observatory-night-relight): `$0.0013` (~10s)
- [upscale](https://runware.ai/docs/models/real-esrgan/examples#example-herbarium-curator-catalogue-portrait): `$0.0026` (~16s)
- [upscale](https://runware.ai/docs/models/real-esrgan/examples#example-saffron-apiary-smoker-colorway): `$0.0026` (~13s)

billed by compute time, so a request costs what it took · times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/real-esrgan#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `upscale`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/real-esrgan#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/real-esrgan#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/real-esrgan#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/real-esrgan#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/real-esrgan#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/real-esrgan#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/real-esrgan#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/real-esrgan#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/real-esrgan#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [routing](https://runware.ai/docs/models/real-esrgan#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/real-esrgan#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/real-esrgan#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/real-esrgan#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/real-esrgan#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/real-esrgan#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/real-esrgan#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [image](https://runware.ai/docs/models/real-esrgan#request-inputs-image)

- **Path**: `inputs.image`
- **Type**: `string`
- **Required**: true

Image input (UUID, URL, Data URI, or Base64).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/real-esrgan#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `runware:504@1`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [upscaleFactor](https://runware.ai/docs/models/real-esrgan#request-upscalefactor)

- **Type**: `integer`

Factor by which to upscale the generated image. A value of 2 doubles width and height.

**Allowed values**: `2` `4`

## Response Parameters

### [taskType](https://runware.ai/docs/models/real-esrgan#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `upscale`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/real-esrgan#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/real-esrgan#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/real-esrgan#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/real-esrgan#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/real-esrgan#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [cost](https://runware.ai/docs/models/real-esrgan#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### upscale (upscale)

![Herbarium Curator Catalogue Portrait](https://assets.runware.ai/examples/real-esrgan/d4231e32-0850-4ad7-b370-c6aa6819c0c3.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "8f6cbb6d-7e10-4334-9471-7d6c77df94bd",
  "model": "runware:504@1",
  "upscaleFactor": 4,
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/f9f39878-58b9-4254-9b40-d2a0613c0deb.jpg"
  }
}
```

---

### upscale (upscale)

![Prosthetic Running Blade Catalog Photo](https://assets.runware.ai/examples/real-esrgan/6cc814c0-da95-4ee6-bb42-aad9150505a1.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "200dfb96-5a33-45f8-96a5-6fd9b62b3618",
  "model": "runware:504@1",
  "upscaleFactor": 4,
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/491bf548-85e6-4afb-92c8-4420c5963153.jpg"
  }
}
```

---

### upscale (upscale)

![Desert Observatory Night Relight](https://assets.runware.ai/examples/real-esrgan/42e5c1ad-cf9e-4a7a-9fc2-edc1603240a2.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "9ef9bf83-9aed-4c73-9301-6fb6424f7e0a",
  "model": "runware:504@1",
  "upscaleFactor": 4,
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/3623ba64-a143-45b5-809b-31e6ee6c13eb.jpg"
  }
}
```

---

### upscale (upscale)

![Saffron Apiary Smoker Colorway](https://assets.runware.ai/examples/real-esrgan/b7bc5a3b-a077-4306-b170-444a1d3ba876.jpg)

```json
{
  "taskType": "upscale",
  "taskUUID": "f807e06d-115e-4926-82c4-0a168ea00421",
  "model": "runware:504@1",
  "upscaleFactor": 4,
  "inputs": {
    "image": "https://assets.runware.ai/assets/inputs/834e30e7-92c3-454c-aa98-fa0db0ea0a32.jpg"
  }
}
```