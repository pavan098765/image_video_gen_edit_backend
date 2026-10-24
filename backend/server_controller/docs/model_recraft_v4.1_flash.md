---
title: Recraft V4.1 Flash | Runware Docs
url: https://runware.ai/docs/models/recraft-v4-1-flash
description: Fast, low-cost design-focused image generation for high-volume work and rapid iteration
---
# Recraft V4.1 Flash

![Recraft V4.1 Flash](https://assets.runware.ai/covers/recraft-v4-1-flash.jpg)

Recraft V4.1 Flash is the speed tier of the Recraft V4.1 family, a dedicated fast raster model for text-to-image generation at 1K resolution. It keeps output quality close to Recraft V4.1, with strong results in photography, portraits, atmosphere, complex scenes, mixed media, and logo concepts. It suits high-volume generation, rapid iteration, and design exploration where latency and cost matter most.

- **Status**: live
- **ID**: `recraft:v4.1@flash`
- Try in Playground
- **Creator**: Recraft
- **Release Date**: September 23, 2026
- **Capabilities**: Text to Image, Checkpoint

## Pricing

$0.007 per run · ~4s per request

**4 example generations**

- [text-to-image](https://runware.ai/docs/models/recraft-v4-1-flash/examples#example-wildlife-culvert-retrofit-explainer): `$0.007` (~4s)
- [text-to-image](https://runware.ai/docs/models/recraft-v4-1-flash/examples#example-circular-brickmaker-editorial-portrait): `$0.007` (~3s)
- [text-to-image](https://runware.ai/docs/models/recraft-v4-1-flash/examples#example-deep-sea-robotics-engineer-portrait): `$0.007` (~4s)
- [text-to-image](https://runware.ai/docs/models/recraft-v4-1-flash/examples#example-rare-book-conservator-profile-avatar): `$0.007` (~3s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

`width` and `height` must be used together.

---

The following dimension combinations are supported:

| Configuration | Dimensions |
| --- | --- |
| `1K (1:1)` | `1024x1024` |
| `1K (2:1)` | `1536x768` |
| `1K (1:2)` | `768x1536` |
| `1K (3:2)` | `1280x832` |
| `1K (2:3)` | `832x1280` |
| `1K (4:3)` | `1216x896` |
| `1K (3:4)` | `896x1216` |
| `1K (5:4)` | `1152x896` |
| `1K (4:5)` | `896x1152` |
| `1K (6:10)` | `832x1344` |
| `1K (14:10)` | `1280x896` |
| `1K (10:14)` | `896x1280` |
| `1K (16:9)` | `1344x768` |
| `1K (9:16)` | `768x1344` |

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/recraft-v4-1-flash#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/recraft-v4-1-flash#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/recraft-v4-1-flash#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/recraft-v4-1-flash#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/recraft-v4-1-flash#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/recraft-v4-1-flash#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/recraft-v4-1-flash#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/recraft-v4-1-flash#request-uploadendpoint)

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

### [safety](https://runware.ai/docs/models/recraft-v4-1-flash#request-safety)

- **Path**: `safety.checkContent`
- **Type**: `object (1 property)`

Content safety checking configuration for image generation.

#### [checkContent](https://runware.ai/docs/models/recraft-v4-1-flash#request-safety-checkcontent)

- **Path**: `safety.checkContent`
- **Type**: `boolean`

Enable or disable content safety checking. Increases total generation time.

### [ttl](https://runware.ai/docs/models/recraft-v4-1-flash#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/recraft-v4-1-flash#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/recraft-v4-1-flash#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `20`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/recraft-v4-1-flash#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words, such as a style or a feature. A model that cannot meet them is not chosen. Output size is set with width and height rather than described here.

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/recraft-v4-1-flash#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `recraft:v4.1@flash`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/recraft-v4-1-flash#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `1`
- **Max**: `10000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [width](https://runware.ai/docs/models/recraft-v4-1-flash#request-width)

- **Type**: `integer`
- **Paired with**: height

Width of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [height](https://runware.ai/docs/models/recraft-v4-1-flash#request-height)

- **Type**: `integer`
- **Paired with**: width

Height of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [backgroundColor](https://runware.ai/docs/models/recraft-v4-1-flash#request-settings-backgroundcolor)

- **Path**: `settings.backgroundColor`
- **Type**: `object (1 property)`

Background color used for the generation.

**Learn more** (1 resource):

- [Color Palettes](https://runware.ai/docs/models/recraft-v4-1/guides/color-palettes) (learn)

#### [rgb](https://runware.ai/docs/models/recraft-v4-1-flash#request-settings-backgroundcolor-rgb)

- **Path**: `settings.backgroundColor.rgb`
- **Type**: `array of integers`
- **Required**: true
- **Min**: `0`
- **Max**: `255`
- **Min items**: `items: 3`

Color as `[red, green, blue]` array.

### [colors](https://runware.ai/docs/models/recraft-v4-1-flash#request-settings-colors)

- **Path**: `settings.colors`
- **Type**: `array of objects (1 property)`

Colors defining the palette for the generation.

**Learn more** (1 resource):

- [Color Palettes](https://runware.ai/docs/models/recraft-v4-1/guides/color-palettes) (learn)

#### [rgb](https://runware.ai/docs/models/recraft-v4-1-flash#request-settings-colors-rgb)

- **Path**: `settings.colors.rgb`
- **Type**: `array of integers`
- **Required**: true
- **Min**: `0`
- **Max**: `255`
- **Min items**: `items: 3`

Color as `[red, green, blue]` array.

**Features**

Standalone addons and post-processing features.

### [watermark](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark)

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

#### [text](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-text)

- **Path**: `advancedFeatures.watermark.text`
- **Type**: `string`
- **Min**: `2`
- **Max**: `32`

Watermark text.

#### [image](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-image)

- **Path**: `advancedFeatures.watermark.image`
- **Type**: `string`

Watermark image (UUID, URL, Data URI, or Base64).

#### [displayPosition](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-displayposition)

- **Path**: `advancedFeatures.watermark.displayPosition`
- **Type**: `string`

Watermark position.

**Allowed values**: `top-left` `top-center` `top-right` `center-left` `center-center` `center-right` `bottom-left` `bottom-center` `bottom-right` `tiled`

#### [opacity](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-opacity)

- **Path**: `advancedFeatures.watermark.opacity`
- **Type**: `float`
- **Min**: `0.1`
- **Max**: `1`
- **Step**: `0.01`

Watermark opacity.

#### [fontColor](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-fontcolor)

- **Path**: `advancedFeatures.watermark.fontColor`
- **Type**: `string`

Text color in hex format.

#### [bgColor](https://runware.ai/docs/models/recraft-v4-1-flash#request-advancedfeatures-watermark-bgcolor)

- **Path**: `advancedFeatures.watermark.bgColor`
- **Type**: `string`

Background color in hex format.

## Response Parameters

### [taskType](https://runware.ai/docs/models/recraft-v4-1-flash#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/recraft-v4-1-flash#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/recraft-v4-1-flash#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/recraft-v4-1-flash#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/recraft-v4-1-flash#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/recraft-v4-1-flash#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [seed](https://runware.ai/docs/models/recraft-v4-1-flash#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [NSFWContent](https://runware.ai/docs/models/recraft-v4-1-flash#response-nsfwcontent)

- **Type**: `boolean`

Flag indicating if NSFW content was detected.

### [cost](https://runware.ai/docs/models/recraft-v4-1-flash#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-image (text-to-image)

![Wildlife Culvert Retrofit Explainer](https://assets.runware.ai/examples/recraft-v4-1-flash/a65e523d-0944-43ea-b8ed-6480378aa85a.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "5b301a39-f351-48ad-88fd-e2191a287e29",
  "model": "recraft:v4.1@flash",
  "positivePrompt": "Finished square editorial cutaway illustration for a municipal wildlife-crossing retrofit guide. Show a three-quarter axonometric section through a rural two-lane road and the stream culvert beneath it. The road, gravel foundation, soil layers, roots, concrete culvert walls, flowing shallow water, stones and planted banks are all clearly visible in one coherent cutaway. Inside the culvert, include raised dry passage ledges securely bracketed along both walls above the waterline, gently sloped access ramps connecting them to the natural banks, and subtle protective fencing that guides animals toward the safe route. A European otter walks along one ledge while a second otter approaches from the reed-filled stream bank; small trout are visible in the water for scale and ecological context. Make every structural relationship immediately understandable without labels, arrows, numbers or text. Precision technical drawing meets natural-history illustration: crisp graphite contours, opaque gouache color, delicate screen-print grain, carefully rendered concrete, aggregate, damp soil, moss, reeds and rippling water. Restrained palette of deep teal, river blue, warm concrete gray, clay brown, moss green and small safety-orange accents on mounting hardware. Soft overcast daylight, calm practical mood, clean pale background around the isolated cutaway, balanced centered composition, high detail, professional infrastructure-publication quality, no border, no logo, no typography, no photorealistic mockup.",
  "width": 1024,
  "height": 1024
}
```

---

### text-to-image (text-to-image)

![Circular Brickmaker Editorial Portrait](https://assets.runware.ai/examples/recraft-v4-1-flash/6f26ca82-16b5-4aee-b6e0-80163a44111d.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "0f4e3849-e326-4add-bc18-2f1c14bf4108",
  "model": "recraft:v4.1@flash",
  "positivePrompt": "Polished square editorial photograph for a contemporary architecture and design magazine, profiling a female materials engineer who manufactures low-carbon bricks from demolition waste. She stands behind a scarred steel worktable inside a working materials laboratory, looking calmly and confidently into the camera. One dust-covered hand holds a jagged fragment of recycled concrete aggregate; the other rests on a beautifully finished speckled masonry block. She wears a deep cobalt work jumpsuit with rolled sleeves, practical safety glasses pushed onto her head, realistic skin texture, loose hair, and fine mineral dust on her hands and clothing. Arrange broken terracotta, crushed concrete, pale stone chips, sample bricks, metal molds, and labeled-but-unreadable specimen jars across the foreground like an intentional material study. In the background, show a hydraulic brick press, stacked curing blocks, hanging tools, and translucent workshop windows, all authentic and slightly worn. Distinctive art direction: rigorous industrial geometry softened by human warmth, mineral palette of chalk white, rust, clay red, concrete gray, and cobalt blue. Mixed cool window light and warm sodium workshop light, subtle direct flash, crisp tactile detail, restrained contrast, gentle 35mm film grain, natural imperfections, premium documentary photography. Eye-level 35mm lens, three-quarter portrait, symmetrical but not sterile composition, subject clearly separated from the layered workshop, deep enough focus to reveal the manufacturing context. No typography, no logo, no border, no magazine mockup, no staged advertising polish.",
  "width": 1024,
  "height": 1024
}
```

---

### text-to-image (text-to-image)

![Deep-Sea Robotics Engineer Portrait](https://assets.runware.ai/examples/recraft-v4-1-flash/428f05e4-a18c-4fac-b78e-d5ecfaa0e1e0.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "8cfb30d5-e2f5-4a62-a526-417bb3d33695",
  "model": "recraft:v4.1@flash",
  "positivePrompt": "Square editorial portrait of a veteran deep-sea robotics engineer in her late 50s, a Black woman with warm deep-brown skin, intelligent dark eyes, subtle smile lines, and short salt-and-pepper microbraids swept neatly back. Head-and-shoulders composition, facing the camera with calm authority and an approachable, quietly confident expression. She wears a refined graphite technical overshirt over a burnt-orange mock-neck base layer, practical and contemporary, with one small titanium pressure-sensor component held loosely at the bottom edge of the frame as a subtle clue to her profession. Photograph her inside a working oceanographic robotics hangar: softly blurred autonomous underwater vehicle with a rounded white hull and articulated manipulator arm in the background, faint workshop rails, cables, and marine equipment rendered as elegant bokeh rather than clutter. Intentional cinematic color contrast of cool deep-ocean cyan, graphite, white, and restrained safety orange. Large diffused key light from camera left creates soft dimensional modeling across her face; cool ambient hangar light adds a delicate cyan rim along her shoulders and hair; natural catchlights in both eyes. Eye-level camera, 85mm portrait-lens character, shallow depth of field, crisp focus on the eyes, realistic pores and skin texture, individually resolved hair, accurate anatomy, nuanced fabric texture, premium documentary photography with the polish of a major science magazine. Clean square crop with comfortable breathing room around the head, no text, no logo, no badge, no watermark, no excessive retouching.",
  "width": 1024,
  "height": 1024
}
```

---

### text-to-image (text-to-image)

![Rare Book Conservator Profile Avatar](https://assets.runware.ai/examples/recraft-v4-1-flash/5b974336-cc6c-4463-8ef4-1d323513080c.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "e5fa2090-491b-445c-ae89-48d9be58c857",
  "model": "recraft:v4.1@flash",
  "positivePrompt": "Polished square profile avatar of a distinguished South Asian woman in her early 50s who works as a museum rare book conservator. Head-and-shoulders composition, centered and facing the viewer with a calm, perceptive expression, silver-streaked dark hair in a loose low bun, round tortoiseshell glasses, oxblood linen work jacket over a parchment-colored blouse, one tiny brass bookbinder’s pin at the collar. Contemporary editorial portrait rendered as richly layered cut paper, opaque gouache, delicate ink hatching, and restrained gold-leaf accents, combining the tactile character of an illuminated manuscript with sophisticated modern magazine illustration. Behind her head, an abstract circular motif built from marbled endpaper waves, archival tissue shapes, fine bookbinding thread, and small geometric gilded fragments; decorative rather than literal, with a strong silhouette that remains legible at thumbnail size. Warm parchment background, intentional palette of oxblood red, verdigris green, midnight ink blue, aged ivory, and antique gold. Soft frontal illumination, subtle dimensional paper shadows, crisp facial features, nuanced skin tone, meticulous handcrafted texture, balanced negative space, premium cultural-institution art direction. No lettering, logos, watermark, frame, interface, or mockup.",
  "width": 1024,
  "height": 1024
}
```