---
title: Muse Image | Runware Docs
url: https://runware.ai/docs/models/meta-muse-image
description: Agentic image generation and editing with multi-reference composition, clean text rendering, and precise local edits
---
# Muse Image

![Muse Image](https://assets.runware.ai/covers/meta-muse-image.jpg)

Muse Image is Meta's flagship image generation model from Meta Superintelligence Labs. It is built for prompt-faithful image creation, precision editing, and multi-reference composition, with strong text rendering and the ability to refine existing photos through localized markup-based edits. Meta positions it as an agentic image model that plans layouts, uses search and coding tools to improve accuracy, blends multiple visual references intelligently, and handles both creative generation and practical visual tasks such as infographics, QR codes, restorations, product-style mockups, and photobomber removal.

- **Status**: live
- **ID**: `meta:muse@image`
- Try in Playground
- **Creator**: Meta
- **Release Date**: July 7, 2026
- **Capabilities**: Text to Image, Image to Image, Edit, Checkpoint

## Pricing

$0.01 per run · ~22s per request

**4 example generations**

- [reference-to-image](https://runware.ai/docs/models/meta-muse-image/examples#example-beekeeping-swarm-crew-shirt-graphic) · 1344×1792: `$0.01` (~18s)
- [text-to-image](https://runware.ai/docs/models/meta-muse-image/examples#example-equine-ultrasound-campaign-hero) · 2048×1152: `$0.01` (~36s)
- [text-to-image](https://runware.ai/docs/models/meta-muse-image/examples#example-antikythera-exhibition-banner-expansion) · 2048×1152: `$0.01` (~26s)
- [reference-to-image](https://runware.ai/docs/models/meta-muse-image/examples#example-cello-restoration-social-media-ad) · 1344×1792: `$0.01` (~18s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

`resolution` cannot be used with `width/height`.

---

When `resolution` is provided, `inputs.referenceImages` is required.

---

`width` and `height` must be used together.

---

The following dimension combinations are supported:

| Configuration | Dimensions |
| --- | --- |
| `2K (1:1)` | `1600x1600` |
| `2K (3:2)` | `1920x1280` |
| `2K (2:3)` | `1280x1920` |
| `2K (4:3)` | `1792x1344` |
| `2K (3:4)` | `1344x1792` |
| `2K (16:9)` | `2048x1152` |
| `2K (9:16)` | `1152x2048` |
| `2K (21:9)` | `2352x1008` |

## Notes

Muse Image works in two modes, and `inputs.referenceImages` is what switches between them:

- **Text-to-image:** Provide a prompt on its own.
- **Editing and composition:** Provide `inputs.referenceImages` (up to 10). One image is edited according to the prompt. Several are composed into a single output, which also makes this the way to hold a subject or a style steady across a series.

**Sizing:** Pass a `width`/`height` pair from the supported set to pin the frame. With a reference attached, leave the size out and the output follows the reference's aspect ratio instead of one of the fixed pairs, and `resolution` gives the same result. Every pair renders at roughly the same 2.5 megapixels, so the pair chooses shape rather than size.

**Prompting:** The prompt accepts up to 200,000 characters and is required in both modes, including edits, where it describes the change rather than the whole image.

> [!NOTE]
> A reference larger than the requested output is downscaled to fit rather than rejected, so a 6000 px source contributes at roughly 2K. Send references near the output size when fine detail has to survive.

**Built-in tools:** `settings.webSearch`, `settings.imageSearch` and `settings.shell` are all on by default and included in the price. Web search grounds the image in current facts, image search pulls visual references for real places and products, and shell lets the model compute a layout before rendering, which is what makes charts and tables come out structured. Turning web search off also removes the browser tools.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/meta-muse-image#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/meta-muse-image#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/meta-muse-image#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Image output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/meta-muse-image#request-outputformat)

- **Type**: `string`
- **Default**: `JPG`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`JPG\`: Best for photorealistic images with smaller file sizes (no transparency).
- \`PNG\`: Lossless compression, supports high quality and transparency (alpha channel).
- \`WEBP\`: Modern format providing superior compression and transparency support.

> [!NOTE]
> \*\*Transparency\*\*: If you are using features like background removal or LayerDiffuse that require transparency, you must select a format that supports an alpha channel (e.g., \`PNG\`, \`WEBP\`, \`TIFF\`). \`JPG\` does not support transparency.

**Allowed values**: `JPG` `PNG` `WEBP`

### [outputQuality](https://runware.ai/docs/models/meta-muse-image#request-outputquality)

- **Type**: `integer`
- **Min**: `20`
- **Max**: `99`
- **Default**: `95`

Compression quality of the output. Higher values preserve quality but increase file size.

### [webhookURL](https://runware.ai/docs/models/meta-muse-image#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/meta-muse-image#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/meta-muse-image#request-uploadendpoint)

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

### [safety](https://runware.ai/docs/models/meta-muse-image#request-safety)

- **Path**: `safety.checkContent`
- **Type**: `object (1 property)`

Content safety checking configuration for image generation.

#### [checkContent](https://runware.ai/docs/models/meta-muse-image#request-safety-checkcontent)

- **Path**: `safety.checkContent`
- **Type**: `boolean`

Enable or disable content safety checking. Increases total generation time.

### [ttl](https://runware.ai/docs/models/meta-muse-image#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/meta-muse-image#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/meta-muse-image#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `20`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/meta-muse-image#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/meta-muse-image#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/meta-muse-image#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/meta-muse-image#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/meta-muse-image#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/meta-muse-image#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/meta-muse-image#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [referenceImages](https://runware.ai/docs/models/meta-muse-image#request-inputs-referenceimages)

- **Path**: `inputs.referenceImages`
- **Type**: `array of strings`
- **Min items**: `1`
- **Max items**: `10`

List of reference images (UUID, URL, Data URI, or Base64).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/meta-muse-image#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `meta:muse@image`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/meta-muse-image#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `1`
- **Max**: `200000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [width](https://runware.ai/docs/models/meta-muse-image#request-width)

- **Type**: `integer`
- **Default**: `1600`
- **Paired with**: height

Width of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [height](https://runware.ai/docs/models/meta-muse-image#request-height)

- **Type**: `integer`
- **Default**: `1600`
- **Paired with**: width

Height of the generated media in pixels.

**Learn more** (2 resources):

- [Dimensions](https://runware.ai/docs/learn/dimensions) (learn)
- [Image Outpainting: Dimensions Critical For Outpainting](https://runware.ai/docs/learn/image-outpainting#dimensions-critical-for-outpainting) (learn)

### [resolution](https://runware.ai/docs/models/meta-muse-image#request-resolution)

- **Type**: `string`

Resolution preset for the output. When used with input media, automatically matches the aspect ratio from the input.

**Allowed values**: `2K`

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [imageSearch](https://runware.ai/docs/models/meta-muse-image#request-settings-imagesearch)

- **Path**: `settings.imageSearch`
- **Type**: `boolean`
- **Default**: `true`

Pull visual references for real places, products, logos, and styles.

### [moderation](https://runware.ai/docs/models/meta-muse-image#request-settings-moderation)

- **Path**: `settings.moderation`
- **Type**: `string`
- **Default**: `auto`

Content moderation level.

**Allowed values**:

- `auto` Standard filtering that limits creating certain categories of potentially age-inappropriate content.
- `low` Less restrictive filtering.

### [shell](https://runware.ai/docs/models/meta-muse-image#request-settings-shell)

- **Path**: `settings.shell`
- **Type**: `boolean`
- **Default**: `true`

Allow the model to run code to build charts, tables, and structured layouts before rendering.

### [thinkingLevel](https://runware.ai/docs/models/meta-muse-image#request-settings-thinkinglevel)

- **Path**: `settings.thinkingLevel`
- **Type**: `string`
- **Default**: `high`

Controls the depth of internal reasoning the model performs before generating a response.

**Allowed values**: `high` `low`

### [webSearch](https://runware.ai/docs/models/meta-muse-image#request-settings-websearch)

- **Path**: `settings.webSearch`
- **Type**: `boolean`
- **Default**: `true`

Ground the generation in web results for current facts.

**Features**

Standalone addons and post-processing features.

### [watermark](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark)

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

#### [text](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-text)

- **Path**: `advancedFeatures.watermark.text`
- **Type**: `string`
- **Min**: `2`
- **Max**: `32`

Watermark text.

#### [image](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-image)

- **Path**: `advancedFeatures.watermark.image`
- **Type**: `string`

Watermark image (UUID, URL, Data URI, or Base64).

#### [displayPosition](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-displayposition)

- **Path**: `advancedFeatures.watermark.displayPosition`
- **Type**: `string`

Watermark position.

**Allowed values**: `top-left` `top-center` `top-right` `center-left` `center-center` `center-right` `bottom-left` `bottom-center` `bottom-right` `tiled`

#### [opacity](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-opacity)

- **Path**: `advancedFeatures.watermark.opacity`
- **Type**: `float`
- **Min**: `0.1`
- **Max**: `1`
- **Step**: `0.01`

Watermark opacity.

#### [fontColor](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-fontcolor)

- **Path**: `advancedFeatures.watermark.fontColor`
- **Type**: `string`

Text color in hex format.

#### [bgColor](https://runware.ai/docs/models/meta-muse-image#request-advancedfeatures-watermark-bgcolor)

- **Path**: `advancedFeatures.watermark.bgColor`
- **Type**: `string`

Background color in hex format.

## Response Parameters

### [taskType](https://runware.ai/docs/models/meta-muse-image#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `imageInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/meta-muse-image#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [imageUUID](https://runware.ai/docs/models/meta-muse-image#response-imageuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output image.

### [imageURL](https://runware.ai/docs/models/meta-muse-image#response-imageurl)

- **Type**: `string`
- **Format**: `uri`

URL of the output image.

### [imageBase64Data](https://runware.ai/docs/models/meta-muse-image#response-imagebase64data)

- **Type**: `string`

Base64-encoded image data.

### [imageDataURI](https://runware.ai/docs/models/meta-muse-image#response-imagedatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output image.

### [seed](https://runware.ai/docs/models/meta-muse-image#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [NSFWContent](https://runware.ai/docs/models/meta-muse-image#response-nsfwcontent)

- **Type**: `boolean`

Flag indicating if NSFW content was detected.

### [cost](https://runware.ai/docs/models/meta-muse-image#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-image (text-to-image)

![Equine Ultrasound Campaign Hero](https://assets.runware.ai/examples/meta-muse-image/aac2dc0f-091e-42dc-9e7f-86cac878d0d3.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "9a0e1620-9a18-49b4-9b5c-b4f628aecadc",
  "model": "meta:muse@image",
  "positivePrompt": "Create a polished landscape campaign hero for a manufacturer of portable equine veterinary ultrasound equipment. The hero product is a compact, rugged ultrasound console resting on a clean dark-oak examination cart inside an upscale working horse stable. Show credible industrial and medical details: matte charcoal impact-resistant housing, reinforced corners, fold-up high-resolution display, compact sealed keyboard, trackball and control dials, ventilation slots, battery compartment, clearly differentiated probe ports, and one connected linear transducer with its cable arranged naturally beside the unit. The screen displays a realistic grayscale ultrasound scan without patient data, branding, labels, or readable text.\n\nCompose the product prominently in the right-center foreground at a three-quarter angle, fully visible and sharply focused, with enough uncluttered negative space across the left third for a website headline that will be added later. In the softly receding stable aisle, include warm timber stall fronts, black powder-coated hardware, neatly hung leather tack, subtle straw texture, and a calm chestnut horse visible over a stall door in the middle distance. The horse must remain secondary and slightly out of focus. Early morning sunlight enters through high stable windows, creating restrained golden shafts and soft highlights along the equipment edges, balanced by cool neutral shadows. Premium editorial product photography, authentic equine-clinic atmosphere, natural materials, precise perspective, physically plausible reflections, realistic cable behavior, subtle dust in the light, refined warm brown, charcoal, cream, and muted sage palette. Sophisticated, calm, dependable mood. No people, no logos, no invented brand, no promotional lettering, no watermark, no border, no UI overlay, and no excessive props.",
  "width": 2048,
  "height": 1152,
  "settings": {
    "imageSearch": true,
    "shell": true,
    "thinkingLevel": "high",
    "webSearch": true
  }
}
```

---

### text-to-image (text-to-image)

![Antikythera Exhibition Banner Expansion](https://assets.runware.ai/examples/meta-muse-image/c4377922-0cf9-4784-9e6b-57584e678f30.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "b047ff5e-17ae-4050-b340-2d17e1d40867",
  "model": "meta:muse@image",
  "positivePrompt": "Create a finished 16:9 panoramic hero image for a maritime archaeology museum exhibition about the Antikythera mechanism. Compose it as a seamless outpaint-style expansion of an imagined tight vertical conservation photograph: preserve a visually concentrated central zone containing the mechanism’s largest corroded bronze gear fragment, then extend the scene naturally far to both sides with a wider conservation workbench and generous negative space. Show the authentic artifact as irregular, mineral-encrusted bronze fragments with recognizable intermeshing gear teeth, fractured plates, green verdigris, brown corrosion and limestone accretions—scientifically plausible, precious and fragile, not a polished fantasy clock. Top-down oblique museum conservation photography, artifact supported on archival charcoal-gray foam inside a shallow neutral tray. Expand leftward into neatly arranged conservation tools: bamboo picks, a soft sable brush, calipers, cotton gloves and small unlabeled specimen dishes. Expand rightward into partially visible X-radiography films on a cool cyan light pad, archival tissue and additional tiny bronze fragments, all spatially coherent with the central scene. Deliberate horizontal visual rhythm, the main gear fragment near center-right, uncluttered dark slate negative space across the left third for later curatorial typography, but include no text in the image. Controlled raking light reveals gear teeth and layered corrosion; restrained palette of oxidized bronze, verdigris green, graphite gray and muted cyan. Premium institutional editorial art direction, forensic accuracy, subtle shadows, crisp micro-detail, realistic materials, seamless perspective and lighting across the expanded frame. No people, no modern electronics, no decorative steampunk additions, no invented complete machine, no labels, captions, logos, borders, crop marks, UI or mockup presentation.",
  "width": 2048,
  "height": 1152,
  "settings": {
    "imageSearch": true,
    "shell": true,
    "thinkingLevel": "high",
    "webSearch": true
  }
}
```

---

### reference-to-image (reference-to-image)

![Beekeeping Swarm Crew Shirt Graphic](https://assets.runware.ai/examples/meta-muse-image/ada4e940-f6be-4221-a6e0-56fb7af168cf.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "923836ae-b8fa-4641-89a4-e0a0a4e39cbb",
  "model": "meta:muse@image",
  "positivePrompt": "Transform the supplied beekeeper-smoker photograph into a finished portrait-format apparel graphic for a volunteer urban beekeeping swarm-response crew. Preserve the reference object's recognizable conical spout, wire heat guard, wooden bellows, proportions, dents, scratches, and soot marks so this specific piece of field equipment remains identifiable. Redraw it as a bold central woodcut emblem with chunky carved contours and controlled halftone texture. Add curling smoke above it that forms a clean arched frame containing a few small, anatomically credible honeybees in flight. Place the exact headline “SWARM RESPONSE CREW” in large condensed block lettering across the top. Add the exact smaller line “CALL • COLLECT • REHOME” beneath the smoker and the exact year “2026” at the bottom. Use a practical municipal-workwear aesthetic rather than a cute honey-brand look: strong symmetry, assertive hierarchy, rugged but polished linework, and excellent readability from a distance. Limit the design to four flat screen-print inks: carbon black, safety orange, golden ochre, and warm cream. Create believable overprint and slightly distressed ink texture while keeping all lettering crisp and correctly spelled. Center the composition with generous negative space and clean outer edges for easy garment printing. Deliver the standalone flat print artwork on a solid warm-cream background, not a shirt mockup, not a product photograph, no person, no hanger, no fabric folds, no extra logos, and no additional text.",
  "width": 1344,
  "height": 1792,
  "settings": {
    "imageSearch": false,
    "shell": false,
    "thinkingLevel": "low",
    "webSearch": false
  },
  "inputs": {
    "referenceImages": [
      "https://assets.runware.ai/assets/inputs/a2946e24-67ff-4371-a476-1c8cad698a7b.jpg"
    ]
  }
}
```

---

### reference-to-image (reference-to-image)

![Cello Restoration Social Media Ad](https://assets.runware.ai/examples/meta-muse-image/a4c995b7-260f-4971-9387-a682efe3a91d.jpg)

```json
{
  "taskType": "imageInference",
  "taskUUID": "38d1fb35-74e3-450c-8790-23404e06ca85",
  "model": "meta:muse@image",
  "positivePrompt": "Create a finished 1344x1792 portrait social media advertisement for a specialist cello restoration workshop using the supplied reference image. Preserve the exact recognizable cello: its silhouette, proportions, scroll, pegs, strings, bridge, f-holes, reddish-brown varnish, distinctive grain, and visible crack location. Do not replace it with a different instrument. Transform the plain source photo into a refined editorial still life inside a traditional luthier's workshop. Position the cello upright as the dominant central subject, extending from the lower edge toward the upper third, with generous negative space above and to the left for typography. Show the damaged area treated with a subtle luminous gold repair seam, elegant and technically plausible rather than magical, communicating that an heirloom has been restored instead of discarded. Surround it sparingly with authentic hand tools on a dark wooden bench: one small finger plane, a spool of linen thread, a maple shaving, and a folded cream cloth. Background cabinetry should fall into soft shadow with no distracting clutter. Art direction: contemporary luxury campaign meets Dutch still-life photography; deep oxblood, burnt umber, amber, charcoal, and warm ivory palette; directional window light from upper left; crisp detail on the cello and repair; soft atmospheric falloff; tactile wood, varnish, and cloth textures; quiet, assured, emotionally resonant mood. Add clean, accurately rendered editorial typography integrated into the composition, not inside a button. Large warm-ivory serif headline in the upper-left negative space: “DON’T RETIRE THE SOUND.” Beneath it, much smaller uppercase sans-serif copy: “CELLO RESTORATION · SPRING INTAKE”. At the bottom-left, add a restrained line: “Repair the instrument. Keep the history.” Maintain generous margins and a clear mobile-feed hierarchy. No logo, no invented company name, no website, no price, no app interface, no social media UI, no mockup frame, no extra instruments, no musicians, and no fantasy effects.",
  "width": 1344,
  "height": 1792,
  "settings": {
    "imageSearch": false,
    "shell": false,
    "thinkingLevel": "low",
    "webSearch": false
  },
  "inputs": {
    "referenceImages": [
      "https://assets.runware.ai/assets/inputs/5fe95bf0-3261-404c-b3c5-ad2549346e08.jpg"
    ]
  }
}
```