---
title: Model Schemas | Runware Docs
url: https://runware.ai/docs/models-api/model-schemas
description: One request shape covers every combination of inputs a model accepts. Learn how to build on a model's schema, and how to map it to separate modes.
relatedDocuments:
  - https://runware.ai/docs/models-api/introduction
  - https://runware.ai/docs/models-api/pricing
  - https://runware.ai/docs/models-api/task-polling
---
## Introduction

Models take more kinds of input with every generation. A video model used to do one thing, text to video or image to video. Current ones accept reference images, videos and audio **combined in one request**, and each release adds combinations.

A product that lists each mode as a separate item has no place for those requests. A first frame is image to video and reference images are reference to video, and **a request that carries both fits neither item**. It needs an item of its own, and so does every combination a new model adds.

Runware describes a request by its inputs instead, which is **the shape models are converging on**. A model is one item, and you send it the inputs you have in any combination it accepts. Every request goes to the same endpoint, and a combination the model gains later is already covered by the integration you wrote.

That integration is written against **the schema each model publishes**, one JSON document stating which inputs and parameters the model accepts and how they combine. Parameters keep the same name and role across models, so a first frame is always `inputs.frameImages`, whoever made the model, and code that reads one schema reads all of them.

The sections below lay out a request and show how to build on the schema, with [MiniMax H3](https://runware.ai/docs/models/minimax-h3) as the example. If you are migrating from a provider that gives each mode its own endpoint, or your product is organized around separate modes and you are keeping them for now, [Working with separate modes](#working-with-separate-modes) maps them onto the same schema.

> [!NOTE]
> Hand this page to your agent
> 
> This page is written to be given to a coding agent. Pass it the [Markdown version](https://runware.ai/docs/models-api/model-schemas.md) along with the model you want to integrate, and it has what it needs to do the work: every step reads from a public JSON document and ends in a request it can validate.

## Where the schema lives

The schemas are served as JSON from **two endpoints**.

| Endpoint | What it holds |
| --- | --- |
| `/docs/models/index.json` | Every public model with its id, AIR, status, capabilities, and the URL of its schema. |
| `/docs/models/<model>/schema.json` | The OpenAPI document for one model. |

Start from the index and follow `schema`.

```bash
curl -s https://runware.ai/docs/models/index.json | jq '.[] | select(.id == "minimax-h3")'
```

```json
{
  "id": "minimax-h3",
  "air": "minimax:h3@0",
  "name": "MiniMax H3",
  "status": "live",
  "capabilities": ["text-to-video", "image-to-video", "video-to-video", "audio-to-video", "edit", "checkpoint", "audio"],
  "schema": "https://runware.ai/docs/models/minimax-h3/schema.json"
}
```

The document that `schema` points at carries everything else an integration reads.

| Location | What it holds |
| --- | --- |
| `info["x-air-id"]` | The value to send as `model`. |
| `info["x-status"]` | `live`, `coming-soon` or `deprecated`. A deprecated model also carries `x-deactivates-at` and `x-replaced-by`. |
| `info["x-modes"]` | Every combination of inputs the model accepts. |
| `info["x-examples"]` | The URL of the model's published examples, each with the request that produced it. |
| `info["x-pricing"]` | Rates and measured runs. [Pricing](https://runware.ai/docs/models-api/pricing) explains the shapes. |
| `components.schemas.RequestBody` | The request, an array of tasks. `items` is the JSON Schema of one task, called the task schema on this page. |
| `components.schemas.ResponseBody` | The response. `properties.data.items` is the JSON Schema of one result. |

## The layers of a request

A task is a flat object with a few nested ones, and nearly every parameter sits in **one of four layers**. The layer tells you how much that parameter varies from one model to the next.

```json
[
  {
    "taskType": "videoInference",
    "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
    "includeCost": true,
    "model": "minimax:h3@0",
    "positivePrompt": "The worker lifts the mature wasabi root smoothly into view.",
    "resolution": "768p",
    "duration": 8,
    "inputs": {
      "frameImages": ["https://assets.runware.ai/assets/inputs/93bc5eb1-8f3d-4c56-9232-06f3da7dce28.jpg"]
    },
    "settings": {
      "promptExpansion": "disabled"
    }
  }
]
```

1. **API** — General to the API and the same on every model. They control delivery and output, such as `deliveryMethod`, `webhookURL`, `outputFormat` and `numberResults`.
2. **Core** — The main parameters, shared across models under the same name, such as `model`, `positivePrompt`, `width`, `height`, `duration` and `seed`. Each model has the ones that apply to it, with its own ranges.
3. **Inputs** — Every asset the model works from, inside `inputs`. Models differ in which media they take, alone or combined.
4. **Settings** — Finer tuning options, inside `settings`. They are specific to a model, a family or a provider, with no single set common to the whole catalog.

A model declares only the parameters it supports, and **anything it does not declare is rejected**. H3 has no `steps` and no `negativePrompt`, so neither appears in its schema and neither can be sent.

## Inputs

Every asset the model works from goes in this object. Each key has **one role across the whole catalog**, so the key alone tells you what the model does with the asset.

| Input | Role |
| --- | --- |
| `referenceImages` | Images that guide the result without fixing any frame, such as a subject or a style. |
| `referenceVideos` | Clips the model takes as a reference for the subject or the motion. |
| `referenceAudios` | Audio the model takes as a reference, such as a voice. |
| `frameImages` | Images pinned to a position on the video timeline. One image is the first frame, two are the first and the last. |
| `image`, `video`, `audio` | The single asset the task operates on: the image to upscale, the video to edit, the audio a video follows. |
| `images`, `videos`, `audios`, `documents` | Several assets read together, mostly attachments for a text model. |
| `seedImage` | The image an image-to-image generation starts from. |
| `maskImage` | The area of the source image to repaint. White is edited and black is preserved. |

Almost every model in the catalog is **described by these inputs alone**, which is what lets one integration cover them all. That consistency never comes at the cost of the developer experience. When a model needs an input of its own, or a different shape for one of these, **its schema defines it**.

Each input **declares its own limits**. H3 takes up to 9 `referenceImages` and up to 2 `frameImages`, both stated as `maxItems` in its schema.

## Building on the schema

The direct integration is **one interface per model**. Each input becomes an upload slot, each core and settings parameter becomes a control, and the schema decides which requests are valid. The API layer stays in your backend, set once for every model.

Every combination the model accepts is available from that one interface. The user fills the slots they need, and **nothing has to be decided per mode**.

### Rules between parameters

Most parameters are valid whatever inputs are sent. H3 has 20 parameters outside `inputs`, and **only three depend on anything else**: `width`, `height` and `resolution`. What ties them to the inputs is a set of rules under `allOf`, at the root of the task schema.

The rules are standard JSON Schema, and nearly all of them take **one of five shapes**.

| Keyword | Meaning | Example |
| --- | --- | --- |
| `dependentRequired` | Parameters that travel together. | `width` requires `height`, and `height` requires `width`. |
| `not` | Parameters that cannot be sent together. | `resolution` with `width` or `height`. |
| `if` / `then` | One parameter being present requires, forbids or narrows another. | With `frameImages`, `width` and `height` are forbidden. |
| `if` / `then` / `else` | The same, with a second rule for when the condition is not met. | With `frameImages`, `width` and `height` are forbidden. Without it, both are required. |
| `oneOf` | A closed list of alternatives. | The `width` and `height` pairs the model accepts. |

This is the H3 rule that removes the size parameters once a frame is sent.

```json
{
  "if": {
    "properties": { "inputs": { "required": ["frameImages"] } },
    "required": ["inputs"]
  },
  "then": {
    "not": {
      "anyOf": [{ "required": ["width"] }, { "required": ["height"] }]
    }
  }
}
```

The condition is **a presence check**. `if` matches when `inputs.frameImages` is sent, and `then` rejects the request if `width` or `height` came with it. With a frame, the output takes its aspect ratio from the image and `resolution` sets its size.

### Dimensions

Many models accept **a closed list of width and height pairs** instead of any size. The list is a `oneOf`, and each entry carries a `title` you can show as a label. This is the start of the H3 list.

```json
{
  "oneOf": [
    { "title": "768p (~16:9)", "properties": { "width": { "const": 1344 }, "height": { "const": 768 } } },
    { "title": "768p (4:3)", "properties": { "width": { "const": 1024 }, "height": { "const": 768 } } },
    { "title": "768p (1:1)", "properties": { "width": { "const": 768 }, "height": { "const": 768 } } }
  ]
}
```

A size outside the list is rejected, and on H3 that includes `1920 × 1080`.

### Limits on the media you send

JSON Schema can check that `referenceVideos` is a list of URLs. It cannot check how long each clip runs. **Limits that depend on the file itself** are in `x-constraints`, at the root of the schema.

```json
[
  {
    "operation": "duration",
    "parameters": ["inputs.referenceVideos", "inputs.referenceAudios"],
    "minimum": 2,
    "maximum": 15,
    "scope": "each"
  },
  {
    "operation": "duration",
    "parameters": ["inputs.referenceVideos", "inputs.referenceAudios"],
    "maximum": 15
  }
]
```

On H3 each reference clip runs between 2 and 15 seconds, and the clips of one request add up to 15 seconds at most. Every entry is built from **the same fields**.

| Field | Meaning |
| --- | --- |
| `operation` | What is measured. One of the operations listed below. |
| `parameters` | The parameters the limit applies to, as paths. `inputs.referenceVideos` bounds a file, and `width` with `height` bounds the output size. |
| `minimum`, `maximum` | The bounds, in the unit of the operation. |
| `allowed` | A closed list of accepted values, used in place of the bounds. |
| `scope` | `each` bounds every file on its own. Without it, the entry bounds all the listed files together. |
| `when` | Makes the entry conditional. It applies only when every parameter in `present` is sent and every parameter in `absent` is left out. |

`operation` takes **one of these values**.

| Operation | What it measures | Unit |
| --- | --- | --- |
| `width` | Width of the output or of an input file | pixels |
| `height` | Height of the output or of an input file | pixels |
| `area` | Pixel count, width times height | pixels |
| `ratio` | Width divided by height | none |
| `duration` | Length of a video or audio file | seconds |
| `frameRate` | Frame rate of a video | frames per second |
| `pageCount` | Pages in a document | count |
| `fileCount` | Files across the listed inputs | count |
| `fileFormat` | Accepted file types, listed in `allowed` | none |
| `fileSize` | Size of a file | bytes |
| `archiveUnzippedSize` | Size of an archive once unpacked | bytes |
| `archiveFileCount` | Files inside an archive | count |
| `archiveImageCount` | Valid images inside an archive | count |

A JSON Schema validator skips `x-constraints`, so a request can pass validation and still break one of these limits. **Check them in your own code** before uploading a file.

### Validating a request

A JSON Schema validator **applies every rule at once**, so an interface can check a request as the user fills it in and before anything is sent.

```javascript
import Ajv from 'ajv/dist/2020.js'

const response = await fetch('https://runware.ai/docs/models/minimax-h3/schema.json')
const document = await response.json()
const task = document.components.schemas.RequestBody.items

const compile = (schema) => new Ajv({ strict: false, validateFormats: false }).compile(schema)
const validate = compile(task)

const base = {
  taskType: 'videoInference',
  taskUUID: crypto.randomUUID(),
  model: 'minimax:h3@0',
  positivePrompt: 'The worker lifts the mature wasabi root smoothly into view.',
}

const firstFrame = {
  frameImages: ['https://assets.runware.ai/assets/inputs/93bc5eb1-8f3d-4c56-9232-06f3da7dce28.jpg'],
}

validate({ ...base, width: 1344, height: 768 })                      // true
validate({ ...base, resolution: '768p', inputs: firstFrame })        // true
validate({ ...base, width: 1344, height: 768, inputs: firstFrame })  // false
```

One schema accepts the request with a frame and the request without one, and rejects the size parameters that do not fit the inputs sent. The rejected call reports `schemaPath`, **the location of the rule that failed**. Here it is `#/allOf/3/then/not`, the fourth rule of the task. An interface can use it to disable the control, and an agent can read the rule and correct the payload.

> [!NOTE]
> Use a validator, not a parser of your own
> 
> The rules can be read by hand, but a parser has to be taught every shape a rule can take, and it breaks the day a model uses one it has never seen. A validator already implements the whole standard, so it stays right for every model, including the ones released later.

A few models tie one value to another, so validate the final request with every value in place. A model can offer durations of 5, 8 and 10 seconds and sizes up to 1080p, and still take **5 or 8 seconds only** at 1080p.

The [TypeScript](https://runware.ai/docs/tools/typescript) and [Python](https://runware.ai/docs/tools/python) SDKs run this validation for you, on every request, before it leaves your machine.

### Published examples

Every model also publishes examples, each with **the request that produced it**. `info["x-examples"]` links to them, and they are there when you would rather adapt a request that ran than write one from an empty object.

The published requests leave out `taskUUID`, so add one before sending.

## Working with separate modes

Some products list text to video and image to video as separate items. That layout comes from models that did one thing each, and it maps onto the same schema: **each item is one combination of inputs** of the same model. This section is for anyone migrating from a provider with one endpoint per mode, and for a product that has that layout and keeps it for now.

### The modes of a model

`info["x-modes"]` lists **every valid combination** of inputs. The list is shortened here.

```bash
curl -s https://runware.ai/docs/models/minimax-h3/schema.json | jq '.info["x-modes"]'
```

```json
[
  { "id": "text", "inputs": [] },
  { "id": "reference-images", "inputs": ["referenceImages"] },
  { "id": "first-frame", "inputs": ["frameImages"], "frames": ["first"] },
  { "id": "first-last-frames", "inputs": ["frameImages"], "frames": ["first", "last"] },
  { "id": "reference-videos", "inputs": ["referenceVideos"] },
  { "id": "reference-images+reference-videos", "inputs": ["referenceImages", "referenceVideos"] }
]
```

| Field | Meaning |
| --- | --- |
| `id` | The inputs of the mode in kebab case, joined with `+`. `text` means no inputs at all. |
| `inputs` | The keys of `inputs` that the mode sends. |
| `frames` | Present on frame modes with fixed positions. The positions `frameImages` fills, so one input yields `first-frame` and `first-last-frames`. |
| `frameCount` | Present on `multiple-frames`, the mode for three or more frames at positions you choose. It gives the `minimum` and `maximum` number of images. |
| `values` | Present when a field inside an input changes what the input combines with. It names that field and the value the mode needs, and the `id` carries the value after a colon, as in `reference-videos:extend`. |

The list already applies the rules between inputs. H3 has no mode that mixes `frameImages` with a reference input, and no `reference-audios` mode on its own, because an audio reference **needs an image or video reference** next to it. A combination missing from the list is one the model rejects.

### Choosing which modes to expose

Every entry is a valid request, so the full list is the whole menu. **One item per mode** is the predictable choice: each item maps to one `id` and shows exactly the inputs it sends.

For fewer items, keep the modes whose inputs are **not contained in a larger one**, and treat their inputs as optional.

```javascript
const modes = document.info['x-modes']

const widest = modes.filter((mode) => (
  !modes.some((other) => (
    other.inputs.length > mode.inputs.length
    && mode.inputs.every((input) => (other.inputs.includes(input)))
  ))
))
```

On H3 that leaves three: `first-frame`, `first-last-frames`, and `reference-images+reference-videos+reference-audios`.

> [!WARNING]
> A narrower mode can accept parameters the wider one rejects. H3 takes `width` and `height` in `text` mode and forbids them once `frameImages` is present, so an item that folds `text` into `first-frame` has to switch its size controls when the frame is left empty.

The list keeps growing as models accept more combinations, and each new one is another item to add. The interface described in [Building on the schema](#building-on-the-schema) **takes them all and stays the same**.

### Parameters of a mode

Most of the schema is the same in every mode. **A few parameters are ruled out**, and those few are all that separates one mode from the next.

| Group | Which parameters | How they behave |
| --- | --- | --- |
| **Inputs** | The keys listed in `inputs` on the `x-modes` entry. | Exactly those, and no other input. |
| **Parameters that change** | The ones named in a rule of the schema. Usually the size parameters. | Allowed, required or forbidden depending on the mode. |
| **Everything else** | The rest of the schema. | The same in every mode. |

On H3 the three parameters named in a rule are **the only difference between its nine modes**.

| Modes of H3 | Size parameters |
| --- | --- |
| `text` | `width` and `height` |
| `first-frame`, `first-last-frames` | `resolution` |
| Every mode with a reference input | `resolution`, or `width` and `height` |

Many models have no rule at all, and each of their modes takes the full schema. For the rest, collecting the names that appear under `allOf` gives **the short list to resolve**. Not every name on it changes with the mode, and the test in the next section settles which ones do.

The snippets in this section reuse `document`, `task`, `compile`, `base` and `firstFrame` from [Validating a request](#validating-a-request).

```javascript
const ruleNames = (node, names = new Set()) => {
  if (Array.isArray(node)) {
    node.forEach((item) => ruleNames(item, names))
  } else if (node && typeof node === 'object') {
    for (const name of node.required ?? []) { names.add(name) }
    for (const name of Object.keys(node.properties ?? {})) { names.add(name) }
    Object.values(node).forEach((value) => ruleNames(value, names))
  }

  return names
}

const changing = [...ruleNames(task.allOf)]
  .filter((name) => (name !== 'inputs' && name in task.properties))

// ['resolution', 'width', 'height']
```

### A schema per mode

Adding one constraint to the task schema **narrows it to a single mode**: the inputs of the mode become required and every other input is ruled out. The result is a regular JSON Schema, so a product with one item per mode can store one of these next to each item.

```javascript
const modeSchema = (mode) => {
  const pinned = {}

  const count = (mode.frames)
    ? { minimum: mode.frames.length, maximum: mode.frames.length }
    : mode.frameCount

  if (count) {
    pinned.frameImages = { minItems: count.minimum, maxItems: count.maximum }
  }

  for (const [path, value] of Object.entries(mode.values ?? {})) {
    const [input, field] = path.split('.')
    pinned[input] = { items: { required: [field], properties: { [field]: { const: value } } } }
  }

  return {
    allOf: [task, {
      required: (mode.inputs.length) ? ['inputs'] : [],
      properties: {
        inputs: { required: mode.inputs, maxProperties: mode.inputs.length, properties: pinned },
      },
    }],
  }
}

const firstFrameMode = modes.find((mode) => (mode.id === 'first-frame'))
const validateMode = compile(modeSchema(firstFrameMode))

const request = { ...base, inputs: firstFrame }

validateMode({ ...request, resolution: '768p' })           // true
validateMode({ ...request, width: 1344, height: 768 })     // false
validateMode({ ...base, width: 1344, height: 768 })        // false
```

The first two calls **settle the size parameters** of `first-frame`: it takes `resolution` and rejects `width` and `height`. The third is a valid `text` request, and the mode schema rejects it because the frame is missing. **Run the same test for each changing parameter** against each mode and you have the form of every mode, with no rule read by hand.

### A working request per mode

The mode schema also picks **the published example that fits the mode**, which gives each item a request that already ran.

```javascript
const { examples } = await (await fetch(document.info['x-examples'])).json()

const starting = examples.find((example) => (
  validateMode({ ...example.request, taskUUID: crypto.randomUUID() })
))

// starting.request: { taskType, model, positivePrompt, resolution, duration, inputs }
```

A mode with no published example starts from the closest one, with its inputs swapped for the inputs of the mode.

## Integrating a new model

1. **Find the model.** Read `/docs/models/index.json`, match on `id` or `air`, and fetch the document at `schema`.
2. **Check its status.** Skip a model that is not `live`. For a `deprecated` one, `x-deactivates-at` is the date it stops working and `x-replaced-by` is the model to move to.
3. **Build the interface from the task schema.** Give each input an upload slot and each core and settings parameter a control.
4. **Write the request.** The examples in `info["x-examples"]` are there if you want one that already ran to start from.
5. **Validate every request** against the task schema, as it is filled in and before sending it.
6. **Send it.** Long-running tasks use `"deliveryMethod": "async"` with [task polling](https://runware.ai/docs/models-api/task-polling) or [webhooks](https://runware.ai/docs/models-api/webhooks).
7. **Read the result** with the shape in `ResponseBody`. Set `includeCost` to get the charge back with it.

A product with separate modes adds two steps after the second: pick its items from `info["x-modes"]`, then build a schema per mode and validate against that one.

Every step reads from the schema, so the same code path **covers the next model** the day its schema is published.