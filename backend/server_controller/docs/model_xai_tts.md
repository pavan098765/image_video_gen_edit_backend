---
title: xAI Text-to-Speech | Runware Docs
url: https://runware.ai/docs/models/xai-tts
description: Expressive text-to-speech with over two dozen voices, speech tags, and multilingual support
---
# xAI Text-to-Speech

![xAI Text-to-Speech](https://assets.runware.ai/covers/xai-tts.jpg)

xAI Text-to-Speech converts text into natural-sounding spoken audio with a single API call. It offers more than two dozen expressive voices, inline speech tags for fine-grained control over pauses, laughter, whispers, and emphasis, and supports over 20 auto-detected languages.

- **Status**: live
- **ID**: `xai:tts@0`
- Try in Playground
- **Creator**: xAI
- **Release Date**: March 16, 2026
- **Zero Data Retention**: Yes
- **Capabilities**: Text to Audio, Checkpoint

## Pricing

$0.000015 per character · ~5s per request

**4 example generations**

- [text-to-audio](https://runware.ai/docs/models/xai-tts/examples#example-antarctic-research-station-pa-announcement): `$0.00583` (~5s)
- [text-to-audio](https://runware.ai/docs/models/xai-tts/examples#example-grid-resilience-podcast-cold-open): `$0.00612` (~5s)
- [text-to-audio](https://runware.ai/docs/models/xai-tts/examples#example-forensic-arboriculture-podcast-intro): `$0.00612` (~5s)
- [text-to-audio](https://runware.ai/docs/models/xai-tts/examples#example-restaurant-oyster-shell-recycling-explainer): `$0.00791` (~6s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/xai-tts#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/xai-tts#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/xai-tts#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Audio output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/xai-tts#request-outputformat)

- **Type**: `string`
- **Default**: `MP3`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`MP3\`: Compressed audio, smaller file size.
- \`WAV\`: Uncompressed, high-quality audio.
- \`FLAC\`: Lossless compression.
- \`OGG\`: Open-source compressed audio format (Vorbis codec).

**Allowed values**: `MP3` `WAV` `FLAC` `OGG`

### [audioSettings](https://runware.ai/docs/models/xai-tts#request-audiosettings)

- **Path**: `audioSettings.bitrate`
- **Type**: `object (3 properties)`

Audio encoding settings for controlling the bitrate, number of channels, and sample rate of the generated audio. Only applicable for lossy output formats (`MP3` and `OGG`). When using lossless formats (`WAV` or `FLAC`), this parameter must not be provided.

The available sample rates and valid bitrate ranges depend on the output format. For `OGG`, bitrate limits also vary by the number of channels.

**MP3 bitrate limits**:

Bitrate limits for MP3 are the same regardless of mono or stereo.

| Sample Rate | Min Bitrate | Max Bitrate |
| --- | --- | --- |
| 8,000 Hz | 8 kbps | 64 kbps |
| 11,025 Hz | 8 kbps | 64 kbps |
| 12,000 Hz | 8 kbps | 64 kbps |
| 16,000 Hz | 8 kbps | 160 kbps |
| 22,050 Hz | 8 kbps | 160 kbps |
| 24,000 Hz | 8 kbps | 160 kbps |
| 32,000 Hz | 32 kbps | 320 kbps |
| 44,100 Hz | 32 kbps | 320 kbps |
| 48,000 Hz | 32 kbps | 320 kbps |

**OGG bitrate limits — Mono (1 channel)**:

| Sample Rate | Min Bitrate | Max Bitrate |
| --- | --- | --- |
| 8,000 Hz | 8 kbps | 40 kbps |
| 12,000 Hz | 16 kbps | 48 kbps |
| 16,000 Hz | 16 kbps | 96 kbps |
| 24,000 Hz | 16 kbps | 80 kbps |
| 48,000 Hz | 32 kbps | 224 kbps |

**OGG bitrate limits — Stereo (2 channels)**:

| Sample Rate | Min Bitrate | Max Bitrate |
| --- | --- | --- |
| 8,000 Hz | 16 kbps | 80 kbps |
| 12,000 Hz | 16 kbps | 96 kbps |
| 16,000 Hz | 24 kbps | 192 kbps |
| 24,000 Hz | 32 kbps | 160 kbps |
| 48,000 Hz | 48 kbps | 256 kbps |

> [!NOTE]
> **Lossless formats**: When `outputFormat` is set to `WAV` or `FLAC`, the `audioSettings` parameter is not available since these formats produce uncompressed or lossless audio with no configurable encoding settings.

#### [bitrate](https://runware.ai/docs/models/xai-tts#request-audiosettings-bitrate)

- **Path**: `audioSettings.bitrate`
- **Type**: `integer`
- **Min**: `8`

Audio bitrate in kbps.

#### [channels](https://runware.ai/docs/models/xai-tts#request-audiosettings-channels)

- **Path**: `audioSettings.channels`
- **Type**: `integer`
- **Default**: `2`

Number of audio channels. 1 for mono, 2 for stereo.

**Allowed values**: `1` `2`

#### [sampleRate](https://runware.ai/docs/models/xai-tts#request-audiosettings-samplerate)

- **Path**: `audioSettings.sampleRate`
- **Type**: `integer`

Audio sample rate in Hz.

### [webhookURL](https://runware.ai/docs/models/xai-tts#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/xai-tts#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/xai-tts#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/xai-tts#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/xai-tts#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/xai-tts#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `4`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/xai-tts#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/xai-tts#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/xai-tts#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/xai-tts#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/xai-tts#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/xai-tts#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/xai-tts#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/xai-tts#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `xai:tts@0`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [speech](https://runware.ai/docs/models/xai-tts#request-speech)

- **Path**: `speech.text`
- **Type**: `object (3 properties)`
- **Required**: true

Settings for speech generation.

#### [text](https://runware.ai/docs/models/xai-tts#request-speech-text)

- **Path**: `speech.text`
- **Type**: `string`
- **Required**: true

Text to convert to speech.

#### [voice](https://runware.ai/docs/models/xai-tts#request-speech-voice)

- **Path**: `speech.voice`
- **Type**: `string`
- **Required**: true
- **Default**: `eve`

Voice identifier to use for speech generation.

**Allowed values**: `carina` `zagan` `helix` `orion` `luna` `iris` `altair` `zenith` `perseus` `helios` `lux` `kepler` `rigel` `cosmo` `celeste` `ursa` `sirius` `lumen` `castor` `naksh` `atlas` `una` `ara` `eve` `leo` `rex` `sal`

#### [language](https://runware.ai/docs/models/xai-tts#request-speech-language)

- **Path**: `speech.language`
- **Type**: `string`
- **Min**: `1`

Language code for speech generation.

**Allowed values**: `en` `ar-EG` `ar-SA` `ar-AE` `bn` `zh` `fr` `de` `hi` `id` `it` `ja` `ko` `pt-BR` `pt-PT` `ru` `es-MX` `es-ES` `tr` `vi`

## Response Parameters

### [taskType](https://runware.ai/docs/models/xai-tts#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/xai-tts#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [audioUUID](https://runware.ai/docs/models/xai-tts#response-audiouuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output audio.

### [audioURL](https://runware.ai/docs/models/xai-tts#response-audiourl)

- **Type**: `string`
- **Format**: `uri`

URL of the output audio.

### [audioBase64Data](https://runware.ai/docs/models/xai-tts#response-audiobase64data)

- **Type**: `string`

Base64-encoded audio data.

### [audioDataURI](https://runware.ai/docs/models/xai-tts#response-audiodatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output audio.

### [seed](https://runware.ai/docs/models/xai-tts#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [cost](https://runware.ai/docs/models/xai-tts#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/xai-tts/e438c811-2f3d-4e7b-b988-57bb33480862.mp3)

> **Prompt**: The storm began ninety-three million miles away. By sunrise, charged particles were pressing against Earth’s magnetic field—and grid operators were watching the needles move. Welcome to The Long Circuit, the podcast about the hidden systems that keep modern life running. Today: how a burst of solar plasma becomes a problem for transformers, satellites, and the people responsible for keeping the lights on.

```json
{
  "taskType": "audioInference",
  "taskUUID": "4f13fed5-2310-4c44-8d41-0a61cb97efd7",
  "model": "xai:tts@0",
  "speech": {
    "text": "The storm began ninety-three million miles away. By sunrise, charged particles were pressing against Earth’s magnetic field—and grid operators were watching the needles move. Welcome to The Long Circuit, the podcast about the hidden systems that keep modern life running. Today: how a burst of solar plasma becomes a problem for transformers, satellites, and the people responsible for keeping the lights on.",
    "voice": "orion",
    "language": "en"
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/xai-tts/23303f66-9dac-46eb-9a24-3a561a538305.mp3)

> **Prompt**: A fallen branch can look like bad luck. To a forensic arborist, it is testimony. The split wood, the hidden decay, even the direction of the wind can reveal what happened—and whether it could have been prevented. Welcome to Rings of Evidence, the podcast where urban trees become witnesses. In this episode, we examine a century-old plane tree, a crowded pavement, and the warning signs everyone walked past.

```json
{
  "taskType": "audioInference",
  "taskUUID": "ae938bee-4c14-4a31-b768-b26e7eba27a5",
  "model": "xai:tts@0",
  "speech": {
    "text": "A fallen branch can look like bad luck. To a forensic arborist, it is testimony. The split wood, the hidden decay, even the direction of the wind can reveal what happened—and whether it could have been prevented. Welcome to Rings of Evidence, the podcast where urban trees become witnesses. In this episode, we examine a century-old plane tree, a crowded pavement, and the warning signs everyone walked past.",
    "voice": "orion",
    "language": "en"
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/xai-tts/ecc966d4-0d41-41b3-be5e-2b9ea74d3790.mp3)

> **Prompt**: Attention, winter-over team. Katabatic winds are now exceeding seventy knots, and visibility beyond the eastern marker is near zero. Exterior travel is suspended until further notice. Please secure all loose equipment, confirm your radio check, and report to the central module by nineteen hundred hours. The storm may sound close tonight, but this station was built for it—and so were we.

```json
{
  "taskType": "audioInference",
  "taskUUID": "d0a168d4-b925-4414-8b9c-d2946d0460fe",
  "model": "xai:tts@0",
  "speech": {
    "text": "Attention, winter-over team. Katabatic winds are now exceeding seventy knots, and visibility beyond the eastern marker is near zero. Exterior travel is suspended until further notice. Please secure all loose equipment, confirm your radio check, and report to the central module by nineteen hundred hours. The storm may sound close tonight, but this station was built for it—and so were we.",
    "voice": "orion",
    "language": "en"
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/xai-tts/90bf0e71-3a89-44eb-87f4-33fd9acd3efb.mp3)

> **Prompt**: Tonight’s oysters can help protect tomorrow’s shoreline. Instead of sending empty shells to a landfill, your restaurant can return them to the coast—where they become the foundation for new oyster reefs. Here’s how it works: collect the shells, schedule a pickup, and the local recovery team handles the rest. The restored reefs create habitat, filter coastal waters, and soften the impact of waves. One small change in your kitchen can make a lasting difference beyond it. Join your community’s oyster shell recycling program.

```json
{
  "taskType": "audioInference",
  "taskUUID": "3717e9dd-326c-4e8b-874f-136a155bfc51",
  "model": "xai:tts@0",
  "speech": {
    "text": "Tonight’s oysters can help protect tomorrow’s shoreline. Instead of sending empty shells to a landfill, your restaurant can return them to the coast—where they become the foundation for new oyster reefs. Here’s how it works: collect the shells, schedule a pickup, and the local recovery team handles the rest. The restored reefs create habitat, filter coastal waters, and soften the impact of waves. One small change in your kitchen can make a lasting difference beyond it. Join your community’s oyster shell recycling program.",
    "voice": "orion",
    "language": "en"
  }
}
```