---
title: Qwen3-TTS 1.7B Base | Runware Docs
url: https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base
description: High-quality multilingual text-to-speech with voice cloning and ultra-low latency
---
# Qwen3-TTS 1.7B Base

![Qwen3-TTS 1.7B Base](https://assets.runware.ai/covers/alibaba-qwen3-tts-1-7b-base.jpg)

Qwen3-TTS 1.7B Base is the foundation text-to-speech model from Alibaba's Qwen3-TTS family. It generates human-like speech across 10+ languages including Chinese, English, Japanese, Korean, and European languages. It supports voice cloning from a 3-second audio sample and achieves latency as low as 97ms for real-time applications.

- **Status**: live
- **ID**: `alibaba:qwen@3-tts-1.7b-base`
- Try in Playground
- **Creator**: Alibaba
- **Release Date**: January 1, 2026
- **Zero Data Retention**: Yes
- **Capabilities**: Audio to Audio, Checkpoint

## Pricing

$0.000015 per character · ~4s per request

**4 example generations**

- [speaker-embedding](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base/examples#example-orbital-greenhouse-intercom-ambience): `$0.0005` (~4s)
- [speaker-embedding](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base/examples#example-coffee-grounds-collection-explainer): `$0.0028` (~5s)
- [voice-cloning](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base/examples#example-rare-seed-licensing-podcast-intro): `$0.0029` (~4s)
- [voice-cloning](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base/examples#example-urban-leak-detection-explainer): `$0.0052` (~4s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

When `settings.xVectorOnly` is `true`, `settings` is required.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Audio output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-outputformat)

- **Type**: `string`
- **Default**: `MP3`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`MP3\`: Compressed audio, smaller file size.
- \`WAV\`: Uncompressed, high-quality audio.
- \`FLAC\`: Lossless compression.
- \`OGG\`: Open-source compressed audio format (Vorbis codec).

**Allowed values**: `MP3` `WAV` `FLAC` `OGG`

### [audioSettings](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-audiosettings)

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

#### [bitrate](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-audiosettings-bitrate)

- **Path**: `audioSettings.bitrate`
- **Type**: `integer`
- **Min**: `8`

Audio bitrate in kbps.

#### [channels](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-audiosettings-channels)

- **Path**: `audioSettings.channels`
- **Type**: `integer`
- **Default**: `2`

Number of audio channels. 1 for mono, 2 for stereo.

**Allowed values**: `1` `2`

#### [sampleRate](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-audiosettings-samplerate)

- **Path**: `audioSettings.sampleRate`
- **Type**: `integer`

Audio sample rate in Hz.

### [webhookURL](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `4`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [audio](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-inputs-audio)

- **Path**: `inputs.audio`
- **Type**: `string`
- **Required**: true

Audio input (UUID or URL).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `alibaba:qwen@3-tts-1.7b-base`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [speech](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-speech)

- **Path**: `speech.text`
- **Type**: `object (4 properties)`
- **Required**: true

Settings for speech generation.

#### [text](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-speech-text)

- **Path**: `speech.text`
- **Type**: `string`
- **Required**: true
- **Max**: `2000`

Text to convert to speech.

#### [voice](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-speech-voice)

- **Path**: `speech.voice`
- **Type**: `string`
- **Required**: true
- **Value**: `clone`

Voice identifier to use for speech generation.

#### [language](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-speech-language)

- **Path**: `speech.language`
- **Type**: `string`
- **Min**: `1`
- **Default**: `Auto`

Language code for speech generation.

**Allowed values**: `Auto` `Chinese` `English` `Japanese` `Korean` `German` `French` `Russian` `Portuguese` `Spanish` `Italian`

#### [speed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-speech-speed)

- **Path**: `speech.speed`
- **Type**: `float`
- **Min**: `0.25`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Playback speed of the generated speech.

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [maxNewTokens](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-settings-maxnewtokens)

- **Path**: `settings.maxNewTokens`
- **Type**: `integer`
- **Default**: `2048`

Audio output token cap. Higher values allow longer audio but risk hangs.

### [transcript](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-settings-transcript)

- **Path**: `settings.transcript`
- **Type**: `string`

Transcript of the reference audio. Required for ICL mode, optional for x-vector-only mode.

### [xVectorOnly](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#request-settings-xvectoronly)

- **Path**: `settings.xVectorOnly`
- **Type**: `boolean`

If true, uses speaker embedding only (no transcript needed, lower similarity). If false, uses ICL mode (requires transcript, higher quality).

## Response Parameters

### [taskType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [audioUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-audiouuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output audio.

### [audioURL](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-audiourl)

- **Type**: `string`
- **Format**: `uri`

URL of the output audio.

### [audioBase64Data](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-audiobase64data)

- **Type**: `string`

Base64-encoded audio data.

### [audioDataURI](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-audiodatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output audio.

### [seed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [cost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-base#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### voice-cloning (voice-cloning)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-base/59a1c4da-f9a1-4c90-a883-dd7e0128e489.wav)

> **Prompt**: Una fuga subterránea puede pasar meses sin verse, pero no sin escucharse. Nuestros sensores acústicos analizan la red durante la noche, identifican sonidos anómalos y señalan el tramo exacto que necesita atención. Así, el equipo municipal puede reparar antes, excavar menos y evitar que miles de litros de agua potable se pierdan bajo la ciudad.

```json
{
  "taskType": "audioInference",
  "taskUUID": "cacdb98b-1d02-4048-92be-7e7e21be706e",
  "model": "alibaba:qwen@3-tts-1.7b-base",
  "speech": {
    "text": "Una fuga subterránea puede pasar meses sin verse, pero no sin escucharse. Nuestros sensores acústicos analizan la red durante la noche, identifican sonidos anómalos y señalan el tramo exacto que necesita atención. Así, el equipo municipal puede reparar antes, excavar menos y evitar que miles de litros de agua potable se pierdan bajo la ciudad.",
    "voice": "clone",
    "language": "Spanish"
  },
  "settings": {
    "transcript": "Today, the morning light made every familiar street feel new again.",
    "xVectorOnly": false
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/9c7e0f15-6070-4526-a1fc-41013cad9e81.mp3"
  }
}
```

---

### speaker-embedding (speaker-embedding)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-base/68cd37ee-ab73-44c6-88f3-764aa32f182f.wav)

> **Prompt**: Yesterday’s coffee grounds still have work to do. We collect them from local cafés and turn them into rich mushroom substrate, keeping useful material in circulation, one bucket at a time.

```json
{
  "taskType": "audioInference",
  "taskUUID": "925323c3-084b-4c81-9930-96b78535f4ad",
  "model": "alibaba:qwen@3-tts-1.7b-base",
  "speech": {
    "text": "Yesterday’s coffee grounds still have work to do. We collect them from local cafés and turn them into rich mushroom substrate, keeping useful material in circulation, one bucket at a time.",
    "voice": "clone",
    "language": "English"
  },
  "settings": {
    "xVectorOnly": true
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/06a2dbe3-5891-45f1-a53b-55ccf01235fc.mp3"
  }
}
```

---

### speaker-embedding (speaker-embedding)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-base/7f6ad73d-ac97-4a2c-9085-3ae6c2181243.wav)

> **Prompt**: 水耕栽培区画の散水を開始します。通路の青い表示線から離れてください。

```json
{
  "taskType": "audioInference",
  "taskUUID": "45b97d4d-4dff-4cf0-b2af-f8187f7e4536",
  "model": "alibaba:qwen@3-tts-1.7b-base",
  "speech": {
    "text": "水耕栽培区画の散水を開始します。通路の青い表示線から離れてください。",
    "voice": "clone",
    "language": "Japanese"
  },
  "settings": {
    "xVectorOnly": true
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/17e50fde-abe7-47ee-a9a3-5175214461af.mp3"
  }
}
```

---

### voice-cloning (voice-cloning)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-base/3546274e-4d01-4ed3-ad05-97856cf6cb45.wav)

> **Prompt**: Welcome to Field Rights. Today, we follow a drought-tolerant sorghum trait from a university greenhouse to a licensing dispute in Kansas—and ask when public research becomes private leverage.

```json
{
  "taskType": "audioInference",
  "taskUUID": "5a17d95d-011c-45b8-8380-1fc98bb5d89b",
  "model": "alibaba:qwen@3-tts-1.7b-base",
  "speech": {
    "text": "Welcome to Field Rights. Today, we follow a drought-tolerant sorghum trait from a university greenhouse to a licensing dispute in Kansas—and ask when public research becomes private leverage.",
    "voice": "clone",
    "language": "English"
  },
  "settings": {
    "transcript": "Today, the morning light made every familiar street feel new again.",
    "xVectorOnly": false
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/23b5f12c-dd5c-4cf1-882b-65d9c33bd790.mp3"
  }
}
```