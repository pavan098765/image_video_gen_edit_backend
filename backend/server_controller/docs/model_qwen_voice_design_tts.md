---
title: Qwen3-TTS 1.7B VoiceDesign | Runware Docs
url: https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign
description: Text-to-speech with voice creation from natural language descriptions
---
# Qwen3-TTS 1.7B VoiceDesign

![Qwen3-TTS 1.7B VoiceDesign](https://assets.runware.ai/covers/alibaba-qwen3-tts-1-7b-voicedesign.jpg)

Qwen3-TTS 1.7B VoiceDesign is a text-to-speech model from Alibaba that creates custom voices from natural language descriptions specifying emotion, tone, and prosody. It supports voice cloning from a 3-second audio sample, generates speech in 10+ languages including Chinese, English, Japanese, Korean, and European languages, and achieves latency as low as 97ms.

- **Status**: live
- **ID**: `alibaba:qwen@3-tts-1.7b-voicedesign`
- Try in Playground
- **Creator**: Alibaba
- **Release Date**: January 1, 2026
- **Zero Data Retention**: Yes
- **Capabilities**: Text to Audio, Checkpoint

## Pricing

$0.000015 per character · ~6s per request

**4 example generations**

- [text-to-audio](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign/examples#example-semiconductor-cleanroom-intercom-ambience): `$0.0042` (~4s)
- [text-to-audio](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign/examples#example-municipal-water-leak-explainer-voiceover): `$0.0063` (~6s)
- [text-to-audio](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign/examples#example-shellfish-farm-monitoring-explainer): `$0.0078` (~6s)
- [text-to-audio](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign/examples#example-forensic-entomology-training-narration): `$0.0108` (~9s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Audio output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-outputformat)

- **Type**: `string`
- **Default**: `MP3`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`MP3\`: Compressed audio, smaller file size.
- \`WAV\`: Uncompressed, high-quality audio.
- \`FLAC\`: Lossless compression.
- \`OGG\`: Open-source compressed audio format (Vorbis codec).

**Allowed values**: `MP3` `WAV` `FLAC` `OGG`

### [audioSettings](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-audiosettings)

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

#### [bitrate](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-audiosettings-bitrate)

- **Path**: `audioSettings.bitrate`
- **Type**: `integer`
- **Min**: `8`

Audio bitrate in kbps.

#### [channels](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-audiosettings-channels)

- **Path**: `audioSettings.channels`
- **Type**: `integer`
- **Default**: `2`

Number of audio channels. 1 for mono, 2 for stereo.

**Allowed values**: `1` `2`

#### [sampleRate](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-audiosettings-samplerate)

- **Path**: `audioSettings.sampleRate`
- **Type**: `integer`

Audio sample rate in Hz.

### [webhookURL](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `4`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `alibaba:qwen@3-tts-1.7b-voicedesign`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-positiveprompt)

- **Type**: `string`
- **Required**: true

Natural language voice description. Example: 'A warm, cheerful male voice with moderate pace'.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [speech](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-speech)

- **Path**: `speech.text`
- **Type**: `object (4 properties)`
- **Required**: true

Settings for speech generation.

#### [text](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-speech-text)

- **Path**: `speech.text`
- **Type**: `string`
- **Required**: true
- **Max**: `2000`

Text to convert to speech.

#### [voice](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-speech-voice)

- **Path**: `speech.voice`
- **Type**: `string`
- **Required**: true
- **Value**: `design`

Voice identifier to use for speech generation.

#### [language](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-speech-language)

- **Path**: `speech.language`
- **Type**: `string`
- **Min**: `1`
- **Default**: `Auto`

Language code for speech generation.

**Allowed values**: `Auto` `Chinese` `English` `Japanese` `Korean` `German` `French` `Russian` `Portuguese` `Spanish` `Italian`

#### [speed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-speech-speed)

- **Path**: `speech.speed`
- **Type**: `float`
- **Min**: `0.25`
- **Max**: `4`
- **Step**: `0.01`
- **Default**: `1`

Playback speed of the generated speech.

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [maxNewTokens](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#request-settings-maxnewtokens)

- **Path**: `settings.maxNewTokens`
- **Type**: `integer`
- **Default**: `2048`

Audio output token cap. Higher values allow longer audio but risk hangs.

## Response Parameters

### [taskType](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [audioUUID](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-audiouuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output audio.

### [audioURL](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-audiourl)

- **Type**: `string`
- **Format**: `uri`

URL of the output audio.

### [audioBase64Data](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-audiobase64data)

- **Type**: `string`

Base64-encoded audio data.

### [audioDataURI](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-audiodatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output audio.

### [seed](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [cost](https://runware.ai/docs/models/alibaba-qwen3-tts-1-7b-voicedesign#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-voicedesign/305df46b-2f0a-435d-b79d-069be68f75d9.wav)

> **Prompt**: By the time harmful organisms are visible, a shellfish crop may already be at risk. TideTrace detects environmental DNA in routine water samples, revealing emerging threats days earlier than visual checks alone. Farm teams receive a clear risk summary, location data, and practical next steps—without installing new equipment offshore. Earlier warning means more time to adjust intake, protect vulnerable stock, and document every decision. TideTrace turns a bottle of seawater into a clearer plan for the week ahead.

```json
{
  "taskType": "audioInference",
  "taskUUID": "84e865ab-7863-4a78-b6dd-54cb300bcaef",
  "model": "alibaba:qwen@3-tts-1.7b-voicedesign",
  "positivePrompt": "A woman in her late forties with a low, clear alto and a subtle west-coast Scottish accent. Calm, credible, and scientifically precise, with the grounded confidence of an experienced marine biologist rather than a commercial announcer. Measured pacing, crisp technical pronunciation, restrained urgency in the opening, and a gently optimistic lift toward the close.",
  "speech": {
    "text": "By the time harmful organisms are visible, a shellfish crop may already be at risk. TideTrace detects environmental DNA in routine water samples, revealing emerging threats days earlier than visual checks alone. Farm teams receive a clear risk summary, location data, and practical next steps—without installing new equipment offshore. Earlier warning means more time to adjust intake, protect vulnerable stock, and document every decision. TideTrace turns a bottle of seawater into a clearer plan for the week ahead.",
    "voice": "design",
    "language": "English"
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-voicedesign/3bde7b57-9100-4ed9-835e-addf930f5792.wav)

> **Prompt**: A hidden water-main leak can run for months before it reaches the surface. By then, treated water is lost, roads are damaged, and repairs become emergencies. Continuous acoustic monitoring listens for the distinctive signal of escaping water, then helps crews narrow the search before they excavate. The result is earlier detection, fewer unnecessary roadworks, and maintenance planned around the city—not the crisis.

```json
{
  "taskType": "audioInference",
  "taskUUID": "b66c6f62-24db-4e85-91f8-315c0622e7c1",
  "model": "alibaba:qwen@3-tts-1.7b-voicedesign",
  "positivePrompt": "A composed female voice in her late forties with a low, clear alto timbre and a subtle Scandinavian English accent. Precise consonants, measured pacing, and calm authority suited to a municipal infrastructure film. Sound practical and trustworthy rather than promotional, with restrained urgency on risks and a warmer, optimistic lift when describing outcomes.",
  "speech": {
    "text": "A hidden water-main leak can run for months before it reaches the surface. By then, treated water is lost, roads are damaged, and repairs become emergencies. Continuous acoustic monitoring listens for the distinctive signal of escaping water, then helps crews narrow the search before they excavate. The result is earlier detection, fewer unnecessary roadworks, and maintenance planned around the city—not the crisis.",
    "voice": "design",
    "language": "English",
    "speed": 0.96
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-voicedesign/70d48363-bda3-4a3d-8a02-c5f34f5b9586.wav)

> **Prompt**: Insect evidence can help estimate a minimum postmortem interval, but only when the scene is documented before specimens are disturbed. Begin by photographing visible larvae, pupae, and adult insects in place. Record ambient temperature, ground temperature, and the temperature inside the largest larval mass. Next, collect specimens from several locations on and beneath the remains. Preserve one sample immediately, and keep a second sample alive with a small amount of tissue for rearing. Never rely on size alone to identify a larval stage. Species, temperature, and access to the remains all affect development. Your final estimate must combine laboratory identification with weather records and scene conditions.

```json
{
  "taskType": "audioInference",
  "taskUUID": "3109d85b-e3cb-4482-ac67-e677f7b713ff",
  "model": "alibaba:qwen@3-tts-1.7b-voicedesign",
  "positivePrompt": "A mature feminine contralto with a subtle South African English accent and the calm authority of an experienced forensic lecturer. Low, lightly textured timbre; measured pacing; crisp scientific diction; restrained seriousness; and brief, purposeful pauses between procedural steps. Clear and instructive rather than theatrical, with quiet emphasis on cautions and key terms.",
  "speech": {
    "text": "Insect evidence can help estimate a minimum postmortem interval, but only when the scene is documented before specimens are disturbed. Begin by photographing visible larvae, pupae, and adult insects in place. Record ambient temperature, ground temperature, and the temperature inside the largest larval mass. Next, collect specimens from several locations on and beneath the remains. Preserve one sample immediately, and keep a second sample alive with a small amount of tissue for rearing. Never rely on size alone to identify a larval stage. Species, temperature, and access to the remains all affect development. Your final estimate must combine laboratory identification with weather records and scene conditions.",
    "voice": "design",
    "language": "English",
    "speed": 0.92
  }
}
```

---

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/alibaba-qwen3-tts-1-7b-voicedesign/9f8afa6d-b556-411c-a7f4-e2bab4679718.wav)

> **Prompt**: Attention, photolithography team. Bay twelve is entering ultraviolet calibration. Confirm visor seals before crossing the amber line. Automated carriers have priority in the east corridor. Particle count remains within tolerance. Resume wafer transfer when the blue beacon is steady.

```json
{
  "taskType": "audioInference",
  "taskUUID": "247d977d-cb20-4fd4-aa99-3199dc7ad041",
  "model": "alibaba:qwen@3-tts-1.7b-voicedesign",
  "positivePrompt": "A composed middle-aged Taiwanese woman speaking international English with a subtle Mandarin accent. Low mezzo timbre, restrained authority, immaculate consonants, measured procedural pacing, brief natural pauses, and a cool yet reassuring emotional tone. Precise and professional, never theatrical.",
  "speech": {
    "text": "Attention, photolithography team. Bay twelve is entering ultraviolet calibration. Confirm visor seals before crossing the amber line. Automated carriers have priority in the east corridor. Particle count remains within tolerance. Resume wafer transfer when the blue beacon is steady.",
    "voice": "design",
    "language": "English"
  }
}
```