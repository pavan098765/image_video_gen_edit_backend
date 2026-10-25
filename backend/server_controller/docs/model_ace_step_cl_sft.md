---
title: ACE-Step v1.5 XL SFT | Runware Docs
url: https://runware.ai/docs/models/ace-step-v1-5-xl-sft
description: Highest-quality 4B music generation model with CFG-controlled prompt adherence
---
# ACE-Step v1.5 XL SFT

![ACE-Step v1.5 XL SFT](https://assets.runware.ai/covers/ace-step-v1-5-xl-sft.jpg)

ACE-Step v1.5 XL SFT is the supervised fine-tuned 4B DiT variant in the ACE-Step 1.5 XL line. It is positioned as the highest-quality XL option, combining 50-step CFG inference with stronger prompt adherence and refined audio quality for text-to-music, cover, and repaint workflows when final output quality matters more than speed or broader editing task coverage.

- **Status**: live
- **ID**: `runware:ace-step@v1.5-xl-sft`
- Try in Playground
- **Release Date**: April 2, 2026
- **Zero Data Retention**: Yes
- **Capabilities**: Text to Audio, Audio to Audio, Checkpoint

## Pricing

$0.00195 from, per run · ~11s per request

**Measured runs**

- duration 60 · steps 20: `$0.00195` (~6s)
- duration 30 · steps 50: `$0.00285` (~5s)
- duration 60 · steps 50: `$0.0039` (~8s)
- duration 60 · steps 100: `$0.00675` (~9s)

**4 example generations**

- [text-to-audio](https://runware.ai/docs/models/ace-step-v1-5-xl-sft/examples#example-retro-tech-podcast-theme-song) · 60s: `$0.0039` (~13s)
- [audio-repaint](https://runware.ai/docs/models/ace-step-v1-5-xl-sft/examples#example-maritime-claims-podcast-segment-bumper): `$0.0096` (~24s)
- [audio-extend](https://runware.ai/docs/models/ace-step-v1-5-xl-sft/examples#example-neutrino-observatory-ambient-extension): `$0.0115` (~31s)
- [audio-driven](https://runware.ai/docs/models/ace-step-v1-5-xl-sft/examples#example-flooded-archive-salvage-audio-lesson): `$0.0144` (~32s)

billed by compute time, so a request costs what it took · times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

When `inputs.audio` is provided, `duration` cannot be used.

---

When `negativePrompt` is provided, `CFGScale` must be greater than `1`.

---

When `settings.repaintingStart` or `settings.repaintingEnd` is provided, `inputs.audio` is required.

---

When `strength` is provided, `inputs.audio` is required.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Audio output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-outputformat)

- **Type**: `string`
- **Default**: `MP3`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`MP3\`: Compressed audio, smaller file size.
- \`WAV\`: Uncompressed, high-quality audio.
- \`FLAC\`: Lossless compression.
- \`OGG\`: Open-source compressed audio format (Vorbis codec).

**Allowed values**: `MP3` `WAV` `FLAC` `OGG`

### [audioSettings](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-audiosettings)

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

#### [bitrate](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-audiosettings-bitrate)

- **Path**: `audioSettings.bitrate`
- **Type**: `integer`
- **Min**: `8`

Audio bitrate in kbps.

#### [channels](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-audiosettings-channels)

- **Path**: `audioSettings.channels`
- **Type**: `integer`
- **Default**: `2`

Number of audio channels. 1 for mono, 2 for stereo.

**Allowed values**: `1` `2`

#### [sampleRate](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-audiosettings-samplerate)

- **Path**: `audioSettings.sampleRate`
- **Type**: `integer`

Audio sample rate in Hz.

### [webhookURL](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `4`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Inputs**

Input resources for the task (images, audio, etc). These must be nested inside the \`inputs\` object.

### [audio](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-inputs-audio)

- **Path**: `inputs.audio`
- **Type**: `string`

Audio input (UUID or URL).

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `runware:ace-step@v1.5-xl-sft`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-positiveprompt)

- **Type**: `string`
- **Required**: true
- **Min**: `2`
- **Max**: `3000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [negativePrompt](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-negativeprompt)

- **Type**: `string`
- **Min**: `2`
- **Max**: `3000`

Prompt to guide what to exclude from generation. Ignored when guidance is disabled (CFGScale ≤ 1).

**Learn more** (1 resource):

- [Prompts: Negative Prompts](https://runware.ai/docs/learn/prompts#negative-prompts) (learn)

### [duration](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-duration)

- **Type**: `float`
- **Min**: `30`
- **Max**: `300`
- **Step**: `0.1`
- **Default**: `60`

Length of the generated audio track in seconds.

### [seed](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-seed)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `2147483647`

Random seed for reproducible generation. When not provided, a random seed is generated in the unsigned 32-bit range.

### [steps](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-steps)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `300`
- **Default**: `50`

Total number of denoising steps. Higher values generally produce more detailed results but take longer.

**Learn more** (1 resource):

- [Steps](https://runware.ai/docs/learn/steps) (learn)

### [CFGScale](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-cfgscale)

- **Type**: `float`
- **Min**: `1`
- **Max**: `30`
- **Step**: `0.01`
- **Default**: `10`

Guidance scale representing how closely the output will resemble the prompt. Higher values produce results more aligned with the prompt.

**Learn more** (1 resource):

- [Cfg Scale](https://runware.ai/docs/learn/cfg-scale) (learn)

### [strength](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-strength)

- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0.5`

Fraction of steps using the input source instead of generated output.

**Learn more** (3 resources):

- [Image To Image: Strength The Transformation Intensity](https://runware.ai/docs/learn/image-to-image#strength-the-transformation-intensity) (learn)
- [Image Inpainting: Strength Controlling Transformation Intensity](https://runware.ai/docs/learn/image-inpainting#strength-controlling-transformation-intensity) (learn)
- [Image Outpainting: Strength Understanding The Critical Threshold](https://runware.ai/docs/learn/image-outpainting#strength-understanding-the-critical-threshold) (learn)

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [bpm](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-bpm)

- **Path**: `settings.bpm`
- **Type**: `integer`
- **Min**: `30`
- **Max**: `300`

Beats per minute. If not set, the model decides automatically.

### [cfgIntervalEnd](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-cfgintervalend)

- **Path**: `settings.cfgIntervalEnd`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `1`

Diffusion ratio where CFG ends (1.0 = last step).

### [cfgIntervalStart](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-cfgintervalstart)

- **Path**: `settings.cfgIntervalStart`
- **Type**: `float`
- **Min**: `0`
- **Max**: `1`
- **Step**: `0.01`
- **Default**: `0`

Diffusion ratio where CFG begins (0.0 = first step).

### [guidanceType](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-guidancetype)

- **Path**: `settings.guidanceType`
- **Type**: `string`
- **Default**: `apg`

Controls how guidance is applied during generation.

**Allowed values**:

- `apg` Adaptive Projected Guidance.
- `cfg` Classifier-Free Guidance.

### [keyScale](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-keyscale)

- **Path**: `settings.keyScale`
- **Type**: `string`

Musical key and scale in '{Note}{Accidental} {Mode}' format (e.g. 'C major', 'F# minor', 'Bb major').

### [lyrics](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-lyrics)

- **Path**: `settings.lyrics`
- **Type**: `string`
- **Min**: `10`
- **Max**: `3000`

Song lyrics, typically formatted like a lyrics website.

### [repaintingEnd](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-repaintingend)

- **Path**: `settings.repaintingEnd`
- **Type**: `float`
- **Min**: `0`
- **Max**: `300`

End time in seconds for repaint region. Requires input audio. Values beyond audio duration append new audio.

### [repaintingStart](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-repaintingstart)

- **Path**: `settings.repaintingStart`
- **Type**: `float`
- **Min**: `-300`
- **Max**: `300`

Start time in seconds for repaint region. Requires input audio. Negative values prepend audio before the start.

### [timeSignature](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-timesignature)

- **Path**: `settings.timeSignature`
- **Type**: `integer`

Beats per measure. If not set, the model decides automatically.

**Allowed values**: `2` `3` `4` `6`

### [vocalLanguage](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-settings-vocallanguage)

- **Path**: `settings.vocalLanguage`
- **Type**: `string`
- **Default**: `en`

ISO 639-1 language code for vocals. `unknown` for instrumental or auto detection.

**Allowed values**: `unknown` `ar` `az` `bg` `bn` `ca` `cs` `da` `de` `el` `en` `es` `fa` `fi` `fr` `he` `hi` `hr` `ht` `hu` `id` `is` `it` `ja` `ko` `la` `lt` `ms` `ne` `nl` `no` `pa` `pl` `pt` `ro` `ru` `sa` `sk` `sr` `sv` `sw` `ta` `te` `th` `tl` `tr` `uk` `ur` `vi` `yue` `zh`

**Features**

Standalone addons and post-processing features.

### [modularScheduler](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#request-advancedfeatures-modularscheduler)

- **Path**: `advancedFeatures.modularScheduler`
- **Type**: `object`

Manual scheduler configuration for fine-grained control over the diffusion process (sampler, modifiers, etc.).

## Response Parameters

### [taskType](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [audioUUID](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-audiouuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output audio.

### [audioURL](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-audiourl)

- **Type**: `string`
- **Format**: `uri`

URL of the output audio.

### [audioBase64Data](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-audiobase64data)

- **Type**: `string`

Base64-encoded audio data.

### [audioDataURI](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-audiodatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output audio.

### [seed](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [cost](https://runware.ai/docs/models/ace-step-v1-5-xl-sft#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/ace-step-v1-5-xl-sft/96866198-a3b1-47f2-9dcb-4a62d42c4780.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "115e410c-4433-4ac9-b165-01d5a6250693",
  "model": "runware:ace-step@v1.5-xl-sft",
  "positivePrompt": "Original retro-futurist electro-soul podcast intro with crisp broken-beat production and the intrigue of a technology documentary. Feature punchy Linn-style drums, warm analog synth bass, glassy electric piano, muted funk guitar, arpeggiated monosynth, and brief vocoder responses. A close-miked female contralto sings with calm authority, precise diction, and understated curiosity; melodic throughout rather than spoken. Polished, intelligent, kinetic, and slightly mysterious without becoming ominous. Build a complete 60-second arrangement: four-bar instrumental hook, restrained verse, rising pre-chorus, memorable full chorus with layered harmonies, then a clean branded outro sting. Keep the lead vocal prominent and every lyric intelligible. Broadcast-ready mix with tight low end, dry drums, subtle stereo synth motion, and a decisive ending rather than a fade.",
  "duration": 60,
  "settings": {
    "lyrics": "[Intro]\nUnder the floor, behind the light,\nYesterday is working through the night.\n\n[Verse]\nEvery little circuit has a maker and a date,\nA factory, a promise, and a twist of hidden fate.\nFrom traffic lights to satellites, the old machines persist;\nWe follow every signal that the future somehow missed.\n\n[Pre-Chorus]\nTrace the hands, decode the signs,\nHistory hums between the lines.\n\n[Chorus]\nSilicon Afterlife,\nWhere the buried systems come alive.\nChip by chip and wire by wire,\nWe find the ghosts inside the fire.\nSilicon Afterlife,\nOld ideas that still survive.\nPower down, but stories stay alive.\n\n[Outro]\nSilicon Afterlife.",
    "bpm": 118,
    "keyScale": "D minor",
    "timeSignature": 4,
    "vocalLanguage": "en"
  }
}
```

---

### audio-driven (audio-driven)

[Listen to audio](https://assets.runware.ai/examples/ace-step-v1-5-xl-sft/7ffad22c-cbb6-47c9-b760-12bed99c2fcc.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "a2da45c2-9af1-43cb-bf66-df5f2c3adda1",
  "model": "runware:ace-step@v1.5-xl-sft",
  "positivePrompt": "Reinterpret the entire supplied track as a polished e-learning narration clip for museum and library disaster-response training. Make the transformation immediately noticeable: replace the source's retro funk instrumentation with prepared piano, bass clarinet, restrained cello harmonics, brushed frame drum, soft analog pulse, and subtle room tone suggesting a quiet conservation laboratory after a flood. Retain the recognizable three-note lead motif, syncopated bass rhythm, steady pulse, four-part arrangement, major transition points, and final resolution from the source. The mood is calm urgency rather than drama. Use an articulate, reassuring adult female narrator with a natural instructional cadence, precise consonants, and brief pauses between actions. Speak, do not sing, the following text verbatim: “When water reaches a paper collection, begin with safety, not speed. Confirm the electricity is isolated, wear nitrile gloves, and photograph each shelf before moving anything. Separate wet volumes from damp ones. Support saturated books from beneath; never pull them by the spine. Interleave only when pages can open without resistance. Freeze material that cannot be air-dried within forty-eight hours. Label every crate with room, shelf, and sequence. The goal is not to make objects look dry. The goal is to preserve evidence, order, and options for the conservator.” Keep the narration centered, intimate, and consistently intelligible. Shape the arrangement around the speech with subtle musical ducking, using the instrumental opening and closing to present the retained motif clearly. Finished editorial audio with balanced dynamics and refined documentary sound.",
  "negativePrompt": "Sung narration, chanting, rap delivery, theatrical acting, alarm sounds, cinematic impacts, horror atmosphere, excessive reverb, busy accompaniment beneath speech, distorted voice, doubled voice, background conversations, added words, omitted instructions, abrupt edit points, or an unresolved ending.",
  "strength": 0.55,
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/8d839ab4-3add-4531-bd98-5149dd2e42f6.mp3"
  }
}
```

---

### audio-extend (audio-extend)

[Listen to audio](https://assets.runware.ai/examples/ace-step-v1-5-xl-sft/77fa4664-8a8c-4c4c-a1b9-0f9f8e89da0c.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "1a5d1cb8-c67d-4cc7-97ff-542df91fde86",
  "model": "runware:ace-step@v1.5-xl-sft",
  "positivePrompt": "Continue the supplied underground neutrino observatory ambience seamlessly after its 30-second endpoint, appending audio through 45 seconds. Preserve the exact spatial character, noise floor, deep ventilation drone, cooling-water movement, restrained electrical hum, sparse relay ticks, photomultiplier-like clicks, and slow irregular machinery pulse already established. Develop the existing rhythm subtly: let the relay pattern become slightly more interconnected, introduce one distant low metal resonance, and allow a faint glassy electronic overtone to emerge from the hum. Keep the extension immersive, clinical, cavernous, and mysterious, suitable as a looping ambient bed for a science museum installation. Maintain realistic scale and smooth continuity at the join. No restart, transition cue, new genre, or obvious musical section. Finish still in motion with unresolved machinery and room tone.",
  "negativePrompt": "Vocals, speech, choir, melody, drums, cinematic trailer hits, risers, alarms, sirens, explosions, jump scares, harsh distortion, abrupt edits, silence, fade-out, final chord, rhythmic reset, bright pop production",
  "strength": 0.65,
  "settings": {
    "repaintingStart": 24,
    "repaintingEnd": 45
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/36f32b52-75dd-4fdd-b851-6492b1963bc6.mp3"
  }
}
```

---

### audio-repaint (audio-repaint)

[Listen to audio](https://assets.runware.ai/examples/ace-step-v1-5-xl-sft/b2cdc81c-7ff8-46b1-b806-4d903e2fd122.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "a11d34e4-b519-4f10-be3c-e724ce1a7b79",
  "model": "runware:ace-step@v1.5-xl-sft",
  "positivePrompt": "Replace only the musical passage from 8 to 18 seconds with new content for a premium investigative maritime-insurance podcast bumper. Preserve the untouched audio before and after the replacement. Instead of the source's bowed-vibraphone melody, create a tense but understated interlocking figure using dry marimba, muted prepared piano, and a low plucked cello ostinato. Add tiny metallic accents reminiscent of drafting tools and cargo seals, treated as refined percussion rather than literal sound effects. Keep the existing tempo, pulse, tonal center, intimate studio acoustics, and restrained analog warmth. Enter seamlessly from the prepared-piano rhythm at 8 seconds, develop a memorable four-note question-and-answer motif, then simplify during the final two seconds so the passage reconnects naturally and invisibly with the original material at 18 seconds. Editorial, intelligent, meticulous, and modern; no vocals, speech, sea shanty references, waves, foghorns, EDM drops, trailer booms, or abrupt splice.",
  "strength": 0.75,
  "settings": {
    "repaintingStart": 8,
    "repaintingEnd": 18
  },
  "inputs": {
    "audio": "https://assets.runware.ai/assets/inputs/a2be1c20-67e0-4321-8fc3-8a1fe2452b82.mp3"
  }
}
```