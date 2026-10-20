---
title: MiniMax Music 2.6 | Runware Docs
url: https://runware.ai/docs/models/minimax-music-2-6
description: Promptable full-song generation with vocals, lyrics, BPM and key control
---
# MiniMax Music 2.6

![MiniMax Music 2.6](https://assets.runware.ai/covers/minimax-music-2-6.jpg)

MiniMax Music 2.6 is MiniMax’s latest music generation model for full vocal songs and instrumentals from text prompts. It supports natural-language prompts or detailed production-style instructions, follows specified BPM and key with high reliability, and exposes fine-grained song structure control through section tags. The same Music API also supports instrumental generation, lyrics-assisted workflows, and synchronous or streaming delivery.

- **Status**: live
- **ID**: `minimax:music@2.6`
- Try in Playground
- **Creator**: MiniMax
- **Release Date**: April 10, 2026
- **Capabilities**: Text to Audio, Checkpoint

## Pricing

$0.15 per run · ~2m 21s per request

**4 example generations**

- [instrumental-music](https://runware.ai/docs/models/minimax-music-2-6/examples#example-urban-infrastructure-podcast-theme): `$0.15` (~2m 21s)
- [text-to-audio](https://runware.ai/docs/models/minimax-music-2-6/examples#example-community-seed-library-folk-anthem): `$0.15` (~2m 32s)
- [lyrics-to-music](https://runware.ai/docs/models/minimax-music-2-6/examples#example-womens-rugby-championship-anthem): `$0.15` (~2m 21s)
- [optimized-lyrics-to-music](https://runware.ai/docs/models/minimax-music-2-6/examples#example-manuscript-handling-microlearning-song): `$0.15` (~1m 42s)

times vary, measured end to end · [How pricing works](https://runware.ai/docs/models-api/pricing)

## Compatibility & Validation

When `settings.instrumental` is `true`, `settings.lyrics` cannot be used.

---

When `settings.instrumental` is `true`, `settings.lyricsOptimizer` cannot be `true`.

---

When `settings.instrumental` is `true`, `positivePrompt` is required.

---

When none of `settings.instrumental` or `settings.lyricsOptimizer` are provided, `settings.lyrics` is required.

---

When `settings.lyrics` is not provided, `positivePrompt` is required.

## Request Parameters

**API Options**

Platform-level options for task execution and delivery.

### [taskType](https://runware.ai/docs/models/minimax-music-2-6#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task being performed.

### [taskUUID](https://runware.ai/docs/models/minimax-music-2-6#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

### [outputType](https://runware.ai/docs/models/minimax-music-2-6#request-outputtype)

- **Type**: `string`
- **Default**: `URL`

Audio output type.

**Allowed values**: `URL` `base64Data` `dataURI`

### [outputFormat](https://runware.ai/docs/models/minimax-music-2-6#request-outputformat)

- **Type**: `string`
- **Default**: `MP3`

Specifies the file format of the generated output. The available values depend on the task type and the specific model's capabilities.

- \`MP3\`: Compressed audio, smaller file size.
- \`WAV\`: Uncompressed, high-quality audio.
- \`FLAC\`: Lossless compression.
- \`OGG\`: Open-source compressed audio format (Vorbis codec).

**Allowed values**: `MP3` `WAV` `FLAC` `OGG`

### [audioSettings](https://runware.ai/docs/models/minimax-music-2-6#request-audiosettings)

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

#### [bitrate](https://runware.ai/docs/models/minimax-music-2-6#request-audiosettings-bitrate)

- **Path**: `audioSettings.bitrate`
- **Type**: `integer`
- **Min**: `8`

Audio bitrate in kbps.

#### [channels](https://runware.ai/docs/models/minimax-music-2-6#request-audiosettings-channels)

- **Path**: `audioSettings.channels`
- **Type**: `integer`
- **Default**: `2`

Number of audio channels. 1 for mono, 2 for stereo.

**Allowed values**: `1` `2`

#### [sampleRate](https://runware.ai/docs/models/minimax-music-2-6#request-audiosettings-samplerate)

- **Path**: `audioSettings.sampleRate`
- **Type**: `integer`

Audio sample rate in Hz.

### [webhookURL](https://runware.ai/docs/models/minimax-music-2-6#request-webhookurl)

- **Type**: `string`
- **Format**: `uri`

Webhook URL that receives JSON responses via HTTP POST when generation tasks complete. For batch requests with multiple results, each completed item triggers a separate webhook call as it becomes available.

**Learn more** (1 resource):

- [Webhooks](https://runware.ai/docs/models-api/webhooks) (platform)

### [deliveryMethod](https://runware.ai/docs/models/minimax-music-2-6#request-deliverymethod)

- **Type**: `string`
- **Default**: `sync`

Determines how the API delivers task results.

**Allowed values**:

- `sync` Returns complete results directly in the API response.
- `async` Returns an immediate acknowledgment with the task UUID. Poll for results using getResponse.

**Learn more** (1 resource):

- [Task Polling](https://runware.ai/docs/models-api/task-polling) (platform)

### [uploadEndpoint](https://runware.ai/docs/models/minimax-music-2-6#request-uploadendpoint)

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

### [ttl](https://runware.ai/docs/models/minimax-music-2-6#request-ttl)

- **Type**: `integer`
- **Min**: `60`

Time-to-live (TTL) in seconds for generated content. Only applies when `outputType` is `URL`.

### [includeCost](https://runware.ai/docs/models/minimax-music-2-6#request-includecost)

- **Type**: `boolean`

Include task cost in the response.

### [numberResults](https://runware.ai/docs/models/minimax-music-2-6#request-numberresults)

- **Type**: `integer`
- **Min**: `1`
- **Max**: `4`
- **Default**: `1`

Number of results to generate. Each result uses a different seed, producing variations of the same parameters.

### [routing](https://runware.ai/docs/models/minimax-music-2-6#request-routing)

- **Path**: `routing.preferences`
- **Type**: `object (6 properties)`

Model router options. Only read when `model` is `auto`, and never sent to the model.

#### [preferences](https://runware.ai/docs/models/minimax-music-2-6#request-routing-preferences)

- **Path**: `routing.preferences`
- **Type**: `object (3 properties)`

Relative importance of each dimension when models are compared. Weights, not thresholds.

##### [quality](https://runware.ai/docs/models/minimax-music-2-6#request-routing-preferences-quality)

- **Path**: `routing.preferences.quality`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [speed](https://runware.ai/docs/models/minimax-music-2-6#request-routing-preferences-speed)

- **Path**: `routing.preferences.speed`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

##### [cost](https://runware.ai/docs/models/minimax-music-2-6#request-routing-preferences-cost)

- **Path**: `routing.preferences.cost`
- **Type**: `integer`
- **Min**: `0`
- **Max**: `5`

Importance of one dimension. `0` leaves it out of the decision.

#### [maxPrice](https://runware.ai/docs/models/minimax-music-2-6#request-routing-maxprice)

- **Path**: `routing.maxPrice`
- **Type**: `float`
- **Min**: `0`

Upper bound on what this request may cost, in USD. A model whose cost cannot be bounded is not chosen.

#### [requirements](https://runware.ai/docs/models/minimax-music-2-6#request-routing-requirements)

- **Path**: `routing.requirements`
- **Type**: `string`
- **Max**: `3000`

Conditions the result has to meet, in plain words. A model that cannot meet them is not chosen.

**Core Parameters**

Primary parameters that define the task output.

### [model](https://runware.ai/docs/models/minimax-music-2-6#request-model)

- **Type**: `string`
- **Required**: true
- **Value**: `minimax:music@2.6`

Send `auto` instead of an AIR identifier to have the model router choose the model for this task. The rest of the request is unchanged, every value you send is kept, and only the parameters you leave out are filled in. See [Model Router](https://runware.ai/docs/models-api/model-router).

**Learn more** (3 resources):

- [Text To Image: Model Selection](https://runware.ai/docs/learn/text-to-image#model-selection) (learn)
- [Image Inpainting: Model Specialized Inpainting Models](https://runware.ai/docs/learn/image-inpainting#model-specialized-inpainting-models) (learn)
- [Image Outpainting: Other Critical Parameters](https://runware.ai/docs/learn/image-outpainting#other-critical-parameters) (learn)

### [positivePrompt](https://runware.ai/docs/models/minimax-music-2-6#request-positiveprompt)

- **Type**: `string`
- **Required**: conditional — Required when settings.instrumental is provided or settings.lyrics is not provided
- **Min**: `1`
- **Max**: `2000`

Text prompt describing elements to include in the generated output.

**Learn more** (1 resource):

- [Prompts](https://runware.ai/docs/learn/prompts) (learn)

### [negativePrompt](https://runware.ai/docs/models/minimax-music-2-6#request-negativeprompt)

- **Type**: `string`
- **Min**: `1`
- **Max**: `3500`

Prompt to guide what to exclude from generation. Ignored when guidance is disabled (CFGScale ≤ 1).

**Learn more** (1 resource):

- [Prompts: Negative Prompts](https://runware.ai/docs/learn/prompts#negative-prompts) (learn)

### [seed](https://runware.ai/docs/models/minimax-music-2-6#request-seed)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `1000000`

Random seed for reproducible generation. When not provided, a random seed is generated in the unsigned 32-bit range.

**Learn more** (1 resource):

- [Seed](https://runware.ai/docs/learn/seed) (learn)

**Settings**

Technical parameters to fine-tune the inference process. These must be nested inside the \`settings\` object.

### [instrumental](https://runware.ai/docs/models/minimax-music-2-6#request-settings-instrumental)

- **Path**: `settings.instrumental`
- **Type**: `boolean`

Generate instrumental output without vocals.

### [lyrics](https://runware.ai/docs/models/minimax-music-2-6#request-settings-lyrics)

- **Path**: `settings.lyrics`
- **Type**: `string`
- **Min**: `1`
- **Max**: `3500`

Structured lyrics with section tags such as \[Intro\], \[Verse\], \[Chorus\], \[Bridge\], \[Outro\], and \[Inst\]. For instrumental generation, a tag-only skeleton is acceptable.

### [lyricsOptimizer](https://runware.ai/docs/models/minimax-music-2-6#request-settings-lyricsoptimizer)

- **Path**: `settings.lyricsOptimizer`
- **Type**: `boolean`

Optimize provided lyrics or generate them from the prompt when omitted.

## Response Parameters

### [taskType](https://runware.ai/docs/models/minimax-music-2-6#response-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `audioInference`

Identifier for the type of task this response belongs to.

### [taskUUID](https://runware.ai/docs/models/minimax-music-2-6#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [audioUUID](https://runware.ai/docs/models/minimax-music-2-6#response-audiouuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID of the output audio.

### [audioURL](https://runware.ai/docs/models/minimax-music-2-6#response-audiourl)

- **Type**: `string`
- **Format**: `uri`

URL of the output audio.

### [audioBase64Data](https://runware.ai/docs/models/minimax-music-2-6#response-audiobase64data)

- **Type**: `string`

Base64-encoded audio data.

### [audioDataURI](https://runware.ai/docs/models/minimax-music-2-6#response-audiodatauri)

- **Type**: `string`
- **Format**: `uri`

Data URI of the output audio.

### [seed](https://runware.ai/docs/models/minimax-music-2-6#response-seed)

- **Type**: `integer`

The seed used for generation. If none was provided, shows the randomly generated seed.

### [cost](https://runware.ai/docs/models/minimax-music-2-6#response-cost)

- **Type**: `float`

Task cost in USD. Present when `includeCost` is set to `true` in the request.

## Examples

### text-to-audio (text-to-audio)

[Listen to audio](https://assets.runware.ai/examples/minimax-music-2-6/0a26e81a-1fc6-4192-b2d1-dbbf2d671fa4.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "19552120-8e28-4a38-a8c2-da122e436288",
  "model": "minimax:music@2.6",
  "positivePrompt": "Create a bright, contemporary folk-soul campaign anthem for a public seed library that lends locally adapted vegetable, herb, and flower seeds. Blend Appalachian string-band energy with New Orleans brass: crisp acoustic guitar, clawhammer banjo, upright bass, handclaps, brushed snare, tambourine, warm piano, trumpet, trombone, and a joyful final horn lift. Upbeat around 112 BPM in D major, organic live-room production, catchy and welcoming rather than rustic or nostalgic. Feature a warm, clear female lead with conversational phrasing, supported by a diverse neighborhood choir in the choruses and call-and-response moments. The lyrical narrative should begin with one person opening a small paper seed packet, follow the seeds through shared gardens, balconies, school plots, summer harvests, and community meals, then return to saving and sharing seeds for the next season. Emphasize biodiversity, practical generosity, patience, and neighbors learning from one another without sounding preachy. Build from an intimate verse into a broad, memorable chorus centered on the idea “take one, grow one, bring one back,” followed by a second verse, a short breakdown with handclaps and voices, a triumphant final chorus, and a concise communal outro. Radio-ready, family-friendly, sincere, celebratory, and easy for a crowd to sing after one listen.",
  "settings": {
    "instrumental": false,
    "lyricsOptimizer": true
  }
}
```

---

### instrumental-music (instrumental-music)

[Listen to audio](https://assets.runware.ai/examples/minimax-music-2-6/1ddfce06-d6d3-4dda-872b-56a63a9b943f.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "c9fdfc10-1713-48cc-83c0-0a88ba93b478",
  "model": "minimax:music@2.6",
  "positivePrompt": "Instrumental podcast intro for a journalistic series investigating sewers, substations, tunnels, water mains, and other hidden urban infrastructure. Contemporary chamber-electronic score with dry prepared piano, bass clarinet, muted trombone pulses, marimba, brushed snare, low pizzicato cello, restrained analog synth bass, and precise metallic percussion resembling valves and conduit without becoming literal sound effects. No vocals, spoken words, chants, or vocal samples. Begin with isolated piano taps and a curious two-note bass-clarinet motif; introduce an asymmetric marimba ostinato and quiet mechanical percussion; build into a confident, tightly layered investigative groove; briefly strip back to cello and piano; then resolve with a memorable three-note brass-and-marimba sonic signature and a clean final hit. Approximately 104 BPM, D minor moving toward F major at the resolution. Intelligent, tactile, civic, slightly mysterious, and forward-looking rather than ominous. Polished public-radio production, strong low-mid detail, crisp transient definition, generous space for a host voiceover, and a decisive broadcast-ready ending.",
  "settings": {
    "instrumental": true
  }
}
```

---

### lyrics-to-music (lyrics-to-music)

[Listen to audio](https://assets.runware.ai/examples/minimax-music-2-6/4b3ac314-583c-4fba-8174-a2f0ff0dd270.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "f74670e0-0600-468c-9707-9a9a9496e0c5",
  "model": "minimax:music@2.6",
  "positivePrompt": "A fearless electro-samba stadium anthem for an international women’s rugby championship. Build the arrangement from thunderous surdo and repinique drums, sharp brass stabs, syncopated handclaps, deep modern synth bass, and sparse electric guitar accents. Begin with a tense percussion-and-voice call, drive the verses with tight rhythmic momentum, then expand into a huge melodic chorus designed for a stadium to sing together. Powerful female alto lead with crisp diction, controlled grit, and genuine athletic intensity; layered all-female team chants answer key phrases. The bridge should briefly pull back to heartbeat percussion and intimate lead vocal before a triumphant final chorus with full brass and choir. Polished broadcast-ready production, kinetic, resilient, communal, and celebratory. Perform the supplied lyrics clearly and exactly.",
  "settings": {
    "instrumental": false,
    "lyrics": "[Intro]\nHey! Hey!\nBoots to the earth, eyes on the line\nOne breath, one beat, now is our time\n\n[Verse 1]\nTape on our wrists, steady in the rain\nEvery hard yard, we earn it again\nCall from the captain, hands down the line\nWe turn the whole field when the timing is right\n\n[Pre-Chorus]\nWhen the pressure closes in\nWe lock our shoulders, we begin\n\n[Chorus]\nRun it, run it, over the line\nFifteen hearts moving in time\nHold it, hold it, never let go\nWe are the current, feel how it flows\nSide by side when the whole world calls\nWe rise together, we carry it all\n\n[Verse 2]\nLow in the scrum, then open and wide\nTrust in the hands and the voice at our side\nSpace for a second, we cut through the seam\nOne player breaks, but the run belongs to the team\n\n[Pre-Chorus]\nWhen the pressure closes in\nWe lock our shoulders, we begin\n\n[Chorus]\nRun it, run it, over the line\nFifteen hearts moving in time\nHold it, hold it, never let go\nWe are the current, feel how it flows\nSide by side when the whole world calls\nWe rise together, we carry it all\n\n[Bridge]\nTo every girl with a ball in her hands\nYour name has a place in the roar of the stands\nBring all your courage, bring all you know\nThe field gets wider wherever you go\n\n[Final Chorus]\nRun it, run it, over the line\nFifteen hearts moving in time\nHold it, hold it, never let go\nWe are the current, feel how it flows\nSide by side when the whole world calls\nWe rise together, we carry it all\n\n[Outro]\nOne team! One sound!\nTogether we rise\nOne team! One sound!\nTogether we rise",
    "lyricsOptimizer": false
  }
}
```

---

### optimized-lyrics-to-music (optimized-lyrics-to-music)

[Listen to audio](https://assets.runware.ai/examples/minimax-music-2-6/a7a8cfb6-64f9-4d19-9918-d5bb7f12e2cf.mp3)

```json
{
  "taskType": "audioInference",
  "taskUUID": "1030be5c-e0c7-4289-a6e8-f55d5cd860fc",
  "model": "minimax:music@2.6",
  "positivePrompt": "Educational chamber-pop song for a museum conservation training course, 92 BPM, D minor with a gentle major-key lift in the chorus. Intimate harpsichord, muted viola da gamba, soft frame drum, pizzicato strings, and subtle modern bass; clean, dry production with no cinematic effects. One articulate mezzo-soprano lead, warm and authoritative, using crisp diction and a lightly spoken-sung delivery so every instruction remains intelligible. Brief two-part vocal harmonies may reinforce the chorus mnemonic. The mood is attentive, calm, elegant, and reassuring rather than theatrical. Shape the arrangement like a concise learning chapter: sparse orientation, procedural verses, memorable recap chorus, caution-focused bridge, and a resolved final summary. Preserve the practical conservation guidance in the supplied lyrics while refining meter, rhyme, and natural phrasing. The learner should remember: prepare a clean workspace, wash and dry hands, remove jewelry, use supports, turn pages from strong edges without forcing them, keep food and liquids away, and report damage instead of repairing it.",
  "settings": {
    "lyrics": "[Intro]\nBefore the cover opens, pause and clear the way,\nA quiet desk, a steady light, a careful start today.\n\n[Verse 1]\nWash and dry your hands completely,\nMove all drinks and food away.\nTake off watches, rings, and bracelets,\nGive the manuscript safe space.\nSet the cradle on the table,\nLet it bear the binding's weight.\nNever press a cover flatter\nThan the book will tolerate.\n\n[Chorus]\nClean, clear, cradle, care,\nSupport the book and stay aware.\nSlow hands, strong edge, never force,\nPause when damage changes course.\nClean, clear, cradle, care,\nEvery page is ours to share.\n\n[Verse 2]\nTurn one leaf from a sound margin,\nNot the painted, fragile place.\nGuide it gently, never tug it,\nKeep a calm and even pace.\nIf a page resists your movement,\nStop; do not persuade it through.\nAsk the supervising keeper\nWhat the safest step will be.\n\n[Bridge]\nLoose thread, lifting gold, a tear along the fold:\nDo not tape or trim or mend what time has made so old.\nLeave each fragment where you find it, close the cover only when\nA conservator has guided what should happen next, and when.\n\n[Final Chorus]\nClean, clear, cradle, care,\nSupport the book and stay aware.\nSlow hands, strong edge, never force,\nPause when damage changes course.\nClean, clear, cradle, care,\nRecord concerns and always share.\n\n[Outro]\nPrepare, support, and turn with care;\nWhen something changes, stop and share.",
    "lyricsOptimizer": true,
    "instrumental": false
  }
}
```