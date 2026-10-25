---
title: One API for image, video, audio, 3D, and text | Runware Docs
url: https://runware.ai/docs/models-api/introduction
description: Hundreds of models on Runware are addressable by their model identifier and reachable through a single call.
relatedDocuments:
  - https://runware.ai/docs/models-api/authentication
  - https://runware.ai/docs/models-api/model-search
  - https://runware.ai/docs/tools/typescript
---
## Introduction

Runware exposes a **single endpoint** for image, video, audio, 3D, and text generation. Models are addressed by their **model identifier**, and the same request shape works across every modality.

## One shape, every modality

**Image Request**:

```json
{
  "taskType": "imageInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "xai:grok-imagine@image-2.0",
  "positivePrompt": "A golden retriever on a park bench, cinematic lighting",
  "width": 1024,
  "height": 1024
}
```

**Video Request**:

```json
{
  "taskType": "videoInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "klingai:kling-video@3-4k",
  "positivePrompt": "Ocean waves crashing on a beach at sunset",
  "width": 3840,
  "height": 2160,
  "duration": 8
}
```

**Audio Request**:

```json
{
  "taskType": "audioInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "fishaudio:s2.1@pro",
  "speech": {
    "text": "Welcome to Runware."
  }
}
```

**3D Request**:

```json
{
  "taskType": "3dInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "tripo:v3.1@0",
  "positivePrompt": "A vintage Polaroid camera, detailed mesh"
}
```

**Text Request**:

```json
{
  "taskType": "textInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "anthropic:claude@opus-4.8",
  "messages": [
    { "role": "user", "content": "Write a haiku about the ocean." }
  ]
}
```

**Image Response**:

```json
{
  "taskType": "imageInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "imageUUID": "77da2d99-a6d3-44d9-b8c0-ae9fb06b6200",
  "imageURL": "https://im.runware.ai/image/os/a14d18/ws/2/ii/77da2d99-a6d3-44d9-b8c0-ae9fb06b6200.jpg",
  "cost": 0.06
}
```

**Video Response**:

```json
{
  "taskType": "videoInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "videoUUID": "b7db282d-2943-4f12-992f-77df3ad3ec71",
  "videoURL": "https://vm.runware.ai/video/os/a14d18/ws/2/vi/b7db282d-2943-4f12-992f-77df3ad3ec71.mp4",
  "cost": 3.36
}
```

**Audio Response**:

```json
{
  "taskType": "audioInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "audioUUID": "f1e2d3c4-b5a6-7890-1234-567890abcdef",
  "audioURL": "https://am.runware.ai/audio/os/a14d18/ws/2/ai/f1e2d3c4-b5a6-7890-1234-567890abcdef.mp3",
  "cost": 0.0003
}
```

**3D Response**:

```json
{
  "taskType": "3dInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "outputs": {
    "files": [
      {
        "uuid": "8c2e6d99-7404-43e6-a8f0-7a3b8d8d9f0c",
        "url": "https://im.runware.ai/image/os/a14d18/ws/5/ii/8c2e6d99-7404-43e6-a8f0-7a3b8d8d9f0c.glb"
      }
    ]
  },
  "cost": 0.3
}
```

**Text Response**:

```json
{
  "taskType": "textInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "text": "Ocean breathes the dawn\nWaves whisper their endless songs\nMoon hangs silent watch",
  "finishReason": "stop",
  "cost": 0.0005
}
```

**452 hosted models**. Plus thousands of community and user-uploaded models that run on any of our supported architectures.

[Browse all models](https://runware.ai/docs/models)

## Getting started

1. [Sign up](https://runware.ai/signup) for a Runware account.
2. Generate an API key from your dashboard.
3. Read [Authentication](https://runware.ai/docs/models-api/authentication) to learn how to send the key with your requests.
4. Browse [Models](https://runware.ai/docs/models) to find the right one for your use case.

> [!NOTE]
> The fastest path is the [TypeScript SDK](https://runware.ai/docs/tools/typescript) or [Python SDK](https://runware.ai/docs/tools/python). Both wrap REST and WebSocket transports and validate requests against the model's JSON Schema before sending. For AI agents, see the [MCP integration](https://runware.ai/docs/tools/mcp) and [Skills](https://runware.ai/docs/tools/skills).

## Your first request

The same call in cURL, TypeScript, or Python.

**TypeScript**:

```typescript
import { createClient } from '@runware/sdk'

const client = await createClient({ apiKey: process.env.RUNWARE_API_KEY })
await client.connect()

const [result] = await client.run({
  model: 'xai:grok-imagine@image-2.0',
  positivePrompt: 'A golden retriever on a park bench, cinematic lighting',
  width: 1024,
  height: 1024
})
```

**Python**:

```python
import asyncio
import os

from runware import Runware

async def main():
    async with Runware(api_key=os.environ["RUNWARE_API_KEY"]) as client:
        results = await client.run({
            "model": "xai:grok-imagine@image-2.0",
            "positivePrompt": "A golden retriever on a park bench, cinematic lighting",
            "width": 1024,
            "height": 1024
        })

asyncio.run(main())
```

**cURL**:

```bash
curl https://api.runware.ai/v1 \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "taskType": "imageInference",
      "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
      "model": "xai:grok-imagine@image-2.0",
      "positivePrompt": "A golden retriever on a park bench, cinematic lighting",
      "width": 1024,
      "height": 1024
    }
  ]'
```

**CLI**:

```bash
runware run xai:grok-imagine@image-2.0 \
  positivePrompt="A golden retriever on a park bench, cinematic lighting" \
  width=1024 \
  height=1024
```

**JSON**:

```json
{
  "taskType": "imageInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "xai:grok-imagine@image-2.0",
  "positivePrompt": "A golden retriever on a park bench, cinematic lighting",
  "width": 1024,
  "height": 1024
}
```

## Core API concepts

### Task-based architecture

Every Runware request is a **task**. Each task is processed independently, so you can send one task or batch many in a single call. Long-running tasks (video, large images) run asynchronously and deliver results as they complete.

### Anatomy of a request

Hover any field below to see what it does. The same shape applies to every task type on Runware.

```json
[
  {
    "taskType": "imageInference",
    "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
    "model": "xai:grok-imagine@image-2.0",
    "positivePrompt": "A golden retriever on a park bench",
    "width": 1024,
    "height": 1024
  }
]
```

1. **Array of tasks** — Every request is an array. Send one task or batch many in a single call.
2. **Task object** — Each item in the array is one task. The same object shape works for image, video, audio, 3D, and text.
3. **Task type** — Picks the operation. `imageInference`, `videoInference`, `audioInference`, `3dInference`, `textInference`, plus utilities.
4. **Task UUID** — A UUID v4 you generate. Used to match the response and to trace the task in your dashboard.
5. **Model** — Address any model on the platform by its model identifier: `creator:family@version`.

### Common response structure

Every response echoes `taskType` and `taskUUID` from the request, and adds the modality-specific output: `imageURL`, `videoURL`, `audioURL`, mesh files for 3D, or `text` for LLMs. Set **`includeCost: true`** on the request to get a `cost` field back.

When a task produces multiple results, each completes independently and arrives in its own message. Output URLs are retained for **7 days** by default. Set `ttl` on the request to change the retention.

```json
{
  "data": [
    {
      "taskType": "imageInference",
      "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
      "imageUUID": "77da2d99-a6d3-44d9-b8c0-ae9fb06b6200",
      "imageURL": "https://im.runware.ai/image/os/a14d18/ws/2/ii/a770f077-f413-47de-9dac-be0b26a35da6.jpg",
      "cost": 0.0013
    },
    {
      "taskType": "videoInference",
      "taskUUID": "b880f077-e514-58ef-0ebd-ce1c37b46eb7",
      "videoUUID": "b7db282d-2943-4f12-992f-77df3ad3ec71",
      "videoURL": "https://vm.runware.ai/video/os/a14d18/ws/2/vi/b7db282d-2943-4f12-992f-77df3ad3ec71.mp4",
      "cost": 0.18
    }
  ]
}
```