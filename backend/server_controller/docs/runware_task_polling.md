---
title: Task Polling | Runware Docs
url: https://runware.ai/docs/models-api/task-polling
description: Retrieve results from async operations using the getResponse task. Learn how to poll for status updates and fetch completed results.
relatedDocuments:
  - https://runware.ai/docs/models-api/webhooks
  - https://runware.ai/docs/models-api/introduction
---
## Introduction

Some operations like video generation require extended processing time. Instead of holding the connection open, you can set `"deliveryMethod": "async"` on your request to **queue the task for asynchronous processing**. You receive an acknowledgment immediately, then use the `getResponse` task to poll for status updates and retrieve the final result when ready.

[Webhooks](https://runware.ai/docs/models-api/webhooks) are the alternative: results arrive by HTTP POST as soon as they are ready, **with no polling at all**.

> [!NOTE]
> If you are using the [TypeScript](https://runware.ai/docs/tools/typescript) or [Python](https://runware.ai/docs/tools/python) SDK, async polling is handled automatically. You do not need to call `getResponse` directly. This page is relevant if you are integrating with the raw API.

### How it works

When you call `getResponse` with a task UUID from an async operation, the system:

1. **Checks active operations** and finds the running async task and all its generations.
2. **Returns the current status** of each generation: `processing`, `success`, or `error`.
3. **Provides results as they become available.** Completed generations include their final outputs, so you can access partial results without waiting for everything to finish.

> [!NOTE]
> Polling best practices
> 
> Implement **exponential backoff** when polling to avoid overwhelming the API. Start with 1-2 second intervals and gradually increase the delay between requests.
> 
> For tasks with predictable durations (like video generation), consider adding an **initial delay** before your first poll to reduce unnecessary requests.

## Request

The Runware API always accepts an array of objects as input, where each object represents a **specific task to be performed**. The structure varies depending on the workflow and features used.

The following example shows the structure of a request object.

**TypeScript**:

```typescript
import { createClient } from '@runware/sdk'

const client = await createClient({ apiKey: process.env.RUNWARE_API_KEY })
await client.connect()

const [result] = await client.run({
  taskType: 'getResponse',
  taskUUID: '50836053-a0ee-4cf5-b9d6-ae7c5d140ada'
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
            "taskType": "getResponse",
            "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada"
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
      "taskType": "getResponse",
      "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada"
    }
  ]'
```

**CLI**:

```bash
runware result 50836053-a0ee-4cf5-b9d6-ae7c5d140ada
```

**JSON**:

```json
{
  "taskType": "getResponse",
  "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada"
}
```

---

### [taskType](#request-tasktype)

- **Type**: `string`
- **Required**: true
- **Value**: `getResponse`

Identifier for the type of task being performed.

### [taskUUID](#request-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier for tracking tasks and matching async responses. Must be unique per task.

## Response

**All requests return the same response format.** Processing and successful generations appear in the `data` array, and failed ones move to the `errors` array.

```json
{
  "data": [
    {
      "taskType": "videoInference",
      "taskUUID": "24cd5dff-cb81-4db5-8506-b72a9425f9d1",
      "status": "processing",
      "progress": 47
    },
    {
      "taskType": "videoInference",
      "taskUUID": "24cd5dff-cb81-4db5-8506-b72a9425f9d1",
      "status": "success",
      "videoUUID": "b7db282d-2943-4f12-992f-77df3ad3ec71",
      "videoURL": "https://vm.runware.ai/video/os/a14d18/ws/2/vi/b7db282d-2943-4f12-992f-77df3ad3ec71.mp4",
      "cost": 0.18
    }
  ],
  "errors": [
    {
      "code": "timeoutProvider",
      "status": "error",
      "message": "The external provider did not respond within the timeout window. The request was automatically terminated.",
      "documentation": "https://runware.ai/docs/models-api/errors",
      "taskUUID": "24cd5dff-cb81-4db5-8506-b72a9425f9d1"
    }
  ]
}
```

---

### [taskType](#response-tasktype)

- **Type**: `string`
- **Required**: true

Identifier for the type of task this response belongs to.

**Possible values**: `authentication` `imageInference` `videoInference` `audioInference` `textInference` `3dInference` `modelSearch` `modelUpload` `accountManagement` `imageUpload` `mediaStorage` `getResponse` `caption` `controlNetPreprocess` `imageMasking` `promptEnhance` `removeBackground` `upscale` `vectorize` `training` `tools` `ping`

### [taskUUID](#response-taskuuid)

- **Type**: `string`
- **Required**: true
- **Format**: `UUID v4`

UUID v4 identifier echoed from the original request, used to match async responses to their tasks.

### [status](#response-status)

- **Type**: `string`
- **Required**: true

Current status of the task.

**Possible values**: `processing` `success` `error`

### [progress](#response-progress)

- **Type**: `integer`
- **Min**: `0`
- **Max**: `100`

Task progress as a percentage from 0 to 100. Returned while `status` is `processing`, and only for tasks that support progress reporting.

### [error](#response-error)

- **Path**: `error.code`
- **Type**: `object (2 properties)`

Error details if the task failed.

#### [code](#response-error-code)

- **Path**: `error.code`
- **Type**: `string`

Error code.

#### [message](#response-error-message)

- **Path**: `error.message`
- **Type**: `string`

Error message description.