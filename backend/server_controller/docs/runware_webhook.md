---
title: Webhooks | Runware Docs
url: https://runware.ai/docs/models-api/webhooks
description: Receive real-time notifications when your tasks complete using webhooks. Learn how to configure, secure, and handle webhook deliveries from Runware.
relatedDocuments:
  - https://runware.ai/docs/models-api/task-polling
  - https://runware.ai/docs/models-api/introduction
---
## Introduction

Webhooks deliver an HTTP POST the moment a generation task completes, **so nothing has to poll for the result**. Runware sends the complete response to the URL you name.

They earn their place on **long-running work like video generation**, on batches whose items finish at different times, and wherever another system has to react the moment a result exists.

For an alternative approach using client-side polling, see [Task Polling](https://runware.ai/docs/models-api/task-polling).

## How it works

You provide a webhook URL when submitting a task. Runware processes the request asynchronously and POSTs the complete task response as JSON when it completes. **Your endpoint must answer with a 2xx status code to confirm receipt**, or delivery is retried automatically (see [Retry behavior](#retry-behavior)).

In a batch, **each completed item triggers its own webhook call** as it becomes available.

## Configuration

Include the `webhookURL` parameter in your API request:

**TypeScript**:

```typescript
import { createClient } from '@runware/sdk'

const client = await createClient({ apiKey: process.env.RUNWARE_API_KEY })
await client.connect()

const [result] = await client.run({
  model: 'runware:100@1',
  positivePrompt: 'a serene landscape',
  width: 1024,
  height: 1024,
  webhookURL: 'https://api.example.com/webhooks/runware'
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
            "model": "runware:100@1",
            "positivePrompt": "a serene landscape",
            "width": 1024,
            "height": 1024,
            "webhookURL": "https://api.example.com/webhooks/runware"
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
      "model": "runware:100@1",
      "positivePrompt": "a serene landscape",
      "width": 1024,
      "height": 1024,
      "webhookURL": "https://api.example.com/webhooks/runware"
    }
  ]'
```

**CLI**:

```bash
runware run runware:100@1 \
  positivePrompt="a serene landscape" \
  width=1024 \
  height=1024 \
  webhookURL=https://api.example.com/webhooks/runware
```

**JSON**:

```json
{
  "taskType": "imageInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "model": "runware:100@1",
  "positivePrompt": "a serene landscape",
  "width": 1024,
  "height": 1024,
  "webhookURL": "https://api.example.com/webhooks/runware"
}
```

### Authentication

**Webhooks authenticate through URL parameters.** Put a token, an API key or your own tracking parameters directly in the webhook URL:

**Authentication token**:

```text
https://api.example.com/webhooks/runware?token=your_auth_token
```

**API key**:

```text
https://api.example.com/webhooks/runware?apiKey=sk_live_abc123
```

**Custom parameters**:

```text
https://api.example.com/webhooks/runware?projectId=proj_789&userId=12345
```

> [!WARNING]
> Always use HTTPS for webhook URLs to ensure data transmission is encrypted. Validate authentication parameters in your endpoint before processing webhook data.

## Receiving notifications

The webhook POST body carries the complete JSON response for your task. **The structure matches the standard API response for that task type**:

**TypeScript**:

```typescript
import { createClient } from '@runware/sdk'

const client = await createClient({ apiKey: process.env.RUNWARE_API_KEY })
await client.connect()

const [result] = await client.run({
  imageUUID: '550e8400-e29b-41d4-a716-446655440000',
  imageURL: 'https://im.runware.ai/image/os/a14d18/ws/2/ii/550e8400-e29b-41d4-a716-446655440000.jpg',
  model: 'runware:100@1',
  width: 1024,
  height: 1024,
  cost: 0.004,
  createdAt: '2026-01-16T10:30:45Z'
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
            "imageUUID": "550e8400-e29b-41d4-a716-446655440000",
            "imageURL": "https://im.runware.ai/image/os/a14d18/ws/2/ii/550e8400-e29b-41d4-a716-446655440000.jpg",
            "model": "runware:100@1",
            "width": 1024,
            "height": 1024,
            "cost": 0.004,
            "createdAt": "2026-01-16T10:30:45Z"
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
      "imageUUID": "550e8400-e29b-41d4-a716-446655440000",
      "imageURL": "https://im.runware.ai/image/os/a14d18/ws/2/ii/550e8400-e29b-41d4-a716-446655440000.jpg",
      "model": "runware:100@1",
      "width": 1024,
      "height": 1024,
      "cost": 0.004,
      "createdAt": "2026-01-16T10:30:45Z"
    }
  ]'
```

**CLI**:

```bash
runware run runware:100@1 \
  imageUUID=550e8400-e29b-41d4-a716-446655440000 \
  imageURL=https://im.runware.ai/image/os/a14d18/ws/2/ii/550e8400-e29b-41d4-a716-446655440000.jpg \
  width=1024 \
  height=1024 \
  cost=0.004 \
  createdAt=2026-01-16T10:30:45Z
```

**JSON**:

```json
{
  "taskType": "imageInference",
  "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
  "imageUUID": "550e8400-e29b-41d4-a716-446655440000",
  "imageURL": "https://im.runware.ai/image/os/a14d18/ws/2/ii/550e8400-e29b-41d4-a716-446655440000.jpg",
  "model": "runware:100@1",
  "width": 1024,
  "height": 1024,
  "cost": 0.004,
  "createdAt": "2026-01-16T10:30:45Z"
}
```

## Building your endpoint

Your endpoint must answer with a status code between 200 and 299 to confirm receipt, **ideally within 5 seconds**. If the work is heavy, return the 200 first and do the work afterwards.

**Node.js (Express)**:

```javascript
const express = require('express');
const app = express();

app.use(express.json());

app.post('/webhooks/runware', (req, res) => {
  const webhookData = req.body;
  
  // Verify authentication
  if (req.query.token !== process.env.WEBHOOK_TOKEN) {
    return res.status(401).send('Unauthorized');
  }

  // Process asynchronously
  processWebhook(webhookData).catch(console.error);
  
  // Respond immediately
  res.status(200).send('OK');
});

async function processWebhook(data) {
  console.log('Task completed:', data.taskUUID);
  // Handle the completed task...
}

app.listen(3000);
```

**Python (Flask)**:

```python
from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/webhooks/runware', methods=['POST'])
def webhook():
    if request.args.get('token') != os.environ.get('WEBHOOK_TOKEN'):
        return 'Unauthorized', 401
    
    webhook_data = request.json
    process_webhook(webhook_data)
    
    return 'OK', 200

def process_webhook(data):
    print(f"Task completed: {data['taskUUID']}")
    # Handle the completed task...

if __name__ == '__main__':
    app.run(port=3000)
```

**PHP**:

```php
<?php
if ($_GET['token'] !== getenv('WEBHOOK_TOKEN')) {
    http_response_code(401);
    exit('Unauthorized');
}

$webhookData = json_decode(file_get_contents('php://input'), true);
processWebhook($webhookData);

http_response_code(200);
echo 'OK';

function processWebhook($data) {
    error_log("Task completed: " . $data['taskUUID']);
    // Handle the completed task...
}
```

## Retry behavior

A failed delivery is **retried with exponential backoff**. The first retry happens after 250ms and each one doubles the previous delay, up to a maximum of 8 seconds. Random jitter of up to 20% keeps simultaneous retries from colliding, and **retries stop as soon as a 2xx arrives**.

Example retry timing: 250ms, 500ms, 1s, 2s, 4s, then 8s for subsequent attempts.

## Testing locally

For local development, use ngrok to expose your local server:

```bash
# Start your local server
npm start

# In another terminal, start ngrok
ngrok http 3000

# Use the ngrok URL as your webhook
https://abc123.ngrok.io/webhooks/runware
```

You can also simulate webhook delivery with curl:

```bash
curl -X POST "http://localhost:3000/webhooks/runware?token=test" \
-H "Content-Type: application/json" \
-d '{"taskType":"imageInference","taskUUID":"test-uuid"}'
```

## Handling duplicates

**A webhook can arrive more than once**, from a network problem or a retry. Handle duplicates by tracking the task UUIDs you have already processed:

```javascript
const processed = new Set();

app.post('/webhooks/runware', (req, res) => {
const { taskUUID } = req.body;

if (processed.has(taskUUID)) {
  return res.status(200).send('Already processed');
}

processed.add(taskUUID);
processWebhook(req.body);
res.status(200).send('OK');
});
```

## Troubleshooting

If nothing arrives, check that **your endpoint is publicly reachable and answering with 2xx**. Look for the incoming requests in your server logs, and confirm the SSL certificate is valid on an HTTPS endpoint. In local development, ngrok gives you a reachable URL.

If your endpoint times out, return the 200 first and move the heavy work to a background job. **Delivery counts as successful only when a 2xx arrives inside the timeout window.**