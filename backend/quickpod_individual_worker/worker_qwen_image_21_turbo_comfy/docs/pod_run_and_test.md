# Qwen Image 2.1 Turbo pod: run and test

Hand this file to the model that writes the test plan. That model should produce SSH steps and prompts. It should not edit workflow JSON or add a photoshoot-specific graph.

All commands below are run **inside the pod** after SSH, unless noted. The container workdir is `/app/worker`. ComfyUI is `/app/comfyui`. Results land in `/workspace`.

## What the two workflows are

The graph is generic. Image roles exist only in the prompt for that job. Nothing in the node graph means "baby", "parent", or "backdrop".

| File | Who uses it |
| --- | --- |
| `workflow/qwen_image_2_1_turbo_multiref_api.json` | `/run`. Text-to-image when the job has no images. Reference edit when the job includes images. |
| `workflow/qwen_image_2_1_turbo_multiref_workflow.json` | ComfyUI in the browser on port **8188**. Copied to `/app/comfyui/user/default/workflows/` at startup. |

Shared settings: GGUF Turbo unet, Qwen3-VL text encoder, Qwen Image 2.1 VAE, euler, cfg 1, fixed 8-step Turbo sigmas. Do not change the sigma node.

`TextEncodeQwenImage21` accepts up to 16 images. This pod uses **10**.

| Order sent | Encoder input | Tag in the prompt |
| --- | --- | --- |
| 1st image | `images.image_1` | `<image1>` |
| 2nd image | `images.image_2` | `<image2>` |
| Nth image | `images.image_N` | `<imageN>` |

Rules the test prompts must follow:

- Do not skip a number. If image 2 is omitted, image 3 is encoded as `<image2>`.
- One image: the prompt does not need a tag. The model still sees that image.
- Two or more images: the prompt must say what to take from each `<imageN>`.
- The first image sets the canvas when `custom_size` is off. That is a size rule, not a role.
- `custom_size` on (the text-to-image default) uses `ResolutionSelector` width and height. On a reference edit, leave size fields unset so the canvas follows the first image. Set `width` and `height`, or `aspect_ratio`, only when the test wants a chosen frame.
- The worker does not rewrite the prompt. Paste the final instruction.
- cfg stays 1. The negative prompt has no effect at cfg 1.

## Weights

`models_manifest.json` downloads three files into `/app/comfyui/models/`:

| File | Repo |
| --- | --- |
| `unet/qwen-image-2.1-turbo-Q8_0.gguf` | `unsloth/Qwen-Image-2.1-Turbo-GGUF` |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | `Comfy-Org/Qwen-Image-2.1` |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | `Comfy-Org/Qwen-Image-2.1` |

These are multi-gigabyte. The GGUF is the large file. First boot can take a long time.

## 1. Watch the download

The container command is `src/start.sh`. That starts the API on port **8000**, which downloads missing weights and only then starts ComfyUI. Progress for that path is the **container stdout**, not the ComfyUI log. ComfyUI's own log (`/app/comfyui/comfyui_server.log`) stays empty until the weights exist.

In the platform log, or `docker logs -f` on the container, wait for these lines in order:

1. `Downloading Qwen Image 2.1 Turbo weights if missing...`
2. Hugging Face progress for each file, then `Downloaded ...` or `Already present: ...`
3. `All LTX-2.5 INT8 weights are present` (that sentence is the success line, even though these weights are Qwen)
4. `Starting ComfyUI server...`
5. `ComfyUI server is ready.`
6. `Validated workflows ... — all node types registered.`
7. `ComfyUI ready, worker is now READY`

While that runs, a second SSH shell can show bytes growing:

```bash
watch -n 5 'du -sh /app/comfyui/models /app/comfyui/models/unet /app/comfyui/models/text_encoders /app/comfyui/models/vae 2>/dev/null; echo ---; find /app/comfyui/models -type f -printf "%s %p\n" 2>/dev/null'
```

`Ctrl-C` stops `watch`. Files are done when all three paths exist and their sizes stop changing.

To force a visible Hugging Face bar **before** the server starts, only do this if `start.sh` is not already running (a second download will fight the first):

```bash
cd /app/worker
export HF_HUB_DISABLE_PROGRESS_BARS=0
python -u src/download_models.py
```

The bar is on that terminal. When it exits 0, start the server:

```bash
cd /app/worker
bash src/start.sh
```

If a file returns 401 or 403, export `HF_TOKEN` and run the download again. Do not set `SKIP_RUNTIME_MODEL_DOWNLOAD=1` unless the three files are already on disk.

## 2. Wait until jobs are accepted

From inside the pod:

```bash
curl -s http://127.0.0.1:8000/status
```

Submit jobs only when `startup_completed` and `comfyui_ready` are true and `comfyui_error` is null. Until then, `POST /run` returns 503.

ComfyUI itself:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/system_stats
```

`200` means the UI is up. From a browser, open `http://<pod-public-host>:8188`. Inside the pod the API is `http://127.0.0.1:8000`.

## 3. Submit a job

`POST /run` JSON:

| Field | Required | Value |
| --- | --- | --- |
| `job_id` | yes | unique string |
| `prompt` | yes | the edit or generation instruction |
| `model` | yes | `qwen_image_21_turbo` |
| `upload_url` | yes | `http://dummy-upload/local` skips the upload and leaves the file on the pod |
| `result_key` | yes | any string, for example `local/test.png` |
| `params` | no | see below |

`params` keys that matter:

| Key | Effect |
| --- | --- |
| `init_image` | One `https://` URL, or a list of URLs, in tag order |
| `image_1` … `image_10` | Same idea, one URL per slot. If any of these are set, `init_image` is ignored |
| `prompt` | Used only when the top-level `prompt` is empty |
| `width`, `height` | Both set: sample at that size. On a reference job this turns the explicit canvas on |
| `aspect_ratio` | For example `2:3`. Also turns the explicit canvas on |
| `custom_size` | `true` or `false`. Overrides the default |
| `seed` | integer |
| `resolution` | encoder pixel budget, default `1024`. `0` keeps each reference's own size, snapped to 32 |

Text-to-image (no images). Canvas is the resolution selector, default about 1024 square:

```bash
curl -s -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "t2i-001",
    "prompt": "A simple studio portrait, soft window light, plain gray backdrop, sharp focus.",
    "model": "qwen_image_21_turbo",
    "upload_url": "http://dummy-upload/local",
    "result_key": "local/t2i-001.png",
    "params": {"seed": 1}
  }'
```

Reference edit. Image order is the tag order. No width or height, so the canvas follows the first image:

```bash
curl -s -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "ref-001",
    "prompt": "Keep the person in <image1> unchanged. Place them in the room shown in <image2>. Match the room light.",
    "model": "qwen_image_21_turbo",
    "upload_url": "http://dummy-upload/local",
    "result_key": "local/ref-001.png",
    "params": {
      "seed": 1,
      "init_image": [
        "https://example.com/person.png",
        "https://example.com/room.png"
      ]
    }
  }'
```

Replace the example URLs with real `https` links the pod can download. A 202 response looks like `{"status":"accepted","job_id":"ref-001",...}`.

Poll until `jobs["ref-001"].status` is `completed` or `failed`:

```bash
curl -s http://127.0.0.1:8000/status
```

On success the PNG is the path logged as `Inference completed, output: ...`, under `/workspace`. ComfyUI also writes a copy under `/app/comfyui/output/` with prefix `Qwen_image_2.1_multiref`.

While it runs, ComfyUI's log is:

```bash
tail -n 80 /app/comfyui/comfyui_server.log
```

or `curl -s 'http://127.0.0.1:8000/logs/comfyui?tail=80'`.

## 4. Same test from the ComfyUI UI

1. Open port 8188 and load `qwen_image_2_1_turbo_multiref_workflow.json`.
2. Text-to-image: leave every `reference_*` node muted (they ship muted). Leave `custom_size` on. Type the prompt. Queue.
3. Reference edit: unmute `reference_1`, then `reference_2`, and so on, with no gaps. Set each file from `/app/comfyui/input` (upload in the UI, or copy files into that folder). Turn `custom_size` off unless the test wants a chosen width and height. Queue.

## Instructions for the model writing the baby-photoshoot test

Write a short test plan for a human on this pod. Do not change workflow files, node types, the Turbo schedule, or the model names.

The photoshoot is only the prompt plus which photo is uploaded in which slot. Pick a slot map for the test and state it in plain language, for example:

- `<image1>`: the baby's own photo, identity to keep
- `<image2>`: a parent photo, only if that person should appear
- `<image3>`: a backdrop or set photo
- further slots: extra views of the same baby, or a wrap and prop photo

Then give:

1. The SSH checks from sections 1 and 2, unchanged.
2. One text-to-image curl with a shoot-style prompt and no `init_image`, so the pod is proven before any reference photo is used.
3. Two or three reference curls. Each prompt is one paragraph, uses `<image1>`, `<image2>`, … in order, and says what stays and what changes. Do not send `width` or `height` on the first reference test.
4. One reference curl that also sets `aspect_ratio` to `2:3` or `3:4`, and say that this one uses an explicit canvas.
5. The UI steps from section 4 for the same slot map, including "turn custom_size off" on the reference run.
6. What to look at in `/workspace` and what failure looks like (`/status` job `failed`, or `/run` 503 because weights are still downloading).

Use real image URLs only if the human supplies them. Otherwise leave `https://...` placeholders and say which real photo belongs in each slot.
