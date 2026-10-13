# Build ComfyUI worker images with GitHub Actions

Windows and Quickpod cannot reliably build these images (Docker Desktop on the
Windows laptop is stopped / too little RAM; Quickpod DIND is not the path).
Use this workflow on `ubuntu-latest` with `linux/amd64` Buildx, then pull the
tags onto a Quickpod GPU pod for testing.

Images are **runtime-weight**: ComfyUI + nodes are cloned at **build** time;
Hugging Face weights download on **first boot** after the worker registers.

## Images this workflow produces

| Workflow choice | Docker Hub tag | Worker folder | Runtime weights |
|---|---|---|---|
| `minimax_h3_comfy` | `pavanalur3/quickpod_worker:minimax_h3_comfy` | `worker_minimax_h3_comfy` | MiniMax H3 fused turbo INT8 + Seed Hunter extras |
| `ltx25_int8_comfy` | `pavanalur3/quickpod_worker:ltx25_int8_comfy` | `worker_ltx25_int8_comfy` | Lightricks LTX-2.5 INT8 AllModes |
| `qwen_image_21_turbo_comfy` | `pavanalur3/quickpod_worker:qwen_image_21_turbo_comfy` | `worker_qwen_image_21_turbo_comfy` | AtomicChat Qwen Image 2.1 Turbo **Q8_0 GGUF** + Comfy-Org CLIP/VAE |
| `minimax_music3_comfy` | `pavanalur3/quickpod_worker:minimax_music3_comfy` | `worker_minimax_music3_comfy` | MiniMax Music 3 **int8 convrot DiT** + pruned int8 encoder + DAV VAE |
| `all` | all four tags above, four parallel jobs | | |

Each push also tags `:name-<git sha>` so you can pin a build.

Base image for all four: `pavanalur3/deps-base-new:py311-cu128`.

## One-time GitHub setup

1. Repo **Settings → Secrets and variables → Actions**.
2. Add repository secrets:
   - `DOCKERHUB_USERNAME` — Docker Hub user (`pavanalur3`)
   - `DOCKERHUB_TOKEN` — Docker Hub **access token** with Read & Write (not the account password)
3. Confirm GitHub Actions is enabled for this repo.
4. The workflow file is `.github/workflows/build-comfy-workers.yml`. It only
   runs on **workflow_dispatch** (manual). It does not run on push.

The GH runner must be able to **pull** the private/public base image
`pavanalur3/deps-base-new:py311-cu128`. If that image is private, the same
Docker Hub token must have pull access.

## How to run a build

1. Open the repo on GitHub → **Actions**.
2. In the left sidebar choose **Build ComfyUI worker images**.
3. Click **Run workflow**.
4. Branch: `main` (or the branch that contains the worker folders).
5. **Which image to build and push**:
   - First time: pick `all` (four parallel jobs, ~1–2 hours each).
   - Later: pick one worker after a small code change.
6. Click **Run workflow**.

Expect:

- Disk-free step first (`jlumbroso/free-disk-space`) so ComfyUI + node clones fit.
- `docker/setup-buildx-action` then `docker/login-action`.
- `docker/build-push-action` with `platforms: linux/amd64`, `push: true`.
- Job timeout **360 minutes**. MiniMax H3 is the heaviest (many custom nodes).
  Music 3 is the lightest (native ComfyUI, no extra node packs).
- `fail-fast: false` so one failed worker does not cancel the other three.

If a job fails with `no space left on device`, re-run that single worker; the
free-disk-space action is already enabled. Do not enable GitHub cache for these
images unless you have verified runner disk headroom.

## After the images are on Docker Hub

On a Quickpod (or any) GPU pod:

```bash
docker pull pavanalur3/quickpod_worker:qwen_image_21_turbo_comfy
docker pull pavanalur3/quickpod_worker:minimax_h3_comfy
docker pull pavanalur3/quickpod_worker:ltx25_int8_comfy
docker pull pavanalur3/quickpod_worker:minimax_music3_comfy
```

Quickpod templates in `backend/server_controller/src/templates.json` must:

- Use those image tags.
- Map ports **8000** (FastAPI) and **8188** (ComfyUI UI).
- Set `MODEL_KEY` to `qwen_image_21_turbo`, `minimax_h3`, `ltx25_int8`, or `minimax_music3`.
- Pass `CONTROLLER_URL`, `CONTROLLER_SHARED_SECRET`, `CONTROLLER_API_KEY`, `HF_TOKEN`.
- **Start command**: `/app/worker/src/start.sh`  
  Quickpod often overrides the Dockerfile `CMD`. If you leave the template
  `cli_command` as a raw `uvicorn ...` line, the process may start with the
  wrong working directory. `start.sh` always `cd /app/worker` first.

After you create the four Quickpod templates, paste the real UUIDs into
`backend/server_controller/src/server_config.py`:

```
MINIMAX_H3_COMFY_TEMPLATE_UUID=...
LTX25_INT8_COMFY_TEMPLATE_UUID=...
QWEN_IMAGE_21_TURBO_COMFY_TEMPLATE_UUID=...
MINIMAX_MUSIC3_COMFY_TEMPLATE_UUID=...
```

Disk sizes already configured: H3 250 GB, LTX 120 GB, Qwen turbo 80 GB, Music 3 80 GB.

First boot downloads weights **after** the worker registers so the controller
does not kill the pod. `/run` returns 503 until ComfyUI is ready.

## Qwen Image 2.1 Turbo notes

Attached graph `qwenImage21Turbo8Steps_turbo8StepV10.json` is the default.

**AtomicChat GGUF can be used, with these constraints:**

- Swap **only** the UNET: `UNETLoader` → `UnetLoaderGGUF` pointing at
  `Qwen-Image-2.1-Turbo-Abliterated-Uncensored-Q8_0.gguf` in `models/unet/`.
- Keep Comfy-Org **CLIP** (`qwen3vl_8b_int8_convrot.safetensors`, type `qwen_image`)
  and **VAE** (`qwen_image_2.1_vae_bf16.safetensors`). AtomicChat CLIP GGUF /
  `mmproj` files are llama.cpp-only and cannot replace `CLIPLoader`.
- Pin **molbal/ComfyUI-GGUF** (`b77b9d8124656aa1a548d95c617da4d16cab1a2e`).
  city96’s pack does not know `qwen_image` / `qwen_image21` and will raise
  `Unexpected architecture type`.
- AtomicChat’s README still says the GGUF has not been run in ComfyUI. If the
  file’s architecture tag is unexpected, the first `/run` will fail at UNET
  load; the rest of the 8-step turbo graph (ManualSigmas, SamplerCustom euler,
  cfg 1) stays the same.

Optional `/run` params: `unet_name` (other AtomicChat quants you copy in),
`width`/`height`/`megapixels`/`aspect_ratio`, `seed`, `cfg`,
`first_frame_url` / `init_image` (wired to `TextEncodeQwenImage21.image_1`).

## MiniMax Music 3 notes

Two graphs were attached. **Use `audio_minimax_music_3.json`** (official
template): same inner graph as `04479f67187d.json`, plus `SaveAudioAdvanced`
and tiled VAE decode. Default DiT is **int8 convrot** for 16 GB-class GPUs
(`params.unet_name=fp16` switches back to the official fp16 DiT if you add
that file to the pod). `params.tiled_decode=true` by default.

Pass `params.caption` (or the job `prompt`) and `params.lyrics`.
`max_duration` default 60 seconds (model supports up to ~300).

## Regenerating a Dockerfile after lock changes

From each worker folder:

```bash
python src/generate_dockerfile.py
```

Commit the generated `Dockerfile` before you trigger Actions. The workflow
builds the committed file; it does not run `generate_dockerfile.py`.
