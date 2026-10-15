# Quickpod NFS volume (planned)

**Status:** not wired for weights yet. MiniMax H3 already honors Triton cache on the volume when present. Do not assume `/data` exists in new start scripts, Dockerfiles, or downloaders.

Goal: one NFS volume attached at pod create, holding **MiniMax H3 + MiniMax Music 3 + LTX 2.5** weights (and later SolAttn Triton kernels). Comfy starts, finds files, loads to VRAM, **no Hugging Face download**. Size the quota to files + 5–10 GB headroom. Billing is on **provisioned quota**, not used bytes.

Quickpod / Podstack default mount is **`/data`**. Never `mkdir /data` in an image or start script — that hides a missing attach and writes to overlay disk.

## Layout (when enabled)

```text
/data/
  models/                         # Comfy folder_paths tree (shared by H3, Music 3, LTX)
    diffusion_models/
    text_encoders/
    vae/
    loras/MINIMAX/
    latent_upscale_models/
    vae_approx/
    unet/                         # only if Qwen GGUF shares this volume later
  triton/                         # SolAttn / Triton JIT cache (H3 Fast T2V)
```

Keep Comfy `output/` and `temp/` on the **pod disk** (`/app/comfyui/...`, `/workspace`), not NFS.

## Env vars (use these names in future code)

| Variable | Purpose |
|---|---|
| `QUICKPOD_DATA_DIR` | Mount root if not `/data`. Example: `/data`. |
| `MODELS_DIR` | Comfy weights root. Planned NFS: `/data/models`. Image default: `/app/comfyui/models`. |
| `SKIP_RUNTIME_MODEL_DOWNLOAD` | `1` when the volume is filled. Startup must not call Hub. Fail fast if a required dest is missing. |
| `TRITON_CACHE_DIR` | SolAttn kernel cache. NFS: `/data/triton`. Local fallback: `/root/.triton`. |
| `COMFYUI_EXTRA_MODEL_PATHS` | Optional path to `extra_model_paths.yaml` with `base_path: /data` if we do not symlink `MODELS_DIR`. |

H3 `start.sh` today: if `TRITON_CACHE_DIR` is unset or still `/root/.triton`, and `/data/triton` (or `$QUICKPOD_DATA_DIR/triton`) **already exists**, use it. It will not create `/data`.

## How Comfy should see weights

Pick one, do not mix:

1. `MODELS_DIR=/data/models` (downloaders and loaders already read this), or
2. Symlink `/app/comfyui/models` → `/data/models`, or
3. `extra_model_paths.yaml` `base_path: /data` with the same subfolders.

Loaders use **filenames only** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`). Dest paths in each worker `models_manifest.json` must match this tree.

## Fill rules

- Fill **once** from one pod or an offline machine. Then treat the volume as **read-mostly**.
- Several pods may **read** the same volume. Do not run two `hf_hub_download` / `download_models.py` into it at once.
- Do not put Hugging Face cache (`.incomplete`, `~/.cache/huggingface`) on the volume if we are sizing for 5–10 GB headroom.
- `download_models.py` “already present if size > 0” will skip a **truncated** file. NFS fill must copy complete files (or check sizes against Hub).

## Size (cost-sensitive)

Approximate GiB on disk, quality Music + H3 + LTX INT8, no Hub cache:

| Slice | ~GB |
|---|---|
| Music 3 quality (FP16 DiT + BF16 TE + VAE) | 22 |
| H3 (fused T2V + fl2va I2V + CLIP + VAEs + LoRA + upscaler) | 62 |
| LTX 2.5 INT8 | 40 |
| Files | **~124** |
| Headroom | **+10** |
| Volume | **~135 GB** |

If the product only offers 10 / 25 / 50 / 100 / 250 GB presets, pick **250** (100 is short). Triton cache is small (hundreds of MB to a few GB), not the size driver.

Drop Music INT8 (~12 GB) if 3090 always uses quality. Music + H3 only ≈ 94 GB files.

## What future PRs must not do

- Bake weights into Docker images “because NFS is not ready” **and** also download them at runtime onto NFS without `SKIP_RUNTIME_MODEL_DOWNLOAD`.
- Point `WORKSPACE_DIR` / Comfy output at `/data`.
- Assume gcc/Triton JIT can write only to `/root/.triton` — honor `TRITON_CACHE_DIR`.
- Concurrent Hub downloads into `/data/models`.
- `mkdir -p /data` in Dockerfile or start.sh.

## Enablement checklist (later)

1. Create volume, attach at pod create, confirm `/data` is the NFS mount (`mount | grep /data`).
2. Layout as above; copy weights once; `SKIP_RUNTIME_MODEL_DOWNLOAD=1`.
3. Set `MODELS_DIR=/data/models` (or symlink / extra_model_paths).
4. `mkdir /data/triton` on the volume, or `TRITON_CACHE_DIR=/data/triton`.
5. First H3 T2V after gcc image: SolAttn may JIT into that cache; later pods skip compile if GPU arch matches (Ampere vs Ada vs Blackwell are **separate** caches — do not share one cache dir across GPU generations if it causes kernel mismatch).
6. Smoke T2V/I2V with Hub disabled and no `download_models` progress.

Workers in scope: `worker_minimax_h3_comfy`, `worker_minimax_music3_comfy`, `worker_ltx25_int8_comfy`. Qwen Image is not on this volume unless we explicitly add `unet/` later.
