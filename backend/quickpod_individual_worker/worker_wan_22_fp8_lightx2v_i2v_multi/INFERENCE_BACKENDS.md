# WAN 2.2 FP8 LightX2V I2V — swappable inference backends

Omni multi-worker resolves the I2V entrypoint from the embedded bundle under  
`worker_wan_22_fp8_lightx2v_i2v_multi/src/inference/worker_wan_22_fp8_lightx2v_i2v/src/`.

## Environment

| Variable | Values | Default |
|----------|--------|---------|
| `WAN_I2V_INFERENCE_MODULE` | `inference.py`, `inference_sdnq_space_parity.py`, `inference_hub_space_torchao.py` | `inference.py` (this repo’s I2V multi **Dockerfile** sets `inference_sdnq_space_parity.py`, a thin re-export of `inference.py`) |
| `WAN_I2V_DISABLE_AUTO_DOWNLOAD` | `1` / `true` / `yes` | off — when set, V2/V3 **do not** run `snapshot_download` for WAN weights; missing dirs raise errors |

Only these basenames are accepted (no directories / path traversal).

**Tokenizer / `MODEL_LOCAL_DIR`:** V2 does **not** download a missing `spiece.model` (or any tokenizer shard) from a third-party WAN repo. The **rockapaper** snapshot under `MODEL_LOCAL_DIR` must include a complete `tokenizer/` (or whatever the pipeline expects). If `from_pretrained` fails, fix the bake / snapshot, not a hard-coded fallback URL.

**Space behavior reference (no AoTI in this path):** Gradio parity source lives at `worker_wan_22_fp8_lightx2v_i2v_multi/wan2-2-fp8da-aoti-preview2/` (optional checkout). That Space also uses **torchao** + **aoti** on load; this worker’s SDNQ path only mirrors schedulers, resize, `flow_shift`, and `pipe` kwargs.

### Auto-download (V2 / V3)

- **V2:** If `MODEL_LOCAL_DIR` is missing or does not look like a diffusers WAN tree, **`snapshot_download(MODEL_REPO, local_dir=MODEL_LOCAL_DIR)`** runs before `from_pretrained(..., local_files_only=True)`.
- **V3:** If `WAN_I2V_HUB_LOCAL_ONLY=1` **or** `WAN_I2V_HUB_LOCAL_DIR` is set, weights are snapshotted to `WAN_I2V_HUB_LOCAL_DIR` or `MODELS_DIR/hub_wan_i2v_<repo>__...` when that folder is empty/incomplete. Otherwise `from_pretrained(repo_id, local_files_only=False)` uses the normal Hugging Face cache (online).
- **RIFE / Real-ESRGAN:** When interpolation or upscaling is enabled, **`video_postprocessing.download_rife_model`** / **`download_realesrgan_model`** are always invoked first (they no-op if files already exist). RIFE **Python code** must still be present under `rife/` next to `video_postprocessing.py`; only weights are downloaded.

## Backends

| File | Model / stack | Disk (typical) | Extra deps |
|------|----------------|----------------|------------|
| **V1** `inference.py` | SDNQ rockapaper (local) | ~30 GB baked | `requirements.txt` |
| **V2** `inference_sdnq_space_parity.py` | Same as V1: re-exports `inference.py` (Gradio `SCHEDULER_MAP` parity in code; no torchao/AoTI). | ~30 GB | Same as V1 |
| **V3** `inference_hub_space_torchao.py` | Hub `WanImageToVideoPipeline` + **torchao** + **aoti** (`zerogpu-aoti/Wan2` fp8da) | ~70 GB class | `pip install -r requirements-wan-i2v-hub-space.txt` |

### V3 environment

| Variable | Purpose |
|----------|---------|
| `REPO_ID` / `WAN_I2V_HUB_REPO_ID` / `HF_REPO_WAN_I2V` | Hub model id (default: WAMU-Merge WAN2.2 I2V Lightning) |
| `WAN_I2V_HUB_LOCAL_DIR` | If set, WAN weights are stored/loaded from this directory (auto `snapshot_download` when empty) |
| `WAN_I2V_HUB_LOCAL_ONLY` | `1` → snapshot into default disk path under `MODELS_DIR` if needed, then `local_files_only=True` |
| `WAN_I2V_HUB_CPU_OFFLOAD` | `1` → sequential/model CPU offload instead of full `.to(cuda)` before quantize (may break Space parity) |

V3 post-RIFE/upscale uses the same **`video_postprocessing`** stack as V1/V2 (not Space `train_log` RIFE).

## Pod copy (minimal test)

**V1 / V2:** `inference.py` or `inference_sdnq_space_parity.py`, `video_postprocessing.py`, `rife/`, `sdnq/`, `requirements.txt`, model tree under `MODEL_LOCAL_DIR`.

**V3:** above plus `inference_hub_space_torchao.py`, `requirements-wan-i2v-hub-space.txt`, full Hub snapshot + compatible torch/torchao/aoti.

## API params (V2-focused)

See [API_REQUEST_FORMAT.md](API_REQUEST_FORMAT.md). Common additions for V2:

- `duration_seconds` / `duration` → `num_frames` (8–160 @ 16 fps) unless `num_frames` is set
- `last_image` / `last_frame_image` (presigned URL, same rules as `init_image`)
- `guidance_scale_2`, `scheduler` / `scheduler_name`, `flow_shift`
- `resize_mode`: `area480` (default) or `wan_space`
- `video_quality` (1–10) for `export_to_video`
- `frame_multiplier` 32 / 64 / 128 → enables RIFE with target output fps (via **video_postprocessing**); overrides `rife_target_fps` when set and `> 16`
- `output_type` (optional) — passed to `WanImageToVideoPipeline.__call__` when supported (e.g. `"np"` matches Gradio Space; default leaves diffusers default / PIL path)

V1 ignores `last_image` and V2-only keys safely where not read.
