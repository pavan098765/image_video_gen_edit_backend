# Server Controller Operations Runbook

## Fixed Fleet Guardrails

### What is enforced
- One active pod per machine.
- Slot cap based on:
  - `FIXED_FLEET_MACHINE_IDS` length when allowlist is configured.
  - Otherwise `FIXED_FLEET_MAX_PODS`.
- In-flight claim protection prevents parallel create races from over-allocating.

### Claim decision outcomes
Create requests are denied when any of the following is true:
- machine is outside `FIXED_FLEET_MACHINE_IDS` (if allowlist configured),
- machine already has an active pod,
- machine is already claimed by another in-flight create,
- total `active_pods + in_flight_claims` reached slot cap.

### Why allowlist matters
- With `FIXED_FLEET_MACHINE_IDS` set, placement is pinned to those exact machines.
- Without allowlist, cap is still enforced, but machine identity can vary per rentable offer availability.

## Dynamic ETA Learning

### Source of truth
- ETA learning is updated from successful `POST /worker/job_complete`.
- Runtime sample priority:
  1) `metrics.processing_time_seconds` from worker payload,
  2) fallback `Job.processing_time_seconds`.

### Rollout flags
- `SCHEDULER_DYNAMIC_ETA_ENABLED`
- `SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY`
- `SCHEDULER_DYNAMIC_ETA_ALLOWED_GROUPS`
- `SCHEDULER_DYNAMIC_ETA_MIN_SAMPLES`
- `SCHEDULER_DYNAMIC_ETA_ALPHA`
- `SCHEDULER_DYNAMIC_ETA_MIN_SECONDS`
- `SCHEDULER_DYNAMIC_ETA_MAX_SECONDS`

### Rollout sequence
1. Observe-only:
   - enable `SCHEDULER_DYNAMIC_ETA_ENABLED=true`
   - keep `SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY=true`
2. Canary:
   - set `SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY=false`
   - set `SCHEDULER_DYNAMIC_ETA_ALLOWED_GROUPS=<group_id>`
3. Full:
   - keep `SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY=false`
   - unset/empty `SCHEDULER_DYNAMIC_ETA_ALLOWED_GROUPS`

## Soft Churn Governor (No Hard Traffic Stop)

### Purpose
Reduce create/destroy churn risk under spikes without hard blocking all jobs.

### Behavior
- Tracks rolling pod ops from create and destroy events.
- When threshold is exceeded, scheduler admission becomes more conservative by:
  - increasing safety buffer,
  - increasing effective drain guard.

### Flags
- `SCHEDULER_SOFT_GOVERNOR_ENABLED`
- `SCHEDULER_SOFT_GOVERNOR_OPS_WINDOW_SECONDS`
- `SCHEDULER_SOFT_GOVERNOR_OPS_THRESHOLD`
- `SCHEDULER_SOFT_GOVERNOR_EXTRA_SAFETY_SECONDS`
- `SCHEDULER_SOFT_GOVERNOR_EXTRA_DRAIN_GUARD_SECONDS`

## Validation Checklist

### Pinning and slot controls
Use `GET /admin/scheduler/status` and verify:
- `fixed_fleet.claim_logic.one_active_pod_per_machine == true`
- `fixed_fleet.claim_logic.allowlist_enforced_when_configured` matches intent
- `fixed_fleet.active_pod_count <= slot cap`
- all `active_pods[].machine_id` are in allowlist when configured

### Churn metrics
Check `fixed_fleet.pod_operation_metrics`:
- `pod_creates_1h`
- `pod_destroys_1h`
- `pod_ops_rolling_1h`

Check `fixed_fleet.soft_governor`:
- `active`
- `ops_in_window`
- applied extra safety/drain values

### ETA learning
After traffic runs:
- verify logs contain `dynamic_eta_learned`.
- verify scheduler logs show dynamic ETA observation/use by model.

## Auto-prompt from asset (Nemotron VL via NVIDIA)

### Endpoint
- `POST /describe_asset_for_prompt` (Firebase Bearer). Body: `input_r2_key`, `media_type` (`image` only), optional `content_type`.
- Client flow: `POST /get_media_upload_urls` → PUT image bytes to R2 → call describe with the returned key. For video, upload a single JPEG frame the same way.
- The server **downloads** the object from R2 (after `head_object` size check), builds Nemotron-style `messages`: system `/think` for images, user `content` = text + `image_url` with `data:{mime};base64,...` (same pattern as NVIDIA samples). Default model `nvidia/nemotron-nano-12b-v2-vl`.

### Environment variables
- `ASSET_PROMPT_VLM_ENABLED` — set to `true` to enable the route (default off).
- `NVIDIA_API_KEY` or `NVIDIA_LLM_API` — Bearer `nvapi-...` (`NVIDIA_API_KEY` preferred; do not ship to mobile clients).
- `NVIDIA_CHAT_COMPLETIONS_URL` — optional (default `https://integrate.api.nvidia.com/v1/chat/completions`).
- `ASSET_PROMPT_NVIDIA_MODELS` or `NVIDIA_VISION_MODELS` — optional comma-separated models (default `nvidia/nemotron-nano-12b-v2-vl`).
- `NVIDIA_SYSTEM_PROMPT_IMAGE` / `NVIDIA_SYSTEM_PROMPT_VIDEO` — optional (defaults `/think` and `/no_think` for video parts if used later).
- `ASSET_PROMPT_VLM_TIMEOUT` — HTTP timeout seconds (optional; default higher for large base64 payloads).
- `ASSET_PROMPT_VLM_DAILY_LIMIT_PER_USER` — max describe calls per user per UTC day (optional, default 50).
- `ASSET_PROMPT_MAX_IMAGE_BYTES` — max R2 object size (optional).
- `NVIDIA_MAX_TOKENS` (default `4096`), `NVIDIA_TEMPERATURE` (default `1`), `NVIDIA_TOP_P`, `NVIDIA_FREQUENCY_PENALTY`, `NVIDIA_PRESENCE_PENALTY` — payload tuning.
- `NVIDIA_STREAM` — set `true` to use SSE streaming to the NVIDIA API (default off; server still returns one JSON response to clients).

### Prompt enhancement (same Nemotron stack)
- `PROMPT_ENHANCEMENT_ENABLED` — `true` to enable.
- `PROMPT_ENHANCEMENT_NVIDIA_MODELS` or `NVIDIA_TEXT_MODELS` — optional (default `nvidia/nemotron-nano-12b-v2-vl`).
- `PROMPT_ENHANCEMENT_TIMEOUT` — optional timeout seconds.
