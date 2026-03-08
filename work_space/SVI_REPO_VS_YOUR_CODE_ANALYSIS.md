# SVI Long Video: Repo vs Your Code — Analysis & Recommendations

## Executive Summary

**Inference count:** Both approaches use **N separate inference calls** for N clips. The repo does NOT do "one generation internally" — it loops and calls `pipe()` once per clip, same as you. For 4 clips → 4 inferences in both cases.

**Root cause of visible transitions:** Your code uses **pixel-space handoff only** (last frame → next clip's start_image). The repo uses **latent bridging** (`prev_last_latent`) — passing the raw denoised latent directly between clips without decode/re-encode. This is the main reason their transitions are smoother.

---

## 1. What the Repo Code Does Better

### 1.1 Latent Bridging (`prev_last_latent`) — **CRITICAL**

| Your Code | Repo Code |
|-----------|-----------|
| Last frame → decode to pixels → PIL → resize → next clip's `image` | Raw latent from clip N-1 → passed directly to clip N's conditioning |
| **Decode → re-encode round-trip** causes quality loss and artifacts at boundaries | **No round-trip** — latent flows directly into positions 1..K of the conditioning tensor |

**Why this matters:**  
When you pass the last frame as a pixel image, the next clip must:
1. VAE-encode it again (lossy)
2. Treat it as a new conditioning signal (slight distribution shift)

The repo instead injects `prev_last_latent` at positions 1..`num_motion_latent` of the conditioning tensor. The model was trained for this — SVI's "Error-Recycling" fine-tuning teaches the model to continue from its own latent output. No decode/re-encode = no boundary artifacts from that path.

**Your diffusers `WanImageToVideoPipeline` does NOT support `prev_last_latent`.** It only accepts a single `image` for I2V conditioning. This is a pipeline-level limitation.

---

### 1.2 Anchor Image

| Your Code | Repo Code |
|-----------|-----------|
| Clip 0: original image. Clip N>0: last frame of clip N-1 only | **Every clip** gets `anchor=original_image` at position 0 of conditioning |

**Why this matters:**  
The anchor keeps identity, color, and framing stable across many clips. Without it, small drift accumulates and transitions become more noticeable. The repo re-injects the original image every call; your code only uses it for clip 0.

**Your diffusers pipeline** takes one `image` per call — you cannot pass both anchor and motion frame separately.

---

### 1.3 Overlap Frame Handling

| Your Code | Repo Code |
|-----------|-----------|
| `overlap=1` — drop 1 frame at each join | `num_overlap_frame=4` (default) or 5 — drop 4–5 frames at each join |

**Why this matters:**  
The first few frames of each new clip are "warm-up" — least consistent with the previous clip. Dropping more of them hides the transition. With overlap=1, you keep almost all of that warm-up, so the join is more visible.

**You can implement this** — just change your `stitch_clips(overlap=1)` to `stitch_clips(overlap=4)` or `overlap=5`.

---

### 1.4 Multi-Frame Pixel Conditioning (`num_motion_frame`)

| Your Code | Repo Code |
|-----------|-----------|
| Single frame: last frame only | Can use last 4 frames: `current_input_image = video_frames[-4:]` |

**Repo default:** `num_motion_frame=4` — passes a list of 4 frames as `input_image`.  
**Your code:** Uses 1 frame only.

**Caveat:** `WanImageToVideoPipeline` (diffusers) typically expects a single image. You would need to check if it accepts a list; if not, this is pipeline-dependent.

---

### 1.5 Separate High/Low Noise Transformers + LoRAs

| Your Code | Repo Code |
|-----------|-----------|
| Single merged model (rockapaper SDNQ) | Separate `dit` (high noise) + `dit2` (low noise) with separate SVI LoRAs |

Wan2.2 has two transformer stages:
- **High noise:** structure, identity (steps 0–3)
- **Low noise:** texture, detail (steps 3+)

The repo loads separate LoRAs for each. Your merged model bakes both into one checkpoint. The structural benefit of two-phase denoising is preserved in the pipeline design; your merged model approximates it but may behave slightly differently.

---

### 1.6 CFG and Seed Strategy

| Your Code | Repo Code |
|-----------|-----------|
| `GUIDANCE_SCALE=1.5`, `SEED + clip_idx` | `cfg_scale=4.0` (or 5.0), `seed=clip_idx * seed_multiplier` |

Your model is LightX2V-distilled (steps=6, cfg=1.5). The repo uses native Wan2.2 (no LightX2V) with cfg=4–5. Different models, different sweet spots. Your settings are appropriate for your merged Lightning model.

---

## 2. What You Can Implement in Your Current Diffusers Code

These changes **do not** require switching pipelines:

### 2.1 Increase Overlap Frames (High Impact, Easy)

```python
# Change from:
final_frames = stitch_clips(all_clips, overlap=1)

# To:
NUM_OVERLAP_FRAME = 5   # Match repo / SVI Pro 2.0 recommendation
final_frames = stitch_clips(all_clips, overlap=NUM_OVERLAP_FRAME)
```

This drops the first 5 frames of each continuation clip instead of 1. Those frames are the least consistent; hiding them reduces visible transitions.

**Note:** Your total frame count changes:
- Before: `NUM_CLIPS * 81 - (NUM_CLIPS - 1) * 1`
- After:  `81 + (NUM_CLIPS - 1) * (81 - 5)` = `81 + 76*(NUM_CLIPS-1)` for 4 clips → 309 frames

---

### 2.2 Reduce or Disable Noise Augmentation

```python
# Try:
NOISE_AUG_STRENGTH = 0.0   # was 0.02
```

The repo does **not** add noise to the handoff frame. Your noise aug was meant to simulate SVI's error-recycling, but without latent bridging it may introduce extra inconsistency. Test with 0.0 and see if transitions improve.

---

### 2.3 Optional: Blend Last Few Frames at Join (Experimental)

You could blend the last N frames of clip N-1 with the first N frames of clip N at the stitch point. This is a post-processing heuristic, not in the repo, but can soften visible cuts:

```python
def blend_at_join(frames_before, frames_after, blend_frames=3):
    """Cross-fade the last blend_frames of before with first blend_frames of after."""
    # ... alpha blend from 1→0 and 0→1 over blend_frames
```

Use sparingly; too much blending can look blurry.

---

## 3. What You Cannot Implement Without Changing Pipelines

| Feature | Why Not Possible |
|---------|------------------|
| **prev_last_latent** | `WanImageToVideoPipeline` has no such parameter. Conditioning is image-only. |
| **Anchor image** | Pipeline accepts one `image` per call. No separate anchor slot. |
| **num_motion_latent** | Latent-level conditioning is internal to `WanVideoSviProPipeline`. |
| **Multi-frame input_image** | Diffusers Wan I2V typically expects a single image; would need to verify API. |

---

## 4. Recommended Path: Use DiffSynth's WanVideoSviProPipeline

To get **latent bridging** and **anchor image**, you need the SVI pipeline. You already have a reference script: `wan22_svi_long_video_inference.py` (in your Downloads).

### 4.1 Model Layout Difference

| Your Current Setup | SVI Pro 2.0 Setup |
|--------------------|-------------------|
| Single merged model: `rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64` | Original Wan2.2 weights + separate SVI LoRAs |
| No LoRA loading at runtime | `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors` + `_low_noise_lora_v2.0_pro.safetensors` |

### 4.2 Installation

```bash
git clone https://github.com/vita-epfl/Stable-Video-Infinity.git -b svi_wan22
cd Stable-Video-Infinity
pip install -e .
pip install flash_attn==2.8.0.post2   # Required — artifacts without it
```

### 4.3 Key Parameters (from repo / wan22_svi_long_video_inference.py)

```python
NUM_MOTION_LATENT = 1    # Latent frames bridged between clips (1 recommended)
NUM_OVERLAP_FRAME = 5    # Frames dropped from start of each new clip
NUM_MOTION_FRAME  = 1    # Pixel-space frames for input_image (1 for Pro 2.0)
anchor            = anchor_image   # Original image, every clip
prev_last_latent  = prev_last_latent  # From previous clip's output
```

---

## 5. Summary Table

| Aspect | Your Code | Repo Code | Can Fix in Yours? |
|--------|-----------|-----------|-------------------|
| Inference count | N calls for N clips | N calls for N clips | N/A (same) |
| Handoff | Pixel only (last frame) | **Latent + pixel** | No — need SVI pipeline |
| Anchor image | Only clip 0 | Every clip | No — need SVI pipeline |
| Overlap frames | 1 | 4–5 | **Yes** — increase to 5 |
| Noise on handoff | 0.02 | None | **Yes** — try 0.0 |
| Multi-frame input | 1 frame | Up to 4 frames | Unclear (diffusers API) |
| Pipeline | WanImageToVideoPipeline | WanVideoSviProPipeline | Must switch for full SVI |

---

## 6. Action Items

**Quick wins (stay on diffusers):**
1. Set `NUM_OVERLAP_FRAME = 5` and use it in `stitch_clips()`.
2. Set `NOISE_AUG_STRENGTH = 0.0` and compare.

**Full solution (smoother transitions):**
1. Use `wan22_svi_long_video_inference.py` with DiffSynth's `WanVideoSviProPipeline`.
2. Download original Wan2.2 weights + SVI Pro 2.0 LoRAs.
3. Run with `prev_last_latent`, `anchor`, and `num_overlap_frame=5`.
