# WAN 2.2 I2V: quantization, offload, and OOM — root causes and fix plan

## Short answer

- **[`quantize_wan.py`](backend/quickpod_individual_worker/worker_wan_22_fp8_lightx2v_i2v_multi/src/inference/worker_wan_22_fp8_lightx2v_i2v/src/quantize_wan.py)** is largely **doing the right thing** for LoRA fuse → per-component TorchAO quant → save. The **text_encoder “fix”** (keep merged **BF16** UMT5, do **not** TorchAo-quantize it) is **correct** for Transformers’ load/init path (`_init_weights` / `normal_` on quantized tensors → `NotImplementedError`).
- The pain comes from **three separate mechanisms** that interact badly:
  1. **TorchAO `Float8Tensor` / affine quant tensors** + **Accelerate CPU offload hooks** (`set_module_tensor_to_device` → `tensor.to(dtype)`) → **hard crash** on sequential / model offload.
  2. **`balanced` offload** in [`inference.py`](backend/quickpod_individual_worker/worker_wan_22_fp8_lightx2v_i2v_multi/src/inference/worker_wan_22_fp8_lightx2v_i2v/src/inference.py) pins **both** `transformer` **and** `transformer_2` on GPU — for 14B-class I2V with long sequences, that **often OOMs 24 GB** even with FP8 **weights** (activations + two stacks dominate).
  3. **`safe_serialization=True`** for TorchAO exports is a **different failure mode** (round-trip / loader support), not the fix for offload.

---

## 1. What the code actually does (audit)

### 1.1 Quantization script — [`quantize_wan.py`](backend/quickpod_individual_worker/worker_wan_22_fp8_lightx2v_i2v_multi/src/inference/worker_wan_22_fp8_lightx2v_i2v/src/quantize_wan.py)

| Step | Behavior | Assessment |
|------|----------|------------|
| Base merge on CPU | `WanImageToVideoPipeline.from_pretrained(..., device_map="cpu")` | Safe for RAM |
| LoRA load + `fuse_lora` on `transformer` / `transformer_2` | Weights merged into BF16 staging | Good |
| Staging `save_pretrained(STAGING_DIR, safe_serialization=False)` | Large `.bin` tree | Expected |
| `text_encoder` | **Copied** from staging; **not** reloaded with `TorchAoConfig` | **Correct** — avoids Transformers init on quantized tensors |
| `transformer` / `transformer_2` | `AutoModel.from_pretrained` + `DiffusersTorchAoConfig` + `save_pretrained(..., safe_serialization=False)` | Matches diffusers “quantize diffusion transformer” pattern |

**`WAN_QUANT_TEXT_ENCODER_MODE`**: intentionally **ignored** with a warning if set — the script never applies TorchAO to TE; that is **by design**, not an incomplete TODO.

**`safe_serialization=False` (forced on quantized saves, line ~325)**: custom TorchAO tensor types historically **do not** round-trip cleanly through safetensors the way plain `float32`/`bfloat16` tensors do. If you switch to `True` for Hub convenience, expect **load errors, partial loads, or subtle bugs** unless your exact `diffusers`/`torchao` pair officially supports safetensors for those checkpoints.

### 1.2 Inference — [`inference.py`](backend/quickpod_individual_worker/worker_wan_22_fp8_lightx2v_i2v_multi/src/inference/worker_wan_22_fp8_lightx2v_i2v/src/inference.py)

- **TorchAO `.bin` path**: loads `transformer` / `transformer_2` via `AutoModel`, TE via `_load_wan_text_encoder_from_folder` (forces `UMT5EncoderModel` when config says so — **correct** vs wrong full `UMT5Model`).
- After load (torchao-bin case): **`pipe.to("cpu")`** then `configure_wan_pipe_memory` — normalizes devices and avoids some tie/device bugs before offload.
- **`sequential` / default `model` offload**: uses diffusers → accelerate hooks that **cast/moves** parameters; **TorchAO `Float8Tensor` does not implement `aten.to` dtype dispatch** → `NotImplementedError` (your original traceback).
- **`balanced`**: appends `transformer` and `transformer_2` to `_exclude_from_cpu_offload`, then **`.to(cuda)` both**, then `enable_sequential_cpu_offload` for the **rest**. So **both** denoising stacks stay resident on GPU — minimum VRAM is **high**, especially with **long `num_frames`** and **attention** cost.

### 1.3 Dependencies — [`requirements.txt`](backend/quickpod_individual_worker/worker_wan_22_fp8_lightx2v_i2v_multi/src/requirements.txt)

- `diffusers @ main`, `accelerate` git, `torch==2.11.0`, unpinned `torchao` → **moving targets**. Log line **“Unable to import `torchao` Tensor objects”** indicates **registration / version skew** and should be treated as **high priority** to pin compatible versions.

---

## 2. Why you see each symptom

### A. Sequential / model offload + TorchAO checkpoint → crash

**Cause**: Not a bad quantize script; **Accelerate + TorchAO tensor subclass** incompatibility on parameter `.to(dtype)` during `pre_forward`.

**Implication**: With **current** TorchAO FP8 shards, **you cannot rely on** `enable_sequential_cpu_offload` / `enable_model_cpu_offload` **for modules that hold Float8 weights**, unless upstream fixes dispatch or diffusers stops casting those params.

### B. Balanced → OOM on RTX 4090 (24 GB)

**Cause**: **By design**, `balanced` keeps **both** experts on GPU. FP8 **weights** save memory vs BF16, but **activation memory** for video DiT (many latent frames) is large; two transformers on device **plus** peak activations **exceeds 24 GB** for typical `480×832` / high frame counts.

**4090 vs 5090**: 32 GB may survive where 24 GB does not — still tune `num_frames`, resolution, and attention settings.

### C. `safe_serialization=True` “worked at quant time” but inference broke later

**Cause**: Different serialization path for custom tensors; loaders may not reconstruct the same parameter types, or Hub/diffusers version on consumer differs from quant machine.

### D. Text encoder encoding errors vs `quantize_wan` TE policy

**Two different classes of bugs**:

- **Wrong architecture** (`UMT5Model` vs `UMT5EncoderModel`) → **fixed in inference** by `_load_wan_text_encoder_from_folder`.
- **TorchAO on TE during quant** → **avoided in `quantize_wan.py`** on purpose.

If encoding still fails with BF16 TE, the bug is **elsewhere** (tokenizer, prompt length, `device_map` injection fallback, weight tie warnings).

---

## 3. Design goals mapped to viable strategies

| Goal | TorchAO FP8 + current offload | Practical path |
|------|------------------------------|----------------|
| 24 GB, correct video, **no** crash | Sequential/model offload **broken**; balanced **OOM** | **BnB int8/4bit** for `transformer`/`transformer_2` **or** aggressive **activation** controls + **5090** tier for FP8+balanced |
| 32 GB | May work with **balanced** + tuned frames / slicing | Same as above; measure headroom |
| Hub upload, reproducible load | Prefer **one** serialization story per backend; **pin** versions in Docker | Document `quantize_*` + `requirements.lock` |

---

## 4. Fix plan (implementation order)

### Phase A — Reproducibility and diagnostics (blocking)

1. **Pin** `torchao`, `accelerate`, `diffusers` (stop floating `main` for production workers), **verify** `import torchao` with **no** “Unable to import Tensor objects” in the **same** image you run inference.
2. Add a **small VRAM probe** log after load: report **allocated/reserved** with `torch.cuda.mem_get_info()` and which offload mode is active — clarify OOM vs hook crash.

### Phase B — Inference: tiered offload policy (24 GB vs 32 GB)

1. **Document** that **TorchAO `.bin` + sequential/model offload is unsupported** until upstream fixes; default env docs should **not** promise sequential for FP8 TorchAO.
2. Implement **`WAN_I2V_OFFLOAD_MODE`** extensions, for example:
   - **`pin_high_only`** (name TBD): exclude + pin **only** `transformer` — **only viable** if the **other** stack’s weights use a backend compatible with offload hooks (e.g. BnB), or if upstream fixes TorchAO movement. **Pure dual TorchAO + pin one / offload other** may **still crash** when the offloaded expert uses Float8 hooks.
   - Prefer **BnB** checkpoint when **sequential/model** offload is required on 24 GB.

3. Tighten **activation** pressure (already partially there): `WAN_I2V_ATTENTION_SLICING`, VAE tiling/slicing, avoid `none` on 24 GB unless bench confirms.

### Phase C — Quantization: second pipeline for deploy reality

1. Add **`quantize_wan_bnb.py`**: same LoRA fuse + staging, then `BitsAndBytesConfig` (`load_in_8bit` / `load_in_4bit`) for `transformer` / `transformer_2`; optional TE 8-bit via `transformers.BitsAndBytesConfig` with **fallback** BF16 TE copy (mirror current TE safety).
2. **Validate** output video against BF16 baseline; note [diffusers#11006](https://github.com/huggingface/diffusers/issues/11006) risk for Wan + quant.

### Phase D — Inference loader for BnB exports

1. Detect BnB folders (`quantization_config` in saved configs) vs TorchAO `.bin`.
2. Load **`AutoModel.from_pretrained`** without assuming TorchAO-only; ensure **`bitsandbytes`** in worker requirements.
3. Re-test **`sequential`** / **`model`** offload on 4090 — BnB Linear weights typically survive `set_module_tensor_to_device` better than `Float8Tensor`.

### Phase E — Optional upstream / long-term

1. Track **PyTorch AO** + **Accelerate** issues for `Float8Tensor.to(dtype)`; if fixed, re-evaluate TorchAO + offload on pinned versions.
2. If diffusers adds **block-level CPU offload** for `WanTransformer3DModel`, integrate as a middle ground — **verify in diffusers source** before relying on it.

---

## 5. Answers to your direct questions

- **Is `quantize_wan.py` the issue?** **Not** for TE handling or LoRA fuse; the main issues are **offload + tensor type** and **VRAM of dual pinned transformers**, not the quant recipe itself.
- **Are the TE fixes right?** **Yes**: BF16 TE in the bundle + **encoder-only** class in inference is the standard-safe combination for this stack.
- **Why so many issues?** You are combining **(1)** dual large transformers, **(2)** aggressive VRAM savings via **subclassed quantized weights**, **(3)** diffusers’ offload **hooks** that assume ordinary `Tensor` casts, and **(4)** unpinned bleeding-edge deps — each layer can fail independently.

---

## 6. Suggested acceptance criteria

- **4090 (24 GB)**: One documented path (**BnB 8-bit or 4-bit** + sequential/model offload OR BF16 staging + strong slicing) that completes a reference job (fixed resolution, frames, steps) **without OOM** and **without** `Float8Tensor` dispatch errors.
- **5090 (32 GB)**: Optional **TorchAO FP8 + balanced** path if benchmarks show headroom; otherwise same as 4090 for one image.
- **Hub**: One quant script output layout + **frozen** `requirements` string in README or Docker label for that revision.
