#!/usr/bin/env python3
"""
SDNQ quantization for **Lightricks LTX-2.3** Diffusers checkpoints.

Supports **both** public layouts:
  - **Base (full)**: ``dg845/LTX-2.3-Diffusers`` → ``LTX2Pipeline`` / ``LTX2ConditionPipeline`` (same weights tree; model_index picks one class).
  - **Distilled I2V**: ``dg845/LTX-2.3-Distilled-Diffusers`` — weights are **image-to-video**; the public snapshot often sets ``_class_name`` to ``LTX2Pipeline`` even though inference should use ``LTX2ImageToVideoPipeline``. This script loads distilled runs with ``LTX2ImageToVideoPipeline.from_pretrained`` so the saved tree matches the worker.

For **base**, load uses ``DiffusionPipeline.from_pretrained`` (class from ``model_index.json``).
Failures print a **full traceback** (no silent except).

Optional — **merge Civitai (or local) LoRA into transformer weights**, then quantize the **fused** checkpoint
(no separate adapter in the saved tree). Requires ``pip install requests`` for Civitai URLs.

Requirements:
  pip install sdnq diffusers accelerate transformers safetensors huggingface_hub torch requests

Examples:
  # Quantize only the base (full) repo
  python quantize_sdnq_ltx2_3.py --repo dg845/LTX-2.3-Diffusers \\
      --variant base --dtype fp8 --svd-rank 64 --svd-steps 16

  # Quantize only distilled I2V
  python quantize_sdnq_ltx2_3.py --repo dg845/LTX-2.3-Distilled-Diffusers \\
      --variant distilled --dtype fp8 --svd-rank 64 --svd-steps 16

  # One command: download (if needed) and quantize **both** defaults sequentially
  python quantize_sdnq_ltx2_3.py --run-both-hf --dtype fp8 --svd-rank 64 --svd-steps 16

  # Custom pair of repos
  python quantize_sdnq_ltx2_3.py --run-both-hf \\
      --base-repo dg845/LTX-2.3-Diffusers \\
      --distilled-repo dg845/LTX-2.3-Distilled-Diffusers \\
      --dtype fp8 --svd-rank 64 --svd-steps 16

  # Distilled I2V + Civitai LoRA (download + bake + SDNQ). Token: --civitai-token or CIVITAI_API_KEY.
  export CIVITAI_API_KEY="..."
  python3 quantize_sdnq_ltx2_3.py --repo dg845/LTX-2.3-Distilled-Diffusers --variant distilled \\
      --civitai-lora-url "https://civitai.com/models/<id>/..." \\
      --lora-strength 1.0 --dtype fp8 --svd-rank 64 --svd-steps 16

  # Pin Civitai version id (model page URL) and add a local LoRA after the downloaded one:
  python3 quantize_sdnq_ltx2_3.py --repo dg845/LTX-2.3-Distilled-Diffusers --variant distilled \\
      --civitai-lora-url "https://civitai.com/models/<id>/..." --civitai-version-id 2772932 \\
      --lora /path/to/extra.safetensors:0.5 --dtype fp8 --svd-rank 64 --svd-steps 16

Notes:
  - VAE is not quantized. Spatial upsampler is a **separate** repo; quantize it with a dedicated pass
    on that snapshot if needed (this script still loads it as a normal Diffusers folder if you point
    ``--model-dir`` there).
  - Point inference worker env ``LTX2_3_FULL_LOCAL`` / ``LTX2_3_DISTILLED_LOCAL`` at the **output**
    directories from this script; install ``sdnq`` so Diffusers can load pre-quantized weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback
from typing import Any

import torch
from safetensors.torch import load_file

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from diffusers import DiffusionPipeline
except ImportError as e:
    raise SystemExit(f"diffusers required: pip install -U diffusers  ({e})") from e

from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Local SDNQ on path (same pattern as quantize_sdnq_multi.py)
# ---------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"Added local SDNQ source: {SDNQ_SRC_DIR}")

from sdnq import sdnq_post_load_quant, SDNQConfig  # noqa: E402
from sdnq.loader import save_sdnq_model  # noqa: E402

MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
LORA_SAVE_DIR = os.path.join(MODELS_DIR, "loras")

CIVITAI_API_BASE = "https://civitai.com/api/v1"
_CIVITAI_MODEL_PATH_RE = re.compile(
    r"civitai\.com/(?:models/|api/v1/models/)(\d+)", re.I
)
_CIVITAI_DOWNLOAD_VER_RE = re.compile(r"civitai\.com/api/download/models/(\d+)", re.I)

# LoRA tensor key suffixes (PEFT + diffusers); same set as quantize_sdnq_multi.py — no Wan block remapping here.
LORA_SUFFIXES = [
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    ".lora_down.default.weight",
    ".lora_up.default.weight",
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
]

LTX_LORA_STRIP_PREFIXES = [
    "base_model.model.transformer_2.",
    "base_model.model.transformer.",
    "transformer_2.",
    "transformer.",
]

DTYPE = torch.bfloat16
QUANTIZATION_DEVICE = "cuda"
RETURN_DEVICE = "cpu"

DEFAULT_BASE_REPO = "dg845/LTX-2.3-Diffusers"
DEFAULT_DISTILLED_REPO = "dg845/LTX-2.3-Distilled-Diffusers"

# model_index.json _class_name values we accept for each variant
VARIANT_EXPECTED_CLASSES: dict[str, frozenset[str]] = {
    "base": frozenset({"LTX2Pipeline", "LTX2ConditionPipeline"}),
    # HF distilled repo sometimes ships _class_name=LTX2Pipeline though weights are I2V-only.
    "distilled": frozenset({"LTX2ImageToVideoPipeline", "LTX2Pipeline"}),
}

MODULES_TO_NOT_CONVERT = [
    "correction_coefs",
    "prediction_coefs",
    "lm_head",
    "embedding_projection",
]

# model_index.json keys that must contain checkpoint shards (not just config / tokenizer json).
COMPONENTS_NEEDING_WEIGHT_BLOBS = frozenset({
    "text_encoder",
    "text_encoder_2",
    "transformer",
    "transformer_2",
    "vae",
    "vocoder",
    "image_encoder",
    "audio_encoder",
    "unet",
    "image_projection",
})


# ---------------------------------------------------------------------------
# Civitai LoRA URL → local .safetensors (aligned with worker ltx2_3_inference_v3)
# ---------------------------------------------------------------------------
def _civitai_headers(token: str | None) -> dict[str, str]:
    tok = (token or os.environ.get("CIVITAI_API_KEY") or os.environ.get("CIVITAI_API_TOKEN") or "").strip()
    if not tok:
        return {}
    return {"Authorization": f"Bearer {tok}"}


def _civitai_primary_safetensors_name(version_obj: dict[str, Any]) -> str:
    files = version_obj.get("files") or []
    for f in files:
        if f.get("primary") and str(f.get("name", "")).lower().endswith(".safetensors"):
            return str(f["name"])
    for f in files:
        if str(f.get("name", "")).lower().endswith(".safetensors"):
            return str(f["name"])
    raise ValueError("Civitai version has no .safetensors file")


def _pick_civitai_version(model_json: dict[str, Any], version_id: int | None) -> dict[str, Any]:
    versions = list(model_json.get("modelVersions") or [])
    if not versions:
        raise ValueError("Civitai model has no published versions")
    if version_id is not None:
        for v in versions:
            if int(v.get("id", 0)) == int(version_id):
                return v
        raise ValueError(f"civitai_version_id {version_id} not in model {model_json.get('id')}")
    prefer = [v for v in versions if "2.3" in str(v.get("baseModel", ""))]
    pool = prefer if prefer else versions
    return sorted(pool, key=lambda v: str(v.get("publishedAt") or ""), reverse=True)[0]


def _civitai_download_version(
    version_id: int, weight_basename: str, cache_dir: str, headers: dict[str, str]
) -> str:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("pip install requests for Civitai LoRA download")
    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", weight_basename)
    dest = os.path.join(cache_dir, f"civitai_v{version_id}_{safe}")
    if os.path.isfile(dest) and os.path.getsize(dest) > 4096:
        print(f"   Civitai LoRA cache hit: {dest}")
        return dest
    url = f"https://civitai.com/api/download/models/{int(version_id)}"
    r = requests.get(url, headers=headers, stream=True, timeout=600)
    if r.status_code == 401:
        r.close()
        raise RuntimeError(
            "Civitai returned 401 — set CIVITAI_API_KEY or pass --civitai-token."
        )
    r.raise_for_status()
    tmp = dest + ".partial"
    try:
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    print(f"   Civitai LoRA saved: {dest}")
    return dest


def resolve_civitai_lora_file(
    raw_url: str,
    *,
    civitai_version_id: int | None,
    token: str | None,
    cache_dir: str,
) -> str:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("pip install requests for Civitai LoRA download")
    raw = raw_url.strip()
    headers = _civitai_headers(token)
    mv = _CIVITAI_DOWNLOAD_VER_RE.search(raw)
    if mv:
        version_id = int(mv.group(1))
        vr = requests.get(
            f"{CIVITAI_API_BASE}/model-versions/{version_id}",
            headers=headers,
            timeout=120,
        )
        vr.raise_for_status()
        wn = _civitai_primary_safetensors_name(vr.json())
    else:
        mm = _CIVITAI_MODEL_PATH_RE.search(raw)
        if not mm:
            raise ValueError(
                f"Civitai URL not recognized (need .../models/{{id}} or .../download/models/{{vid}}): {raw[:120]}"
            )
        model_id = int(mm.group(1))
        mr = requests.get(
            f"{CIVITAI_API_BASE}/models/{model_id}",
            headers=headers,
            timeout=120,
        )
        mr.raise_for_status()
        ver = _pick_civitai_version(mr.json(), civitai_version_id)
        version_id = int(ver["id"])
        wn = _civitai_primary_safetensors_name(ver)
    return os.path.abspath(_civitai_download_version(version_id, wn, cache_dir, headers))


def parse_lora_path_strength(spec: str, default_strength: float) -> tuple[str, float]:
    parts = spec.rsplit(":", 1)
    if len(parts) == 2:
        try:
            return parts[0], float(parts[1])
        except ValueError:
            pass
    return spec, default_strength


def inject_lora_ltx_transformer(
    model: torch.nn.Module, lora_path: str, strength: float, label: str = ""
) -> tuple[int, int]:
    """Bake LoRA into linear weights. LTX-safe: strip PEFT prefixes only (no Wan block remapping)."""
    tag = f" ({label})" if label else ""
    print(f"   Injecting LoRA{tag}: {os.path.basename(lora_path)} strength={strength}")
    raw_sd = load_file(lora_path, device="cpu")
    all_keys = list(raw_sd.keys())
    if all(k.startswith("diffusion_model.") for k in all_keys):
        raise RuntimeError(
            "Full ComfyUI LoRA (diffusion_model.* keys only) is not supported for LTX bake; "
            "use Diffusers/PEFT .safetensors from Civitai."
        )

    sd = raw_sd
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    skipped_no_sfx = 0

    for key, val in sd.items():
        k = key
        for pfx in LTX_LORA_STRIP_PREFIXES:
            if k.startswith(pfx):
                k = k[len(pfx) :]
                break
        sfx = ""
        for s in LORA_SUFFIXES:
            if k.endswith(s):
                sfx = s
                k = k[: -len(s)]
                break
        if not sfx:
            skipped_no_sfx += 1
            continue
        if sfx in (
            ".lora_A.weight",
            ".lora_A.default.weight",
            ".lora_down.weight",
            ".lora_down.default.weight",
        ):
            pairs.setdefault(k, {})["down"] = val
        elif sfx in (
            ".lora_B.weight",
            ".lora_B.default.weight",
            ".lora_up.weight",
            ".lora_up.default.weight",
        ):
            pairs.setdefault(k, {})["up"] = val
        elif sfx == ".alpha":
            pairs.setdefault(k, {})["alpha"] = val

    del sd
    gc.collect()
    if skipped_no_sfx > 0:
        print(f"   Skipped {skipped_no_sfx} keys with unrecognised LoRA suffixes")

    model_params = dict(model.named_parameters())
    total = len([p for p in pairs.values() if "down" in p and "up" in p])
    applied = skipped_missing = skipped_incomplete = 0

    print(f"   Baking {total} LoRA layers into transformer...")
    for i, (base_key, parts) in enumerate(pairs.items()):
        if i % 50 == 0 and i > 0:
            print(f"   Baking... {i}/{len(pairs)}", end="\r", flush=True)
        if "down" not in parts or "up" not in parts:
            skipped_incomplete += 1
            continue
        weight_key = base_key + ".weight"
        if weight_key not in model_params:
            skipped_missing += 1
            continue
        param = model_params[weight_key]
        dtype = param.dtype
        down = parts["down"].to(dtype)
        up = parts["up"].to(dtype)
        rank = down.shape[0]
        scale = (parts["alpha"].float().item() / rank) if "alpha" in parts else 1.0
        if down.dim() == 4:
            down = down.squeeze(3).squeeze(2)
        if up.dim() == 4:
            up = up.squeeze(3).squeeze(2)
        delta = (up @ down) * (scale * strength)
        if param.shape != delta.shape:
            try:
                delta = delta.view(param.shape)
            except Exception:
                del down, up, delta
                skipped_missing += 1
                continue
        with torch.no_grad():
            param.data.add_(delta)
        del down, up, delta
        applied += 1

    del pairs, model_params
    gc.collect()
    print()
    print(
        f"   Result: {applied}/{total} layers baked | {skipped_missing} not in model | "
        f"{skipped_incomplete} incomplete"
    )
    if applied == 0:
        raise RuntimeError("No LoRA weights were baked — check LoRA / base checkpoint compatibility.")
    if applied < max(1, int(total * 0.8)):
        print(f"   WARNING: only {applied}/{total} layers applied ({100 * applied // max(total, 1)}%)")
    return applied, total


def inject_lora_list_ltx(model: torch.nn.Module, lora_list: list[tuple[str, float]], label: str) -> None:
    for idx, (path, strength) in enumerate(lora_list, 1):
        print(f"   [{idx}/{len(lora_list)}] ", end="")
        inject_lora_ltx_transformer(model, path, strength, label=label)


def merge_loras_into_ltx_pipe(
    pipe: DiffusionPipeline,
    lora_t1: list[tuple[str, float]],
    lora_t2: list[tuple[str, float]],
) -> None:
    if not lora_t1 and not lora_t2:
        return
    print("\nBaking LoRA(s) into pipeline transformer weights (permanent merge before SDNQ)...")
    if lora_t1 and getattr(pipe, "transformer", None) is not None:
        inject_lora_list_ltx(pipe.transformer, lora_t1, "transformer")
        gc.collect()
    effective_t2 = lora_t2 if lora_t2 else lora_t1
    if effective_t2 and getattr(pipe, "transformer_2", None) is not None:
        inject_lora_list_ltx(pipe.transformer_2, effective_t2, "transformer_2")
        gc.collect()
    print("LoRA baking complete.\n")


def build_lora_merge_list(args: argparse.Namespace) -> tuple[list[tuple[str, float]], str]:
    """Returns (ordered list of path, strength), suffix for output dir (empty if no loras)."""
    default_s = float(args.lora_strength)
    items: list[tuple[str, float]] = []
    os.makedirs(LORA_SAVE_DIR, exist_ok=True)

    if getattr(args, "civitai_lora_url", None) and args.civitai_lora_url.strip():
        p = resolve_civitai_lora_file(
            args.civitai_lora_url.strip(),
            civitai_version_id=args.civitai_version_id,
            token=getattr(args, "civitai_token", None),
            cache_dir=LORA_SAVE_DIR,
        )
        items.append((p, default_s))

    for spec in args.lora or []:
        path, s = parse_lora_path_strength(spec.strip(), default_s)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise SystemExit(f"Not a file: {path!r}")
        items.append((path, s))

    if not items:
        return [], ""

    first_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", os.path.splitext(os.path.basename(items[0][0]))[0])[
        :48
    ]
    suffix = f"lora_{first_stem}"
    if len(items) > 1:
        suffix += f"_x{len(items)}"
    return items, suffix


def _folder_has_model_weights(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for _root, _, files in os.walk(folder):
        for f in files:
            fl = f.lower()
            if fl.endswith(".safetensors"):
                return True
            if fl.endswith(".bin") and "model" in fl:
                return True
            if fl.endswith(".safetensors.index.json"):
                return True
    return False


def validate_diffusers_weight_files(model_path: str) -> None:
    """
    Fail fast if the tree looks like a partial download (common for LTX: huge Gemma text_encoder).
    """
    idx_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(idx_path):
        raise RuntimeError(f"Missing model_index.json under {model_path}")
    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)
    errors: list[str] = []
    for name in sorted(index.keys()):
        if name.startswith("_"):
            continue
        if name not in COMPONENTS_NEEDING_WEIGHT_BLOBS:
            continue
        comp_dir = os.path.join(model_path, name)
        if not os.path.isdir(comp_dir):
            errors.append(f"  - {name}: folder missing ({comp_dir})")
            continue
        if not _folder_has_model_weights(comp_dir):
            listing = sorted(os.listdir(comp_dir))[:30]
            errors.append(
                f"  - {name}: no .safetensors / model *.bin weights (files: {listing})"
            )
    if errors:
        raise RuntimeError(
            "Diffusers snapshot looks incomplete (sub-model folders exist but weight files are missing).\n"
            "This often happens if the download was interrupted or only part of the repo was copied.\n"
            + "\n".join(errors)
            + "\n\nRepair examples:\n"
            f"  huggingface-cli download dg845/LTX-2.3-Diffusers --local-dir {model_path!r}\n"
            "Or delete that directory and re-run quantization with:\n"
            "  python quantize_sdnq_ltx2_3.py --repo dg845/LTX-2.3-Diffusers --redownload ...\n"
            "Gated repos: export HF_TOKEN=...\n"
            "To bypass this check (not recommended): --skip-snapshot-verify"
        )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), seconds % 60
    return f"{h}h {m}m {s:.2f}s"


def print_timing_table(title: str, rows: list[tuple[str, float]]) -> None:
    """Pretty-print labeled wall-clock segments (seconds)."""
    print(f"\n{title}")
    print("-" * 52)
    wlab = max(len(r[0]) for r in rows) if rows else 0
    for label, sec in rows:
        print(f"  {label:<{wlab}}  {format_time(sec)}")
    print("-" * 52)


def get_model_size_gb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total_bytes / (1024**3)


def get_dir_size_gb(directory: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)


def download_model(
    model_repo: str,
    token: str | None,
    revision: str | None,
    *,
    redownload: bool,
    skip_snapshot_verify: bool,
) -> tuple[str, float]:
    repo_name = model_repo.split("/")[-1]
    model_local_dir = os.path.join(MODELS_DIR, repo_name)

    idx = os.path.join(model_local_dir, "model_index.json")
    if os.path.isdir(model_local_dir) and os.path.isfile(idx):
        if redownload:
            print(f"Removing existing snapshot (--redownload): {model_local_dir}")
            shutil.rmtree(model_local_dir)
        else:
            print(f"Using existing snapshot at {model_local_dir}")
            if not skip_snapshot_verify:
                validate_diffusers_weight_files(model_local_dir)
                print("   Snapshot validation OK.")
            return model_local_dir, 0.0

    print(f"Downloading {model_repo} to {model_local_dir}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.perf_counter()
    snapshot_download(
        repo_id=model_repo,
        local_dir=model_local_dir,
        token=token,
        revision=revision,
        resume_download=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Download finished in {format_time(elapsed)}")
    if not skip_snapshot_verify:
        validate_diffusers_weight_files(model_local_dir)
        print("   Post-download snapshot validation OK.")
    return model_local_dir, elapsed


def peek_pipeline_class(model_path: str) -> str | None:
    p = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("_class_name") or "") or None
    except OSError:
        return None


def validate_variant(model_path: str, variant: str) -> None:
    """Raise if --variant is set (not auto) and _class_name does not match."""
    if variant == "auto":
        return
    cls = peek_pipeline_class(model_path)
    expected = VARIANT_EXPECTED_CLASSES.get(variant, frozenset())
    if not cls:
        raise ValueError(f"No _class_name in model_index.json under {model_path}")
    if cls in expected:
        note = ""
        if variant == "distilled" and cls == "LTX2Pipeline":
            note = " — snapshot label; quantized load uses LTX2ImageToVideoPipeline"
        print(f"   Variant check OK ({variant}): {cls}{note}")
        return
    # Relaxed: HF / diffusers bumps sometimes rename; keep LTX2 family rules
    if variant == "base" and cls.startswith("LTX2") and "ImageToVideo" not in cls:
        print(f"   Variant check OK ({variant}, relaxed): {cls}")
        return
    if variant == "distilled" and cls.startswith("LTX2") and "ImageToVideo" in cls:
        print(f"   Variant check OK ({variant}, relaxed): {cls}")
        return
    raise ValueError(
        f"For --variant {variant!r}, expected one of {sorted(expected)} (or compatible LTX2* name), "
        f"got _class_name={cls!r}. Use --variant auto to skip this check."
    )


def _load_as_ltx2_i2v(variant: str, model_path: str, cls_name: str | None) -> bool:
    """Distilled checkpoints must be loaded as LTX2ImageToVideoPipeline (worker matches this)."""
    if variant == "distilled":
        return True
    if variant != "auto":
        return False
    if cls_name == "LTX2ImageToVideoPipeline":
        return True
    # Mislabeled distilled folder on Hub
    if cls_name == "LTX2Pipeline" and "distilled" in os.path.basename(model_path.rstrip("/\\")).lower():
        return True
    return False


def load_ltx_pipeline(
    model_path: str, *, skip_snapshot_verify: bool, variant: str = "auto"
) -> DiffusionPipeline:
    print(f"\nLoading DiffusionPipeline from {model_path}...")
    cls_name = peek_pipeline_class(model_path)
    if cls_name:
        print(f"   model_index.json _class_name: {cls_name}")
    if not skip_snapshot_verify:
        validate_diffusers_weight_files(model_path)
        print("   Pre-load snapshot validation OK.")
    t0 = time.time()
    try:
        if _load_as_ltx2_i2v(variant, model_path, cls_name):
            from diffusers import LTX2ImageToVideoPipeline

            print("   Using LTX2ImageToVideoPipeline.from_pretrained (distilled I2V).")
            pipe = LTX2ImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
            )
    except Exception:
        print("ERROR: Pipeline load failed — full traceback:")
        traceback.print_exc()
        raise
    print(f"   Loaded class: {pipe.__class__.__name__} in {format_time(time.time() - t0)}")
    return pipe


def list_pipeline_modules(pipe: DiffusionPipeline) -> None:
    names = []
    for n in sorted(dir(pipe)):
        if n.startswith("_"):
            continue
        try:
            m = getattr(pipe, n, None)
        except Exception:
            continue
        if isinstance(m, torch.nn.Module):
            names.append((n, type(m).__name__))
    print("   nn.Module submodules on pipeline:")
    for n, t in names:
        print(f"      - {n}: {t}")


def quantize_component(
    component: torch.nn.Module,
    component_name: str,
    weights_dtype: str,
    svd_rank: int,
    svd_steps: int,
    use_svd: bool,
) -> torch.nn.Module:
    print(f"\n   Quantizing {component_name} ({weights_dtype} + SVD rank {svd_rank}, steps {svd_steps})...")
    pre_size = get_model_size_gb(component)
    print(f"   Original size: {pre_size:.2f} GB")

    t0 = time.time()
    component = sdnq_post_load_quant(
        component,
        weights_dtype=weights_dtype,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        torch_dtype=DTYPE,
        quantization_device=QUANTIZATION_DEVICE,
        return_device=RETURN_DEVICE,
        modules_to_not_convert=list(MODULES_TO_NOT_CONVERT),
        add_skip_keys=True,
    )
    elapsed = time.time() - t0
    post_size = get_model_size_gb(component)
    red = (1 - post_size / pre_size) * 100 if pre_size > 0 else 0
    print(
        f"   Quantized: {pre_size:.2f} GB -> {post_size:.2f} GB "
        f"({red:.1f}% reduction) in {format_time(elapsed)}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return component


def quantize_ltx_pipeline(
    pipe: DiffusionPipeline,
    output_dir: str,
    weights_dtype: str,
    svd_rank: int,
    svd_steps: int,
    use_svd: bool,
    quantize_text_encoder: bool,
    quantize_vocoder: bool,
    skip_transformer_2: bool,
) -> tuple[str, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SDNQ quantization")

    print(f"\n   GPU: {torch.cuda.get_device_name(0)}")

    q0 = time.perf_counter()

    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer = quantize_component(
            pipe.transformer, "transformer", weights_dtype, svd_rank, svd_steps, use_svd
        )
        gc.collect()

    if (
        not skip_transformer_2
        and hasattr(pipe, "transformer_2")
        and pipe.transformer_2 is not None
    ):
        pipe.transformer_2 = quantize_component(
            pipe.transformer_2, "transformer_2", weights_dtype, svd_rank, svd_steps, use_svd
        )
        gc.collect()

    if quantize_text_encoder and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = quantize_component(
            pipe.text_encoder, "text_encoder", weights_dtype, svd_rank, svd_steps, use_svd
        )
        gc.collect()

    if quantize_vocoder and hasattr(pipe, "vocoder") and pipe.vocoder is not None:
        pipe.vocoder = quantize_component(
            pipe.vocoder, "vocoder", weights_dtype, svd_rank, svd_steps, use_svd
        )
        gc.collect()

    print("\n   VAE / non-selected modules left at original precision (not quantized).")
    quantize_elapsed = time.perf_counter() - q0
    print(f"   All selected components done in {format_time(quantize_elapsed)}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving quantized pipeline to {output_dir}...")
    t_save0 = time.perf_counter()
    sdnq_config = SDNQConfig(
        weights_dtype=weights_dtype,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        add_skip_keys=False,
        modules_to_not_convert=list(MODULES_TO_NOT_CONVERT),
    )
    save_sdnq_model(
        pipe,
        output_dir,
        max_shard_size="5GB",
        is_pipeline=True,
        sdnq_config=sdnq_config,
    )
    save_elapsed = time.perf_counter() - t_save0
    print(f"   Saved in {format_time(save_elapsed)}")
    return output_dir, quantize_elapsed, save_elapsed


def default_output_dir(model_tag: str, dtype: str, use_svd: bool, svd_rank: int, suffix: str = "") -> str:
    svd_tag = "nosvd" if not use_svd else f"svd_r{svd_rank}"
    name = f"{model_tag}_sdnq_{dtype}_{svd_tag}"
    if suffix:
        name = f"{name}_{suffix}"
    return os.path.join(MODELS_DIR, name)


def run_one_checkpoint(
    *,
    model_path: str,
    model_tag: str,
    variant: str,
    output_dir: str | None,
    args: argparse.Namespace,
    timing_label: str = "this checkpoint",
) -> tuple[str, float]:
    validate_variant(model_path, variant)
    use_svd = not args.no_svd
    checkpoint_wall0 = time.perf_counter()

    t0 = time.perf_counter()
    lora_list, lora_out_suffix = build_lora_merge_list(args)
    sec_lora_prepare = time.perf_counter() - t0

    if output_dir:
        out = os.path.abspath(output_dir)
    else:
        out = default_output_dir(
            model_tag, args.dtype, use_svd, args.svd_rank, suffix=lora_out_suffix
        )

    print("\n" + "=" * 60)
    print("LTX-2.3 SDNQ quantization")
    print("=" * 60)
    print(f"   Input : {model_path}")
    print(f"   Variant: {variant}")
    print(f"   Dtype : {args.dtype}  SVD: {use_svd} (rank={args.svd_rank}, steps={args.svd_steps})")
    if lora_list:
        print(f"   LoRA  : merge {len(lora_list)} file(s) into transformer (then quantize)")
    print(f"   Output: {out}")
    print("=" * 60)

    t0 = time.perf_counter()
    pipe = load_ltx_pipeline(
        model_path, skip_snapshot_verify=args.skip_snapshot_verify, variant=variant
    )
    sec_load = time.perf_counter() - t0
    if args.verbose:
        list_pipeline_modules(pipe)

    t0 = time.perf_counter()
    if lora_list:
        merge_loras_into_ltx_pipe(pipe, lora_list, [])
    sec_bake = time.perf_counter() - t0

    _, sec_quantize, sec_save = quantize_ltx_pipeline(
        pipe,
        output_dir=out,
        weights_dtype=args.dtype,
        svd_rank=args.svd_rank,
        svd_steps=args.svd_steps,
        use_svd=use_svd,
        quantize_text_encoder=args.quantize_text_encoder,
        quantize_vocoder=args.quantize_vocoder,
        skip_transformer_2=args.skip_transformer_2,
    )

    sec_checkpoint_total = time.perf_counter() - checkpoint_wall0
    timing_rows: list[tuple[str, float]] = [
        ("LoRA prepare (resolve/download)", sec_lora_prepare),
        ("Pipeline load (from_pretrained)", sec_load),
        ("LoRA bake (merge into weights)", sec_bake),
        ("SDNQ quantize (components)", sec_quantize),
        ("Save pipeline (save_sdnq_model)", sec_save),
        (f"Subtotal ({timing_label})", sec_checkpoint_total),
    ]
    print_timing_table(f"=== Wall time — {timing_label} ===", timing_rows)

    try:
        o1 = get_dir_size_gb(model_path)
        o2 = get_dir_size_gb(out)
        print(f"\nDirectory size (approx): {o1:.2f} GB -> {o2:.2f} GB")
    except OSError:
        pass

    print("\nDone.")
    print(
        "Load with the same pipeline class as the source (e.g. LTX2Pipeline / "
        "LTX2ImageToVideoPipeline) and `from_pretrained` after importing `sdnq`."
    )
    return out, sec_checkpoint_total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize LTX-2.3 Diffusers checkpoints (base + distilled) with SDNQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", type=str, help="Single HF repo id to quantize")
    src.add_argument("--model-dir", type=str, help="Local directory with model_index.json")
    src.add_argument(
        "--run-both-hf",
        action="store_true",
        help=f"Quantize {DEFAULT_BASE_REPO} then {DEFAULT_DISTILLED_REPO} (override with --base-repo / --distilled-repo)",
    )

    p.add_argument(
        "--variant",
        choices=("auto", "base", "distilled"),
        default="auto",
        help="Expected pipeline class in model_index.json (strict check). Use 'base' for LTX-2.3-Diffusers, "
        "'distilled' for Distilled-Diffusers (accepts LTX2Pipeline or LTX2ImageToVideoPipeline in model_index; "
        "load always uses LTX2ImageToVideoPipeline).",
    )
    p.add_argument("--base-repo", type=str, default=DEFAULT_BASE_REPO)
    p.add_argument("--distilled-repo", type=str, default=DEFAULT_DISTILLED_REPO)

    p.add_argument("--dtype", choices=["uint4", "int8", "fp8"], default="fp8")
    p.add_argument("--svd-rank", type=int, default=64)
    p.add_argument("--svd-steps", type=int, default=16)
    p.add_argument("--no-svd", action="store_true", help="Disable SVD part of SDNQ")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--token", type=str, default=None, help="HF token (or HF_TOKEN env)")
    p.add_argument("--revision", type=str, default=None, help="HF revision (branch / tag / commit)")
    p.add_argument(
        "--quantize-text-encoder",
        action="store_true",
        help="Also run SDNQ on pipe.text_encoder (Gemma; experimental).",
    )
    p.add_argument(
        "--quantize-vocoder",
        action="store_true",
        help="Also run SDNQ on pipe.vocoder if present (experimental).",
    )
    p.add_argument(
        "--skip-transformer-2",
        action="store_true",
        help="Do not quantize transformer_2 even if the pipeline has one.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List pipeline nn.Module children after load.",
    )
    p.add_argument(
        "--redownload",
        action="store_true",
        help="With --repo: delete existing snapshot under work_space/models/<repo_name> and download again.",
    )
    p.add_argument(
        "--skip-snapshot-verify",
        action="store_true",
        help="Skip pre-load check for .safetensors/.bin in text_encoder/transformer/vae (not recommended).",
    )
    p.add_argument(
        "--civitai-lora-url",
        type=str,
        default="",
        help="Civitai model page or https://civitai.com/api/download/models/{versionId} — downloads primary .safetensors.",
    )
    p.add_argument(
        "--civitai-token",
        type=str,
        default=None,
        help="Civitai API Bearer token (or set CIVITAI_API_KEY / CIVITAI_API_TOKEN).",
    )
    p.add_argument(
        "--civitai-version-id",
        type=int,
        default=None,
        help="Pin Civitai model version when using a model page URL.",
    )
    p.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="Strength for --civitai-lora-url and for --lora entries without :strength suffix.",
    )
    p.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="PATH[:STRENGTH]",
        help="Local LoRA .safetensors (repeatable). Applied after Civitai URL LoRA if both set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    script_t0 = time.perf_counter()

    if args.run_both_hf:
        if args.output_dir:
            raise SystemExit(
                "--output-dir is not supported with --run-both-hf (each repo gets its own default "
                "output under models/). Run two separate invocations with --repo to set custom paths."
            )
        jobs = [
            ("base", args.base_repo),
            ("distilled", args.distilled_repo),
        ]
        total_hf = 0.0
        total_ckpt = 0.0
        for variant, repo in jobs:
            model_path, hf_sec = download_model(
                repo,
                token=token,
                revision=args.revision,
                redownload=args.redownload,
                skip_snapshot_verify=args.skip_snapshot_verify,
            )
            total_hf += hf_sec
            model_tag = repo.split("/")[-1]
            _, ck_sec = run_one_checkpoint(
                model_path=model_path,
                model_tag=model_tag,
                variant=variant,
                output_dir=None,
                args=args,
                timing_label=f"{variant} ({model_tag})",
            )
            total_ckpt += ck_sec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        script_total = time.perf_counter() - script_t0
        print_timing_table(
            "=== Wall time — full script (--run-both-hf) ===",
            [
                ("HF snapshots (download or cache)", total_hf),
                ("All checkpoints (load+bake+quantize+save)", total_ckpt),
                ("End-to-end (process + overhead)", script_total),
            ],
        )
        return

    hf_sec = 0.0
    if args.model_dir:
        model_path = os.path.abspath(args.model_dir)
        if not os.path.isdir(model_path):
            raise SystemExit(f"Not a directory: {model_path}")
        model_tag = os.path.basename(model_path.rstrip(os.sep))
    else:
        model_path, hf_sec = download_model(
            args.repo,
            token=token,
            revision=args.revision,
            redownload=args.redownload,
            skip_snapshot_verify=args.skip_snapshot_verify,
        )
        model_tag = args.repo.split("/")[-1]

    _, ck_sec = run_one_checkpoint(
        model_path=model_path,
        model_tag=model_tag,
        variant=args.variant,
        output_dir=args.output_dir,
        args=args,
        timing_label=f"{model_tag}",
    )
    script_total = time.perf_counter() - script_t0
    print_timing_table(
        "=== Wall time — full script ===",
        [
            ("HF snapshot (download or cache)", hf_sec),
            ("Checkpoint (load+bake+quantize+save)", ck_sec),
            ("End-to-end (process + overhead)", script_total),
        ],
    )


if __name__ == "__main__":
    main()
