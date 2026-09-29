import os
import sys
from typing import Any, Dict, List, Set  # noqa: F401  (Any/Dict used in startup hook below)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QPW_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
COMMON_SRC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if COMMON_SRC not in sys.path:
    sys.path.insert(0, COMMON_SRC)

_LTX_INF_SRC = os.path.join(THIS_DIR, "inference", "worker_ltx2_3", "src")
if os.path.isdir(_LTX_INF_SRC) and _LTX_INF_SRC not in sys.path:
    sys.path.insert(0, _LTX_INF_SRC)

try:
    from merge import main as _merge_main

    _merge_main()
except Exception as _merge_error:
    print(f"merge preflight failed: {_merge_error}")

from multi_model_runtime import MultiModelWorker  # type: ignore
from omni_config import (  # type: ignore
    OMNI_SHARED_BAKED_MODELS,
    get_effective_model_eta_seconds,
    get_effective_supported_models,
    omni_download_plan_excluding,
)

from inference_router import get_registry, preprocess

LTX23_CANONICAL_DOWNLOAD_MODEL = "ltx2_3_t2v"
LTX23_ALIASES: frozenset = frozenset(
    {"ltx2_3_i2v_distilled", "ltx2_3_flf2v", "ltx2_3_iv2v_distilled"}
)
LTX23_ALL_KEYS: frozenset = frozenset(
    {LTX23_CANONICAL_DOWNLOAD_MODEL} | LTX23_ALIASES
)

BAKED_MODELS: frozenset = frozenset(LTX23_ALL_KEYS)

MODELS_ROOT = os.environ.get("LTX2_3_MODELS_ROOT", "/app/models/ltx2_3")
DISTILLED_DIR = os.environ.get("LTX2_3_DISTILLED_LOCAL", f"{MODELS_ROOT}/distilled")
SPATIAL_DIR = os.environ.get("LTX2_3_SPATIAL_UPSAMPLER_LOCAL", f"{MODELS_ROOT}/spatial_upsampler")

DISTILLED_REPO = os.environ.get(
    "LTX2_3_DISTILLED_REPO", "rockapaper/LTX-2.3-Distilled-Diffusers_tenc_uint4_sdnq_r64_s16"
).strip()
SPATIAL_REPO = os.environ.get(
    "LTX2_3_SPATIAL_UPSAMPLER_REPO", "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
).strip()


def _ltx23_canonical_download_spec() -> Dict[str, Any]:
    """Single canonical LTX multi-snapshot spec (distilled + spatial only)."""
    return {
        "model": LTX23_CANONICAL_DOWNLOAD_MODEL,
        "multi_snapshot": True,
        "local_dir": MODELS_ROOT,
        "download_priority": 10,
        "snapshots": [
            {"repo_id": DISTILLED_REPO, "target_dir": DISTILLED_DIR},
            {"repo_id": SPATIAL_REPO, "target_dir": SPATIAL_DIR},
        ],
    }


def _build_download_plan() -> List[Dict[str, Any]]:
    """
    Runtime download plan: omni plan (for non-LTX omni siblings) minus LTX keys + shared baked utilities,
    with the LTX canonical spec prepended so re-bake paths are consistent even if baked weights missing.
    """
    excluded: Set[str] = set(BAKED_MODELS) | set(OMNI_SHARED_BAKED_MODELS)
    plan = omni_download_plan_excluding(excluded)
    return [_ltx23_canonical_download_spec()] + plan


class Ltx23MultiModelWorker(MultiModelWorker):
    """One HF multi-snapshot download (ltx2_3_t2v) satisfies all four advertised LTX keys."""

    def _model_weights_ready(self, model: str) -> bool:
        mk = (model or "").strip().lower()
        if mk in LTX23_ALIASES:
            return super()._model_weights_ready(LTX23_CANONICAL_DOWNLOAD_MODEL)
        return super()._model_weights_ready(model)

    def _model_in_download_plan(self, model: str) -> bool:
        mk = (model or "").strip().lower()
        if mk in LTX23_ALIASES:
            return super()._model_in_download_plan(LTX23_CANONICAL_DOWNLOAD_MODEL)
        return super()._model_in_download_plan(model)

    def _model_download_failed(self, model: str) -> bool:
        mk = (model or "").strip().lower()
        if mk in LTX23_ALIASES:
            return super()._model_download_failed(LTX23_CANONICAL_DOWNLOAD_MODEL)
        return super()._model_download_failed(model)

    async def _reprioritize_download_model(self, model: str) -> Dict[str, Any]:
        mk = (model or "").strip().lower()
        if mk in LTX23_ALIASES:
            return await super()._reprioritize_download_model(LTX23_CANONICAL_DOWNLOAD_MODEL)
        return await super()._reprioritize_download_model(model)


worker = Ltx23MultiModelWorker(
    model_registry=get_registry(),
    supported_models=get_effective_supported_models(),
    model_eta_seconds=get_effective_model_eta_seconds(),
    download_plan=_build_download_plan(),
    preprocess_hook=preprocess,
)

app = worker.create_fastapi_app(
    title=os.getenv("MODEL_KEY", "ltx2-3-diffusers-multi-worker")
)

import asyncio  # noqa: E402  (imported here so we don't disturb the import order above)
import logging  # noqa: E402

_lora_log = logging.getLogger("ltx_civitai_lora")


async def _warmup_civitai_loras_background() -> None:
    """
    Download every manifest LoRA into LTX2_3_LORA_CACHE if not already present.

    Runs once at startup, off the request path, so cold cold-starts don't block
    worker registration / heartbeat. Idempotent: ``_resolve_civitai_lora_file``
    returns the cached path if the file already exists.

    Soft-fails entry-by-entry — missing CIVITAI_API_KEY or transient HTTP errors
    do not crash the worker.
    """
    try:
        from omni_config import get_ltx2_3_civitai_loras  # type: ignore
        loras = get_ltx2_3_civitai_loras()
    except Exception as e:
        _lora_log.warning("manifest LoRA catalog unavailable (%r); skipping startup warmup", e)
        return
    if not loras:
        _lora_log.info("no LoRAs in manifest catalog; nothing to warm up")
        return

    if not (os.environ.get("CIVITAI_API_KEY") or "").strip():
        _lora_log.warning(
            "CIVITAI_API_KEY not set on this pod; skipping startup pre-warm of %d LoRA(s) "
            "(they will be fetched on first use)",
            len(loras),
        )
        return

    try:
        from ltx2_3_inference import _resolve_civitai_lora_file  # type: ignore
    except Exception as e:
        _lora_log.warning("cannot import _resolve_civitai_lora_file (%r); aborting warmup", e)
        return

    cache_dir = os.environ.get(
        "LTX2_3_LORA_CACHE",
        os.path.join(os.environ.get("LTX2_3_MODELS_ROOT", "/app/models/ltx2_3"), "lora_cache"),
    )

    loop = asyncio.get_running_loop()

    async def _warm_one(entry: Dict[str, Any]) -> None:
        nm = entry.get("name") or entry["url"]
        params: Dict[str, Any] = {
            "civitai_lora_model_url": entry["url"],
            "lora_cache_dir": cache_dir,
        }
        vid = entry.get("version_id")
        if vid is not None:
            params["civitai_version_id"] = int(vid)
        try:
            local_path, version_id = await loop.run_in_executor(
                None, _resolve_civitai_lora_file, params
            )
            _lora_log.info("warmed CivitAI LoRA %s v%s -> %s", nm, version_id, local_path)
        except Exception as e:
            _lora_log.warning("CivitAI LoRA warmup failed (%s): %r", nm, e)

    await asyncio.gather(*(_warm_one(e) for e in loras))


@app.on_event("startup")
async def _ltx_civitai_lora_warmup_on_startup() -> None:
    # Fire-and-forget; do NOT await, so registration + heartbeat are not blocked
    # by potentially-slow CivitAI downloads.
    asyncio.create_task(_warmup_civitai_loras_background())
