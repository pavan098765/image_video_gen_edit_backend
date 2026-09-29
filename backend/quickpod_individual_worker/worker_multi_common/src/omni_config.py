"""
Omni multi-worker: shared model lists, registry merge, download plan specs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

from image_registry import build_image_registry, image_preprocess_hook
from omni_runtime_manifest import load_runtime_manifest
from qwen_2511_lora_registry import build_qwen_lora_supplement_hub_files
from video_registry import build_hunyuan_registry, build_wan_registry

# Model ids must match worker_multi_common/build_inference_zips.py MODELS (multi_worker_dir +
# inference_primary_dir) and omni_manifest.json.
# When OMNI_SLIM_INFERENCE_SRC=1, static registry omits these; inference comes from R2
# zip after HF weights download (see omni_download_plan_excluding + inference_archive).
# OMNI_MULTI_IMAGE_ID must match the multi worker folder basename (set in Dockerfile).
OMNI_SLIM_DEFER_INFERENCE_TO_PLUGIN: frozenset = frozenset(
    {
        "qwen_image_edit_2511_rapid_aio",
        "z_image_turbo",
        "flux_klein",
        "sdxl_t2i",
        "diffusers_outpaint",
        "wan_22_fp8_lightx2v_t2v",
        "wan_22_fp8_lightx2v_i2v",
        "hunyan15_480p_t2v_distilled",
        "hunyan15_480p_i2v_step_distilled",
        "ltx2_3_t2v",
        "ltx2_3_i2v_distilled",
        "ltx2_3_flf2v",
        "ltx2_3_iv2v_distilled",
    }
)

MULTI_IMAGE_ID_TO_PRIMARY_MODEL: Dict[str, str] = {
    "worker_flux_klein_9b_multi": "flux_klein",
    "worker_qwen_image_edit_2511_rapid_aio_multi": "qwen_image_edit_2511_rapid_aio",
    "worker_z_image_turbo_multi": "z_image_turbo",
    "worker_stable_diffusion_xl_multi": "sdxl_t2i",
    "worker_stable_diffusion_xl_outpaint_multi": "diffusers_outpaint",
    "worker_wan_22_fp8_lightx2v_t2v_multi": "wan_22_fp8_lightx2v_t2v",
    "worker_wan_22_fp8_lightx2v_i2v_multi": "wan_22_fp8_lightx2v_i2v",
    "worker_hunyan15_480p_t2V_distilled_multi": "hunyan15_480p_t2v_distilled",
    "worker_hunyan15_480p_i2V_step_distilled_multi": "hunyan15_480p_i2v_step_distilled",
    "worker_ltx2_3_multi": "ltx2_3_t2v",
    # Shared-small omni bases: no primary in DEFER set — all deferred models use R2 only.
    "worker_basic_utils_multi": "",
    "worker_video_upscalers_multi": "",
    "worker_face_enhancer_multi": "",
    "worker_image_upscaler_aurasrv2_multi": "",
    "worker_image_upscaler_swin2sr_multi": "",
    "worker_glm_ocr_multi": "",
}

OMNI_SUPPORTED_MODELS: List[str] = [
    "qwen_image_edit_2511_rapid_aio",
    "z_image_turbo",
    "flux_klein",
    "sdxl_t2i",
    "diffusers_outpaint",
    "basic_utils",
    "aurasrv2",
    "video_upscaler",
    "swin2sr",
    "face_enhancer",
    "wan_22_fp8_lightx2v_t2v",
    "wan_22_fp8_lightx2v_i2v",
    "hunyan15_480p_t2v_distilled",
    "hunyan15_480p_i2v_step_distilled",
    "glm_ocr",
    "ltx2_3_t2v",
    "ltx2_3_i2v_distilled",
    "ltx2_3_flf2v",
    "ltx2_3_iv2v_distilled",
]
# Baked into every omni multi image via run_omni_full_bake + image_registry sibling code.
# Excluded from HF/runtime download plan on all bases (union with each worker's BAKED_MODELS).
OMNI_SHARED_BAKED_MODELS: frozenset = frozenset(
    {
        "basic_utils",
        "aurasrv2",
        "swin2sr",
        "video_upscaler",
        "face_enhancer",
        "glm_ocr",
    }
)

# Diffusers API compatibility groups. Models pinned to a specific diffusers major
# version cannot share a pod with models pinned to a different version (incompatible APIs).
# Omitted from this map = version-agnostic (shared utilities use transformers/torch directly).
OMNI_MODEL_DIFFUSERS_GROUP: Dict[str, str] = {
    # diffusers 0.38.x group (LTX 2.3 only; uses LTX2ConditionPipeline requiring 0.38+)
    "ltx2_3_t2v": "v038",
    "ltx2_3_i2v_distilled": "v038",
    "ltx2_3_flf2v": "v038",
    "ltx2_3_iv2v_distilled": "v038",
    # diffusers 0.36.x group (all other primary models)
    "qwen_image_edit_2511_rapid_aio": "v036",
    "z_image_turbo": "v036",
    "flux_klein": "v036",
    "sdxl_t2i": "v036",
    "diffusers_outpaint": "v036",
    "wan_22_fp8_lightx2v_t2v": "v036",
    "wan_22_fp8_lightx2v_i2v": "v036",
    "hunyan15_480p_t2v_distilled": "v036",
    "hunyan15_480p_i2v_step_distilled": "v036",
    # Shared utilities (basic_utils / aurasrv2 / swin2sr / video_upscaler / face_enhancer / glm_ocr)
    # are version-agnostic — no entry = compatible with any pod.
}


def pod_diffusers_group() -> str:
    """Pod's diffusers bucket from ENV (``v036`` / ``v038``); empty = no filter."""
    return (os.getenv("OMNI_POD_DIFFUSERS_GROUP") or "").strip().lower()


def is_model_compatible_with_pod(model: str) -> bool:
    """
    True if ``model`` can run on the current pod's diffusers bucket, or it is
    version-agnostic (no entry in ``OMNI_MODEL_DIFFUSERS_GROUP``), or the pod has
    not declared a group (``OMNI_POD_DIFFUSERS_GROUP`` unset — legacy behavior).
    """
    pod_group = pod_diffusers_group()
    if not pod_group:
        return True
    mk = (model or "").strip().lower()
    if not mk:
        return True
    model_group = OMNI_MODEL_DIFFUSERS_GROUP.get(mk)
    if not model_group:
        return True
    return model_group == pod_group


OMNI_MODEL_ETA_SECONDS: Dict[str, int] = {
    "qwen_image_edit_2511_rapid_aio": 70,
    "z_image_turbo": 55,
    "flux_klein": 65,
    "sdxl_t2i": 70,
    "diffusers_outpaint": 95,
    "basic_utils": 40,
    "aurasrv2": 60,
    "video_upscaler": 220,
    "swin2sr": 80,
    "face_enhancer": 65,
    "wan_22_fp8_lightx2v_t2v": 140,
    "wan_22_fp8_lightx2v_i2v": 165,
    "hunyan15_480p_t2v_distilled": 170,
    "hunyan15_480p_i2v_step_distilled": 190,
    "glm_ocr": 45,
    "ltx2_3_t2v": 200,
    "ltx2_3_i2v_distilled": 220,
    "ltx2_3_flf2v": 200,
    "ltx2_3_iv2v_distilled": 220,
}

WAN_I2V_LOCAL = os.getenv(
    "WAN_I2V_MODEL_DIR",
    "/app/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64",
)
WAN_T2V_LOCAL = os.getenv(
    "WAN_T2V_MODEL_DIR",
    "/app/models/fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32",
)
HUNYUAN_I2V_LOCAL = os.getenv("HUNYAN15_MODEL_LOCAL_PATH", "/app/models/hunyan15")
HUNYUAN_T2V_LOCAL = os.getenv("HUNYAN15_T2V_LOCAL_PATH", "/app/models/hunyan15_t2v")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")


def build_omni_registry(worker_root: str) -> Dict[str, Any]:
    reg: Dict[str, Any] = {}
    reg.update(build_image_registry(worker_root))
    reg.update(build_wan_registry(worker_root))
    reg.update(build_hunyuan_registry(worker_root))
    slim = os.getenv("OMNI_SLIM_INFERENCE_SRC", "").lower() in ("1", "true", "yes")
    if slim:
        mid = os.getenv("OMNI_MULTI_IMAGE_ID", "").strip()
        primary = (MULTI_IMAGE_ID_TO_PRIMARY_MODEL.get(mid) or "").strip().lower()
        for mk in OMNI_SLIM_DEFER_INFERENCE_TO_PLUGIN:
            key = mk.strip().lower()
            if primary and key == primary:
                continue
            reg.pop(key, None)
    return reg


def omni_preprocess_hook(
    model: str, mode: Optional[str], params: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    return image_preprocess_hook(model, mode, params, job_id)


def _all_omni_download_specs() -> List[Dict[str, Any]]:
    flux_dir = os.path.join(MODELS_DIR, "flux2_klein_9b_sdnq_4bit_svd_r32")
    nunchaku_dir = os.getenv(
        "NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH", "/app/models/nunchaku_z_image_turbo"
    )
    z_base_dir = os.getenv("Z_IMAGE_TURBO_LOCAL_PATH", "/app/models/z_image_turbo")
    qwen_dir = os.getenv("QWEN_2511_LOCAL_PATH", "/app/models/qwen_image_edit_2511")
    aio_transformer_parent = os.getenv(
        "AIO_TRANSFORMER_LOCAL_PATH", "/app/models/aio_transformers"
    )
    _aio_raw = os.getenv("AIO_VERSION", "v19").strip()
    aio_ver = _aio_raw if _aio_raw.startswith("v") else f"v{_aio_raw}"
    sdxl_dir = os.getenv("SDXL_LOCAL_PATH", "/app/models/sdxl_lighting_8_steps")
    outpaint_root = "/app/models/sdxl_outpaint"
    glm_ocr_dir = os.getenv("GLM_OCR_LOCAL_PATH", "/app/models/glm_ocr")
    lora_root = os.getenv("LORA_ADAPTERS_LOCAL_PATH", "/app/models/lora_adapters")

    qwen_lora_supplements = build_qwen_lora_supplement_hub_files()

    return [
        {
            "model": "qwen_image_edit_2511_rapid_aio",
            "multi_snapshot": True,
            "local_dir": qwen_dir,
            "supplement_dest_dir": lora_root,
            "supplement_hub_files": qwen_lora_supplements,
            "snapshots": [
                {
                    "repo_id": os.getenv(
                        "HF_REPO_QWEN_BAKED", "Qwen/Qwen-Image-Edit-2511"
                    ),
                    "target_dir": qwen_dir,
                    "ignore_patterns": ["transformer/*"],
                },
                {
                    "repo_id": os.getenv(
                        "HF_REPO_QWEN_AIO_TRANSFORMER",
                        "Pr0f3ssi0n4ln00b/Phr00t-Qwen-Rapid-AIO",
                    ),
                    "target_dir": aio_transformer_parent,
                    "allow_patterns": [f"{aio_ver}/transformer/*"],
                },
            ],
        },
        {
            "model": "z_image_turbo",
            "repo_id": os.getenv("HF_REPO_Z_IMAGE_TURBO", "Tongyi-MAI/Z-Image-Turbo"),
            "local_dir": z_base_dir,
            "ignore_patterns": ["transformer/*"],
            "supplement_dest_dir": nunchaku_dir,
            "supplement_hub_files": [
                {
                    "repo_id": os.getenv(
                        "HF_REPO_NUNCHAKU_Z", "nunchaku-tech/nunchaku-z-image-turbo"
                    ),
                    "filename": "svdq-fp4_r128-z-image-turbo.safetensors",
                },
                {
                    "repo_id": os.getenv(
                        "HF_REPO_NUNCHAKU_Z", "nunchaku-tech/nunchaku-z-image-turbo"
                    ),
                    "filename": "svdq-int4_r128-z-image-turbo.safetensors",
                },
            ],
        },
        {
            "model": "flux_klein",
            "repo_id": os.getenv(
                "HF_REPO_FLUX_KLEIN_BAKED",
                "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
            ),
            "local_dir": flux_dir,
        },
        {
            "model": "sdxl_t2i",
            "repo_id": os.getenv("HF_REPO_SDXL_LIGHTING", "HyHorX/SDXL-Lighting-8-steps"),
            "local_dir": sdxl_dir,
        },
        {
            "model": "diffusers_outpaint",
            "multi_snapshot": True,
            "local_dir": outpaint_root,
            "snapshots": [
                {
                    "repo_id": "xinsir/controlnet-union-sdxl-1.0",
                    "target_dir": f"{outpaint_root}/controlnet",
                },
                {
                    "repo_id": "madebyollin/sdxl-vae-fp16-fix",
                    "target_dir": f"{outpaint_root}/vae",
                },
                {
                    "repo_id": "SG161222/RealVisXL_V5.0_Lightning",
                    "target_dir": f"{outpaint_root}/pipeline",
                    "ignore_patterns": [
                        "text_encoder/model.safetensors",
                        "text_encoder_2/model.safetensors",
                        "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
                        "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
                        "vae/diffusion_pytorch_model.safetensors",
                        "RealVisXL_V5.0_Lightning_fp32.safetensors",
                    ],
                },
            ],
        },
        {
            "model": "wan_22_fp8_lightx2v_t2v",
            "repo_id": os.getenv(
                "HF_REPO_WAN_T2V",
                "rockapaper/wan22-14b-t2v-lightning-sdnq-fp8-svd-r32",
            ),
            "local_dir": WAN_T2V_LOCAL,
        },
        {
            "model": "wan_22_fp8_lightx2v_i2v",
            "repo_id": os.getenv(
                "HF_REPO_WAN_I2V",
                "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64",
            ),
            "local_dir": WAN_I2V_LOCAL,
        },
        {
            "model": "hunyan15_480p_t2v_distilled",
            "repo_id": os.getenv(
                "HF_REPO_HUNYUAN_T2V",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
            ),
            "local_dir": HUNYUAN_T2V_LOCAL,
            "supplement_hub_files": [
                {
                    "repo_id": "lightx2v/Hy1.5-Distill-Models",
                    "filename": "hy1.5_t2v_480p_lightx2v_4step.safetensors",
                    "target_relpath": "transformer/diffusion_pytorch_model.safetensors",
                }
            ],
        },
        {
            "model": "hunyan15_480p_i2v_step_distilled",
            "repo_id": os.getenv(
                "HF_REPO_HUNYUAN_I2V",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
            ),
            "local_dir": HUNYUAN_I2V_LOCAL,
        },
        {
            "model": "glm_ocr",
            "repo_id": os.getenv("HF_REPO_GLM_OCR", "zai-org/GLM-OCR"),
            "local_dir": glm_ocr_dir,
            "download_priority": 80,
        },
    ]


def _apply_canonical_qwen_lora_supplements(specs: List[Dict[str, Any]]) -> None:
    """
    After manifest merge: use ``supplement_hub_files`` from ``omni_manifest.json`` when it
    lists the full LoRA set. If missing, incomplete, or ``supplement_dest_dir`` is empty,
    backfill from ``qwen_2511_lora_registry`` (same 15 files as Qwen multi bake).
    """
    lora_root = os.getenv("LORA_ADAPTERS_LOCAL_PATH", "/app/models/lora_adapters")
    full = build_qwen_lora_supplement_hub_files()
    expected_n = len(full)
    mk = "qwen_image_edit_2511_rapid_aio"
    for spec in specs:
        if str(spec.get("model", "")).strip().lower() != mk:
            continue
        have = spec.get("supplement_hub_files") or []
        dest = str(spec.get("supplement_dest_dir") or "").strip()
        if len(have) >= expected_n and dest:
            continue
        spec["supplement_dest_dir"] = dest or lora_root
        spec["supplement_hub_files"] = [dict(x) for x in full]


def _merge_download_specs_from_manifest(
    base: List[Dict[str, Any]], manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Replace or append specs by ``model`` id; manifest entries must be full _download_task specs."""
    extras = manifest.get("download_specs")
    if not extras or not isinstance(extras, list):
        return base
    overrides: Dict[str, Dict[str, Any]] = {}
    for item in extras:
        if not isinstance(item, dict):
            continue
        mk = str(item.get("model", "")).strip().lower()
        if not mk:
            continue
        overrides[mk] = dict(item)
    if not overrides:
        return base
    out: List[Dict[str, Any]] = []
    used: Set[str] = set()
    for spec in base:
        mk = str(spec.get("model", "")).strip().lower()
        if mk in overrides:
            out.append(overrides[mk])
            used.add(mk)
        else:
            out.append(dict(spec))
    for mk, spec in overrides.items():
        if mk not in used:
            out.append(spec)
    return out


def get_effective_supported_models() -> List[str]:
    """Default omni list plus manifest add/remove; used for worker advertisement."""
    manifest = load_runtime_manifest()
    remove = {
        str(x).strip().lower()
        for x in (manifest.get("supported_models_remove") or [])
        if str(x).strip()
    }
    add: List[str] = []
    for x in manifest.get("supported_models_add") or []:
        s = str(x).strip().lower()
        if s:
            add.append(s)
    out: List[str] = []
    seen: Set[str] = set()
    for m in OMNI_SUPPORTED_MODELS:
        ml = m.strip().lower()
        if ml in remove or ml in seen:
            continue
        seen.add(ml)
        out.append(ml)
    for m in add:
        if m in remove or m in seen:
            continue
        seen.add(m)
        out.append(m)
    # Filter out models incompatible with this pod's diffusers bucket.
    out = [m for m in out if is_model_compatible_with_pod(m)]
    return out


def get_effective_model_eta_seconds() -> Dict[str, int]:
    manifest = load_runtime_manifest()
    d = dict(OMNI_MODEL_ETA_SECONDS)
    upd = manifest.get("model_eta_seconds")
    if isinstance(upd, dict):
        for k, v in upd.items():
            try:
                d[str(k).strip().lower()] = int(v)
            except (TypeError, ValueError):
                continue
    return d


def get_model_categories() -> Dict[str, List[str]]:
    """Image vs video model ids from manifest (adaptive download / future tuning)."""
    manifest = load_runtime_manifest()
    cats = manifest.get("model_categories")
    if isinstance(cats, dict):
        out: Dict[str, List[str]] = {}
        for k, vs in cats.items():
            if isinstance(vs, list):
                out[str(k)] = [str(v).strip().lower() for v in vs if str(v).strip()]
        if out:
            return out
    return {
        "image": [
            "qwen_image_edit_2511_rapid_aio",
            "z_image_turbo",
            "flux_klein",
            "sdxl_t2i",
            "diffusers_outpaint",
            "basic_utils",
            "aurasrv2",
            "swin2sr",
            "face_enhancer",
            "glm_ocr",
        ],
        "video": [
            "wan_22_fp8_lightx2v_t2v",
            "wan_22_fp8_lightx2v_i2v",
            "hunyan15_480p_t2v_distilled",
            "hunyan15_480p_i2v_step_distilled",
            "video_upscaler",
            "ltx2_3_t2v",
            "ltx2_3_i2v_distilled",
            "ltx2_3_flf2v",
            "ltx2_3_iv2v_distilled",
        ],
    }


def get_ltx2_3_civitai_loras() -> List[Dict[str, Any]]:
    """
    LTX-2.3 CivitAI LoRA catalog from the runtime manifest.

    Returns a list of normalized dicts: ``{"name": str, "url": str, "strength": float,
    "baked": bool, "version_id": Optional[int]}``. Entries with no usable ``url`` are dropped.
    Used by ``worker_ltx2_3_multi`` (prep_model bake + runtime startup pre-warm + per-job
    name lookup). Other multi workers ignore this list — LTX is in a different diffusers
    bucket and is filtered out of their download plans.
    """
    manifest = load_runtime_manifest()
    raw = manifest.get("ltx2_3_civitai_loras")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        version_id_raw = item.get("version_id")
        try:
            version_id = int(version_id_raw) if version_id_raw is not None else None
        except (TypeError, ValueError):
            version_id = None
        out.append({
            "name": (str(item.get("name") or "").strip() or None),
            "url": url,
            "strength": strength,
            "baked": bool(item.get("baked", False)),
            "version_id": version_id,
        })
    return out


def find_ltx2_3_lora_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Look up a LoRA in the manifest catalog by its ``name`` field (case-insensitive)."""
    if not name:
        return None
    needle = str(name).strip().lower()
    if not needle:
        return None
    for entry in get_ltx2_3_civitai_loras():
        nm = (entry.get("name") or "").strip().lower()
        if nm and nm == needle:
            return entry
    return None


def omni_download_plan_excluding(baked_models: Set[str]) -> List[Dict[str, Any]]:
    """
    Return the full runtime download plan, merged with the R2/HTTP manifest.

    Steps:
    1. Start with built-in HF specs (minus baked models).
    2. Overlay/append manifest ``download_specs`` (HF overrides or new HF-only models).
    3. Attach ``inference_archive`` from manifest ``inference_plugins`` onto matching specs;
       new plugin-only models (code but no HF weights) get a spec with no repo_id.
    4. Sort by ``download_priority`` (int, lower = earlier; default 100).
       Manifest specs should set a priority value so new models start downloading
       promptly rather than after all built-in models.
    """
    baked = {m.strip().lower() for m in baked_models if m and str(m).strip()}
    manifest = load_runtime_manifest()

    base = [
        dict(spec)
        for spec in _all_omni_download_specs()
        if str(spec.get("model", "")).strip().lower() not in baked
    ]

    # Step 2: apply HF overrides/additions from manifest
    merged = _merge_download_specs_from_manifest(base, manifest)
    _apply_canonical_qwen_lora_supplements(merged)

    # Step 3: attach inference_archive from manifest inference_plugins
    plugins_by_model: Dict[str, Any] = {}
    for p in (manifest.get("inference_plugins") or []):
        if not isinstance(p, dict):
            continue
        mk = str(p.get("model") or "").strip().lower()
        if mk:
            plugins_by_model[mk] = p

    if plugins_by_model:
        covered: Set[str] = set()
        for spec in merged:
            mk = str(spec.get("model") or "").strip().lower()
            if mk in plugins_by_model:
                spec["inference_archive"] = plugins_by_model[mk]
                covered.add(mk)
        for mk, plugin_spec in plugins_by_model.items():
            if mk not in covered and mk not in baked:
                # Code-only plugin: no HF weights, just inference archive
                merged.append({
                    "model": mk,
                    "local_dir": "",
                    "inference_archive": plugin_spec,
                    "download_priority": int(plugin_spec.get("download_priority", 50)),
                })

    # Step 4: filter baked (again after additions) and sort by priority
    out = [
        spec for spec in merged
        if (str(spec.get("model") or "").strip().lower() not in baked)
        and is_model_compatible_with_pod(str(spec.get("model") or ""))
    ]
    out.sort(key=lambda s: int(s.get("download_priority") or 100))
    return out
