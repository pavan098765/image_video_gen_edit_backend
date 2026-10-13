import contextlib
import importlib.util
import os
import sys
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional


def _module_name_from_path(path: str) -> str:
    base = os.path.basename(path).replace(".", "_")
    return f"multi_worker_mod_{abs(hash(path))}_{base}"


@contextlib.contextmanager
def _module_dir_on_syspath(module_path: str) -> Iterator[None]:
    d = os.path.dirname(os.path.abspath(module_path))
    old = list(sys.path)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        yield
    finally:
        sys.path[:] = old


@lru_cache(maxsize=64)
def _load_module(path: str):
    name = _module_name_from_path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with _module_dir_on_syspath(path):
        spec.loader.exec_module(module)
    return module


def _lazy_call(module_path: str, function_name: str):
    def _wrapped(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        mod = _load_module(module_path)
        fn = getattr(mod, function_name)
        return fn(job_id, prompt, params)

    return _wrapped


def _lazy_call_no_prompt(module_path: str, function_name: str):
    def _wrapped(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        mod = _load_module(module_path)
        fn = getattr(mod, function_name)
        return fn(job_id, params)

    return _wrapped


def _lazy_hook(module_path: str, function_name: str):
    def _wrapped():
        mod = _load_module(module_path)
        fn = getattr(mod, function_name, None)
        if callable(fn):
            return fn()
        return None

    return _wrapped


def _resolve_module_path(worker_root: str, worker_dir: str, filename: str) -> str:
    """
    Resolve inference module path.
    Priority:
      1) Embedded copy under /app/worker/src/inference/... (or QPW_ROOT override)
      2) Legacy sibling path under /app/<worker_dir>/src
    """
    explicit_root = os.getenv("QPW_ROOT", "").strip()
    embedded_roots = []
    if explicit_root:
        embedded_roots.append(explicit_root)
    embedded_roots.extend(
        [
            os.path.join(worker_root, "worker", "src", "inference"),
            "/app/worker/src/inference",
            os.path.join(os.getcwd(), "worker", "src", "inference"),
        ]
    )

    seen = set()
    for root in embedded_roots:
        if not root:
            continue
        norm = os.path.normpath(root)
        if norm in seen:
            continue
        seen.add(norm)
        path = os.path.join(norm, worker_dir, "src", filename)
        if os.path.exists(path):
            return path

    legacy_roots = [worker_root, "/app", os.getcwd()]
    for root in legacy_roots:
        if not root:
            continue
        norm = os.path.normpath(root)
        if norm in seen:
            continue
        seen.add(norm)
        path = os.path.join(norm, worker_dir, "src", filename)
        if os.path.exists(path):
            return path

    return os.path.join(worker_root, worker_dir, "src", filename)


def build_image_registry(worker_root: str) -> Dict[str, Any]:
    """
    Build a multi-model image registry by reusing existing inference modules.

    worker_root: the directory that contains all sibling worker directories
                 (e.g. /app in Docker, or quickpod_individual_worker locally).
    """
    qwen_mod = _resolve_module_path(
        worker_root, "worker_qwen_image_edit_2511_rapid_aio", "qwen_image_edit_2511_rapid_aio_inference.py"
    )
    flux_mod = _resolve_module_path(worker_root, "worker_flux_klein_9b", "flux_klein_inference.py")
    z_mod = _resolve_module_path(worker_root, "worker_z_image_turbo", "z_image_turbo.py")
    sdxl_mod = _resolve_module_path(worker_root, "worker_stable_diffusion_xl", "sdxl_T2I_inference.py")
    outpaint_mod = _resolve_module_path(
        worker_root, "worker_stable_diffusion_xl_outpaint", "diffuser_outpaint_inference.py"
    )
    aura_mod = _resolve_module_path(worker_root, "worker_image_upscaler_aurasrv2", "aurasrv2_upscaler.py")
    swin_mod = _resolve_module_path(worker_root, "worker_image_upscaler_swin2sr", "swin2sr_upscaler.py")
    face_mod = _resolve_module_path(worker_root, "worker_face_enhancer", "inference_face_enhancer.py")
    video_up_mod = _resolve_module_path(worker_root, "worker_video_upscalers", "inference.py")
    glm_ocr_mod = _resolve_module_path(worker_root, "worker_glm_ocr", "glm_ocr_inference.py")
    rmbg_mod = _resolve_module_path(worker_root, "worker_basic_utils", "bg_remover.py")
    lama_mod = _resolve_module_path(worker_root, "worker_basic_utils", "simple_lama_inpaint.py")

    def _run_basic_utils(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        run_inference_simple_lama_inpaint = getattr(
            _load_module(lama_mod), "run_inference_simple_lama_inpaint"
        )
        run_inference_rmbg = getattr(_load_module(rmbg_mod), "run_inference_rmbg")
        mode = str(params.get("mode") or "bg_remove").strip().lower()
        if mode == "lama_inpaint":
            return run_inference_simple_lama_inpaint(job_id, prompt, params)
        return run_inference_rmbg(job_id, prompt, params)

    return {
        "qwen_image_edit_2511_rapid_aio": {
            "callable": _lazy_call(
                qwen_mod, "run_inference_qwen_image_edit_2511_rapid_aio"
            ),
            "prewarm": _lazy_hook(qwen_mod, "get_pipeline"),
            "unload": _lazy_hook(qwen_mod, "unload_models"),
        },
        "flux_klein": {
            "callable": _lazy_call(flux_mod, "run_inference_flux_klein"),
            "prewarm": _lazy_hook(flux_mod, "get_pipeline"),
            "unload": _lazy_hook(flux_mod, "unload_models"),
        },
        "z_image_turbo": {
            "callable": _lazy_call(z_mod, "run_inference_z_image_turbo"),
            "prewarm": _lazy_hook(z_mod, "get_pipeline"),
            "unload": _lazy_hook(z_mod, "unload_models"),
        },
        "sdxl_t2i": {
            "callable": _lazy_call(sdxl_mod, "run_inference_sdxl_t2i"),
            "prewarm": _lazy_hook(sdxl_mod, "get_pipeline"),
            "unload": _lazy_hook(sdxl_mod, "unload_models"),
        },
        "diffusers_outpaint": {
            "callable": _lazy_call(outpaint_mod, "run_inference_sdxl_outpaint"),
            "prewarm": _lazy_hook(outpaint_mod, "get_sdxl_outpaint_pipeline"),
            "unload": _lazy_hook(outpaint_mod, "unload_sdxl_outpaint"),
        },
        "aurasrv2": {
            "callable": _lazy_call_no_prompt(aura_mod, "run_inference_aurasr_upscale"),
            "unload": _lazy_hook(aura_mod, "unload_aura_sr"),
        },
        "swin2sr": {
            "callable": _lazy_call_no_prompt(swin_mod, "run_inference_swin2sr"),
            "unload": _lazy_hook(swin_mod, "unload_swin2sr_models"),
        },
        "face_enhancer": {
            "callable": _lazy_call_no_prompt(face_mod, "run_inference_face_enhancer"),
            "unload": _lazy_hook(face_mod, "unload_models"),
        },
        "video_upscaler": {
            "callable": _lazy_call(video_up_mod, "run_inference_video_upscaler"),
            "unload": _lazy_hook(video_up_mod, "unload_models"),
        },
        "basic_utils": _run_basic_utils,
        "glm_ocr": {
            "callable": _lazy_call(glm_ocr_mod, "run_inference_glm_ocr"),
            "prewarm": _lazy_hook(glm_ocr_mod, "preload_pipeline"),
            "unload": _lazy_hook(glm_ocr_mod, "unload_pipeline"),
        },
    }


def registry_entry_from_py(
    module_path: str,
    run_inference_name: str,
    *,
    prewarm_name: Optional[str] = None,
    unload_name: Optional[str] = None,
    no_prompt: bool = False,
) -> Dict[str, Any]:
    """
    Registry entry for a single lazy-loaded .py file (used by R2/runtime inference plugins).
    """
    if no_prompt:
        call = _lazy_call_no_prompt(module_path, run_inference_name)
    else:
        call = _lazy_call(module_path, run_inference_name)
    entry: Dict[str, Any] = {"callable": call}
    if prewarm_name:
        entry["prewarm"] = _lazy_hook(module_path, prewarm_name)
    if unload_name:
        entry["unload"] = _lazy_hook(module_path, unload_name)
    return entry


def assert_worker_media_params_are_presigned_urls(
    params: Dict[str, Any], job_id: str
) -> None:
    """
    Reject embedded base64/data URIs and non-URL strings for known media fields.
    Controller is expected to send only http(s) presigned GET URLs.
    """

    def _looks_like_embedded_blob(s: str) -> bool:
        if s.startswith("data:"):
            return True
        if len(s) < 500:
            return False
        sample = s[:100].replace("/", "").replace("+", "").replace("=", "")
        return sample.isalnum()

    media_keys = (
        "init_image",
        "mask_image",
        "image",
        "image_input",
        "input_video",
        "ref_image",
        "ref_video",
        "control_image",
        "source_image",
    )
    for k in media_keys:
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            for i, item in enumerate(v):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"[{job_id}] {k}[{i}] must be a non-empty http(s) URL string"
                    )
                if _looks_like_embedded_blob(item) or not (
                    item.startswith("http://") or item.startswith("https://")
                ):
                    raise ValueError(
                        f"[{job_id}] {k}[{i}] must be a presigned http(s) URL, not embedded media"
                    )
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        if _looks_like_embedded_blob(v) or not (
            v.startswith("http://") or v.startswith("https://")
        ):
            raise ValueError(
                f"[{job_id}] {k} must be a presigned http(s) URL, not embedded media"
            )


def image_preprocess_hook(
    model: str, mode: Optional[str], params: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    """
    Ensure cross-model compatibility for utility models.
    """
    assert_worker_media_params_are_presigned_urls(params or {}, job_id)
    out = dict(params or {})
    if model == "basic_utils":
        if "mode" not in out and mode:
            out["mode"] = mode
        out.setdefault("mode", "bg_remove")
    return out

