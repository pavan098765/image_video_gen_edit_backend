import contextlib
import importlib.util
import os
import sys
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Iterator, Optional

import requests
from PIL import Image


# WAN I2V: swappable inference module (basename only — no path traversal)
_WAN_I2V_INFERENCE_ALLOWLIST = frozenset(
    {
        "inference.py",
        "inference_sdnq_space_parity.py",
        "inference_hub_space_torchao.py",
    }
)

# WAN T2V: swappable inference module (basename only — no path traversal)
_WAN_T2V_INFERENCE_ALLOWLIST = frozenset(
    {
        "inference.py",
        "inference_sdnq_space_parity.py",
    }
)


def _wan_i2v_inference_basename() -> str:
    raw = os.getenv("WAN_I2V_INFERENCE_MODULE", "inference.py").strip()
    base = os.path.basename(raw)
    if base != raw or ".." in raw or "/" in raw or "\\" in raw:
        base = "inference.py"
    if base not in _WAN_I2V_INFERENCE_ALLOWLIST:
        base = "inference.py"
    return base


def _wan_t2v_inference_basename() -> str:
    raw = os.getenv("WAN_T2V_INFERENCE_MODULE", "inference.py").strip()
    base = os.path.basename(raw)
    if base != raw or ".." in raw or "/" in raw or "\\" in raw:
        base = "inference.py"
    if base not in _WAN_T2V_INFERENCE_ALLOWLIST:
        base = "inference.py"
    return base


def _module_name_from_path(path: str) -> str:
    base = os.path.basename(path).replace(".", "_")
    return f"multi_worker_video_mod_{abs(hash(path))}_{base}"


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


@lru_cache(maxsize=32)
def _load_module(path: str):
    name = _module_name_from_path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with _module_dir_on_syspath(path):
        spec.loader.exec_module(module)
    return module


def _resolve_init_image_to_pil(raw: Any, timeout: int = 120) -> Optional[Image.Image]:
    if raw is None:
        return None
    if isinstance(raw, str) and (raw.startswith("http://") or raw.startswith("https://")):
        resp = requests.get(raw, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if isinstance(raw, str) and os.path.exists(raw):
        return Image.open(raw).convert("RGB")
    if isinstance(raw, str) and raw.strip():
        raise ValueError(
            "init_image must be an http(s) presigned URL or an existing local file path"
        )
    return None


def _resolve_module_path(worker_root: str, worker_dir: str, filename: str) -> str:
    explicit_root = os.getenv("QPW_ROOT", "").strip()
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)
    candidates.extend(
        [
            os.path.join(worker_root, "worker", "src", "inference"),
            "/app/worker/src/inference",
            os.path.join(os.getcwd(), "worker", "src", "inference"),
            worker_root,
            "/app",
            os.getcwd(),
        ]
    )
    seen = set()
    for root in candidates:
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


def build_wan_registry(worker_root: str) -> Dict[str, Any]:
    def _wan_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        t2v_fn = _wan_t2v_inference_basename()
        t2v_path = _resolve_module_path(
            worker_root, "worker_wan_22_fp8_lightx2v_t2v", t2v_fn
        )
        run_inference_wan_22_t2v = getattr(_load_module(t2v_path), "run_inference_wan_22_t2v")
        return run_inference_wan_22_t2v(job_id, prompt, params)

    def _wan_i2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        i2v_fn = _wan_i2v_inference_basename()
        i2v_path = _resolve_module_path(
            worker_root, "worker_wan_22_fp8_lightx2v_i2v", i2v_fn
        )
        run_inference_wan_22_i2v = getattr(
            _load_module(i2v_path), "run_inference_wan_22_i2v"
        )
        p = dict(params or {})
        init_image = _resolve_init_image_to_pil(
            p.pop("init_image", None) or p.pop("image", None)
        )
        last_raw = p.pop("last_image", None) or p.pop("last_frame_image", None)
        last_image = _resolve_init_image_to_pil(last_raw)
        return run_inference_wan_22_i2v(
            job_id, prompt, p, init_image=init_image, last_image=last_image
        )

    return {
        "wan_22_fp8_lightx2v_t2v": _wan_t2v,
        "wan_22_fp8_lightx2v_i2v": _wan_i2v,
    }


def build_hunyuan_registry(worker_root: str) -> Dict[str, Any]:
    t2v_path = _resolve_module_path(
        worker_root,
        "worker_hunyan15_480p_t2V_distilled",
        "hunyan15_video_inference.py",
    )
    i2v_path = _resolve_module_path(
        worker_root,
        "worker_hunyan15_480p_i2V_step_distilled",
        "hunyan15_video_inference.py",
    )

    def _t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        run_inference_hunyan15_t2v = getattr(
            _load_module(t2v_path), "run_inference_hunyan15_t2v"
        )
        return run_inference_hunyan15_t2v(job_id, prompt, params)

    def _i2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
        run_inference_hunyan15_i2v = getattr(
            _load_module(i2v_path), "run_inference_hunyan15_i2v"
        )
        p = dict(params or {})
        image_input = (
            p.pop("image_input", None)
            or p.pop("image", None)
            or p.pop("init_image", None)
        )
        if not image_input:
            raise ValueError("image_input/init_image is required for hunyuan i2v")
        return run_inference_hunyan15_i2v(job_id, prompt, image_input, p)

    return {
        "hunyan15_480p_t2v_distilled": _t2v,
        "hunyan15_480p_i2v_step_distilled": _i2v,
    }


def passthrough_preprocess(
    model: str, mode: Optional[str], params: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    return dict(params or {})

