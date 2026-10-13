import os
from typing import Any, Dict, Optional

from ltx2_3_inference import (
    run_inference_ltx2_3_flf2v,
    run_inference_ltx2_3_i2v_distilled,
    run_inference_ltx2_3_iv2v_distilled,
    run_inference_ltx2_3_t2v,
    unload_pipelines,
)


def get_registry() -> Dict[str, Any]:
    return {
        "ltx2_3_t2v": {
            "callable": run_inference_ltx2_3_t2v,
            "unload": unload_pipelines,
        },
        "ltx2_3_i2v_distilled": {
            "callable": run_inference_ltx2_3_i2v_distilled,
            "unload": unload_pipelines,
        },
        "ltx2_3_flf2v": {
            "callable": run_inference_ltx2_3_flf2v,
            "unload": unload_pipelines,
        },
        "ltx2_3_iv2v_distilled": {
            "callable": run_inference_ltx2_3_iv2v_distilled,
            "unload": unload_pipelines,
        },
    }


def _looks_like_embedded_blob(s: str) -> bool:
    if s.startswith("data:"):
        return True
    if len(s) < 500:
        return False
    sample = s[:100].replace("/", "").replace("+", "").replace("=", "")
    return sample.isalnum()


def _assert_url_field(params: Dict[str, Any], key: str, job_id: str) -> None:
    v = params.get(key)
    if v is None:
        return
    if not isinstance(v, str) or not v.strip():
        return
    if _looks_like_embedded_blob(v) or not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError(
            f"[{job_id}] {key} must be a presigned http(s) URL, not embedded media"
        )


def _flf2v_first_last_explicitly_set(out: Dict[str, Any]) -> bool:
    first = (
        out.get("first_frame_image")
        or out.get("first_image")
    )
    last = (
        out.get("last_frame_image")
        or out.get("last_image")
    )
    if not (isinstance(first, str) and first.strip()):
        return False
    if not (isinstance(last, str) and last.strip()):
        return False
    return True


def _map_flf2v_init_image_list_to_first_last(
    out: Dict[str, Any], job_id: str
) -> None:
    """Controller bulk two-frame jobs send init_image as [start_url, end_url]."""
    raw = out.get("init_image")
    if raw is None:
        return
    if not isinstance(raw, list):
        return
    if _flf2v_first_last_explicitly_set(out):
        return
    n = len(raw)
    if n == 0:
        return
    if n != 2:
        raise ValueError(
            f"[{job_id}] ltx2_3_flf2v: expected init_image to be a list of exactly "
            f"2 presigned http(s) URLs [first_frame, last_frame], got length {n}"
        )
    a, b = raw[0], raw[1]
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(
            f"[{job_id}] ltx2_3_flf2v: init_image list entries must be URL strings"
        )
    out["first_frame_image"] = a.strip()
    out["last_frame_image"] = b.strip()
    out.pop("init_image", None)
    # Single-image key used by i2v should not remain for FLF2V inference.
    out.pop("image", None)


def _iv2v_assert_required_urls(out: Dict[str, Any], job_id: str) -> None:
    """iv2v requires both a condition_video and a condition_image as presigned URLs."""
    cv = out.get("condition_video")
    ci = out.get("condition_image")
    if not (isinstance(cv, str) and cv.strip()):
        raise ValueError(
            f"[{job_id}] ltx2_3_iv2v_distilled: condition_video is required as a presigned http(s) URL"
        )
    if not (isinstance(ci, str) and ci.strip()):
        raise ValueError(
            f"[{job_id}] ltx2_3_iv2v_distilled: condition_image is required as a presigned http(s) URL"
        )


def preprocess(
    model: str, mode: Optional[str], params: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    from image_registry import assert_worker_media_params_are_presigned_urls  # type: ignore

    out = dict(params or {})
    assert_worker_media_params_are_presigned_urls(out, job_id)
    mk = (model or "").strip().lower()
    if mk == "ltx2_3_flf2v":
        _map_flf2v_init_image_list_to_first_last(out, job_id)
    if mk == "ltx2_3_iv2v_distilled":
        _iv2v_assert_required_urls(out, job_id)
    for k in (
        "first_frame_image",
        "last_frame_image",
        "first_image",
        "last_image",
        "condition_video",
        "condition_image",
    ):
        _assert_url_field(out, k, job_id)
    return out
