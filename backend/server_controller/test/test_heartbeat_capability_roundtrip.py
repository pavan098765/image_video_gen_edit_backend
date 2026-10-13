"""
Heartbeat <-> Worker.info <-> scheduler_strategy round-trip test.

Ensures the shape that MultiModelWorker emits in heartbeat (supported_models, model_eta_seconds,
metrics.download_state) survives the heartbeat-sync normalization in routes.py and is readable
by scheduler_strategy._resolve_runtime_supported_models + _find_pending_candidate download_state
usage.

This is a logic-only test: it does NOT boot FastAPI or touch a real DB. It inlines the normalization
path from routes.sync_heartbeat_cache_to_db so any schema drift will break the assertion here first.

Run: `python test/test_heartbeat_capability_roundtrip.py`
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def _normalize_supported(raw: Any) -> List[str]:
    out: List[str] = []
    if not raw:
        return out
    for m in raw or []:
        if isinstance(m, str):
            m_norm = m.strip().lower()
            if m_norm and m_norm not in out:
                out.append(m_norm)
    return out


def _normalize_eta(raw: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        try:
            val = int(v)
        except Exception:
            continue
        if val <= 0:
            continue
        out[k.strip().lower()] = val
    return out


def _normalize_download_state(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for m_key, m_state in raw.items():
        if not isinstance(m_key, str):
            continue
        model_key = m_key.strip().lower()
        if not model_key:
            continue
        if isinstance(m_state, dict):
            compact: Dict[str, Any] = {}
            status_val = m_state.get("status")
            if isinstance(status_val, str) and status_val.strip():
                compact["status"] = status_val.strip()
            ready_in = m_state.get("ready_in_seconds")
            if isinstance(ready_in, (int, float)) and ready_in >= 0:
                compact["ready_in_seconds"] = int(ready_in)
            eta_download = m_state.get("eta_seconds")
            if isinstance(eta_download, (int, float)) and eta_download > 0:
                compact["eta_seconds"] = int(eta_download)
            if compact:
                out[model_key] = compact
    return out


def _apply_heartbeat_to_info(worker_info: Dict[str, Any], hb_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of sync_heartbeat_cache_to_db normalization in routes.py."""
    info = dict(worker_info) if isinstance(worker_info, dict) else {}
    hb_supported = hb_payload.get("supported_models")
    hb_eta = hb_payload.get("model_eta_seconds")
    hb_metrics = hb_payload.get("metrics") or {}
    hb_download_state_raw = hb_metrics.get("download_state") if isinstance(hb_metrics, dict) else None

    normalized_supported = _normalize_supported(hb_supported)
    if normalized_supported:
        info["supported_models"] = normalized_supported
    eta_map = _normalize_eta(hb_eta)
    if eta_map:
        info["model_eta_seconds"] = eta_map
    ds = _normalize_download_state(hb_download_state_raw)
    if ds:
        info["download_state"] = ds
    return info


def _resolve_runtime_supported_models(worker_info: Dict[str, Any], worker_model: str = "") -> set:
    """Mirror of scheduler_strategy._resolve_runtime_supported_models."""
    if not isinstance(worker_info, dict):
        return set()
    raw_supported = worker_info.get("supported_models")
    if not isinstance(raw_supported, list):
        return set()
    supported = set()
    for m in raw_supported:
        if isinstance(m, str) and m.strip():
            supported.add(m.strip().lower())
    if worker_model and isinstance(worker_model, str):
        supported.add(worker_model.strip().lower())
    return supported


def test_round_trip_basic():
    # Simulated heartbeat payload as sent by MultiModelWorker.heartbeat_loop
    hb = {
        "worker_id": "pod-abc123",
        "pod_uuid": "pod-abc123",
        "timestamp": 1700000000,
        "status": "ready",
        "supported_models": [
            "Flux_Klein",  # mixed case on purpose — must be normalized
            "z_image_turbo",
            "glm_ocr",
            "",  # empty should be dropped
        ],
        "model_eta_seconds": {
            "flux_klein": 65,
            "z_image_turbo": 55,
            "bogus": "not_int",  # must be dropped
            "glm_ocr": 0,  # must be dropped (<= 0)
        },
        "metrics": {
            "queue_size": 0,
            "active_jobs": 0,
            "download_state": {
                "sdxl_t2i": {"status": "downloading", "ready_in_seconds": 30, "eta_seconds": 70},
                "qwen_image_edit_2511_rapid_aio": {"status": "pending", "ready_in_seconds": 90},
                "broken_row": "not a dict",  # must be dropped
            },
        },
    }
    info = _apply_heartbeat_to_info({}, hb)

    assert info["supported_models"] == ["flux_klein", "z_image_turbo", "glm_ocr"], info["supported_models"]
    assert info["model_eta_seconds"] == {"flux_klein": 65, "z_image_turbo": 55}, info["model_eta_seconds"]
    assert "download_state" in info
    ds = info["download_state"]
    assert "sdxl_t2i" in ds and ds["sdxl_t2i"]["status"] == "downloading"
    assert ds["sdxl_t2i"]["ready_in_seconds"] == 30
    assert ds["sdxl_t2i"]["eta_seconds"] == 70
    assert "qwen_image_edit_2511_rapid_aio" in ds
    assert "broken_row" not in ds

    resolved = _resolve_runtime_supported_models(info, worker_model="flux_klein")
    assert "flux_klein" in resolved
    assert "z_image_turbo" in resolved
    assert "glm_ocr" in resolved


def test_ltx_omni_advertises_four_modes():
    """LTX multi worker should advertise all 4 modes once canonical download completes."""
    hb = {
        "supported_models": [
            "ltx2_3_t2v",
            "ltx2_3_i2v_distilled",
            "ltx2_3_flf2v",
            "ltx2_3_iv2v_distilled",
            "basic_utils",
            "glm_ocr",
        ],
        "model_eta_seconds": {
            "ltx2_3_t2v": 200,
            "ltx2_3_i2v_distilled": 220,
            "ltx2_3_flf2v": 200,
            "ltx2_3_iv2v_distilled": 220,
        },
        "metrics": {
            "download_state": {
                "flux_klein": {"status": "downloading", "ready_in_seconds": 45},
            }
        },
    }
    info = _apply_heartbeat_to_info({}, hb)
    resolved = _resolve_runtime_supported_models(info, worker_model="ltx2_3_t2v")
    for key in (
        "ltx2_3_t2v",
        "ltx2_3_i2v_distilled",
        "ltx2_3_flf2v",
        "ltx2_3_iv2v_distilled",
        "basic_utils",
        "glm_ocr",
    ):
        assert key in resolved, f"missing {key} in {sorted(resolved)}"
    assert info["download_state"]["flux_klein"]["status"] == "downloading"


def test_wan_i2v_baked_capability():
    """WAN I2V multi advertises wan_22_fp8_lightx2v_i2v + shared-bake utilities immediately."""
    hb = {
        "supported_models": [
            "wan_22_fp8_lightx2v_i2v",
            "basic_utils",
            "aurasrv2",
            "swin2sr",
            "video_upscaler",
            "face_enhancer",
            "glm_ocr",
        ],
        "metrics": {
            "download_state": {
                "flux_klein": {"status": "downloading"},
                "z_image_turbo": {"status": "pending"},
            }
        },
    }
    info = _apply_heartbeat_to_info({}, hb)
    resolved = _resolve_runtime_supported_models(info, worker_model="wan_22_fp8_lightx2v_i2v")
    assert "wan_22_fp8_lightx2v_i2v" in resolved
    assert "glm_ocr" in resolved
    # Models that are still downloading should NOT appear in supported_models (worker filters before heartbeat)
    assert "flux_klein" not in resolved


def _run() -> int:
    tests = [
        test_round_trip_basic,
        test_ltx_omni_advertises_four_modes,
        test_wan_i2v_baked_capability,
    ]
    failed: List[str] = []
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except AssertionError as e:
            failed.append(f"{t.__name__}: {e}")
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed.append(f"{t.__name__}: {type(e).__name__}: {e}")
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(json.dumps({"failed": failed}, indent=2))
        return 1
    print("all heartbeat round-trip tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
