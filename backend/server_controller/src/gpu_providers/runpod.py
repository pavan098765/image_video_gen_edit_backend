"""RunPod GPU pods (REST v2, with v1 fallback)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

from gpu_providers.base import CreatedPod, RUNPOD_GPU_IDS, WorkerSpec
from gpu_providers.spec import inject_runtime_env

logger = logging.getLogger("gpu_providers.runpod")

V2_BASE = "https://api.runpod.io/v2"
V1_BASE = "https://rest.runpod.io/v1"


def _api_key() -> str:
    return (os.getenv("RUNPOD_API_KEY") or "").strip()


def is_configured() -> bool:
    return bool(_api_key())


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _cloud_order() -> list:
    preferred = (os.getenv("RUNPOD_CLOUD") or "COMMUNITY").strip().upper()
    if preferred not in ("COMMUNITY", "SECURE"):
        preferred = "COMMUNITY"
    other = "SECURE" if preferred == "COMMUNITY" else "COMMUNITY"
    return [preferred, other]


def public_url_for_pod(pod_id: str, http_port: int = 8000) -> str:
    return f"https://{pod_id}-{http_port}.proxy.runpod.net"


def _create_v2(body: Dict[str, Any]) -> Optional[dict]:
    resp = requests.post(f"{V2_BASE}/pods", headers=_headers(), json=body, timeout=60)
    if resp.status_code in (200, 201):
        return resp.json()
    logger.info("RunPod v2 create %s: %s", resp.status_code, (resp.text or "")[:400])
    return None


def _create_v1(body: Dict[str, Any]) -> Optional[dict]:
    resp = requests.post(f"{V1_BASE}/pods", headers=_headers(), json=body, timeout=60)
    if resp.status_code in (200, 201):
        return resp.json()
    logger.info("RunPod v1 create %s: %s", resp.status_code, (resp.text or "")[:400])
    return None


def _parse_id(data: dict) -> str:
    for key in ("id", "podId", "pod_id"):
        val = data.get(key)
        if val:
            return str(val)
    return ""


def create(spec: WorkerSpec) -> Optional[CreatedPod]:
    if not is_configured():
        return None
    port = spec.http_port or 8000
    for gpu_token in spec.allowed_gpus:
        gpu_id = RUNPOD_GPU_IDS.get(gpu_token)
        if not gpu_id:
            continue
        for cloud in _cloud_order():
            # CONTAINER_LABEL is our uuid; RunPod's real id is stored as provider_id.
            import uuid as uuidlib

            our_id = str(uuidlib.uuid4())
            env = inject_runtime_env(spec, provider="runpod", pod_label=our_id)
            v2_body = {
                "name": f"kala-{spec.model[:24]}-{our_id[:8]}",
                "image": spec.image,
                "gpu": {"id": gpu_id, "count": 1},
                "disk": max(int(spec.disk_gb or 40), 20),
                "env": env,
                "ports": [f"{port}/http"],
                "cloud": cloud,
            }
            data = _create_v2(v2_body)
            provider_id = _parse_id(data) if data else ""
            if not provider_id:
                v1_body = {
                    "name": f"kala-{spec.model[:24]}-{our_id[:8]}",
                    "imageName": spec.image,
                    "gpuTypeId": gpu_id,
                    "gpuCount": 1,
                    "containerDiskInGb": max(int(spec.disk_gb or 40), 20),
                    "volumeInGb": 0,
                    "ports": f"{port}/http",
                    "env": env,
                    "cloudType": cloud,
                    "supportPublicIp": True,
                }
                data = _create_v1(v1_body)
                provider_id = _parse_id(data) if data else ""
            if not provider_id:
                continue
            url = public_url_for_pod(provider_id, port)
            logger.info(
                "RunPod pod created id=%s cloud=%s gpu=%s model=%s",
                provider_id,
                cloud,
                gpu_id,
                spec.model,
            )
            return CreatedPod(
                provider="runpod",
                pod_id=our_id,
                public_url=url,
                gpu_type=gpu_id,
                machine_id=f"runpod:{gpu_token}",
                meta={"provider_id": provider_id, "cloud": cloud, "http_port": port},
            )
    return None


def destroy(pod_id: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    if not is_configured():
        return False
    real_id = str((meta or {}).get("provider_id") or pod_id)
    last_err = None
    for url in (f"{V2_BASE}/pods/{real_id}", f"{V1_BASE}/pods/{real_id}"):
        try:
            resp = requests.delete(url, headers=_headers(), timeout=30)
            if resp.status_code in (200, 204, 404):
                logger.info("RunPod destroy %s -> %s", real_id, resp.status_code)
                return True
            last_err = f"{resp.status_code} {(resp.text or '')[:200]}"
        except Exception as e:
            last_err = str(e)
    logger.warning("RunPod destroy failed for %s: %s", real_id, last_err)
    return False
