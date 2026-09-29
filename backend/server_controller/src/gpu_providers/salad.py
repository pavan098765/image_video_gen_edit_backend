"""SaladCloud container groups (one replica = one worker)."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from gpu_providers.base import CreatedPod, WorkerSpec
from gpu_providers.spec import inject_runtime_env

logger = logging.getLogger("gpu_providers.salad")

API_ROOT = "https://api.salad.com/api/public"

_GPU_CLASS_CACHE: Dict[str, List[dict]] = {}


def _api_key() -> str:
    return (os.getenv("SALAD_API_KEY") or "").strip()


def _org() -> str:
    return (os.getenv("SALAD_ORGANIZATION") or os.getenv("SALAD_ORG") or "").strip()


def _project() -> str:
    return (os.getenv("SALAD_PROJECT") or "").strip()


def is_configured() -> bool:
    return bool(_api_key() and _org() and _project())


def _headers() -> Dict[str, str]:
    return {
        "Salad-Api-Key": _api_key(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return f"{API_ROOT}/organizations/{_org()}/projects/{_project()}"


def _gpu_classes() -> List[dict]:
    org = _org()
    if org in _GPU_CLASS_CACHE:
        return _GPU_CLASS_CACHE[org]
    items: List[dict] = []
    try:
        resp = requests.get(
            f"{API_ROOT}/organizations/{org}/gpu-classes",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            items = data.get("items") or data.get("gpu_classes") or []
            if not isinstance(items, list):
                items = []
    except Exception as e:
        logger.info("Salad gpu-classes fetch failed: %s", e)
    _GPU_CLASS_CACHE[org] = items
    return items


def _gpu_class_ids_for_token(token: str) -> List[str]:
    needle = token.upper()
    ids: List[str] = []
    for item in _gpu_classes():
        name = str(item.get("name") or item.get("title") or "").upper()
        gid = item.get("id") or item.get("gpu_class_id")
        if gid and needle in name:
            ids.append(str(gid))
    return ids


def _safe_name(model: str, suffix: str) -> str:
    raw = f"sc-{model}-{suffix}".lower()
    raw = re.sub(r"[^a-z0-9-]", "-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-")
    if not raw or not raw[0].isalpha():
        raw = f"sc{raw}"
    return raw[:62].rstrip("-") or "scworker"


def _group_url(data: dict) -> str:
    net = data.get("networking") or {}
    dns = net.get("dns") or net.get("fqdn") or data.get("dns") or ""
    if dns:
        dns = str(dns).strip()
        if dns.startswith("http"):
            return dns.rstrip("/")
        return f"https://{dns}".rstrip("/")
    return ""


def _start_group(name: str) -> None:
    try:
        requests.post(
            f"{_base()}/containers/{name}/start",
            headers=_headers(),
            timeout=30,
        )
    except Exception as e:
        logger.debug("Salad start %s: %s", name, e)


def create(spec: WorkerSpec) -> Optional[CreatedPod]:
    if not is_configured():
        return None
    import uuid as uuidlib

    our_id = str(uuidlib.uuid4())
    name = _safe_name(spec.model, our_id[:8])
    port = spec.http_port or 8000
    env = inject_runtime_env(spec, provider="salad", pod_label=our_id)
    gpu_ids: List[str] = []
    gpu_label = ""
    for token in spec.allowed_gpus:
        ids = _gpu_class_ids_for_token(token)
        if ids:
            gpu_ids.extend(ids)
            gpu_label = gpu_label or token
    gpu_ids = list(dict.fromkeys(gpu_ids))
    if not gpu_ids:
        logger.warning("Salad: no GPU class ids matched %s", spec.allowed_gpus)
        return None
    body = {
        "name": name,
        "display_name": f"kala {spec.model} {our_id[:8]}",
        "container": {
            "image": spec.image,
            "resources": {
                "cpu": int(os.getenv("SALAD_CPU", "8")),
                "memory": int(os.getenv("SALAD_MEMORY_MB", "32768")),
                "gpu_classes": gpu_ids,
            },
            "environment_variables": env,
        },
        "autostart_policy": True,
        "restart_policy": "always",
        "replicas": 1,
        "networking": {
            "protocol": "http",
            "port": port,
            "auth": False,
        },
    }
    try:
        resp = requests.post(
            f"{_base()}/containers",
            headers=_headers(),
            json=body,
            timeout=60,
        )
    except Exception as e:
        logger.info("Salad create failed: %s", e)
        return None
    if resp.status_code not in (200, 201):
        logger.info("Salad create %s: %s", resp.status_code, (resp.text or "")[:400])
        return None
    data = resp.json() or {}
    group_name = str(data.get("name") or name)
    _start_group(group_name)
    url = _group_url(data)
    if not url:
        # DNS can lag; poll a few times.
        for _ in range(12):
            time.sleep(5)
            try:
                g = requests.get(
                    f"{_base()}/containers/{group_name}",
                    headers=_headers(),
                    timeout=20,
                )
                if g.status_code == 200:
                    url = _group_url(g.json() or {})
                    if url:
                        break
            except Exception:
                pass
    logger.info("Salad group %s url=%s model=%s", group_name, url or "(pending)", spec.model)
    return CreatedPod(
        provider="salad",
        pod_id=our_id,
        public_url=url,
        gpu_type=gpu_label or "salad-gpu",
        machine_id=f"salad:{gpu_label or 'gpu'}",
        meta={"provider_id": group_name, "http_port": port},
    )


def destroy(pod_id: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    if not is_configured():
        return False
    name = str((meta or {}).get("provider_id") or pod_id)
    try:
        resp = requests.delete(
            f"{_base()}/containers/{name}",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code in (200, 202, 204, 404):
            logger.info("Salad destroy %s -> %s", name, resp.status_code)
            return True
        logger.warning("Salad destroy %s -> %s %s", name, resp.status_code, (resp.text or "")[:200])
    except Exception as e:
        logger.warning("Salad destroy failed for %s: %s", name, e)
    return False
