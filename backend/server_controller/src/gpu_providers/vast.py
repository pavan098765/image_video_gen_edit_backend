"""Vast.ai search-and-rent GPU instances."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from gpu_providers.base import CreatedPod, VAST_GPU_NAMES, WorkerSpec
from gpu_providers.spec import inject_runtime_env

logger = logging.getLogger("gpu_providers.vast")

BASE = "https://console.vast.ai/api/v0"


def _api_key() -> str:
    return (os.getenv("VAST_API_KEY") or os.getenv("VASTAI_API_KEY") or "").strip()


def is_configured() -> bool:
    return bool(_api_key())


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _search_offers(gpu_name: str) -> List[dict]:
    query = {
        "gpu_name": {"eq": gpu_name},
        "num_gpus": {"eq": 1},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "type": "ondemand",
        "reliability": {"gte": float(os.getenv("VAST_MIN_RELIABILITY", "0.95"))},
        "inet_down": {"gte": 50},
        "direct_port_count": {"gte": 1},
        "order": [["dph_total", "asc"]],
        "limit": 15,
    }
    try:
        resp = requests.post(f"{BASE}/bundles/", headers=_headers(), json=query, timeout=30)
        if resp.status_code == 200:
            data = resp.json() or {}
            offers = data.get("offers") or data.get("matches") or []
            if isinstance(offers, list) and offers:
                return offers
    except Exception as e:
        logger.info("Vast POST search failed: %s", e)
    try:
        import json as jsonlib

        resp = requests.get(
            f"{BASE}/bundles/",
            headers=_headers(),
            params={"q": jsonlib.dumps(query)},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            offers = data.get("offers") or data.get("matches") or []
            if isinstance(offers, list):
                return offers
    except Exception as e:
        logger.info("Vast GET search failed: %s", e)
    return []


def _offer_price(offer: dict) -> float:
    for key in ("dph_total", "dph_base", "search", "hourly_cost"):
        try:
            val = offer.get(key)
            if val is not None:
                return float(val)
        except (TypeError, ValueError):
            continue
    return 1e9


def _extract_endpoint(inst: dict, http_port: int) -> str:
    ip = (
        inst.get("public_ipaddr")
        or inst.get("public_ip")
        or inst.get("ssh_host")
        or ""
    )
    ip = str(ip).strip()
    ports = inst.get("ports") or {}
    host_port = None
    for key in (f"{http_port}/tcp", str(http_port), f"{http_port}"):
        mapping = ports.get(key) if isinstance(ports, dict) else None
        if mapping is None:
            continue
        host_port = _port_from_mapping(mapping)
        if host_port:
            break
    if not host_port:
        try:
            host_port = int(inst.get("direct_port_start") or 0) or None
        except (TypeError, ValueError):
            host_port = None
    if ip and host_port:
        return f"http://{ip}:{host_port}"
    if ip:
        return f"http://{ip}:{http_port}"
    return ""


def _port_from_mapping(mapping: Any) -> Optional[int]:
    if mapping is None:
        return None
    if isinstance(mapping, int):
        return mapping
    if isinstance(mapping, str):
        # "1.2.3.4:41234" or "41234"
        tail = mapping.rsplit(":", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return None
    if isinstance(mapping, (list, tuple)) and mapping:
        first = mapping[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                return int(first[1])
            except (TypeError, ValueError):
                return None
        return _port_from_mapping(first)
    if isinstance(mapping, dict):
        for k in ("HostPort", "host_port", "external_port", "PublicPort"):
            if mapping.get(k) is not None:
                try:
                    return int(mapping[k])
                except (TypeError, ValueError):
                    pass
    return None


def get_instance(instance_id: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{BASE}/instances/{instance_id}/",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.json() or {}
        inst = data.get("instances") or data
        if isinstance(inst, list) and inst:
            return inst[0]
        if isinstance(inst, dict):
            return inst
    except Exception as e:
        logger.debug("Vast get instance %s: %s", instance_id, e)
    return None


def wait_for_endpoint(instance_id: str, http_port: int, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = get_instance(instance_id)
        if inst:
            status = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
            if status in ("exited", "unknown", "offline", "error"):
                logger.warning("Vast instance %s status=%s", instance_id, status)
                return ""
            url = _extract_endpoint(inst, http_port)
            if url and status == "running":
                return url
        time.sleep(5)
    inst = get_instance(instance_id)
    return _extract_endpoint(inst, http_port) if inst else ""


def create(spec: WorkerSpec) -> Optional[CreatedPod]:
    if not is_configured():
        return None
    import uuid as uuidlib

    our_id = str(uuidlib.uuid4())
    port = spec.http_port or 8000
    env = inject_runtime_env(spec, provider="vast", pod_label=our_id)
    env_body = dict(env)
    env_body[f"-p {port}:{port}"] = "1"
    for gpu_token in spec.allowed_gpus:
        gpu_name = VAST_GPU_NAMES.get(gpu_token)
        if not gpu_name:
            continue
        offers = _search_offers(gpu_name)
        offers = sorted(offers, key=_offer_price)
        for offer in offers[:8]:
            offer_id = offer.get("id") or offer.get("ask_id")
            if not offer_id:
                continue
            body = {
                "image": spec.image,
                "disk": max(int(spec.disk_gb or 40), 16),
                "label": f"kala-{spec.model[:20]}-{our_id[:8]}",
                "runtype": "args",
                "target_state": "running",
                "cancel_unavail": True,
                "env": env_body,
            }
            try:
                resp = requests.put(
                    f"{BASE}/asks/{offer_id}/",
                    headers=_headers(),
                    json=body,
                    timeout=60,
                )
            except Exception as e:
                logger.info("Vast rent offer %s failed: %s", offer_id, e)
                continue
            if resp.status_code not in (200, 201):
                logger.info(
                    "Vast rent offer %s -> %s %s",
                    offer_id,
                    resp.status_code,
                    (resp.text or "")[:300],
                )
                continue
            data = resp.json() or {}
            instance_id = data.get("new_contract") or data.get("id")
            if not instance_id:
                continue
            instance_id = str(instance_id)
            url = wait_for_endpoint(instance_id, port, timeout=180)
            logger.info(
                "Vast instance %s gpu=%s url=%s model=%s",
                instance_id,
                gpu_name,
                url or "(pending)",
                spec.model,
            )
            return CreatedPod(
                provider="vast",
                pod_id=our_id,
                public_url=url,
                hourly_cost=_offer_price(offer) if _offer_price(offer) < 1e8 else 0.0,
                gpu_type=gpu_name,
                machine_id=f"vast:{gpu_token}",
                meta={
                    "provider_id": instance_id,
                    "offer_id": offer_id,
                    "http_port": port,
                },
            )
    return None


def refresh_public_url(meta: Dict[str, Any], http_port: int = 8000) -> str:
    instance_id = str((meta or {}).get("provider_id") or "")
    if not instance_id:
        return ""
    inst = get_instance(instance_id)
    if not inst:
        return ""
    return _extract_endpoint(inst, int((meta or {}).get("http_port") or http_port))


def destroy(pod_id: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    if not is_configured():
        return False
    real_id = str((meta or {}).get("provider_id") or pod_id)
    try:
        resp = requests.delete(
            f"{BASE}/instances/{real_id}/",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code in (200, 204, 404):
            logger.info("Vast destroy %s -> %s", real_id, resp.status_code)
            return True
        logger.warning("Vast destroy %s -> %s %s", real_id, resp.status_code, (resp.text or "")[:200])
    except Exception as e:
        logger.warning("Vast destroy failed for %s: %s", real_id, e)
    return False
