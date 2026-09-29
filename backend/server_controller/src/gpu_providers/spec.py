"""Resolve Docker image + env for a model from templates.json and controller env."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gpu_providers.base import WorkerSpec, normalize_gpu_tokens

logger = logging.getLogger("gpu_providers.spec")

_TEMPLATES_CACHE: Optional[List[dict]] = None


def _templates_path() -> Path:
    return Path(__file__).resolve().parent.parent / "templates.json"


def load_templates() -> List[dict]:
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    import json

    path = _templates_path()
    if not path.is_file():
        _TEMPLATES_CACHE = []
        return _TEMPLATES_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _TEMPLATES_CACHE = data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to load templates.json: %s", e)
        _TEMPLATES_CACHE = []
    return _TEMPLATES_CACHE


def parse_docker_options(docker_options: str) -> Tuple[Dict[str, str], List[str]]:
    env: Dict[str, str] = {}
    ports: List[str] = []
    if not docker_options:
        return env, ports
    try:
        parts = shlex.split(docker_options)
    except ValueError:
        parts = docker_options.split()
    i = 0
    while i < len(parts):
        token = parts[i]
        if token in ("-e", "--env") and i + 1 < len(parts):
            kv = parts[i + 1]
            if "=" in kv:
                key, val = kv.split("=", 1)
                env[key.strip()] = val
            i += 2
            continue
        if token in ("-p", "--publish") and i + 1 < len(parts):
            ports.append(parts[i + 1])
            i += 2
            continue
        i += 1
    return env, ports


def _controller_env_overrides() -> Dict[str, str]:
    public_url = (os.getenv("CONTROLLER_PUBLIC_URL") or "").strip()
    shared = os.getenv("CONTROLLER_SHARED_SECRET") or ""
    api_keys = os.getenv("CONTROLLER_API_KEYS") or os.getenv("CONTROLLER_API_KEY") or ""
    api_key = api_keys.split(",")[0].strip() if api_keys else shared
    hf = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_WRITE_REPO") or "").strip()
    out = {}
    if public_url:
        out["CONTROLLER_URL"] = public_url.rstrip("/")
    if shared:
        out["CONTROLLER_SHARED_SECRET"] = shared
    if api_key:
        out["CONTROLLER_API_KEY"] = api_key
    if hf:
        out["HF_TOKEN"] = hf
    crypto = os.getenv("WORKER_TRANSPORT_CRYPTO")
    if crypto:
        out["WORKER_TRANSPORT_CRYPTO"] = crypto
    enc_key = os.getenv("WORKER_PAYLOAD_ENCRYPTION_KEY")
    if enc_key:
        out["WORKER_PAYLOAD_ENCRYPTION_KEY"] = enc_key
    for r2_key in (
        "R2_ENDPOINT",
        "R2_BUCKET",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_TOKEN",
        "R2_PUBLIC_URL",
    ):
        val = os.getenv(r2_key)
        if val:
            out[r2_key] = val
    return out


def _template_by_uuid(template_uuid: Optional[str]) -> Optional[dict]:
    if not template_uuid:
        return None
    want = template_uuid.strip().lower()
    for row in load_templates():
        tu = str(row.get("template_uuid") or "").strip().lower()
        if tu and tu == want:
            return row
    return None


def _template_by_model(model: str) -> Optional[dict]:
    key = (model or "").strip().lower()
    if not key:
        return None
    best = None
    for row in load_templates():
        env, _ = parse_docker_options(str(row.get("docker_options") or ""))
        mk = str(env.get("MODEL_KEY") or "").strip().lower()
        name = str(row.get("template_name") or "").lower()
        if mk == key or key in mk or mk in key:
            return row
        if key and key.replace("_", "") in name.replace("_", "").replace("-", ""):
            best = best or row
    return best


def http_port_from_publish(ports: List[str], default: int = 8000) -> int:
    for p in ports:
        # "8000:8000" or "8000"
        left = p.split("/")[0]
        host, _, container = left.partition(":")
        try:
            if container:
                return int(container)
            return int(host)
        except ValueError:
            continue
    return default


def build_worker_spec(
    model: str,
    variant: Optional[str],
    template_uuid: Optional[str],
    disk_gb: int,
    allowed_gpus: Optional[List[str]],
) -> Optional[WorkerSpec]:
    row = _template_by_uuid(template_uuid) or _template_by_model(model)
    if not row:
        logger.warning("No templates.json entry for model=%s template=%s", model, template_uuid)
        return None
    image = str(row.get("image_path") or "").strip()
    if not image:
        logger.warning("templates.json row missing image_path for model=%s", model)
        return None
    env, ports = parse_docker_options(str(row.get("docker_options") or ""))
    env.update(_controller_env_overrides())
    if "MODEL_KEY" not in env or not env.get("MODEL_KEY"):
        env["MODEL_KEY"] = model
    command = str(row.get("cli_command") or "").strip()
    http_port = http_port_from_publish(ports, 8000)
    disk = int(row.get("disk_space") or disk_gb or 50)
    if disk_gb and disk_gb > disk:
        disk = disk_gb
    return WorkerSpec(
        model=model,
        variant=variant,
        image=image,
        disk_gb=disk,
        env=env,
        allowed_gpus=normalize_gpu_tokens(allowed_gpus),
        command=command,
        http_port=http_port,
    )


def inject_runtime_env(spec: WorkerSpec, *, provider: str, pod_label: str, public_url: str = "") -> Dict[str, str]:
    env = dict(spec.env)
    env["GPU_PROVIDER"] = provider
    env["CONTAINER_LABEL"] = pod_label
    if public_url:
        env["PUBLIC_BASE_URL"] = public_url.rstrip("/")
        # Workers that only know PUBLIC_IPADDR still get a hostname they can register;
        # controller overwrites the endpoint with public_url anyway.
        host = public_url
        for prefix in ("https://", "http://"):
            if host.startswith(prefix):
                host = host[len(prefix) :]
                break
        host = host.split("/")[0].split(":")[0]
        if host:
            env["PUBLIC_IPADDR"] = host
    return env
