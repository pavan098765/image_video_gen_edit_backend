"""Shared types and GPU-name mapping for external GPU providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Tokens from model_requirements allowed_gpu_names / variant_allowed_gpus.
GPU_TOKEN_ALIASES = {
    "5090": ["5090"],
    "4090": ["4090"],
    "4080": ["4080"],
    "4070": ["4070"],
    "3090": ["3090"],
    "3080": ["3080"],
    "A6000": ["A6000", "RTX A6000"],
    "A5000": ["A5000", "RTX A5000"],
}

RUNPOD_GPU_IDS = {
    "5090": "NVIDIA GeForce RTX 5090",
    "4090": "NVIDIA GeForce RTX 4090",
    "4080": "NVIDIA GeForce RTX 4080",
    "4070": "NVIDIA GeForce RTX 4070",
    "3090": "NVIDIA GeForce RTX 3090",
    "3080": "NVIDIA GeForce RTX 3080",
    "A6000": "NVIDIA RTX A6000",
    "A5000": "NVIDIA RTX A5000",
}

VAST_GPU_NAMES = {
    "5090": "RTX 5090",
    "4090": "RTX 4090",
    "4080": "RTX 4080",
    "4070": "RTX 4070",
    "3090": "RTX 3090",
    "3080": "RTX 3080",
    "A6000": "RTX A6000",
    "A5000": "RTX A5000",
}


def normalize_gpu_tokens(allowed_gpus: Optional[List[str]]) -> List[str]:
    """Map raw GPU strings to canonical tokens (4090, 5090, ...)."""
    if not allowed_gpus:
        return ["4090", "5090", "3090"]
    out: List[str] = []
    seen = set()
    for raw in allowed_gpus:
        text = str(raw or "").upper()
        token = None
        for key in ("5090", "4090", "4080", "4070", "3090", "3080", "A6000", "A5000"):
            if key in text:
                token = key
                break
        if not token:
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out or ["4090", "5090", "3090"]


@dataclass
class WorkerSpec:
    """Everything needed to launch the same Docker worker on a non-Quickpod host."""

    model: str
    variant: Optional[str]
    image: str
    disk_gb: int
    env: Dict[str, str]
    allowed_gpus: List[str]
    command: str = ""
    http_port: int = 8000


@dataclass
class CreatedPod:
    provider: str
    pod_id: str
    public_url: str = ""
    hourly_cost: float = 0.0
    gpu_type: str = ""
    machine_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
