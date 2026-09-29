"""Admit Studio jobs onto a Vultr VM using bible §11 reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

JobClass = Literal["chrome_heavy", "ffmpeg_edit", "doc_ingest", "llm_only"]

RESERVE = {
    "chrome_heavy": {"ram_mb": 3500, "vcpu": 3, "chrome": 1},
    "ffmpeg_edit": {"ram_mb": 1500, "vcpu": 2, "chrome": 0},
    "doc_ingest": {"ram_mb": 2000, "vcpu": 2, "chrome": 0},
    "llm_only": {"ram_mb": 500, "vcpu": 0, "chrome": 0},
}

VM_RAM_MB = 16 * 1024
VM_VCPU = 6


@dataclass
class VmSlots:
    ram_free: int = VM_RAM_MB
    vcpu_free: int = VM_VCPU
    chrome_free: int = 1
    queue: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    hour_started_at: float = 0.0
    vultr_id: str | None = None


def class_for_pipeline(pipeline: str) -> JobClass:
    if pipeline in {"clip-factory", "podcast-repurpose", "shorts", "talking-head"}:
        return "ffmpeg_edit"
    if pipeline == "documentary-montage":
        return "doc_ingest"
    return "chrome_heavy"


def can_admit(slots: VmSlots, job_class: JobClass) -> bool:
    need = RESERVE[job_class]
    return (
        slots.ram_free >= need["ram_mb"]
        and slots.vcpu_free >= need["vcpu"]
        and slots.chrome_free >= need["chrome"]
    )


def admit(slots: VmSlots, job_id: str, job_class: JobClass) -> bool:
    if not can_admit(slots, job_class):
        return False
    need = RESERVE[job_class]
    slots.ram_free -= need["ram_mb"]
    slots.vcpu_free -= need["vcpu"]
    slots.chrome_free -= need["chrome"]
    slots.running.append(job_id)
    return True


def release(slots: VmSlots, job_id: str, job_class: JobClass) -> None:
    if job_id in slots.running:
        slots.running.remove(job_id)
        need = RESERVE[job_class]
        slots.ram_free += need["ram_mb"]
        slots.vcpu_free += need["vcpu"]
        slots.chrome_free += need["chrome"]
