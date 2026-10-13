"""Vultr Cloud Compute pool: create from snapshot, never halt, destroy after billed hour."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

from studio_capacity import VmSlots, admit, can_admit, class_for_pipeline

VULTR_API = "https://api.vultr.com/v2"


class VultrPool:
    def __init__(self) -> None:
        self.api_key = os.getenv("VULTR_API_KEY", "")
        self.snapshot_id = os.getenv("VULTR_SNAPSHOT_ID") or os.getenv("SNAPSHOT_ID", "")
        self.plan = os.getenv("VULTR_PLAN", "vc2-6c-16gb")
        self.region = os.getenv("VULTR_REGION", "ewr")
        self.max_instances = int(os.getenv("MAX_INSTANCES", "2"))
        self.wait_threshold_sec = int(os.getenv("STUDIO_SCALE_WAIT_SEC", "480"))
        self.vms: list[VmSlots] = []
        self._created_at: dict[str, float] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def create_instance(self) -> str:
        mock = os.getenv("STUDIO_VULTR_MOCK", "").strip() in {"1", "true", "yes"}
        if not self.api_key or not self.snapshot_id:
            if not mock:
                raise RuntimeError(
                    "VULTR_API_KEY and VULTR_SNAPSHOT_ID are required in production. "
                    "Set STUDIO_VULTR_MOCK=1 only in tests."
                )
            fake = f"mock-{int(time.time())}"
            slots = VmSlots(hour_started_at=time.time(), vultr_id=fake)
            self.vms.append(slots)
            self._created_at[fake] = time.time()
            return fake
        body = {
            "region": self.region,
            "plan": self.plan,
            "snapshot_id": self.snapshot_id,
            "label": "kala-studio-engine",
            "hostname": "kala-studio-engine",
        }
        resp = requests.post(f"{VULTR_API}/instances", json=body, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        instance_id = resp.json()["instance"]["id"]
        slots = VmSlots(hour_started_at=time.time(), vultr_id=instance_id)
        self.vms.append(slots)
        self._created_at[instance_id] = time.time()
        return instance_id

    def destroy_instance(self, instance_id: str) -> None:
        if self.api_key and not instance_id.startswith("mock-"):
            requests.delete(f"{VULTR_API}/instances/{instance_id}", headers=self._headers(), timeout=30)
        self.vms = [v for v in self.vms if v.vultr_id != instance_id]
        self._created_at.pop(instance_id, None)

    def never_halt(self, instance_id: str) -> None:
        # Halt still bills; this pool only creates or destroys.
        _ = instance_id

    def place_job(self, job_id: str, pipeline: str, queued_at: Optional[float] = None) -> dict[str, Any]:
        job_class = class_for_pipeline(pipeline)
        now = queued_at or time.time()
        for vm in self.vms:
            if admit(vm, job_id, job_class):
                return {"action": "start", "vultr_id": vm.vultr_id, "queued": False}
        if self.vms:
            vm = self.vms[0]
            vm.queue.append(job_id)
            wait = now - (queued_at or now)
            if wait >= self.wait_threshold_sec and len(self.vms) < self.max_instances:
                vid = self.create_instance()
                return {"action": "scale_out", "vultr_id": vid, "queued": True}
            return {"action": "queued", "vultr_id": vm.vultr_id, "queued": True}
        if len(self.vms) < self.max_instances:
            vid = self.create_instance()
            vm = self.vms[-1]
            admit(vm, job_id, job_class)
            return {"action": "provisioning", "vultr_id": vid, "queued": False}
        return {"action": "queued_max", "vultr_id": None, "queued": True}

    def release_job(self, job_id: str, pipeline: str) -> None:
        job_class = class_for_pipeline(pipeline)
        for vm in self.vms:
            if job_id in vm.running or job_id in vm.queue:
                if job_id in vm.queue:
                    vm.queue.remove(job_id)
                from studio_capacity import release

                release(vm, job_id, job_class)
                return

    def reap_idle(self) -> list[str]:
        destroyed = []
        for vm in list(self.vms):
            if vm.running or vm.queue:
                continue
            started = self._created_at.get(vm.vultr_id or "", time.time())
            if time.time() - started >= 3600:
                if vm.vultr_id:
                    self.destroy_instance(vm.vultr_id)
                    destroyed.append(vm.vultr_id)
        return destroyed


pool = VultrPool()
