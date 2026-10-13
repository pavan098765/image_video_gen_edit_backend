"""
Scheduler strategy abstraction for job-to-pod assignment.

Phase 0 introduces a pluggable interface so we can keep legacy behavior
and progressively implement fixed-fleet windowed scheduling.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import requests

from worker_transport_crypto import encode_worker_request_body
from security import sign_payload_bytes

logger = logging.getLogger("scheduler_strategy")

MODE_LEGACY_DYNAMIC_FLEET = "legacy_dynamic_fleet"
MODE_FIXED_FLEET_WINDOWED = "fixed_fleet_windowed"

STUDIO_MODELS = {"studio", "studio_engine"}


def _is_studio_model(model: Optional[str]) -> bool:
    return (model or "").strip().lower() in STUDIO_MODELS


def _is_studio_engine_identity(value: Optional[str]) -> bool:
    return (value or "").strip().lower() == "studio_engine"


class SchedulerStrategy(ABC):
    """Contract for all scheduler strategies."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Scheduler mode identifier."""
        raise NotImplementedError

    @abstractmethod
    async def assign_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        """Assign job and return worker_id."""
        raise NotImplementedError

    @abstractmethod
    async def simulate_assignment(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return dry-run scheduling details without side effects."""
        raise NotImplementedError


@dataclass
class LegacyDynamicFleetScheduler(SchedulerStrategy):
    """Current production scheduling behavior."""

    pod_manager: Any

    @property
    def mode(self) -> str:
        return MODE_LEGACY_DYNAMIC_FLEET

    async def assign_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        if _is_studio_model(model):
            raise RuntimeError(
                "Studio jobs are pulled by studio_engine on Vultr, not dispatched to QuickPod"
            )
        return await self.pod_manager.dispatch_job(job_id, model, mode)

    async def simulate_assignment(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "job_id": job_id,
            "model": model,
            "decision": "not_available",
            "reason": "legacy scheduler has no dry-run decision API",
        }


@dataclass
class FixedFleetWindowedScheduler(SchedulerStrategy):
    """
    New mode placeholder.

    Phase 0 keeps behavior equivalent to legacy while architecture is wired.
    Subsequent phases will replace this implementation with fixed-fleet TTL logic.
    """

    pod_manager: Any

    @property
    def mode(self) -> str:
        return MODE_FIXED_FLEET_WINDOWED

    def _get_model_requirement(self, model: str) -> Dict[str, Any]:
        req = {}
        try:
            req = self.pod_manager.model_requirements.get(model, {})
        except Exception:
            req = {}
        return req if isinstance(req, dict) else {}

    def _pending_grace_enabled(self) -> bool:
        return os.getenv("PENDING_MODEL_GRACE_ENABLED", "true").lower() == "true"

    def _pending_max_wait_seconds(self) -> int:
        return max(0, int(os.getenv("PENDING_MODEL_MAX_WAIT_SECONDS", "20")))

    def _pending_poll_interval_seconds(self) -> float:
        interval_ms = max(100, int(os.getenv("PENDING_MODEL_POLL_INTERVAL_MS", "500")))
        return float(interval_ms) / 1000.0

    def _pending_hint_timeout_seconds(self) -> int:
        return max(1, int(os.getenv("PENDING_MODEL_HINT_TIMEOUT_SECONDS", "3")))

    def _increment_metric(self, key: str):
        metrics = getattr(self.pod_manager, "fixed_fleet_metrics", None)
        if isinstance(metrics, dict):
            metrics[key] = int(metrics.get(key, 0)) + 1

    def _resolve_static_supported_models(self, model: str) -> Set[str]:
        req = self._get_model_requirement(model)
        group_id = req.get("group_id")
        explicit_supported = req.get("supported_models")

        static_supported: Set[str] = {model}
        if isinstance(explicit_supported, list):
            for m in explicit_supported:
                if isinstance(m, str) and m.strip():
                    static_supported.add(m.strip().lower())
            return static_supported

        if not group_id:
            return static_supported

        # Discover all models in the same group_id.
        try:
            all_requirements = self.pod_manager.model_requirements or {}
        except Exception:
            all_requirements = {}
        for model_name, model_req in all_requirements.items():
            if (
                isinstance(model_name, str)
                and isinstance(model_req, dict)
                and model_req.get("group_id") == group_id
            ):
                static_supported.add(model_name.strip().lower())
        return static_supported

    def _resolve_runtime_supported_models(self, pod_uuid: str) -> Set[str]:
        # Runtime capabilities are stored in Worker.info by register/heartbeat.
        try:
            from db import SessionLocal, Worker
        except Exception:
            return set()

        with SessionLocal() as db:
            worker = (
                db.query(Worker)
                .filter(Worker.container_id == pod_uuid)
                .order_by(Worker.updated_at.desc())
                .first()
            )
            if not worker or not isinstance(worker.info, dict):
                return set()

            info = worker.info
            raw_supported = info.get("supported_models")
            if not isinstance(raw_supported, list):
                return set()

            supported = set()
            for m in raw_supported:
                if isinstance(m, str) and m.strip():
                    supported.add(m.strip().lower())
            if worker.model and isinstance(worker.model, str):
                supported.add(worker.model.strip().lower())
            return supported

    def _pod_supports_model(self, pod_uuid: str, pod_baked_model: str, requested_model: str) -> bool:
        requested = requested_model.strip().lower()
        static_supported = self._resolve_static_supported_models(pod_baked_model)
        runtime_supported = self._resolve_runtime_supported_models(pod_uuid)

        # Hybrid static fallback:
        # - If runtime supported list exists, require intersection with static support.
        # - If runtime list missing, trust static support map.
        if runtime_supported:
            return requested in (static_supported & runtime_supported)
        return requested in static_supported

    def _get_model_eta_seconds(self, model: str) -> int:
        static_eta = self._get_static_model_eta_seconds(model)
        dynamic = self._get_dynamic_model_eta_seconds(model, static_eta)
        if dynamic is None:
            return static_eta
        return int(dynamic)

    def _get_worker_for_pod(self, pod_uuid: str):
        try:
            from db import SessionLocal, Worker
        except Exception:
            return None

        with SessionLocal() as db:
            return (
                db.query(Worker)
                .filter(Worker.container_id == pod_uuid)
                .order_by(Worker.updated_at.desc())
                .first()
            )

    async def _probe_worker_health(self, endpoint: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
        endpoint = (endpoint or "").strip()
        if not endpoint:
            return None
        health_url = f"{endpoint.rstrip('/')}/health"

        def _fetch():
            resp = requests.get(health_url, timeout=timeout_seconds)
            resp.raise_for_status()
            body = resp.json()
            return body if isinstance(body, dict) else {}

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.debug("pending_wait health probe failed endpoint=%s err=%s", endpoint, e)
            return None

    async def _send_download_priority_hint(
        self, endpoint: str, model: str, timeout_seconds: int
    ) -> Optional[Dict[str, Any]]:
        endpoint = (endpoint or "").strip()
        if not endpoint:
            return None
        hint_url = f"{endpoint.rstrip('/')}/download_priority"
        payload = {"model": model}

        def _send():
            body = encode_worker_request_body(payload)
            headers = {"Content-Type": "application/json"}
            try:
                headers["X-Signature"] = sign_payload_bytes(body)
            except ValueError:
                pass
            resp = requests.post(
                hint_url, data=body, headers=headers, timeout=timeout_seconds
            )
            resp.raise_for_status()
            resp_json = resp.json()
            return resp_json if isinstance(resp_json, dict) else {}

        try:
            return await asyncio.to_thread(_send)
        except Exception as e:
            logger.debug("pending_wait hint failed endpoint=%s model=%s err=%s", endpoint, model, e)
            return None

    async def _find_pending_candidate(
        self,
        model: str,
        model_eta_seconds: int,
        safety_seconds: int,
        drain_guard_seconds: int,
        max_wait_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        now = datetime.utcnow()
        ttl_seconds = int(os.getenv("POD_MAX_LIFETIME_SECONDS", "840"))
        requested = (model or "").strip().lower()
        candidates: List[Dict[str, Any]] = []

        async with self.pod_manager._lock:
            pods = list(self.pod_manager.individual_pods.items())

        for pod_uuid, info in pods:
            if info.meta.get("is_cache_refresh_pod"):
                continue
            status_value = str(getattr(info.status, "value", info.status)).lower()
            if status_value != "ready":
                continue
            if info.machine_id and self.pod_manager.is_machine_bad(info.machine_id):
                continue
            if self.pod_manager.lifecycle.is_recently_destroyed(pod_uuid):
                continue

            pod_model = (info.model or "").strip().lower()
            if not pod_model:
                continue
            if _is_studio_engine_identity(pod_model) and not _is_studio_model(requested):
                continue
            if _is_studio_model(requested):
                continue

            # Candidate must be statically group-compatible with requested model.
            static_supported = self._resolve_static_supported_models(pod_model)
            if requested not in static_supported:
                continue

            worker = self._get_worker_for_pod(pod_uuid)
            if not worker:
                continue
            endpoint = (getattr(worker, "endpoint", None) or "").strip()
            if not endpoint:
                continue

            runtime_supported = self._resolve_runtime_supported_models(pod_uuid)
            if requested in runtime_supported:
                # Already runtime-ready; this path is only for pending.
                continue

            worker_info = worker.info if isinstance(worker.info, dict) else {}
            download_state = worker_info.get("download_state")
            if not isinstance(download_state, dict):
                continue
            model_state = download_state.get(requested)
            if not isinstance(model_state, dict):
                continue
            status = str(model_state.get("status", "")).strip().lower()
            if status not in {"pending", "downloading"}:
                continue

            ready_in_seconds = model_state.get("ready_in_seconds")
            if isinstance(ready_in_seconds, (int, float)) and ready_in_seconds >= 0:
                ready_in = int(ready_in_seconds)
            else:
                ready_in = max_wait_seconds

            age_seconds = max(0, int((now - info.created_at).total_seconds()))
            remaining_seconds = max(0, ttl_seconds - age_seconds)
            queued_runtime_seconds = int(info.queue_size) * model_eta_seconds
            predicted_finish_seconds = (
                queued_runtime_seconds + max(0, ready_in) + model_eta_seconds + safety_seconds
            )
            can_wait_and_admit = (
                remaining_seconds > drain_guard_seconds
                and predicted_finish_seconds < remaining_seconds
            )
            if not can_wait_and_admit:
                continue

            candidates.append(
                {
                    "pod_uuid": pod_uuid,
                    "endpoint": endpoint,
                    "machine_id": info.machine_id,
                    "queue_size": int(info.queue_size),
                    "remaining_seconds": remaining_seconds,
                    "ready_in_seconds": max(0, ready_in),
                    "download_status": status,
                }
            )

        if not candidates:
            return None
        candidates.sort(
            key=lambda x: (
                int(x.get("ready_in_seconds", max_wait_seconds)),
                int(x.get("queue_size", 0)),
                -int(x.get("remaining_seconds", 0)),
            )
        )
        return candidates[0]

    def _get_static_model_eta_seconds(self, model: str) -> int:
        req = self._get_model_requirement(model)
        eta = req.get("eta_seconds_default")
        if isinstance(eta, (int, float)) and eta > 0:
            return int(eta)
        return int(os.getenv("SCHEDULER_DEFAULT_MODEL_ETA_SECONDS", "180"))

    def _is_dynamic_eta_model_allowed(self, model: str) -> bool:
        allowed_groups_raw = os.getenv("SCHEDULER_DYNAMIC_ETA_ALLOWED_GROUPS", "").strip()
        if not allowed_groups_raw:
            return True
        allowed_groups = {
            x.strip().lower() for x in allowed_groups_raw.split(",") if x.strip()
        }
        req = self._get_model_requirement(model)
        group_id = str(req.get("group_id") or "").strip().lower()
        return bool(group_id and group_id in allowed_groups)

    def _get_dynamic_model_eta_seconds(self, model: str, static_eta: int) -> Optional[int]:
        enabled = os.getenv("SCHEDULER_DYNAMIC_ETA_ENABLED", "false").lower() == "true"
        if not enabled:
            return None
        if not self._is_dynamic_eta_model_allowed(model):
            return None

        min_samples = max(1, int(os.getenv("SCHEDULER_DYNAMIC_ETA_MIN_SAMPLES", "5")))
        observe_only = (
            os.getenv("SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY", "true").lower() == "true"
        )
        min_seconds = max(1, int(os.getenv("SCHEDULER_DYNAMIC_ETA_MIN_SECONDS", "10")))
        max_seconds = max(
            min_seconds, int(os.getenv("SCHEDULER_DYNAMIC_ETA_MAX_SECONDS", "1200"))
        )

        try:
            from db import SessionLocal, ModelEtaStat
        except Exception:
            return None

        model_norm = (model or "").strip().lower()
        if not model_norm:
            return None

        with SessionLocal() as db:
            row = db.query(ModelEtaStat).filter(ModelEtaStat.model == model_norm).first()
            if not row:
                return None
            sample_count = int(row.sample_count or 0)
            if sample_count < min_samples:
                return None
            learned = float(row.ewma_eta_seconds or 0.0)
            if learned <= 0:
                return None
            learned = max(float(min_seconds), min(float(max_seconds), learned))
            if observe_only:
                logger.info(
                    "dynamic_eta observe_only model=%s learned=%.2fs static=%ss samples=%s",
                    model_norm,
                    learned,
                    static_eta,
                    sample_count,
                )
                return None
            return int(round(learned))

    async def _evaluate_candidate_pods(
        self, model: str, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        ttl_seconds = int(os.getenv("POD_MAX_LIFETIME_SECONDS", "840"))
        drain_guard_seconds = int(os.getenv("POD_DRAIN_GUARD_SECONDS", "90"))
        safety_seconds = int(os.getenv("SCHEDULER_ADMISSION_SAFETY_SECONDS", "30"))
        model_eta_seconds = self._get_model_eta_seconds(model)
        static_eta_seconds = self._get_static_model_eta_seconds(model)

        soft_governor = {}
        if hasattr(self.pod_manager, "get_soft_governor_state"):
            try:
                soft_governor = await self.pod_manager.get_soft_governor_state()
            except Exception as e:
                logger.debug("soft governor lookup failed: %s", e)
                soft_governor = {}
        if soft_governor.get("active"):
            safety_seconds += int(soft_governor.get("extra_safety_seconds", 0))
            drain_guard_seconds += int(soft_governor.get("extra_drain_guard_seconds", 0))

        async with self.pod_manager._lock:
            pods = list(self.pod_manager.individual_pods.items())

        evaluated = []
        for pod_uuid, info in pods:
            if info.meta.get("is_cache_refresh_pod"):
                continue

            status_value = str(getattr(info.status, "value", info.status)).lower()
            if status_value != "ready":
                continue

            if info.machine_id and self.pod_manager.is_machine_bad(info.machine_id):
                continue
            if self.pod_manager.lifecycle.is_recently_destroyed(pod_uuid):
                continue

            pod_model = (info.model or "").strip().lower()
            if not pod_model:
                continue
            if _is_studio_engine_identity(pod_model) and not _is_studio_model(model):
                continue
            if _is_studio_model(model):
                continue
            if not self._pod_supports_model(pod_uuid, pod_model, model):
                continue

            age_seconds = max(0, int((now - info.created_at).total_seconds()))
            remaining_seconds = max(0, ttl_seconds - age_seconds)
            queued_runtime_seconds = int(info.queue_size) * model_eta_seconds
            predicted_finish_seconds = queued_runtime_seconds + model_eta_seconds + safety_seconds
            can_admit = (
                remaining_seconds > drain_guard_seconds
                and predicted_finish_seconds < remaining_seconds
            )
            evaluated.append(
                {
                    "pod_uuid": pod_uuid,
                    "pod_model": pod_model,
                    "machine_id": info.machine_id,
                    "queue_size": int(info.queue_size),
                    "age_seconds": age_seconds,
                    "remaining_seconds": remaining_seconds,
                    "model_eta_seconds": model_eta_seconds,
                    "queued_runtime_seconds": queued_runtime_seconds,
                    "predicted_finish_seconds": predicted_finish_seconds,
                    "can_admit": can_admit,
                }
            )

        feasible = [x for x in evaluated if x["can_admit"]]
        best = min(feasible, key=lambda x: x["predicted_finish_seconds"]) if feasible else None
        return {
            "evaluated": evaluated,
            "feasible": feasible,
            "best": best,
            "ttl_seconds": ttl_seconds,
            "drain_guard_seconds": drain_guard_seconds,
            "safety_seconds": safety_seconds,
            "model_eta_seconds": model_eta_seconds,
            "static_eta_seconds": static_eta_seconds,
            "eta_source": (
                "dynamic"
                if int(model_eta_seconds) != int(static_eta_seconds)
                else "static"
            ),
            "soft_governor": soft_governor,
        }

    async def simulate_assignment(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        evaluation = await self._evaluate_candidate_pods(model=model, mode=mode)
        evaluated: List[Dict[str, Any]] = evaluation["evaluated"]
        feasible: List[Dict[str, Any]] = evaluation["feasible"]
        best = evaluation["best"]

        return {
            "mode": self.mode,
            "job_id": job_id,
            "model": model,
            "decision": "admit_to_existing_ready_pod" if best else "no_feasible_ready_pod",
            "reason": (
                "pod_can_finish_before_ttl_cutoff"
                if best
                else "ttl_or_capacity_constraints_prevent_admission"
            ),
            "fleet_snapshot": {
                "pods_considered": len(evaluated),
                "pods_feasible": len(feasible),
                "ttl_seconds": evaluation["ttl_seconds"],
                "drain_guard_seconds": evaluation["drain_guard_seconds"],
                "safety_seconds": evaluation["safety_seconds"],
                "model_eta_seconds": evaluation["model_eta_seconds"],
                "static_eta_seconds": evaluation["static_eta_seconds"],
                "eta_source": evaluation["eta_source"],
                "soft_governor": evaluation.get("soft_governor", {}),
            },
            "selected": best,
        }

    async def assign_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        if _is_studio_model(model):
            raise RuntimeError(
                "Studio jobs are pulled by studio_engine on Vultr, not dispatched to QuickPod"
            )
        # Decision order (strict fixed-mode):
        # 1) feasible existing READY pod
        # 2) create new pod and assign there
        # 3) fallback to legacy dispatch only if create failed
        try:
            from db import SessionLocal, Worker, utc_naive_now
        except Exception:
            SessionLocal = None

        def _assign_to_worker_on_pod(pod_uuid: str, decision_label: str, details: Dict[str, Any]) -> Optional[str]:
            if SessionLocal is None:
                return None
            with SessionLocal() as db:
                worker = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.model != "studio_engine",
                    )
                    .order_by(Worker.updated_at.desc())
                    .first()
                )
                if worker and _is_studio_engine_identity(worker.model):
                    return None
                if worker and worker.status == "ready":
                    worker.status = "dispatching"
                    worker.last_active = utc_naive_now()
                    db.commit()
                    logger.info(
                        "fixed_fleet_admission decision=%s job=%s model=%s pod=%s machine=%s queue=%s remaining=%s template_uuid=%s",
                        decision_label,
                        job_id,
                        model,
                        pod_uuid,
                        details.get("machine_id"),
                        details.get("queue_size"),
                        details.get("remaining_seconds"),
                        details.get("template_uuid"),
                    )
                    if decision_label == "existing_feasible":
                        self._increment_metric("admission_existing_feasible")
                    elif decision_label == "created_new_pod":
                        self._increment_metric("admission_created_new_pod")
                    elif decision_label == "pending_wait_ready":
                        self._increment_metric("admission_pending_wait_success")
                    return worker.worker_id
            return None

        try:
            evaluation = await self._evaluate_candidate_pods(model=model, mode=mode)
            best = evaluation["best"]
            if best:
                pod_uuid = best["pod_uuid"]
                reserved = await self.pod_manager.reserve_slot_on_pod(pod_uuid, job_id)
                if reserved:
                    worker_id = _assign_to_worker_on_pod(
                        pod_uuid=pod_uuid,
                        decision_label="existing_feasible",
                        details=best,
                    )
                    if worker_id:
                        return worker_id
                    # Could not resolve usable worker; release reserved slot immediately.
                    await self.pod_manager.release_slot_on_pod(pod_uuid, job_id)

            # Pending-model grace path: ask a compatible READY pod to prioritize download, then wait briefly.
            if self._pending_grace_enabled():
                max_wait_seconds = self._pending_max_wait_seconds()
                poll_interval_seconds = self._pending_poll_interval_seconds()
                hint_timeout_seconds = self._pending_hint_timeout_seconds()
                if max_wait_seconds > 0:
                    pending = await self._find_pending_candidate(
                        model=model,
                        model_eta_seconds=int(evaluation["model_eta_seconds"]),
                        safety_seconds=int(evaluation["safety_seconds"]),
                        drain_guard_seconds=int(evaluation["drain_guard_seconds"]),
                        max_wait_seconds=max_wait_seconds,
                    )
                    if pending:
                        wait_start = time.monotonic()
                        self._increment_metric("admission_pending_wait_started")
                        logger.info(
                            "fixed_fleet_admission decision=pending_wait_started job=%s model=%s pod=%s machine=%s queue=%s remaining=%s est_ready_in=%s",
                            job_id,
                            model,
                            pending.get("pod_uuid"),
                            pending.get("machine_id"),
                            pending.get("queue_size"),
                            pending.get("remaining_seconds"),
                            pending.get("ready_in_seconds"),
                        )
                        await self._send_download_priority_hint(
                            endpoint=str(pending.get("endpoint") or ""),
                            model=model,
                            timeout_seconds=hint_timeout_seconds,
                        )
                        deadline = wait_start + max_wait_seconds
                        hinted_pod_uuid = str(pending.get("pod_uuid") or "")
                        while time.monotonic() < deadline and hinted_pod_uuid:
                            health = await self._probe_worker_health(
                                endpoint=str(pending.get("endpoint") or ""),
                                timeout_seconds=hint_timeout_seconds,
                            )
                            advertised = []
                            if isinstance(health, dict):
                                advertised = health.get("advertised_supported_models") or []
                            advertised_set = {
                                str(x).strip().lower()
                                for x in advertised
                                if isinstance(x, str) and str(x).strip()
                            }
                            if (model or "").strip().lower() in advertised_set:
                                reserved = await self.pod_manager.reserve_slot_on_pod(
                                    hinted_pod_uuid, job_id
                                )
                                if reserved:
                                    worker_id = _assign_to_worker_on_pod(
                                        pod_uuid=hinted_pod_uuid,
                                        decision_label="pending_wait_ready",
                                        details=pending,
                                    )
                                    if worker_id:
                                        waited_ms = int((time.monotonic() - wait_start) * 1000)
                                        metrics = getattr(self.pod_manager, "fixed_fleet_metrics", None)
                                        if isinstance(metrics, dict):
                                            metrics["admission_pending_wait_total_ms"] = int(
                                                metrics.get("admission_pending_wait_total_ms", 0)
                                            ) + waited_ms
                                        logger.info(
                                            "fixed_fleet_admission decision=pending_wait_ready job=%s model=%s pod=%s waited_ms=%s",
                                            job_id,
                                            model,
                                            hinted_pod_uuid,
                                            waited_ms,
                                        )
                                        return worker_id
                                    await self.pod_manager.release_slot_on_pod(
                                        hinted_pod_uuid, job_id
                                    )
                            await asyncio.sleep(poll_interval_seconds)
                        waited_ms = int((time.monotonic() - wait_start) * 1000)
                        self._increment_metric("admission_pending_wait_timeout")
                        metrics = getattr(self.pod_manager, "fixed_fleet_metrics", None)
                        if isinstance(metrics, dict):
                            metrics["admission_pending_wait_total_ms"] = int(
                                metrics.get("admission_pending_wait_total_ms", 0)
                            ) + waited_ms
                        logger.info(
                            "fixed_fleet_admission decision=pending_wait_timeout job=%s model=%s pod=%s waited_ms=%s",
                            job_id,
                            model,
                            hinted_pod_uuid,
                            waited_ms,
                        )

            # No feasible existing pod -> try create new pod first.
            chosen_template_uuid: Optional[str] = None
            try:
                chosen_template_uuid = self.pod_manager.get_template_for_variant(model, mode)
            except Exception:
                chosen_template_uuid = None
            logger.info(
                "fixed_fleet_admission decision=creating_new_pod job=%s model=%s mode=%s template_uuid=%s reason=no_feasible_existing_pod",
                job_id,
                model,
                mode,
                chosen_template_uuid,
            )
            created_pod_uuid = await self.pod_manager.create_machine_based_pod(model, mode)
            if created_pod_uuid:
                reserved = await self.pod_manager.reserve_slot_on_pod(created_pod_uuid, job_id)
                if reserved:
                    worker_id = _assign_to_worker_on_pod(
                        pod_uuid=created_pod_uuid,
                        decision_label="created_new_pod",
                        details={
                            "machine_id": None,
                            "queue_size": 0,
                            "remaining_seconds": None,
                            "template_uuid": chosen_template_uuid,
                        },
                    )
                    if worker_id:
                        return worker_id
                    await self.pod_manager.release_slot_on_pod(created_pod_uuid, job_id)
                logger.warning(
                    "fixed_fleet_admission decision=created_new_pod_but_unassignable job=%s model=%s pod=%s template_uuid=%s",
                    job_id,
                    model,
                    created_pod_uuid,
                    chosen_template_uuid,
                )
        except Exception as e:
            self._increment_metric("admission_pending_wait_error")
            logger.warning(
                "fixed_fleet_admission decision=evaluation_or_create_error job=%s model=%s err=%s",
                job_id,
                model,
                e,
            )

        logger.info(
            "fixed_fleet_admission decision=fallback_legacy_after_create_fail job=%s model=%s",
            job_id,
            model,
        )
        self._increment_metric("admission_fallback_legacy")
        return await self.pod_manager.dispatch_job(job_id, model, mode)


@dataclass
class SchedulerRouter:
    """
    Runtime router for active scheduler mode with optional shadow logging.
    """

    active: SchedulerStrategy
    legacy: SchedulerStrategy
    shadow_mode: bool = False

    @property
    def mode(self) -> str:
        return self.active.mode

    async def _log_shadow_decision(self, job_id: str, model: str, mode: Optional[str]) -> None:
        try:
            decision = await self.active.simulate_assignment(job_id, model, mode)
            logger.info(
                "scheduler_shadow_decision mode=%s job=%s model=%s decision=%s reason=%s selected_pod=%s snapshot=%s",
                decision.get("mode"),
                job_id,
                model,
                decision.get("decision"),
                decision.get("reason"),
                (decision.get("selected") or {}).get("pod_uuid"),
                decision.get("fleet_snapshot"),
            )
        except Exception as e:
            logger.warning(
                "scheduler_shadow_decision_failed mode=%s job=%s model=%s err=%s",
                self.active.mode,
                job_id,
                model,
                e,
            )

    async def assign_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        if _is_studio_model(model):
            raise RuntimeError(
                "Studio jobs are pulled by studio_engine on Vultr, not dispatched to QuickPod"
            )
        if self.shadow_mode and self.active.mode != MODE_LEGACY_DYNAMIC_FLEET:
            await self._log_shadow_decision(job_id, model, mode)
        return await self.active.assign_job(job_id, model, mode)


def build_scheduler_strategy(pod_manager, scheduler_mode: str, shadow_mode: bool) -> SchedulerRouter:
    """Factory for scheduler router."""
    legacy = LegacyDynamicFleetScheduler(pod_manager=pod_manager)
    if scheduler_mode == MODE_FIXED_FLEET_WINDOWED:
        active: SchedulerStrategy = FixedFleetWindowedScheduler(pod_manager=pod_manager)
    else:
        active = legacy

    logger.info(
        "Initialized scheduler strategy mode=%s shadow_mode=%s",
        active.mode,
        shadow_mode,
    )
    return SchedulerRouter(active=active, legacy=legacy, shadow_mode=shadow_mode)

