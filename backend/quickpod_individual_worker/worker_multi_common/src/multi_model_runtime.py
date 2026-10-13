import asyncio
import concurrent.futures
import gc
import hashlib
import hmac
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:  # pragma: no cover
    hf_hub_download = None
    snapshot_download = None


logger = logging.getLogger("multi_model_worker")


def cleanup_gpu(verbose: bool = False):
    try:
        if verbose and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                "[GPU] Before cleanup: allocated=%.2fMB reserved=%.2fMB",
                allocated,
                reserved,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.empty_cache()
        if verbose and torch.cuda.is_available():
            logger.info(
                "[GPU] After cleanup: allocated=%.2fMB",
                torch.cuda.memory_allocated() / 1024**2,
            )
    except Exception as e:
        logger.warning("[GPU] cleanup warning: %s", e)


class RunRequest(BaseModel):
    job_id: str
    prompt: Optional[str] = ""
    model: str
    mode: Optional[str] = None
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str


class BulkRunRequest(BaseModel):
    jobs: List[RunRequest]


class DownloadPriorityRequest(BaseModel):
    model: str


class MultiModelWorker:
    """
    Coordinator-only runtime:
    - app.py orchestrates queue/controller/health
    - inference remains in separate modules (router map callables)
    """

    def __init__(
        self,
        model_registry: Dict[str, Any],
        supported_models: List[str],
        model_eta_seconds: Dict[str, int],
        download_plan: List[Dict[str, Any]],
        preprocess_hook: Optional[
            Callable[[str, Optional[str], Dict[str, Any], str], Dict[str, Any]]
        ] = None,
    ):
        self.model_registry = model_registry
        self.supported_models = [m.strip().lower() for m in supported_models if m]
        self.model_eta_seconds = {
            k.strip().lower(): int(v) for k, v in (model_eta_seconds or {}).items()
        }
        self.download_plan = download_plan or []
        self.download_models = {
            str(spec.get("model", "")).strip().lower()
            for spec in self.download_plan
            if isinstance(spec, dict) and str(spec.get("model", "")).strip()
        }
        self.preprocess_hook = preprocess_hook

        self.container_id = os.getenv("CONTAINER_LABEL")
        self.pod_uuid = self.container_id
        self.public_ip = os.getenv("PUBLIC_IPADDR")
        self.worker_id = self.pod_uuid
        self.controller_url = os.getenv("CONTROLLER_URL", "").strip()
        self.controller_shared_secret = os.getenv("CONTROLLER_SHARED_SECRET", "")
        self.controller_api_key = os.getenv("CONTROLLER_API_KEY", "")

        # Crypto mode banner — must match controller. Log once at startup so misconfig is
        # visible before the first register/heartbeat hits the wire.
        try:
            from worker_transport_crypto import (  # noqa: PLC0415
                crypto_enabled as _wt_crypto_enabled,
                transport_crypto_explicitly_on as _wt_explicit_on,
                get_fernet as _wt_get_fernet,
            )
            _wt_env = os.getenv("WORKER_TRANSPORT_CRYPTO", "").strip()
            _wt_has_key = _wt_get_fernet() is not None
            logger.info(
                "WORKER_TRANSPORT_CRYPTO=%r explicit_on=%s fernet_key_available=%s payload_encryption=%s "
                "(MUST match controller; set WORKER_TRANSPORT_CRYPTO=1 on both sides)",
                _wt_env,
                _wt_explicit_on(),
                _wt_has_key,
                _wt_crypto_enabled(),
            )
        except Exception as _crypto_log_err:
            logger.warning("Could not log worker transport crypto mode: %s", _crypto_log_err)

        internal_port = 8000
        external_port = os.getenv(f"QUICKPOD_PORT_{internal_port}")
        self.port = int(external_port) if external_port else internal_port

        self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
        self.workspace_dir = os.getenv("WORKSPACE_DIR", "/workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)

        self.job_queue_maxsize = int(os.getenv("JOB_QUEUE_MAX", "30"))
        self.download_max_parallel = min(
            32,
            max(1, int(os.getenv("DOWNLOAD_MAX_PARALLEL", "8"))),
        )
        dl_threads = min(
            64,
            max(self.download_max_parallel + 4, self.download_max_parallel * 2),
        )

        self.inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference"
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="io"
        )
        self.download_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=dl_threads, thread_name_prefix="download"
        )
        self.prewarm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prewarm"
        )

        self._pending_jobs: List[Dict[str, Any]] = []
        self._schedule_cond = asyncio.Condition()
        self.approx_queue_size = 0
        self.job_status: Dict[str, Dict[str, Any]] = {}
        self.job_status_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(self.max_concurrent_jobs)

        self.pod_start_time = time.time()
        self.first_inference_start_time: Optional[float] = None
        self.first_inference_lock = asyncio.Lock()

        self.last_model_key: Optional[str] = None
        self.prewarmed_models: set[str] = set()
        self.single_active_fallback = False
        self.prewarm_enabled = os.getenv("MODEL_PREWARM_ENABLED", "true").lower() == "true"
        self.prewarm_min_free_gb = float(os.getenv("MODEL_PREWARM_MIN_FREE_GB", "6.0"))
        self.startup_completed = False
        self.download_state: Dict[str, Dict[str, Any]] = {
            m: {"status": "pending"} for m in self.download_models
        }
        self.downloader_started = False
        self._download_tasks: Dict[str, asyncio.Task] = {}
        self._download_plan_by_model: Dict[str, Dict[str, Any]] = {}
        self._download_queue_order: List[str] = []
        for spec in self.download_plan:
            model_key = str(spec.get("model", "")).strip().lower()
            if not model_key:
                continue
            self._download_plan_by_model[model_key] = spec
            self._download_queue_order.append(model_key)

        # Sort initial queue by download_priority (lower = first).
        # Baked-model specs that aren't in the queue are excluded already.
        self._download_queue_order.sort(
            key=lambda mk: int(
                (self._download_plan_by_model.get(mk) or {}).get("download_priority") or 100
            )
        )

        self._download_state_lock = asyncio.Lock()
        self._download_plan_lock = asyncio.Lock()
        self._prewarm_lock = asyncio.Lock()

        self._latest_queue_snapshot: Dict[str, int] = {}
        self._model_inference_seconds: Dict[str, float] = {}
        self._adaptive_parallel_cap = self.download_max_parallel
        self._adaptive_download_parallel = (
            os.getenv("ADAPTIVE_DOWNLOAD_PARALLEL", "true").lower() == "true"
        )
        self._model_categories: Dict[str, List[str]] = {"image": [], "video": []}
        try:
            from omni_config import get_model_categories  # noqa: PLC0415

            self._model_categories = get_model_categories()
        except Exception:
            pass

    def _model_in_download_plan(self, model: str) -> bool:
        return (model or "").strip().lower() in self.download_models

    def _model_weights_ready(self, model: str) -> bool:
        mk = (model or "").strip().lower()
        if not self._model_in_download_plan(mk):
            return True
        state = self.download_state.get(mk) or {}
        st = str(state.get("status", "pending"))
        if st.startswith("ready"):
            return True
        if st.startswith("skipped:"):
            return True
        return False

    def _model_download_failed(self, model: str) -> bool:
        mk = (model or "").strip().lower()
        state = self.download_state.get(mk) or {}
        return str(state.get("status", "")) == "failed"

    def _estimated_inference_seconds(self, model: str) -> float:
        """Server EWMA when present; else heuristic from manifest ``model_categories``."""
        m = (model or "").strip().lower()
        if m in self._model_inference_seconds:
            return float(self._model_inference_seconds[m])
        video = set(self._model_categories.get("video") or [])
        image = set(self._model_categories.get("image") or [])
        if m in video:
            return 150.0
        if m in image:
            return 35.0
        return 45.0

    def _compute_adaptive_parallelism(self) -> int:
        """
        Parallel HF downloads tuned from local pending jobs and register-time EWMAs.
        Heartbeat ``queue_snapshot`` is stored for future global tuning; v1 is local-only.
        """
        cap = self._adaptive_parallel_cap

        models_remaining = sum(
            1
            for mk in self.download_models
            if not self._model_weights_ready(mk) and not self._model_download_failed(mk)
        )
        if models_remaining == 0:
            return 1

        if self._pending_jobs:
            first_model = str(self._pending_jobs[0].get("model", "")).strip().lower()
            if not self._model_weights_ready(first_model) and first_model in self.download_models:
                return 1

        if not self._pending_jobs:
            return cap

        runway_seconds = 0.0
        for job in self._pending_jobs:
            m = str(job.get("model", "")).strip().lower()
            if self._model_weights_ready(m):
                runway_seconds += self._estimated_inference_seconds(m)

        if runway_seconds > 120:
            return cap
        if runway_seconds > 60:
            return max(1, cap * 3 // 4)
        if runway_seconds > 30:
            return max(1, cap // 2)
        return max(1, cap // 4)

    async def _enqueue_job(self, item: Dict[str, Any]) -> int:
        async with self._schedule_cond:
            if len(self._pending_jobs) >= self.job_queue_maxsize:
                raise RuntimeError("queue full")
            self._pending_jobs.append(item)
            self.approx_queue_size = len(self._pending_jobs)
            self._schedule_cond.notify_all()
            return len(self._pending_jobs)

    async def _notify_schedule(self) -> None:
        async with self._schedule_cond:
            self._schedule_cond.notify_all()

    def _peek_next_job_id(self) -> Optional[str]:
        try:
            if self._pending_jobs:
                return (self._pending_jobs[0] or {}).get("job_id")
        except Exception:
            return None
        return None

    def _peek_next_job_model(self) -> Optional[str]:
        try:
            if self._pending_jobs:
                return (self._pending_jobs[0] or {}).get("model")
        except Exception:
            return None
        return None

    async def _peek_next_runnable_job_id(self) -> Optional[str]:
        async with self._schedule_cond:
            for it in self._pending_jobs:
                m = str(it.get("model", "")).strip().lower()
                if self._model_weights_ready(m):
                    return (it or {}).get("job_id")
        return None

    async def _peek_next_runnable_job_model(self) -> Optional[str]:
        async with self._schedule_cond:
            for it in self._pending_jobs:
                m = str(it.get("model", "")).strip().lower()
                if self._model_weights_ready(m):
                    return m
        return None

    async def _fail_job_weights(self, item: Dict[str, Any], error: str) -> None:
        job_id = str(item.get("job_id", ""))
        result_key = item.get("result_key")
        async with self.job_status_lock:
            self.job_status[job_id] = {
                "status": "failed",
                "error": error,
                "finished_at": datetime.utcnow().isoformat(),
            }
        if self.controller_url and job_id and not job_id.startswith("autotest-"):
            fail_payload = {
                "job_id": job_id,
                "result_r2_key": result_key,
                "failed": True,
                "error": error,
                "next_job_id": await self._peek_next_runnable_job_id(),
            }
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    self.io_executor,
                    self._post_signed_json,
                    f"{self.controller_url.rstrip('/')}/job_complete",
                    fail_payload,
                    10,
                    5,
                )
            except Exception:
                pass

    async def _drain_failed_weight_jobs(self) -> None:
        doomed: List[Dict[str, Any]] = []
        async with self._schedule_cond:
            i = 0
            while i < len(self._pending_jobs):
                it = self._pending_jobs[i]
                m = str(it.get("model", "")).strip().lower()
                if self._model_in_download_plan(m) and self._model_download_failed(m):
                    doomed.append(self._pending_jobs.pop(i))
                    self.approx_queue_size = len(self._pending_jobs)
                    continue
                i += 1
        for it in doomed:
            await self._fail_job_weights(
                it, "Model weights download failed permanently for this pod"
            )

    async def _pop_next_ready_job(self) -> Optional[Dict[str, Any]]:
        async with self._schedule_cond:
            for i, it in enumerate(self._pending_jobs):
                m = str(it.get("model", "")).strip().lower()
                if self._model_weights_ready(m):
                    job = self._pending_jobs.pop(i)
                    self.approx_queue_size = len(self._pending_jobs)
                    return job
        return None

    async def _wait_for_schedule_wakeup(self) -> None:
        boost: List[str] = []
        async with self._schedule_cond:
            for it in self._pending_jobs:
                m = str(it.get("model", "")).strip().lower()
                if (
                    self._model_in_download_plan(m)
                    and not self._model_weights_ready(m)
                    and not self._model_download_failed(m)
                ):
                    boost.append(m)
        for m in dict.fromkeys(boost):
            await self._reprioritize_download_model(m)
        async with self._schedule_cond:
            if not self._pending_jobs:
                await self._schedule_cond.wait()
                return
            try:
                await asyncio.wait_for(self._schedule_cond.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    def _resolve_inference_callable(self, model: str, mode: Optional[str]):
        if model not in self.model_registry:
            raise ValueError(f"Unknown model: {model}")
        entry = self.model_registry[model]
        if callable(entry):
            return entry
        if isinstance(entry, dict):
            callables = entry.get("modes")
            if isinstance(callables, dict):
                if not mode:
                    raise ValueError(f"mode is required for model '{model}'")
                if mode not in callables:
                    raise ValueError(f"Unsupported mode '{mode}' for model '{model}'")
                return callables[mode]
            func = entry.get("callable")
            if callable(func):
                return func
        raise ValueError(f"Invalid registry entry for model '{model}'")

    def _resolve_model_entry(self, model: str) -> Dict[str, Any]:
        entry = self.model_registry.get(model)
        if callable(entry):
            return {"callable": entry}
        if isinstance(entry, dict):
            return entry
        return {}

    def _advertised_supported_models(self) -> List[str]:
        supported = set(self.supported_models)
        for m in list(supported):
            state = self.download_state.get(m)
            if isinstance(state, dict) and state.get("status") in {
                "pending",
                "downloading",
                "failed",
            }:
                supported.discard(m)
        return sorted(list(supported))

    def _has_prewarm_headroom(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            return free_gb >= self.prewarm_min_free_gb
        except Exception:
            return False

    async def _reprioritize_download_model(self, model: str) -> Dict[str, Any]:
        model_key = (model or "").strip().lower()
        if not model_key:
            return {"status": "ignored", "reason": "empty_model"}
        if model_key not in self.download_models:
            return {
                "status": "ignored",
                "reason": "model_not_in_download_plan",
                "model": model_key,
            }

        async with self._download_state_lock:
            state = self.download_state.get(model_key, {})
            state_status = str(state.get("status", "pending"))
            if state_status.startswith("ready"):
                return {
                    "status": "ignored",
                    "reason": "already_ready",
                    "model": model_key,
                }
            if state_status == "downloading":
                return {
                    "status": "accepted",
                    "reason": "already_downloading",
                    "model": model_key,
                }
            if state_status == "failed":
                self.download_state[model_key] = {"status": "pending", "retry": True}

        async with self._download_plan_lock:
            if model_key not in self._download_queue_order:
                if model_key in self.download_models:
                    self._download_queue_order.insert(0, model_key)
                    return {
                        "status": "accepted",
                        "reason": "re_enqueued_after_failure",
                        "model": model_key,
                        "queue_order": list(self._download_queue_order),
                    }
                return {
                    "status": "ignored",
                    "reason": "not_pending_in_queue",
                    "model": model_key,
                }
            self._download_queue_order = [
                model_key,
                *[x for x in self._download_queue_order if x != model_key],
            ]
            return {
                "status": "accepted",
                "reason": "reprioritized_to_front",
                "model": model_key,
                "queue_order": list(self._download_queue_order),
            }

    @staticmethod
    def _invoke_entry_hook(entry: Dict[str, Any], hook_name: str):
        hook = entry.get(hook_name) if isinstance(entry, dict) else None
        if callable(hook):
            hook()

    async def _run_inference_and_save(
        self,
        job_id: str,
        prompt: Optional[str],
        model: str,
        mode: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, float]]:
        loop = asyncio.get_event_loop()
        params = dict(params or {})
        params.setdefault("output_dir", self.workspace_dir)

        async with self.first_inference_lock:
            if self.first_inference_start_time is None:
                self.first_inference_start_time = time.time()
                startup_time = self.first_inference_start_time - self.pod_start_time
            else:
                startup_time = None

        if self.preprocess_hook:
            params = await loop.run_in_executor(
                self.io_executor, self.preprocess_hook, model, mode, params, job_id
            )

        inference_func = self._resolve_inference_callable(model, mode)
        model_key = f"{model}:{mode or ''}"
        if self.single_active_fallback:
            # Keep only active model under pressure to avoid repeated OOM churn.
            for other_model in list(self.prewarmed_models):
                if other_model == model:
                    continue
                entry = self._resolve_model_entry(other_model)
                self._invoke_entry_hook(entry, "unload")
                self.prewarmed_models.discard(other_model)

        if self.last_model_key and self.last_model_key != model_key:
            cleanup_gpu(verbose=False)

        start = time.time()
        try:
            out_path = await loop.run_in_executor(
                self.inference_executor,
                inference_func,
                job_id,
                prompt or "",
                params,
            )
            processing_time = time.time() - start
            metrics = {"processing_time_seconds": round(processing_time, 2)}
            if startup_time is not None:
                metrics["startup_time_seconds"] = round(startup_time, 2)
            return out_path, metrics
        finally:
            self.last_model_key = model_key
            self.prewarmed_models.add(model)

    async def _prewarm_model_if_possible(self, model: str):
        if not self.prewarm_enabled or self.single_active_fallback:
            return
        if model in self.prewarmed_models:
            return
        if model not in self.model_registry:
            return
        if not self._has_prewarm_headroom():
            return
        entry = self._resolve_model_entry(model)
        prewarm_fn = entry.get("prewarm")
        if not callable(prewarm_fn):
            return
        async with self._prewarm_lock:
            if model in self.prewarmed_models:
                return
            if not self._has_prewarm_headroom():
                return
            # Hybrid policy: keep active model + one likely-next prewarmed model.
            for other_model in list(self.prewarmed_models):
                if other_model == model:
                    continue
                other_entry = self._resolve_model_entry(other_model)
                self._invoke_entry_hook(other_entry, "unload")
                self.prewarmed_models.discard(other_model)
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(self.prewarm_executor, prewarm_fn)
                self.prewarmed_models.add(model)
                logger.info("prewarm complete for model=%s", model)
            except Exception as e:
                msg = str(e).lower()
                logger.warning("prewarm failed for model=%s error=%s", model, e)
                if "out of memory" in msg or "oom" in msg:
                    self.single_active_fallback = True
                    cleanup_gpu(verbose=True)
                    logger.warning(
                        "prewarm disabled due to OOM; staying single-active until restart"
                    )

    @staticmethod
    def _sign_payload_bytes(secret: str, payload_bytes: bytes) -> str:
        mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
        return "sha256=" + mac.hexdigest()

    def _signed_headers_for_wire(self, body_bytes: bytes) -> Dict[str, str]:
        sec = (self.controller_shared_secret or "").strip()
        if sec:
            sig = self._sign_payload_bytes(sec, body_bytes)
        else:
            sig = "sha256="
        headers = {"X-Signature": sig, "Content-Type": "application/json"}
        if self.controller_api_key:
            headers["Authorization"] = f"Bearer {self.controller_api_key}"
        return headers

    def _post_signed_json(
        self, url: str, payload: dict, timeout: int = 10, max_retries: int = 3
    ) -> requests.Response:
        from worker_transport_crypto import encode_worker_request_body

        body_bytes = encode_worker_request_body(payload)
        headers = self._signed_headers_for_wire(body_bytes)
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url, data=body_bytes, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise last_error
        raise last_error  # pragma: no cover

    @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
    def register_with_controller(self):
        if not self.controller_url:
            return {"skipped": True}
        payload = {
            "worker_id": self.worker_id,
            "models": list(self.model_registry.keys()),
            "supported_models": self._advertised_supported_models(),
            "model_eta_seconds": self.model_eta_seconds,
            "endpoint": f"http://{self.public_ip}:{self.port}",
            "container_id": self.pod_uuid,
            "pod_uuid": self.pod_uuid,
            "public_ip": self.public_ip,
            "info": {
                "started_at": datetime.utcnow().isoformat(),
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "queue_capacity": self.job_queue_maxsize,
                "capability_source": "worker_runtime",
            },
        }
        url = f"{self.controller_url.rstrip('/')}/register_worker"
        resp = self._post_signed_json(url, payload, timeout=15, max_retries=6).json()
        mis = resp.get("model_inference_seconds")
        if isinstance(mis, dict):
            for k, v in mis.items():
                try:
                    self._model_inference_seconds[str(k).strip().lower()] = float(v)
                except (TypeError, ValueError):
                    continue
        return resp

    @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def send_heartbeat(self):
        if not self.controller_url:
            return {"skipped": True}
        queue_size = self.approx_queue_size
        active_jobs = sum(
            1 for v in self.job_status.values() if v.get("status") == "running"
        )
        status = "busy" if (active_jobs > 0 or queue_size > 0) else "ready"
        if not self.startup_completed:
            status = "starting"
        payload = {
            "worker_id": self.worker_id,
            "pod_uuid": self.pod_uuid,
            "timestamp": int(time.time()),
            "status": status,
            "supported_models": self._advertised_supported_models(),
            "model_eta_seconds": self.model_eta_seconds,
            "metrics": {
                "queue_size": queue_size,
                "active_jobs": active_jobs,
                "max_concurrent": self.max_concurrent_jobs,
                "download_state": self.download_state,
                "prewarmed_models": sorted(list(self.prewarmed_models)),
                "single_active_fallback": self.single_active_fallback,
            },
        }
        url = f"{self.controller_url.rstrip('/')}/heartbeat"
        resp = self._post_signed_json(url, payload, timeout=8, max_retries=3).json()
        qs = resp.get("queue_snapshot")
        if isinstance(qs, dict):
            snap: Dict[str, int] = {}
            for k, v in qs.items():
                try:
                    snap[str(k).strip().lower()] = int(v)
                except (TypeError, ValueError):
                    continue
            self._latest_queue_snapshot = snap
        return resp

    async def _post_job_started(self, job_id: str):
        if not self.controller_url:
            return
        payload = {"job_id": job_id, "worker_id": self.worker_id}
        url = f"{self.controller_url.rstrip('/')}/job_started"
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self.io_executor, self._post_signed_json, url, payload, 8, 3
            )
        except Exception as e:
            logger.debug("[%s] job_started notify failed: %s", job_id, e)

    @staticmethod
    def _copy_reuse_components(target: str, reuse_spec: Dict[str, Any]):
        source_dir = str(reuse_spec.get("source_dir") or "").strip()
        components = reuse_spec.get("components") or []
        if not source_dir or not os.path.isdir(source_dir):
            return
        for component in components:
            name = str(component).strip()
            if not name:
                continue
            src = os.path.join(source_dir, name)
            dst = os.path.join(target, name)
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    @staticmethod
    def _check_required_paths(target: str, required_paths: List[str]) -> bool:
        for rel_path in required_paths or []:
            check_path = os.path.join(target, rel_path)
            if not os.path.exists(check_path):
                return False
        return True

    @staticmethod
    def _run_snapshot(
        repo_id: str,
        target: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ):
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
        )

    def _supplement_hub_files(self, spec: Dict[str, Any]) -> None:
        items = spec.get("supplement_hub_files") or []
        if not items:
            return
        base_dest = str(spec.get("supplement_dest_dir") or spec.get("local_dir") or "").strip()
        if not base_dest:
            return
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub hf_hub_download unavailable")
        os.makedirs(base_dest, exist_ok=True)
        for it in items:
            rid = str((it or {}).get("repo_id") or "").strip()
            fn = str((it or {}).get("filename") or "").strip()
            if not rid or not fn:
                continue
            target_relpath = str((it or {}).get("target_relpath") or "").strip().replace(
                "\\", "/"
            )

            if target_relpath:
                work = os.path.join(base_dest, ".hf_supplement_work")
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work, exist_ok=True)
                try:
                    hf_hub_download(
                        repo_id=rid,
                        filename=fn,
                        local_dir=work,
                        resume_download=True,
                    )
                    downloaded = os.path.join(work, fn)
                    if not os.path.isfile(downloaded):
                        downloaded = ""
                        for root, _, files in os.walk(work):
                            if fn in files:
                                downloaded = os.path.join(root, fn)
                                break
                    if not downloaded or not os.path.isfile(downloaded):
                        logger.warning(
                            "supplement file missing after download repo=%s filename=%s",
                            rid,
                            fn,
                        )
                        continue
                    final = os.path.join(
                        base_dest, target_relpath.replace("/", os.sep)
                    )
                    tl = target_relpath.lower()
                    if tl.startswith("transformer/"):
                        tdir = os.path.join(base_dest, "transformer")
                        if os.path.isdir(tdir):
                            for c in os.listdir(tdir):
                                if c.endswith(".safetensors") or c.endswith(
                                    ".index.json"
                                ):
                                    try:
                                        os.remove(os.path.join(tdir, c))
                                    except OSError:
                                        pass
                    os.makedirs(os.path.dirname(final), exist_ok=True)
                    if os.path.exists(final):
                        try:
                            os.remove(final)
                        except OSError:
                            pass
                    shutil.move(downloaded, final)
                finally:
                    shutil.rmtree(work, ignore_errors=True)
            else:
                hf_hub_download(
                    repo_id=rid,
                    filename=fn,
                    local_dir=base_dest,
                    resume_download=True,
                )

    def _hot_register(
        self,
        model_key: str,
        registry_entry: Any,
    ) -> None:
        """Add a newly-downloaded model to the live registry so requests can be served."""
        mk = (model_key or "").strip().lower()
        if not mk or registry_entry is None:
            return
        self.model_registry[mk] = registry_entry
        if mk not in [m.strip().lower() for m in self.supported_models]:
            self.supported_models.append(mk)
        logger.info("hot_register: model=%s added to registry", mk)
        # Also add to download_models so _model_weights_ready works correctly for it
        self.download_models.add(mk)

    def _download_task(self, spec: Dict[str, Any]) -> Tuple[str, str, Optional[Any]]:
        target = str(spec.get("local_dir") or "").strip()
        if target:
            os.makedirs(target, exist_ok=True)
        repo_id = spec.get("repo_id")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        if spec.get("multi_snapshot"):
            if snapshot_download is None:
                return target, "failed:huggingface_hub_missing", None
            for snap in spec.get("snapshots") or []:
                tgt = str((snap or {}).get("target_dir") or "").strip()
                srepo = str((snap or {}).get("repo_id") or "").strip()
                if not tgt or not srepo:
                    continue
                os.makedirs(tgt, exist_ok=True)
                self._run_snapshot(
                    srepo,
                    tgt,
                    revision=(snap or {}).get("revision"),
                    allow_patterns=(snap or {}).get("allow_patterns"),
                    ignore_patterns=(snap or {}).get("ignore_patterns"),
                )
            # Apply optional supplement files (e.g. Qwen LoRAs) for multi-snapshot specs too.
            self._supplement_hub_files(spec)
            logger.info(
                "download ready model=%s mode=multi_snapshot",
                spec.get("model", ""),
            )
            registry_entry = self._try_install_inference_archive(spec)
            return target, "ready:multi_snapshot", registry_entry

        if not repo_id:
            # Code-only plugin (no HF weights): install archive and register.
            registry_entry = self._try_install_inference_archive(spec)
            return target, "ready:code_only", registry_entry

        if snapshot_download is None:
            return target, "failed:huggingface_hub_missing", None

        partial_profiles = spec.get("partial_profiles") or []
        for profile in partial_profiles:
            profile_name = str(profile.get("name") or "partial_transformer")
            try:
                self._copy_reuse_components(
                    target, profile.get("reuse_components") or spec.get("reuse_components") or {}
                )
                self._run_snapshot(
                    repo_id=repo_id,
                    target=target,
                    revision=spec.get("revision"),
                    allow_patterns=profile.get("allow_patterns"),
                    ignore_patterns=profile.get("ignore_patterns"),
                )
                required_paths = profile.get("required_paths") or []
                if not self._check_required_paths(target, required_paths):
                    raise RuntimeError("required_paths_missing_after_partial_download")
                self._supplement_hub_files(spec)
                logger.info(
                    "download ready model=%s mode=%s",
                    spec.get("model", repo_id),
                    profile_name,
                )
                registry_entry = self._try_install_inference_archive(spec)
                return target, f"ready:{profile_name}", registry_entry
            except Exception as e:
                logger.warning(
                    "partial download failed model=%s mode=%s error=%s",
                    spec.get("model", repo_id),
                    profile_name,
                    e,
                )

        self._run_snapshot(
            repo_id=repo_id,
            target=target,
            revision=spec.get("revision"),
            allow_patterns=spec.get("allow_patterns"),
            ignore_patterns=spec.get("ignore_patterns"),
        )
        self._supplement_hub_files(spec)
        logger.info(
            "download ready model=%s mode=full_snapshot",
            spec.get("model", repo_id),
        )
        registry_entry = self._try_install_inference_archive(spec)
        return target, "ready:full_snapshot", registry_entry

    @staticmethod
    def _try_install_inference_archive(spec: Dict[str, Any]) -> Optional[Any]:
        """
        If spec has an ``inference_archive`` dict, download+install the plugin .py and
        return a ready registry entry dict.  Returns None if no archive or on error.
        Runs in the download executor thread — no async.
        """
        archive_spec = spec.get("inference_archive")
        if not archive_spec or not isinstance(archive_spec, dict):
            return None
        model = str(spec.get("model") or "").strip().lower()
        try:
            from omni_runtime_plugins import (  # noqa: PLC0415
                install_inference_plugin,
                registry_entry_from_plugin_spec,
            )
            plugins_root = os.getenv("OMNI_RUNTIME_PLUGINS_DIR", "/app/runtime_plugins")
            timeout = float(os.getenv("OMNI_RUNTIME_MANIFEST_TIMEOUT", "120"))
            py_path = install_inference_plugin(archive_spec, model, plugins_root, timeout)
            if py_path:
                return registry_entry_from_plugin_spec(archive_spec, py_path)
        except Exception as e:
            logger.warning("inference_archive install failed model=%s err=%s", model, e)
        return None

    async def _download_one_model(self, model_key: str, spec: Dict[str, Any]) -> None:
        max_retries = int(os.getenv("DOWNLOAD_MAX_RETRIES", "3"))
        retry_backoff_base = 15
        attempt = 0
        success = False
        last_err: Optional[Exception] = None
        while attempt < max_retries and not success:
            attempt += 1
            async with self._download_state_lock:
                self.download_state[model_key] = {
                    "status": "downloading",
                    "started_at": datetime.utcnow().isoformat(),
                    "attempt": attempt,
                }
            loop = asyncio.get_running_loop()
            try:
                _, status, registry_entry = await loop.run_in_executor(
                    self.download_executor, self._download_task, spec
                )
                async with self._download_state_lock:
                    self.download_state[model_key]["status"] = status
                    self.download_state[model_key][
                        "completed_at"
                    ] = datetime.utcnow().isoformat()
                if registry_entry is not None:
                    self._hot_register(model_key, registry_entry)
                success = True
            except Exception as e:
                last_err = e
                logger.warning(
                    "download failed model=%s attempt=%d/%d error=%s",
                    model_key,
                    attempt,
                    max_retries,
                    e,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_backoff_base * attempt)
                else:
                    async with self._download_state_lock:
                        self.download_state[model_key]["status"] = "failed"
                        self.download_state[model_key]["error"] = str(e)
                        self.download_state[model_key][
                            "completed_at"
                        ] = datetime.utcnow().isoformat()
        if not success and last_err:
            logger.error(
                "download gave up model=%s after %d attempts: %s",
                model_key,
                max_retries,
                last_err,
            )
        async with self._schedule_cond:
            self._schedule_cond.notify_all()

    async def _downloader_loop(self):
        if self.downloader_started:
            return
        self.downloader_started = True
        while True:
            max_p = (
                self._compute_adaptive_parallelism()
                if self._adaptive_download_parallel
                else self.download_max_parallel
            )
            async with self._download_plan_lock:
                rotated = 0
                q_len = len(self._download_queue_order)
                while (
                    len(self._download_tasks) < max_p
                    and self._download_queue_order
                    and rotated < q_len
                ):
                    mk = self._download_queue_order[0]
                    if self._model_weights_ready(mk):
                        self._download_queue_order.pop(0)
                        asyncio.create_task(self._notify_schedule())
                        rotated = 0
                        q_len = len(self._download_queue_order)
                        continue
                    if mk in self._download_tasks:
                        self._download_queue_order.append(
                            self._download_queue_order.pop(0)
                        )
                        rotated += 1
                        continue
                    self._download_queue_order.pop(0)
                    spec = self._download_plan_by_model.get(mk)
                    if not isinstance(spec, dict):
                        continue
                    self._download_tasks[mk] = asyncio.create_task(
                        self._download_one_model(mk, spec)
                    )
                    rotated = 0
                    q_len = len(self._download_queue_order)
            if not self._download_tasks and not self._download_queue_order:
                break
            if self._download_tasks:
                done, _ = await asyncio.wait(
                    set(self._download_tasks.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0,
                )
                for t in done:
                    rm: Optional[str] = None
                    for k, v in list(self._download_tasks.items()):
                        if v is t:
                            rm = k
                            break
                    if rm is not None:
                        del self._download_tasks[rm]
                    try:
                        t.result()
                    except Exception as e:
                        logger.debug("download task finished with error: %s", e)
                    asyncio.create_task(self._notify_schedule())
            else:
                await asyncio.sleep(0.05)

    async def _upload_with_retry(
        self, file_path: str, upload_url: str, job_id: str, max_retries: int = 3
    ) -> bool:
        loop = asyncio.get_event_loop()

        def _upload():
            with open(file_path, "rb") as f:
                resp = requests.put(upload_url, data=f, timeout=60)
                resp.raise_for_status()

        for attempt in range(max_retries):
            try:
                await loop.run_in_executor(self.io_executor, _upload)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("[%s] upload retry %s: %s", job_id, attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("[%s] upload failed: %s", job_id, e)
                    return False
        return False

    async def process_job_item(self, item: Dict[str, Any]):
        job_id = item["job_id"]
        model = item["model"]
        mode = item.get("mode")
        prompt = item.get("prompt") or ""
        params = item.get("params") or {}
        upload_url = item["upload_url"]
        result_key = item["result_key"]
        start = datetime.utcnow()

        async with self.job_status_lock:
            self.job_status[job_id] = {
                "status": "running",
                "started_at": start.isoformat(),
                "model": model,
                "mode": mode,
            }
        try:
            async with self.semaphore:
                await self._post_job_started(job_id)
                out_local, metrics = await self._run_inference_and_save(
                    job_id, prompt, model, mode, params
                )
                if not upload_url.startswith("http://dummy-upload"):
                    ok = await self._upload_with_retry(out_local, upload_url, job_id)
                    if not ok:
                        raise RuntimeError("result upload failed")
                # Clean up local output file after successful upload
                try:
                    if out_local and os.path.isfile(out_local):
                        os.remove(out_local)
                except Exception as _cleanup_err:
                    logger.debug("[%s] output cleanup failed: %s", job_id, _cleanup_err)
                if self.controller_url and not job_id.startswith("autotest-"):
                    payload = {
                        "job_id": job_id,
                        "result_r2_key": result_key,
                        "next_job_id": await self._peek_next_runnable_job_id(),
                        "metrics": metrics,
                    }
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self.io_executor,
                        self._post_signed_json,
                        f"{self.controller_url.rstrip('/')}/job_complete",
                        payload,
                        10,
                        5,
                    )
                done = datetime.utcnow()
                async with self.job_status_lock:
                    self.job_status[job_id] = {
                        "status": "completed",
                        "result_r2_key": result_key,
                        "finished_at": done.isoformat(),
                        "processing_time": (done - start).total_seconds(),
                    }
                next_model = await self._peek_next_runnable_job_model()
                if isinstance(next_model, str):
                    asyncio.create_task(self._prewarm_model_if_possible(next_model))
        except Exception as e:
            logger.exception("[%s] job failed: %s", job_id, e)
            if self.controller_url and not job_id.startswith("autotest-"):
                fail_payload = {
                    "job_id": job_id,
                    "result_r2_key": None,
                    "failed": True,
                    "error": str(e),
                    "next_job_id": await self._peek_next_runnable_job_id(),
                }
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        self.io_executor,
                        self._post_signed_json,
                        f"{self.controller_url.rstrip('/')}/job_complete",
                        fail_payload,
                        10,
                        5,
                    )
                except Exception:
                    pass
            async with self.job_status_lock:
                self.job_status[job_id] = {
                    "status": "failed",
                    "error": str(e),
                    "finished_at": datetime.utcnow().isoformat(),
                }

    async def worker_loop(self):
        while True:
            await self._drain_failed_weight_jobs()
            job = await self._pop_next_ready_job()
            if job is None:
                await self._wait_for_schedule_wakeup()
                continue
            await self.process_job_item(job)

    async def heartbeat_loop(self):
        while True:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.io_executor, self.send_heartbeat)
            except Exception as e:
                logger.debug("heartbeat failed: %s", e)
            await asyncio.sleep(self.heartbeat_interval)

    async def record_job_status_later(self, job_id: str, model: str):
        try:
            async with self.job_status_lock:
                self.job_status[job_id] = {
                    "status": "queued",
                    "queued_at": datetime.utcnow().isoformat(),
                    "model": model,
                }
        except Exception:
            pass

    def create_fastapi_app(self, title: str) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            asyncio.create_task(self.worker_loop())
            asyncio.create_task(self._downloader_loop())
            try:
                self.register_with_controller()
            except Exception as e:
                logger.warning("register failed on startup: %s", e)
            self.startup_completed = True
            asyncio.create_task(self.heartbeat_loop())
            yield
            self.startup_completed = False
            self.inference_executor.shutdown(wait=False, cancel_futures=False)
            self.io_executor.shutdown(wait=False, cancel_futures=False)
            self.download_executor.shutdown(wait=False, cancel_futures=False)
            self.prewarm_executor.shutdown(wait=False, cancel_futures=False)
            cleanup_gpu(verbose=False)

        app = FastAPI(title=title, lifespan=lifespan)

        @app.post("/run")
        async def run_endpoint(request: Request):
            from worker_transport_crypto import (
                decode_worker_request_body,
                verify_controller_signature,
            )

            body = await request.body()
            if not verify_controller_signature(
                request.headers.get("X-Signature"), body
            ):
                raise HTTPException(
                    status_code=403, detail="Invalid or missing controller signature"
                )
            try:
                data = decode_worker_request_body(body)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            req = RunRequest.model_validate(data)
            if req.model not in self.model_registry:
                raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
            async with self.job_status_lock:
                if req.job_id in self.job_status and self.job_status[req.job_id].get(
                    "status"
                ) in ("queued", "running"):
                    raise HTTPException(status_code=409, detail="Job already queued/running")
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            try:
                qp = await self._enqueue_job(payload)
            except RuntimeError:
                raise HTTPException(status_code=503, detail="Worker queue full") from None
            asyncio.create_task(self.record_job_status_later(req.job_id, req.model))
            return {
                "status": "accepted",
                "job_id": req.job_id,
                "queue_position": qp,
                "worker_id": self.worker_id,
            }

        @app.post("/download_priority")
        async def download_priority_endpoint(request: Request):
            from worker_transport_crypto import (
                decode_worker_request_body,
                verify_controller_signature,
            )

            body = await request.body()
            if not verify_controller_signature(
                request.headers.get("X-Signature"), body
            ):
                raise HTTPException(
                    status_code=403, detail="Invalid or missing controller signature"
                )
            try:
                data = decode_worker_request_body(body)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            req = DownloadPriorityRequest.model_validate(data)
            result = await self._reprioritize_download_model(req.model)
            await self._notify_schedule()
            return {
                **result,
                "worker_id": self.worker_id,
            }

        @app.post("/run_bulk")
        async def run_bulk_endpoint(request: Request):
            from worker_transport_crypto import (
                decode_worker_request_body,
                verify_controller_signature,
            )

            body = await request.body()
            if not verify_controller_signature(
                request.headers.get("X-Signature"), body
            ):
                raise HTTPException(
                    status_code=403, detail="Invalid or missing controller signature"
                )
            try:
                data = decode_worker_request_body(body)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            req = BulkRunRequest.model_validate(data)
            if not req.jobs:
                raise HTTPException(status_code=400, detail="No jobs provided")
            async with self._schedule_cond:
                cur = len(self._pending_jobs)
            available = self.job_queue_maxsize - cur
            if available < len(req.jobs):
                raise HTTPException(
                    status_code=503,
                    detail=f"Not enough queue capacity. Available: {available}",
                )
            accepted = []
            for job in req.jobs:
                if job.model not in self.model_registry:
                    continue
                payload = job.model_dump() if hasattr(job, "model_dump") else job.dict()
                try:
                    await self._enqueue_job(payload)
                except RuntimeError:
                    break
                asyncio.create_task(self.record_job_status_later(job.job_id, job.model))
                accepted.append(job.job_id)
            return {
                "status": "accepted",
                "job_ids": accepted,
                "total_accepted": len(accepted),
                "total_requested": len(req.jobs),
                "worker_id": self.worker_id,
                "queue_size": self.approx_queue_size,
            }

        @app.get("/health")
        async def health():
            async with self.job_status_lock:
                active_jobs = sum(
                    1
                    for j in self.job_status.values()
                    if j.get("status") == "running"
                )
                queued_jobs = sum(
                    1 for j in self.job_status.values() if j.get("status") == "queued"
                )
            state = "busy" if (active_jobs > 0 or queued_jobs > 0) else "ok"
            if not self.startup_completed:
                state = "starting"
            return {
                "status": state,
                "worker_id": self.worker_id,
                "queue_size": self.approx_queue_size,
                "queue_capacity": self.job_queue_maxsize,
                "active_jobs": active_jobs,
                "queued_jobs": queued_jobs,
                "supported_models": self.supported_models,
                "advertised_supported_models": self._advertised_supported_models(),
                "downloads": self.download_state,
                "startup_completed": self.startup_completed,
                "ready": self.startup_completed
                and self.approx_queue_size < self.job_queue_maxsize,
                "healthy": True,
            }

        @app.get("/status")
        async def status():
            async with self.job_status_lock:
                jobs_copy = dict(self.job_status)
            return {
                "queue_size": self.approx_queue_size,
                "queue_capacity": self.job_queue_maxsize,
                "jobs": jobs_copy,
                "worker_id": self.worker_id,
                "supported_models": self.supported_models,
                "advertised_supported_models": self._advertised_supported_models(),
                "model_eta_seconds": self.model_eta_seconds,
                "downloads": self.download_state,
            }

        return app

