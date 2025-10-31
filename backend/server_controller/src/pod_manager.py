# pod_manager.py
import os
import time
import json
import tempfile
import logging
import asyncio
import enum
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from quickpod_api import QuickPodManager  # shared global QuickPodManager

logger = logging.getLogger("pod_manager")


class PodStrategyType(str, enum.Enum):
    WARM_AND_INDIVIDUAL = "warm_and_individual"
    MACHINE_BASED = "machine_based"


MACHINE_PRIORITY_LIST: List[Tuple[str, str]] = [
    ("562", "4x 5090"),
    ("14790", "4x 5090"),
    ("14780", "6x 5090"),
    ("14762", "1x 5090"),
    ("275", "1x 5090"),
    ("14774", "4x 4090"),
    ("266", "4x 4090"),
    ("416", "2x 4090"),
]

MAX_JOBS_PER_POD = 100
MAX_SAME_JOB_PER_POD = 100


@dataclass
class PodInfo:
    pod_uuid: str
    model: str
    mode: Optional[str]
    status: str
    created_at: datetime
    last_active: datetime
    queue_size: int = 0
    error_count: int = 0
    machine_id: Optional[str] = None
    total_jobs_processed: int = 0
    unique_job_ids: set = field(default_factory=set)
    job_id_counts: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class PodManager:
    def __init__(self, email: str, password: str, warm_pod_uuid: Optional[str]):
        self.email = email
        self.password = password
        self.warm_pod_uuid = warm_pod_uuid
        self.warm_pod_manager = (
            QuickPodManager(email, password, warm_pod_uuid) if warm_pod_uuid else None
        )
        self.own_warm_pod: bool = False
        self.individual_pods: Dict[str, PodInfo] = {}
        self.individual_pod_managers: Dict[str, QuickPodManager] = {}
        self._lock = asyncio.Lock()

        self.pending_jobs: Dict[str, set] = {}
        self._jobs_lock = asyncio.Lock()
        self.recently_destroyed: Dict[str, float] = {}
        # Track machines that failed outputs recently
        self.bad_machines: Dict[str, float] = {}
        self.bad_machine_cooldown = int(
            os.getenv("BAD_MACHINE_COOLDOWN", "600")
        )  # 10 min

        self.pod_strategy = PodStrategyType(
            os.getenv("POD_STRATEGY", PodStrategyType.WARM_AND_INDIVIDUAL.value)
        )
        self.max_individual_pods = int(os.getenv("MAX_INDIVIDUAL_PODS", "10"))
        self.pod_startup_timeout = int(os.getenv("POD_STARTUP_TIMEOUT", "60"))
        self.template_uuids = {
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl": os.getenv("SDXL_TEMPLATE_UUID"),
            "diffusers_outpaint": os.getenv("DIFFUSERS_OUTPAINT_TEMPLATE_UUID"),
            "qwen_image_edit": os.getenv("QWEN_IMAGE_EDIT_TEMPLATE_UUID"),
            "qwen_image_edit_2509": os.getenv("QWEN_IMAGE_EDIT_2509_TEMPLATE_UUID"),
            "flux_kontext_dev": os.getenv("FLUX_KONTEXT_DEV_TEMPLATE_UUID"),
            "fastwan_21_1b": os.getenv("FASTWAN_21_1B_TEMPLATE_UUID"),
            "fastwan_22_5b": os.getenv("FASTWAN_22_5B_TEMPLATE_UUID"),
            "aurasrv2": os.getenv("AURASRV2_TEMPLATE_UUID"),
            "realesrgan": os.getenv("REALESRGAN_TEMPLATE_UUID"),
            "swin2sr": os.getenv("SWIN2SR_TEMPLATE_UUID"),
            "basic_utils": os.getenv("BASIC_UTILS_TEMPLATE_UUID"),
        }
        self.template_disk_size = {
            "sd15": 30,
            "sd35": 40,
            "sdxl": 30,
            "diffusers_outpaint": 50,
            "qwen_image_edit": 80,
            "qwen_image_edit_2509": 85,
            "flux_kontext_dev": 45,
            "fastwan_21_1b": 60,
            "fastwan_22_5b": 60,
            "aurasrv2": 30,
            "realesrgan": 30,
            "swin2sr": 30,
            "basic_utils": 30,
        }

        # 🔹 Shared global QuickPodManager
        self.global_qp = QuickPodManager(email, password)
        self._global_qp_lock = asyncio.Lock()
        self._global_qp_logged_in = False
        self._global_qp_last_login = None

        # 🔹 Machine cache persistence
        self.machine_cache_path = os.getenv("MACHINE_CACHE_PATH", "machine_cache.json")
        self.machine_cache = {
            "machines": {},
            "meta": {"version": 1, "updated_at": None},
        }
        self._machine_cache_lock = asyncio.Lock()

        # 🔹 Fallback event tracking (for autoscaler)
        self.fallback_events = deque()
        self.last_autoscale_ts = 0
        self.autoscale_cooldown = int(os.getenv("AUTOSCALE_COOLDOWN", "600"))

    @property
    def active_pods(self) -> Dict[str, PodInfo]:
        """
        Returns a unified dict of all currently tracked pods (warm + individual).
        This is used by the controller to validate incoming worker registrations.
        """
        pods = dict(self.individual_pods)
        if self.warm_pod_uuid and self.own_warm_pod:
            pods[self.warm_pod_uuid] = PodInfo(
                pod_uuid=self.warm_pod_uuid,
                model="warm",
                mode=None,
                status="ready",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
            )
        return pods

    # --------------------------------------------------------------------------
    # MACHINE CACHE HELPERS
    # --------------------------------------------------------------------------
    async def load_machine_cache(self):
        """Load machine_cache.json from disk (if exists)."""
        async with self._machine_cache_lock:
            if not os.path.exists(self.machine_cache_path):
                return
            try:
                with open(self.machine_cache_path, "r") as f:
                    self.machine_cache = json.load(f)
                    logger.info(
                        "Loaded machine cache with %d machines",
                        len(self.machine_cache.get("machines", {})),
                    )
            except Exception:
                logger.exception("Failed to load machine cache file")

    async def _persist_machine_cache(self):
        async with self._machine_cache_lock:
            try:
                cache_dir = os.path.dirname(self.machine_cache_path)
                if cache_dir and not os.path.exists(cache_dir):
                    # create with safe perms
                    os.makedirs(cache_dir, exist_ok=True)
                # write to a temp file in same directory (avoid cross-device rename issues)
                tmp_path = f"{self.machine_cache_path}.tmp"

                def _write():
                    with open(tmp_path, "w") as f:
                        json.dump(self.machine_cache, f, indent=2)
                    os.replace(tmp_path, self.machine_cache_path)

                await asyncio.to_thread(_write)
                logger.debug("Persisted machine cache to %s", self.machine_cache_path)
            except Exception:
                # include path in logs for easier debugging
                logger.exception(
                    "Failed to persist machine cache (path=%s)", self.machine_cache_path
                )

    async def mark_image_cached_on_machine(self, machine_id: str, image_name: str):
        """Mark an image as cached on a given machine."""
        async with self._machine_cache_lock:
            now = datetime.utcnow().isoformat() + "Z"
            mc = self.machine_cache.setdefault("machines", {})
            m = mc.setdefault(
                str(machine_id), {"cached_images": {}, "is_priority": False}
            )
            m["cached_images"][image_name] = {
                "added_at": now,
                "score": m["cached_images"].get(image_name, {}).get("score", 0.0) + 1.0,
            }
            m["last_seen"] = now
            self.machine_cache["meta"]["updated_at"] = now
        await self._persist_machine_cache()
        logger.info(f"✅ Cached record updated: {image_name} on machine {machine_id}")

    # --------------------------------------------------------------------------
    # AUTOSCALER HELPERS
    # --------------------------------------------------------------------------
    async def record_fallback_event(self, model: str, machine_id: str):
        now = time.time()
        self.fallback_events.append((now, model, machine_id))
        window = int(os.getenv("FALLBACK_WINDOW_SECONDS", "600"))
        while self.fallback_events and now - self.fallback_events[0][0] > window:
            self.fallback_events.popleft()

    async def autoscaler_loop(self):
        """Detect fallback frequency and trigger warm pod spin-up."""
        FALLBACK_THRESHOLD = int(os.getenv("FALLBACK_THRESHOLD", "5"))
        QUEUE_MIN_FOR_SCALE = int(os.getenv("QUEUE_MIN_FOR_SCALE", "3"))
        LOAD_SCALE_THRESHOLD = int(os.getenv("LOAD_SCALE_THRESHOLD", "15"))
        while True:
            try:
                now = time.time()
                if now - self.last_autoscale_ts < self.autoscale_cooldown:
                    await asyncio.sleep(5)
                    continue

                # 🔹 Load-based autoscaling
                async with self._lock:
                    model_pods = {}
                    for pod_uuid, info in self.individual_pods.items():
                        if info.status in ("ready", "starting"):
                            model_pods.setdefault(info.model, []).append(info)

                for model, jobs in list(self.pending_jobs.items()):
                    pending_count = len(jobs)
                    active_pods = model_pods.get(model, [])
                    no_ready = all(p.status != "ready" for p in active_pods)
                    if pending_count >= LOAD_SCALE_THRESHOLD and no_ready:
                        logger.info(
                            f"⚡ Load autoscaler trigger: model={model}, pending={pending_count}, pods={len(active_pods)}"
                        )
                        self.last_autoscale_ts = now
                        await self.scale_up_unified_warm_pod_for_model(model)
                        break

                counts: Dict[str, int] = {}
                for ts, model, mid in self.fallback_events:
                    counts[model] = counts.get(model, 0) + 1

                for model, cnt in counts.items():
                    if cnt >= FALLBACK_THRESHOLD:
                        logger.info(
                            f"Autoscaler trigger: model={model}, fallback_count={cnt}"
                        )
                        self.last_autoscale_ts = now
                        # call scaler (simple implementation; queue checks can be added later)
                        await self.scale_up_unified_warm_pod_for_model(model)
                        break

            except Exception:
                logger.exception("Autoscaler loop error")
            await asyncio.sleep(5)

    async def scale_up_unified_warm_pod_for_model(self, model: str):
        """Create a unified warm pod on a chosen machine when fallback frequency high.

        NOTE: This is intentionally conservative. You can later implement logic to pick the
        cheapest available machine_id (e.g., by reading machine metadata / costs).
        """
        try:
            logger.info(f"🚀 Scaling unified warm pod for model={model}")
            # For now, create machine-based pod using the priority list selection logic.
            # You may adapt to pass a preferred_machine_id (quickpod_api.get_best_offer supports it).
            pod_uuid = await self.create_machine_based_pod(model, mode=None)
            if pod_uuid:
                logger.info(f"Scaled unified warm pod {pod_uuid} for model {model}")
            else:
                logger.warning("Scale attempt failed (no pod created)")
        except Exception:
            logger.exception("Failed to scale unified warm pod")

    # --------------------------------------------------------------------------
    # GLOBAL LOGIN / RETRY
    # --------------------------------------------------------------------------
    async def _ensure_global_login(self, force_refresh: bool = False):
        async with self._global_qp_lock:
            needs_login = (
                force_refresh
                or not self._global_qp_logged_in
                or not self.global_qp._is_token_valid()
            )
            if needs_login:
                try:
                    await asyncio.to_thread(self.global_qp.login)
                    self._global_qp_logged_in = True
                    self._global_qp_last_login = datetime.utcnow()
                    logger.info("QuickPodManager: global login successful.")
                except Exception as e:
                    logger.error("Global QuickPodManager login failed: %s", e)
                    raise

    async def _call_with_retry(self, func, *args, **kwargs):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            logger.warning("QuickPod API call failed (%s); retrying once...", e)
            await asyncio.sleep(2)
            await self._ensure_global_login(force_refresh=True)
            return await asyncio.to_thread(func, *args, **kwargs)

    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------
    async def initialize(self) -> None:
        await self.load_machine_cache()
        if not self.warm_pod_manager:
            logger.info("No warm pod configured.")
            self.own_warm_pod = False
            return
        try:
            logger.info("Initializing warm pod...")
            await self._ensure_global_login()
            resp = await asyncio.to_thread(self.warm_pod_manager.start_pod)
            if isinstance(resp, dict) and "busy" in str(resp.get("error", "")).lower():
                logger.warning("Warm pod busy, cannot acquire.")
                self.own_warm_pod = False
            else:
                self.own_warm_pod = True
                logger.info("Warm pod started successfully.")
        except Exception as e:
            logger.exception("Failed to initialize warm pod: %s", e)
            self.own_warm_pod = False

    # --------------------------------------------------------------------------
    # JOB DISPATCH (entry)
    # --------------------------------------------------------------------------
    async def dispatch_job(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> str:
        if self.pod_strategy == PodStrategyType.WARM_AND_INDIVIDUAL:
            return await self._dispatch_job_warm_and_individual(job_id, model, mode)
        else:
            return await self._dispatch_job_machine_based(job_id, model, mode)

    # --------------------------------------------------------------------------
    # WARM + INDIVIDUAL DISPATCH
    # --------------------------------------------------------------------------
    async def _dispatch_job_warm_and_individual(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> str:
        """Dispatch flow when running warm+individual strategy.

        Prefer the warm pod worker if available; otherwise create an individual pod.
        """
        from db import SessionLocal, Worker, Job
        from job_manager import JobState

        await self.add_pending_job(job_id, model)

        # 1) try to find a ready warm worker in DB
        with SessionLocal() as db:
            w = (
                db.query(Worker)
                .filter(
                    Worker.is_warm_pod == True,
                    Worker.model == model,
                    Worker.status == "ready",
                )
                .order_by(Worker.last_active.desc())
                .first()
            )
            if w:
                # mark dispatching and return worker_id
                w.status = "dispatching"
                w.last_active = datetime.utcnow()
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job.status = JobState.DISPATCHING.value
                    job.updated_at = datetime.utcnow()
                db.commit()
                logger.info("Dispatching job %s to warm worker %s", job_id, w.worker_id)
                return w.worker_id

        # 2) no warm worker available -> create an individual pod and wait for registration
        pod_uuid = await self.create_individual_pod(model, mode)
        if not pod_uuid:
            raise RuntimeError("Failed to create individual pod")

        # wait for worker registration
        start = time.time()
        timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        with SessionLocal() as db:
            while time.time() - start < timeout:
                w = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.model == model,
                        Worker.status == "ready",
                    )
                    .first()
                )
                if w:
                    w.status = "dispatching"
                    w.last_active = datetime.utcnow()
                    # update job state
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        job.status = JobState.DISPATCHING.value
                        job.updated_at = datetime.utcnow()
                    db.commit()
                    return w.worker_id
                await asyncio.sleep(1)

        raise RuntimeError(f"Worker registration timeout for pod {pod_uuid}")

    # --------------------------------------------------------------------------
    # MACHINE-BASED DISPATCH
    # --------------------------------------------------------------------------
    async def _dispatch_job_machine_based(
        self, job_id: str, model: str, mode: Optional[str] = None
    ) -> str:
        from db import SessionLocal, Worker, Job
        from job_manager import JobState

        await self.add_pending_job(job_id, model)

        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if info.machine_id and self.is_machine_bad(info.machine_id):
                    continue  # don't reuse pods on blacklisted machine
                if (
                    info.model == model
                    and info.status == "ready"
                    and self.can_reuse_pod(info, job_id)
                ):
                    info.last_active = datetime.utcnow()
                    self.update_pod_job_counts(info, job_id)
                    logger.info(
                        "Reusing machine-based pod %s for job %s", pod_uuid, job_id
                    )
                    with SessionLocal() as db:
                        w = (
                            db.query(Worker)
                            .filter(
                                Worker.container_id == pod_uuid, Worker.model == model
                            )
                            .first()
                        )
                        if w:
                            w.status = "dispatching"
                            w.last_active = datetime.utcnow()
                            job = db.query(Job).filter(Job.job_id == job_id).first()
                            if job:
                                job.status = JobState.DISPATCHING.value
                                job.updated_at = datetime.utcnow()
                            db.commit()
                            return w.worker_id

        # Attempt to create on priority machines
        pod_uuid = await self.create_machine_based_pod(model, mode)
        if not pod_uuid:
            raise RuntimeError("Failed to create machine-based pod")

        # wait for worker registration
        start = time.time()
        timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        from db import SessionLocal

        while time.time() - start < timeout:
            with SessionLocal() as db:
                w = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.model == model,
                        Worker.status == "ready",
                    )
                    .first()
                )
                if w:
                    w.status = "dispatching"
                    w.last_active = datetime.utcnow()
                    db.commit()
                    return w.worker_id
            await asyncio.sleep(1)

        raise RuntimeError(f"Worker registration timeout for pod {pod_uuid}")

    # --------------------------------------------------------------------------
    # CREATE MACHINE-BASED POD (updated for caching & fallback tracking)
    # --------------------------------------------------------------------------

    def _is_4090_allowed_model(self, model: str) -> bool:
        """Return True if this model is allowed to run on 4090 machines."""
        model_lower = model.lower()
        allowed_keywords = [
            "upscale", "aurasr", "esrgan", "swin",
            "diffuser", "sdxl", "outpaint", "sd35", "sd15", "diffusion"
        ]
        return any(k in model_lower for k in allowed_keywords)

    async def create_machine_based_pod(
        self, model: str, mode: Optional[str] = None
    ) -> Optional[str]:
        """Create a new machine-based pod, trying priority machines, then cached machines,
        then global fallback. Blacklists machines on certain failures to avoid immediate retries.
        """
        await self._ensure_global_login()

        template_id = self._get_template_for_model(model)
        if not template_id:
            logger.error(f"No template found for model {model}")
            return None

        # Build candidate machine list (str ids), preserving order:
        candidates: List[Tuple[str, Optional[str]]] = []

        # ✅ Filter machine candidates based on model type
        use_4090 = self._is_4090_allowed_model(model)
        filtered_priority_list = []

        for mid, desc in MACHINE_PRIORITY_LIST:
            if "4090" in desc and not use_4090:
                logger.info(f"⛔ Skipping 4090 machine {mid} for model {model} (not allowed)")
                continue
            if "5090" in desc and use_4090:
                # Still allow 5090 as backup for 4090 models, optional
                filtered_priority_list.append((mid, desc))
                continue
            filtered_priority_list.append((mid, desc))

        # 1) replace MACHINE_PRIORITY_LIST with filtered one
        for mid, desc in filtered_priority_list:
            candidates.append((str(mid), desc))
        
        # 2) machines from machine_cache that either have this model cached or are marked priority
        try:
            async with self._machine_cache_lock:
                mc = (
                    self.machine_cache.get("machines", {}) if self.machine_cache else {}
                )
                # sort cached machines by (is_priority, score of model if exists, last_seen) descending
                scored = []
                for mid, info in mc.items():
                    is_pr = info.get("is_priority", False)
                    cached_images = info.get("cached_images", {})
                    model_info = cached_images.get(model, {})
                    score = model_info.get("score", 0.0)
                    last_seen = info.get("last_seen") or ""
                    scored.append((mid, is_pr, score, last_seen))
                # prefer is_priority, then higher score, then recent last_seen
                scored.sort(key=lambda x: (not x[1], -x[2], x[3]))
                for mid, is_pr, score, last_seen in scored:
                    # don't duplicate
                    if str(mid) not in {c[0] for c in candidates}:
                        candidates.append((str(mid), "cached"))
        except Exception:
            logger.exception(
                "Error while reading machine cache; proceeding with defaults"
            )

        # dedupe while preserving order
        seen = set()
        candidate_list = []
        for mid, desc in candidates:
            if mid in seen:
                continue
            seen.add(mid)
            candidate_list.append((mid, desc))

        logger.info(
            "Machine candidates for model %s: %s", model, [c[0] for c in candidate_list]
        )

        # Try each candidate machine using QuickPod global API asking for offer for a specific machine (if supported)
        for candidate_mid, desc in candidate_list:
            mid = str(candidate_mid)
            # skip bad machines
            if self.is_machine_bad(mid):
                logger.info(f"⛔ Skipping blacklisted machine {mid}")
                continue

            try:
                # get best offer specifically for this machine (some implementations return tuple)
                offer_result = await self._call_with_retry(
                    self.global_qp.get_best_offer, mid
                )
                # offer_result may be (offer_id, actual_machine_id) or just offer_id
                if not offer_result:
                    logger.debug("No offer for machine %s", mid)
                    continue
                if isinstance(offer_result, tuple):
                    offer_id, actual_machine_id = offer_result
                    actual_mid = str(actual_machine_id)
                else:
                    offer_id = offer_result
                    actual_mid = mid

                # if QuickPod returned a different actual machine, normalize that too
                if actual_mid != mid:
                    logger.warning(
                        f"Preferred machine {mid} unavailable; QuickPod suggests {actual_mid} instead"
                    )
                    # skip if suggested machine is blacklisted
                    if self.is_machine_bad(actual_mid):
                        logger.info(
                            f"⛔ QuickPod's suggested machine {actual_mid} is blacklisted; skipping"
                        )
                        continue
                    chosen_machine = actual_mid
                else:
                    chosen_machine = mid

                logger.info(
                    f"🚀 Creating pod for {model} on machine {chosen_machine} (candidate {mid}, desc={desc})"
                )
                pod_data = await self._call_with_retry(
                    self.global_qp.create_pod,
                    offer_id,
                    template_id,
                    self.template_disk_size.get(model, 100),
                )
                if not isinstance(pod_data, dict):
                    logger.warning(
                        "create_pod returned unexpected type for machine %s",
                        chosen_machine,
                    )
                    continue

                pod_uuid = (
                    pod_data.get("pod_uuid")
                    or pod_data.get("uuid")
                    or pod_data.get("id")
                )
                if not pod_uuid:
                    logger.warning(
                        "No pod uuid returned for machine %s", chosen_machine
                    )
                    continue

                # register pod
                async with self._lock:
                    self.individual_pods[pod_uuid] = PodInfo(
                        pod_uuid=pod_uuid,
                        model=model,
                        mode=mode,
                        status="starting",
                        created_at=datetime.utcnow(),
                        last_active=datetime.utcnow(),
                        machine_id=str(chosen_machine),
                    )
                    self.individual_pod_managers[pod_uuid] = QuickPodManager(
                        self.email, self.password, pod_uuid
                    )

                # wait for worker registration
                start_time = time.time()
                while time.time() - start_time < self.pod_startup_timeout:
                    if await self.check_pod_ready(pod_uuid):
                        async with self._lock:
                            self.individual_pods[pod_uuid].status = "ready"
                        # record cache that image is on this machine (best-effort)
                        try:
                            await self.mark_image_cached_on_machine(
                                str(chosen_machine), model
                            )
                        except Exception:
                            logger.debug(
                                "mark_image_cached_on_machine failed, continuing"
                            )
                        logger.info(
                            "✅ Pod %s ready on machine %s for %s",
                            pod_uuid,
                            chosen_machine,
                            model,
                        )
                        return pod_uuid
                    await asyncio.sleep(5)

                # startup timeout: destroy and mark machine bad to avoid immediate retry
                logger.error(
                    "Pod %s startup timeout on machine %s; destroying pod and marking machine bad",
                    pod_uuid,
                    chosen_machine,
                )
                try:
                    await self.destroy_individual_pod(pod_uuid)
                except Exception:
                    logger.exception("Failed to destroy timed out pod %s", pod_uuid)

                # mark machine bad on startup timeout
                try:
                    self.mark_machine_bad(chosen_machine)
                except Exception:
                    logger.exception(
                        "Failed to mark machine bad after timeout: %s", chosen_machine
                    )

            except RuntimeError as re:
                msg = str(re).lower()
                logger.exception(
                    "Runtime error creating pod on candidate %s: %s", mid, re
                )
                # if DB or transient error - skip candidate; if repeated errors occur we will fallback
                # Optionally mark machine bad for non-transient errors
                continue
            except Exception as e:
                logger.exception(
                    "Unexpected error creating pod on candidate %s: %s", mid, e
                )
                # mark candidate as bad to avoid retrying aggressively
                try:
                    self.mark_machine_bad(mid)
                except Exception:
                    pass
                continue

        # Global fallback: ask for the best offer without preferred machine
        try:
            logger.warning(
                "⚠️ All preferred/cached machines exhausted. Trying global fallback offer..."
            )
            offer = await self._call_with_retry(self.global_qp.get_best_offer)
            if not offer:
                logger.error("❌ No global fallback offer available.")
                return None

            if isinstance(offer, tuple):
                global_offer_id, fallback_machine_id = offer
                fallback_mid = str(fallback_machine_id) if fallback_machine_id else None
            else:
                global_offer_id = offer
                fallback_mid = None

            if fallback_mid and self.is_machine_bad(fallback_mid):
                logger.error(
                    f"❌ Global fallback machine {fallback_mid} is blacklisted. Aborting fallback."
                )
                return None

            pod_data = await self._call_with_retry(
                self.global_qp.create_pod,
                global_offer_id,
                template_id,
                self.template_disk_size.get(model, 100),
            )
            if not isinstance(pod_data, dict):
                logger.error("Invalid fallback pod creation response.")
                return None
            pod_uuid = (
                pod_data.get("pod_uuid") or pod_data.get("uuid") or pod_data.get("id")
            )
            if not pod_uuid:
                logger.error("No pod UUID returned for fallback pod creation.")
                return None

            chosen_mid = fallback_mid or "fallback"
            async with self._lock:
                self.individual_pods[pod_uuid] = PodInfo(
                    pod_uuid=pod_uuid,
                    model=model,
                    mode=mode,
                    status="starting",
                    created_at=datetime.utcnow(),
                    last_active=datetime.utcnow(),
                    machine_id=str(chosen_mid),
                )
                self.individual_pod_managers[pod_uuid] = QuickPodManager(
                    self.email, self.password, pod_uuid
                )

            start_time = time.time()
            while time.time() - start_time < self.pod_startup_timeout:
                if await self.check_pod_ready(pod_uuid):
                    async with self._lock:
                        self.individual_pods[pod_uuid].status = "ready"
                    logger.info("✅ Fallback pod %s ready for %s", pod_uuid, model)
                    return pod_uuid
                await asyncio.sleep(5)

            logger.error("⚠️ Fallback pod %s startup timeout for %s", pod_uuid, model)
            try:
                await self.destroy_individual_pod(pod_uuid)
            except Exception:
                pass
            if fallback_mid:
                self.mark_machine_bad(fallback_mid)
            return None

        except Exception as e:
            logger.exception("⚠️ Global fallback path failed for %s: %s", model, e)
            return None

    async def destroy_unexpected_pod(self, pod_uuid: str):
        """Force-stop and destroy a pod that was not requested by the orchestrator."""
        try:
            logger.warning(f"💣 Destroying unexpected pod {pod_uuid} to prevent cost.")

            # Create a temporary QuickPodManager bound to this specific pod UUID
            temp_manager = QuickPodManager(self.email, self.password, pod_uuid)

            # Stop and destroy via proper calls (each manager knows its pod_uuid)
            await asyncio.to_thread(temp_manager.stop_pod)
            await asyncio.sleep(2)  # small grace period
            await asyncio.to_thread(temp_manager.destroy_pod)

            logger.info(f"✅ Unexpected pod {pod_uuid} stopped and destroyed.")
        except Exception as e:
            logger.error(f"❌ Failed to destroy unexpected pod {pod_uuid}: {e}")
        finally:
            self.recently_destroyed[pod_uuid] = time.time()

    # --------------------------------------------------------------------------
    # INDIVIDUAL POD (WARM+INDIVIDUAL)
    # --------------------------------------------------------------------------
    async def create_individual_pod(
        self, model: str, mode: Optional[str] = None
    ) -> Optional[str]:
        await self._ensure_global_login()
        async with self._lock:
            if len(self.individual_pods) >= self.max_individual_pods:
                raise RuntimeError("Maximum number of individual pods reached")

        try:
            offer_id = await self._call_with_retry(self.global_qp.get_best_offer)
            if not offer_id:
                raise RuntimeError("No offer returned by QuickPod")

            template_uuid = self._get_template_for_model(model)
            pod_data = await self._call_with_retry(
                self.global_qp.create_pod,
                offer_id,
                template_uuid,
                self.template_disk_size.get(model, 100),
            )

            pod_uuid = (
                pod_data.get("pod_uuid") or pod_data.get("uuid") or pod_data.get("id")
            )
            if not pod_uuid:
                raise RuntimeError("No pod UUID returned")

            async with self._lock:
                self.individual_pods[pod_uuid] = PodInfo(
                    pod_uuid=pod_uuid,
                    model=model,
                    mode=mode,
                    status="starting",
                    created_at=datetime.utcnow(),
                    last_active=datetime.utcnow(),
                )
                self.individual_pod_managers[pod_uuid] = QuickPodManager(
                    self.email, self.password, pod_uuid
                )
            return pod_uuid
        except Exception as e:
            logger.exception("Failed to create individual pod: %s", e)
        return None

    # --------------------------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------------------------
    async def stop_individual_pod(self, pod_uuid: str) -> None:
        await self._ensure_global_login()
        async with self._lock:
            pod_manager = self.individual_pod_managers.get(pod_uuid) or self.global_qp
            exists = pod_uuid in self.individual_pods
        if not exists:
            return
        try:
            await self._call_with_retry(pod_manager.stop_pod)
        except Exception as e:
            logger.warning("stop_individual_pod failed: %s", e)
        await asyncio.sleep(1)
        try:
            await self._call_with_retry(pod_manager.destroy_pod)
        except Exception as e:
            logger.warning("destroy_pod failed: %s", e)

        # Remove from internal caches and also clear DB worker rows referencing this pod_uuid
        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        # mark recently destroyed
        self.recently_destroyed[pod_uuid] = time.time()

        # best-effort DB cleanup: clear container_id on any worker row that referenced this pod_uuid
        try:
            from db import SessionLocal, Worker
            db = SessionLocal()
            try:
                affected = db.query(Worker).filter(Worker.container_id == pod_uuid).all()
                for w in affected:
                    logger.info("DB sync: clearing container_id for worker %s after stop_individual_pod(%s)", w.worker_id, pod_uuid)
                    w.container_id = None
                    w.status = "offline"
                    w.last_active = datetime.utcnow()
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("DB cleanup failed in stop_individual_pod for pod %s", pod_uuid)
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to import DB for cleanup in stop_individual_pod")

        logger.info("Pod %s cleaned up (stop_individual_pod)", pod_uuid)


    async def destroy_individual_pod(self, pod_uuid: str) -> None:
        await self._ensure_global_login()
        async with self._lock:
            manager = self.individual_pod_managers.get(pod_uuid)
            exists = pod_uuid in self.individual_pods

        if not exists:
            # still mark recently_destroyed to avoid immediate re-register
            self.recently_destroyed[pod_uuid] = time.time()
            logger.debug("destroy_individual_pod: pod %s not tracked in individual_pods", pod_uuid)
            return

        try:
            use_temp = True
            if manager and getattr(manager, "pod_uuid", None) == pod_uuid:
                use_temp = False

            if use_temp:
                # create a temporary manager bound to correct pod_uuid
                tmp_mgr = QuickPodManager(self.email, self.password, pod_uuid)
                await self._call_with_retry(tmp_mgr.destroy_pod)
            else:
                await self._call_with_retry(manager.destroy_pod)
        except Exception as e:
            logger.warning("destroy_individual_pod failed: %s", e)

        # Remove from internal structures
        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        # record recently destroyed timestamp
        self.recently_destroyed[pod_uuid] = time.time()

        # best-effort DB sync: clear container_id on workers that referenced this pod_uuid
        try:
            from db import SessionLocal, Worker
            db = SessionLocal()
            try:
                affected = db.query(Worker).filter(Worker.container_id == pod_uuid).all()
                for w in affected:
                    logger.info("DB sync: clearing container_id for worker %s after destroy_individual_pod(%s)", w.worker_id, pod_uuid)
                    w.container_id = None
                    w.status = "terminated"
                    w.last_active = datetime.utcnow()
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("DB cleanup failed in destroy_individual_pod for pod %s", pod_uuid)
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to import DB for cleanup in destroy_individual_pod")

        logger.info("Individual pod %s destroyed (db worker rows cleared)", pod_uuid)

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _get_template_for_model(self, model: str) -> Optional[str]:
        return self.template_uuids.get(model) or self.template_uuids.get("default")

    def can_reuse_pod(self, pod_info: PodInfo, job_id: str) -> bool:
        if pod_info.total_jobs_processed >= MAX_JOBS_PER_POD:
            return False
        if job_id in pod_info.job_id_counts:
            return pod_info.job_id_counts[job_id] < MAX_SAME_JOB_PER_POD
        return len(pod_info.unique_job_ids) < MAX_JOBS_PER_POD

    def update_pod_job_counts(self, pod_info: PodInfo, job_id: str) -> None:
        pod_info.total_jobs_processed += 1
        pod_info.unique_job_ids.add(job_id)
        pod_info.job_id_counts[job_id] = pod_info.job_id_counts.get(job_id, 0) + 1

    async def check_pod_ready(self, pod_uuid: str) -> bool:
        from db import SessionLocal, Worker

        with SessionLocal() as db:
            w = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
            if w and w.status in ("ready", "processing"):
                return True
        return False

    async def add_pending_job(self, job_id: str, model: str):
        async with self._jobs_lock:
            if model not in self.pending_jobs:
                self.pending_jobs[model] = set()
            self.pending_jobs[model].add(job_id)

    async def remove_pending_job(self, job_id: str, model: str):
        async with self._jobs_lock:
            if model in self.pending_jobs:
                self.pending_jobs[model].discard(job_id)
                if not self.pending_jobs[model]:
                    del self.pending_jobs[model]

    async def has_pending_jobs_for_model(self, model: str) -> bool:
        async with self._jobs_lock:
            return model in self.pending_jobs and bool(self.pending_jobs[model])

    async def should_keep_pod_alive(self, pod_info: PodInfo) -> bool:
        if self.pod_strategy == PodStrategyType.WARM_AND_INDIVIDUAL:
            return True
        return await self.has_pending_jobs_for_model(pod_info.model)

    async def cleanup_inactive_pods(self) -> None:
        now = datetime.utcnow()
        to_stop = []
        async with self._lock:
            for pod_uuid, info in list(self.individual_pods.items()):
                inactive = (now - info.last_active) > timedelta(minutes=30)
                starting_timed_out = info.status == "starting" and (
                    now - info.created_at
                ) > timedelta(minutes=10)
                if info.error_count >= 3 or inactive or starting_timed_out:
                    to_stop.append(pod_uuid)
        if not to_stop:
            return
        logger.info("Cleaning up %d pods: %s", len(to_stop), to_stop)
        for pod_uuid in to_stop:
            try:
                await self.stop_individual_pod(pod_uuid)
            except Exception:
                logger.exception("Error cleaning up pod %s", pod_uuid)

    async def stop_warm_pod(self, pod_uuid: str) -> None:
        if not self.warm_pod_manager or pod_uuid != self.warm_pod_uuid:
            logger.warning("stop_warm_pod called for non-warm pod %s", pod_uuid)
            return
        try:
            await self._call_with_retry(self.warm_pod_manager.stop_pod)
            self.own_warm_pod = False
        except Exception:
            logger.exception("Error stopping warm pod %s", pod_uuid)

    def mark_machine_bad(self, machine_id: str):
        """Mark machine bad using normalized str keys and log clearly."""
        if not machine_id:
            return
        mid = str(machine_id)
        expiry = time.time() + self.bad_machine_cooldown
        self.bad_machines[mid] = expiry
        logger.warning(
            f"⚠️ Marking machine {mid} as bad until {expiry} (cooldown {self.bad_machine_cooldown}s)"
        )

    def is_machine_bad(self, machine_id: str) -> bool:
        """Normalize key to str and check expiry; remove stale entries."""
        if not machine_id:
            return False
        mid = str(machine_id)
        expiry = self.bad_machines.get(mid, 0)
        if expiry < time.time():
            # expired or non-existent
            self.bad_machines.pop(mid, None)
            return False
        return True
