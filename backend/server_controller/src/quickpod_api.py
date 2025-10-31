# quickpod_api.py
from datetime import datetime, timedelta
import json
import os
import requests
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("quickpod")
logger.setLevel(logging.INFO)

QUICKPOD_BASE = os.getenv("QUICKPOD_BASE", "https://api.quickpod.org")

BASE_URL = "https://api.quickpod.org/update"
RENTABLE_URL = "https://api.quickpod.org/rentable"


class QuickPodManager:
    """
    Thin wrapper to call QuickPod API. This class is simple and stateless.
    For operations which target a specific pod, instantiate with pod_uuid.
    """

    def __init__(
        self,
        email: str,
        password: str,
        pod_uuid: Optional[str] = None,
        api_base: str = None,
    ):
        self.email = email or os.getenv("QUICKPOD_EMAIL")
        self.password = password or os.getenv("QUICKPOD_PASSWORD")
        self.pod_uuid = pod_uuid
        self.token = None
        self.token_acquired_at: Optional[datetime] = None
        # assume tokens valid for 6 hours unless API tells us otherwise
        self.token_ttl = timedelta(hours=6)
        self.base = api_base or QUICKPOD_BASE
        print(f"QuickPodManager initialized for {self.email} with pod_uuid={pod_uuid}")

    def login(self):
        url = f"{BASE_URL}/auth/login"
        payload = {"email": self.email, "password": self.password}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.token = data.get("authToken") or data.get("token") or data.get("jwt")
        if not self.token:
            raise RuntimeError(f"Login failed: {data}")
        print("Login successful. Token acquired.")
        self.token_acquired_at = datetime.utcnow()
        # if response provides expiry, set token_ttl accordingly (best-effort)
        expires_at = data.get("expires_at") or data.get("expiry")
        if expires_at:
            try:
                # try parse ISO timestamp
                self.token_ttl = datetime.fromisoformat(expires_at) - datetime.utcnow()
            except Exception:
                pass
        logger.info("Login successful. Token acquired.")
        return self.token
    
    def _is_token_valid(self) -> bool:
        if not self.token or not self.token_acquired_at:
            return False
        return (datetime.utcnow() - self.token_acquired_at) < self.token_ttl

    def _auth_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        # reuse token when valid; otherwise login
        if force_refresh or not self._is_token_valid():
            self.login()
        return {"Authorization": f"Bearer {self.token}"}


    def _make_request(
    self,
    method: str,
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    retry_on_401: bool = True,
    **kwargs,
    ) -> Any:
        url = f"{self.base.rstrip('/')}/{endpoint.lstrip('/')}"
        merged_headers = self._auth_headers() or {}
        if headers:
            merged_headers.update(headers)
        if params is None and self.pod_uuid:
            params = {"pod_uuid": self.pod_uuid}
        resp = requests.request(
            method, url, headers=merged_headers, params=params, timeout=30, **kwargs
        )

        # if unauthorized, try refresh once
        if resp.status_code == 401 and retry_on_401:
            logger.warning("401 returned from QuickPod; forcing token refresh and retrying.")
            merged_headers = self._auth_headers(force_refresh=True)
            resp = requests.request(method, url, headers=merged_headers, params=params, timeout=30, **kwargs)

        raw_text = resp.text[:4096]  # short preview
        try:
            data = resp.json()
        except Exception as je:
            # Log the raw response for easier debugging and re-raise http error
            logger.error("QuickPod returned non-JSON response (status=%s) for %s %s: %s", resp.status_code, method, endpoint, raw_text)
            resp.raise_for_status()
            return {"raw": raw_text}

        # If API returned structured error, include the full body in the exception message
        if isinstance(data, dict) and "error" in data:
            logger.error("QuickPod API error (status=%s) endpoint=%s: %s", resp.status_code, endpoint, json.dumps(data))
            # raise including original payload for easier debugging
            raise RuntimeError(f"API Error: {data.get('error')} (full_response={json.dumps(data)})")
        resp.raise_for_status()
        return data

    # ---------------------------
    # Template Management
    # ---------------------------
    def get_templates(self):
        """Fetch available templates from QuickPod API."""
        return self._make_request("GET", "templates")

    def get_template_by_name(self, template_name: str) -> Optional[str]:
        """Return template UUID by template name, or None if not found."""
        templates = self.get_templates()

        for template in templates:
            if template.get("template_name") == template_name:
                return template.get("template_uuid")
        return None

    # ---------------------------
    # Pod Management
    # ---------------------------
    def start_pod(self):
        """Start a pod by UUID (GET with query param)."""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot start pod.")
        return self._make_request(
            "GET", "update/startpod", params={"pod_uuid": self.pod_uuid}
        )

    def stop_pod(self):
        """Stop a pod by UUID."""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot stop pod.")
        return self._make_request(
            "GET", "update/stoppod", params={"pod_uuid": self.pod_uuid}
        )

    def restart_pod(self):
        """Restart a pod by UUID."""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot restart pod.")
        return self._make_request(
            "GET", "update/restartpod", params={"pod_uuid": self.pod_uuid}
        )

    def destroy_pod(self):
        """Destroy the current pod (frees resources)."""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot destroy pod.")
        logger.info("QuickPodManager.destroy_pod called for pod_uuid=%s", self.pod_uuid)
        return self._make_request(
            "GET", "update/destroypod", params={"pod_uuid": self.pod_uuid}
        )

    def delete_pod(self, *args, **kwargs):
        """Alias for destroy_pod to match main.py expectations."""
        return self.destroy_pod(*args, **kwargs)

    def get_logs(self):
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot fetch logs.")
        return self._make_request(
            "GET", "update/podlogs", params={"pod_uuid": self.pod_uuid}
        )

    # ---------------------------
    # Machine Listing + Pod Creation
    # ---------------------------

    def get_best_offer(self, preferred_machine_id: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """
        Fetch rentable machines and return (offer_id, machine_id) tuple for the best available match.
        Never creates pods. Selection priority:
        1. Preferred machine (if specified)
        2. MACHINE_ID env
        3. WARM_MACHINE_ID env
        4. Fallback — cheapest offer by GPU tier (e.g. 5090 > 4090 > 4080)
        """
        try:
            resp = requests.get(RENTABLE_URL, timeout=30)
            resp.raise_for_status()
            offers = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch offers: {e}")

        if not offers:
            return None, None

        # --- Helpers ---
        def extract_offer_id(o): 
            return o.get("offer_id") or o.get("id")

        def extract_machine_id(o):
            for key in ("machines_id", "machine_id", "machineId"):
                if key in o and o[key] is not None:
                    return int(o[key])
            return None

        def is_multi_gpu_offer(o):
            name = str(o.get("offer_name", "")).upper()
            if name.startswith(("2X", "3X", "4X")):
                return True
            g = str(o.get("gpu_indexes", "") or "")
            tokens = [t for t in g.split("|") if t.strip()]
            return len(tokens) > 1

        def extract_price(o):
            try:
                return float(o.get("hourly_cost", 1e9))
            except Exception:
                return 1e9

        def has_valid_tflops(o):
            try:
                return float(o.get("tflops", 0)) > 0
            except Exception:
                return False

        # --- 1️⃣ Preferred machine ---
        if preferred_machine_id:
            filtered = [
                o for o in offers 
                if extract_machine_id(o) == int(preferred_machine_id)
                and not is_multi_gpu_offer(o)
                and has_valid_tflops(o)
            ]
            if filtered:
                oid = extract_offer_id(filtered[0])
                mid = extract_machine_id(filtered[0])
                if oid:
                    logger.info(f"✅ Using preferred machine {mid} offer_id={oid}")
                    return oid, mid
            logger.warning(f"No offers found for preferred machine {preferred_machine_id}")

        # --- 2️⃣ Env machine fallback ---
        for env_key in ("MACHINE_ID", "WARM_MACHINE_ID"):
            env_val = os.getenv(env_key)
            if env_val:
                filtered = [
                    o for o in offers
                    if extract_machine_id(o) == int(env_val)
                    and not is_multi_gpu_offer(o)
                    and has_valid_tflops(o)
                ]
                if filtered:
                    oid = extract_offer_id(filtered[0])
                    mid = extract_machine_id(filtered[0])
                    logger.info(f"✅ Using {env_key}={mid} offer_id={oid}")
                    return oid, mid

        # --- 3️⃣ Fallback by GPU tier & cheapest ---
        gpu_tiers = ["5090", "4090", "4080TI", "4080"]
        
        for tier in gpu_tiers:
            tier_offers = [
                o for o in offers
                if tier in str(o.get("offer_name", "")).upper()
                and not is_multi_gpu_offer(o)
                and has_valid_tflops(o)
            ]
            tier_offers.sort(key=extract_price)
            if tier_offers:
                cheapest = tier_offers[0]
                oid = extract_offer_id(cheapest)
                mid = extract_machine_id(cheapest)
                logger.info(
                    f"⚙️ Using fallback offer_id={oid} (GPU={tier}, price={extract_price(cheapest):.3f}, tflops={cheapest.get('tflops')}) machine_id={mid}"
                )
                return oid, mid

        logger.error("❌ No suitable offers found with tflops > 0.")
        return None, None


    def create_pod(self, offer_id: int, template_uuid: str, disk_size: int = 50):
        """Create a new pod with given offer_id, template_uuid, and disk size."""
        body = {
            "offers_id": offer_id,
            "disk_size": str(disk_size),
            "template_uuid": template_uuid,
        }

        # Correct endpoint: /update/createpod
        data = self._make_request("POST", "update/createpod", json=body)
        if data is None:
            raise RuntimeError("No data returned from create_pod API call")
        if data.get("status") != "success":
            raise RuntimeError(f"Pod creation failed: {data}")
        else:
            logger.info("Pod creation successful.")
        logger.info(f"QuickPod API response: {json.dumps(data, indent=2)}")

        self.pod_uuid = data.get("pod_uuid") or data.get("uuid") or data.get("id")
        return data
    
if __name__ == "__main__":
        # Create manager instance with environment variables
        manager = QuickPodManager(email="pavanalur3@gmail.com", password="C@9rw6fxATDm8sM")  # Will use env vars
        # Get templates and save to file
        data = manager.get_templates()
        with open("quickpod_templates.json", "w") as f:
            json.dump(data, f, indent=2)
        print("\n✅ QuickPod templates saved to quickpod_templates.json\n")
