# quickpod_api.py
import os
import requests
import logging
from typing import Optional, Dict, Any

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

    def __init__(self, email: str, password: str, pod_uuid: Optional[str] = None, api_base: str = None):
        self.email = email or os.getenv("QUICKPOD_EMAIL")
        self.password = password or os.getenv("QUICKPOD_PASSWORD")
        self.pod_uuid = pod_uuid
        self.token = None
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
        return self.token

    def _auth_headers(self) -> Dict[str, str]:
        if not self.token:
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    def _make_request(self, method: str, endpoint: str, headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        url = f"{self.base.rstrip('/')}/{endpoint.lstrip('/')}"

        merged_headers = self._auth_headers() or {}
        if headers:
            merged_headers.update(headers)

        if params is None and self.pod_uuid:
            params = {"pod_uuid": self.pod_uuid}

        try:
            resp = requests.request(method, url, headers=merged_headers, params=params, timeout=30, **kwargs)
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            resp.raise_for_status()
            return {"raw": resp.text}

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")

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
    def start_pod(self, payload: Optional[dict] = None):
        """Start a pod. Accepts optional payload for flexibility."""
        return self._make_request("POST", "pods/start", json=payload or {})

    def stop_pod(self, payload: Optional[dict] = None):
        """Stop a pod. Accepts optional payload for flexibility."""
        return self._make_request("POST", "pods/stop", json=payload or {})

    def restart_pod(self, payload: Optional[dict] = None):
        """Restart a pod. Accepts optional payload for flexibility."""
        return self._make_request("POST", "pods/restart", json=payload or {})

    def destroy_pod(self):
        """Permanently destroy the current pod (frees resources)."""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot destroy pod.")
        return self._make_request("POST", "pods/destroy", json={"pod_uuid": self.pod_uuid})

    def delete_pod(self, *args, **kwargs):
        """Alias for destroy_pod to match main.py expectations."""
        return self.destroy_pod(*args, **kwargs)

    def get_logs(self):
        return self._make_request("GET", "podlogs")

    # ---------------------------
    # Machine Listing + Pod Creation
    # ---------------------------
    def get_best_offer(self):
        """Fetch rentable machines and return best offer_id."""
        resp = requests.get(RENTABLE_URL)
        resp.raise_for_status()
        offers = resp.json()

        filtered = [
            o for o in offers
            if "5090" in o.get("offer_name", "")
            and not o.get("occupied")
            and o.get("num_gpus") == 1
            and float(o.get("_machines", {}).get("inet_down", 0)) >= 300
            and float(o.get("_machines", {}).get("inet_up", 0)) >= 300
        ]

        if not filtered:
            raise RuntimeError("No suitable offers found")

        filtered.sort(key=lambda x: float(x.get("hourly_cost", 9999)))
        return filtered[0]["id"]

    def create_pod(self, offer_id: int, template_uuid: str, disk_size: int = 50000):
        """Create a new pod with given offer_id, template_uuid, and disk size."""
        body = {
            "offers_id": offer_id,
            "disk_size": disk_size,
            "template_uuid": template_uuid,
        }
        url = f"{BASE_URL}/createpod"
        resp = requests.post(url, headers=self._auth_headers(), json=body)
        resp.raise_for_status()
        data = resp.json()
        self.pod_uuid = data.get("pod_uuid")
        # data contains: pod_uuid, public_ipaddr, open_port_start, open_port_end, status
        return data
