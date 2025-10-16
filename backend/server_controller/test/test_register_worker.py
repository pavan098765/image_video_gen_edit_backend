from typing import Any, Dict, Optional, List
import requests
import json

BASE_URL = "https://api.quickpod.org/update"


class QuickPodClient:
    def __init__(self, email: str = None, password: str = None):
        self.email = email or "pavanalur3@gmail.com"
        self.password = password or "C@9rw6fxATDm8sM"
        self.token: Optional[str] = None
        self.pod_uuid: Optional[str] = None
        self.base = BASE_URL

    # --------------------------------------------------
    # AUTH
    # --------------------------------------------------
    def login(self) -> str:
        """Authenticate with the QuickPod API and get JWT token."""
        url = f"{self.base}/auth/login"
        payload = {"email": self.email, "password": self.password}

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Login request failed: {str(e)}")

        self.token = data.get("authToken") or data.get("token") or data.get("jwt")
        if not self.token:
            raise RuntimeError(f"Login failed: No token in response - {data}")

        print("✅ Login successful. Token acquired.")
        return self.token

    def _auth_headers(self) -> Dict[str, str]:
        """Return authorization headers, logging in if necessary."""
        if not self.token:
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        _base = self.base.replace("update","")
        """Generic request helper with auth and error handling."""
        url = f"{_base.rstrip('/')}/{endpoint.lstrip('/')}"
        merged_headers = self._auth_headers()
        if headers:
            merged_headers.update(headers)
        if params is None and self.pod_uuid:
            params = {"pod_uuid": self.pod_uuid}

        resp = requests.request(
            method, url, headers=merged_headers, params=params, timeout=30, **kwargs
        )
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return {"raw": resp.text}

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")

        resp.raise_for_status()
        return data

    # --------------------------------------------------
    # TEMPLATE METHODS
    # --------------------------------------------------
    def get_templates(self, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all available templates for the authenticated user."""
        templates = self._make_request("GET", "templates")

        if not templates:
            raise RuntimeError("No templates returned from API")

        print(f"📦 Fetched {len(templates)} templates from QuickPod API")

        if output_file:
            with open(output_file, "w") as f:
                json.dump(templates, f, indent=2)
            print(f"💾 Templates saved to {output_file}")

        # Display some useful info
        for t in templates[:10]:  # print first 10
            print(f" - {t.get('template_name')} ({t.get('template_uuid')})")

        return templates

    def get_template_by_name(self, template_name: str) -> Optional[str]:
        """Return template UUID by template name, or None if not found."""
        templates = self.get_templates()
        for template in templates:
            if template.get("template_name") == template_name:
                return template.get("template_uuid")
        return None

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    try:
        client = QuickPodClient()
        client.login()
        templates = client.get_templates(output_file="templates.json")
        print(f"\n✅ Successfully retrieved {len(templates)} templates.\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
