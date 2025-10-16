import requests
import os
import json

BASE = "https://genedit.mooo.com"

# Must be one of CONTROLLER_API_KEYS (not just SHARED_SECRET unless you set it there too)
API_KEY = os.getenv("CONTROLLER_API_KEY", "SECRETKEY")

def test_submit():
    payload = {
        "model": "sd15",
        "prompt": "A sunset over mountains",
        "params": {"timeout": 120}
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(f"{BASE}/submit", json=payload, headers=headers, verify=False)
    print("Status:", r.status_code)
    try:
        print("Response:", r.json())
        return r.json()
    except Exception:
        print("Raw Response:", r.text)
        return {}

if __name__ == "__main__":
    job = test_submit()
    print("Job ID:", job.get("job_id"))
