# salad_api.py
import os
import requests
import json
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

SALAD_API_KEY = os.getenv("SALAD_API_KEY")
SALAD_ORGANIZATION = os.getenv("SALAD_ORGANIZATION")
SALAD_PROJECT = os.getenv("SALAD_PROJECT")
IMAGE_PREFIX = os.getenv("SALAD_WORKER_IMAGE_PREFIX", "")
API_BASE = "https://api.salad.com/api/public"

HEADERS = {
    "Salad-Api-Key": SALAD_API_KEY,
    "accept": "application/json",
    "content-type": "application/json"
}

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(int(os.getenv("SALAD_RETRY_ATTEMPTS", "5"))))
def start_salad_worker(worker_name: str, image: str, gpu_classes: list, env: dict, cpu=2, memory=24576, port=8080):
    """
    Create a container on Salad. Returns the JSON response.
    """
    url = f"{API_BASE}/organizations/{os.getenv('SALAD_ORGANIZATION')}/projects/{os.getenv('SALAD_PROJECT')}/containers"
    payload = {
        "container": {
            "resources": {
                "cpu": cpu,
                "memory": memory,
                "gpu_classes": gpu_classes,
                "image": image
            },
            "restart_policy": "never",
            "networking": {
                "protocol": "http",
                "auth": True,
                "port": port
            },
            "name": worker_name,
            "replicas": 1
        },
        "env": env
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Salad API error {r.status_code}: {r.text}")
    resp = r.json()
    return resp

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
def delete_salad_worker(container_id: str):
    """
    Delete container by id. Adjust endpoint if necessary.
    """
    url = f"{API_BASE}/organizations/{os.getenv('SALAD_ORGANIZATION')}/projects/{os.getenv('SALAD_PROJECT')}/containers/{container_id}"
    r = requests.delete(url, headers=HEADERS, timeout=30)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Failed to delete container {container_id} : {r.status_code} {r.text}")
    return r.status_code
