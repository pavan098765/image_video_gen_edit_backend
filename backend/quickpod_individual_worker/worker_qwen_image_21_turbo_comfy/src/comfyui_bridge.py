#!/usr/bin/env python3
"""
comfyui_bridge.py
=================

Manages a ComfyUI server as a subprocess and provides a thin Python client
to queue prompts, poll for completion, and retrieve outputs.

ComfyUI listens on 0.0.0.0:8188 so the Civitai LTX-2.5 workflow UI can be
opened on the pod. FastAPI remains on port 8000.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("comfyui_bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/app/comfyui")
COMFYUI_LISTEN = os.environ.get("COMFYUI_LISTEN", "0.0.0.0")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = os.environ.get("COMFYUI_URL", f"http://127.0.0.1:{COMFYUI_PORT}")

# Max time (seconds) to wait for ComfyUI to become ready on startup
COMFYUI_STARTUP_TIMEOUT = int(os.environ.get("COMFYUI_STARTUP_TIMEOUT", "7200"))

# Max time (seconds) to wait for a single prompt to complete
COMFYUI_PROMPT_TIMEOUT = int(os.environ.get("COMFYUI_PROMPT_TIMEOUT", "3600"))

# Polling interval when waiting for prompt completion
POLL_INTERVAL = 2.0


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_comfyui_process: Optional[subprocess.Popen] = None

# Cards that report more than 16 GiB can hold the text encoder, unet, and VAE together.
_HIGHVRAM_MIN_MIB = 16 * 1024


def _gpu_vram_mib() -> Optional[int]:
    """Total memory of GPU 0 in MiB, or None when it cannot be read."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        return int(float(first))
    except ValueError:
        return None


def _vram_launch_args() -> List[str]:
    """Use --highvram only when the card is larger than 16 GiB."""
    mib = _gpu_vram_mib()
    if mib is None:
        logger.info("GPU VRAM was not detected; starting ComfyUI in normal VRAM mode")
        return []
    if mib > _HIGHVRAM_MIN_MIB:
        logger.info(
            "GPU VRAM %d MiB is above 16 GiB; starting ComfyUI with --highvram",
            mib,
        )
        return ["--highvram"]
    logger.info(
        "GPU VRAM %d MiB is 16 GiB or less; starting ComfyUI in normal VRAM mode",
        mib,
    )
    return []


def start_server() -> subprocess.Popen:
    """
    Launch ComfyUI as a headless background process.

    Returns the Popen handle so the caller can manage it.
    """
    global _comfyui_process

    if _comfyui_process is not None and _comfyui_process.poll() is None:
        logger.info("ComfyUI server already running (pid %d)", _comfyui_process.pid)
        return _comfyui_process

    main_py = os.path.join(COMFYUI_DIR, "main.py")
    if not os.path.isfile(main_py):
        raise FileNotFoundError(f"ComfyUI main.py not found: {main_py}")

    import sys as _sys
    python_bin = _sys.executable or "python3"

    cmd = [
        python_bin, main_py,
        "--listen", COMFYUI_LISTEN,
        "--port", str(COMFYUI_PORT),
        "--disable-auto-launch",
        *_vram_launch_args(),
    ]

    comfyui_log_path = os.path.join(COMFYUI_DIR, "comfyui_server.log")
    _comfyui_log_file = open(comfyui_log_path, "w")

    logger.info("Starting ComfyUI server: %s", " ".join(cmd))
    logger.info("ComfyUI logs: %s", comfyui_log_path)
    _comfyui_process = subprocess.Popen(
        cmd,
        cwd=COMFYUI_DIR,
        stdout=_comfyui_log_file,
        stderr=_comfyui_log_file,
    )
    logger.info("ComfyUI server started (pid %d)", _comfyui_process.pid)
    return _comfyui_process


def get_comfyui_log_path() -> str:
    """Return the path to the ComfyUI server log file (for tail/debug)."""
    return os.path.join(COMFYUI_DIR, "comfyui_server.log")


def wait_until_ready(timeout: int = COMFYUI_STARTUP_TIMEOUT) -> bool:
    """
    Block until ComfyUI responds to ``GET /system_stats``.

    Returns True if ready, raises TimeoutError otherwise.
    """
    deadline = time.time() + timeout
    logger.info("Waiting for ComfyUI to become ready (timeout=%ds)...", timeout)

    while time.time() < deadline:
        try:
            resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
            if resp.status_code == 200:
                logger.info("ComfyUI is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        time.sleep(2)

    raise TimeoutError(
        f"ComfyUI did not become ready within {timeout}s. "
        f"Check ComfyUI logs for errors."
    )


def stop_server() -> None:
    """Terminate the ComfyUI subprocess gracefully."""
    global _comfyui_process
    if _comfyui_process is None:
        return
    if _comfyui_process.poll() is None:
        logger.info("Stopping ComfyUI server (pid %d)...", _comfyui_process.pid)
        _comfyui_process.terminate()
        try:
            _comfyui_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("ComfyUI did not stop in time, killing...")
            _comfyui_process.kill()
            _comfyui_process.wait(timeout=5)
    _comfyui_process = None
    logger.info("ComfyUI server stopped.")


def health_check() -> bool:
    """Quick non-blocking check: is the ComfyUI server responding?"""
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def is_server_running() -> bool:
    """Check if the ComfyUI subprocess is alive."""
    return _comfyui_process is not None and _comfyui_process.poll() is None


# ---------------------------------------------------------------------------
# Prompt API
# ---------------------------------------------------------------------------


def queue_prompt(api_workflow: dict, client_id: Optional[str] = None) -> str:
    """Submit an API-format workflow to ComfyUI."""
    if client_id is None:
        client_id = str(uuid.uuid4())

    payload = {
        "prompt": api_workflow,
        "client_id": client_id,
    }

    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        error_text = resp.text
        try:
            error_data = resp.json()
            logger.error("ComfyUI prompt rejection - full response: %s", json.dumps(error_data, indent=2)[:5000])
            if "node_errors" in error_data and error_data["node_errors"]:
                error_text = json.dumps(error_data["node_errors"], indent=2)
            elif "error" in error_data and error_data["error"]:
                error_text = json.dumps(error_data["error"], indent=2)
            else:
                error_text = json.dumps(error_data, indent=2)[:3000]
        except Exception:
            pass
        raise RuntimeError(
            f"ComfyUI rejected prompt (HTTP {resp.status_code}): {error_text}"
        )

    data = resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {data}")

    logger.info("Prompt queued: prompt_id=%s", prompt_id)
    return prompt_id


def wait_for_completion(
    prompt_id: str,
    timeout: int = COMFYUI_PROMPT_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> dict:
    """Poll ``GET /history/{prompt_id}`` until the prompt finishes."""
    deadline = time.time() + timeout
    logger.info("Waiting for prompt %s to complete (timeout=%ds)...", prompt_id, timeout)

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=10,
            )
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    entry = history[prompt_id]

                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        raise RuntimeError(
                            f"ComfyUI prompt {prompt_id} failed: {messages}"
                        )

                    if status.get("status_str") == "success" or entry.get("outputs"):
                        logger.info("Prompt %s completed successfully.", prompt_id)
                        return entry
        except requests.exceptions.ConnectionError:
            logger.warning("ComfyUI connection lost while polling, retrying...")
        except requests.exceptions.Timeout:
            pass

        time.sleep(poll_interval)

    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def get_output_files(history_entry: dict) -> List[Dict[str, Any]]:
    """Extract output file info from a completed history entry."""
    outputs = history_entry.get("outputs", {})
    files: List[Dict[str, Any]] = []

    for node_id, node_output in outputs.items():
        for key in ("gifs", "images", "videos"):
            items = node_output.get(key, [])
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    files.append({
                        "node_id": node_id,
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "format": item.get("format", ""),
                    })

    return files


def get_output_file_path(file_info: dict) -> str:
    """Resolve a ComfyUI output file info dict to an absolute file path."""
    output_type = file_info.get("type", "output")
    subfolder = file_info.get("subfolder", "")
    filename = file_info["filename"]

    if output_type == "output":
        base = os.path.join(COMFYUI_DIR, "output")
    elif output_type == "temp":
        base = os.path.join(COMFYUI_DIR, "temp")
    elif output_type == "input":
        base = os.path.join(COMFYUI_DIR, "input")
    else:
        base = os.path.join(COMFYUI_DIR, "output")

    full_path = os.path.join(base, subfolder, filename)
    return os.path.abspath(full_path)


# ---------------------------------------------------------------------------
# Input management
# ---------------------------------------------------------------------------


def save_image_to_input(image_bytes: bytes, filename: str) -> str:
    """Save raw image bytes to ComfyUI's ``input/`` directory."""
    return save_bytes_to_input(image_bytes, filename)


def save_bytes_to_input(data: bytes, filename: str) -> str:
    """Save raw bytes (image/audio/video) to ComfyUI's ``input/`` directory."""
    input_dir = os.path.join(COMFYUI_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    out_path = os.path.join(input_dir, filename)
    with open(out_path, "wb") as f:
        f.write(data)
    logger.info("Saved input file: %s (%d bytes)", out_path, len(data))
    return filename


def interrupt() -> None:
    """Ask ComfyUI to stop the current prompt (personal-queue preemption)."""
    try:
        resp = requests.post(f"{COMFYUI_URL}/interrupt", timeout=10)
        logger.info("ComfyUI interrupt HTTP %s", resp.status_code)
    except Exception:
        logger.exception("ComfyUI interrupt failed")


def upload_image_to_comfyui(image_bytes: bytes, filename: str) -> str:
    """Upload an image to ComfyUI via ``/upload/image``."""
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to upload image to ComfyUI: {resp.text}")
    data = resp.json()
    return data.get("name", filename)


def run_prompt(
    api_workflow: dict,
    timeout: int = COMFYUI_PROMPT_TIMEOUT,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Queue a prompt, wait for completion, and return the output files."""
    prompt_id = queue_prompt(api_workflow)
    history = wait_for_completion(prompt_id, timeout=timeout)
    output_files = get_output_files(history)
    return prompt_id, output_files
