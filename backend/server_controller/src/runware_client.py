#!/usr/bin/env python3
"""Runware REST client: one /v1 task array, async delivery, getResponse polling."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("runware_client")

RUNWARE_API = os.getenv("RUNWARE_API_BASE", "https://api.runware.ai/v1").rstrip("/")

# FLUX.2 klein: width and height together, multiples of 16, 128–2048.
FLUX_SIZES: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
    "21:9": (1344, 576),
}

# Muse rejects every pair except this 2K set. resolution cannot be mixed with width/height.
MUSE_SIZES: Dict[str, Tuple[int, int]] = {
    "1:1": (1600, 1600),
    "3:2": (1920, 1280),
    "2:3": (1280, 1920),
    "4:3": (1792, 1344),
    "3:4": (1344, 1792),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "21:9": (2352, 1008),
}

SIZE_TABLES = {"flux": FLUX_SIZES, "muse": MUSE_SIZES}

MEDIA_URL_KEYS = ("imageURL", "audioURL", "videoURL")


def _api_key() -> str:
    return (os.getenv("RUNWARE_API_KEY") or "").strip()


def webhook_token() -> str:
    key = _api_key()
    if not key:
        return ""
    digest = hmac.new(key.encode("utf-8"), b"runware-webhook", hashlib.sha256).hexdigest()
    return digest[:32]


def verify_webhook_token(token: str) -> bool:
    expected = webhook_token()
    if not expected or not token:
        return False
    return hmac.compare_digest(expected, token.strip())


def webhook_url() -> Optional[str]:
    public = (os.getenv("CONTROLLER_PUBLIC_URL") or os.getenv("CONTROLLER_URL") or "").rstrip("/")
    token = webhook_token()
    if not public or not token:
        return None
    return f"{public}/runware/callback?token={token}"


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("true", "1", "yes", "on"):
            return True
        if raw in ("false", "0", "no", "off"):
            return False
    return None


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith("{{") and text.endswith("}}"):
        return ""
    return text


def _http_urls(value: Any) -> List[str]:
    items = value if isinstance(value, list) else [value]
    urls: List[str] = []
    for item in items:
        if isinstance(item, str):
            src = item.strip()
            if src.startswith("http://") or src.startswith("https://"):
                urls.append(src)
    return urls


def snap_size(width: Any, height: Any, table: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Pick the allowed frame whose aspect is closest to the requested pixels."""
    w = _as_int(width)
    h = _as_int(height)
    if not w or not h or w <= 0 or h <= 0:
        return table["1:1"]
    target = w / h
    best = table["1:1"]
    best_diff = abs((best[0] / best[1]) - target)
    for pair in table.values():
        diff = abs((pair[0] / pair[1]) - target)
        if diff < best_diff:
            best = pair
            best_diff = diff
    return best


def choose_size(args: Dict[str, Any], table: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    aspect = _clean_text(args.get("aspect_ratio")).replace(" ", "")
    if aspect in table:
        return table[aspect]
    return snap_size(args.get("width"), args.get("height"), table)


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            cleaned = _drop_empty(item)
            if cleaned in (None, "", [], {}):
                continue
            out[key] = cleaned
        return out
    if isinstance(value, list):
        return [item for item in (_drop_empty(v) for v in value) if item not in (None, "", [], {})]
    return value


def assemble_task(
    cfg: Dict[str, Any],
    args: Dict[str, Any],
    task_uuid: str,
    webhook: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one Runware task from a model config and mapped arguments."""
    provider = cfg.get("provider") or {}
    spec = cfg.get("runware") or {}
    profile = str(spec.get("profile") or "")
    task: Dict[str, Any] = {
        "taskType": spec.get("task_type"),
        "taskUUID": task_uuid,
        "model": provider.get("model_id"),
        "outputType": "URL",
        "outputFormat": spec.get("output_format") or "PNG",
        "includeCost": True,
        "deliveryMethod": "async",
    }
    # Upscale tasks return one image. Runware rejects numberResults on them.
    if profile not in ("upscale_p", "upscale_esrgan"):
        task["numberResults"] = 1
    if webhook:
        task["webhookURL"] = webhook

    if profile in ("flux_image", "muse_image"):
        _fill_image(task, args, profile)
    elif profile == "upscale_p":
        _fill_p_upscale(task, args)
    elif profile == "upscale_esrgan":
        _fill_esrgan(task, args)
    elif profile == "speech_xai":
        _fill_xai_speech(task, args)
    elif profile == "speech_design":
        _fill_design_speech(task, args)
    elif profile == "speech_clone":
        _fill_clone_speech(task, args)
    elif profile == "music_ace":
        _fill_ace(task, args)
    elif profile == "music_minimax":
        _fill_minimax(task, args)
    else:
        raise ValueError(f"Unknown Runware profile: {profile}")

    if not task.get("taskType") or not task.get("model"):
        raise ValueError("Runware taskType or model_id missing")
    return _drop_empty(task)


def _fill_image(task: Dict[str, Any], args: Dict[str, Any], profile: str) -> None:
    prompt = _clean_text(args.get("positivePrompt") or args.get("prompt"))
    if not prompt:
        raise ValueError("Prompt is required")
    task["positivePrompt"] = prompt[:10000 if profile == "flux_image" else 200000]
    raw_refs = args.get("referenceImages")
    refs = _http_urls(raw_refs)
    if "referenceImages" in args and not refs:
        raise ValueError("Reference image must be a downloadable URL")
    table = MUSE_SIZES if profile == "muse_image" else FLUX_SIZES
    if profile == "flux_image":
        refs = refs[:4]
    else:
        refs = refs[:10]
    omit_size = profile == "muse_image" and bool(refs)
    if refs:
        task["inputs"] = {"referenceImages": refs}
    if not omit_size:
        width, height = choose_size(args, table)
        task["width"] = width
        task["height"] = height
    if profile == "flux_image":
        task["steps"] = 4
    if profile == "muse_image":
        settings: Dict[str, Any] = {}
        thinking = _clean_text(args.get("thinkingLevel")).lower()
        if thinking in ("high", "low"):
            settings["thinkingLevel"] = thinking
        for key in ("webSearch", "imageSearch", "shell"):
            flag = _as_bool(args.get(key))
            if flag is not None:
                settings[key] = flag
        moderation = _clean_text(args.get("moderation")).lower()
        if moderation in ("auto", "low"):
            settings["moderation"] = moderation
        if settings:
            task["settings"] = settings


def _fill_p_upscale(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    image = _http_urls(args.get("image"))
    if not image:
        raise ValueError("An image is required")
    factor = _as_int(args.get("upscaleFactor")) or 2
    task["upscaleFactor"] = max(2, min(16, factor))
    task["inputs"] = {"image": image[0]}
    settings: Dict[str, Any] = {}
    if _as_bool(args.get("enhanceDetails")) is True:
        settings["enhanceDetails"] = True
    if _as_bool(args.get("realism")) is True:
        settings["realism"] = True
    if settings:
        task["settings"] = settings


def _fill_esrgan(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    image = _http_urls(args.get("image"))
    if not image:
        raise ValueError("An image is required")
    factor = _as_int(args.get("upscaleFactor")) or 2
    task["upscaleFactor"] = 4 if factor >= 4 else 2
    task["inputs"] = {"image": image[0]}


def _fill_xai_speech(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    text = _clean_text(args.get("text") or args.get("prompt"))
    if not text:
        raise ValueError("Speech text is required")
    voice = _clean_text(args.get("voice")) or "eve"
    language = _clean_text(args.get("language")) or "en"
    task["speech"] = {"text": text, "voice": voice, "language": language}


def _fill_design_speech(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    script = _clean_text(args.get("text") or args.get("prompt"))
    description = _clean_text(args.get("positivePrompt") or args.get("voice_description"))
    if not script:
        raise ValueError("Speech text is required")
    if not description:
        raise ValueError("A voice description is required")
    language = _clean_text(args.get("language")) or "Auto"
    task["positivePrompt"] = description[:2000]
    task["speech"] = {
        "text": script[:2000],
        "voice": "design",
        "language": language,
    }


def _fill_clone_speech(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    script = _clean_text(args.get("text") or args.get("prompt"))
    audio = _http_urls(args.get("audio"))
    if not script:
        raise ValueError("Speech text is required")
    if not audio:
        raise ValueError("A reference voice recording is required")
    language = _clean_text(args.get("language")) or "Auto"
    task["inputs"] = {"audio": audio[0]}
    task["speech"] = {
        "text": script[:2000],
        "voice": "clone",
        "language": language,
    }
    # ICL mode requires a transcript of the sample. The app only sends the
    # recording, so use speaker embedding unless a transcript is provided.
    transcript = _clean_text(args.get("transcript"))
    if transcript:
        task["settings"] = {"transcript": transcript[:2000], "xVectorOnly": False}
    else:
        task["settings"] = {"xVectorOnly": True}


def _fill_ace(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    prompt = _clean_text(args.get("positivePrompt") or args.get("prompt"))
    if len(prompt) < 2:
        raise ValueError("Describe the music in at least a few words")
    task["positivePrompt"] = prompt[:3000]
    audio_urls = _http_urls(args.get("audio"))
    if audio_urls:
        task["inputs"] = {"audio": audio_urls[0]}
        try:
            strength = float(args.get("strength"))
        except (TypeError, ValueError):
            strength = 0.55
        task["strength"] = round(max(0.0, min(1.0, strength)), 2)
    else:
        duration = _as_int(args.get("duration")) or 60
        task["duration"] = max(30, min(240, duration))
    task["steps"] = 50
    negative = _clean_text(args.get("negativePrompt"))
    if len(negative) >= 2:
        task["negativePrompt"] = negative[:3000]
        task["CFGScale"] = 4
    settings: Dict[str, Any] = {}
    lyrics = _clean_text(args.get("lyrics"))
    if len(lyrics) >= 10:
        settings["lyrics"] = lyrics[:3000]
    bpm = _as_int(args.get("bpm"))
    if bpm is not None:
        settings["bpm"] = max(30, min(300, bpm))
    key_scale = _clean_text(args.get("keyScale"))
    if key_scale and key_scale.lower() != "auto":
        settings["keyScale"] = key_scale[:40]
    meter = _as_int(args.get("timeSignature"))
    if meter in (2, 3, 4, 6):
        settings["timeSignature"] = meter
    language = _clean_text(args.get("vocalLanguage"))
    if language and language.lower() not in ("auto", ""):
        settings["vocalLanguage"] = "unknown" if language.lower() == "unknown" else language[:8]
    if audio_urls:
        start = _as_int(args.get("repaintingStart"))
        end = _as_int(args.get("repaintingEnd"))
        if start is not None and end is not None and end > start:
            settings["repaintingStart"] = max(-300, min(300, start))
            settings["repaintingEnd"] = max(0, min(300, end))
    if settings:
        task["settings"] = settings


def _fill_minimax(task: Dict[str, Any], args: Dict[str, Any]) -> None:
    prompt = _clean_text(args.get("positivePrompt") or args.get("prompt"))
    lyrics = _clean_text(args.get("lyrics"))
    instrumental = _as_bool(args.get("instrumental"))
    optimizer = _as_bool(args.get("lyricsOptimizer")) is True
    if instrumental is True:
        lyrics = ""
        optimizer = False
    if not lyrics and not optimizer and instrumental is not False:
        instrumental = True
    if instrumental is True:
        if len(prompt) < 1:
            raise ValueError("Describe the track")
        task["positivePrompt"] = prompt[:2000]
        task["settings"] = {"instrumental": True}
    elif lyrics:
        if not prompt:
            prompt = "Vocal song matching the lyrics"
        task["positivePrompt"] = prompt[:2000]
        task["settings"] = {
            "lyrics": lyrics[:3500],
            "instrumental": False,
            "lyricsOptimizer": optimizer,
        }
    elif optimizer and prompt:
        task["positivePrompt"] = prompt[:2000]
        task["settings"] = {"lyricsOptimizer": True, "instrumental": False}
    else:
        raise ValueError("Add lyrics, or turn vocals off")


def _walk_tasks(payload: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(_walk_tasks(item))
        return found
    if not isinstance(payload, dict):
        return found
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                found.append(item)
        return found
    if any(key in payload for key in ("taskUUID", "taskUuid", *MEDIA_URL_KEYS, "status")):
        found.append(payload)
    return found


def _error_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("errors"), list):
        return [item for item in payload["errors"] if isinstance(item, dict)]
    return []


def media_url(payload: Any) -> Optional[str]:
    for task in _walk_tasks(payload):
        for key in MEDIA_URL_KEYS:
            value = task.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        files = task.get("outputs")
        if isinstance(files, dict):
            rows = files.get("files")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and isinstance(row.get("url"), str):
                        return row["url"]
    return None


def task_uuid_of(payload: Any) -> Optional[str]:
    for task in _walk_tasks(payload):
        for key in ("taskUUID", "taskUuid"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for err in _error_items(payload):
        value = err.get("taskUUID") or err.get("taskUuid")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def failure_message(payload: Any) -> Optional[str]:
    if media_url(payload):
        return None
    messages: List[str] = []
    for err in _error_items(payload):
        text = err.get("message") or err.get("code")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    for task in _walk_tasks(payload):
        if str(task.get("status") or "").lower() != "error":
            continue
        nested = task.get("error")
        if isinstance(nested, dict):
            text = nested.get("message") or nested.get("code")
            if isinstance(text, str) and text.strip():
                messages.append(text.strip())
        elif isinstance(task.get("message"), str):
            messages.append(task["message"].strip())
    if not messages:
        return None
    return messages[0][:500]


def is_processing(payload: Any) -> bool:
    if media_url(payload) or failure_message(payload):
        return False
    tasks = _walk_tasks(payload)
    if not tasks and not _error_items(payload):
        return False
    if any(str(task.get("status") or "").lower() == "processing" for task in tasks):
        return True
    # Async ack echoes the task UUID and has no media yet.
    return task_uuid_of(payload) is not None


def _post(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    key = _api_key()
    if not key:
        raise RuntimeError("RUNWARE_API_KEY is not configured")
    resp = requests.post(
        RUNWARE_API,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=tasks,
        timeout=90,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"errors": [{"message": (resp.text or "")[:500]}]}
    if not isinstance(body, dict):
        body = {"data": body if isinstance(body, list) else []}
    if resp.status_code >= 400:
        raise RuntimeError(failure_message(body) or f"Runware HTTP {resp.status_code}")
    return body


def submit_task(task: Dict[str, Any]) -> Dict[str, Any]:
    body = _post([task])
    err = failure_message(body)
    if err and not media_url(body):
        raise RuntimeError(err)
    return body


def get_response(task_uuid: str) -> Dict[str, Any]:
    return _post([{"taskType": "getResponse", "taskUUID": task_uuid}])


def new_task_uuid() -> str:
    return str(uuid.uuid4())
