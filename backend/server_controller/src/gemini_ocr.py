#!/usr/bin/env python3
"""Gemini 3.5 Flash Lite OCR for Extract Text (not a Kie model)."""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("gemini_ocr")

GEMINI_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-3.5-flash-lite")


def extract_text_from_image_url(image_url: str) -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()
    data = resp.content
    mime = resp.headers.get("Content-Type") or "image/jpeg"
    if ";" in mime:
        mime = mime.split(";", 1)[0].strip()
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Extract all readable text from this image. "
                            "Preserve line breaks. Return only the transcribed text."
                        )
                    },
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ]
    }
    r = requests.post(url, params={"key": api_key}, json=payload, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini OCR HTTP {r.status_code}: {r.text[:800]}")
    body = r.json()
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini OCR empty response: {body}")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [p.get("text") or "" for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RuntimeError("Gemini OCR returned no text")
    return text
