# moderation_client.py
"""
OpenAI Moderations client.

Separate from the request gate so tests can mock the HTTP call.
The API key is read from the environment (OPENAI_API_KEY) and is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("moderation_client")

MODERATION_URL = "https://api.openai.com/v1/moderations"
MODERATION_MODEL = "omni-moderation-latest"
DEFAULT_TIMEOUT_SECONDS = 5.0
# OpenAI has to download R2 objects, so image requests get a longer budget.
IMAGE_MODERATION_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5

# 429 and 5xx are retried. Other 4xx (auth, bad request) will not succeed on retry.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ModerationUnavailable(Exception):
    """Moderation could not be completed. Callers must fail closed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class ModerationVerdict:
    """Decision data derived from an OpenAI moderations response."""

    flagged: bool
    categories: List[str]
    scores: Dict[str, float] = field(default_factory=dict)


def parse_allow_moderation_flagged(raw: Optional[str]) -> bool:
    """
    Parse ALLOW_MODERATION_FLAGGED.

    Only the case-insensitive tokens ``true`` and ``false`` are accepted.
    Missing or unparseable values default to False (block flagged content).
    """
    if raw is None:
        return False
    value = raw.strip().lower()
    if value == "true":
        return True
    return False


def allow_flagged_from_env() -> bool:
    """Current ALLOW_MODERATION_FLAGGED value from the process environment."""
    return parse_allow_moderation_flagged(os.getenv("ALLOW_MODERATION_FLAGGED"))


def openai_api_key_from_env() -> str:
    """OPENAI_API_KEY from the process environment, stripped. Empty when unset."""
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def interpret_moderation_response(payload: Any) -> ModerationVerdict:
    """
    Reduce an OpenAI moderations body to a verdict.

    The request is flagged when any ``results[]`` entry has ``flagged == true``.
    Category names are included only when their value is strictly true
    (``null`` and ``false`` are not flagged). Scores are kept for logs only.
    """
    if not isinstance(payload, dict):
        raise ModerationUnavailable("invalid_response")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ModerationUnavailable("invalid_response")

    flagged = False
    categories: List[str] = []
    seen = set()
    scores: Dict[str, float] = {}

    for result in results:
        if not isinstance(result, dict):
            raise ModerationUnavailable("invalid_response")
        flag = result.get("flagged")
        if not isinstance(flag, bool):
            raise ModerationUnavailable("invalid_response")
        flagged = flagged or flag

        raw_categories = result.get("categories", {})
        if raw_categories is None:
            raw_categories = {}
        if not isinstance(raw_categories, dict):
            raise ModerationUnavailable("invalid_response")
        for name, value in raw_categories.items():
            if value is True and name not in seen:
                seen.add(name)
                categories.append(str(name))

        raw_scores = result.get("category_scores", {})
        if raw_scores is None:
            continue
        if not isinstance(raw_scores, dict):
            continue
        for name, value in raw_scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            current = scores.get(str(name))
            if current is None or number > current:
                scores[str(name)] = number

    categories.sort()
    return ModerationVerdict(flagged=flagged, categories=categories, scores=scores)


class ModerationClient:
    """POST /v1/moderations with a short timeout and limited retries."""

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_SECONDS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.api_key = api_key or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._transport = transport

    async def moderate(self, moderation_input: Any) -> Dict[str, Any]:
        """
        Classify ``moderation_input`` (string, list of strings, or multimodal parts).

        Returns the parsed JSON body. Raises ModerationUnavailable on timeout,
        transport errors, retry-exhausted 429/5xx, or an invalid body.
        The API key and request content are not included in the exception.
        """
        if not self.api_key:
            raise ModerationUnavailable("missing_api_key")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {"model": MODERATION_MODEL, "input": moderation_input}
        reason = "transport_error"
        attempts = self.max_retries + 1

        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            for attempt in range(attempts):
                try:
                    response = await client.post(MODERATION_URL, headers=headers, json=body)
                except httpx.TimeoutException:
                    reason = "timeout"
                except httpx.TransportError:
                    reason = "transport_error"
                else:
                    if response.status_code == 200:
                        try:
                            payload = response.json()
                        except Exception:
                            raise ModerationUnavailable("invalid_response")
                        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                            raise ModerationUnavailable("invalid_response")
                        return payload
                    reason = f"http_{response.status_code}"
                    if response.status_code not in RETRYABLE_STATUS:
                        raise ModerationUnavailable(reason)

                if attempt >= self.max_retries:
                    raise ModerationUnavailable(reason)

                # Do not log the body or the Authorization header.
                logger.warning(
                    "openai_moderation_retry attempt=%s reason=%s",
                    attempt + 1,
                    reason,
                )
                delay = self.backoff_base * (2 ** attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise ModerationUnavailable(reason)
