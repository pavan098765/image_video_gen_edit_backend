# content_moderation.py
"""
Inbound content-moderation gate for frontend requests.

Runs before the route handler. Worker callbacks, provider webhooks, and
no-content operational routes are on EXEMPT_PATH_PREFIXES. Every other
path is in scope: if the body or query has user text, an image URL, or an
uploaded image object key, the OpenAI moderations API is called. Uploaded
images are presigned for a few minutes so OpenAI can fetch them; the bytes
are not downloaded here. Requests with nothing to moderate are allowed
without an API call.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from moderation_client import (
    IMAGE_MODERATION_TIMEOUT_SECONDS,
    ModerationClient,
    ModerationUnavailable,
    ModerationVerdict,
    allow_flagged_from_env,
    interpret_moderation_response,
    openai_api_key_from_env,
)

logger = logging.getLogger("content_moderation")

MAX_IMAGE_BYTES = 20 * 1024 * 1024
# One moderations call stays small; larger uploads are split and run together.
IMAGE_BATCH_SIZE = 10
R2_PRESIGN_SECONDS = 180

FLAGGED_MESSAGE = "Your request was blocked because it may violate our content policy."
UNAVAILABLE_MESSAGE = "Content moderation is temporarily unavailable. Please try again."
IMAGE_TOO_LARGE_MESSAGE = "An image in the request exceeds the 20 MB moderation limit."

# Routes that are not end-user content submissions. New frontend routes are
# covered automatically; add a prefix here only for worker, webhook, or
# no-content operational paths that must not depend on the moderation API.
EXEMPT_PATH_PREFIXES: Sequence[str] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/register_worker",
    "/heartbeat",
    "/job_started",
    "/job_complete",
    "/personal/pull",
    "/personal/complete",
    "/personal/pause",
    "/kie/callback",
    "/runware/callback",
    "/ads",
    "/studio/internal",
)

# Normalized (lowercase, alphanumeric only) field names that carry user text.
# IDs, enums, tokens, and R2 keys are intentionally absent.
_TEXT_KEYS = frozenset({
    "prompt",
    "negativeprompt",
    "positiveprompt",
    "text",
    "message",
    "caption",
    "description",
    "lyrics",
    "lyric",
    "title",
    "style",
    "topic",
    "audience",
    "script",
    "scriptline",
    "copy",
    "query",
    "instruction",
    "instructions",
    "content",
    "body",
    "comment",
    "negative",
    "negativetags",
    "tags",
    "idea",
    "subject",
    "story",
    "dialogue",
    "transcript",
    "subtitle",
    "alt",
    "alttext",
    "speechtext",
    "userprompt",
    "customprompt",
    "systemprompt",
    "usermessage",
    "question",
    "bio",
})

_IMAGE_KEYS = frozenset({
    "image",
    "images",
    "imageurl",
    "imageurls",
    "img",
    "initimage",
    "inputimage",
    "refimage",
    "photo",
    "photos",
    "picture",
    "pictures",
    "thumbnail",
    "sourceimage",
    "referenceimage",
    "styleimage",
    "faceimage",
    "controlimage",
    "maskimage",
    "mask",
    "backgroundimage",
    "foregroundimage",
})

_IMAGE_EXT_RE = re.compile(
    r"\.(png|jpe?g|gif|webp|bmp|heic|heif)(?:$|[?#])",
    re.IGNORECASE,
)
_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]")


class R2ImageTooLarge(Exception):
    """An uploaded object is over the moderation size limit."""


@dataclass
class ExtractedContent:
    texts: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    image_r2_keys: List[str] = field(default_factory=list)
    oversized: bool = False


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (key or "").lower())


def is_text_key(norm: str) -> bool:
    return norm in _TEXT_KEYS


def is_image_key(norm: str) -> bool:
    if not norm or "r2" in norm:
        return False
    if norm in _IMAGE_KEYS:
        return True
    return norm.endswith("imageurl") or norm.endswith("imageurls")


def is_image_r2_key_name(norm: str) -> bool:
    """Object keys for uploaded pictures. Masks, video, and audio are not images."""
    if "r2" not in norm:
        return False
    if any(part in norm for part in ("mask", "video", "audio", "latent", "result")):
        return False
    if "image" in norm:
        return True
    # describe_asset sends the uploaded image as input_r2_key.
    return norm == "inputr2key"


def is_exempt_path(path: str, prefixes: Sequence[str] = EXEMPT_PATH_PREFIXES) -> bool:
    """True when ``path`` is exactly a prefix or a child of one."""
    cleaned = (path or "").split("?", 1)[0].rstrip("/").lower() or "/"
    for prefix in prefixes:
        pref = prefix.rstrip("/").lower() or "/"
        if cleaned == pref or cleaned.startswith(pref + "/"):
            return True
    return False


def _is_http_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def _data_image_payload(value: str) -> Optional[str]:
    """Return the base64 payload of a data:image URL, or None."""
    lower = value.lower()
    if not lower.startswith("data:image/"):
        return None
    marker = ";base64,"
    idx = lower.find(marker)
    if idx == -1:
        return None
    return value[idx + len(marker):]


def _decoded_base64_size(b64: str) -> Optional[int]:
    compact = "".join(b64.split())
    if not compact or len(compact) % 4 == 1:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/]+=*", compact):
        return None
    padding = compact.count("=")
    if padding > 2:
        return None
    return (len(compact) * 3) // 4 - padding


def _looks_like_image_url(value: str) -> bool:
    return _is_http_url(value) and _IMAGE_EXT_RE.search(value) is not None


def extract_moderatable_content(
    data: Any,
    *,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> ExtractedContent:
    """
    Collect user text, image URLs / data URLs, and uploaded image object keys.

    Field names that are ids, enums, or tokens are ignored. Mask, video, and
    audio object keys are ignored. Inline images whose decoded size exceeds
    ``max_image_bytes`` set ``oversized`` and are not included in the API payload.
    """
    texts: List[str] = []
    images: List[str] = []
    image_r2_keys: List[str] = []
    seen_text = set()
    seen_images = set()
    seen_r2 = set()
    oversized = False

    def add_text(value: str) -> None:
        text = value.strip()
        if not text or text in seen_text:
            return
        seen_text.add(text)
        texts.append(text)

    def add_image(value: str) -> None:
        nonlocal oversized
        url = value.strip()
        if not url or url in seen_images:
            return
        payload = _data_image_payload(url)
        if payload is not None:
            size = _decoded_base64_size(payload)
            if size is None:
                return
            if size > max_image_bytes:
                oversized = True
                return
            seen_images.add(url)
            images.append(url)
            return
        if _is_http_url(url):
            seen_images.add(url)
            images.append(url)

    def add_r2_key(value: str) -> None:
        key = value.strip()
        if not key or key in seen_r2:
            return
        seen_r2.add(key)
        image_r2_keys.append(key)

    def consider(key: Optional[str], value: str) -> None:
        raw = value.strip()
        if not raw:
            return
        if _data_image_payload(raw) is not None:
            add_image(raw)
            return
        norm = _norm_key(key or "")
        if is_image_r2_key_name(norm):
            if _is_http_url(raw):
                add_image(raw)
            else:
                add_r2_key(raw)
            return
        if is_image_key(norm):
            if _is_http_url(raw):
                add_image(raw)
            return
        if is_text_key(norm):
            if _looks_like_image_url(raw):
                add_image(raw)
                return
            add_text(raw)

    def walk(node: Any, key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, str(child_key))
            return
        if isinstance(node, list):
            for child in node:
                walk(child, key)
            return
        if isinstance(node, str):
            consider(key, node)

    walk(data)
    return ExtractedContent(
        texts=texts,
        images=images,
        image_r2_keys=image_r2_keys,
        oversized=oversized,
    )


def build_moderation_input(extracted: ExtractedContent) -> Any:
    """
    Build the OpenAI ``input`` value.

    A single text becomes a string. Several texts become a list of strings.
    Any image forces the multimodal part list. Returns None when there is
    nothing to send.
    """
    if not extracted.texts and not extracted.images:
        return None
    if extracted.images:
        parts: List[dict] = [{"type": "text", "text": text} for text in extracted.texts]
        parts.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in extracted.images
        )
        return parts
    if len(extracted.texts) == 1:
        return extracted.texts[0]
    return list(extracted.texts)


def moderation_inputs(extracted: ExtractedContent, batch_size: int = IMAGE_BATCH_SIZE) -> List[Any]:
    """One OpenAI input, or several chunks when there are more than ``batch_size`` images."""
    if not extracted.images:
        single = build_moderation_input(extracted)
        return [] if single is None else [single]
    if len(extracted.images) <= batch_size:
        built = build_moderation_input(extracted)
        return [] if built is None else [built]
    batches: List[Any] = []
    for start in range(0, len(extracted.images), batch_size):
        part = ExtractedContent(
            texts=list(extracted.texts),
            images=extracted.images[start:start + batch_size],
        )
        built = build_moderation_input(part)
        if built is not None:
            batches.append(built)
    return batches


async def _moderate_inputs(client, inputs: Sequence[Any]) -> List[ModerationVerdict]:
    async def one(item: Any) -> ModerationVerdict:
        raw = await client.moderate(item)
        return interpret_moderation_response(raw)

    if len(inputs) == 1:
        return [await one(inputs[0])]
    return list(await asyncio.gather(*(one(item) for item in inputs)))


def merge_verdicts(verdicts: Sequence[ModerationVerdict]) -> ModerationVerdict:
    """Flagged if any batch is flagged. Categories and scores are combined."""
    flagged = False
    categories: List[str] = []
    seen = set()
    scores: dict = {}
    for verdict in verdicts:
        flagged = flagged or verdict.flagged
        for name in verdict.categories:
            if name not in seen:
                seen.add(name)
                categories.append(name)
        for name, score in verdict.scores.items():
            current = scores.get(name)
            if current is None or score > current:
                scores[name] = score
    categories.sort()
    return ModerationVerdict(flagged=flagged, categories=categories, scores=scores)


def _valid_object_key(key: str) -> bool:
    if not key or len(key) > 1024:
        return False
    if key.startswith("/") or "\\" in key or ".." in key:
        return False
    return not any(ord(ch) < 32 for ch in key)


def _r2_client_and_bucket():
    """S3 client and bucket from app config. Both are None when R2 is not configured."""
    from config import ConfigManager

    manager = ConfigManager.get_instance()
    return manager.get_s3_client(), manager.config.r2.bucket


def _head_and_sign(client, bucket: str, key: str, max_image_bytes: int, expires_in: int) -> str:
    """HEAD the object, then sign a GET URL. Exceptions carry no key or URL."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        raise ModerationUnavailable("r2_unavailable") from None
    size = head.get("ContentLength") if isinstance(head, dict) else None
    if isinstance(size, int) and size > max_image_bytes:
        raise R2ImageTooLarge()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception:
        raise ModerationUnavailable("r2_unavailable") from None
    if not isinstance(url, str) or not _is_http_url(url):
        raise ModerationUnavailable("r2_unavailable")
    return url


async def resolve_r2_image_urls(
    keys: Sequence[str],
    *,
    max_image_bytes: int,
    client=None,
    bucket: Optional[str] = None,
    expires_in: int = R2_PRESIGN_SECONDS,
) -> List[str]:
    """
    Presign uploaded images so OpenAI can fetch them.

    Does not download the bytes. Raises R2ImageTooLarge or ModerationUnavailable.
    """
    unique: List[str] = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        if not _valid_object_key(key):
            raise ModerationUnavailable("r2_unavailable")
        unique.append(key)
    if not unique:
        return []

    if client is None:
        client, bucket = await asyncio.to_thread(_r2_client_and_bucket)
    if client is None or not bucket:
        raise ModerationUnavailable("r2_unavailable")

    async def one(key: str) -> str:
        return await asyncio.to_thread(
            _head_and_sign, client, bucket, key, max_image_bytes, expires_in
        )

    results = await asyncio.gather(*(one(key) for key in unique), return_exceptions=True)
    urls: List[str] = []
    unavailable: Optional[ModerationUnavailable] = None
    for result in results:
        if isinstance(result, R2ImageTooLarge):
            raise R2ImageTooLarge()
        if isinstance(result, Exception):
            unavailable = result if isinstance(result, ModerationUnavailable) else ModerationUnavailable("r2_unavailable")
            continue
        urls.append(result)
    if unavailable is not None:
        raise unavailable
    return urls


def request_id_from_request(request: Request) -> str:
    raw = request.headers.get("x-request-id") or request.headers.get("x-correlation-id") or ""
    cleaned = _REQUEST_ID_RE.sub("", raw)[:128]
    return cleaned or str(uuid.uuid4())


def user_id_from_request(request: Request) -> Optional[str]:
    """Best-effort user id from a bearer JWT payload. The token is not verified or logged."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    token = header[7:].strip()
    parts = token.split(".")
    if len(parts) < 2:
        return None
    segment = parts[1]
    padding = "=" * (-len(segment) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(segment + padding))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("user_id", "uid", "sub"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _error_response(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    error = {"code": code, "message": message}
    error.update(extra)
    return JSONResponse(status_code=status_code, content={"error": error})


def flagged_response(categories: List[str]) -> JSONResponse:
    return _error_response(
        400,
        "content_moderation_flagged",
        FLAGGED_MESSAGE,
        flagged_categories=categories,
    )


def unavailable_response() -> JSONResponse:
    return _error_response(503, "moderation_unavailable", UNAVAILABLE_MESSAGE)


def image_too_large_response() -> JSONResponse:
    return _error_response(400, "content_moderation_image_too_large", IMAGE_TOO_LARGE_MESSAGE)


def _query_payload(request: Request) -> dict:
    grouped: dict = {}
    for key, value in request.query_params.multi_items():
        grouped.setdefault(key, []).append(value)
    flat = {}
    for key, values in grouped.items():
        flat[key] = values if len(values) > 1 else values[0]
    return flat


async def read_request_payload(request: Request) -> Any:
    """JSON body, form body, and query params. Invalid JSON is ignored (the route will reject it)."""
    pieces: List[Any] = []
    query = _query_payload(request)
    if query:
        pieces.append(query)

    body = await request.body()
    if not body:
        return pieces[0] if len(pieces) == 1 else pieces

    content_type = (request.headers.get("content-type") or "").lower()
    stripped = body.lstrip()
    if "application/json" in content_type or stripped[:1] in (b"{", b"["):
        try:
            pieces.append(json.loads(body))
        except json.JSONDecodeError:
            pass
    elif "application/x-www-form-urlencoded" in content_type:
        parsed = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=False)
        flat = {key: values if len(values) > 1 else values[0] for key, values in parsed.items()}
        if flat:
            pieces.append(flat)
    elif content_type.startswith("text/plain"):
        text = body.decode("utf-8", errors="replace").strip()
        if text:
            pieces.append({"prompt": text})

    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return pieces


def default_moderation_client() -> ModerationClient:
    return ModerationClient(api_key=openai_api_key_from_env())


class ContentModerationMiddleware(BaseHTTPMiddleware):
    """Block or audit inbound user content before the route handler runs."""

    def __init__(
        self,
        app: ASGIApp,
        client_factory=None,
        exempt_prefixes: Optional[Sequence[str]] = None,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        r2_resolver=None,
    ):
        super().__init__(app)
        self.client_factory = client_factory or default_moderation_client
        self.exempt_prefixes = tuple(exempt_prefixes) if exempt_prefixes is not None else EXEMPT_PATH_PREFIXES
        self.max_image_bytes = max_image_bytes
        self.r2_resolver = r2_resolver or resolve_r2_image_urls

    async def dispatch(self, request: Request, call_next):
        if is_exempt_path(request.url.path, self.exempt_prefixes):
            return await call_next(request)

        request_id = request_id_from_request(request)
        request.state.request_id = request_id
        user_id = user_id_from_request(request) or "-"

        try:
            payload = await read_request_payload(request)
            extracted = extract_moderatable_content(payload, max_image_bytes=self.max_image_bytes)
        except Exception as exc:
            logger.warning(
                "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                "action=blocked reason=extract_failed error_type=%s",
                request_id,
                user_id,
                request.url.path,
                type(exc).__name__,
            )
            return unavailable_response()

        if extracted.oversized:
            logger.warning(
                "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                "action=blocked reason=image_too_large",
                request_id,
                user_id,
                request.url.path,
            )
            return image_too_large_response()

        if extracted.image_r2_keys:
            try:
                signed = await self.r2_resolver(
                    extracted.image_r2_keys,
                    max_image_bytes=self.max_image_bytes,
                )
            except R2ImageTooLarge:
                logger.warning(
                    "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                    "action=blocked reason=image_too_large",
                    request_id,
                    user_id,
                    request.url.path,
                )
                return image_too_large_response()
            except ModerationUnavailable as exc:
                logger.warning(
                    "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                    "action=blocked reason=%s",
                    request_id,
                    user_id,
                    request.url.path,
                    exc.reason,
                )
                return unavailable_response()
            except Exception as exc:
                logger.warning(
                    "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                    "action=blocked reason=unexpected error_type=%s",
                    request_id,
                    user_id,
                    request.url.path,
                    type(exc).__name__,
                )
                return unavailable_response()
            if (
                not isinstance(signed, list)
                or len(signed) != len(extracted.image_r2_keys)
                or any(not isinstance(url, str) or not _is_http_url(url) for url in signed)
            ):
                logger.warning(
                    "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                    "action=blocked reason=r2_unavailable",
                    request_id,
                    user_id,
                    request.url.path,
                )
                return unavailable_response()
            for url in signed:
                if url not in extracted.images:
                    extracted.images.append(url)

        inputs = moderation_inputs(extracted)
        if not inputs:
            return await call_next(request)

        allow_flagged = allow_flagged_from_env()
        try:
            client = self.client_factory()
            if extracted.images and isinstance(client, ModerationClient):
                client.timeout = max(client.timeout, IMAGE_MODERATION_TIMEOUT_SECONDS)
            verdict = merge_verdicts(await _moderate_inputs(client, inputs))
        except ModerationUnavailable as exc:
            logger.warning(
                "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                "action=blocked reason=%s",
                request_id,
                user_id,
                request.url.path,
                exc.reason,
            )
            return unavailable_response()
        except Exception as exc:
            logger.warning(
                "content_moderation request_id=%s user_id=%s path=%s flagged=- categories=[] "
                "action=blocked reason=unexpected error_type=%s",
                request_id,
                user_id,
                request.url.path,
                type(exc).__name__,
            )
            return unavailable_response()

        if verdict.flagged and allow_flagged:
            action = "allowed-despite-flag"
        elif verdict.flagged:
            action = "blocked"
        else:
            action = "allowed"

        log = logger.warning if verdict.flagged else logger.info
        log(
            "content_moderation request_id=%s user_id=%s path=%s flagged=%s categories=%s action=%s scores=%s",
            request_id,
            user_id,
            request.url.path,
            verdict.flagged,
            verdict.categories,
            action,
            verdict.scores,
        )

        request.state.moderation_flagged = action == "allowed-despite-flag"
        if action == "blocked":
            return flagged_response(verdict.categories)
        return await call_next(request)
