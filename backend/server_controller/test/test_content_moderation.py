"""OpenAI moderation gate: client, content extraction, and middleware decisions."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from content_moderation import (  # noqa: E402
    EXEMPT_PATH_PREFIXES,
    R2ImageTooLarge,
    ContentModerationMiddleware,
    build_moderation_input,
    extract_moderatable_content,
    is_exempt_path,
    moderation_inputs,
    resolve_r2_image_urls,
)
from moderation_client import (  # noqa: E402
    IMAGE_MODERATION_TIMEOUT_SECONDS,
    MODERATION_MODEL,
    MODERATION_URL,
    ModerationClient,
    ModerationUnavailable,
    interpret_moderation_response,
    parse_allow_moderation_flagged,
)

PROMPT = "UNIQUE_PROMPT_DO_NOT_LOG"
API_KEY = "sk-unit-test-key"
IMAGE_URL = "https://cdn.example/cat.png"


class FakeModeration:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else {
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}]
        }
        self.error = error
        self.calls = []

    async def moderate(self, moderation_input):
        self.calls.append(moderation_input)
        if self.error:
            raise self.error
        return self.payload


def _flagged_payload():
    return {
        "id": "modr-test",
        "model": MODERATION_MODEL,
        "results": [
            {
                "flagged": False,
                "categories": {"violence": False, "illicit": None},
                "category_scores": {"violence": 0.01, "illicit": None},
            },
            {
                "flagged": True,
                "categories": {
                    "violence": True,
                    "harassment": False,
                    "illicit": None,
                    "hate": True,
                },
                "category_scores": {"violence": 0.861234, "hate": 0.5, "illicit": None},
            },
        ],
    }


def _clear_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_MODERATION_FLAGGED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _app(fake, monkeypatch, flag=None, max_image_bytes=20 * 1024 * 1024, r2_resolver=None):
    _clear_flag(monkeypatch)
    if flag is not None:
        monkeypatch.setenv("ALLOW_MODERATION_FLAGGED", flag)
    hits = {"n": 0}
    app = FastAPI()
    app.add_middleware(
        ContentModerationMiddleware,
        client_factory=lambda: fake,
        max_image_bytes=max_image_bytes,
        r2_resolver=r2_resolver,
    )

    @app.post("/submit")
    async def submit(request: Request):
        hits["n"] += 1
        body = await request.json()
        return {
            "ok": True,
            "prompt": body.get("prompt"),
            "moderation_flagged": getattr(request.state, "moderation_flagged", False) is True,
        }

    @app.post("/echo-form")
    async def echo_form(request: Request):
        hits["n"] += 1
        return {"ok": True}

    @app.api_route("/health", methods=["GET", "POST"])
    async def health():
        hits["n"] += 1
        return {"ok": True}

    @app.api_route("/heartbeat", methods=["GET", "POST"])
    async def heartbeat():
        hits["n"] += 1
        return {"ok": True}

    @app.get("/search")
    async def search():
        hits["n"] += 1
        return {"ok": True}

    return app, hits


def _client(app):
    return TestClient(app)


def _bearer(user_id="user-123"):
    segment = base64.urlsafe_b64encode(json.dumps({"user_id": user_id, "uid": user_id}).encode()).decode().rstrip("=")
    return f"Bearer header.{segment}.sig"


def test_parse_allow_flag_only_accepts_true_and_false():
    assert parse_allow_moderation_flagged(None) is False
    assert parse_allow_moderation_flagged("") is False
    assert parse_allow_moderation_flagged("true") is True
    assert parse_allow_moderation_flagged("TRUE") is True
    assert parse_allow_moderation_flagged("  False ") is False
    assert parse_allow_moderation_flagged("yes") is False
    assert parse_allow_moderation_flagged("1") is False
    assert parse_allow_moderation_flagged("maybe") is False


def test_null_categories_are_not_flagged_names():
    verdict = interpret_moderation_response(_flagged_payload())
    assert verdict.flagged is True
    assert verdict.categories == ["hate", "violence"]
    assert "illicit" not in verdict.categories
    assert verdict.scores["violence"] == pytest.approx(0.861234)

    clean = interpret_moderation_response({
        "results": [{
            "flagged": False,
            "categories": {"violence": True, "illicit": None, "harassment": False},
        }]
    })
    assert clean.flagged is False
    assert clean.categories == ["violence"]

    empty = interpret_moderation_response({
        "results": [{"flagged": True, "categories": {"illicit": None, "illicit/violent": None}}]
    })
    assert empty.flagged is True
    assert empty.categories == []


def test_invalid_moderation_body_fails_closed():
    with pytest.raises(ModerationUnavailable):
        interpret_moderation_response({"results": None})
    with pytest.raises(ModerationUnavailable):
        interpret_moderation_response({"results": [{"categories": {}}]})
    with pytest.raises(ModerationUnavailable):
        interpret_moderation_response([])


def test_exempt_list_covers_workers_and_not_frontend_routes():
    for path in (
        "/health",
        "/health/",
        "/docs",
        "/register_worker",
        "/heartbeat",
        "/job_started",
        "/job_complete",
        "/personal/pull",
        "/personal/complete",
        "/kie/callback",
        "/runware/callback",
        "/ads/admob/ssv",
        "/studio/internal/next-job",
    ):
        assert is_exempt_path(path), path
    for path in (
        "/submit",
        "/auth/sync",
        "/studio/jobs",
        "/describe_asset_for_prompt",
        "/get_media_upload_urls",
        "/job_status",
        "/admin/personal",
        "/healthcheck",
        "/on_device/job_complete",
    ):
        assert not is_exempt_path(path), path
    assert "/health" in EXEMPT_PATH_PREFIXES


def test_extract_text_image_and_ignore_non_content():
    payload = {
        "model": "flux",
        "prompt": "a red balloon",
        "fcm_token": "device-token",
        "input_image_r2_keys": ["inputs/a.png", ["inputs/b.png", "inputs/a.png"]],
        "mask_image_r2_key": "masks/a.png",
        "input_video_r2_key": "videos/a.mp4",
        "params": {
            "mode": "txt2img",
            "width": 1024,
            "negative_prompt": "blurry",
            "init_image": IMAGE_URL,
            "init_image_r2_key": "inputs/c.png",
            "prompt": "a red balloon",
        },
    }
    extracted = extract_moderatable_content(payload)
    assert extracted.texts == ["a red balloon", "blurry"]
    assert extracted.images == [IMAGE_URL]
    assert extracted.image_r2_keys == ["inputs/a.png", "inputs/b.png", "inputs/c.png"]
    assert extracted.oversized is False
    assert build_moderation_input(extracted) == [
        {"type": "text", "text": "a red balloon"},
        {"type": "text", "text": "blurry"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]

    text_only = extract_moderatable_content({"prompt": "a red balloon", "model": "flux"})
    assert build_moderation_input(text_only) == "a red balloon"

    image_only = extract_moderatable_content({"params": {"image_urls": [IMAGE_URL]}})
    assert build_moderation_input(image_only) == [
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]

    nothing = extract_moderatable_content({
        "model": "flux",
        "params": {"mode": "txt2img", "width": 512},
        "fcm_token": "tok",
        "mask_image_r2_key": "masks/a.png",
        "input_video_r2_key": "videos/a.mp4",
        "prompt": "   ",
    })
    assert nothing.image_r2_keys == []
    assert build_moderation_input(nothing) is None


def test_oversize_and_unreadable_images():
    raw = base64.b64encode(b"x" * 30).decode()
    url = f"data:image/png;base64,{raw}"
    oversized = extract_moderatable_content({"image_url": url}, max_image_bytes=10)
    assert oversized.oversized is True
    assert oversized.images == []

    fits = extract_moderatable_content({"image_url": url}, max_image_bytes=64)
    assert fits.oversized is False
    assert fits.images == [url]

    junk = extract_moderatable_content({"image_url": "data:image/png;base64,@@@"})
    assert junk.images == []
    assert junk.oversized is False
    assert build_moderation_input(junk) is None


def test_flag_true_flagged_passes_and_logs_without_content(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="content_moderation")
    fake = FakeModeration(_flagged_payload())
    app, hits = _app(fake, monkeypatch, flag="true")
    response = _client(app).post(
        "/submit",
        json={"prompt": PROMPT, "model": "flux"},
        headers={"x-request-id": "req-1", "authorization": _bearer("user-123")},
    )
    assert response.status_code == 200
    assert response.json()["prompt"] == PROMPT
    assert response.json()["moderation_flagged"] is True
    assert hits["n"] == 1
    assert fake.calls == [PROMPT]
    assert "allowed-despite-flag" in caplog.text
    assert "req-1" in caplog.text
    assert "user-123" in caplog.text
    assert "hate" in caplog.text
    assert "violence" in caplog.text
    assert "0.861234" in caplog.text
    assert PROMPT not in caplog.text
    assert API_KEY not in caplog.text


def test_flag_false_flagged_blocks_before_handler(monkeypatch):
    fake = FakeModeration(_flagged_payload())
    app, hits = _app(fake, monkeypatch, flag="false")
    response = _client(app).post("/submit", json={"prompt": PROMPT, "model": "flux"})
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "content_moderation_flagged",
            "message": "Your request was blocked because it may violate our content policy.",
            "flagged_categories": ["hate", "violence"],
        }
    }
    assert "0.861234" not in response.text
    assert "category_scores" not in response.text
    assert PROMPT not in response.text
    assert hits["n"] == 0
    assert len(fake.calls) == 1


def test_missing_and_invalid_flag_block_flagged_content(monkeypatch):
    fake = FakeModeration(_flagged_payload())
    app, hits = _app(fake, monkeypatch, flag=None)
    missing = _client(app).post("/submit", json={"prompt": PROMPT})
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "content_moderation_flagged"
    assert hits["n"] == 0

    monkeypatch.setenv("ALLOW_MODERATION_FLAGGED", "maybe")
    invalid = _client(app).post("/submit", json={"prompt": PROMPT})
    assert invalid.status_code == 400
    assert hits["n"] == 0


def test_not_flagged_passes_even_if_a_category_is_true(monkeypatch):
    fake = FakeModeration({
        "results": [{
            "flagged": False,
            "categories": {"violence": True, "illicit": None},
            "category_scores": {"violence": 0.2},
        }]
    })
    app, hits = _app(fake, monkeypatch, flag="false")
    response = _client(app).post("/submit", json={"prompt": "a red balloon"})
    assert response.status_code == 200
    assert response.json()["moderation_flagged"] is False
    assert hits["n"] == 1
    assert "0.2" not in response.text

    allowed = _app(fake, monkeypatch, flag="true")
    clean = _client(allowed[0]).post("/submit", json={"prompt": "a red balloon"})
    assert clean.status_code == 200
    assert clean.json()["moderation_flagged"] is False


def test_moderation_error_and_timeout_return_503(monkeypatch):
    fake = FakeModeration(error=ModerationUnavailable("timeout"))
    app, hits = _app(fake, monkeypatch, flag="true")
    response = _client(app).post("/submit", json={"prompt": PROMPT})
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "moderation_unavailable",
            "message": "Content moderation is temporarily unavailable. Please try again.",
        }
    }
    assert PROMPT not in response.text
    assert hits["n"] == 0

    broken = FakeModeration(payload={"results": None})
    app, hits = _app(broken, monkeypatch, flag="false")
    bad = _client(app).post("/submit", json={"prompt": PROMPT})
    assert bad.status_code == 503
    assert bad.json()["error"]["code"] == "moderation_unavailable"
    assert hits["n"] == 0


def test_text_image_and_mixed_inputs_are_sent(monkeypatch):
    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="false")
    client = _client(app)

    text = client.post("/submit", json={"model": "flux", "prompt": "a red balloon"})
    assert text.status_code == 200
    assert fake.calls[-1] == "a red balloon"

    image = client.post("/submit", json={"model": "flux", "params": {"image_urls": [IMAGE_URL]}})
    assert image.status_code == 200
    assert fake.calls[-1] == [{"type": "image_url", "image_url": {"url": IMAGE_URL}}]

    mixed = client.post("/submit", json={
        "prompt": "a red balloon",
        "params": {"negative_prompt": "blurry", "init_image": IMAGE_URL},
    })
    assert mixed.status_code == 200
    assert fake.calls[-1] == [
        {"type": "text", "text": "a red balloon"},
        {"type": "text", "text": "blurry"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]
    assert hits["n"] == 3


def test_no_moderatable_content_skips_api(monkeypatch):
    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="false")
    response = _client(app).post("/submit", json={
        "model": "flux",
        "prompt": "  ",
        "params": {"mode": "txt2img", "width": 512},
        "fcm_token": "tok",
        "mask_image_r2_key": "masks/a.png",
        "input_video_r2_key": "videos/a.mp4",
    })
    assert response.status_code == 200
    assert fake.calls == []
    assert hits["n"] == 1


def test_exempt_routes_skip_api_even_with_a_prompt(monkeypatch):
    fake = FakeModeration(_flagged_payload())
    app, hits = _app(fake, monkeypatch, flag="false")
    client = _client(app)
    health = client.post("/health", json={"prompt": PROMPT})
    heartbeat = client.post("/heartbeat", json={"prompt": PROMPT})
    assert health.status_code == 200
    assert heartbeat.status_code == 200
    assert fake.calls == []
    assert hits["n"] == 2


def test_query_and_form_text_are_moderated(monkeypatch):
    fake = FakeModeration(_flagged_payload())
    app, hits = _app(fake, monkeypatch, flag="false")
    client = _client(app)
    query = client.get("/search", params={"prompt": PROMPT})
    assert query.status_code == 400
    assert query.json()["error"]["code"] == "content_moderation_flagged"
    assert fake.calls[-1] == PROMPT
    assert hits["n"] == 0

    form = client.post(
        "/echo-form",
        content=b"prompt=hello+there",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert form.status_code == 400
    assert fake.calls[-1] == "hello there"


def test_oversize_image_blocks_without_calling_api(monkeypatch):
    fake = FakeModeration()
    raw = base64.b64encode(b"x" * 30).decode()
    app, hits = _app(fake, monkeypatch, flag="true", max_image_bytes=8)
    response = _client(app).post("/submit", json={
        "prompt": "a red balloon",
        "image_url": f"data:image/png;base64,{raw}",
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_moderation_image_too_large"
    assert fake.calls == []
    assert hits["n"] == 0


def test_missing_api_key_fails_closed_without_network(monkeypatch):
    _clear_flag(monkeypatch)
    hits = {"n": 0}
    app = FastAPI()
    app.add_middleware(ContentModerationMiddleware)

    @app.post("/submit")
    async def submit():
        hits["n"] += 1
        return {"ok": True}

    response = _client(app).post("/submit", json={"prompt": PROMPT})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "moderation_unavailable"
    assert hits["n"] == 0
    assert API_KEY not in response.text


def test_frontend_path_cannot_bypass_gate(monkeypatch):
    fake = FakeModeration(_flagged_payload())
    app, _hits = _app(fake, monkeypatch, flag="false")
    response = _client(app).post("/admin/personal", json={"prompt": PROMPT, "model": "flux"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_moderation_flagged"
    assert len(fake.calls) == 1


def test_middleware_runs_after_the_path_whitelist():
    from middleware import setup_middleware

    app = FastAPI()
    setup_middleware(app, debug=False)
    names = [middleware.cls.__name__ for middleware in app.user_middleware]
    assert names.index("AggressiveSecurityMiddleware") < names.index("PathWhitelistMiddleware")
    assert names.index("PathWhitelistMiddleware") < names.index("ContentModerationMiddleware")


def test_config_loader_parses_flag_and_does_not_log_the_key(monkeypatch, caplog):
    monkeypatch.setenv("ALLOW_MODERATION_FLAGGED", "TRUE")
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    from config import ConfigManager

    manager = ConfigManager.get_instance()
    manager._load_moderation_config()
    assert manager.config.moderation.allow_flagged is True
    assert manager.config.moderation.openai_api_key == API_KEY

    monkeypatch.setenv("ALLOW_MODERATION_FLAGGED", "nope")
    manager._load_moderation_config()
    assert manager.config.moderation.allow_flagged is False

    caplog.set_level(logging.INFO, logger="config")
    manager.log_config()
    assert API_KEY not in caplog.text
    assert "openai_api_key_set=True" in caplog.text
    assert "allow_flagged=False" in caplog.text


def _ok_body():
    return {
        "id": "modr-1",
        "model": MODERATION_MODEL,
        "results": [{"flagged": False, "categories": {"violence": False}, "category_scores": {"violence": 0.0}}],
    }


def test_client_posts_model_and_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    client = ModerationClient(api_key=API_KEY, transport=httpx.MockTransport(handler), backoff_base=0)
    payload = asyncio.run(client.moderate("hello"))
    assert payload["results"][0]["flagged"] is False
    assert seen["url"] == MODERATION_URL
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["content_type"].startswith("application/json")
    assert seen["body"] == {"model": MODERATION_MODEL, "input": "hello"}


def test_client_retries_429_and_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        if calls["n"] == 2:
            return httpx.Response(503, content=PROMPT.encode())
        return httpx.Response(200, json=_ok_body())

    client = ModerationClient(api_key=API_KEY, max_retries=2, backoff_base=0, transport=httpx.MockTransport(handler))
    payload = asyncio.run(client.moderate("hello"))
    assert payload["results"][0]["flagged"] is False
    assert calls["n"] == 3


def test_client_stops_after_two_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, content=PROMPT.encode())

    client = ModerationClient(api_key=API_KEY, max_retries=2, backoff_base=0, transport=httpx.MockTransport(handler))
    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(client.moderate("hello"))
    assert caught.value.reason == "http_500"
    assert calls["n"] == 3
    assert PROMPT not in str(caught.value)
    assert API_KEY not in str(caught.value)


def test_client_does_not_retry_other_4xx_or_invalid_json():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, content=b"bad key")

    client = ModerationClient(api_key=API_KEY, max_retries=2, backoff_base=0, transport=httpx.MockTransport(handler))
    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(client.moderate("hello"))
    assert caught.value.reason == "http_401"
    assert calls["n"] == 1

    def bad_json(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"not-json")

    calls["n"] = 0
    client = ModerationClient(api_key=API_KEY, max_retries=2, backoff_base=0, transport=httpx.MockTransport(bad_json))
    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(client.moderate("hello"))
    assert caught.value.reason == "invalid_response"
    assert calls["n"] == 1


def test_client_timeout_retries_then_fails_closed():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out")

    client = ModerationClient(
        api_key=API_KEY,
        timeout=5.0,
        max_retries=2,
        backoff_base=0,
        transport=httpx.MockTransport(handler),
    )
    assert client.timeout == 5.0
    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(client.moderate(PROMPT))
    assert caught.value.reason == "timeout"
    assert calls["n"] == 3
    assert PROMPT not in str(caught.value)
    assert API_KEY not in str(caught.value)


def test_client_missing_key_does_not_call_network():
    called = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json=_ok_body())

    client = ModerationClient(api_key="", transport=httpx.MockTransport(handler))
    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(client.moderate("hello"))
    assert caught.value.reason == "missing_api_key"
    assert called["n"] == 0


def test_uploaded_r2_image_is_presigned_and_moderated(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="content_moderation")
    seen = {}

    async def resolver(keys, *, max_image_bytes):
        seen["keys"] = list(keys)
        seen["max"] = max_image_bytes
        return [f"https://r2.example/signed-{key.split('/')[-1][0]}" for key in keys]

    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="false", r2_resolver=resolver)
    response = _client(app).post("/submit", json={
        "model": "flux",
        "prompt": "a red balloon",
        "input_image_r2_keys": ["inputs/a.png", "inputs/a.png"],
        "mask_image_r2_key": "masks/skip.png",
        "input_video_r2_key": "videos/skip.mp4",
        "params": {"init_image_r2_key": "inputs/b.png"},
    })
    assert response.status_code == 200
    assert hits["n"] == 1
    assert seen["keys"] == ["inputs/a.png", "inputs/b.png"]
    assert seen["max"] == 20 * 1024 * 1024
    assert fake.calls == [[
        {"type": "text", "text": "a red balloon"},
        {"type": "image_url", "image_url": {"url": "https://r2.example/signed-a"}},
        {"type": "image_url", "image_url": {"url": "https://r2.example/signed-b"}},
    ]]
    assert "signed-a" not in caplog.text
    assert "inputs/a.png" not in caplog.text


def test_r2_image_only_submit_is_moderated(monkeypatch):
    async def resolver(keys, *, max_image_bytes):
        return [f"https://r2.example/{key}" for key in keys]

    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="false", r2_resolver=resolver)
    response = _client(app).post("/submit", json={
        "model": "flux",
        "input_image_r2_keys": [["inputs/a.png", "inputs/b.png"]],
    })
    assert response.status_code == 200
    assert hits["n"] == 1
    assert fake.calls == [[
        {"type": "image_url", "image_url": {"url": "https://r2.example/inputs/a.png"}},
        {"type": "image_url", "image_url": {"url": "https://r2.example/inputs/b.png"}},
    ]]


def test_oversize_r2_image_blocks_without_calling_openai(monkeypatch):
    async def resolver(keys, *, max_image_bytes):
        raise R2ImageTooLarge()

    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="true", r2_resolver=resolver)
    response = _client(app).post("/submit", json={
        "prompt": "a red balloon",
        "input_image_r2_keys": ["inputs/big.png"],
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_moderation_image_too_large"
    assert fake.calls == []
    assert hits["n"] == 0


def test_r2_signing_failure_fails_closed(monkeypatch):
    async def resolver(keys, *, max_image_bytes):
        raise ModerationUnavailable("r2_unavailable")

    fake = FakeModeration()
    app, hits = _app(fake, monkeypatch, flag="true", r2_resolver=resolver)
    response = _client(app).post("/submit", json={"input_image_r2_keys": ["inputs/a.png"]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "moderation_unavailable"
    assert fake.calls == []
    assert hits["n"] == 0
    assert "inputs/a.png" not in response.text


def test_many_r2_images_are_batched_with_the_prompt(monkeypatch):
    async def resolver(keys, *, max_image_bytes):
        return [f"https://r2.example/{i}" for i in range(len(keys))]

    fake = FakeModeration()
    app, _hits = _app(fake, monkeypatch, flag="false", r2_resolver=resolver)
    keys = [f"inputs/{i}.png" for i in range(11)]
    response = _client(app).post("/submit", json={"prompt": "a red balloon", "input_image_r2_keys": keys})
    assert response.status_code == 200
    assert len(fake.calls) == 2
    first, second = fake.calls
    assert first[0] == {"type": "text", "text": "a red balloon"}
    assert len(first) == 11
    assert second[0] == {"type": "text", "text": "a red balloon"}
    assert second[1] == {"type": "image_url", "image_url": {"url": "https://r2.example/10"}}


def test_image_batches_keep_text_on_every_chunk():
    extracted = extract_moderatable_content({
        "prompt": "a red balloon",
        "params": {"image_urls": [f"https://cdn.example/{i}.png" for i in range(11)]},
    })
    batches = moderation_inputs(extracted)
    assert len(batches) == 2
    assert batches[0][0]["text"] == "a red balloon"
    assert len(batches[0]) == 11
    assert batches[1][0]["text"] == "a red balloon"
    assert len(batches[1]) == 2


class _FakeR2:
    def __init__(self, sizes):
        self.sizes = sizes
        self.signed = []

    def head_object(self, Bucket, Key):
        if Key not in self.sizes:
            raise RuntimeError(f"missing {Key}")
        return {"ContentLength": self.sizes[Key]}

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        assert ClientMethod == "get_object"
        assert Params["Bucket"] == "bucket"
        assert ExpiresIn == 180
        self.signed.append(Params["Key"])
        return f"https://files.example/{Params['Key']}?sig=secret"


def test_resolve_r2_checks_size_and_presigns_without_downloading():
    store = _FakeR2({"ok": 100, "also": 20 * 1024 * 1024})
    urls = asyncio.run(resolve_r2_image_urls(
        ["ok", "also", "ok"],
        max_image_bytes=20 * 1024 * 1024,
        client=store,
        bucket="bucket",
    ))
    assert urls == [
        "https://files.example/ok?sig=secret",
        "https://files.example/also?sig=secret",
    ]
    assert store.signed == ["ok", "also"]


def test_resolve_r2_oversize_and_head_failure(monkeypatch):
    store = _FakeR2({"big": 20 * 1024 * 1024 + 1})
    with pytest.raises(R2ImageTooLarge):
        asyncio.run(resolve_r2_image_urls(
            ["big"],
            max_image_bytes=20 * 1024 * 1024,
            client=store,
            bucket="bucket",
        ))
    assert store.signed == []

    class Boom:
        def head_object(self, Bucket, Key):
            raise RuntimeError("inputs/secret.png")

        def generate_presigned_url(self, *args, **kwargs):
            raise AssertionError("should not sign")

    with pytest.raises(ModerationUnavailable) as caught:
        asyncio.run(resolve_r2_image_urls(
            ["inputs/secret.png"],
            max_image_bytes=20 * 1024 * 1024,
            client=Boom(),
            bucket="bucket",
        ))
    assert caught.value.reason == "r2_unavailable"
    assert "secret" not in str(caught.value)

    monkeypatch.setattr(
        "content_moderation._r2_client_and_bucket",
        lambda: (None, None),
    )
    with pytest.raises(ModerationUnavailable) as missing:
        asyncio.run(resolve_r2_image_urls(["inputs/a.png"], max_image_bytes=100))
    assert missing.value.reason == "r2_unavailable"


def test_r2_image_uses_the_longer_moderation_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body())

    client = ModerationClient(
        api_key=API_KEY,
        timeout=5.0,
        backoff_base=0,
        transport=httpx.MockTransport(handler),
    )

    async def resolver(keys, *, max_image_bytes):
        return ["https://r2.example/a.png"]

    app, hits = _app(client, monkeypatch, flag="false", r2_resolver=resolver)
    response = _client(app).post("/submit", json={"input_image_r2_keys": ["inputs/a.png"]})
    assert response.status_code == 200
    assert hits["n"] == 1
    assert client.timeout == IMAGE_MODERATION_TIMEOUT_SECONDS
