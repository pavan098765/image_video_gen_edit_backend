"""Local TestClient coverage for /studio/* (not live genedit.mooo.com)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("VULTR_API_KEY", "")
os.environ.setdefault("VULTR_SNAPSHOT_ID", "")
os.environ.setdefault("STUDIO_VULTR_MOCK", "1")

from db import CreditTransaction, Job, User, Worker  # noqa: E402
from job_manager import JobManager, JobState, is_studio_model  # noqa: E402
from studio_capacity import class_for_pipeline  # noqa: E402
from studio_routes import studio_router, _client_auth  # noqa: E402
import studio_routes  # noqa: E402
from vultr_pool import VultrPool  # noqa: E402
import vultr_pool as vultr_pool_mod  # noqa: E402


def _engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def db_session():
    engine = _engine()
    User.__table__.create(engine)
    Job.__table__.create(engine)
    CreditTransaction.__table__.create(engine)
    Worker.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    u1 = User(id="u1", firebase_uid="f1", email="a@x.com", total_credits=100, is_active=True)
    u2 = User(id="u2", firebase_uid="f2", email="b@x.com", total_credits=100, is_active=True)
    u0 = User(id="u0", firebase_uid="f0", email="z@x.com", total_credits=0, is_active=True)
    session.add_all([u1, u2, u0])
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(db_session, monkeypatch):
    current = {"user_id": "u1", "firebase_uid": "f1"}

    async def fake_auth():
        return dict(current)

    monkeypatch.setattr(
        "credit_service.get_credit_cost_breakdown",
        lambda *a, **k: (25, {"pricing_type": "standard"}),
    )

    def _db():
        class _Keep:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def close(self):
                return None

        return _Keep(db_session)

    monkeypatch.setattr(studio_routes, "_db", _db)
    monkeypatch.setattr("credit_service.SessionLocal", lambda: db_session)

    pool = VultrPool()
    pool.max_instances = 1
    pool.vms = []
    monkeypatch.setattr(vultr_pool_mod, "pool", pool)
    monkeypatch.setattr(studio_routes, "vultr_pool", pool)

    studio_routes._pending_pull.clear()

    app = FastAPI()
    app.include_router(studio_router)
    app.dependency_overrides[_client_auth] = fake_auth
    tc = TestClient(app)
    tc.current = current
    tc.pool = pool
    tc.db = db_session
    return tc


def _job_body(**kwargs):
    body = {
        "pipeline": "animated-explainer",
        "review_mode": True,
        "asset_keys": [],
        "prefs": {
            "topic": "How photosynthesis works",
            "duration_seconds": 30,
            "platform_profile": "16:9",
            "delivery_promise": "mixed",
        },
    }
    body.update(kwargs)
    return body


def test_jwt_401():
    async def deny():
        raise HTTPException(status_code=401, detail="missing token")

    app = FastAPI()
    app.include_router(studio_router)
    app.dependency_overrides[_client_auth] = deny
    tc = TestClient(app)
    r = tc.get("/studio/jobs/any")
    assert r.status_code == 401


def test_footage_required_400(client):
    r = client.post(
        "/studio/jobs",
        json=_job_body(pipeline="talking-head", prefs={"topic": "clip", "talking": True}),
    )
    assert r.status_code == 400


def test_owner_isolation(client):
    r = client.post("/studio/jobs", json=_job_body())
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    client.current["user_id"] = "u2"
    client.current["firebase_uid"] = "f2"
    r2 = client.get(f"/studio/jobs/{job_id}")
    assert r2.status_code in (403, 404)


def test_two_chrome_jobs_queue_one_vm(client):
    a = client.post("/studio/jobs", json=_job_body())
    b = client.post("/studio/jobs", json=_job_body())
    assert a.status_code == 200
    assert b.status_code == 200
    assert len(client.pool.vms) == 1
    assert b.json()["placement"]["queued"] is True
    assert b.json()["placement"]["action"] == "queued"


def test_gpu_submit_never_calls_vultr():
    src = (Path(__file__).resolve().parents[1] / "src" / "routes.py").read_text(
        encoding="utf-8"
    )
    assert "vultr_pool" not in src
    assert "place_job" not in src


def test_mock_agent_heartbeat_and_pull(client):
    created = client.post("/studio/jobs", json=_job_body())
    assert created.status_code == 200
    job_id = created.json()["job_id"]
    hb = client.post(
        "/studio/internal/heartbeat",
        json={"worker_id": "studio-mock", "vultr_id": "mock-1"},
    )
    assert hb.status_code == 200
    assert hb.json()["model"] == "studio_engine"
    nxt = client.get("/studio/internal/next-job")
    assert nxt.status_code == 200
    payload = nxt.json()
    assert payload["job_id"] == job_id
    assert "email" not in payload
    assert "firebase" not in str(payload).lower()
    ready = client.post(
        "/studio/internal/review_ready",
        json={
            "job_id": job_id,
            "storyboard": {
                "job_id": job_id,
                "scenes": [{"id": "sc1", "approved": False, "poster_key": "p.png"}],
            },
        },
    )
    assert ready.status_code == 200
    st = client.get(f"/studio/jobs/{job_id}")
    assert st.json()["status"] == "review"


def test_no_refund_while_reviewing(client):
    created = client.post("/studio/jobs", json=_job_body())
    job_id = created.json()["job_id"]
    user = client.db.query(User).filter(User.id == "u1").first()
    locked_balance = user.total_credits
    client.post("/studio/internal/review_ready", json={"job_id": job_id, "storyboard": {"scenes": []}})
    fail = client.post("/studio/internal/failed", json={"job_id": job_id, "error": "nope"})
    assert fail.json().get("skipped_refund") is True
    client.db.refresh(user)
    assert user.total_credits == locked_balance


def test_zero_credits_402(client):
    client.current["user_id"] = "u0"
    client.current["firebase_uid"] = "f0"
    r = client.post("/studio/jobs", json=_job_body())
    assert r.status_code == 402


def test_gpu_cannot_enter_reviewing():
    assert not is_studio_model("wan_22_fp8_lightx2v_t2v")
    assert is_studio_model("studio")


def test_class_for_pipeline():
    assert class_for_pipeline("animated-explainer") == "chrome_heavy"
    assert class_for_pipeline("clip-factory") == "ffmpeg_edit"
    assert class_for_pipeline("shorts") == "ffmpeg_edit"
    assert class_for_pipeline("documentary-montage") == "doc_ingest"


def test_auto_router_talking_and_long():
    from studio_routes import route_auto_pipeline

    assert route_auto_pipeline({"talking": True}, ["k"]) == "talking-head"
    assert route_auto_pipeline({"long_video": True}, ["k"]) == "clip-factory"
    assert route_auto_pipeline({"native_short": True}, []) == "shorts"
    assert route_auto_pipeline({}, []) == "animated-explainer"


def test_job_manager_reviewing_gpu_rejected(db_session):
    job = Job(job_id="gpu1", model="wan_22", prompt="x", params={}, status="running", user_id="u1")
    db_session.add(job)
    db_session.commit()
    bind = db_session.get_bind()
    Factory = sessionmaker(bind=bind)
    mgr = JobManager(session_factory=Factory)
    from exceptions import JobStateError

    with pytest.raises(JobStateError):
        mgr.update_job_state("gpu1", JobState.REVIEWING)


def test_studio_credits_scale_with_duration_and_tier():
    from credit_service import calculate_job_cost

    draft = calculate_job_cost("studio", {"prefs": {"duration_seconds": 30, "quality_tier": "draft"}})
    hero = calculate_job_cost("studio", {"prefs": {"duration_seconds": 120, "quality_tier": "hero", "emit_shorts": True}})
    assert draft == 10  # floor
    assert hero == 28 * 2 + 10
