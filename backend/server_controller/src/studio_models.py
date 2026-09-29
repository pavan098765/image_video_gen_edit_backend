from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PIPELINES = [
    "animated-explainer",
    "animation",
    "character-animation",
    "cinematic",
    "clip-factory",
    "documentary-montage",
    "hybrid",
    "podcast-repurpose",
    "screen-demo",
    "talking-head",
    "shorts",
    "auto",
]

FOOTAGE_REQUIRED = {
    "talking-head",
    "clip-factory",
    "hybrid",
    "podcast-repurpose",
}


class StudioPrefs(BaseModel):
    topic: Optional[str] = None
    audience: Optional[str] = None
    duration_seconds: int = Field(default=30, ge=5, le=1200)
    language: str = "en"
    platform_profile: Literal["16:9", "9:16", "1:1", "21:9"] = "16:9"
    style_playbook: str = "clean-professional"
    voice_id: Optional[str] = None
    tts_provider: Literal["kokoro", "piper", "elevenlabs"] = "kokoro"
    captions: bool = True
    karaoke: bool = True
    music: bool = True
    real_footage_only: bool = False
    budget_cap_usd: float = Field(default=2.0, ge=0)
    delivery_promise: Literal["motion_led", "mixed"] = "mixed"
    render_runtime: Optional[Literal["remotion", "hyperframes", "ffmpeg", "motion_canvas"]] = None
    talking: bool = False
    long_video: bool = False
    quality_tier: Literal["draft", "studio", "hero"] = "studio"
    emit_shorts: bool = False
    native_short: bool = False
    reference_key: Optional[str] = None
    mode: Optional[Literal["from_topic", "from_footage", "from_evidence", "shorts"]] = None

    @field_validator("topic")
    @classmethod
    def topic_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v


class StudioJobCreate(BaseModel):
    pipeline: str
    prefs: StudioPrefs
    review_mode: bool = True
    asset_keys: list[str] = Field(default_factory=list)
    fcm_token: Optional[str] = None

    @field_validator("pipeline")
    @classmethod
    def pipeline_ok(cls, v: str) -> str:
        if v not in PIPELINES:
            raise ValueError(f"unknown pipeline {v}")
        return v


class SceneAction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    action: Literal["approve", "regenerate", "replace", "patch_copy"]
    script_line: Optional[str] = Field(default=None, alias="copy")
    replace_r2_key: Optional[str] = None
    explore_job_id: Optional[str] = None


class StudioUploadRequest(BaseModel):
    job_id: Optional[str] = None
    filename: str
    content_type: str = "application/octet-stream"


class SceneCut(BaseModel):
    scene_id: str
    in_seconds: Optional[float] = None
    out_seconds: Optional[float] = None


class ContinueJobRequest(BaseModel):
    cuts: Optional[list[SceneCut]] = None
