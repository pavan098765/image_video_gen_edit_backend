"""Unit tests for LTX2.3 worker helpers (no GPU / no model download)."""

from __future__ import annotations

import os
import sys

import pytest

# Same import path as runtime worker (app cwd = this ``src`` directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
_INF = os.path.join(_HERE, "inference", "worker_ltx2_3", "src")
if _INF not in sys.path:
    sys.path.insert(0, _INF)

import ltx2_3_inference as ltx


def test_inference_profile_defaults_to_distilled() -> None:
    assert ltx._inference_profile_dict({}) == "distilled"


def test_inference_profile_aliases() -> None:
    assert ltx._normalize_inference_profile("HYBRID") == "hybrid_two_stage"
    assert ltx._normalize_inference_profile("full") == "teacher"


def test_inference_profile_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown inference_profile"):
        ltx._normalize_inference_profile("not_a_profile")


def test_lora_exclusive_raises() -> None:
    p = {
        "civitai_lora_model_url": "https://civitai.com/api/download/models/1",
        "hf_lora_model_id": "x/y",
    }
    with pytest.raises(ValueError, match="either Civitai"):
        ltx._assert_lora_source_exclusive(p)


def test_main_pipe_cache_key_includes_profile_and_source() -> None:
    p = {
        "inference_profile": "teacher",
        "offload": "none",
        "base_model_repo": "my/custom/teacher-repo",
    }
    tok = ltx._ltx_job_params.set(p)
    try:
        ltx._OFFLOAD_MODE = "none"
        key = ltx._main_pipe_cache_key("t2v_distilled")
        assert key[0] == "t2v_distilled"
        assert key[1] == "none"
        assert key[2] == "teacher"
        assert key[3] == "my/custom/teacher-repo"
    finally:
        ltx._ltx_job_params.reset(tok)
        ltx._OFFLOAD_MODE = ""


def test_merge_distilled_extras_allowlist_only() -> None:
    p = {"distilled_pipe_extra": {"cross_attention_kwargs": {"foo": 1}, "rejected": 2}}
    k = ltx._merge_distilled_extras({"guidance_scale": 1.0}, p)
    assert k["cross_attention_kwargs"] == {"foo": 1}
    assert "rejected" not in k


def test_sigmas_kw_empty_when_none() -> None:
    assert ltx._sigmas_kw(None) == {}


def test_teacher_steps_defaults() -> None:
    assert ltx._stage1_steps_teacher({}) == 30
    assert ltx._stage2_steps_teacher({}) == 30
