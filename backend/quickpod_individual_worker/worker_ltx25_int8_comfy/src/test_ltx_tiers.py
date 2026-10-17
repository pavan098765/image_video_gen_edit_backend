"""Studio quality names map onto skip, speed, and quality. Frames stay put."""

from ltx_patch import apply_api_job_params, apply_studio_quality, resolve_stage2, wants_skip_stage2


def _graph():
    return {
        "344": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["377", 0]}, "_meta": {"title": "Stage 1 sample"}},
        "367": {"class_type": "LTXVSeparateAVLatent", "inputs": {}},
        "371": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}},
        "373": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "374": {"class_type": "VAEDecodeTiled", "inputs": {}},
        "376": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": ""}},
        "358": {"class_type": "LTXVAudioVAEDecode", "inputs": {}},
        "407": {"class_type": "VRAM_Debug", "inputs": {}},
        "408": {"class_type": "LTXControls", "inputs": {}},
        "409": {"class_type": "LoadImage", "inputs": {"image": "keep-first.png"}},
        "411": {"class_type": "LoadImage", "inputs": {"image": "keep-last.png"}},
        "422": {"class_type": "LoraLoader", "inputs": {}},
        "423": {"class_type": "LTXStage2", "inputs": {"stage2_mode": "speed"}},
    }


def test_quality_names():
    assert apply_studio_quality({}) == ""
    draft = {"quality": "draft"}
    assert apply_studio_quality(draft) == "draft"
    assert wants_skip_stage2(draft)
    fast = {"quality": "fast_refine"}
    assert apply_studio_quality(fast) == "fast_refine"
    assert resolve_stage2(fast) == resolve_stage2({"stage2_mode": "speed"})
    assert not wants_skip_stage2(fast)
    full = {"quality": "full_refine"}
    assert apply_studio_quality(full) == "full_refine"
    assert "quality" in resolve_stage2(full).lower() or resolve_stage2(full) == resolve_stage2({"quality_mode": "quality"})


def test_frames_and_latent():
    graph = _graph()
    try:
        apply_api_job_params(graph, "a ridge", {"quality": "fast_refine"}, "ltx25_int8_flf2v", "first.png", "last.png")
    except RuntimeError:
        pass
    assert graph["409"]["inputs"]["image"] in ("first.png", "keep-first.png")
    assert "ltx_load_latent" not in graph or graph.get("344", {}).get("inputs", {}).get("latent_image") != ["ltx_load_latent", 0]
    continued = _graph()
    apply_api_job_params(
        continued,
        "a ridge",
        {"quality": "draft", "continuation_latent": "prev.latent"},
        "ltx25_int8_i2v",
        "first.png",
        None,
    )
    assert continued["344"]["inputs"]["latent_image"] == ["ltx_load_latent", 0]
    assert continued["409"]["inputs"]["image"] == "first.png"


if __name__ == "__main__":
    test_quality_names()
    test_frames_and_latent()
    print("ltx tiers ok")
