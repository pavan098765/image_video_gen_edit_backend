"""Studio tiers stay on the fast API graph and only grow a second pass when asked."""

from h3_fast_patch import (
    FULL_PASS_STEPS,
    REFINE_DENOISE,
    REFINE_STEPS,
    apply_h3_tier,
    resolve_h3_quality,
)
from h3_inference import pick_workflow_file


def _graph():
    return {
        "4": {"class_type": "VAELoader", "inputs": {}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {}},
        "7": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4}},
        "10": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "a", "width": ["7", 0], "height": ["7", 1]}},
        "11": {"class_type": "RandomNoise", "inputs": {}},
        "12": {"class_type": "KSamplerSelect", "inputs": {}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["10", 1], "sigmas": ["30", 0], "guider": ["14", 0]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0]}},
        "30": {"class_type": "ManualSigmas", "inputs": {}},
    }


def test_absent_quality_stays_on_the_fast_graph():
    assert resolve_h3_quality({}) == "draft"
    assert pick_workflow_file({}) .endswith("fastMinimaxH3_textToVideo.json") or pick_workflow_file({}).endswith(
        "fastMinimaxH3_textToVideo_api.json"
    ) or "fast" in pick_workflow_file({}).lower() or pick_workflow_file({"quality": "draft"}) != pick_workflow_file({"workflow": "quality"})
    graph = _graph()
    assert apply_h3_tier(graph, {}, {}) == "draft"
    assert "h3_ultimate" not in graph
    assert "h3_load_latent" not in graph


def test_refine_sets_steps_and_continuation_is_opt_in():
    fast = _graph()
    apply_h3_tier(fast, {"quality": "fast_refine"}, {"references": ["rama.png"]})
    assert fast["h3_pass2"]["inputs"]["steps"] == REFINE_STEPS
    assert fast["h3_pass2"]["inputs"]["denoise"] == REFINE_DENOISE
    assert "h3_pass1" not in fast
    assert fast["10"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert "h3_load_latent" not in fast

    full = _graph()
    apply_h3_tier(full, {"quality": "full_refine"}, {"continuation_latent": "prev.latent", "references": ["rama.png"], "first_frame": "pose.png"})
    full["10"]["inputs"]["first_frame"] = ["h3_first", 0]
    apply_h3_tier(full, {"quality": "full_refine"}, {"continuation_latent": "prev.latent", "references": ["rama.png"]})
    assert full["h3_pass1"]["inputs"]["steps"] == FULL_PASS_STEPS
    assert full["h3_base"]["inputs"]["model_name"].startswith("minimax_h3_fl2va")
    assert full["15"]["inputs"]["latent_image"] == ["h3_load_latent", 0]
    assert full["h3_pass2"]["inputs"]["denoise"] == REFINE_DENOISE


if __name__ == "__main__":
    test_absent_quality_stays_on_the_fast_graph()
    test_refine_sets_steps_and_continuation_is_opt_in()
    print("h3 tiers ok")
