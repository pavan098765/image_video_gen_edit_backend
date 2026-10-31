"""Prompt style book: each Kie and Runware text model has its own refine and invent brief."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prompt_enhancement import instruction_for  # noqa: E402


# model, field, unused-name that still resolves to the old shared kind
OWNED = [
    ("flux2_klein_9b", "prompt", "plain_picture"),
    ("flux2_klein_9b_edit", "prompt", "sample_edit"),
    ("qwen2_1_t2i", "prompt", "plain_picture"),
    ("qwen2_1_i2i", "prompt", "sample_i2i"),
    ("grok_imagine_t2i", "prompt", "plain_picture"),
    ("grok_imagine_edit", "prompt", "sample_edit"),
    ("muse_image", "prompt", "plain_picture"),
    ("muse_image_edit", "prompt", "sample_edit"),
    ("krea2_turbo", "prompt", "plain_picture"),
    ("krea2_turbo_edit", "prompt", "sample_edit"),
    ("recraft_v41_flash", "prompt", "plain_picture"),
    ("twinflow_z_image", "prompt", "plain_picture"),
    ("juggernaut_lightning", "prompt", "plain_picture"),
    ("juggernaut_lightning_expand", "prompt", "sample_expand"),
    ("z_image_turbo_edit", "prompt", "sample_edit"),
    ("z_image_turbo_inpaint", "prompt", "sample_inpaint"),
    ("z_image_turbo_expand", "prompt", "sample_expand"),
    ("suno_v6", "prompt", "suno_missing"),
    ("suno_v6", "style", "suno_missing"),
    ("suno_v6", "lyrics", "suno_missing"),
    ("suno_v6", "extend_lyrics", "suno_missing"),
    ("suno_v6", "negative", "suno_missing"),
    ("suno_v6", "negative_tags", "suno_missing"),
    ("suno_music", "prompt", "suno_missing"),
    ("suno_music", "style", "suno_missing"),
    ("suno_music", "lyrics", "suno_missing"),
    ("suno_music", "negative", "suno_missing"),
    ("suno_sfx", "prompt", "suno_missing"),
    ("ace_step_music", "prompt", "ace_step_missing"),
    ("ace_step_music", "lyrics", "ace_step_missing"),
    ("ace_step_music", "negative_prompt", "ace_step_missing"),
    ("minimax_music_26", "prompt", "minimax_music_missing"),
    ("minimax_music_26", "lyrics", "minimax_music_missing"),
    ("xai_tts", "prompt", "other_tts"),
    ("qwen_voice_design", "prompt", "other_voice"),
    ("qwen_voice_design", "voice_description", "other_voice"),
    ("qwen_voice_clone", "prompt", "other_voice"),
]

LEAKED_NAMES = (
    "flux",
    "klein",
    "qwen",
    "grok",
    "muse",
    "krea",
    "recraft",
    "twinflow",
    "juggernaut",
    "suno",
    "ace-step",
    "minimax",
)


def test_each_provider_model_has_its_own_refine_and_invent():
    for model, field, probe in OWNED:
        refine = instruction_for(model, field, False)
        invent = instruction_for(model, field, True)
        assert refine and invent
        assert refine != invent
        assert refine != instruction_for(probe, field, False)
        assert invent != instruction_for(probe, field, True)
        folded = f"{refine}\n{invent}".lower()
        for name in LEAKED_NAMES:
            assert name not in folded


def test_format_rules_from_the_vendor_guides():
    assert "keyword lists" in instruction_for("flux2_klein_9b", "prompt", False).lower()
    assert "quotation marks" in instruction_for("recraft_v41_flash", "prompt", False).lower()
    assert "[Chorus]" in instruction_for("suno_v6", "lyrics", False)
    assert "[Chorus]" in instruction_for("suno_v6", "lyrics", True)
    extend = instruction_for("suno_v6", "extend_lyrics", False).lower()
    assert "next sung section" in extend
    assert "complete song" not in extend


def test_empty_box_selects_invent():
    def invent_for(text):
        return not (text or "").strip()

    assert invent_for("") is True
    assert invent_for("   ") is True
    assert invent_for("a red fox") is False
    empty = instruction_for("flux2_klein_9b", "prompt", invent_for(""))
    typed = instruction_for("flux2_klein_9b", "prompt", invent_for("a red fox"))
    assert empty == instruction_for("flux2_klein_9b", "prompt", True)
    assert typed == instruction_for("flux2_klein_9b", "prompt", False)
    assert empty != typed


def test_local_video_models_keep_the_shared_video_brief():
    shared = instruction_for("other_t2v", "prompt", False)
    for model in (
        "ltx2_3_t2v",
        "ltx2_3_cinereal",
        "ltx25_int8_t2v",
        "ltx25_int8_i2v",
        "minimax_h3",
        "wan_22",
    ):
        assert instruction_for(model, "prompt", False) == shared
