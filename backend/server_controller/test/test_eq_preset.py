"""Equalizer preset ids stay on the VLC list and its style aliases."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import asset_prompt_vlm  # noqa: E402


def test_only_an_exact_id_is_kept():
    assert asset_prompt_vlm.normalize_eq_preset("rock") == "rock"
    assert asset_prompt_vlm.normalize_eq_preset("  hiphop  ") == "hiphop"
    assert asset_prompt_vlm.normalize_eq_preset("Rock") == "rock"
    assert asset_prompt_vlm.normalize_eq_preset("rock.") == "flat"
    assert asset_prompt_vlm.normalize_eq_preset("use rock") == "flat"
    assert asset_prompt_vlm.normalize_eq_preset("") == "flat"
    assert asset_prompt_vlm.normalize_eq_preset("bassboost") == "flat"


def test_prompt_lists_every_id_and_alias():
    prompt = asset_prompt_vlm.eq_preset_prompt("Night Drive", "trap", "808", "yeah")
    for preset_id in asset_prompt_vlm.EQ_PRESET_IDS:
        assert preset_id in prompt
    assert "hiphop: fullbass" in prompt
    assert "bollywood: pop" in prompt
    assert "Night Drive" in prompt
    assert len(asset_prompt_vlm.EQ_PRESET_ALLOWLIST) == len(asset_prompt_vlm.EQ_PRESET_IDS)
