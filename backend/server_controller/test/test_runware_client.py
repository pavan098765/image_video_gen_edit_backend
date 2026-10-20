"""Runware task assembly and response parsing. No network."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from runware_client import (  # noqa: E402
    assemble_task,
    failure_message,
    is_processing,
    media_url,
    snap_size,
    MUSE_SIZES,
)


def _cfg(profile, model, task_type, output_format="PNG"):
    return {
        "provider": {"model_id": model},
        "runware": {
            "profile": profile,
            "task_type": task_type,
            "output_format": output_format,
        },
    }


def test_flux_snaps_widescreen_and_keeps_step_multiple():
    task = assemble_task(
        _cfg("flux_image", "runware:400@2", "imageInference"),
        {"positivePrompt": "a fox", "width": 896, "height": 512},
        "11111111-1111-4111-8111-111111111111",
    )
    assert task["model"] == "runware:400@2"
    assert task["taskType"] == "imageInference"
    assert task["width"] == 1280
    assert task["height"] == 720
    assert task["width"] % 16 == 0 and task["height"] % 16 == 0
    assert task["steps"] == 4
    assert task["deliveryMethod"] == "async"
    assert "inputs" not in task


def test_muse_uses_only_documented_frames_and_drops_size_when_editing():
    made = assemble_task(
        _cfg("muse_image", "meta:muse@image", "imageInference"),
        {"positivePrompt": "a clinic", "aspect_ratio": "16:9"},
        "22222222-2222-4222-8222-222222222222",
    )
    assert (made["width"], made["height"]) == MUSE_SIZES["16:9"]

    edited = assemble_task(
        _cfg("muse_image", "meta:muse@image", "imageInference"),
        {
            "positivePrompt": "change the jacket",
            "width": 896,
            "height": 896,
            "referenceImages": ["https://cdn.example/a.jpg", "not-a-url"],
        },
        "33333333-3333-4333-8333-333333333333",
    )
    assert "width" not in edited and "height" not in edited
    assert edited["inputs"]["referenceImages"] == ["https://cdn.example/a.jpg"]


def test_muse_rejects_a_non_url_reference():
    try:
        assemble_task(
            _cfg("muse_image", "meta:muse@image", "imageInference"),
            {"positivePrompt": "edit", "referenceImages": ["{{init_image}}"]},
            "44444444-4444-4444-8444-444444444444",
        )
    except ValueError as exc:
        assert "downloadable" in str(exc)
    else:
        raise AssertionError("placeholder reference was accepted")


def test_upscale_factors_stay_inside_each_model():
    p_image = assemble_task(
        _cfg("upscale_p", "prunaai:p-image@upscale", "upscale"),
        {
            "image": "https://cdn.example/in.jpg",
            "upscaleFactor": 4,
            "enhanceDetails": True,
            "realism": False,
        },
        "55555555-5555-4555-8555-555555555555",
    )
    assert p_image["upscaleFactor"] == 4
    assert p_image["inputs"]["image"] == "https://cdn.example/in.jpg"
    assert p_image["settings"] == {"enhanceDetails": True}

    esrgan = assemble_task(
        _cfg("upscale_esrgan", "runware:504@1", "upscale"),
        {"image": "https://cdn.example/in.jpg", "upscaleFactor": 3},
        "66666666-6666-4666-8666-666666666666",
    )
    assert esrgan["upscaleFactor"] == 2


def test_speech_and_music_shapes():
    speech = assemble_task(
        _cfg("speech_xai", "xai:tts@0", "audioInference", "MP3"),
        {"text": "Hello there", "voice": "orion", "language": "en"},
        "77777777-7777-4777-8777-777777777777",
    )
    assert speech["outputFormat"] == "MP3"
    assert speech["speech"] == {"text": "Hello there", "voice": "orion", "language": "en"}

    design = assemble_task(
        _cfg("speech_design", "alibaba:qwen@3-tts-1.7b-voicedesign", "audioInference", "MP3"),
        {"text": "Hello", "positivePrompt": "A calm narrator"},
        "88888888-8888-4888-8888-888888888888",
    )
    assert design["speech"]["voice"] == "design"
    assert design["positivePrompt"] == "A calm narrator"

    clone = assemble_task(
        _cfg("speech_clone", "alibaba:qwen@3-tts-1.7b-base", "audioInference", "MP3"),
        {"text": "Hello", "audio": "https://cdn.example/voice.mp3"},
        "99999999-9999-4999-8999-999999999999",
    )
    assert clone["speech"]["voice"] == "clone"
    assert clone["inputs"]["audio"] == "https://cdn.example/voice.mp3"
    assert clone["settings"] == {"xVectorOnly": True}

    ace = assemble_task(
        _cfg("music_ace", "runware:ace-step@v1.5-xl-sft", "audioInference", "MP3"),
        {"positivePrompt": "soft piano", "duration": 12, "lyrics": "too short"},
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )
    assert ace["duration"] == 30
    assert "settings" not in ace

    arranged = assemble_task(
        _cfg("music_ace", "runware:ace-step@v1.5-xl-sft", "audioInference", "MP3"),
        {
            "positivePrompt": "soft piano",
            "duration": 90,
            "lyrics": "[Verse]\nHome again tonight",
            "bpm": 96,
            "keyScale": "D major",
            "timeSignature": 3,
            "vocalLanguage": "en",
            "negativePrompt": "no drums",
        },
        "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
    )
    assert arranged["duration"] == 90
    assert arranged["settings"]["bpm"] == 96
    assert arranged["settings"]["keyScale"] == "D major"
    assert arranged["settings"]["timeSignature"] == 3
    assert arranged["negativePrompt"] == "no drums"

    instrumental = assemble_task(
        _cfg("music_minimax", "minimax:music@2.6", "audioInference", "MP3"),
        {"positivePrompt": "city lights"},
        "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    )
    assert instrumental["settings"] == {"instrumental": True}

    vocal = assemble_task(
        _cfg("music_minimax", "minimax:music@2.6", "audioInference", "MP3"),
        {"positivePrompt": "anthem", "lyrics": "[Chorus]\nWe stay"},
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    )
    assert vocal["settings"]["lyrics"].startswith("[Chorus]")
    assert vocal["settings"]["instrumental"] is False
    assert vocal["settings"]["lyricsOptimizer"] is False


def test_response_parser_covers_webhook_poll_and_errors():
    webhook = {
        "taskType": "imageInference",
        "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
        "imageURL": "https://im.runware.ai/image/out.jpg",
    }
    assert media_url(webhook).endswith("out.jpg")
    assert failure_message(webhook) is None

    processing = {
        "data": [
            {
                "taskType": "audioInference",
                "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
                "status": "processing",
            }
        ]
    }
    assert is_processing(processing)
    assert media_url(processing) is None

    failed = {
        "errors": [
            {
                "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
                "message": "The external provider did not respond",
            }
        ]
    }
    assert failure_message(failed).startswith("The external provider")
    assert is_processing(failed) is False

    audio = {"data": [{"audioURL": "https://am.runware.ai/audio/out.mp3", "status": "success"}]}
    assert media_url(audio).endswith("out.mp3")


def test_snap_falls_back_to_square():
    assert snap_size(None, None, MUSE_SIZES) == MUSE_SIZES["1:1"]
