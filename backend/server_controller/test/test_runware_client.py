"""Runware task assembly and response parsing. No network."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from external_api_router import (  # noqa: E402
    ExternalApiRouter,
    ExternalExecutionResult,
    carry_moderation_flag,
)
from runware_client import (  # noqa: E402
    assemble_task,
    failure_message,
    is_processing,
    media_url,
    snap_size,
    FLUX_SIZES,
    MUSE_SIZES,
    RECRAFT_SIZES,
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


def test_muse_forwards_canvas_settings_and_can_turn_them_off():
    task = assemble_task(
        _cfg("muse_image", "meta:muse@image", "imageInference"),
        {
            "positivePrompt": "a clinic",
            "aspect_ratio": "1:1",
            "thinkingLevel": "low",
            "webSearch": False,
            "imageSearch": True,
            "shell": False,
            "moderation": "auto",
        },
        "99999999-9999-4999-8999-999999999999",
    )
    assert task["settings"] == {
        "thinkingLevel": "low",
        "webSearch": False,
        "imageSearch": True,
        "shell": False,
        "moderation": "auto",
    }


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
    assert "numberResults" not in p_image

    esrgan = assemble_task(
        _cfg("upscale_esrgan", "runware:504@1", "upscale"),
        {"image": "https://cdn.example/in.jpg", "upscaleFactor": 3},
        "66666666-6666-4666-8666-666666666666",
    )
    assert esrgan["upscaleFactor"] == 2
    assert "numberResults" not in esrgan


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
    assert arranged["CFGScale"] == 4

    remix = assemble_task(
        _cfg("music_ace", "runware:ace-step@v1.5-xl-sft", "audioInference", "MP3"),
        {
            "positivePrompt": "Reinterpret the track as a quiet piano cover",
            "duration": 90,
            "audio": "https://cdn.example/song.mp3",
            "strength": 0.55,
        },
        "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
    )
    assert "duration" not in remix
    assert remix["inputs"]["audio"] == "https://cdn.example/song.mp3"
    assert remix["strength"] == 0.55
    assert "repaintingStart" not in remix.get("settings", {})

    repaint = assemble_task(
        _cfg("music_ace", "runware:ace-step@v1.5-xl-sft", "audioInference", "MP3"),
        {
            "positivePrompt": "Replace the middle passage with a dry marimba figure",
            "audio": "https://cdn.example/song.mp3",
            "strength": 0.75,
            "repaintingStart": 8,
            "repaintingEnd": 18,
        },
        "ffffffff-ffff-4fff-8fff-ffffffffffff",
    )
    assert repaint["strength"] == 0.75
    assert repaint["settings"]["repaintingStart"] == 8
    assert repaint["settings"]["repaintingEnd"] == 18

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


def _task(model, prompt, params, task_uuid="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"):
    router = ExternalApiRouter()
    cfg = router.get_model_config(model)
    args = router._build_arguments(cfg, prompt, params)
    return assemble_task(cfg, args, task_uuid)


def test_new_image_models_build_the_documented_task():
    glimpse = _task("twinflow_z_image", "a fox", {"aspect_ratio": "1:1"})
    assert glimpse["model"] == "runware:twinflow-z-image-turbo@0"
    assert glimpse["steps"] == 4
    assert (glimpse["width"], glimpse["height"]) == FLUX_SIZES["1:1"]
    assert "inputs" not in glimpse

    poster = _task("recraft_v41_flash", "a poster", {"aspect_ratio": "16:9"})
    assert poster["model"] == "recraft:v4.1@flash"
    assert (poster["width"], poster["height"]) == RECRAFT_SIZES["16:9"]
    assert (poster["width"], poster["height"]) == (1344, 768)
    assert "steps" not in poster

    palette = _task(
        "krea2_turbo_edit",
        "match the coat",
        {"init_image": [f"https://cdn.example/{i}.jpg" for i in range(5)]},
    )
    assert palette["model"] == "krea:krea@2-turbo"
    assert palette["steps"] == 8
    assert palette["inputs"]["referenceImages"] == [f"https://cdn.example/{i}.jpg" for i in range(4)]

    touch = _task("z_image_turbo_edit", "change the sky", {"init_image": "https://cdn.example/a.jpg"})
    assert touch["model"] == "runware:z-image@turbo"
    assert touch["steps"] == 8
    assert touch["inputs"] == {"seedImage": "https://cdn.example/a.jpg"}
    assert "width" not in touch and "height" not in touch

    filled = _task("z_image_turbo_inpaint", "  ", {
        "init_image": "https://cdn.example/a.jpg",
        "mask_image": "https://cdn.example/mask.png",
    })
    assert filled["positivePrompt"] == "Fill the painted area so it matches the surrounding image."
    assert filled["inputs"]["seedImage"] == "https://cdn.example/a.jpg"
    assert filled["inputs"]["maskImage"] == "https://cdn.example/mask.png"
    assert "width" not in filled and "steps" in filled and filled["steps"] == 8

    vista = _task("juggernaut_lightning_expand", "", {
        "init_image": "https://cdn.example/a.jpg",
        "aspect_ratio": "9:16",
    })
    assert vista["model"] == "rundiffusion:110@101"
    assert vista["steps"] == 4
    assert vista["inputs"]["seedImage"] == "https://cdn.example/a.jpg"
    assert (vista["width"], vista["height"]) == FLUX_SIZES["9:16"]
    assert vista["positivePrompt"].startswith("Extend the scene")

    widened = _task("z_image_turbo_expand", "more sky", {
        "init_image": ["https://cdn.example/a.jpg", "https://cdn.example/b.jpg"],
        "aspect_ratio": "16:9",
    })
    assert widened["inputs"]["seedImage"] == "https://cdn.example/a.jpg"
    assert (widened["width"], widened["height"]) == FLUX_SIZES["16:9"]
    assert widened["steps"] == 8


def test_seed_and_mask_reject_non_urls():
    try:
        _task("z_image_turbo_edit", "edit", {"init_image": "{{init_image}}"})
    except ValueError as exc:
        assert "image" in str(exc).lower()
    else:
        raise AssertionError("placeholder seed was accepted")

    try:
        _task("z_image_turbo_inpaint", "edit", {
            "init_image": "https://cdn.example/a.jpg",
            "mask_image": "masks/local.png",
        })
    except ValueError as exc:
        assert "mask" in str(exc).lower()
    else:
        raise AssertionError("non-url mask was accepted")


def test_qwen_stays_on_runware_unless_this_job_was_flagged():
    router = ExternalApiRouter()
    seen = {}

    def runware(cfg, args, prompt, params):
        seen["type"] = cfg["provider"]["type"]
        seen["model"] = cfg["provider"]["model_id"]
        seen["args"] = args
        return ExternalExecutionResult(success=True, pending=True, external_task_id="rw", provider="runware")

    def kie(cfg, args, prompt, params):
        seen["type"] = cfg["provider"]["type"]
        seen["model"] = cfg["provider"]["model_id"]
        seen["args"] = args
        return ExternalExecutionResult(success=True, pending=True, external_task_id="kie", provider="kie")

    router._run_runware = runware
    router._run_kie = kie

    clean = router.execute("qwen2_1_t2i", "a fox", {"aspect_ratio": "16:9"})
    assert clean.provider == "runware"
    assert seen == {
        "type": "runware",
        "model": "alibaba:qwen-image@2.1-turbo",
        "args": seen["args"],
    }
    assert "steps" not in seen["args"]
    task = assemble_task(router.resolve_model_config("qwen2_1_t2i", {}), seen["args"], "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    assert "steps" not in task
    assert (task["width"], task["height"]) == FLUX_SIZES["16:9"]

    edited = _task("qwen2_1_i2i", "change the jacket", {
        "init_image": ["https://cdn.example/a.jpg", "https://cdn.example/b.jpg"],
        "aspect_ratio": "auto",
    })
    assert edited["model"] == "alibaba:qwen-image@2.1-turbo"
    assert edited["inputs"]["referenceImages"] == [
        "https://cdn.example/a.jpg",
        "https://cdn.example/b.jpg",
    ]
    assert "steps" not in edited
    assert "width" not in edited and "height" not in edited

    seen.clear()
    flagged = router.execute("qwen2_1_i2i", "change the jacket", {
        "_moderation_flagged": True,
        "init_image": "https://cdn.example/a.jpg",
    })
    assert flagged.provider == "kie"
    assert seen["type"] == "kie"
    assert seen["model"] == "qwen2-1/image-to-image"
    assert seen["args"]["image_urls"] == ["https://cdn.example/a.jpg"]

    seen.clear()
    not_this_job = router.execute("qwen2_1_t2i", "a fox", {"_moderation_flagged": "true"})
    assert not_this_job.provider == "runware"
    assert seen["model"] == "alibaba:qwen-image@2.1-turbo"

    prepared = carry_moderation_flag(
        {"_moderation_flagged": True, "_batch_id": "b"},
        {"aspect_ratio": "1:1"},
    )
    assert prepared["_moderation_flagged"] is True
    assert "_batch_id" not in prepared
    assert "_moderation_flagged" not in carry_moderation_flag({"_moderation_flagged": False}, {})


def test_flagged_qwen_text_uses_the_kie_text_model():
    router = ExternalApiRouter()
    seen = {}

    def kie(cfg, args, prompt, params):
        seen["model"] = cfg["provider"]["model_id"]
        return ExternalExecutionResult(success=True, pending=True, external_task_id="kie", provider="kie")

    router._run_kie = kie
    router._run_runware = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runware"))
    result = router.execute("qwen2_1_t2i", "a fox", {"_moderation_flagged": True, "aspect_ratio": "1:1"})
    assert result.provider == "kie"
    assert seen["model"] == "qwen2-1/text-to-image"
