"""Suno V6 studio: easy-mode pipeline, follow-ups, and 4x pricing."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import suno_v6  # noqa: E402
from credit_service import calculate_job_cost  # noqa: E402
from external_api_router import ExternalApiRouter  # noqa: E402


IDEA = "A nostalgic song about childhood memories and growing up in a small town"


def _lyrics_event():
    return suno_v6.interpret({
        "code": 200,
        "msg": "All generated successfully.",
        "data": {
            "callbackType": "complete",
            "task_id": "lyrics-task",
            "data": [
                {
                    "status": "complete",
                    "title": "Starry Night Dreams",
                    "text": "[Verse]\nMoonlight spreads across the windowsill\n[Chorus]\nIn starry dreams we find tomorrow",
                },
                {
                    "status": "complete",
                    "title": "Main Street Summer",
                    "text": "[Verse]\nLights reflect in your bright eyes",
                },
            ],
        },
    })


def _song_event(callback_type="complete"):
    return suno_v6.interpret({
        "code": 200,
        "msg": "All generated successfully.",
        "data": {
            "callbackType": callback_type,
            "task_id": "song-task",
            "data": [
                {
                    "id": "audio-1",
                    "audio_url": "https://cdn.example/a.mp3",
                    "stream_audio_url": "https://cdn.example/a-stream",
                    "image_url": "https://cdn.example/a.jpg",
                    "prompt": "[Verse]\nMoonlight",
                    "title": "Starry Night Dreams",
                    "tags": "nostalgic, acoustic",
                    "duration": 198.44,
                }
            ],
        },
    })


def test_easy_idea_starts_with_lyrics_then_song_then_covers():
    prepared = suno_v6.prepare(IDEA, {"mode": "easy"})
    assert prepared.kie_model == "ai-music-api/generate-lyrics"
    assert prepared.kie_input == {"prompt": IDEA}
    assert prepared.music["phase"] == "writing_lyrics"
    assert prepared.music["commercial_rights"] is True

    lyrics_step = suno_v6.advance(prepared.context, prepared.music, _lyrics_event())
    assert lyrics_step["action"] == "submit"
    assert lyrics_step["kie_model"] == "ai-music-api/generate"
    song = lyrics_step["kie_input"]
    assert song["custom_mode"] is False
    assert song["instrumental"] is False
    assert song["model"] == "V6"
    assert song["prompt"] == IDEA
    assert song["lyrics"].startswith("[Verse]")
    assert "title" not in song
    assert "duration" not in song
    assert "vocal_gender" not in song
    assert len(lyrics_step["music"]["lyrics"]) == 2

    song_step = suno_v6.advance(lyrics_step["context"], lyrics_step["music"], _song_event())
    assert song_step["action"] == "submit"
    assert song_step["kie_model"] == "ai-music-api/cover-generate"
    assert song_step["kie_input"] == {"task_id": "song-task"}
    assert song_step["optional"] is True
    assert song_step["music"]["tracks"][0]["id"] == "audio-1"
    assert song_step["music"]["tracks"][0]["cover_url"] == "https://cdn.example/a.jpg"
    assert song_step["music"]["phase"] == "cover_art"

    covers = suno_v6.interpret({
        "code": 200,
        "msg": "success",
        "data": {
            "taskId": "cover-task",
            "images": ["https://cdn.example/cover-1.png", "https://cdn.example/cover-2.png"],
        },
    })
    done = suno_v6.advance(song_step["context"], song_step["music"], covers)
    assert done["action"] == "finish"
    assert done["primary_url"] == "https://cdn.example/a.mp3"
    assert done["music"]["covers"] == ["https://cdn.example/cover-1.png", "https://cdn.example/cover-2.png"]
    assert done["music"]["phase"] == "complete"
    assert done["music"]["commercial_rights"] is True


def test_first_track_stays_in_progress():
    prepared = suno_v6.prepare(IDEA, {"mode": "easy", "lyrics": "[Chorus]\nAlready written"})
    event = _song_event("first")
    step = suno_v6.advance(prepared.context, prepared.music, event)
    assert step["action"] == "update"
    assert step["music"]["phase"] == "first_track"
    assert step["context"]["phase"] == "song"


def test_easy_with_lyrics_skips_the_lyrics_call():
    prepared = suno_v6.prepare(
        "summer road trip",
        {"mode": "easy", "lyrics": "[Chorus]\nWindows down\n", "style": "acoustic pop", "vocal_gender": "female", "duration": 90},
    )
    assert prepared.kie_model == "ai-music-api/generate"
    assert prepared.kie_input["custom_mode"] is False
    assert prepared.kie_input["lyrics"].startswith("[Chorus]")
    assert prepared.kie_input["style"] == "acoustic pop"
    assert prepared.kie_input["prompt"] == "summer road trip"
    assert "vocal_gender" not in prepared.kie_input
    assert "duration" not in prepared.kie_input


def test_easy_instrumental_does_not_ask_for_lyrics():
    prepared = suno_v6.prepare("warm piano in an empty church", {"mode": "easy", "instrumental": True})
    assert prepared.kie_model == "ai-music-api/generate"
    assert prepared.kie_input["custom_mode"] is False
    assert prepared.kie_input["instrumental"] is True
    assert "lyrics" not in prepared.kie_input
    assert "audio_weight" not in prepared.kie_input


def test_references_use_simple_mode_and_keep_photos():
    prepared = suno_v6.prepare(
        IDEA,
        {
            "mode": "easy",
            "lyrics": "[Verse]\nHold this photo",
            "image_urls": ["https://cdn.example/photo.png"],
            "duration": 120,
        },
    )
    assert prepared.kie_input["custom_mode"] is False
    assert prepared.kie_input["image_urls"] == ["https://cdn.example/photo.png"]
    assert "duration" not in prepared.kie_input
    assert "Song length" in prepared.music["note"]


def test_a_reference_alone_is_enough_in_easy_mode():
    prepared = suno_v6.prepare(
        "",
        {"mode": "easy", "image_urls": ["https://cdn.example/photo.png"]},
    )
    assert prepared.kie_model == "ai-music-api/generate"
    assert prepared.kie_input["custom_mode"] is False
    assert prepared.kie_input["image_urls"] == ["https://cdn.example/photo.png"]
    assert "prompt" not in prepared.kie_input
    assert "lyrics" not in prepared.kie_input


def test_custom_mode_keeps_length_voice_and_variety():
    prepared = suno_v6.prepare(
        "",
        {
            "mode": "custom",
            "style": "jazz trio",
            "lyrics": "[Chorus]\nNight train",
            "vocal_gender": "male",
            "duration": 180,
            "variety": 2,
        },
    )
    assert prepared.kie_input["custom_mode"] is True
    assert prepared.kie_input["title"]
    assert prepared.kie_input["style"] == "jazz trio"
    assert prepared.kie_input["vocal_gender"] == "m"
    assert prepared.kie_input["duration"] == 180
    assert prepared.kie_input["variety"] == 2
    try:
        suno_v6.prepare("", {"mode": "custom"})
        raise AssertionError("expected a validation error")
    except ValueError as exc:
        assert "lyrics" in str(exc).lower() or "style" in str(exc).lower()
    try:
        suno_v6.prepare("song", {"mode": "custom", "style": "jazz", "image_urls": ["https://cdn.example/a.png"]})
        raise AssertionError("expected references to be rejected")
    except ValueError as exc:
        assert "Easy" in str(exc)


def test_replace_and_persona_limits():
    try:
        suno_v6.prepare("swap the chorus", {
            "action": "replace",
            "task_id": "song-task",
            "audio_id": "audio-1",
            "tags": "jazz",
            "title": "Night",
            "full_lyrics": "[Verse]\nOld\n[Chorus]\nNew",
            "infill_start_s": 10,
            "infill_end_s": 18,
        })
        raise AssertionError("expected a 10 second minimum")
    except ValueError as exc:
        assert "10" in str(exc)

    ok = suno_v6.prepare("a softer chorus", {
        "action": "replace",
        "task_id": "song-task",
        "audio_id": "audio-1",
        "tags": "jazz",
        "title": "Night",
        "full_lyrics": "[Verse]\nOld\n[Chorus]\nNew",
        "infill_start_s": 10.5,
        "infill_end_s": 25,
        "source_duration": 100,
    })
    assert ok.kie_model == "ai-music-api/replace-section"
    assert ok.kie_input["task_id"] == "song-task"
    assert "duration" not in ok.kie_input

    try:
        suno_v6.prepare("", {
            "action": "persona",
            "task_id": "song-task",
            "audio_id": "audio-1",
            "name": "Night singer",
            "description": "A close-mic alto with a dusty chorus",
            "vocal_start": 0,
            "vocal_end": 8,
        })
        raise AssertionError("expected the persona window to be rejected")
    except ValueError as exc:
        assert "10" in str(exc)


def test_upload_is_only_for_owned_audio():
    try:
        suno_v6.prepare("", {
            "action": "stems",
            "audio_urls": ["https://cdn.example/mine.mp3"],
            "stem_type": "separate_vocal",
        })
        raise AssertionError("expected a rights confirmation")
    except ValueError as exc:
        assert "copyright" in str(exc).lower()

    stems = suno_v6.prepare("", {
        "action": "stems",
        "audio_urls": ["https://cdn.example/mine.mp3"],
        "stem_type": "separate_vocal",
        "confirm_audio_rights": True,
    })
    assert stems.kie_input["audio_url"] == "https://cdn.example/mine.mp3"
    assert "task_id" not in stems.kie_input
    assert "audio_id" not in stems.kie_input

    try:
        suno_v6.prepare("", {
            "action": "stems",
            "task_id": "song-task",
            "audio_id": "audio-1",
            "audio_urls": ["https://cdn.example/mine.mp3"],
            "confirm_audio_rights": True,
        })
        raise AssertionError("expected mixed sources to be rejected")
    except ValueError as exc:
        assert "not both" in str(exc).lower()

    replaced = suno_v6.prepare("a softer chorus", {
        "action": "replace",
        "upload_url": "https://cdn.example/mine.mp3",
        "confirm_audio_rights": True,
        "variant": "V6",
        "tags": "jazz",
        "title": "Night",
        "full_lyrics": "[Verse]\nOld\n[Chorus]\nNew",
        "infill_start_s": 10.5,
        "infill_end_s": 25,
        "source_duration": 100,
    })
    assert replaced.kie_input["upload_url"] == "https://cdn.example/mine.mp3"
    assert replaced.kie_input["model"] == "V6"
    assert "task_id" not in replaced.kie_input

    for action in ("video", "covers", "persona"):
        try:
            suno_v6.prepare("", {
                "action": action,
                "audio_urls": ["https://cdn.example/mine.mp3"],
                "confirm_audio_rights": True,
                "name": "Night",
                "description": "A close-mic alto with a dusty chorus",
            })
            raise AssertionError(f"expected {action} to reject an upload")
        except ValueError as exc:
            assert "made here" in str(exc).lower()


def test_video_stems_and_sounds_payloads():
    video = suno_v6.prepare("", {
        "action": "video",
        "source_task_id": "song-task",
        "audio_id": "audio-1",
        "author": "Ava",
        "domain_name": "songs.example",
    })
    assert video.kie_model == "ai-music-api/create-music-video"
    assert video.kie_input["author"] == "Ava"

    stems = suno_v6.prepare("", {
        "action": "stems",
        "task_id": "song-task",
        "audio_id": "audio-1",
        "stem_type": "split_stem_advanced",
        "stem_name": "piano",
    })
    assert stems.kie_input["stem_name"] == "Piano"
    assert stems.kie_input["type"] == "split_stem_advanced"

    sounds = suno_v6.prepare("soft rain on a tin roof", {
        "action": "sounds",
        "variant": "v6_mini",
        "sound_loop": True,
        "sound_tempo": 72,
        "sound_key": "d#m",
    })
    assert sounds.kie_input["model"] == "V6_MINI"
    assert sounds.kie_input["sound_key"] == "D#m"
    assert sounds.kie_input["sound_loop"] is True


def test_callbacks_for_video_and_stems():
    video = suno_v6.interpret({
        "code": 200,
        "msg": "success",
        "data": {"task_id": "video-task", "video_url": "https://cdn.example/clip.mp4"},
    })
    assert video["disposition"] == "success"
    assert video["kind"] == "video"
    stems = suno_v6.interpret({
        "code": 200,
        "msg": "vocal separation generated successfully.",
        "data": {
            "task_id": "stem-task",
            "vocal_removal_info": {
                "instrumental_url": "https://cdn.example/inst.mp3",
                "vocal_url": "https://cdn.example/vox.mp3",
                "origin_url": "",
            },
        },
    })
    assert stems["kind"] == "stems"
    assert [item["name"] for item in stems["stems"]] == ["instrumental", "vocal"]
    pending = suno_v6.interpret({"code": 200, "data": {"state": "generating", "taskId": "song-task"}})
    assert pending["disposition"] == "pending"


def test_lyrics_failure_still_composes_the_song():
    prepared = suno_v6.prepare(IDEA, {"mode": "easy"})
    failed = {"disposition": "failed", "kind": "", "error": "Try a different idea", "tracks": [], "lyrics": [], "covers": [], "stems": [], "video_url": "", "persona": None, "task_id": ""}
    step = suno_v6.advance(prepared.context, prepared.music, failed)
    assert step["action"] == "submit"
    assert step["kie_model"] == "ai-music-api/generate"
    assert step["kie_input"]["style"]


def test_prices_match_the_sell_schedule():
    with patch("credit_service.get_model_cost_info", return_value={"cost": 65, "offer": 0}), \
         patch("credit_service._resolve_effective_offer", return_value=(0, {})):
        assert calculate_job_cost("suno_v6", {"action": "song"}) == 65
        assert calculate_job_cost("suno_v6", {}) == 65
        assert calculate_job_cost("suno_v6", {"action": "sounds"}) == 15
        assert calculate_job_cost("suno_v6", {"action": "replace"}) == 65
        assert calculate_job_cost("suno_v6", {"action": "lyrics"}) == 5
        assert calculate_job_cost("suno_v6", {"action": "covers"}) == 10
        assert calculate_job_cost("suno_v6", {"action": "video"}) == 20
        assert calculate_job_cost("suno_v6", {"action": "stems"}) == 20
        assert calculate_job_cost("suno_v6", {"action": "stems", "stem_type": "split_stem"}) == 40
        assert calculate_job_cost("suno_v6", {"action": "stems", "stem_type": "split_stem_advanced"}) == 30
        assert calculate_job_cost("suno_v6", {"action": "persona"}) == 10
        assert calculate_job_cost("suno_v6", {"action": "voice"}) == 2
        assert suno_v6.SONG_USER_CREDITS == 65


def test_router_submits_the_lyrics_task():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "external_api_config.json"
        path.write_text(json.dumps({
            "models": {
                "suno_v6": {
                    "route_policy": "external_only",
                    "provider": {"type": "kie", "model_id": "ai-music-api/generate"},
                    "kie_profile": "suno_v6",
                    "param_map": {},
                }
            }
        }), encoding="utf-8")
        router = ExternalApiRouter(str(path))
        with patch.dict("os.environ", {"CONTROLLER_PUBLIC_URL": "https://api.example.test", "KIE_API_KEY": "test-key"}), \
             patch("kie_client.ensure_credits", return_value=(True, "")), \
             patch("kie_client.create_task", return_value="task-1") as create:
            result = router.execute("suno_v6", IDEA, {"mode": "easy"})
        assert result.pending is True
        assert result.external_task_id == "task-1"
        assert create.call_args.args[0] == "ai-music-api/generate-lyrics"
        assert result.raw["music"]["phase"] == "writing_lyrics"
        assert result.raw["suno_context"]["phase"] == "lyrics"
