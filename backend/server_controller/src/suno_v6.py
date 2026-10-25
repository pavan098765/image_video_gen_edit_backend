"""Suno V6 music studio on Kie.

One app model, ``suno_v6``, drives the Kie music endpoints: lyrics, full songs,
sound loops, cover art, section replace, stems, personas, custom voices, and
music videos. Easy mode writes lyrics first, then the song, then extra covers.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

MODEL_NAME = "suno_v6"
# Sell price. 1 credit is 0.3 INR, so a song is about 20 INR.
# Kie bills about 6 INR ($0.06) per song, and that includes commercial rights.
SONG_USER_CREDITS = 65

VARIANTS = ("V6", "V6_MINI", "V6_WILD")
COMMERCIAL_ACTIONS = ("song", "sounds", "replace")

# Smaller actions keep the same ratio to a song. Override with tool_config actionCosts.
ACTION_CREDITS = {
    "song": SONG_USER_CREDITS,
    "sounds": 15,
    "replace": SONG_USER_CREDITS,
    "lyrics": 5,
    "covers": 10,
    "video": 20,
    "stems": 20,
    "stems_split": 40,
    "stems_advanced": 30,
    "persona": 10,
    "voice": 2,
}

KIE_MODELS = {
    "song": "ai-music-api/generate",
    "lyrics": "ai-music-api/generate-lyrics",
    "covers": "ai-music-api/cover-generate",
    "sounds": "ai-music-api/sounds",
    "replace": "ai-music-api/replace-section",
    "stems": "ai-music-api/separate-vocals",
    "video": "ai-music-api/create-music-video",
    "persona": "ai-music-api/generate-persona",
    "voice": "ai-music-api/create-voice",
}

PHASE_LABELS = {
    "writing_lyrics": "Writing lyrics",
    "composing": "Composing your song",
    "first_track": "First version is ready",
    "cover_art": "Designing cover art",
    "separating": "Separating stems",
    "rendering_video": "Rendering the music video",
    "replacing": "Replacing that section",
    "creating_voice": "Saving the voice",
    "creating_persona": "Creating the persona",
    "complete": "Ready",
}

SOUND_KEYS = (
    "Any", "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
SKILL_LEVELS = ("beginner", "intermediate", "advanced", "professional")
STEM_TYPES = ("separate_vocal", "split_stem", "split_stem_advanced")
STEM_NAMES = (
    "Lead Vocal", "Drum Kit", "Kick", "Snare", "Risers", "Bass", "Backing Vocals", "Piano",
    "Electric Guitar", "Percussion", "String Section", "Synth", "Acoustic Guitar",
    "Sound Effects", "Synth Pad", "Synth Bass", "Guitar", "Brass Section", "Organ",
    "Electronic Drum Kit", "Lead Electric Guitar", "Synth Keys", "Rhythm Electric Guitar",
    "Electric Piano", "Upright Bass", "Keyboards", "Distorted Electric Guitar", "Synth Strings",
    "Synth Lead", "Woodwinds", "Rhythm Acoustic Guitar", "Flute", "Harp", "Tambourine",
    "Trumpet", "Arpeggiator", "Accordion", "Fiddle", "Pedal Steel Guitar", "Synth Voice",
    "Violin", "Digital Piano", "Synth Brass", "Mandolin", "Choir", "Banjo", "Bells",
    "Clarinet", "Tenor Saxophone", "Trombone", "Shaker", "French Horn", "Glockenspiel",
    "Electric Bass", "Cello", "Timpani", "Harmonica", "Marimba", "Vibraphone",
    "Lap Steel Guitar", "Saxophone", "Orchestra", "Horns", "Cymbals", "Hand Clap", "Oboe",
    "Celesta", "Congas", "Drone", "Alto Saxophone", "Double Bass", "Ukulele", "Harpsichord",
    "Baritone Saxophone", "Xylophone", "Tuba", "Bass Guitar", "Whistle", "Lead Guitar",
    "Rhodes", "808", "Bongos", "Bassoon", "Cowbell", "Viola", "Sitar", "Steel Drums",
    "Piccolo", "Theremin", "Bagpipes", "Hi-Hat", "Music Box", "Melodica", "Tabla", "Koto",
    "Djembe", "Taiko", "Didgeridoo",
)

_PENDING_STATES = {
    "waiting", "queuing", "queueing", "generating", "processing", "pending", "running",
    "submitted", "in_progress",
}
_FAIL_STATES = {"fail", "failed", "error", "cancelled", "canceled"}
_OK_STATES = {"success", "succeed", "succeeded", "completed", "complete", "done"}


class Prepared:
    def __init__(
        self,
        kie_model: str,
        kie_input: Dict[str, Any],
        context: Dict[str, Any],
        music: Dict[str, Any],
        kind: str,
    ):
        self.kie_model = kie_model
        self.kie_input = kie_input
        self.context = context
        self.music = music
        self.kind = kind  # async, persona, or voice


def user_credit_cost(params: Optional[Dict[str, Any]], cost_info: Optional[Dict[str, Any]] = None) -> int:
    """User credits for one request. A full song is 65."""
    action = normalize_action((params or {}).get("action"))
    key = action
    if action == "stems":
        stem_type = _stem_type(params or {})
        if stem_type == "split_stem":
            key = "stems_split"
        elif stem_type == "split_stem_advanced":
            key = "stems_advanced"
        else:
            key = "stems"
    overrides = {}
    if isinstance(cost_info, dict) and isinstance(cost_info.get("actionCosts"), dict):
        overrides = cost_info["actionCosts"]
    if key in overrides:
        return max(1, int(overrides[key]))
    return ACTION_CREDITS[key]


def normalize_action(value: Any) -> str:
    raw = str(value or "song").strip().lower().replace("-", "_")
    aliases = {
        "song": "song",
        "generate": "song",
        "music": "song",
        "lyrics": "lyrics",
        "lyric": "lyrics",
        "covers": "covers",
        "cover": "covers",
        "cover_art": "covers",
        "sounds": "sounds",
        "sound": "sounds",
        "sfx": "sounds",
        "replace": "replace",
        "replace_section": "replace",
        "section": "replace",
        "stems": "stems",
        "stem": "stems",
        "separate": "stems",
        "video": "video",
        "music_video": "video",
        "mp4": "video",
        "persona": "persona",
        "voice": "voice",
    }
    action = aliases.get(raw)
    if not action:
        raise ValueError("Choose a music action: song, lyrics, covers, video, stems, replace, sounds, persona, or voice.")
    return action


def prepare(prompt: str, params: Optional[Dict[str, Any]], *, for_submit: bool = False) -> Prepared:
    """Build the first Kie task for this request. Raises ValueError on bad input."""
    params = params or {}
    action = normalize_action(params.get("action"))
    variant = _variant(params.get("variant") or params.get("model_variant"))
    easy = _is_easy(params)
    idea = _clip(prompt or params.get("prompt") or "", 3000)
    style = _clip(params.get("style") or "", 1000)
    title = _clip(params.get("title") or "", 80)
    lyrics = _clip(params.get("lyrics") or "", 5000)
    negative = _clip(params.get("negative_tags") or params.get("negativeTags") or "", 1000)
    instrumental = _bool(params.get("instrumental"), False)
    media = _collect_media(params, strict=not for_submit)
    _reject_too_many(media, style=style, lyrics=lyrics, will_add_lyrics=False)

    context: Dict[str, Any] = {
        "action": action,
        "easy": easy,
        "variant": variant,
        "instrumental": instrumental,
        "prompt": idea,
        "style": style,
        "title": title,
        "lyrics": lyrics,
        "negative_tags": negative,
        "vocal_gender": _vocal_gender(params.get("vocal_gender")),
        "duration": _duration(params.get("duration")),
        "style_weight": _unit(params.get("style_weight"), "style_weight"),
        "weirdness_constraint": _unit(params.get("weirdness_constraint"), "weirdness"),
        "audio_weight": _unit(params.get("audio_weight"), "audio_weight"),
        "variety": _variety(params.get("variety")),
        "persona_id": _clip(params.get("persona_id") or "", 200),
        "persona_model": _persona_model(params.get("persona_model")),
        "image_urls": media["images"],
        "video_urls": media["videos"],
        "audio_urls": media["audios"],
        "source_task_id": _clip(params.get("source_task_id") or params.get("task_id") or "", 128),
        "audio_id": _clip(params.get("audio_id") or "", 128),
        "source_duration": _optional_float(params.get("source_duration")),
        "phase": "song",
    }
    if context["persona_id"] and not context["persona_model"]:
        context["persona_model"] = "style_persona"

    if action == "song":
        return _prepare_song(context, for_submit=for_submit)
    if action == "lyrics":
        return _simple(context, "lyrics", "writing_lyrics", {"prompt": _lyrics_prompt(idea)})
    if action == "covers":
        _reject_upload(context, "New cover art")
        if not context["source_task_id"]:
            raise ValueError("Pick a song you made here for new cover art.")
        return _simple(context, "covers", "cover_art", {"task_id": context["source_task_id"]})
    if action == "video":
        _reject_upload(context, "A music video")
        _require_source(context, "A music video")
        payload = {"task_id": context["source_task_id"], "audio_id": context["audio_id"]}
        author = _clip(params.get("author") or "", 50)
        domain = _clip(params.get("domain_name") or params.get("domain") or "", 50)
        if author:
            payload["author"] = author
        if domain:
            payload["domain_name"] = domain
        return _simple(context, "video", "rendering_video", payload)
    if action == "stems":
        return _prepare_stems(context, params)
    if action == "replace":
        return _prepare_replace(context, params)
    if action == "sounds":
        return _prepare_sounds(context, params)
    if action == "persona":
        return _prepare_persona(context, params)
    if action == "voice":
        return _prepare_voice(context, params)
    raise ValueError("Choose a music action: song, lyrics, covers, video, stems, replace, sounds, persona, or voice.")


def callback_summary(payload: Dict[str, Any]) -> str:
    """One log line for a Kie music callback. URLs are omitted."""
    data, result_json = _unwrap(payload if isinstance(payload, dict) else {})
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    obj = result_json.get("resultObject") if isinstance(result_json.get("resultObject"), dict) else {}
    callback = str(data.get("callbackType") or data.get("callback_type") or "").strip().lower()
    state = str(data.get("state") or data.get("status") or "").strip().lower()
    counts = []
    raw_ids = []
    for name, source in (("data", data), ("response", response), ("result", obj), ("json", result_json)):
        if not isinstance(source, dict):
            continue
        for key in ("data", "sunoData", "suno_data", "clips"):
            items = source.get(key)
            if not isinstance(items, list):
                continue
            counts.append(f"{name}.{key}={len(items)}")
            for item in items:
                if not isinstance(item, dict):
                    continue
                clip_id = _first_text(item.get("id"), item.get("audio_id"), item.get("audioId")) or "-"
                audio = bool(_first_text(item.get("audio_url"), item.get("audioUrl"), item.get("source_audio_url")))
                stream = bool(_first_text(
                    item.get("stream_audio_url"), item.get("stream_url"),
                    item.get("streamAudioUrl"), item.get("source_stream_audio_url"),
                ))
                raw_ids.append(f"{clip_id}:audio={int(audio)}:stream={int(stream)}")
    event = interpret(payload if isinstance(payload, dict) else {})
    parsed = [
        f"{track.get('id') or '-'}:audio={int(bool(track.get('audio_url')))}:stream={int(bool(track.get('stream_url')))}"
        for track in event.get("tracks") or []
    ]
    return (
        f"code={payload.get('code') if isinstance(payload, dict) else '-'} "
        f"msg={payload.get('msg') if isinstance(payload, dict) else '-'} "
        f"callback={callback or '-'} state={state or '-'} "
        f"raw={counts or ['none']} raw_clips={raw_ids or ['none']} "
        f"parsed={len(parsed)} {parsed or ['none']} "
        f"disposition={event.get('disposition')} kind={event.get('kind') or '-'}"
    )


def interpret(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Read a Kie callback or recordInfo body."""
    data, result_json = _unwrap(payload)
    state = str(data.get("state") or data.get("status") or "").strip().lower()
    callback = str(data.get("callbackType") or data.get("callback_type") or "").strip().lower()
    task_id = _first_text(
        data.get("taskId"), data.get("task_id"), payload.get("taskId"), payload.get("task_id")
    )
    code = payload.get("code")
    tracks = _tracks_from(data, result_json)
    lyrics = _lyrics_from(data, result_json)
    covers = _covers_from(data, result_json)
    stems = _stems_from(data, result_json)
    video_url = _video_from(data, result_json)
    persona = _persona_from(data, result_json)

    event = {
        "disposition": "pending",
        "kind": "",
        "task_id": task_id,
        "tracks": tracks,
        "lyrics": lyrics,
        "covers": covers,
        "stems": stems,
        "video_url": video_url,
        "persona": persona,
        "error": "",
    }
    if state in _FAIL_STATES or (code not in (None, 200, "200") and not _has_assets(event)):
        event["disposition"] = "failed"
        event["error"] = _fail_text(payload, data)
        return event
    if callback in ("text", "first"):
        event["disposition"] = "progress"
        event["kind"] = "tracks" if tracks else "lyrics" if lyrics else ""
        return event
    if state in _PENDING_STATES and callback != "complete":
        if tracks or lyrics:
            event["disposition"] = "progress"
            event["kind"] = "tracks" if tracks else "lyrics"
            return event
        event["disposition"] = "pending"
        return event

    if persona:
        event["kind"] = "persona"
    elif video_url and not tracks:
        event["kind"] = "video"
    elif covers and not tracks:
        event["kind"] = "covers"
    elif stems and not tracks:
        event["kind"] = "stems"
    elif lyrics and not tracks:
        event["kind"] = "lyrics"
    elif tracks:
        event["kind"] = "tracks"

    done = callback == "complete" or state in _OK_STATES or bool(event["kind"])
    if done and event["kind"]:
        event["disposition"] = "success"
        return event
    if done and not event["kind"]:
        event["disposition"] = "failed"
        event["error"] = _fail_text(payload, data) or "The music service finished without a file."
        return event
    if code not in (None, 200, "200"):
        event["disposition"] = "failed"
        event["error"] = _fail_text(payload, data)
        return event
    event["disposition"] = "pending"
    return event


def advance(context: Dict[str, Any], music: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Decide the next pipeline step after a callback. Does not call the network."""
    ctx = copy.deepcopy(context or {})
    current = copy.deepcopy(music or _blank_music(ctx, "composing"))
    disposition = event.get("disposition") or "pending"
    kind = event.get("kind") or ""
    phase = ctx.get("phase") or "song"

    if disposition == "pending":
        playable = any(_track_has_audio(track) for track in current.get("tracks") or [])
        # A long wait can accept the one clip that already arrived, even when
        # the latest poll has no new file yet.
        if not (ctx.get("accept_single_track") and playable):
            return _step("update", ctx, current)
        kind = "tracks"
    if disposition == "progress":
        _apply_assets(current, event)
        playable = any(_track_has_audio(track) for track in current.get("tracks") or [])
        if playable and phase == "song":
            current["phase"] = "first_track"
            current["label"] = PHASE_LABELS["first_track"]
            if hold_for_second_version(ctx, current):
                return _step("update", ctx, current)
            if ctx.get("accept_single_track"):
                kind = "tracks"
            else:
                return _step("update", ctx, current)
        else:
            return _step("update", ctx, current)
    if disposition == "failed":
        if phase == "covers":
            current["phase"] = "complete"
            current["label"] = PHASE_LABELS["complete"]
            current["note"] = "Extra covers were not ready. Each version still has its own cover."
            return _step("finish", ctx, current, primary_url=_primary_audio(current))
        if phase == "lyrics":
            ctx["phase"] = "song"
            ctx["lyrics_error"] = event.get("error") or "Lyrics were not ready"
            current["phase"] = "composing"
            current["label"] = PHASE_LABELS["composing"]
            return _submit_song(ctx, current, lyric=None)
        return _step("fail", ctx, current, error=event.get("error") or "Music generation failed")

    # Success. Ignore a duplicate callback once the pipeline has moved on.
    if kind == "lyrics":
        if phase != "lyrics":
            return _step("update", ctx, current)
        lyrics = [item for item in (event.get("lyrics") or []) if item.get("text")]
        current["lyrics"] = lyrics
        if not lyrics:
            ctx["phase"] = "song"
            return _submit_song(ctx, current, lyric=None)
        ctx["phase"] = "song"
        current["lyrics_used"] = 0
        current["phase"] = "composing"
        current["label"] = PHASE_LABELS["composing"]
        return _submit_song(ctx, current, lyric=lyrics[0])
    if kind == "tracks":
        if phase not in ("song", "sounds", "replace"):
            _apply_assets(current, event)
            return _step("update", ctx, current)
        _apply_assets(current, event)
        if event.get("task_id") and not current.get("task_id"):
            current["task_id"] = event["task_id"]
        if not ctx.get("music_task_id"):
            ctx["music_task_id"] = current.get("task_id") or event.get("task_id")
        if hold_for_second_version(ctx, current):
            ctx["phase"] = "song"
            current["phase"] = "first_track"
            current["label"] = PHASE_LABELS["first_track"]
            return _step("update", ctx, current)
        if phase == "song" and ctx.get("easy") and ctx.get("action") == "song" and ctx.get("music_task_id"):
            ctx["phase"] = "covers"
            current["phase"] = "cover_art"
            current["label"] = PHASE_LABELS["cover_art"]
            return _step(
                "submit",
                ctx,
                current,
                kie_model=KIE_MODELS["covers"],
                kie_input={"task_id": ctx["music_task_id"]},
                primary_url=_primary_audio(current),
                optional=True,
            )
        ctx["phase"] = "done"
        current["phase"] = "complete"
        current["label"] = PHASE_LABELS["complete"]
        return _step("finish", ctx, current, primary_url=_primary_audio(current))
    if kind == "covers":
        if phase != "covers" and ctx.get("action") != "covers":
            return _step("update", ctx, current)
        current["covers"] = list(event.get("covers") or [])
        ctx["phase"] = "done"
        current["phase"] = "complete"
        current["label"] = PHASE_LABELS["complete"]
        primary = _primary_audio(current) or (current["covers"][0] if current["covers"] else None)
        return _step("finish", ctx, current, primary_url=primary)
    if kind == "video":
        current["video_url"] = event.get("video_url")
        if event.get("task_id") and ctx.get("action") == "video":
            current["task_id"] = event["task_id"]
        ctx["phase"] = "done"
        current["phase"] = "complete"
        current["label"] = PHASE_LABELS["complete"]
        return _step("finish", ctx, current, primary_url=current.get("video_url"))
    if kind == "stems":
        current["stems"] = list(event.get("stems") or [])
        if event.get("task_id") and ctx.get("action") == "stems":
            current["task_id"] = event["task_id"]
        ctx["phase"] = "done"
        current["phase"] = "complete"
        current["label"] = PHASE_LABELS["complete"]
        primary = current["stems"][0]["url"] if current["stems"] else None
        return _step("finish", ctx, current, primary_url=primary)
    if kind == "persona":
        current["persona"] = event.get("persona")
        ctx["phase"] = "done"
        current["phase"] = "complete"
        current["label"] = PHASE_LABELS["complete"]
        return _step("finish", ctx, current)
    return _step("fail", ctx, current, error="The music service returned a result this app could not read.")


def persona_music(music: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    current = copy.deepcopy(music or {})
    current["persona"] = {
        "id": str(data.get("persona_id") or ""),
        "name": data.get("name") or "",
        "description": data.get("description") or "",
    }
    current["phase"] = "complete"
    current["label"] = PHASE_LABELS["complete"]
    return current


def is_suno_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if data.get("callbackType") or data.get("callback_type"):
        return True
    if isinstance(data.get("vocal_removal_info"), dict) or data.get("video_url"):
        return True
    return False


def _prepare_song(context: Dict[str, Any], *, for_submit: bool) -> Prepared:
    idea = context["prompt"]
    has_references = bool(context["image_urls"] or context["video_urls"] or context["audio_urls"])
    if not context["easy"]:
        _validate_custom_song(context)
        if has_references:
            raise ValueError("Photos, video, and audio references belong in Easy mode. Remove them, or switch to Easy.")
        context["phase"] = "song"
        music = _blank_music(context, "composing")
        return Prepared(KIE_MODELS["song"], _custom_song_input(context, None), context, music, "async")

    # Easy is non-custom. Length, voice, variety, and style weights are ignored.
    dropped_length = context.get("duration")
    for key in ("duration", "style_weight", "weirdness_constraint", "audio_weight", "variety"):
        context[key] = None
    context["vocal_gender"] = ""

    if has_references:
        _reject_too_many(
            {"images": context["image_urls"], "videos": context["video_urls"], "audios": context["audio_urls"]},
            style=context["style"],
            lyrics=context["lyrics"],
            will_add_lyrics=not context["instrumental"] and not context["lyrics"] and bool(idea),
        )

    def easy_song(sheets: Optional[List[Dict[str, str]]] = None) -> Prepared:
        context["phase"] = "song"
        music = _blank_music(context, "composing")
        if sheets:
            music["lyrics"] = sheets
            music["lyrics_used"] = 0
        if has_references and dropped_length:
            music["note"] = "Song length is chosen by the model when you add a photo, video, or audio reference."
        return Prepared(KIE_MODELS["song"], _referenced_song_input(context, None), context, music, "async")

    if context["instrumental"]:
        if not (idea or context["style"] or context["negative_tags"] or has_references):
            raise ValueError("Describe the instrumental, add a style, or attach a reference.")
        return easy_song()
    if context["lyrics"]:
        if not context["title"]:
            context["title"] = _title_from(context["lyrics"])
        return easy_song([{"title": context["title"], "text": context["lyrics"]}])
    if idea:
        context["phase"] = "lyrics"
        music = _blank_music(context, "writing_lyrics")
        if has_references and dropped_length:
            music["note"] = "Song length is chosen by the model when you add a photo, video, or audio reference."
        return Prepared(
            KIE_MODELS["lyrics"],
            {"prompt": _lyrics_prompt(idea)},
            context,
            music,
            "async",
        )
    if has_references:
        return easy_song()
    raise ValueError("Describe the song you want, paste lyrics, or add a photo, clip, or track.")


def _validate_custom_song(context: Dict[str, Any]) -> None:
    if context["instrumental"]:
        if not (context["style"] or context["negative_tags"] or context["prompt"]):
            raise ValueError("An instrumental needs a style or a short description.")
        if not context["style"] and context["prompt"]:
            context["style"] = _clip(context["prompt"], 1000)
        if not context["title"]:
            context["title"] = _title_from(context["style"] or context["prompt"])
        return
    if not (context["style"] or context["lyrics"] or context["negative_tags"] or context["prompt"]):
        raise ValueError("Add lyrics, a style, or a description of the song.")
    if not context["lyrics"] and context["prompt"]:
        context["lyrics"] = _clip(context["prompt"], 5000)
    if not context["title"]:
        context["title"] = _title_from(context["lyrics"] or context["style"] or context["prompt"])
    if not (context["style"] or context["lyrics"] or context["negative_tags"]):
        raise ValueError("Add lyrics or a style.")


def _submit_song(ctx: Dict[str, Any], music: Dict[str, Any], lyric: Optional[Dict[str, str]]) -> Dict[str, Any]:
    has_references = bool(ctx.get("image_urls") or ctx.get("video_urls") or ctx.get("audio_urls"))
    if lyric and not ctx.get("instrumental"):
        ctx["lyrics"] = lyric.get("text") or ctx.get("lyrics") or ""
        if lyric.get("title") and not ctx.get("title"):
            ctx["title"] = _clip(lyric["title"], 80)
    # Easy stays non-custom even after lyrics are written and even with no reference file.
    payload = _referenced_song_input(ctx, lyric) if ctx.get("easy") or has_references else _custom_song_input(ctx, lyric)
    ctx["phase"] = "song"
    return _step(
        "submit",
        ctx,
        music,
        kie_model=KIE_MODELS["song"],
        kie_input=payload,
        optional=False,
    )


def _custom_song_input(ctx: Dict[str, Any], lyric: Optional[Dict[str, str]]) -> Dict[str, Any]:
    instrumental = bool(ctx.get("instrumental"))
    lyric_text = ""
    lyric_title = ""
    if lyric:
        lyric_text = lyric.get("text") or ""
        lyric_title = lyric.get("title") or ""
    title = _clip(ctx.get("title") or lyric_title or _title_from(ctx.get("prompt") or lyric_text or ctx.get("style") or ""), 80)
    style = _clip(ctx.get("style") or "", 1000)
    if not style:
        style = _style_from(ctx.get("prompt") or "")
    lyrics = "" if instrumental else _clip(lyric_text or ctx.get("lyrics") or "", 5000)
    payload: Dict[str, Any] = {
        "custom_mode": True,
        "instrumental": instrumental,
        "model": ctx.get("variant") or "V6",
        "title": title or "Untitled",
    }
    if style:
        payload["style"] = style
    if lyrics:
        payload["lyrics"] = lyrics
    if ctx.get("negative_tags"):
        payload["negative_tags"] = ctx["negative_tags"]
    if not (payload.get("style") or payload.get("lyrics") or payload.get("negative_tags")):
        payload["style"] = "original"
    _add_custom_controls(payload, ctx, instrumental)
    return payload


def _referenced_song_input(ctx: Dict[str, Any], lyric: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """References only work in non-custom mode. Lyrics can ride along as an attachment."""
    instrumental = bool(ctx.get("instrumental"))
    payload: Dict[str, Any] = {
        "custom_mode": False,
        "instrumental": instrumental,
        "model": ctx.get("variant") or "V6",
    }
    idea = _clip(ctx.get("prompt") or "", 3000)
    if idea:
        payload["prompt"] = idea
    lyric_text = ""
    if lyric and lyric.get("text"):
        lyric_text = lyric["text"]
    elif ctx.get("lyrics"):
        lyric_text = ctx["lyrics"]
    if lyric_text and not instrumental:
        payload["lyrics"] = _clip(lyric_text, 5000)
    if ctx.get("style"):
        payload["style"] = _clip(ctx["style"], 1000)
    if ctx.get("image_urls"):
        payload["image_urls"] = list(ctx["image_urls"][:5])
    if ctx.get("video_urls"):
        payload["video_urls"] = list(ctx["video_urls"][:1])
    if ctx.get("audio_urls"):
        payload["audio_urls"] = list(ctx["audio_urls"])
    if not any(payload.get(key) for key in ("style", "lyrics", "image_urls", "video_urls", "audio_urls")):
        payload["style"] = _style_from(idea) or "original"
    return payload


def _add_custom_controls(
    payload: Dict[str, Any],
    ctx: Dict[str, Any],
    instrumental: bool,
    *,
    include_duration: bool = True,
) -> None:
    if ctx.get("vocal_gender") and not instrumental:
        payload["vocal_gender"] = ctx["vocal_gender"]
    if include_duration and ctx.get("duration") is not None:
        payload["duration"] = ctx["duration"]
    for key in ("style_weight", "weirdness_constraint"):
        if ctx.get(key) is not None:
            payload[key] = ctx[key]
    if ctx.get("audio_weight") is not None and not instrumental:
        payload["audio_weight"] = ctx["audio_weight"]
    if ctx.get("variety") is not None:
        payload["variety"] = ctx["variety"]
    if ctx.get("persona_id"):
        payload["persona_id"] = ctx["persona_id"]
        payload["persona_model"] = ctx.get("persona_model") or "style_persona"


def _prepare_sounds(context: Dict[str, Any], params: Dict[str, Any]) -> Prepared:
    idea = _clip(context["prompt"], 500)
    if not idea:
        raise ValueError("Describe the sound you want.")
    payload: Dict[str, Any] = {"prompt": idea, "model": context["variant"]}
    if _bool(params.get("sound_loop") or params.get("loop"), False):
        payload["sound_loop"] = True
    tempo = params.get("sound_tempo") if params.get("sound_tempo") not in (None, "") else params.get("bpm")
    if tempo not in (None, ""):
        try:
            bpm = int(tempo)
        except (TypeError, ValueError):
            raise ValueError("Tempo must be a whole number from 1 to 300.")
        if bpm < 1 or bpm > 300:
            raise ValueError("Tempo must be a whole number from 1 to 300.")
        payload["sound_tempo"] = bpm
    raw_key = params.get("sound_key") if params.get("sound_key") not in (None, "") else params.get("key")
    if raw_key not in (None, ""):
        key = _choice(raw_key, SOUND_KEYS)
        if not key:
            raise ValueError("Pick a musical key from the list, or leave it on Any.")
        if key != "Any":
            payload["sound_key"] = key
    if _bool(params.get("grab_lyrics"), False):
        payload["grab_lyrics"] = True
    context["phase"] = "sounds"
    return Prepared(KIE_MODELS["sounds"], payload, context, _blank_music(context, "composing"), "async")


def _prepare_replace(context: Dict[str, Any], params: Dict[str, Any]) -> Prepared:
    prompt = _clip(params.get("prompt") or context["prompt"] or "", 5000)
    tags = _clip(params.get("tags") or params.get("style") or context["style"] or "", 1000)
    title = _clip(params.get("title") or context["title"] or "", 80)
    full_lyrics = _clip(params.get("full_lyrics") or "", 5000)
    section_lyrics = _clip(params.get("lyrics") or context["lyrics"] or "", 5000)
    if not prompt:
        raise ValueError("Describe what the replaced section should become.")
    if not tags:
        raise ValueError("Add style tags for the replaced section.")
    if not title:
        raise ValueError("Add a title.")
    if not full_lyrics:
        raise ValueError("full_lyrics must be the entire song after the edit.")
    start = _seconds(params.get("infill_start_s") if params.get("infill_start_s") is not None else params.get("start_seconds"), "start")
    end = _seconds(params.get("infill_end_s") if params.get("infill_end_s") is not None else params.get("end_seconds"), "end")
    if end <= start:
        raise ValueError("The replacement has to end after it starts.")
    if end - start < 10:
        raise ValueError("Replace at least 10 seconds.")
    source_duration = context.get("source_duration")
    if source_duration and (end - start) > (float(source_duration) * 0.5) + 0.001:
        raise ValueError("The replacement cannot be longer than half the song.")
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "tags": tags,
        "title": title,
        "infill_start_s": round(start, 2),
        "infill_end_s": round(end, 2),
        "full_lyrics": full_lyrics,
    }
    if section_lyrics:
        payload["lyrics"] = section_lyrics
    if context["negative_tags"]:
        payload["negative_tags"] = context["negative_tags"]
    _add_custom_controls(payload, context, False, include_duration=False)
    upload = context["audio_urls"][0] if context["audio_urls"] else ""
    has_track = bool(context["source_task_id"] and context["audio_id"])
    if has_track and upload:
        raise ValueError("Pick a saved song or upload audio, not both.")
    if has_track:
        payload["task_id"] = context["source_task_id"]
        payload["audio_id"] = context["audio_id"]
    elif upload:
        _require_audio_rights(params)
        payload["upload_url"] = upload
        payload["model"] = context["variant"]
    else:
        raise ValueError("Pick a song you made here, or upload audio you own.")
    context["phase"] = "replace"
    return Prepared(KIE_MODELS["replace"], payload, context, _blank_music(context, "replacing"), "async")


def _prepare_stems(context: Dict[str, Any], params: Dict[str, Any]) -> Prepared:
    stem_type = _stem_type(params)
    payload: Dict[str, Any] = {"type": stem_type}
    if stem_type == "split_stem_advanced":
        name = _choice(params.get("stem_name"), STEM_NAMES)
        if not name:
            raise ValueError("Pick a stem from the instrument list.")
        payload["stem_name"] = name
    upload = context["audio_urls"][0] if context["audio_urls"] else ""
    has_track = bool(context["source_task_id"] and context["audio_id"])
    if (context["source_task_id"] or context["audio_id"]) and upload:
        raise ValueError("Pick a saved song or upload audio, not both.")
    if has_track:
        payload["task_id"] = context["source_task_id"]
        payload["audio_id"] = context["audio_id"]
    elif upload:
        _require_audio_rights(params)
        payload["audio_url"] = upload
    else:
        raise ValueError("Pick a song you made here, or upload audio you own.")
    context["phase"] = "stems"
    context["stem_type"] = stem_type
    return Prepared(KIE_MODELS["stems"], payload, context, _blank_music(context, "separating"), "async")


def _prepare_persona(context: Dict[str, Any], params: Dict[str, Any]) -> Prepared:
    _reject_upload(context, "A persona")
    _require_source(context, "A persona")
    name = _clip(params.get("name") or params.get("persona_name") or "", 80)
    description = _clip(params.get("description") or params.get("persona_description") or "", 1000)
    if not name:
        raise ValueError("Name the persona.")
    if not description:
        raise ValueError("Describe the persona's style and character.")
    start = _optional_float(params.get("vocal_start"))
    end = _optional_float(params.get("vocal_end"))
    start = 0.0 if start is None else start
    end = 30.0 if end is None else end
    if start < 0 or end < 0 or end <= start:
        raise ValueError("The persona window has to end after it starts.")
    span = end - start
    if span < 10 or span > 30:
        raise ValueError("The persona window must be between 10 and 30 seconds.")
    payload: Dict[str, Any] = {
        "task_id": context["source_task_id"],
        "audio_id": context["audio_id"],
        "name": name,
        "description": description,
        "vocal_start": round(start, 2),
        "vocal_end": round(end, 2),
    }
    style = _clip(params.get("style") or context["style"] or "", 200)
    if style:
        payload["style"] = style
    context["phase"] = "persona"
    music = _blank_music(context, "creating_persona")
    return Prepared(KIE_MODELS["persona"], payload, context, music, "persona")


def _prepare_voice(context: Dict[str, Any], params: Dict[str, Any]) -> Prepared:
    if not context["source_task_id"]:
        raise ValueError("A custom voice needs the validation task_id.")
    verify = ""
    if context["audio_urls"]:
        verify = context["audio_urls"][0]
    if not verify:
        raise ValueError("Upload the sung validation phrase.")
    payload: Dict[str, Any] = {"task_id": context["source_task_id"], "verify_url": verify}
    voice_name = _clip(params.get("voice_name") or params.get("name") or "", 80)
    description = _clip(params.get("description") or "", 500)
    style = _clip(params.get("style") or "", 200)
    skill = _choice(params.get("singer_skill_level") or params.get("skill"), SKILL_LEVELS)
    if voice_name:
        payload["voice_name"] = voice_name
    if description:
        payload["description"] = description
    if style:
        payload["style"] = style
    if skill:
        payload["singer_skill_level"] = skill
    context["phase"] = "voice"
    music = _blank_music(context, "creating_voice")
    music["voice_id"] = context["source_task_id"]
    music["phase"] = "complete"
    music["label"] = "Voice saved"
    music["note"] = "The voice can take about a minute to become usable. Pass it as persona_id with persona_model set to voice_persona."
    return Prepared(KIE_MODELS["voice"], payload, context, music, "voice")


def _simple(context: Dict[str, Any], action: str, phase: str, payload: Dict[str, Any]) -> Prepared:
    if action == "lyrics" and not payload.get("prompt"):
        raise ValueError("Describe the lyrics you want.")
    context["phase"] = action if action != "song" else "song"
    if action == "lyrics":
        context["phase"] = "lyrics"
    elif action == "covers":
        context["phase"] = "covers"
    elif action == "video":
        context["phase"] = "video"
    return Prepared(KIE_MODELS[action], payload, context, _blank_music(context, phase), "async")


def _blank_music(context: Dict[str, Any], phase: str) -> Dict[str, Any]:
    action = context.get("action") or "song"
    return {
        "action": action,
        "phase": phase,
        "label": PHASE_LABELS.get(phase, "Working"),
        "variant": context.get("variant") or "V6",
        "easy": bool(context.get("easy")),
        "instrumental": bool(context.get("instrumental")),
        "commercial_rights": action in COMMERCIAL_ACTIONS,
        "task_id": None,
        "source_task_id": context.get("source_task_id") or None,
        "audio_id": context.get("audio_id") or None,
        "tracks": [],
        "lyrics": [],
        "covers": [],
        "stems": [],
        "video_url": None,
        "persona": None,
        "voice_id": None,
    }


def hold_for_second_version(
    context: Optional[Dict[str, Any]],
    music: Optional[Dict[str, Any]],
    event: Optional[Dict[str, Any]] = None,
) -> bool:
    """Keep a HitMaker song open until both clips are stored.

    A short timer used to claim the Kie task after one clip. The callback that
    carries the second clip was then ignored. The task stays open until two
    clips can play, unless the caller sets accept_single_track after a long wait.
    """
    ctx = context or {}
    current = music or {}
    action = ctx.get("action") or current.get("action") or "song"
    if action != "song" or ctx.get("accept_single_track"):
        return False
    tracks = list(current.get("tracks") or [])
    if event and event.get("tracks"):
        tracks = _merge_tracks(tracks, event.get("tracks") or [])
    playable = [track for track in tracks if _track_has_audio(track)]
    if len(playable) >= 2 or not playable:
        return False
    return True


def _apply_assets(music: Dict[str, Any], event: Dict[str, Any]) -> None:
    if event.get("tracks"):
        music["tracks"] = _merge_tracks(music.get("tracks") or [], event["tracks"])
    if event.get("lyrics"):
        music["lyrics"] = event["lyrics"]
    if event.get("covers"):
        music["covers"] = event["covers"]
    if event.get("stems"):
        music["stems"] = event["stems"]
    if event.get("video_url"):
        music["video_url"] = event["video_url"]
    if event.get("persona"):
        music["persona"] = event["persona"]
    if event.get("task_id") and music.get("action") in COMMERCIAL_ACTIONS and not music.get("task_id"):
        music["task_id"] = event["task_id"]


def _step(
    action: str,
    context: Dict[str, Any],
    music: Dict[str, Any],
    *,
    kie_model: Optional[str] = None,
    kie_input: Optional[Dict[str, Any]] = None,
    primary_url: Optional[str] = None,
    optional: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "action": action,
        "context": context,
        "music": music,
        "kie_model": kie_model,
        "kie_input": kie_input,
        "primary_url": primary_url,
        "optional": optional,
        "error": error,
    }


def _primary_audio(music: Dict[str, Any]) -> Optional[str]:
    for track in music.get("tracks") or []:
        if track.get("audio_url"):
            return track["audio_url"]
    for track in music.get("tracks") or []:
        if track.get("stream_url"):
            return track["stream_url"]
    return None


def _unwrap(payload: Dict[str, Any]):
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    result_json = data.get("resultJson") or data.get("result_json") or {}
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except json.JSONDecodeError:
            result_json = {}
    if not isinstance(result_json, dict):
        result_json = {}
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    if response and not data.get("data") and response.get("data"):
        data = dict(data)
        data["data"] = response.get("data")
    return data, result_json


def _tracks_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_lists = []
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    obj = result_json.get("resultObject") if isinstance(result_json.get("resultObject"), dict) else {}
    for source in (data, response, obj, result_json):
        if not isinstance(source, dict):
            continue
        for key in ("data", "sunoData", "suno_data", "clips"):
            if isinstance(source.get(key), list):
                raw_lists.append(source[key])
    tracks = []
    for raw in raw_lists:
        incoming = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            audio_url = _first_text(
                item.get("audio_url"),
                item.get("audioUrl"),
                item.get("source_audio_url"),
            )
            stream_url = _first_text(
                item.get("stream_audio_url"),
                item.get("stream_url"),
                item.get("streamAudioUrl"),
                item.get("source_stream_audio_url"),
            )
            track_id = _first_text(item.get("id"), item.get("audio_id"), item.get("audioId"))
            if not (audio_url or stream_url or track_id):
                continue
            if item.get("text") and not audio_url and not stream_url:
                continue
            incoming.append({
                "id": track_id or None,
                "title": item.get("title") or "",
                "lyrics": item.get("prompt") or item.get("lyrics") or "",
                "tags": item.get("tags") or "",
                "duration": item.get("duration"),
                "audio_url": audio_url,
                "stream_url": stream_url,
                "cover_url": _first_text(
                    item.get("image_url"),
                    item.get("imageUrl"),
                    item.get("source_image_url"),
                    item.get("cover_url"),
                ),
            })
        tracks = _merge_tracks(tracks, incoming)
    return tracks


def _track_has_audio(track: Dict[str, Any]) -> bool:
    return bool(track.get("audio_url") or track.get("stream_url"))


def _track_key(track: Dict[str, Any]) -> str:
    track_id = str(track.get("id") or "").strip()
    if track_id:
        return "id:" + track_id
    for field in ("audio_url", "stream_url", "download_url"):
        url = str(track.get(field) or "").strip()
        if url:
            return "url:" + url
    return ""


def _merge_tracks(current: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(item) for item in current if isinstance(item, dict)]
    for raw in incoming:
        if not isinstance(raw, dict):
            continue
        key = _track_key(raw)
        if not key:
            out.append(dict(raw))
            continue
        for index, kept in enumerate(out):
            if _track_key(kept) != key:
                continue
            merged = dict(kept)
            for field, value in raw.items():
                if value not in (None, ""):
                    merged[field] = value
            if not merged.get("stream_url") and kept.get("stream_url"):
                merged["stream_url"] = kept["stream_url"]
            out[index] = merged
            break
        else:
            out.append(dict(raw))
    return out


def _lyrics_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = []
    obj = result_json.get("resultObject") if isinstance(result_json.get("resultObject"), dict) else {}
    if isinstance(obj.get("lyricsData"), list):
        raw.extend(obj["lyricsData"])
    if isinstance(data.get("data"), list):
        raw.extend(data["data"])
    lyrics = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or ""
        if not text or item.get("audio_url"):
            continue
        if str(item.get("status") or "complete").lower() == "failed":
            continue
        lyrics.append({"title": item.get("title") or "", "text": text})
    return lyrics


def _covers_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> List[str]:
    images = data.get("images")
    if isinstance(images, list):
        found = []
        for item in images:
            if isinstance(item, str) and item.startswith("http"):
                found.append(item)
            elif isinstance(item, dict):
                url = _first_text(item.get("url"), item.get("image_url"), item.get("imageUrl"))
                if url and url.startswith("http"):
                    found.append(url)
        if found:
            return found
    urls = result_json.get("resultUrls") or result_json.get("result_urls") or []
    if isinstance(urls, list) and urls and all(isinstance(item, str) and _looks_like_image(item) for item in urls):
        return [str(item) for item in urls]
    return []


def _stems_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> List[Dict[str, str]]:
    info = data.get("vocal_removal_info")
    if not isinstance(info, dict):
        obj = result_json.get("resultObject") if isinstance(result_json.get("resultObject"), dict) else {}
        info = obj if any(str(key).endswith("_url") for key in obj.keys()) else None
    stems: List[Dict[str, str]] = []
    if isinstance(info, dict):
        stems.extend(_stem_urls(info, ""))
    urls = result_json.get("resultUrls") or []
    if isinstance(urls, list):
        for url in urls:
            if isinstance(url, str) and url.startswith("http") and not _looks_like_image(url) and not url.lower().split("?")[0].endswith(".mp4"):
                name = url.rsplit("/", 1)[-1].split("?")[0]
                if not any(item.get("url") == url for item in stems):
                    stems.append({"name": name, "url": url})
    return stems


def _stem_urls(info: Dict[str, Any], prefix: str) -> List[Dict[str, str]]:
    stems = []
    for key, value in info.items():
        name = f"{prefix}{key}"
        if isinstance(value, str) and value.startswith("http"):
            label = name[:-4] if name.endswith("_url") else name
            if label == "origin":
                continue
            stems.append({"name": label, "url": value})
        elif isinstance(value, dict):
            stems.extend(_stem_urls(value, f"{name}_"))
    return stems


def _video_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> str:
    for key in ("video_url", "audio_wav_url"):
        value = data.get(key)
        if isinstance(value, str) and value.startswith("http") and key == "video_url":
            return value
    urls = result_json.get("resultUrls") or []
    if isinstance(urls, list):
        for url in urls:
            if isinstance(url, str) and url.lower().split("?")[0].endswith(".mp4"):
                return url
    value = data.get("video_url")
    if isinstance(value, str) and value.startswith("http"):
        return value
    return ""


def _persona_from(data: Dict[str, Any], result_json: Dict[str, Any]) -> Optional[Dict[str, str]]:
    source = data if data.get("persona_id") else result_json
    if not isinstance(source, dict) or not source.get("persona_id"):
        obj = result_json.get("resultObject") if isinstance(result_json.get("resultObject"), dict) else {}
        source = obj if obj.get("persona_id") else None
    if not isinstance(source, dict) or not source.get("persona_id"):
        return None
    return {
        "id": str(source.get("persona_id")),
        "name": source.get("name") or "",
        "description": source.get("description") or "",
    }


def _has_assets(event: Dict[str, Any]) -> bool:
    return bool(event.get("tracks") or event.get("lyrics") or event.get("covers") or event.get("stems") or event.get("video_url") or event.get("persona"))


def _fail_text(payload: Dict[str, Any], data: Dict[str, Any]) -> str:
    for key in ("failMsg", "fail_msg", "error", "errorMessage"):
        value = data.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != "<error detail>":
            return value.strip()
    msg = payload.get("msg") or payload.get("message") or ""
    if isinstance(msg, str) and msg.strip() and msg.strip().lower() not in ("success", "ok"):
        return msg.strip()
    return "Music generation failed"


def _collect_media(params: Dict[str, Any], *, strict: bool) -> Dict[str, List[str]]:
    images = _media_values(params, ("image_urls", "init_image"), ("init_image_r2_keys", "init_image_r2_key", "image_r2_keys"), strict)
    videos = _media_values(params, ("video_urls", "input_video"), ("input_video_r2_key", "video_r2_key"), strict)
    audios = _media_values(
        params,
        ("audio_urls", "reference_audio", "source_audio", "upload_url", "verify_url", "verify_audio"),
        ("audio_r2_keys", "reference_audio_r2_key", "upload_url_r2_key", "verify_audio_r2_key", "verify_url_r2_key"),
        strict,
    )
    if len(images) > 5:
        raise ValueError("Use up to 5 images.")
    if len(videos) > 1:
        raise ValueError("Use one video reference.")
    return {"images": images, "videos": videos, "audios": audios}


def _media_values(params: Dict[str, Any], url_keys, key_fields, strict: bool) -> List[str]:
    found: List[str] = []
    for key in url_keys:
        found.extend(_as_list(params.get(key)))
    http = [item for item in found if item.startswith("http://") or item.startswith("https://")]
    if http:
        return _dedupe(http)
    if found and strict:
        raise ValueError("A reference file is missing its upload link. Upload it and try again.")
    if found and not strict:
        return _dedupe(found)
    pending: List[str] = []
    for key in key_fields:
        pending.extend(_as_list(params.get(key)))
    if pending and strict:
        raise ValueError("A reference file is missing its upload link. Upload it and try again.")
    return _dedupe(pending)


def _reject_too_many(media: Dict[str, List[str]], *, style: str, lyrics: str, will_add_lyrics: bool) -> None:
    count = len(media.get("images") or []) + len(media.get("videos") or []) + len(media.get("audios") or [])
    if style:
        count += 1
    if lyrics or will_add_lyrics:
        count += 1
    if count > 10:
        raise ValueError("Use at most 10 references in total. Style, lyrics, images, video, and audio all count.")


def _is_easy(params: Dict[str, Any]) -> bool:
    mode = str(params.get("mode") or "").strip().lower()
    if mode in ("custom", "advanced"):
        return False
    if mode in ("easy", "simple", "auto"):
        return True
    if params.get("is_custom_mode") is True:
        return False
    return True


def _variant(value: Any) -> str:
    raw = str(value or "V6").strip().upper().replace(".", "_").replace("-", "_")
    aliases = {"V6MINI": "V6_MINI", "V6WILD": "V6_WILD", "MINI": "V6_MINI", "WILD": "V6_WILD"}
    raw = aliases.get(raw, raw)
    if raw not in VARIANTS:
        raise ValueError("Choose V6, V6 Mini, or V6 Wild.")
    return raw


def _vocal_gender(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("", "none", "any"):
        return ""
    if raw in ("m", "male", "man"):
        return "m"
    if raw in ("f", "female", "woman"):
        return "f"
    raise ValueError("Vocal gender must be male or female.")


def _persona_model(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw in ("style", "style_persona"):
        return "style_persona"
    if raw in ("voice", "voice_persona"):
        return "voice_persona"
    raise ValueError("persona_model must be style_persona or voice_persona.")


def _duration(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("Duration must be between 10 and 360 seconds.")
    if seconds < 10 or seconds > 360:
        raise ValueError("Duration must be between 10 and 360 seconds.")
    if abs(seconds - round(seconds)) < 0.001:
        return float(int(round(seconds)))
    return round(seconds, 2)


def _unit(value: Any, label: str) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = round(float(value), 2)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be between 0 and 1.")
    if number < 0 or number > 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return number


def _variety(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("Variety must be a whole number from 0 to 4.")
    if number < 0 or number > 4:
        raise ValueError("Variety must be a whole number from 0 to 4.")
    return number


def _stem_type(params: Dict[str, Any]) -> str:
    raw = str(params.get("stem_type") or params.get("separation") or "separate_vocal").strip().lower()
    if raw not in STEM_TYPES:
        raise ValueError("stem_type must be separate_vocal, split_stem, or split_stem_advanced.")
    return raw


def _choice(value: Any, options) -> str:
    if value in (None, ""):
        return ""
    raw = str(value).strip().lower()
    for option in options:
        if option.lower() == raw:
            return option
    return ""


def _require_source(context: Dict[str, Any], label: str) -> None:
    if not context.get("source_task_id") or not context.get("audio_id"):
        raise ValueError(f"{label} needs one of your songs. Pick a version.")


def _reject_upload(context: Dict[str, Any], label: str) -> None:
    if context.get("audio_urls"):
        raise ValueError(f"{label} only works on a song made here. Pick one of your versions.")


def _require_audio_rights(params: Dict[str, Any]) -> None:
    if not _bool(params.get("confirm_audio_rights"), False):
        raise ValueError("Confirm this audio is yours and is not copyrighted.")


def _seconds(value: Any, label: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Add the {label} time in seconds.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"The {label} time must be a number of seconds.")


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("That number is not valid.")


def _lyrics_prompt(idea: str) -> str:
    text = _clip(idea, 200)
    if not text:
        raise ValueError("Describe the lyrics you want.")
    return text


def _title_from(text: str) -> str:
    line = ""
    for part in str(text or "").replace("\r", "\n").split("\n"):
        cleaned = part.strip().strip("[]")
        if cleaned and not cleaned.lower().startswith(("verse", "chorus", "bridge", "outro", "intro")):
            line = cleaned
            break
    if not line:
        line = "Untitled"
    return _clip(line, 80) or "Untitled"


def _style_from(idea: str) -> str:
    return _clip(idea, 200)


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _dedupe(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _looks_like_image(url: str) -> bool:
    path = url.lower().split("?")[0]
    return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))
