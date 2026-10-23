# Suno V6 music studio

Model id: `suno_v6`

This is a new music model. `suno_music` and `suno_sfx` stay as they are. `suno_v6` is the full studio: an idea becomes lyrics, a finished song with its own cover, and then a video, stems, a section edit, or a reusable persona when the user asks.

A song is **65 credits** (about 20 INR). That price includes commercial rights. One request returns two versions. Do not send `variations` greater than 1; the server forces it back to 1.

## How a job moves

`POST /submit` with `model: "suno_v6"`. Poll `GET /job_status?job_id=...`.

The job stays `running` while lyrics, the song, and extra covers are produced. Read `music` on every poll. It is present before the job completes, so the screen can show lyrics and start playback of the first version early.

`status: "completed"` means the pipeline finished. `result_url` is a short-lived download link for the main file (about 20 minutes). `GET /download_url/{job_id}` mints a fresh link for that same main file. The files inside `music` are the ones to build the player from. Provider files are kept for **14 days**.

`music.commercial_rights` is true for a song, a sound loop, and a section replace.

## What the user pays

| Action | Credits | What they get |
| --- | --- | --- |
| `song` (default) | 65 | Two versions, lyrics, a cover on each version, and in Easy mode a second set of covers. Commercial rights included. |
| `sounds` | 15 | A sound or loop. Commercial rights included. |
| `replace` | 65 | The song with one section regenerated. Commercial rights included. |
| `lyrics` | 5 | Lyric options only, when they want words before a song. |
| `covers` | 10 | Another pair of cover images for a finished song. |
| `video` | 20 | A music video of one version. |
| `stems` (`separate_vocal`) | 20 | Vocals and accompaniment. |
| `stems` (`split_stem`) | 40 | Up to 12 instrument stems. |
| `stems` (`split_stem_advanced`) | 30 | One chosen instrument, extracted. |
| `persona` | 10 | A style persona that later songs can reuse. |
| `voice` | 2 | Saves a sung voice sample. Kie does not charge for this call. |

Lyrics and the extra Easy-mode covers are included in the 65. They are not a second charge. A video, stems, a replace, or a persona is a new submit and a new charge.

## Easy mode

Send `params.mode: "easy"`, or omit `mode`. This is the default.

The prompt is the idea, not the lyric sheet. Example: "a late-night drive through rain, hopeful by the last chorus".

What the server does:

1. If there are no lyrics yet, the song is not instrumental, and there is an idea, it calls the lyrics endpoint. `music.phase` is `writing_lyrics`. A photo, clip, or track by itself skips that call.
2. It submits generate with `custom_mode: false`. The idea stays in `prompt`. Written lyrics ride along as `lyrics`. `music.phase` becomes `composing`.
3. When the first version has audio, `music.phase` becomes `first_track` while `status` is still `running`. Start the player.
4. When both versions are done, it asks for extra cover images. `music.phase` is `cover_art`.
5. `status` becomes `completed`. Each version already has `cover_url`. `music.covers` is the extra pair.

If lyrics fail, the song is still composed from the idea. The user is not failed for a lyric miss.

If they already edited lyrics, send `params.lyrics` and the lyrics call is skipped. Their words are sung.

Instrumental: `params.instrumental: true`. No lyric call. Describe the piece in the prompt, or send `params.style`.

### Optional Easy controls

Easy generate does not accept length, vocal gender, variety, or the style weights. Those belong in Custom mode. A style or negative tags can still ride along, and `variant` stays `V6` unless another variant is sent.

### Reference photo, video, or audio

References only work in Easy mode. Put them on the same submit:

- Up to 5 images: `params.init_image_r2_keys`, or the top-level `input_image_r2_keys` (the server keeps them on this one song).
- One video: `params.input_video_r2_key`. mp4, mov, or webm, up to 100 MB and 241 seconds.
- Audio: `params.reference_audio_r2_key`, `params.audio_r2_keys`, or https URLs in `params.audio_urls`. Each clip is 6 seconds to 30 minutes.
- A sung voice sample for the voice action: `params.verify_audio_r2_key`.

Images should be png, jpeg, or webp. The server rehosts them before the provider sees them.

Style, lyrics, images, the video, and audio clips together cannot exceed 10 attachments. A reference is enough on its own. The idea can be omitted. A reference clip steers a new song. It does not edit or split that file. Editing and splitting are the replace and stems actions. Length, voice, and variety are not sent in Easy mode.

Custom mode rejects reference files. Ask them to switch to Easy.

## Custom mode

`params.mode: "custom"`, or `params.is_custom: true`.

Use this when they want to write the song themselves.

- `title` — up to 80 characters. If they leave it blank, the server takes it from the first lyric line.
- `style` and/or `lyrics`. Lyrics win over the prompt. The prompt is sung only when `lyrics` is empty.
- `instrumental: true` needs a style or a description, and does not take lyrics.
- Optional controls: vocal gender, duration (10–360), weights, variety (0–4), negative tags, persona, variant.

Do not send reference images, video, or audio in Custom mode.

## The `music` object

```json
{
  "action": "song",
  "phase": "complete",
  "label": "Ready",
  "variant": "V6",
  "easy": true,
  "instrumental": false,
  "commercial_rights": true,
  "task_id": "song-task-id",
  "source_task_id": null,
  "audio_id": null,
  "tracks": [
    {
      "id": "audio-id",
      "title": "Starry Night Dreams",
      "lyrics": "[Verse]\n...",
      "tags": "nostalgic, acoustic",
      "duration": 198.44,
      "audio_url": "https://.../song.mp3",
      "stream_url": "https://.../stream",
      "cover_url": "https://.../cover.jpg",
      "download_url": "https://.../fresh-link"
    }
  ],
  "lyrics": [
    {"title": "Starry Night Dreams", "text": "[Verse]\n..."}
  ],
  "lyrics_used": 0,
  "covers": ["https://.../extra-1.png", "https://.../extra-2.png"],
  "stems": [],
  "video_url": null,
  "persona": null,
  "voice_id": null,
  "note": null
}
```

`label` is safe to show as the status line. Phases:

| `phase` | Show |
| --- | --- |
| `writing_lyrics` | Writing lyrics |
| `composing` | Composing. If `lyrics` is already filled, show the sheets. |
| `first_track` | Play `tracks` that have `audio_url` or `stream_url`. Another version is still coming. |
| `cover_art` | Song is done. Extra covers are rendering. |
| `separating` | Stem job. |
| `rendering_video` | Video job. |
| `replacing` | Section replace. |
| `complete` | Done. |

`task_id` and `tracks[].id` are metadata. Save them with the job. Never show them, and never ask the user to type them.

Play `download_url` when it is present. Fall back to `audio_url`, then `stream_url`. `download_url` expires in about 20 minutes. `audio_url` is the 14-day file. For the first version, `GET /download_url/{job_id}` refreshes `result_url`.

There are usually two tracks. Render both as choices, with title and cover. The user picks a version. The client sends that version's ids.

## What to save

The server already keeps `music` on the job. The library is completed `suno_v6` jobs. Read `music` from each one and keep this on the client, next to the job:

```json
{
  "job_id": "job-uuid",
  "action": "song",
  "task_id": "<music.task_id>",
  "commercial_rights": true,
  "variant": "V6",
  "tracks": [
    {
      "id": "<tracks[n].id>",
      "title": "Starry Night Dreams",
      "lyrics": "[Verse]\n...",
      "tags": "nostalgic, acoustic",
      "duration": 198.44,
      "cover_url": "https://.../cover.jpg",
      "audio_url": "https://.../song.mp3"
    }
  ],
  "lyrics": [{"title": "Starry Night Dreams", "text": "[Verse]\n..."}],
  "covers": ["https://.../extra-1.png"],
  "persona_by_track": {"<tracks[0].id>": "<music.persona.id>"}
}
```

Save a persona on its own list too, so a later song can offer it by name:

```json
{"id": "<music.persona.id>", "name": "Night alto", "description": "...", "model": "style_persona"}
```

Update the record when `phase` is `first_track`, and again on `complete`. A replaced song is a new job: save it the same way, with `action: "replace"`.

Only a song from `action: "song"` can become a video, extra covers, or a persona. Those endpoints accept a Generate Music task, not an upload and not a section-replace task. Stems and section replace can use that same saved song, or a file the user uploads.

## What the screen asks for

The user never sees `task_id`, `audio_id`, `source_task_id`, or `persona_id`.

| They want | What they see | What the client sends |
| --- | --- | --- |
| A video, a persona, or new covers | Their songs, as cover cards. They tap a version. Covers use the song, not a version. | `source_task_id` from the saved song, and `audio_id` from the tapped version. Covers send only `source_task_id`. |
| Stems or a section replace of a song they made | The same cards, or "Upload my own audio". | Saved song: `source_task_id` + `audio_id`. Upload: the file key and `confirm_audio_rights: true`. Never both. |
| A saved persona on a new song | A chip with the persona's name. | `persona_id` and `persona_model: "style_persona"`. |
| Style, length, vocal, variety | Chips, a switch, and a slider. | The matching `params` fields. No numeric id, no raw enum typed by hand. |
| A section to replace | The lyric sheet and a range on the waveform. | `infill_start_s`, `infill_end_s`, and `source_duration` from the saved track. Title, tags, and `full_lyrics` start filled from that track. |
| Which lyric sheet to sing | The sheets already on the job. "Sing this" starts a new song. | `params.lyrics` set to that sheet. |

A video, new covers, and a persona have no upload. If the client sends a file for those, the server rejects it. Show the songs they made here.

An upload is only for stems and section replace. The file must be audio they own and that is not copyrighted. The upload control is a checkbox, required before the button enables: "This audio is mine and is not copyrighted." Send `confirm_audio_rights: true` with the file. The server rejects the job without it. Max 20 MB for stems. Read the file's duration in the browser and send it as `source_duration` so the half-song limit can be checked before they pay.

Do not offer video, new covers, or a persona on a sound, a stem, or a section-replace result. Do not offer a second persona on a version that already has one in `persona_by_track`.

## Follow-up submits

Every follow-up is a new `POST /submit` with `model: "suno_v6"`. The ids in these bodies come from the saved song. They are not form fields.

### Music video — 20

```json
{
  "model": "suno_v6",
  "prompt": "",
  "params": {
    "action": "video",
    "source_task_id": "<music.task_id>",
    "audio_id": "<tracks[n].id>",
    "author": "Artist name",
    "domain_name": "yourapp.com"
  }
}
```

The user picks a version of a song they made here. `author` and `domain_name` are optional watermarks, 50 characters each. Prefill `author` from their display name and hide both under "Add a watermark". The file is `music.video_url`. `result_url` is the short download link.

### More covers — 10

```json
{"action": "covers", "source_task_id": "<music.task_id>"}
```

The user picks the song, not a version. Both versions share one saved `task_id`. Only after that song has finished. Results land in `music.covers`. Add them to the saved song's `covers`.

### Stems

```json
{"action": "stems", "source_task_id": "<music.task_id>", "audio_id": "<tracks[n].id>", "stem_type": "separate_vocal"}
```

`stem_type`:

- `separate_vocal` — vocals and accompaniment (20)
- `split_stem` — up to 12 stems (40)
- `split_stem_advanced` — also send `stem_name` (30)

`stem_type` is a segmented control, not a text field. `stem_name` is a searchable list of the names below, and the client sends the canonical spelling.

Results are `music.stems[]` with `name`, `url`, and `download_url`. Asking for the same stem again charges again. Stems are downloads. They are not a new song, so they do not appear as a source for video, covers, or a persona.

`stem_name` values for the advanced picker: Lead Vocal, Drum Kit, Kick, Snare, Risers, Bass, Backing Vocals, Piano, Electric Guitar, Percussion, String Section, Synth, Acoustic Guitar, Sound Effects, Synth Pad, Synth Bass, Guitar, Brass Section, Organ, Electronic Drum Kit, Lead Electric Guitar, Synth Keys, Rhythm Electric Guitar, Electric Piano, Upright Bass, Keyboards, Distorted Electric Guitar, Synth Strings, Synth Lead, Woodwinds, Rhythm Acoustic Guitar, Flute, Harp, Tambourine, Trumpet, Arpeggiator, Accordion, Fiddle, Pedal Steel Guitar, Synth Voice, Violin, Digital Piano, Synth Brass, Mandolin, Choir, Banjo, Bells, Clarinet, Tenor Saxophone, Trombone, Shaker, French Horn, Glockenspiel, Electric Bass, Cello, Timpani, Harmonica, Marimba, Vibraphone, Lap Steel Guitar, Saxophone, Orchestra, Horns, Cymbals, Hand Clap, Oboe, Celesta, Congas, Drone, Alto Saxophone, Double Bass, Ukulele, Harpsichord, Baritone Saxophone, Xylophone, Tuba, Bass Guitar, Whistle, Lead Guitar, Rhodes, 808, Bongos, Bassoon, Cowbell, Viola, Sitar, Steel Drums, Piccolo, Theremin, Bagpipes, Hi-Hat, Music Box, Melodica, Tabla, Koto, Djembe, Taiko, Didgeridoo.

Or they upload their own audio. Show the rights checkbox, then send `reference_audio_r2_key` and `confirm_audio_rights: true`, and do not send `source_task_id` or `audio_id`. Max 20 MB.

### Replace a section — 65

The user picks a saved version. Open its lyric sheet, already filled, and a range on the waveform. Title and style start as `title` and `tags` from that version. `full_lyrics` is the whole sheet after they edit. `source_duration` is `tracks[n].duration`.

```json
{
  "action": "replace",
  "source_task_id": "<music.task_id>",
  "audio_id": "<tracks[n].id>",
  "prompt": "A softer chorus, still the same melody",
  "tags": "acoustic, intimate",
  "title": "Starry Night Dreams",
  "full_lyrics": "[Verse]\n...entire song after the edit...",
  "lyrics": "[Chorus]\njust the new section",
  "infill_start_s": 42.5,
  "infill_end_s": 68.0,
  "source_duration": 198.44
}
```

The range must be at least 10 seconds and must not be more than half of `source_duration`. `full_lyrics` is the whole song after the edit, not only the changed lines. `tags` can be sent as `style`. The result is a new pair of tracks. Save that job on its own. It can be played and downloaded. It is not a source for a video, new covers, or a persona.

Or they upload audio they own. Same sheet and waveform, plus the variant picker and the rights checkbox. Send the file as `reference_audio_r2_key` with `confirm_audio_rights: true` and `variant`, and omit `source_task_id` / `audio_id`. Read the duration from the file and send `source_duration`.

### Persona — 10

The user picks a version of a song they made here that does not already have a persona. One persona per version. The song must already be complete. Name and description are the only empty fields. Prefill `style` from the version's tags. The window is a 10–30 second range on the waveform, defaulting to the first 30 seconds, and it cannot run past `duration`.

```json
{
  "action": "persona",
  "source_task_id": "<music.task_id>",
  "audio_id": "<tracks[n].id>",
  "name": "Night alto",
  "description": "Close-mic alto, dusty chorus, slow acoustic guitar",
  "style": "acoustic",
  "vocal_start": 12,
  "vocal_end": 28
}
```

The result is `music.persona`. Store it on that version and in the persona list. On a later song the user taps the name. The client sends `persona_id` with `persona_model: "style_persona"`.

### Sounds — 15

For beds, loops, and ambience. Not the lyric song flow.

```json
{
  "action": "sounds",
  "variant": "V6",
  "sound_loop": true,
  "sound_tempo": 72,
  "sound_key": "D#m",
  "grab_lyrics": false
}
```

The prompt is the sound, up to 500 characters. Tempo is a slider from 1 to 300, or omitted. Key is a picker: `Any` or a note with optional `#` and `m` (`C`, `F#`, `D#m`, …). Omit the key to let the model choose. Tracks come back in `music.tracks`. Save them for playback. A sound is not a source for video, covers, persona, stems, or replace.

### Lyrics only — 5

```json
{"action": "lyrics"}
```

The prompt is the theme, and only the first 200 characters are sent. `music.lyrics` returns a few titled sheets. "Sing this" on a sheet starts an Easy or Custom song with that sheet in `params.lyrics` and the sheet title in `params.title`. That song is a separate 65-credit job.

### Voice — 2

Kie's create-voice call needs a validation-phrase task id, and that phrase endpoint is not in the API bundle we integrated. Hide this action until the app has that task id.

```json
{
  "action": "voice",
  "source_task_id": "<validation task id>",
  "voice_name": "Ava",
  "description": "Bright tenor",
  "style": "pop belt",
  "singer_skill_level": "intermediate"
}
```

Upload the sung phrase as `verify_audio_r2_key`. Skill is `beginner`, `intermediate`, `advanced`, or `professional`. `music.voice_id` is the validation task id. It can take about a minute before a new song accepts it as `persona_id` with `persona_model: "voice_persona"`.

## Screen shape

Easy, one column:

1. Idea field. Placeholder: "A song about…". Style chips, instrumental switch, vocal, length, and an optional photo, video, or audio clip.
2. Price on the button: 65 credits, two versions, commercial use included.
3. After submit, keep them on the same screen. Use `music.label` as the status. When `lyrics` arrives, show the sheets, with the one in `lyrics_used` marked as the one being sung.
4. When `first_track` arrives, play it with `cover_url`. Don't wait for `completed` to show a player.
5. On `complete`, show both versions side by side. Under the selected version: Make video, Separate vocals, Replace a section, Save persona. New covers sits on the song, once, because both versions share it. Each opens a small sheet and submits a new job. Show that job's price before they confirm. The sheet shows the choice in words ("Version 2 · Starry Night Dreams"), not an id.
6. If they want a different lyric sheet, don't regenerate inside this job. "Sing this" submits a new song with `lyrics` set to the sheet they picked.

Custom is the same player, with title, style, and a lyric editor in place of the idea box. No reference tray. Vocal, length, and variety stay as controls, not text fields. Style weight, weirdness, and audio weight sit under "Fine tune".

Sounds is a separate entry: description, loop switch, tempo slider, key picker. Same player when tracks return. No follow-up actions on a sound.

Stems and replace, opened from anywhere, offer two sources: "My songs" and "Upload audio I own". The upload path shows the rights checkbox and stays disabled until it is checked. Video, covers, and persona offer "My songs" only.

## Limits worth enforcing in the form

- Idea in Easy mode: 3000 characters. The lyric writer only sees the first 200.
- Lyrics: 5000. Style: 1000. Title: 80.
- Duration: 10–360, and only when there is no reference file.
- Images: 5. Video: 1. Attachments of every kind: 10.
- Replace: at least 10 seconds, at most half the song.
- Persona window: 10–30 seconds.
- Variant: three cards, `V6`, `V6_MINI`, `V6_WILD`.
- Uploaded audio for stems or replace: the rights checkbox is on, and the file is not sent together with a saved song.

A 400 from submit is a form error. Show `detail` on the field that caused it. A failed job refunds the credits.

## Submit skeleton

```json
{
  "model": "suno_v6",
  "prompt": "a late-night drive through rain, hopeful by the last chorus",
  "variations": 1,
  "params": {
    "mode": "easy",
    "action": "song",
    "variant": "V6",
    "style": "cinematic pop",
    "vocal_gender": "female"
  }
}
```

Leave `enhance_prompt` off. The server will not rewrite a music idea into an image prompt.
