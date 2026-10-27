#!/usr/bin/env python3
"""Submit the six baby-photoshoot jobs in reference order.

Put every photo in /workspace/style_refs. The script reads only those files.
The script sends the files in the order below and writes <imageN> tags to match.

<image1> is always that shoot's target photo. Output defaults to 1K, about 1 megapixel,
with the same aspect as that target photo. SIZE=2k asks for about 4 megapixels.

File names (png, jpg, jpeg, or webp). Spaces, underscores, and hyphens all match.

  target_1   shoot 1  sleeping newborn, mustard turban
  target_2   shoot 2  spa towel turban, face uncovered
  target_3   shoot 3  overhead music portrait
  target_4   shoot 4  family portrait, pink swaddle
  target_5   shoot 5  parents holding a sleeping newborn
  target_6   shoot 6  father and baby, black and white
  baby_1     baby identity, required
  baby_2     baby identity, used when the file is there
  baby_3     baby identity, used only when baby_2 is there too
  mom_1      mother, shoots 4 and 5
  mom_2      second mother photo, optional, shoots 4 and 5
  dad_1      father, shoots 4, 5, and 6
  dad_2      second father photo, optional, shoots 4, 5, and 6

mother_1 and father_1 are accepted as mom_1 and dad_1.

A gap stops the series: baby_3 is ignored when baby_2 is missing.
Mom photos are not sent on shoots 1, 2, 3, or 6.

  python3 run_baby_shoots.py
  python3 run_baby_shoots.py 1 4
  python3 run_baby_shoots.py all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("API", "http://127.0.0.1:8000")
COMFY = os.environ.get("COMFY", "http://127.0.0.1:8188")
COMFY_LOG = os.environ.get("COMFY_LOG", "/app/comfyui/comfyui_server.log")
REF_DIR_DEFAULT = os.environ.get("REF_DIR", "/workspace/style_refs")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/workspace/results"))
TIMEOUT = int(os.environ.get("SHOOT_TIMEOUT", "1200"))

SHOOTS = [
    {"id": 1, "slug": "sleeping_newborn", "title": "Sleeping newborn, mustard-yellow turban", "mom": False, "dad": False},
    {"id": 2, "slug": "spa_towel", "title": "Spa-themed baby, towel turban, face uncovered", "mom": False, "dad": False},
    {"id": 3, "slug": "music_overhead", "title": "Music-themed overhead baby portrait", "mom": False, "dad": False},
    {"id": 4, "slug": "family_pink_swaddle", "title": "Mother, father, and baby, pink swaddle", "mom": True, "dad": True, "baby_count": 1},
    {"id": 5, "slug": "parents_holding", "title": "Parents holding a sleeping newborn", "mom": True, "dad": True, "baby_count": 1},
    {"id": 6, "slug": "father_baby_mono", "title": "Father and baby, monochrome side profile", "mom": False, "dad": True, "baby_count": 1},
]

EXTS = {".png", ".jpg", ".jpeg", ".webp"}
NAME_RE = re.compile(r"^(target|baby|mom|mother|dad|father)[ _-]?(\d+)$", re.IGNORECASE)
ROLE_ALIAS = {"mother": "mom", "father": "dad"}


def say(text: str = "") -> None:
    print(text, flush=True)


def tag(index: int) -> str:
    return f"<image{index}>"


def join_tags(indexes: list[int]) -> str:
    tags = [tag(i) for i in indexes]
    if len(tags) == 1:
        return tags[0]
    if len(tags) == 2:
        return f"{tags[0]} and {tags[1]}"
    return ", ".join(tags[:-1]) + ", and " + tags[-1]


def consecutive(found: dict[tuple[str, int], Path], role: str) -> list[Path]:
    """Files role_1, role_2, ... until the first missing number."""
    paths: list[Path] = []
    n = 1
    while (role, n) in found:
        paths.append(found[(role, n)])
        n += 1
    return paths


def gap_notes(found: dict[tuple[str, int], Path]) -> None:
    for role in ("baby", "mom", "dad"):
        numbers = sorted(i for (r, i) in found if r == role)
        if not numbers:
            continue
        if numbers[0] != 1:
            say(f"note: {role}_1 is missing, so {role}_{numbers[0]} is not used. Start at {role}_1.")
            continue
        missing = [i for i in range(1, numbers[-1] + 1) if (role, i) not in found]
        if missing:
            say(
                f"note: {role}_{missing[0]} is missing, so later {role}_ files are not used. "
                "Number the files with no gap."
            )


def index_dir(folder: Path) -> tuple[dict[tuple[str, int], Path], list[Path]]:
    found: dict[tuple[str, int], Path] = {}
    unused: list[Path] = []
    if not folder.is_dir():
        return found, unused
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            if path.is_file():
                unused.append(path)
            continue
        match = NAME_RE.match(path.stem.strip())
        if not match:
            unused.append(path)
            continue
        role = ROLE_ALIAS.get(match.group(1).lower(), match.group(1).lower())
        key = (role, int(match.group(2)))
        if key in found:
            say(f"  note: keeping {found[key].name}, ignoring duplicate {path.name}")
            continue
        found[key] = path
    return found, unused


def build_order(found: dict[tuple[str, int], Path], shoot: dict) -> tuple[list[tuple[str, Path]], str | None]:
    target = found.get(("target", shoot["id"]))
    babies = consecutive(found, "baby")
    baby_count = shoot.get("baby_count")
    if baby_count and len(babies) > int(baby_count):
        babies = babies[: int(baby_count)]
    moms = consecutive(found, "mom") if shoot["mom"] else []
    dads = consecutive(found, "dad") if shoot["dad"] else []
    missing = []
    if target is None:
        missing.append(f"target_{shoot['id']}")
    if not babies:
        missing.append("baby_1")
    if shoot["mom"] and not moms:
        missing.append("mom_1")
    if shoot["dad"] and not dads:
        missing.append("dad_1")
    if missing:
        return [], "missing " + ", ".join(missing)
    ordered: list[tuple[str, Path]] = [("target", target)]
    ordered.extend(("baby", path) for path in babies)
    ordered.extend(("mom", path) for path in moms)
    ordered.extend(("dad", path) for path in dads)
    if len(ordered) > 10:
        return [], f"{len(ordered)} images is over the 10-image limit"
    return ordered, None


def slots_for(ordered: list[tuple[str, Path]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"target": [], "baby": [], "mom": [], "dad": []}
    for index, (role, _path) in enumerate(ordered, start=1):
        groups[role].append(index)
    return groups


def prompt_for(shoot_id: int, slots: dict[str, list[int]]) -> str:
    t = tag(slots["target"][0])
    b = join_tags(slots["baby"])
    baby_many = len(slots["baby"]) > 1
    baby_line = (
        f"Treat {b} exclusively as identity references of the same baby, combining the views to preserve the baby's real identity and natural age-appropriate appearance."
        if baby_many
        else f"Treat {b} exclusively as the identity reference of the baby, and preserve that baby's real identity and natural age-appropriate appearance."
    )
    if shoot_id == 1:
        return (
            f"Use {t} as the target photoshoot reference for the pose, composition, camera angle, lighting, props, blanket, and background. "
            f"Replace the baby shown in {t} with the real baby whose identity is shown in {b}. {baby_line} "
            "Recreate the sleeping newborn portrait with the baby wearing a mustard-yellow soft fabric turban, resting peacefully with both arms gently crossed beneath the chin, wrapped in a textured cream or ivory knitted blanket. "
            f"Preserve the target reference's dark charcoal studio background, soft directional lighting, gentle shadows, close framing, and peaceful newborn photography aesthetic. "
            f"Keep the pose and styling faithful to {t} while ensuring the face belongs to the baby in the identity references, not the baby in the target image. "
            "Maintain natural newborn proportions, realistic skin texture, anatomically correct hands, and physically plausible fabric folds. "
            "Do not add accessories, extra fingers, duplicate limbs, text, or watermarks."
        )
    if shoot_id == 2:
        baby_kind = "identity references of the same baby" if baby_many else "the identity reference of the baby"
        return (
            f"The baby's face must be the real baby in {b}. Those photos are {baby_kind}. "
            "Match that baby's eyes, nose, mouth, cheeks, skin tone, and age. "
            f"Do not copy the face from {t}. "
            f"Use {t} only for the body pose, crop, yellow background, lighting, and composition. "
            "Dress the baby in a white bathrobe and wrap a white bath towel around the head, twisted into a conventional towel turban. This must look like a bath towel, not bunny ears, an animal costume, or a hood with ears. "
            "Keep the whole face uncovered: both eyes, the nose, and the mouth are visible, with nothing resting on the face. "
            "Use the same round mustard-yellow cushion and warm yellow studio background, with soft, even professional lighting and a clean spa-portrait aesthetic. "
            f"Keep the composition faithful to {t}; use the identity images only for the baby's face, not their pose. "
            "Ensure realistic towel folds, natural hands, and age-appropriate proportions. "
            "No objects on the face, bunny ears, added accessories, text, or watermarks."
        )
    if shoot_id == 3:
        return (
            f"Use {t} as the target scene reference for the overhead camera angle, arrangement of props, baby's position, background, lighting, and overall layout. "
            f"Replace the baby in {t} with the real baby whose identity is shown in {b}. "
            f"{baby_line} Use the identity photos only to preserve identity, not to copy their poses. "
            "Recreate the scene with the baby lying safely on a mustard-yellow swaddle, wearing oversized white headphones positioned comfortably around the head. "
            f"Arrange the music-player prop and its cable as shown in {t}, with a bright yellow background, handmade felt sun and cloud decorations, and loosely arranged yellow fabric around the edges. "
            "Match the target reference's overhead framing, relative object sizes, spacing, color palette, and soft studio lighting. "
            "Keep the baby's face recognizable as the actual baby from the identity references, with natural age-appropriate proportions and realistic skin texture. "
            "The headphones must fit plausibly without covering or deforming the face; the cable and props must be physically coherent. "
            "Preserve the reference's graphic, carefully styled photoshoot aesthetic without adding unrelated objects, extra limbs, duplicated props, text, or watermarks."
        )
    if shoot_id == 4:
        m = join_tags(slots["mom"])
        d = join_tags(slots["dad"])
        return (
            f"{t} is the canvas. Keep its composition, camera angle, dark lighting, background, and the three body positions. "
            f"Replace the baby in {t} with the baby in {b}, wrapped in a dusty-pink swaddle, with the head and body fully inside the frame. "
            f"Replace the woman in {t} with the woman in {m}, transferring her face so she is unmistakably that woman. "
            f"Replace the man in {t} with the man in {d}, transferring his face so he is unmistakably that man. "
            "The result still shows exactly those three people, with natural hands and realistic contact."
        )
    if shoot_id == 5:
        m = join_tags(slots["mom"])
        d = join_tags(slots["dad"])
        return (
            f"{t} is the canvas. Keep its composition, camera angle, white studio background, soft lighting, and the way the parents hold the sleeping baby. "
            f"Replace the sleeping baby in {t} with the baby in {b}, head and body fully inside the frame and supported by the parents' arms. "
            f"Replace the woman in {t} with the woman in {m}, transferring her face so she is unmistakably that woman, and dress her in beige or cream. "
            f"Replace the man in {t} with the man in {d}, transferring his face so he is unmistakably that man, and dress him in beige or cream. "
            "The result still shows exactly those three people, with natural hands."
        )
    d = join_tags(slots["dad"])
    return (
        f"{t} is the canvas. Keep its side-profile arrangement, camera angle, light background, and the way the man holds the baby. "
        f"Replace the baby in {t} with the baby in {b}, with the head and body fully inside the frame. "
        f"Replace the man in {t} with the man in {d}, transferring his face so he is unmistakably that man, in a simple light-colored shirt. "
        "Render the finished photograph in black and white. The result still shows exactly those two people, with natural hands."
    )


def describe(shoot: dict, ordered: list[tuple[str, Path]], prompt: str) -> None:
    say(f"shoot {shoot['id']}: {shoot['title']}")
    for index, (role, path) in enumerate(ordered, start=1):
        say(f"  <image{index}>  {role:6}  {path}")
    say("  output: 1K by default (set SIZE=2k only on a larger GPU), same aspect as <image1>")
    say(prompt)
    say("")


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 30) -> tuple[int, dict | str]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            code = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        code = exc.code
    except urllib.error.URLError as exc:
        return 0, f"cannot reach {url}: {exc.reason}"
    try:
        return code, json.loads(raw)
    except json.JSONDecodeError:
        return code, raw


def png_snapshot() -> dict[str, float]:
    found: dict[str, float] = {}
    for folder in ("/workspace", "/app/comfyui/output"):
        for path in glob.glob(folder + "/*.png"):
            try:
                found[path] = os.path.getmtime(path)
            except OSError:
                pass
    return found


def image_wh(path: str) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as handle:
            blob = handle.read(65536)
    except OSError:
        return None
    if blob.startswith(b"\x89PNG") and len(blob) >= 24:
        width, height = struct.unpack(">II", blob[16:24])
        return int(width), int(height)
    if not blob.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(blob):
        if blob[index] != 0xFF:
            index += 1
            continue
        marker = blob[index + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", blob[index + 5 : index + 9])
            return int(width), int(height)
        if marker in (0xD8, 0xD9):
            index += 2
            continue
        if index + 4 > len(blob):
            break
        segment = struct.unpack(">H", blob[index + 2 : index + 4])[0]
        index += 2 + segment
    return None


def comfy_step() -> str:
    for route in ("/progress", "/api/progress"):
        code, data = http_json("GET", COMFY + route, timeout=5)
        if code != 200 or not isinstance(data, dict):
            continue
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        value = payload.get("value")
        maximum = payload.get("max")
        if value is None or not maximum:
            continue
        node = payload.get("node") or ""
        label = f"step {value}/{maximum}"
        return f"{label} {node}".strip()
    return ""


def follow_job(job_id: str, before: dict[str, float]) -> tuple[str, str]:
    offset = 0
    if os.path.isfile(COMFY_LOG):
        offset = os.path.getsize(COMFY_LOG)
    else:
        say(f"comfy log not found yet: {COMFY_LOG}")
    deadline = time.time() + TIMEOUT
    started = time.time()
    status = None
    error = None
    prompt_id = ""
    step = ""
    while time.time() < deadline:
        new_lines = []
        if os.path.isfile(COMFY_LOG):
            size = os.path.getsize(COMFY_LOG)
            if size < offset:
                offset = 0
            if size > offset:
                with open(COMFY_LOG, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset = handle.tell()
                for line in chunk.splitlines():
                    text = line.strip()
                    if not text or len(text) > 400:
                        continue
                    low = text.lower()
                    if any(word in low for word in ("error", "exception", "traceback", "%", "sampl", "step", "execut", "progress", "prompt executed", "loaded")):
                        new_lines.append(text)
                    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
                    if match and int(match.group(2)) <= 64:
                        step = f"step {match.group(1)}/{match.group(2)}"
        for line in new_lines:
            say(f"  comfy  {line}")
        live = comfy_step()
        if live:
            step = live
        code, st = http_json("GET", API + "/status", timeout=20)
        queue_size = None
        if isinstance(st, dict):
            job = (st.get("jobs") or {}).get(job_id) or {}
            status = job.get("status")
            error = job.get("error")
            queue_size = st.get("queue_size")
        else:
            status = f"status_http_{code}"
        try:
            _qcode, queue = http_json("GET", COMFY + "/queue", timeout=10)
            if isinstance(queue, dict):
                active = queue.get("queue_running") or []
                if active and isinstance(active[0], list) and len(active[0]) > 1:
                    prompt_id = str(active[0][1])
        except Exception as exc:
            prompt_id = f"queue_error:{exc}"
        elapsed = int(time.time() - started)
        say(
            f"[{time.strftime('%H:%M:%S')}] {job_id}  {status}  {elapsed}s"
            f"  queue={queue_size}  {step or 'waiting for sampler'}"
            + (f"  error={error}" if error else "")
        )
        if status in ("completed", "failed"):
            break
        time.sleep(3)
    else:
        return "timeout", "no completed or failed status before the time limit"

    created = []
    after = png_snapshot()
    for path, mtime in after.items():
        if path not in before or mtime > before.get(path, 0) + 0.01:
            created.append(path)
    created.sort(key=lambda path: after[path], reverse=True)
    if prompt_id and not created:
        _hcode, history = http_json("GET", f"{COMFY}/history/{prompt_id}", timeout=20)
        outputs = history.get(prompt_id, {}).get("outputs") if isinstance(history, dict) else None
        if isinstance(outputs, dict):
            for node in outputs.values():
                if not isinstance(node, dict):
                    continue
                for image in node.get("images") or []:
                    filename = str(image.get("filename") or "")
                    if filename:
                        created.append(str(Path("/app/comfyui/output") / filename))
    if status == "failed":
        return "failed", error or "job failed"
    if not created:
        return "completed", ""
    return "completed", created[0]


def submit(shoot: dict, ordered: list[tuple[str, Path]], prompt: str, index: int, total: int) -> str:
    job_id = f"shoot{shoot['id']:02d}-{int(time.time())}"
    paths = [str(path) for _role, path in ordered]
    body = {
        "job_id": job_id,
        "prompt": prompt,
        "model": "qwen_image_21_turbo",
        "upload_url": "http://dummy-upload/local",
        "result_key": f"local/{job_id}.png",
        "params": {
            "seed": int(os.environ.get("SEED", "1")),
            "init_image": paths,
            "size": os.environ.get("SIZE", "1k"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = OUT_DIR / f"shoot_{shoot['id']:02d}_{shoot['slug']}_request.json"
    record.write_text(json.dumps({"request": body, "files": paths}, indent=2), encoding="utf-8")
    say("")
    tier = str(body["params"]["size"]).upper()
    say(f"=== prompt {index}/{total}  shoot {shoot['id']}  {shoot['title']}  {tier} ===")
    say(f"request {record}")
    before = png_snapshot()
    code, resp = http_json("POST", API + "/run", body, timeout=60)
    say(f"POST /run -> {code} {resp if isinstance(resp, dict) else resp}")
    if code not in (200, 202):
        return "failed"
    status, output = follow_job(job_id, before)
    if status == "completed" and output and os.path.isfile(output):
        dest = OUT_DIR / f"shoot_{shoot['id']:02d}_{shoot['slug']}{Path(output).suffix}"
        shutil.copy2(output, dest)
        wh = image_wh(str(dest))
        pixels = f"  {wh[0]}x{wh[1]}" if wh else ""
        say(f"RESULT {dest}{pixels}")
        if wh and wh[0] * wh[1] < 3_000_000:
            say(f"warning: {wh[0]}x{wh[1]} is below a 2K frame")
        return "completed"
    reason = output or "no output file"
    if "out of memory" in reason.lower():
        reason = "GPU out of memory. This 16GB card cannot sample 2K with several reference photos. Retry with SIZE=1k."
    elif len(reason) > 500:
        reason = reason[:500] + " ..."
    say(f"shoot {shoot['id']} {status}: {reason}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Map shoot photos and submit Qwen baby-photoshoot jobs.")
    parser.add_argument("shoots", nargs="*", help="Shoot numbers, or 'all'. Omit to print the map only.")
    parser.add_argument("--dir", default=REF_DIR_DEFAULT, help="Folder of target_N, baby_N, mom_N, dad_N files.")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        say(f"photo folder does not exist: {folder}")
        say("put target_1, baby_1, mom_1, dad_1, and the rest in /workspace/style_refs")
        return 1
    found, unused = index_dir(folder)
    say(f"photo folder: {folder}")
    gap_notes(found)
    if not found:
        say("no target_, baby_, mom_, or dad_ photos found.")
        say("save files there, then run this script again with no arguments to see the map.")
        return 1
    if unused:
        say("ignored files (rename these if they should be used):")
        for path in unused:
            say(f"  {path.name}")
        say("expected names look like baby_1.jpg, dad_2.png, mom_1.png, target_3.jpg")

    selected: list[dict]
    if not args.shoots:
        selected = SHOOTS
        submit_jobs = False
    elif args.shoots == ["all"]:
        selected = SHOOTS
        submit_jobs = True
    else:
        wanted = []
        for token in args.shoots:
            if not token.isdigit() or int(token) not in range(1, 7):
                say(f"unknown shoot {token!r}. Use 1 2 3 4 5 6 or all.")
                return 2
            wanted.append(int(token))
        selected = [shoot for shoot in SHOOTS if shoot["id"] in wanted]
        submit_jobs = True

    ready: list[tuple[dict, list[tuple[str, Path]], str]] = []
    for shoot in selected:
        ordered, problem = build_order(found, shoot)
        if problem:
            say(f"shoot {shoot['id']}: {shoot['title']} — {problem}")
            say("")
            continue
        prompt = prompt_for(shoot["id"], slots_for(ordered))
        describe(shoot, ordered, prompt)
        ready.append((shoot, ordered, prompt))

    if not submit_jobs:
        say("map only. Nothing was submitted.")
        say("first test:  python3 run_baby_shoots.py 1 4")
        say("all six:     python3 run_baby_shoots.py all")
        return 0 if ready else 1

    if not ready:
        say("nothing to submit.")
        return 1

    code, status = http_json("GET", API + "/status", timeout=20)
    if not isinstance(status, dict):
        say(f"worker status failed: {code} {status}")
        return 1
    say(
        f"startup={status.get('startup_completed')} comfy_ready={status.get('comfyui_ready')} "
        f"error={status.get('comfyui_error')}"
    )
    if not (status.get("startup_completed") and status.get("comfyui_ready") and not status.get("comfyui_error")):
        say("worker is not ready. Wait, then run the same command again.")
        return 1

    total = len(ready)
    for index, (shoot, ordered, prompt) in enumerate(ready, start=1):
        result = submit(shoot, ordered, prompt, index, total)
        if result != "completed":
            say(f"stopped after shoot {shoot['id']} ({result}). Inspect /workspace/results before the next shoot.")
            return 1
    say("finished the requested shoots. Results are in /workspace/results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
