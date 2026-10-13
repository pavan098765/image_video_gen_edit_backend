from __future__ import annotations

from typing import Any


def empty_storyboard(job_id: str) -> dict[str, Any]:
    return {"job_id": job_id, "scenes": [], "compose_strategy": "single_runtime"}


def upsert_scene(board: dict[str, Any], scene_id: str, **fields: Any) -> dict[str, Any]:
    scenes = board.setdefault("scenes", [])
    for scene in scenes:
        if scene.get("id") == scene_id:
            scene.update(fields)
            return board
    row = {"id": scene_id, "approved": False}
    row.update(fields)
    scenes.append(row)
    return board


def all_approved(board: dict[str, Any]) -> bool:
    scenes = board.get("scenes") or []
    return bool(scenes) and all(s.get("approved") for s in scenes)
