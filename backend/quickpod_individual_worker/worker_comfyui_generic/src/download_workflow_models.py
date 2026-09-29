#!/usr/bin/env python3
"""
download_workflow_models.py
===========================

- **model_config.json** (preferred): paths from workflow scans, filled at prep time;
  you set exact ``hf_repo`` / ``hf_path`` per file. Checked first on every download.
- **model_registry.json** (fallback): shared/static catalog when ``model_config`` has
  no complete entry for a path.

Prep should run ``sync_model_config_from_workflows()`` before ``run_download()`` so
``model_config.json`` lists every weight path your workflows reference.

Environment:
  HF_TOKEN                   Hugging Face token (gated repos)
  MODEL_CONFIG_PATH          Override ``model_config.json`` path
  MODEL_REGISTRY_PATH        Override ``model_registry.json`` path
  SKIP_MODEL_DOWNLOAD           1 = skip prep-time downloads only (startup still pulls unless SKIP_RUNTIME...)
  SKIP_RUNTIME_MODEL_DOWNLOAD   1 = skip download in FastAPI startup
  MODEL_DOWNLOAD_NO_HARDLINK=1  Prefer symlink, then copy (skip os.link) if hardlinks fail in your FS
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_REGISTRY = PROJECT_ROOT / "model_registry.json"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "model_config.json"
WORKFLOW_DIR = Path(os.environ.get("WORKFLOW_DIR", str(PROJECT_ROOT / "workflow")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "comfyui" / "models")))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm_rel(p: str) -> str:
    return p.replace("\\", "/").lstrip("/")


def entry_has_hf(entry: Optional[dict]) -> bool:
    if not entry:
        return False
    repo = (entry.get("hf_repo") or entry.get("repo_id") or "").strip()
    path = (entry.get("hf_path") or entry.get("hf_filename") or "").strip()
    return bool(repo and path)


def hf_identity_key(entry: dict) -> Optional[Tuple[str, str, str]]:
    """Group workflow paths that fetch the same remote object (dedupe disk after one hub download)."""
    if not entry_has_hf(entry):
        return None
    repo = (entry.get("hf_repo") or entry.get("repo_id") or "").strip()
    hf_path = _norm_rel(str(entry.get("hf_path") or entry.get("hf_filename") or ""))
    rev = str(entry.get("revision") or "").strip()
    return (repo, hf_path, rev)


def _materialize_same_bytes(src: Path, dst: Path) -> None:
    """One physical file, many ComfyUI-relative names: try hardlink, then relative symlink, then copy."""
    if dst.is_file() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    no_hardlink = os.environ.get("MODEL_DOWNLOAD_NO_HARDLINK", "").strip() == "1"
    if not no_hardlink:
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except OSError:
            pass
    try:
        if dst.exists():
            dst.unlink()
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def resolve_registry_entry(
    relpath: str,
    files_map: dict,
    aliases: Optional[dict] = None,
) -> Tuple[str, dict]:
    aliases = aliases or {}
    canon = _norm_rel(aliases.get(_norm_rel(relpath), _norm_rel(relpath)))
    entry = files_map.get(canon)
    if entry:
        return canon, entry
    entry = files_map.get(_norm_rel(relpath))
    if entry:
        return _norm_rel(relpath), entry
    base = Path(canon).name
    entry = files_map.get(base)
    if entry:
        return base, entry
    return canon, {}


def resolve_entry_for_download(
    relpath: str,
    model_config: dict,
    model_registry: dict,
) -> dict:
    """
    Prefer model_config.files (+ path_aliases), then model_registry.
    """
    cfg_files = model_config.get("files") or {}
    cfg_aliases = model_config.get("path_aliases") or {}
    reg_files = model_registry.get("files") or {}
    reg_aliases = model_registry.get("path_aliases") or {}

    _, ec = resolve_registry_entry(relpath, cfg_files, cfg_aliases)
    if entry_has_hf(ec):
        return ec

    _, er = resolve_registry_entry(relpath, reg_files, reg_aliases)
    if entry_has_hf(er):
        return er

    return {}


def placement_path(models_dir: Path, relpath: str, entry: dict) -> Path:
    if entry.get("local_relpath"):
        rel = _norm_rel(str(entry["local_relpath"]))
    else:
        rel = _norm_rel(relpath)
    return models_dir / rel


def download_one(
    entry: dict,
    dest_file: Path,
    token: Optional[str],
) -> None:
    repo = entry.get("hf_repo") or entry.get("repo_id")
    hf_path = entry.get("hf_path") or entry.get("hf_filename")
    if not repo or not hf_path:
        raise ValueError(f"Entry missing hf_repo or hf_path: {entry!r}")

    revision = entry.get("revision")

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=repo.strip(),
        filename=_norm_rel(hf_path),
        revision=revision if revision else None,
        token=token,
    )
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if os.path.abspath(cached) != os.path.abspath(dest_file):
        shutil.copy2(cached, dest_file)


def collect_needed_paths(workflow_dir: Path) -> dict:
    from workflow_model_scan import collect_unique_refs, list_workflow_jsons

    paths = list_workflow_jsons(workflow_dir)
    return collect_unique_refs(paths)


def _clean_entry_for_config(entry: dict) -> dict:
    out = {}
    for k in ("hf_repo", "hf_path", "revision", "local_relpath"):
        if k in entry and entry[k] not in (None, ""):
            out[k] = entry[k]
    return out


def sync_model_config_from_workflows(
    *,
    workflow_dir: Path = WORKFLOW_DIR,
    config_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Merge workflow-scanned paths into **model_config.json**.

    - Adds new keys with HF fields copied from **model_registry** when possible,
      otherwise ``hf_repo`` / ``hf_path`` empty strings for you to fill.
    - Never removes existing keys. For existing keys, fills only missing
      ``hf_repo`` / ``hf_path`` / ``revision`` / ``local_relpath`` from registry
      if still empty.
    """
    root = project_root or PROJECT_ROOT
    cfg_path = config_path or Path(
        os.environ.get("MODEL_CONFIG_PATH", str(root / "model_config.json"))
    )
    reg_path = registry_path or Path(
        os.environ.get("MODEL_REGISTRY_PATH", str(root / "model_registry.json"))
    )

    needed = collect_needed_paths(workflow_dir)
    if not needed:
        print("No model-like paths in workflows; model_config.json not updated.")
        return cfg_path

    reg: dict = {}
    if reg_path.is_file():
        reg = load_json(reg_path)

    if cfg_path.is_file():
        cfg = load_json(cfg_path)
    else:
        cfg = {}

    cfg.setdefault("path_aliases", {})
    cfg.setdefault("files", {})
    reg_files = reg.get("files") or {}
    reg_aliases = reg.get("path_aliases") or {}

    for rel, _ref in sorted(needed.items(), key=lambda x: x[0]):
        if rel not in cfg["files"]:
            _, rentry = resolve_registry_entry(rel, reg_files, reg_aliases)
            if entry_has_hf(rentry):
                cfg["files"][rel] = _clean_entry_for_config(rentry)
            else:
                cfg["files"][rel] = {"hf_repo": "", "hf_path": ""}
        else:
            ex = cfg["files"][rel]
            if not isinstance(ex, dict):
                cfg["files"][rel] = {"hf_repo": "", "hf_path": ""}
                ex = cfg["files"][rel]
            if not entry_has_hf(ex):
                _, rentry = resolve_registry_entry(rel, reg_files, reg_aliases)
                if entry_has_hf(rentry):
                    for k, v in _clean_entry_for_config(rentry).items():
                        ex.setdefault(k, v)

    cfg["workflows_scanned"] = sorted(p.name for p in workflow_dir.glob("*.json"))

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {cfg_path} ({len(cfg['files'])} file entries).")
    return cfg_path


def run_download(
    *,
    workflow_dir: Path = WORKFLOW_DIR,
    models_dir: Path = MODELS_DIR,
    registry_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    token: Optional[str] = None,
    dry_run: bool = False,
    runtime: bool = False,
) -> Tuple[list, list]:
    """
    Returns (touched_relpaths, missing_hf_relpaths) — missing = no HF mapping
    in model_config or model_registry.
    """
    if (
        not runtime
        and os.environ.get("SKIP_MODEL_DOWNLOAD", "").strip() == "1"
    ):
        print("SKIP_MODEL_DOWNLOAD=1 — skipping prep-time model downloads.")
        return [], []

    if runtime and os.environ.get("SKIP_RUNTIME_MODEL_DOWNLOAD", "").strip() == "1":
        print("SKIP_RUNTIME_MODEL_DOWNLOAD=1 — skipping runtime model downloads.")
        return [], []

    root = PROJECT_ROOT
    cfg_p = config_path or Path(os.environ.get("MODEL_CONFIG_PATH", str(root / "model_config.json")))
    reg_p = registry_path or Path(os.environ.get("MODEL_REGISTRY_PATH", str(root / "model_registry.json")))

    needed = collect_needed_paths(workflow_dir)
    if not needed:
        print("No model-like paths found in workflows.")
        return [], []

    has_cfg = cfg_p.is_file()
    has_reg = reg_p.is_file()
    if not has_cfg and not has_reg:
        print(
            "Neither model_config.json nor model_registry.json found. "
            "Run prep (syncs model_config) or add model_registry.json.",
            file=sys.stderr,
        )
        return [], sorted(needed.keys())

    model_config: Dict[str, Any] = load_json(cfg_p) if has_cfg else {}
    model_registry: Dict[str, Any] = load_json(reg_p) if has_reg else {}

    models_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    done: list[str] = []

    # Pass 1: missing HF map
    rows: List[Tuple[str, Any, dict, Path]] = []
    for rel, ref in sorted(needed.items(), key=lambda x: x[0]):
        entry = resolve_entry_for_download(rel, model_config, model_registry)
        dest = placement_path(models_dir, rel, entry or {})
        if not entry_has_hf(entry):
            missing.append(rel)
            print(
                f"  MISSING HF MAP: {rel} (from {ref.class_type}.{ref.input_name}) — "
                "set in model_config.json or model_registry.json"
            )
            continue
        rows.append((rel, ref, entry, dest))

    # Pass 2: groups sharing the same Hugging Face file (many workflow filenames → one blob)
    groups: Dict[Tuple[str, str, str], List[Tuple[str, Any, dict, Path]]] = defaultdict(list)
    for rel, ref, entry, dest in rows:
        key = hf_identity_key(entry)
        if key:
            groups[key].append((rel, ref, entry, dest))

    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        repo, hf_path, _rev = key
        # Already satisfied paths
        for rel, _ref, _entry, dest in members:
            if dest.is_file() and dest.stat().st_size > 0:
                print(f"  exists: {rel}")

        need_materialize = [
            (rel, ref, entry, dest)
            for rel, ref, entry, dest in members
            if not (dest.is_file() and dest.stat().st_size > 0)
        ]
        if not need_materialize:
            done.extend([rel for rel, _, _, _ in members])
            continue

        # Prefer an existing file in the group as the canonical source
        canonical_src: Optional[Path] = None
        for rel, _ref, _entry, dest in members:
            if dest.is_file() and dest.stat().st_size > 0:
                canonical_src = dest
                break

        if canonical_src is None:
            need_materialize.sort(key=lambda x: x[0])
            rel0, _ref0, entry0, dest0 = need_materialize[0]
            print(f"  download: {rel0} <- {repo}/{hf_path}")
            if dry_run:
                for rel, _, _, _ in members:
                    done.append(rel)
                continue
            try:
                download_one(entry0, dest0, token=token or os.environ.get("HF_TOKEN"))
                canonical_src = dest0
                done.append(rel0)
            except Exception as e:
                print(f"  FAILED {rel0}: {e}", file=sys.stderr)
                raise

        if dry_run:
            for rel, _, _, _ in members:
                if rel not in done:
                    done.append(rel)
            continue

        for rel, _ref, _entry, dest in members:
            if dest.is_file() and dest.stat().st_size > 0:
                if rel not in done:
                    done.append(rel)
                continue
            assert canonical_src is not None
            try:
                _materialize_same_bytes(canonical_src, dest)
                try:
                    canon_disp = str(canonical_src.relative_to(models_dir))
                except ValueError:
                    canon_disp = str(canonical_src)
                print(f"  link: {rel} <- same bytes as {canon_disp}")
                done.append(rel)
            except Exception as e:
                print(f"  FAILED link {rel}: {e}", file=sys.stderr)
                raise

    return done, missing


def ensure_models_for_workflows(
    *,
    workflow_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> None:
    """
    Used at FastAPI startup: download missing weights before ComfyUI serves prompts.
    """
    wd = workflow_dir or Path(os.environ.get("WORKFLOW_DIR", str(PROJECT_ROOT / "workflow")))
    md = models_dir or Path(os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "comfyui" / "models")))
    _, missing = run_download(workflow_dir=wd, models_dir=md, token=token, runtime=True)
    if missing:
        raise RuntimeError(
            "Model files missing Hugging Face mapping (model_config first, registry fallback): "
            + ", ".join(missing[:30])
            + (" ..." if len(missing) > 30 else "")
        )


def write_stub_registry(missing: list[str], out_path: Path) -> None:
    stub = {
        "path_aliases": {},
        "files": {
            rel: {
                "hf_repo": "ORG/REPO",
                "hf_path": f"path/inside/repo/{Path(rel).name}",
            }
            for rel in missing
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", type=Path, default=WORKFLOW_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sync-config-only", action="store_true", help="Only refresh model_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-stub",
        nargs="?",
        const=PROJECT_ROOT / "model_registry.stub.json",
        type=Path,
        help="Write stub for missing HF mappings",
    )
    args = parser.parse_args()

    if args.sync_config_only:
        sync_model_config_from_workflows(workflow_dir=args.workflow_dir)
        return 0

    try:
        done, missing = run_download(
            workflow_dir=args.workflow_dir,
            models_dir=args.models_dir,
            registry_path=args.registry,
            config_path=args.config,
            dry_run=args.dry_run,
        )
    except Exception:
        return 3

    if missing:
        stub_path = args.write_stub
        if stub_path is not None:
            write_stub_registry(missing, stub_path)
            print(f"Wrote stub for {len(missing)} paths -> {stub_path}")
        print(
            f"\n{len(missing)} asset(s) lack HF info. "
            "Edit model_config.json (preferred) or model_registry.json.",
            file=sys.stderr,
        )
        return 2

    print(f"Model download OK ({len(done)} fetched or skipped as existing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
