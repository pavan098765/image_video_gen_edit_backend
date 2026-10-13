#!/usr/bin/env python3
"""
install_workflow_nodes.py
=========================

Resolve custom node git URLs from ComfyUI workflow JSON files using
ComfyUI-Manager's extension-node-map.json (class_type -> repo URL), then
clone into COMFYUI_DIR/custom_nodes and install dependencies.

**Identifiers:** ComfyUI-Manager's published ``extension-node-map.json`` maps **Python node
class names** (workflow ``type`` / API ``class_type``), not the workflow graph's numeric
``id`` (those are for links only). Saved UI workflows often include
``properties.cnr_id`` (Comfy Registry id) and ``properties.aux_id`` (``author/repo``);
those are used as **preferred ``cm-cli install`` names** when they differ from the git
repo folder name. A JSON summary is written when the run finishes (see
``--summary-out`` / ``NODE_INSTALL_SUMMARY_PATH``).

Usage:
  COMFYUI_DIR=/path/to/ComfyUI python install_workflow_nodes.py workflow/a.json [workflow/b.json ...]
  python install_workflow_nodes.py --workflow-dir /path/to/workflow

Environment:
  COMFYUI_DIR          Root ComfyUI checkout (default: ./comfyui next to project)
  MANAGER_BASE_URL     Base URL for Manager raw files (no trailing slash)
  OVERRIDES_PATH       JSON file with resolve_conflicts / missing_nodes / extra_repos
  ALLOW_NODE_PIP_FAIL       If "1", log pip failures but continue (default: strict)
  INSTALL_STRICT_NODE_CONFLICTS  If "1", fail on ambiguous multi-repo nodes (default: auto tie-break)
  USE_COMFYUI_MANAGER_CLI   If "0", only git clone (default: hybrid cm-cli install + git fallback)
  CM_CLI_RESTORE_DEPS       If "1", run cm-cli restore-dependencies after installs
  NODE_INSTALL_SUMMARY_PATH  Write install report JSON here (default: <project>/node_install_summary.json)
  UNLISTED_NODE_HINTS_PATH   Optional JSON: exact + substring_rules for types missing from extension-node-map
  SKIP_UNLISTED_AUTOMATION=1 Skip built-in + unlisted_node_hints.json (only overrides.json missing_nodes)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Install report (written at end of main)
# ---------------------------------------------------------------------------


def _empty_pack_result(pack: str, url: str) -> Dict[str, Any]:
    return {
        "pack": pack,
        "git_url": url,
        "status": "unknown",
        "method": None,
        "cm_cli_name_used": None,
        "warnings": [],
        "errors": [],
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


DEFAULT_MANAGER_BASE = os.environ.get(
    "MANAGER_BASE_URL",
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main",
)

EXTENSION_MAP_FILE = "extension-node-map.json"
COMFYUI_MANAGER_GIT = "https://github.com/ltdrdata/ComfyUI-Manager.git"
CM_CLI_BATCH_SIZE = int(os.environ.get("CM_CLI_BATCH_SIZE", "12"))

# When extension-node-map lists the same node name on many forks, prefer this URL
# substring (case-insensitive) for the given exact class_type string.
PREFERRED_REPO_SUBSTRING: Dict[str, str] = {
    "ColorCorrect": "EllangoK/ComfyUI-post-processing-nodes",
    "ColorMatchToReference": "vrgamegirl19/comfyui-vrgamedevgirl",
    "DownloadAndLoadFlorence2Model": "kijai/ComfyUI-Florence2",
    "Florence2Run": "kijai/ComfyUI-Florence2",
    "FramePackFindNearestBucket": "PlusOne",
    "GetImageSizeAndCount": "cubiq/ComfyUI_essentials",
    "Image Overlay": "jags111/efficiency-nodes-comfyui",
    "ImageResizeKJv2": "kijai/ComfyUI-KJNodes",
    "ImageUpscaleWithModelBatched": "kijai/ComfyUI-KJNodes",
    "Incrementer 🪴": "Eagle-CN/ComfyUI-Addoor",
    "PreviewBridge": "ltdrdata/ComfyUI-Impact-Pack",
    "SimpleMath+": "cubiq/ComfyUI_essentials",
}

# Types often absent from extension-node-map but shipped with ComfyUI (do not clone a repo).
CORE_COMFYUI_UNLISTED_SKIP: Set[str] = {
    "Note",
    "Reroute",
    "PrimitiveNode",
    "MarkdownNote",
}

# extension-node-map misses these; kijai ComfyUI-MMAudio registers these class names.
BUILTIN_UNLISTED_EXACT: Dict[str, str] = {
    "MMAudioFeatureUtilsLoader": "https://github.com/kijai/ComfyUI-MMAudio.git",
    "MMAudioModelLoader": "https://github.com/kijai/ComfyUI-MMAudio.git",
    "MMAudioSampler": "https://github.com/kijai/ComfyUI-MMAudio.git",
}

RGTHREE_GIT = "https://github.com/rgthree/rgthree-comfy.git"
RGTHREE_TYPE_MARKER = " (rgthree)"

_UUID_NODE_TYPE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _fetch_json(url: str) -> Any:
    print(f"Fetching {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "worker_comfyui_generic-install/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def build_reverse_index(extension_map: dict) -> Dict[str, Set[str]]:
    """Map node class name -> set of extension URLs that declare it."""
    rev: Dict[str, Set[str]] = {}
    for url_key, val in extension_map.items():
        if not isinstance(val, list) or len(val) < 1:
            continue
        names = val[0]
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            rev.setdefault(name, set()).add(url_key)
    return rev


def extract_class_types_from_file(path: Path) -> Set[str]:
    """Extract ComfyUI node type names from visual or API workflow JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    types: Set[str] = set()

    if isinstance(data, dict) and "nodes" in data:
        for node in data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            t = node.get("type") or node.get("class_type")
            if isinstance(t, str) and t and not t.startswith("{"):
                types.add(t)
        return types

    if isinstance(data, dict):
        for _nid, node in data.items():
            if isinstance(node, dict) and node.get("class_type"):
                types.add(node["class_type"])
        return types

    return types


def extract_node_registry_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Per workflow node: graph id, class type, and ComfyUI-Manager-friendly ids when present.

    Numeric workflow ``id`` is only for graph wiring; ``cnr_id`` / ``aux_id`` come from
    the desktop UI save format and align better with ``cm-cli`` / the registry.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    wf_name = path.name

    if isinstance(data, dict) and "nodes" in data:
        for node in data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            t = node.get("type") or node.get("class_type")
            if not isinstance(t, str) or not t.strip() or t.startswith("{"):
                continue
            pr = node.get("properties")
            if not isinstance(pr, dict):
                pr = {}
            cnr = pr.get("cnr_id")
            aux = pr.get("aux_id")
            rows.append(
                {
                    "workflow": wf_name,
                    "graph_node_id": node.get("id"),
                    "class_type": t.strip(),
                    "cnr_id": cnr.strip() if isinstance(cnr, str) and cnr.strip() else None,
                    "aux_id": aux.strip() if isinstance(aux, str) and aux.strip() else None,
                }
            )
        return rows

    if isinstance(data, dict):
        for nid, node in data.items():
            if not isinstance(node, dict) or not node.get("class_type"):
                continue
            ct = node["class_type"]
            if not isinstance(ct, str) or not ct.strip():
                continue
            rows.append(
                {
                    "workflow": wf_name,
                    "graph_node_id": str(nid),
                    "class_type": ct.strip(),
                    "cnr_id": None,
                    "aux_id": None,
                }
            )

    return rows


def build_cm_cli_try_names_per_pack(
    url_by_node: Dict[str, str],
    extra_url_set: Set[str],
    registry_rows: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    For each git URL basename (pack folder), ordered names to try with ``cm-cli install``.

    Order: ``cnr_id`` and ``aux_id`` (and ``author/Repo`` tail) from workflow nodes whose
    ``class_type`` maps to that URL, then the default pack folder name.
    """
    hints: Dict[str, List[str]] = {}

    for url in extra_url_set:
        if not is_probably_git_clone_url(url):
            continue
        nu = normalize_git_url(url)
        pack = repo_folder_name(nu)
        if pack == "ComfyUI-Manager":
            continue

        types_for_url = {t for t, u in url_by_node.items() if normalize_git_url(u) == nu}
        ordered: List[str] = []

        def add(x: Optional[str]) -> None:
            if not x:
                return
            s = x.strip()
            if s and s not in ordered:
                ordered.append(s)

        for row in registry_rows:
            if row.get("class_type") not in types_for_url:
                continue
            add(row.get("cnr_id"))
            aid = row.get("aux_id")
            add(aid)
            if isinstance(aid, str) and "/" in aid:
                add(aid.split("/")[-1].strip())

        add(pack)
        hints[pack] = ordered

    return hints


def collect_workflow_paths(workflow_dir: Path) -> List[Path]:
    return sorted(workflow_dir.glob("*.json"))


def load_overrides(path: Optional[Path]) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns (node_name -> git_url, extra_repos).

    JSON shape:
    {
      "resolve_conflicts": { "SomeNode": "https://github.com/org/repo.git" },
      "missing_nodes": { "PrivateNode": "https://..." },
      "extra_repos": ["https://github.com/org/other.git"]
    }
    """
    if path is None or not path.is_file():
        return {}, []

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    by_node: Dict[str, str] = {}
    extras: List[str] = []

    if isinstance(raw, dict):
        for key in ("resolve_conflicts", "missing_nodes"):
            block = raw.get(key)
            if isinstance(block, dict):
                for n, u in block.items():
                    if isinstance(n, str) and isinstance(u, str):
                        by_node[n] = u.strip()
        er = raw.get("extra_repos")
        if isinstance(er, list):
            extras = [str(u).strip() for u in er if str(u).strip()]

    return by_node, extras


def normalize_git_url(url: str) -> str:
    u = url.strip()
    if u.endswith("/"):
        u = u[:-1]
    return u


def repo_folder_name(url: str) -> str:
    u = normalize_git_url(url).rstrip("/")
    base = u.split("/")[-1]
    if base.endswith(".git"):
        base = base[:-4]
    return base or "custom_node"


def is_probably_git_clone_url(url: str) -> bool:
    u = url.lower()
    return "github.com/" in u or u.startswith("git@") or u.endswith(".git")


def load_unlisted_hints_file(project_root: Path) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Optional ``unlisted_node_hints.json`` next to the worker (or ``UNLISTED_NODE_HINTS_PATH``).

    Shape::
      { "exact": { "NodeType": "https://github.com/...git" },
        "substring_rules": [ { "contains": "Substring", "git_url": "https://..." } ] }
    """
    raw_path = os.environ.get("UNLISTED_NODE_HINTS_PATH", "").strip()
    path = Path(raw_path) if raw_path else project_root / "unlisted_node_hints.json"
    if not path.is_file():
        return {}, []

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    exact: Dict[str, str] = {}
    block = raw.get("exact")
    if isinstance(block, dict):
        for k, v in block.items():
            if isinstance(k, str) and isinstance(v, str) and v.strip():
                exact[k] = v.strip()

    subs: List[Tuple[str, str]] = []
    for item in raw.get("substring_rules", []) or []:
        if not isinstance(item, dict):
            continue
        c = item.get("contains")
        u = item.get("git_url")
        if isinstance(c, str) and isinstance(u, str) and c and u.strip():
            subs.append((c, normalize_git_url(u.strip())))

    return exact, subs


def augment_overrides_for_unlisted_node_types(
    all_types: Set[str],
    rev: Dict[str, Set[str]],
    user_node_urls: Dict[str, str],
    project_root: Path,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For node types with **no** extension-node-map entry, add git URLs from optional
    hints file and small built-in rules (rgthree display suffix, MMAudio).

    User ``overrides.json`` entries always win. Does not override extension-map types.

    Returns (merged node->url dict, applied_log, skipped_as_core_or_uuid_log).
    """
    merged = dict(user_node_urls)
    file_exact, file_subs = load_unlisted_hints_file(project_root)
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for nt in sorted(all_types):
        if nt in rev:
            continue
        if nt in merged:
            continue

        if nt in CORE_COMFYUI_UNLISTED_SKIP:
            skipped.append({"node_type": nt, "reason": "assumed_comfyui_core"})
            continue
        if _UUID_NODE_TYPE.match(nt.strip()):
            skipped.append({"node_type": nt, "reason": "uuid_like_subgraph_or_stale_widget"})
            continue

        url: Optional[str] = None
        rule: Optional[str] = None

        if nt in file_exact:
            url, rule = normalize_git_url(file_exact[nt]), "hints_file_exact"
        else:
            for sub, u in file_subs:
                if sub in nt:
                    url, rule = u, f"hints_file_substring:{sub!r}"
                    break

        if not url and nt in BUILTIN_UNLISTED_EXACT:
            url, rule = BUILTIN_UNLISTED_EXACT[nt], "builtin_exact"
        if not url and RGTHREE_TYPE_MARKER in nt:
            url, rule = RGTHREE_GIT, "builtin_rgthree_marker"

        if url and is_probably_git_clone_url(url):
            merged[nt] = url
            applied.append({"node_type": nt, "git_url": url, "source": rule})
            print(f"  auto unlisted: {nt!r} -> {url} ({rule})", file=sys.stderr)

    return merged, applied, skipped


def install_one_repo(
    clone_url: str,
    dest_dir: Path,
    *,
    allow_pip_fail: bool,
    update: bool = False,
) -> Tuple[str, List[str]]:
    """
    Returns (outcome, warnings). outcome is ``cloned`` or ``skipped_existing``.
    """
    warnings: List[str] = []
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    outcome = "skipped_existing"

    if dest_dir.is_dir() and (dest_dir / ".git").is_dir():
        print(f"  Repo exists: {dest_dir.name} (skip clone)")
        if update:
            subprocess.run(
                ["git", "-C", str(dest_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
            )
    else:
        if dest_dir.exists():
            raise RuntimeError(f"Path exists and is not a git repo: {dest_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        outcome = "cloned"

    req = dest_dir / "requirements.txt"
    if req.is_file():
        print(f"  pip install -r requirements.txt ({dest_dir.name})")
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(req)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            msg = r.stderr or r.stdout
            if allow_pip_fail:
                w = f"pip -r requirements.txt failed for {dest_dir.name}: {(msg or '')[:1200]}"
                print(f"  WARNING {w}", file=sys.stderr)
                warnings.append(w)
            else:
                raise RuntimeError(f"pip install failed for {dest_dir}: {msg}")

    install_py = dest_dir / "install.py"
    if install_py.is_file():
        print(f"  python install.py ({dest_dir.name})")
        r = subprocess.run(
            [sys.executable, str(install_py)],
            cwd=str(dest_dir),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"install.py failed for {dest_dir}: {r.stderr or r.stdout}"
            )

    if (dest_dir / "pyproject.toml").is_file() or (dest_dir / "setup.py").is_file():
        print(f"  pip install -e ({dest_dir.name})")
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", str(dest_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            msg = r.stderr or r.stdout
            if allow_pip_fail:
                w = f"pip install -e failed for {dest_dir.name}: {(msg or '')[:1200]}"
                print(f"  WARNING {w}", file=sys.stderr)
                warnings.append(w)
            else:
                raise RuntimeError(f"pip install -e failed for {dest_dir}: {msg}")

    return outcome, warnings


def ensure_comfyui_manager(comfy_dir: Path, *, allow_pip_fail: bool) -> Path:
    """Clone ComfyUI-Manager if missing and install its pip requirements (for cm-cli)."""
    custom_nodes = comfy_dir / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)
    m = custom_nodes / "ComfyUI-Manager"
    if (m / "cm-cli.py").is_file():
        print("ComfyUI-Manager present (cm-cli available).")
    else:
        print(f"Cloning ComfyUI-Manager into {m} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", COMFYUI_MANAGER_GIT, str(m)],
            check=True,
            capture_output=True,
            text=True,
        )
    req = m / "requirements.txt"
    if req.is_file():
        print("pip install ComfyUI-Manager requirements ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(req)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            msg = r.stderr or r.stdout
            if allow_pip_fail:
                print(f"WARNING Manager pip: {msg[:1500]}", file=sys.stderr)
            else:
                raise RuntimeError(f"ComfyUI-Manager requirements failed: {msg}")
    return m


def _cm_cli_install_names(
    cm_cli: Path,
    comfy_dir: Path,
    env: Dict[str, str],
    try_names: List[str],
) -> Tuple[bool, Optional[str], str]:
    """
    Try ``cm-cli install`` with each name (remote mode, then default mode).
    Returns (ok, name_used, last_combined_output).
    """
    last_out = ""
    for name in try_names:
        for mode_args in (["--mode", "remote"], []):
            cmd1 = [sys.executable, str(cm_cli), "install", name, *mode_args]
            r1 = subprocess.run(
                cmd1, cwd=str(comfy_dir), env=env, capture_output=True, text=True
            )
            last_out = (r1.stderr or "") + "\n" + (r1.stdout or "")
            if r1.returncode == 0:
                return True, name, last_out
    return False, None, last_out


def install_custom_nodes_hybrid(
    comfy_dir: Path,
    urls: Set[str],
    *,
    allow_pip_fail: bool,
    use_cm_cli: bool,
    pack_try_names: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Prefer ComfyUI-Manager ``cm-cli.py install`` (same as UI-driven installs),
    then ``git clone`` any pack cm-cli could not install.

    Returns one result dict per pack (see ``_empty_pack_result`` fields).
    """
    custom_nodes = comfy_dir / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)

    git_urls = [normalize_git_url(u) for u in urls if is_probably_git_clone_url(u)]
    report: List[Dict[str, Any]] = []

    if not git_urls:
        print("No git URLs to install.")
        return report

    url_by_pack: Dict[str, str] = {}
    for url in git_urls:
        pn = repo_folder_name(url)
        if pn == "ComfyUI-Manager":
            continue
        url_by_pack[pn] = url

    packs = sorted(url_by_pack.keys())
    for pn in packs:
        rec = _empty_pack_result(pn, url_by_pack[pn])
        report.append(rec)

    if not packs:
        print("Only ComfyUI-Manager in URL set; nothing else to install.")
        return report

    def pack_record(pn: str) -> Dict[str, Any]:
        for r in report:
            if r["pack"] == pn:
                return r
        raise KeyError(pn)

    if not use_cm_cli:
        print("USE_COMFYUI_MANAGER_CLI=0 — using git clone only.")
        for pn in packs:
            url = url_by_pack[pn]
            rec = pack_record(pn)
            print(f"\n== git {url} -> {pn}")
            try:
                outcome, warns = install_one_repo(
                    url, custom_nodes / pn, allow_pip_fail=allow_pip_fail, update=False
                )
                rec["method"] = "git_only"
                rec["status"] = "ok" if not warns else "warning"
                rec["warnings"] = warns
            except (subprocess.CalledProcessError, RuntimeError) as e:
                rec["method"] = "git_only"
                rec["status"] = "failed"
                rec["errors"].append(str(e))
                raise
        return report

    ensure_comfyui_manager(comfy_dir, allow_pip_fail=allow_pip_fail)
    cm_cli = custom_nodes / "ComfyUI-Manager" / "cm-cli.py"
    env = os.environ.copy()
    env["COMFYUI_PATH"] = str(comfy_dir.resolve())

    batch = max(1, CM_CLI_BATCH_SIZE)
    i = 0
    while i < len(packs):
        chunk = packs[i : i + batch]
        cmd = [
            sys.executable,
            str(cm_cli),
            "install",
            *chunk,
            "--mode",
            "remote",
        ]
        print(
            f"\n== cm-cli.py install ({len(chunk)}): {', '.join(chunk[:4])}"
            f"{' ...' if len(chunk) > 4 else ''}"
        )
        r = subprocess.run(cmd, cwd=str(comfy_dir), env=env, capture_output=True, text=True)
        if r.returncode == 0:
            if r.stdout:
                print(r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout)
            for pn in chunk:
                rec = pack_record(pn)
                rec["method"] = "cm_cli_batch"
                rec["status"] = "ok"
                rec["cm_cli_name_used"] = pn
            i += batch
            continue

        print(
            f"cm-cli batch failed (exit {r.returncode}); retrying per-pack. stderr:\n"
            f"{r.stderr or r.stdout}",
            file=sys.stderr,
        )
        for pn in chunk:
            rec = pack_record(pn)
            try_names = pack_try_names.get(pn) or [pn]
            ok, used, out_tail = _cm_cli_install_names(cm_cli, comfy_dir, env, try_names)
            if ok:
                print(f"  cm-cli OK: {pn} (via {used!r})")
                rec["method"] = "cm_cli"
                rec["status"] = "ok"
                rec["cm_cli_name_used"] = used
                snippet = (out_tail or "")[-2000:]
                if "WARNING" in snippet.upper() or "ERROR" in snippet.upper():
                    rec["warnings"].append(f"cm-cli output tail: {snippet}")
                    rec["status"] = "warning"
            else:
                url = url_by_pack[pn]
                print(f"  cm-cli FAILED {pn} -> git clone {url}", file=sys.stderr)
                rec["errors"].append((out_tail or "")[-2500:])
                try:
                    outcome, warns = install_one_repo(
                        url,
                        custom_nodes / pn,
                        allow_pip_fail=allow_pip_fail,
                        update=False,
                    )
                    rec["method"] = "git_fallback"
                    rec["cm_cli_name_used"] = None
                    rec["warnings"].extend(warns)
                    rec["status"] = "ok" if not warns and outcome == "cloned" else "warning"
                    if outcome == "skipped_existing" and not warns:
                        rec["status"] = "ok"
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    rec["method"] = "git_fallback"
                    rec["status"] = "failed"
                    rec["errors"].append(str(e))
                    raise
        i += batch

    if os.environ.get("CM_CLI_RESTORE_DEPS", "").strip() == "1":
        print("\n== cm-cli.py restore-dependencies")
        subprocess.run(
            [sys.executable, str(cm_cli), "restore-dependencies"],
            cwd=str(comfy_dir),
            env=env,
            check=False,
        )

    return report


def is_official_comfyui_repo(url: str) -> bool:
    """True only for the main ComfyUI app repo (not forks named .../ComfyUI-foo)."""
    u = normalize_git_url(url).lower().removesuffix(".git")
    return u == "https://github.com/comfyanonymous/comfyui"


def pick_custom_clone_url(
    node_type: str,
    candidates: List[str],
    tie_breaks: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], bool]:
    """
    Pick one git URL to clone for this node type, or None if ComfyUI core is enough.

    Returns (url_or_none, strict_conflict) — strict_conflict True only when
    INSTALL_STRICT_NODE_CONFLICTS=1 and we refuse to guess.
    """
    strict = os.environ.get("INSTALL_STRICT_NODE_CONFLICTS", "").strip() == "1"
    uniq = sorted(set(candidates))

    if len(uniq) == 1:
        u = uniq[0]
        return (None if is_official_comfyui_repo(u) else u, False)

    if any(is_official_comfyui_repo(c) for c in uniq):
        non_core = [c for c in uniq if not is_official_comfyui_repo(c)]
        if not non_core:
            return (None, False)
        uniq = non_core

    if len(uniq) == 1:
        return (uniq[0], False)

    needle = PREFERRED_REPO_SUBSTRING.get(node_type)
    if needle:
        nl = needle.lower()
        for c in uniq:
            if nl in c.lower():
                return (c, False)

    if node_type.startswith("WanVideo"):
        for c in uniq:
            cl = c.lower()
            if "kijai" in cl and "wanvideo" in cl.replace("-", ""):
                return (c, False)
            if "kijai" in cl and "wan" in cl:
                return (c, False)

    if node_type.startswith("VHS_"):
        for c in uniq:
            if "kosinkadink" in c.lower():
                return (c, False)

    if strict:
        return (None, True)

    pick = sorted(uniq)[0]
    msg = (
        f"  tie-break: {node_type!r} -> {pick} "
        f"(set resolve_conflicts in overrides.json or INSTALL_STRICT_NODE_CONFLICTS=1 to enforce)"
    )
    print(msg, file=sys.stderr)
    if tie_breaks is not None:
        tie_breaks.append(
            {
                "kind": "tie_break",
                "node_type": node_type,
                "chosen_url": pick,
                "candidates": ",".join(sorted(uniq)),
            }
        )
    return (pick, False)


def resolve_urls_for_types(
    all_types: Set[str],
    rev_index: Dict[str, Set[str]],
    overrides_by_node: Dict[str, str],
    tie_breaks: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]]]:
    """
    Returns:
      url_by_node: node_type -> chosen url (custom_nodes clone only)
      unique_urls: sorted unique clone URLs
      conflicts: node -> candidate urls (only when strict mode cannot resolve)
    """
    url_by_node: Dict[str, str] = {}
    conflicts: Dict[str, List[str]] = {}

    for nt in sorted(all_types):
        if nt in overrides_by_node:
            url_by_node[nt] = normalize_git_url(overrides_by_node[nt])
            continue

        candidates = sorted(rev_index.get(nt, ()))
        if not candidates:
            continue

        picked, bad = pick_custom_clone_url(nt, candidates, tie_breaks)
        if bad:
            conflicts[nt] = candidates
        elif picked:
            url_by_node[nt] = normalize_git_url(picked)

    unique_urls = sorted(set(url_by_node.values()))
    return url_by_node, unique_urls, conflicts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Install custom nodes for ComfyUI workflows")
    parser.add_argument("workflows", nargs="*", help="Workflow JSON files")
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        help="Directory of *.json workflows (used if no files passed)",
    )
    parser.add_argument(
        "--comfyui-dir",
        type=Path,
        default=None,
        help="ComfyUI root (else COMFYUI_DIR env or ../comfyui from script)",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="overrides.json path (else OVERRIDES_PATH env)",
    )
    parser.add_argument(
        "--manager-base",
        default=DEFAULT_MANAGER_BASE,
        help="Raw GitHub base for ComfyUI-Manager files",
    )
    parser.add_argument(
        "--allow-pip-fail",
        action="store_true",
        default=os.environ.get("ALLOW_NODE_PIP_FAIL", "").strip() == "1",
        help="Continue if pip install fails",
    )
    parser.add_argument(
        "--no-cm-cli",
        action="store_true",
        help="Git-clone only; do not use ComfyUI-Manager cm-cli.py",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write install report JSON (default: <project>/node_install_summary.json or NODE_INSTALL_SUMMARY_PATH)",
    )
    parser.add_argument(
        "--no-install-summary",
        action="store_true",
        help="Do not write node_install_summary.json",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    comfy_dir = args.comfyui_dir
    if comfy_dir is None:
        cd = os.environ.get("COMFYUI_DIR")
        comfy_dir = Path(cd) if cd else project_root / "comfyui"
    comfy_dir = comfy_dir.resolve()
    custom_nodes = comfy_dir / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)

    over_path = args.overrides
    if over_path is None:
        e = os.environ.get("OVERRIDES_PATH")
        over_path = Path(e) if e else project_root / "overrides.json"
    overrides_nodes, extra_repos = load_overrides(over_path if over_path.is_file() else None)

    wf_paths: List[Path] = []
    for w in args.workflows:
        wf_paths.append(Path(w).resolve())
    if not wf_paths and args.workflow_dir:
        wf_paths = collect_workflow_paths(args.workflow_dir.resolve())
    if not wf_paths:
        default_dir = project_root / "workflow"
        wf_paths = collect_workflow_paths(default_dir)

    if not wf_paths:
        print("No workflow JSON files found.", file=sys.stderr)
        return 2

    all_types: Set[str] = set()
    registry_rows: List[Dict[str, Any]] = []
    for p in wf_paths:
        if not p.is_file():
            print(f"Missing workflow: {p}", file=sys.stderr)
            return 2
        all_types |= extract_class_types_from_file(p)
        registry_rows.extend(extract_node_registry_rows(p))

    print(f"Scanned {len(wf_paths)} workflow(s), {len(all_types)} unique node types")

    map_url = f"{args.manager_base.rstrip('/')}/{EXTENSION_MAP_FILE}"
    extension_map = _fetch_json(map_url)
    if not isinstance(extension_map, dict):
        print("Invalid extension map", file=sys.stderr)
        return 3

    rev = build_reverse_index(extension_map)

    if os.environ.get("SKIP_UNLISTED_AUTOMATION", "").strip() == "1":
        merged_overrides = dict(overrides_nodes)
        unlisted_applied: List[Dict[str, Any]] = []
        unlisted_skipped_core: List[Dict[str, Any]] = []
    else:
        merged_overrides, unlisted_applied, unlisted_skipped_core = (
            augment_overrides_for_unlisted_node_types(
                all_types, rev, dict(overrides_nodes), project_root
            )
        )

    tie_breaks: List[Dict[str, str]] = []
    url_by_node, chosen_urls, conflicts = resolve_urls_for_types(
        all_types, rev, merged_overrides, tie_breaks
    )

    if conflicts:
        print(
            "\n*** UNRESOLVED NODE NAME CONFLICTS (add to overrides.json "
            "or unset INSTALL_STRICT_NODE_CONFLICTS) ***",
            file=sys.stderr,
        )
        for nt, urls in sorted(conflicts.items()):
            print(f"  {nt}:", file=sys.stderr)
            for u in urls:
                print(f"    - {u}", file=sys.stderr)
        return 4

    not_in_map = sorted(t for t in all_types if t not in rev)

    extra_url_set: Set[str] = set()
    for u in extra_repos:
        u = normalize_git_url(u)
        if u:
            extra_url_set.add(u)

    for u in chosen_urls:
        extra_url_set.add(normalize_git_url(u))

    non_git_skipped: List[str] = []
    for u in sorted(extra_url_set):
        if not is_probably_git_clone_url(u):
            print(f"SKIP non-git URL (add manual steps): {u}", file=sys.stderr)
            non_git_skipped.append(u)

    pack_try_names = build_cm_cli_try_names_per_pack(
        url_by_node, extra_url_set, registry_rows
    )

    use_cm = not args.no_cm_cli and os.environ.get("USE_COMFYUI_MANAGER_CLI", "1").strip() != "0"
    print(
        f"Installing {len(extra_url_set)} unique git URLs into {custom_nodes} "
        f"({'hybrid cm-cli + git fallback' if use_cm else 'git only'})"
    )

    summary_out: Optional[Path] = args.summary_out
    if summary_out is None:
        ev = os.environ.get("NODE_INSTALL_SUMMARY_PATH", "").strip()
        summary_out = Path(ev) if ev else project_root / "node_install_summary.json"
    skip_summary = args.no_install_summary or (
        os.environ.get("SKIP_NODE_INSTALL_SUMMARY", "").strip() == "1"
    )

    pack_reports: List[Dict[str, Any]] = []
    fatal: Optional[str] = None
    try:
        pack_reports = install_custom_nodes_hybrid(
            comfy_dir,
            extra_url_set,
            allow_pip_fail=args.allow_pip_fail,
            use_cm_cli=use_cm,
            pack_try_names=pack_try_names,
        )
    except subprocess.CalledProcessError as e:
        fatal = e.stderr or e.stdout or str(e)
        print(fatal, file=sys.stderr)
    except RuntimeError as e:
        fatal = str(e)
        print(fatal, file=sys.stderr)

    summary_doc: Dict[str, Any] = {
        "workflows": [str(p) for p in wf_paths],
        "comfyui_dir": str(comfy_dir),
        "unique_node_types": len(all_types),
        "workflow_registry_rows": len(registry_rows),
        "note": (
            "ComfyUI-Manager extension-node-map keys are node class names (workflow 'type'), "
            "not graph numeric 'id'. When the UI saves properties.cnr_id / properties.aux_id, "
            "those are preferred cm-cli install names (see cm_cli_try_names_per_pack). "
            "Types missing from the map may be auto-mapped via unlisted_node_hints.json and "
            "built-in rules (rgthree suffix, MMAudio); see unlisted_auto_resolved."
        ),
        "unlisted_auto_resolved": unlisted_applied,
        "unlisted_skipped_core_or_uuid": unlisted_skipped_core,
        "tie_breaks": tie_breaks,
        "node_types_not_in_extension_map": not_in_map,
        "non_git_urls_skipped": non_git_skipped,
        "cm_cli_try_names_per_pack": {k: v for k, v in sorted(pack_try_names.items())},
        "packs": pack_reports,
        "counts": {
            "ok": sum(1 for x in pack_reports if x.get("status") == "ok"),
            "warning": sum(1 for x in pack_reports if x.get("status") == "warning"),
            "failed": sum(1 for x in pack_reports if x.get("status") == "failed"),
            "unknown": sum(1 for x in pack_reports if x.get("status") == "unknown"),
        },
    }
    if fatal:
        summary_doc["fatal_error"] = fatal[:8000]

    if not skip_summary and summary_out is not None:
        _write_json(summary_out, summary_doc)
        print(f"\nWrote install summary: {summary_out}")

    c = summary_doc["counts"]
    print(
        f"\nInstall summary: {c['ok']} ok, {c['warning']} warning(s), "
        f"{c['failed']} failed, {c['unknown']} unknown"
    )
    if tie_breaks:
        print(f"  ({len(tie_breaks)} extension-map tie-break(s) — see summary JSON)")
    if fatal:
        return 5

    if not_in_map:
        print("\nNode types not listed in extension-node-map (may be ComfyUI core or unknown):")
        for t in not_in_map:
            print(f"  - {t}")
        print(
            "If ComfyUI startup validation reports these as missing, add them to overrides.json missing_nodes."
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
