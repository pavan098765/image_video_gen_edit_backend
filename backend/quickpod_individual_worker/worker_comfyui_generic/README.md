# worker_comfyui_generic

Headless ComfyUI worker with:

- **Automatic custom node install** from `workflow/*.json` using ComfyUI-Manager’s [`extension-node-map.json`](https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json) (class type → git URL), then **hybrid install**: `ComfyUI-Manager`’s `cm-cli.py install` (batched) with **git clone fallback** when a pack name does not match the registry.
- **Workflow-driven model setup** via **`model_config.json`** (preferred): prep refreshes this file with every model path found in workflows; you set exact **`hf_repo`** / **`hf_path`** per path. **`model_registry.json`** is optional fallback for shared catalogs when `model_config` has no complete entry.
- **Runtime downloads**: before ComfyUI starts, the worker re-runs the same resolver (config → registry) for any file still missing under `comfyui/models/` (disable with `SKIP_RUNTIME_MODEL_DOWNLOAD=1`). Paths that share the same `hf_repo` + `hf_path` (different widget filenames pointing at one HF file) download once, then use **hardlinks** (fallback symlink / copy) so you do not store duplicate gigabytes.
- **Strict installs**: Docker build runs `pip install -r` per custom node without swallowing failures.
- **Startup validation**: After ComfyUI is up, all workflows are checked against `GET /object_info`; missing types block readiness.

## Quick start

1. Copy `overrides.json.example` → `overrides.json` if you need conflict resolution or private repos.
2. Optional: copy `model_registry.example.json` → `model_registry.json` for fallback HF URLs. After first `prep_model.py`, **`model_config.json`** is created/merged from workflows (and pre-filled from the registry where possible).
3. Drop workflow JSON files under `workflow/`.
4. From this directory:

```bash
pip install -r requirements.txt
python src/prep_model.py
docker build -t comfyui-generic-worker .
```

`prep_model.py` clones ComfyUI, installs custom nodes, **syncs `model_config.json`**, **downloads models** into `comfyui/models/`, then writes the `Dockerfile`.

## `model_config.json` (preferred)

- Written/updated every prep from **`workflow/*.json`** (same paths the scanner uses after visual→API conversion).
- Each **`files`** key is a path under `comfyui/models/` (e.g. `loras/WAN/CausVid/foo.safetensors`).
- Values: `{ "hf_repo", "hf_path", "revision"? , "local_relpath"? }` — same shape as the registry.
- **Lookup order** (prep + runtime): `model_config` first, then **`model_registry.json`** for that path.
- Edit this file when you know the exact HF repo/file; keep `model_registry` for reused bundles.

Refresh only the config (no download):

```bash
python src/download_workflow_models.py --sync-config-only
```

See `model_config.example.json`.

## Model registry (`model_registry.json`, fallback)

Optional shared catalog. Same JSON shape as before (`files`, `path_aliases`). Used when `model_config` has no complete `hf_repo`/`hf_path` for a workflow path.

```bash
python src/download_workflow_models.py --dry-run
python src/download_workflow_models.py --write-stub
```

Set **`HF_TOKEN`** for gated repos.

- **`SKIP_MODEL_DOWNLOAD=1`**: skip **prep-time** downloads only; `model_config.json` still syncs; the container can still fetch at startup unless **`SKIP_RUNTIME_MODEL_DOWNLOAD=1`**.

## Overrides (`overrides.json`)

Place next to this README. Used by `install_workflow_nodes.py` and `prep_model.py`.

- **`resolve_conflicts`**: Node class name → single git URL when you want to override auto-resolution.
- **`missing_nodes`**: Types not in `extension-node-map.json`.
- **`extra_repos`**: Extra git URLs to clone.

**Types missing from extension-node-map** (often rgthree `"(rgthree)"` titles, MMAudio, or subgraph UUID “types”): the installer merges **`unlisted_node_hints.json`** (copy from `unlisted_node_hints.example.json`) and small **built-in** rules: [`rgthree-comfy`](https://github.com/rgthree/rgthree-comfy.git) for the ` (rgthree)` name suffix and [kijai/ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio) for `MMAudio*` loaders. Core UI types like `Note` / `Reroute` / `PrimitiveNode` / `MarkdownNote` are skipped (ComfyUI built-in). UUID-shaped strings are skipped (subgraph noise). Anything else still needs an **`exact`** entry in `unlisted_node_hints.json` or **`missing_nodes`** in `overrides.json`.

**Custom node install** resolves the Manager’s duplicate node names by: (1) skipping clones when **`comfyanonymous/ComfyUI`** is among the candidates (core already provides the node); (2) built-in substring preferences for common Wan/KJ/VHS packs; (3) otherwise a deterministic tie-break with a stderr warning. Set **`INSTALL_STRICT_NODE_CONFLICTS=1`** to fail instead of tie-breaking (then use `resolve_conflicts`).

## Generic job API

See [API_REQUEST_FORMAT.md](API_REQUEST_FORMAT.md). Model key: **`comfyui_generic`**.

## Environment

| Variable | Purpose |
|----------|---------|
| `MANAGER_BASE_URL` | Manager raw base URL |
| `USE_COMFYUI_MANAGER_CLI` | `0` = git-clone only (skip `cm-cli.py`; default is hybrid) |
| `CM_CLI_BATCH_SIZE` | Packs per `cm-cli install` batch (default `12`) |
| `CM_CLI_RESTORE_DEPS` | `1` = run `cm-cli restore-dependencies` after installs |
| `NODE_INSTALL_SUMMARY_PATH` | Override path for `node_install_summary.json` (default: worker project root) |
| `SKIP_NODE_INSTALL_SUMMARY` | `1` = do not write the install report JSON |
| `UNLISTED_NODE_HINTS_PATH` | Path to JSON for types missing from Manager map (default: `unlisted_node_hints.json` in this folder) |
| `SKIP_UNLISTED_AUTOMATION` | `1` = disable built-in + hints file; use only `overrides.json` `missing_nodes` |
| `ALLOW_NODE_PIP_FAIL` | `1` = soft pip failures for local node install only |
| `SKIP_MODEL_DOWNLOAD` | `1` = skip HF downloads during **prep** (config still syncs) |
| `MODEL_DOWNLOAD_NO_HARDLINK` | `1` = skip hardlinks when materializing duplicate paths (symlink / copy only) |
| `SKIP_RUNTIME_MODEL_DOWNLOAD` | `1` = worker does not download missing weights at startup |
| `MODEL_CONFIG_PATH` | Alternate path to `model_config.json` |
| `MODEL_REGISTRY_PATH` | Alternate path to registry JSON |
| `SKIP_WORKFLOW_NODE_VALIDATION` | `1` on container = skip startup node checks (debug) |
