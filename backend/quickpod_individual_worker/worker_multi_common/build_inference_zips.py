#!/usr/bin/env python3
"""
Build R2-uploadable inference-code zip archives for every omni model.

Usage:
    python build_inference_zips.py [--out-dir DIR] [--no-manifest-update] [--reset-archive-urls]

Each zip contains Python inference code from the **embedded multi-worker tree**:

``quickpod_individual_worker/<multi_worker_dir>/src/inference/<inference_primary_dir>/src/``

(recursive ``.py`` files, same layout Docker ``COPY`` uses), excluding infrastructure
filenames at that root and thin ``_omni_entry.py`` wrappers where needed.

Output:
    <out_dir>/<model_key>.zip         for each model (upload these to R2)
    omni_manifest.json                updated in-place (inference_plugins metadata;
                                      existing real R2 archive_url values are preserved by default)
    inference_zips/manifest_plugins_snippet.json — copy of plugin list (placeholder URLs)

Archive URLs use the bucket root: https://<bucket>.r2.dev/<model>.zip (no inference_zips/ prefix).

**Shared zip (LTX-2.3):** one archive ``ltx2_3_multi.zip`` backs four ``inference_plugins`` rows
(different ``model`` registry keys and ``callable_name`` values). Each model still installs under
its own ``runtime_plugins/<model>/`` directory; omni duplicates the same bytes on disk unless the
runtime is optimized later.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from typing import Any, Dict, List, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QPW_ROOT = os.path.dirname(SCRIPT_DIR)  # quickpod_individual_worker

INFRA_FILES = frozenset({
    "app.py", "prep_model.py", "generate_dockerfile.py", "merge.py",
    "huggingface_app.py", "requirements.txt",
})

# Prune vendored / obsolete trees from zips (WAN may carry a local ``sdnq`` copy; runtime uses pip).
ZIP_EXCLUDE_DIR_NAMES = frozenset({"_archive", "__pycache__", "sdnq"})

R2_BASE_URL_PLACEHOLDER = "https://<YOUR_R2_PUBLIC_BUCKET>.r2.dev"

# MODELS[].model ids should match omni_manifest inference_plugins / download_specs.
# ``multi_worker_dir`` / ``inference_primary_dir`` must match omni_prep_bake.MULTI_PRIMARY_WORKER_DIR.
# flashvsr_video_upscaler is a dedicated QuickPod template (FlashVSR-Pro + subprocess); not bundled here.

# ── wrapper templates ────────────────────────────────────────────────────
# WAN I2V no longer uses a wrapper: worker_wan_22_fp8_lightx2v_i2v/src/inference.py
# run_inference_wan_22_i2v now accepts URL / path / PIL.Image for init_image & last_image
# directly (pulls from params if kwargs are None), so the plugin path uses inference.py too.

WRAPPER_HUNYUAN_I2V = '''\
"""Omni runtime entry — hunyan15_480p_i2v_step_distilled (resolves image_input)."""
from typing import Any, Dict


def run_inference(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from hunyan15_video_inference import run_inference_hunyan15_i2v
    p = dict(params or {})
    image_input = (
        p.pop("image_input", None)
        or p.pop("image", None)
        or p.pop("init_image", None)
    )
    if not image_input:
        raise ValueError("image_input/init_image is required for hunyuan i2v")
    return run_inference_hunyan15_i2v(job_id, prompt, image_input, p)
'''

WRAPPER_BASIC_UTILS = '''\
"""Omni runtime entry — basic_utils (bg_remove / lama_inpaint dispatch)."""
from typing import Any, Dict


def run_inference(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    mode = str((params or {}).get("mode") or "bg_remove").strip().lower()
    if mode == "lama_inpaint":
        from simple_lama_inpaint import run_inference_simple_lama_inpaint
        return run_inference_simple_lama_inpaint(job_id, prompt, params)
    from bg_remover import run_inference_rmbg
    return run_inference_rmbg(job_id, prompt, params)
'''

WRAPPERS: Dict[str, str] = {
    "hunyuan_i2v": WRAPPER_HUNYUAN_I2V,
    "basic_utils": WRAPPER_BASIC_UTILS,
}

# For _omni_entry.py wrappers, validate the real inference module on disk.
WRAPPER_BACKING: Dict[str, tuple[str, str]] = {
    "hunyuan_i2v": ("hunyan15_video_inference.py", "run_inference_hunyan15_i2v"),
    "basic_utils": ("bg_remover.py", "run_inference_rmbg"),
}

# ── model definitions ────────────────────────────────────────────────────

MODELS: List[Dict[str, Any]] = [
    {
        "model": "qwen_image_edit_2511_rapid_aio",
        "multi_worker_dir": "worker_qwen_image_edit_2511_rapid_aio_multi",
        "inference_primary_dir": "worker_qwen_image_edit_2511_rapid_aio",
        "entry_relpath": "qwen_image_edit_2511_rapid_aio_inference.py",
        "callable_name": "run_inference_qwen_image_edit_2511_rapid_aio",
        "prewarm_fn": "get_pipeline",
        "unload_fn": "unload_models",
    },
    {
        "model": "flux_klein",
        "multi_worker_dir": "worker_flux_klein_9b_multi",
        "inference_primary_dir": "worker_flux_klein_9b",
        "entry_relpath": "flux_klein_inference.py",
        "callable_name": "run_inference_flux_klein",
        "prewarm_fn": "get_pipeline",
        "unload_fn": "unload_models",
    },
    {
        "model": "z_image_turbo",
        "multi_worker_dir": "worker_z_image_turbo_multi",
        "inference_primary_dir": "worker_z_image_turbo",
        "entry_relpath": "z_image_turbo.py",
        "callable_name": "run_inference_z_image_turbo",
        "prewarm_fn": "get_pipeline",
        "unload_fn": "unload_models",
    },
    {
        "model": "sdxl_t2i",
        "multi_worker_dir": "worker_stable_diffusion_xl_multi",
        "inference_primary_dir": "worker_stable_diffusion_xl",
        "entry_relpath": "sdxl_T2I_inference.py",
        "callable_name": "run_inference_sdxl_t2i",
        "prewarm_fn": "get_pipeline",
        "unload_fn": "unload_models",
    },
    {
        "model": "diffusers_outpaint",
        "multi_worker_dir": "worker_stable_diffusion_xl_outpaint_multi",
        "inference_primary_dir": "worker_stable_diffusion_xl_outpaint",
        "entry_relpath": "diffuser_outpaint_inference.py",
        "callable_name": "run_inference_sdxl_outpaint",
        "prewarm_fn": "get_sdxl_outpaint_pipeline",
        "unload_fn": "unload_sdxl_outpaint",
    },
    {
        "model": "aurasrv2",
        "multi_worker_dir": "worker_image_upscaler_aurasrv2_multi",
        "inference_primary_dir": "worker_image_upscaler_aurasrv2",
        "entry_relpath": "aurasrv2_upscaler.py",
        "callable_name": "run_inference_aurasr_upscale",
        "unload_fn": "unload_aura_sr",
        "no_prompt": True,
    },
    {
        "model": "swin2sr",
        "multi_worker_dir": "worker_image_upscaler_swin2sr_multi",
        "inference_primary_dir": "worker_image_upscaler_swin2sr",
        "entry_relpath": "swin2sr_upscaler.py",
        "callable_name": "run_inference_swin2sr",
        "unload_fn": "unload_swin2sr_models",
        "no_prompt": True,
    },
    {
        "model": "face_enhancer",
        "multi_worker_dir": "worker_face_enhancer_multi",
        "inference_primary_dir": "worker_face_enhancer",
        "entry_relpath": "inference_face_enhancer.py",
        "callable_name": "run_inference_face_enhancer",
        "unload_fn": "unload_models",
        "no_prompt": True,
    },
    {
        "model": "video_upscaler",
        "multi_worker_dir": "worker_video_upscalers_multi",
        "inference_primary_dir": "worker_video_upscalers",
        "entry_relpath": "inference.py",
        "callable_name": "run_inference_video_upscaler",
        "unload_fn": "unload_models",
    },
    {
        "model": "basic_utils",
        "multi_worker_dir": "worker_basic_utils_multi",
        "inference_primary_dir": "worker_basic_utils",
        "entry_relpath": "_omni_entry.py",
        "callable_name": "run_inference",
        "wrapper": "basic_utils",
    },
    {
        "model": "wan_22_fp8_lightx2v_t2v",
        "multi_worker_dir": "worker_wan_22_fp8_lightx2v_t2v_multi",
        "inference_primary_dir": "worker_wan_22_fp8_lightx2v_t2v",
        "entry_relpath": "inference.py",
        "callable_name": "run_inference_wan_22_t2v",
    },
    {
        "model": "wan_22_fp8_lightx2v_i2v",
        "multi_worker_dir": "worker_wan_22_fp8_lightx2v_i2v_multi",
        "inference_primary_dir": "worker_wan_22_fp8_lightx2v_i2v",
        "entry_relpath": "inference.py",
        "callable_name": "run_inference_wan_22_i2v",
    },
    {
        "model": "hunyan15_480p_t2v_distilled",
        "multi_worker_dir": "worker_hunyan15_480p_t2V_distilled_multi",
        "inference_primary_dir": "worker_hunyan15_480p_t2V_distilled",
        "entry_relpath": "hunyan15_video_inference.py",
        "callable_name": "run_inference_hunyan15_t2v",
    },
    {
        "model": "hunyan15_480p_i2v_step_distilled",
        "multi_worker_dir": "worker_hunyan15_480p_i2V_step_distilled_multi",
        "inference_primary_dir": "worker_hunyan15_480p_i2V_step_distilled",
        "entry_relpath": "_omni_entry.py",
        "callable_name": "run_inference",
        "wrapper": "hunyuan_i2v",
    },
    {
        "model": "glm_ocr",
        "multi_worker_dir": "worker_glm_ocr_multi",
        "inference_primary_dir": "worker_glm_ocr",
        "entry_relpath": "glm_ocr_inference.py",
        "callable_name": "run_inference_glm_ocr",
        "prewarm_fn": "preload_pipeline",
        "unload_fn": "unload_pipeline",
    },
    {
        "model": "ltx2_3_t2v",
        "multi_worker_dir": "worker_ltx2_3_multi",
        "inference_primary_dir": "worker_ltx2_3",
        "entry_relpath": "ltx2_3_inference.py",
        "callable_name": "run_inference_ltx2_3_t2v",
        "unload_fn": "unload_pipelines",
        "inference_zip_stem": "ltx2_3_multi",
        "inference_zip_variants": [
            {
                "model": "ltx2_3_i2v_distilled",
                "callable_name": "run_inference_ltx2_3_i2v_distilled",
            },
            {
                "model": "ltx2_3_flf2v",
                "callable_name": "run_inference_ltx2_3_flf2v",
            },
            {
                "model": "ltx2_3_iv2v_distilled",
                "callable_name": "run_inference_ltx2_3_iv2v_distilled",
            },
        ],
    },
]


def _is_placeholder_archive_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return not u or "<your_r2" in u or "<your_" in u or "your_r2_public_bucket" in u


def _inference_src_dir(spec: Dict[str, Any]) -> str:
    multi = str(spec.get("multi_worker_dir") or "").strip()
    primary = str(spec.get("inference_primary_dir") or "").strip()
    if not multi or not primary:
        raise ValueError(
            f"{spec.get('model')}: MODELS entry needs multi_worker_dir and inference_primary_dir"
        )
    return os.path.join(QPW_ROOT, multi, "src", "inference", primary, "src")


def _validate_entry_py(
    model: str, src_dir: str, entry_relpath: str, callable_name: str
) -> None:
    """Fail fast if entry file or top-level callable is missing."""
    path = os.path.join(src_dir, entry_relpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{model}: entry_relpath missing: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError as e:
        raise RuntimeError(f"{model}: cannot read {path}: {e}") from e
    needle = f"def {callable_name}("
    if needle not in src:
        raise RuntimeError(
            f"{model}: {entry_relpath} has no `{needle}` — fix MODELS or inference module"
        )


def _collect_inference_zip_members(src_dir: str) -> List[Tuple[str, str]]:
    """Return (zip arcname posix, absolute path) for every inference .py under ``src_dir``."""
    root = os.path.abspath(src_dir)
    if not os.path.isdir(root):
        return []
    out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ZIP_EXCLUDE_DIR_NAMES]
        rel_dir = os.path.relpath(dirpath, root)
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            if rel_dir in (".", "") and fn in INFRA_FILES:
                continue
            full = os.path.join(dirpath, fn)
            if rel_dir in (".", ""):
                arc = fn
            else:
                arc = os.path.join(rel_dir, fn).replace("\\", "/")
            out.append((arc, full))
    out.sort(key=lambda x: x[0])
    return out


def _plugin_entry_from_spec(
    spec: Dict[str, Any],
    *,
    model: str,
    callable_name: str,
    zip_name: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "model": model,
        "archive_url": f"{R2_BASE_URL_PLACEHOLDER}/{zip_name}",
        "entry_relpath": spec["entry_relpath"],
        "callable_name": callable_name,
    }
    if spec.get("prewarm_fn"):
        entry["prewarm_fn"] = spec["prewarm_fn"]
    if spec.get("unload_fn"):
        entry["unload_fn"] = spec["unload_fn"]
    if spec.get("no_prompt"):
        entry["no_prompt"] = True
    return entry


def build_zip(spec: Dict[str, Any], out_dir: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    model = spec["model"]
    try:
        src_dir = _inference_src_dir(spec)
    except ValueError as e:
        print(f"  ERROR {model}: {e}")
        return {}

    if not os.path.isdir(src_dir):
        print(f"  SKIP {model}: {src_dir} not found")
        return {}

    wrapper_key = spec.get("wrapper")
    uses_wrapper = bool(wrapper_key and wrapper_key in WRAPPERS)
    extra_variants = spec.get("inference_zip_variants") or []
    zip_stem = str(spec.get("inference_zip_stem") or "").strip()

    try:
        if uses_wrapper:
            backing = WRAPPER_BACKING.get(str(wrapper_key))
            if backing:
                rel, cname = backing
                _validate_entry_py(model, src_dir, rel, cname)
        else:
            variants_to_check: List[Dict[str, str]] = [
                {"model": model, "callable_name": spec["callable_name"]}
            ]
            for row in extra_variants:
                if not isinstance(row, dict):
                    continue
                m = str(row.get("model") or "").strip()
                c = str(row.get("callable_name") or "").strip()
                if m and c:
                    variants_to_check.append({"model": m, "callable_name": c})
            for v in variants_to_check:
                _validate_entry_py(
                    v["model"], src_dir, spec["entry_relpath"], v["callable_name"]
                )
    except (OSError, RuntimeError) as e:
        print(f"  ERROR {model}: {e}")
        return {}

    if extra_variants and zip_stem:
        zip_name = f"{zip_stem}.zip"
    else:
        zip_name = f"{model}.zip"
    zip_path = os.path.join(out_dir, zip_name)

    members = _collect_inference_zip_members(src_dir)
    if not members and not uses_wrapper:
        print(f"  SKIP {model}: no .py files under {src_dir}")
        return {}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, full in members:
            zf.write(full, arc)
        if uses_wrapper:
            zf.writestr("_omni_entry.py", WRAPPERS[wrapper_key])

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  {zip_name:55s}  {size_kb:7.1f} KB")

    if extra_variants and zip_stem:
        out_list: List[Dict[str, Any]] = []
        for v in [{"model": model, "callable_name": spec["callable_name"]}] + [
            {"model": str(x.get("model") or "").strip(), "callable_name": str(x.get("callable_name") or "").strip()}
            for x in extra_variants
            if isinstance(x, dict) and str(x.get("model") or "").strip() and str(x.get("callable_name") or "").strip()
        ]:
            out_list.append(_plugin_entry_from_spec(spec, model=v["model"], callable_name=v["callable_name"], zip_name=zip_name))
        return out_list

    return _plugin_entry_from_spec(
        spec, model=model, callable_name=spec["callable_name"], zip_name=zip_name
    )


def _merge_inference_plugins(
    built: List[Dict[str, Any]],
    previous: List[Dict[str, Any]],
    *,
    reset_archive_urls: bool,
) -> List[Dict[str, Any]]:
    """
    Use freshly built plugin metadata but keep production archive_url when the new
    value would be a placeholder (avoids wiping real R2 URLs on every zip build).
    """
    prev_by_model: Dict[str, Dict[str, Any]] = {}
    for p in previous:
        if not isinstance(p, dict):
            continue
        mk = str(p.get("model") or "").strip().lower()
        if mk:
            prev_by_model[mk] = p

    out: List[Dict[str, Any]] = []
    for entry in built:
        e = dict(entry)
        mk = str(e.get("model") or "").strip().lower()
        old = prev_by_model.get(mk) if mk else None
        new_url = str(e.get("archive_url") or "")
        if (
            not reset_archive_urls
            and old
            and isinstance(old.get("archive_url"), str)
            and not _is_placeholder_archive_url(old["archive_url"])
            and _is_placeholder_archive_url(new_url)
        ):
            e["archive_url"] = old["archive_url"]
        out.append(e)
    return out


def _update_manifest(
    plugins: List[Dict[str, Any]],
    *,
    reset_archive_urls: bool,
) -> None:
    """Update omni_manifest.json inference_plugins in-place."""
    manifest_path = os.path.join(SCRIPT_DIR, "omni_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  ⚠ {manifest_path} not found — skipping manifest update")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    prev = manifest.get("inference_plugins") or []
    if not isinstance(prev, list):
        prev = []
    manifest["inference_plugins"] = _merge_inference_plugins(
        plugins, prev, reset_archive_urls=reset_archive_urls
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated: {manifest_path}")


def _write_plugins_snippet(plugins: List[Dict[str, Any]]) -> None:
    """Keep inference_zips/manifest_plugins_snippet.json aligned (CI / docs)."""
    snippet_path = os.path.join(SCRIPT_DIR, "inference_zips", "manifest_plugins_snippet.json")
    os.makedirs(os.path.dirname(snippet_path), exist_ok=True)
    with open(snippet_path, "w", encoding="utf-8") as f:
        json.dump(plugins, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote: {snippet_path}")


def main():
    parser = argparse.ArgumentParser(description="Build inference code zips for R2")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(SCRIPT_DIR, "inference_zips"),
        help="Output directory for zip files (default: ./inference_zips)",
    )
    parser.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="Do not modify omni_manifest.json (only build .zip files)",
    )
    parser.add_argument(
        "--reset-archive-urls",
        action="store_true",
        help="Force placeholder archive_url for every model (do not preserve existing R2 URLs)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}\n")

    plugins: List[Dict[str, Any]] = []
    zip_count = 0
    for spec in MODELS:
        entry = build_zip(spec, out_dir)
        if isinstance(entry, list):
            plugins.extend(entry)
            if entry:
                zip_count += 1
        elif entry:
            plugins.append(entry)
            zip_count += 1

    print(f"\n{zip_count} zip file(s) created, {len(plugins)} inference_plugin row(s).")

    if not args.no_manifest_update:
        _update_manifest(plugins, reset_archive_urls=args.reset_archive_urls)
    else:
        print("Skipped omni_manifest.json (--no-manifest-update).")

    _write_plugins_snippet(plugins)

    print(f"\nUpload the .zip files from {out_dir} to your R2 bucket root (same paths as URLs),")
    if args.reset_archive_urls or args.no_manifest_update:
        print(
            f"If URLs are still placeholders, set your public base in omni_manifest.json "
            f"(not {R2_BASE_URL_PLACEHOLDER})."
        )
    else:
        print(
            "Existing production archive_url values in omni_manifest.json were preserved "
            "(use --reset-archive-urls to replace all with placeholders)."
        )


if __name__ == "__main__":
    main()
