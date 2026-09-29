"""
Canonical Qwen-Image-Edit-2511 Rapid AIO LoRA catalog (matches single-worker prep list).

Used by omni_prep_bake (image bake) and omni_config (runtime HF supplements) so prep and
runtime always agree. Do not edit worker_qwen_image_edit_2511_rapid_aio (non-multi); keep
this list in sync with that worker's ADAPTER_SPECS when upstream changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Same adapter keys/repos/weights as worker_qwen_image_edit_2511_rapid_aio/src/prep_model.py
QWEN_2511_LORA_ADAPTERS: Dict[str, Dict[str, Any]] = {
    "Consistance": {
        "repo": "Pr0f3ssi0n4ln00b/QIE_2511_Consistency_Lora",
        "weights": "qe2511_consis_alpha_patched.safetensors",
    },
    "Semirealistic-photo-detailer": {
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
    },
    "AnyPose": {
        "repo": "lilylilith/AnyPose",
        "weights": [
            "2511-AnyPose-base-000006250.safetensors",
            "2511-AnyPose-helper-00006000.safetensors",
        ],
    },
    "Any2Real_2601": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601_A_final_patched.safetensors",
    },
    "Hyperrealistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
    },
    "Ultrarealistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
    },
    "BFS-Best-FaceSwap-merge": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors",
    },
    "F2P": {
        "repo": "DiffSynth-Studio/Qwen-Image-Edit-F2P",
        "weights": "edit_0928_lora_step40000.safetensors",
    },
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
    },
    "Light-Restoration": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
    },
    "Relight": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
    },
    "Edit-Skin": {
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
    },
    "Next-Scene": {
        "repo": "lovis93/next-scene-qwen-image-lora-2509",
        "weights": "next-scene_lora-v2-3000.safetensors",
    },
    "Flat-Log": {
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
    },
}


def build_qwen_lora_supplement_hub_files() -> List[Dict[str, Any]]:
    """Entries for ``supplement_hub_files`` on the qwen multi-snapshot download spec."""
    out: List[Dict[str, Any]] = []
    for adapter_name, cfg in QWEN_2511_LORA_ADAPTERS.items():
        repo = str(cfg.get("repo") or "").strip()
        wraw = cfg.get("weights")
        if not repo or not wraw:
            continue
        files = list(wraw) if isinstance(wraw, (list, tuple)) else [str(wraw)]
        folder = adapter_name.replace("/", "_")
        for fn in files:
            fn = str(fn).strip()
            if not fn:
                continue
            out.append(
                {
                    "repo_id": repo,
                    "filename": fn,
                    "target_relpath": f"{folder}/{fn}",
                }
            )
    return out
