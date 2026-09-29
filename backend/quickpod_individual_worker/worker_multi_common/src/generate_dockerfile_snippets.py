"""
Shared Dockerfile fragments for omni multi workers.
generate_dockerfile.py in each worker should set QPW_ROOT and import this module.
"""

import os
from collections import defaultdict
from typing import Callable, Dict, List, TextIO, Tuple

# Docker rejects a single COPY that lists too many sources (~128 argv slots); there is
# no limit on *how many COPY instructions* a Dockerfile may contain. Split shards use
# one source per COPY below, so huge models only add more layers, not more sources/line.
COPY_MAX_SOURCES_PER_LINE = 120

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
    "flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)


def copy_model_layers(
    f: TextIO, project_root: str, qpw_root: str, max_safetensors_bytes: int
) -> None:
    """
    Emit COPY lines relative to Docker build context = qpw_root (parent of all worker_* dirs).

    Non-split model files sharing the same image destination are batched on one COPY line
    (bounded by COPY_MAX_SOURCES_PER_LINE) to stay under Docker's multi-source limit.

    ``*.part-*`` shards (from ``split -b <fixed size>`` in prep_model, e.g. 500M/700M/850M)
    emit **one COPY per shard**: part count grows with tensor file size, but each instruction
    still has a single source, so the ~126 limit cannot be hit by many splits. To reduce
    layer count instead, shard COPY lines could be batched here the same way as small files
    (at the cost of coarser build cache when one part changes).
    """
    project_root = os.path.abspath(project_root)
    qpw_root = os.path.abspath(qpw_root)
    models_dir = os.path.join(project_root, "models")
    if not os.path.isdir(models_dir):
        return

    def walk_non_chunk() -> List[str]:
        out: List[str] = []
        for root, dirs, files in os.walk(models_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in files:
                if ".part-" in fn:
                    continue
                if fn.endswith(".safetensors"):
                    full = os.path.join(root, fn)
                    if os.path.getsize(full) > max_safetensors_bytes:
                        continue
                out.append(os.path.join(root, fn))
        return out

    def rel_dest_for(abs_path: str) -> Tuple[str, str]:
        rel_src = os.path.relpath(abs_path, qpw_root).replace("\\", "/")
        rel_to_models = os.path.relpath(os.path.dirname(abs_path), models_dir).replace(
            "\\", "/"
        )
        dest_dir = f"/app/models/{rel_to_models}" if rel_to_models != "." else "/app/models"
        return rel_src, dest_dir

    by_dest: Dict[str, List[str]] = defaultdict(list)
    for abs_path in sorted(walk_non_chunk()):
        rel_src, dest_dir = rel_dest_for(abs_path)
        by_dest[dest_dir].append(rel_src)

    for dest_dir in sorted(by_dest.keys()):
        paths = sorted(by_dest[dest_dir])
        for i in range(0, len(paths), COPY_MAX_SOURCES_PER_LINE):
            batch = paths[i : i + COPY_MAX_SOURCES_PER_LINE]
            f.write("COPY " + " ".join(batch) + f" {dest_dir}/\n")

    parts: List[str] = []
    for root, _, files in os.walk(models_dir):
        for fn in files:
            if ".part-" in fn:
                parts.append(os.path.join(root, fn))
    parts.sort()
    for p in parts:
        rel_src, dest_dir = rel_dest_for(p)
        f.write(f"COPY {rel_src} {dest_dir}/\n")


def _dockerfile_embed_all_worker_src() -> bool:
    """Set OMNI_DOCKER_FULL_WORKER_SRC=1 when running generate_dockerfile.py to restore legacy COPY-all behavior."""
    return os.getenv("OMNI_DOCKER_FULL_WORKER_SRC", "").lower() in ("1", "true", "yes")


def write_omni_code_copies(f: TextIO, worker_dir: str) -> None:
    """
    worker_dir: basename of this multi folder (e.g. worker_flux_klein_9b_multi).
    Build context must be qpw_root — the directory that contains worker_dir and worker_multi_common.

    Default (**slim**): COPY only this multi's src and worker_multi_common/src.
    prep_model.py syncs primary/shared inference trees into:
      <multi>/src/inference/<worker_dir>/src
    so runtime can load everything from within the multi folder.

    Legacy (**full**): set ``OMNI_DOCKER_FULL_WORKER_SRC=1`` when running generate_dockerfile.py
    to embed every worker's ``src`` (requires full repo in build context).
    """
    w = str(worker_dir).strip().replace("\\", "/").strip("/")
    if _dockerfile_embed_all_worker_src():
        lines = [
            "# Omni runtime app/router for this specific multi image\n",
            f"COPY {w}/src /app/worker/src\n",
            "# Shared omni runtime framework\n",
            "COPY worker_multi_common/src /app/worker_multi_common/src\n",
            "# Legacy full embed: all inference worker src trees baked into image\n",
            "COPY worker_qwen_image_edit_2511_rapid_aio/src /app/worker_qwen_image_edit_2511_rapid_aio/src\n",
            "COPY worker_flux_klein_9b/src /app/worker_flux_klein_9b/src\n",
            "COPY worker_z_image_turbo/src /app/worker_z_image_turbo/src\n",
            "COPY worker_stable_diffusion_xl/src /app/worker_stable_diffusion_xl/src\n",
            "COPY worker_stable_diffusion_xl_outpaint/src /app/worker_stable_diffusion_xl_outpaint/src\n",
            "COPY worker_image_upscaler_aurasrv2/src /app/worker_image_upscaler_aurasrv2/src\n",
            "COPY worker_image_upscaler_swin2sr/src /app/worker_image_upscaler_swin2sr/src\n",
            "COPY worker_face_enhancer/src /app/worker_face_enhancer/src\n",
            "COPY worker_video_upscalers/src /app/worker_video_upscalers/src\n",
            "COPY worker_basic_utils/src /app/worker_basic_utils/src\n",
            "COPY worker_wan_22_fp8_lightx2v_t2v/src /app/worker_wan_22_fp8_lightx2v_t2v/src\n",
            "COPY worker_wan_22_fp8_lightx2v_i2v/src /app/worker_wan_22_fp8_lightx2v_i2v/src\n",
            "COPY worker_hunyan15_480p_t2V_distilled/src /app/worker_hunyan15_480p_t2V_distilled/src\n",
            "COPY worker_hunyan15_480p_i2V_step_distilled/src /app/worker_hunyan15_480p_i2V_step_distilled/src\n",
            "COPY worker_glm_ocr/src /app/worker_glm_ocr/src\n",
            "COPY worker_ltx2_3/src /app/worker_ltx2_3/src\n",
        ]
        f.writelines(lines)
        return

    f.write("ENV OMNI_SLIM_INFERENCE_SRC=1\n")
    f.write(f"ENV OMNI_MULTI_IMAGE_ID={w}\n")
    f.write("# Omni runtime app/router for this specific multi image\n")
    f.write(f"COPY {w}/src /app/worker/src\n")
    f.write("# Shared omni runtime framework\n")
    f.write("COPY worker_multi_common/src /app/worker_multi_common/src\n")


def write_omni_run_install(f: TextIO) -> None:
    f.write(
        "RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0\n"
    )
    f.write(f"RUN pip install --no-cache-dir {FLASH_ATTN_WHEEL}\n")
    f.write(
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "git openssh-client ffmpeg build-essential \\\n"
        "    && pip install --no-cache-dir -r /app/worker/src/requirements.txt \\\n"
        "    && pip install --no-cache-dir \"huggingface_hub[hf_transfer]\" hf_xet \\\n"
        "    && mkdir -p /app/worker_multi_common \\\n"
        "    && [ -f /app/worker_multi_common/omni_manifest.json ] || printf '{}\\n' > /app/worker_multi_common/omni_manifest.json \\\n"
        "    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n"
    )


def write_ltx2_3_run_install(f: TextIO) -> None:
    """
    LTX-2.3 multi: SDNQ from GitHub at image build; no flash-attn wheel (matches hand-written Dockerfile).
    """
    f.write(
        "RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0\n"
    )
    f.write(
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "git ffmpeg build-essential \\\n"
        "    && git clone --depth 1 https://github.com/Disty0/sdnq.git /opt/sdnq \\\n"
        "    && pip install --no-cache-dir -e /opt/sdnq \\\n"
        "    && pip install --no-cache-dir -r /app/worker/src/requirements.txt \\\n"
        "    && pip install --no-cache-dir \"huggingface_hub[hf_transfer]\" hf_xet \\\n"
        "    && mkdir -p /app/worker_multi_common \\\n"
        "    && [ -f /app/worker_multi_common/omni_manifest.json ] || printf '{}\\n' > /app/worker_multi_common/omni_manifest.json \\\n"
        "    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*\n"
    )


def write_omni_pythonpath(f: TextIO) -> None:
    f.write("ENV PYTHONPATH=/app/worker/src:/app/worker_multi_common/src:${PYTHONPATH}\n")


def write_omni_hf_env(f: TextIO) -> None:
    f.write("ENV HF_HUB_ENABLE_HF_TRANSFER=1\n")
    f.write("ENV HF_XET_HIGH_PERFORMANCE=1\n")
    f.write("# Encrypted JSON + HMAC with controller (set CONTROLLER_SHARED_SECRET at runtime).\n")
    f.write("ENV WORKER_TRANSPORT_CRYPTO=1\n")
    f.write("ENV MODELS_DIR=/app/models\n")
    f.write("# diffusers v0.36.x omni primary models; LTX (v038) overrides via write_flavor_env ltx2_3.\n")
    f.write("ENV OMNI_POD_DIFFUSERS_GROUP=v036\n")
    f.write("ENV QPW_ROOT=/app/worker/src/inference\n")
    f.write("ENV OMNI_RUNTIME_MANIFEST_PATH=/app/worker_multi_common/omni_manifest.json\n")
    f.write("ENV OMNI_RUNTIME_MANIFEST_SYNC_PATH=true\n")
    f.write("ENV QWEN_2511_OFFLINE=1\n")
    f.write("ENV REALESRGAN_LOCAL_PATH=/app/models/realesrgan\n")
    f.write("ENV AURASR_LOCAL_PATH=/app/models/aurasr_v2\n")
    f.write("ENV SWIN2SR_X2_LOCAL_PATH=/app/models/swin2sr/swin2sr_x2\n")
    f.write("ENV SWIN2SR_X4_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4\n")
    f.write("ENV SWIN2SR_X4_CLASSIC_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4_classic\n")
    f.write("ENV SWIN2SR_HF_CACHE_DIR=/app/models/swin2sr_hf_cache\n")
    f.write("ENV BIREFNET_LOCAL_PATH=/app/models/basic_utils/birefnet\n")
    f.write("ENV LAMA_LOCAL_PATH=/app/models/basic_utils/lama/lama.pt\n")
    f.write("ENV GLM_OCR_LOCAL_PATH=/app/models/glm_ocr\n")


def write_dockerfile_header(f: TextIO, base_image: str = "pavanalur3/deps-base-new:py311-cu128") -> None:
    f.write(f"FROM {base_image}\n")
    f.write("WORKDIR /app\n")


def write_dockerfile_footer(f: TextIO) -> None:
    f.write("WORKDIR /app/worker\n")
    f.write("EXPOSE 8000\n")


# Per-flavor extra ENV lines (paths must match inference + download plan)
EXTRA_ENV_BY_FLAVOR: Dict[str, Callable[[TextIO], None]] = {}


def _register_flavor(name: str, fn: Callable[[TextIO], None]) -> None:
    EXTRA_ENV_BY_FLAVOR[name] = fn


def _write_lines(f: TextIO, *lines: str) -> None:
    for line in lines:
        f.write(line if line.endswith("\n") else line + "\n")


_register_flavor(
    "flux_klein",
    lambda f: _write_lines(
        f,
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "qwen",
    lambda f: _write_lines(
        f,
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "sdxl",
    lambda f: _write_lines(
        f,
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "outpaint",
    lambda f: _write_lines(
        f,
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "z_image_turbo",
    lambda f: _write_lines(
        f,
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "wan_i2v",
    lambda f: _write_lines(
        f,
        "ENV MODEL_LOCAL_DIR=/app/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64\n",
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "wan_t2v",
    lambda f: _write_lines(
        f,
        "ENV MODEL_LOCAL_DIR=/app/models/fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32\n",
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)

_register_flavor(
    "hunyuan_i2v",
    lambda f: _write_lines(
        f,
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
    ),
)

_register_flavor(
    "hunyuan_t2v",
    lambda f: _write_lines(
        f,
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
    ),
)

_register_flavor(
    "ltx2_3",
    lambda f: _write_lines(
        f,
        "ENV OMNI_POD_DIFFUSERS_GROUP=v038\n",
        "ENV WORKSPACE_DIR=/workspace\n",
        "ENV LTX2_3_MODELS_ROOT=/app/models/ltx2_3\n",
        "ENV LTX2_3_DISTILLED_LOCAL=/app/models/ltx2_3/distilled\n",
        "ENV LTX2_3_SPATIAL_UPSAMPLER_LOCAL=/app/models/ltx2_3/spatial_upsampler\n",
        "ENV LTX2_3_LORA_CACHE=/app/models/ltx2_3/lora_cache\n",
        "ENV LTX2_3_OFFLOAD_MODE=model\n",
        "ENV LTX2_3_STAGE2_OFFLOAD=sequential\n",
        'ENV CIVITAI_LORA_URL="https://civitai.com/models/1811313/dr34ml4y-all-in-one-nsfw-wanltx2"\n',
        "ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511\n",
        "ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers\n",
        "ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters\n",
        "ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo\n",
        "ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo\n",
        "ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps\n",
        "ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet\n",
        "ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae\n",
        "ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline\n",
        "ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15\n",
        "ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v\n",
    ),
)


def write_flavor_env(f: TextIO, flavor: str) -> None:
    fn = EXTRA_ENV_BY_FLAVOR.get(flavor)
    if fn:
        fn(f)
