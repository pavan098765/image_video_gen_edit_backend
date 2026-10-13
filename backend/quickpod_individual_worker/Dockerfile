FROM pavanalur3/deps-base-new:py311-cu128
WORKDIR /app
COPY worker_flux_klein_9b_multi/models/rife/.gitattributes /app/models/rife/
COPY worker_flux_klein_9b_multi/models/rife/flownet.pkl /app/models/rife/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/README.md /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/.gitattributes /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/birefnet.py /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/config.json /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/.gitignore /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/handler.py /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/requirements.txt /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/BiRefNet_config.py /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/birefnet/model.safetensors /app/models/basic_utils/birefnet/
COPY worker_flux_klein_9b_multi/models/basic_utils/lama/lama.pt /app/models/basic_utils/lama/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4_classic/README.md /app/models/swin2sr/swin2sr_x4_classic/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4_classic/.gitattributes /app/models/swin2sr/swin2sr_x4_classic/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4_classic/config.json /app/models/swin2sr/swin2sr_x4_classic/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4_classic/preprocessor_config.json /app/models/swin2sr/swin2sr_x4_classic/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4_classic/model.safetensors /app/models/swin2sr/swin2sr_x4_classic/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x2/README.md /app/models/swin2sr/swin2sr_x2/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x2/.gitattributes /app/models/swin2sr/swin2sr_x2/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x2/config.json /app/models/swin2sr/swin2sr_x2/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x2/preprocessor_config.json /app/models/swin2sr/swin2sr_x2/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x2/model.safetensors /app/models/swin2sr/swin2sr_x2/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4/README.md /app/models/swin2sr/swin2sr_x4/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4/.gitattributes /app/models/swin2sr/swin2sr_x4/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4/config.json /app/models/swin2sr/swin2sr_x4/
COPY worker_flux_klein_9b_multi/models/swin2sr/swin2sr_x4/preprocessor_config.json /app/models/swin2sr/swin2sr_x4/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/README.md /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/.gitattributes /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/config.json /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/LICENSE.md /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/realesrgan/RealESRGAN_x2plus.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/CodeFormer.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/realesr-general-x4v3.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/GFPGANv1.4.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/GFPGANv1.2.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/RestoreFormer.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/RealESRGAN_x4plus.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/GFPGANv1.3.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth /app/models/realesrgan/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/README.md /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/.gitattributes /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/model_index.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/scheduler/scheduler_config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/scheduler/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/vae/diffusion_pytorch_model.safetensors /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/vae/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/vae/config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/vae/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model.safetensors.index.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/quantization_config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/tokenizer.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/special_tokens_map.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/vocab.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/chat_template.jinja /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/tokenizer_config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/merges.txt /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/added_tokens.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/tokenizer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/generation_config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model.safetensors.index.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/quantization_config.json /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/model.safetensors.part-aa /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/model.safetensors.part-ab /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/model.safetensors.part-ac /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/model.safetensors.part-ad /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/aurasr_v2/model.safetensors.part-ae /app/models/aurasr_v2/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-aa /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ab /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ac /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ad /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ae /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-af /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ag /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ah /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-ai /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00001-of-00002.safetensors.part-aj /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00002-of-00002.safetensors.part-aa /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00002-of-00002.safetensors.part-ab /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00002-of-00002.safetensors.part-ac /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/model-00002-of-00002.safetensors.part-ad /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/text_encoder/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-aa /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ab /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ac /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ad /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ae /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-af /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ag /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ah /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-ai /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00001-of-00002.safetensors.part-aj /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00002-of-00002.safetensors.part-aa /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/diffusion_pytorch_model-00002-of-00002.safetensors.part-ab /app/models/flux2_klein_9b_sdnq_4bit_svd_r32/transformer/
COPY worker_flux_klein_9b_multi/src /app/worker/src
COPY worker_multi_common/src /app/worker_multi_common/src
COPY worker_qwen_image_edit_2511_rapid_aio/src /app/worker_qwen_image_edit_2511_rapid_aio/src
COPY worker_flux_klein_9b/src /app/worker_flux_klein_9b/src
COPY worker_z_image_turbo/src /app/worker_z_image_turbo/src
COPY worker_stable_diffusion_xl/src /app/worker_stable_diffusion_xl/src
COPY worker_stable_diffusion_xl_outpaint/src /app/worker_stable_diffusion_xl_outpaint/src
COPY worker_image_upscaler_aurasrv2/src /app/worker_image_upscaler_aurasrv2/src
COPY worker_image_upscaler_swin2sr/src /app/worker_image_upscaler_swin2sr/src
COPY worker_face_enhancer/src /app/worker_face_enhancer/src
COPY worker_video_upscalers/src /app/worker_video_upscalers/src
COPY worker_basic_utils/src /app/worker_basic_utils/src
COPY worker_wan_22_fp8_lightx2v_t2v/src /app/worker_wan_22_fp8_lightx2v_t2v/src
COPY worker_wan_22_fp8_lightx2v_i2v/src /app/worker_wan_22_fp8_lightx2v_i2v/src
COPY worker_hunyan15_480p_t2V_distilled/src /app/worker_hunyan15_480p_t2V_distilled/src
COPY worker_hunyan15_480p_i2V_step_distilled/src /app/worker_hunyan15_480p_i2V_step_distilled/src

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client ffmpeg build-essential \
    && pip install --no-cache-dir -r /app/worker/src/requirements.txt \
    && pip install --no-cache-dir "huggingface_hub[hf_transfer]" hf_xet \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

ENV PYTHONPATH=/app/worker/src:/app/worker_multi_common/src:${PYTHONPATH}
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV MODELS_DIR=/app/models
ENV QWEN_2511_OFFLINE=1
ENV REALESRGAN_LOCAL_PATH=/app/models/realesrgan
ENV AURASR_LOCAL_PATH=/app/models/aurasr_v2
ENV SWIN2SR_X2_LOCAL_PATH=/app/models/swin2sr/swin2sr_x2
ENV SWIN2SR_X4_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4
ENV SWIN2SR_X4_CLASSIC_LOCAL_PATH=/app/models/swin2sr/swin2sr_x4_classic
ENV SWIN2SR_HF_CACHE_DIR=/app/models/swin2sr_hf_cache
ENV BIREFNET_LOCAL_PATH=/app/models/basic_utils/birefnet
ENV LAMA_LOCAL_PATH=/app/models/basic_utils/lama/lama.pt
ENV QWEN_2511_LOCAL_PATH=/app/models/qwen_image_edit_2511
ENV AIO_TRANSFORMER_LOCAL_PATH=/app/models/aio_transformers
ENV LORA_ADAPTERS_LOCAL_PATH=/app/models/lora_adapters
ENV Z_IMAGE_TURBO_LOCAL_PATH=/app/models/z_image_turbo
ENV NUNCHAKU_Z_IMAGE_TURBO_LOCAL_PATH=/app/models/nunchaku_z_image_turbo
ENV SDXL_LOCAL_PATH=/app/models/sdxl_lighting_8_steps
ENV CTRLNET_LOCAL_PATH=/app/models/sdxl_outpaint/controlnet
ENV VAE_LOCAL_PATH=/app/models/sdxl_outpaint/vae
ENV PIPELINE_LOCAL_PATH=/app/models/sdxl_outpaint/pipeline
ENV HUNYAN15_MODEL_LOCAL_PATH=/app/models/hunyan15
ENV HUNYAN15_T2V_LOCAL_PATH=/app/models/hunyan15_t2v

WORKDIR /app/worker
EXPOSE 8000
