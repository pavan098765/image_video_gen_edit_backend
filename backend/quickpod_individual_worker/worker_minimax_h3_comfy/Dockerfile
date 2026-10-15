FROM pavanalur3/deps-base-new:py311-cu128

WORKDIR /app

ENV COMFYUI_DIR="/app/comfyui"
ENV MODELS_DIR="/app/comfyui/models"
ENV WORKFLOW_DIR="/app/worker/workflow"
ENV WORKSPACE_DIR="/workspace"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"
ENV GIT_TERMINAL_PROMPT=0
ENV CC="gcc"
ENV CXX="g++"
ENV TRITON_CACHE_DIR="/root/.triton"

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ninja-build && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/comfyui /app/worker /workspace /root/.triton

RUN git clone --depth 1 --branch v0.37.0 https://github.com/Comfy-Org/ComfyUI /app/comfyui && pip install --no-cache-dir -r /app/comfyui/requirements.txt

RUN mkdir -p /app/comfyui/custom_nodes /app/comfyui/models /app/comfyui/input /app/comfyui/output /app/comfyui/temp /app/comfyui/user/default/workflows

# Fast MiniMax H3: Kitchen attention + Triton (SolAttn JIT compiles on first GPU job).
RUN pip install --no-cache-dir triton && pip install --no-cache-dir comfy-kitchen || echo 'WARNING: comfy-kitchen/triton install failed; Fast H3 attention may error'

RUN git clone https://github.com/kijai/ComfyUI-KJNodes /app/comfyui/custom_nodes/ComfyUI-KJNodes && git -C /app/comfyui/custom_nodes/ComfyUI-KJNodes checkout d3cfe21625e5170126ce06fbfcfe1d88108688c3 && git -C /app/comfyui/custom_nodes/ComfyUI-KJNodes submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt; fi

RUN git clone https://github.com/kijai/ComfyUI-SolAttn_triton /app/comfyui/custom_nodes/ComfyUI-SolAttn_triton && git -C /app/comfyui/custom_nodes/ComfyUI-SolAttn_triton checkout dfc2e31a41afd72bd53dd2137fc8b2931d5ec192 && git -C /app/comfyui/custom_nodes/ComfyUI-SolAttn_triton submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-SolAttn_triton/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-SolAttn_triton/requirements.txt; fi

RUN git clone https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale /app/comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale && git -C /app/comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale checkout fe6658f6d144066f14150d3526247b417683ff2b && git -C /app/comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale/requirements.txt; fi

RUN git clone https://github.com/rgthree/rgthree-comfy /app/comfyui/custom_nodes/rgthree-comfy && git -C /app/comfyui/custom_nodes/rgthree-comfy checkout 449c58fcdd612f7733e54c51f6758ead63fa180b && git -C /app/comfyui/custom_nodes/rgthree-comfy submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/rgthree-comfy/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/rgthree-comfy/requirements.txt; fi

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite && git -C /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout 4d907bee61e92c2e65af3bd6383a4e4d356126d1 && git -C /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; fi

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack /app/comfyui/custom_nodes/ComfyUI-Impact-Pack && git -C /app/comfyui/custom_nodes/ComfyUI-Impact-Pack checkout 429d0159ad429e64d2b3916e6e7be9c22d025c3c && git -C /app/comfyui/custom_nodes/ComfyUI-Impact-Pack submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt; fi

RUN git clone https://github.com/chrisgoringe/cg-use-everywhere /app/comfyui/custom_nodes/cg-use-everywhere && git -C /app/comfyui/custom_nodes/cg-use-everywhere checkout 50ae9f8c5d8b9538589663c90a15d4067a02969c && git -C /app/comfyui/custom_nodes/cg-use-everywhere submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/cg-use-everywhere/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/cg-use-everywhere/requirements.txt; fi

RUN git clone https://github.com/yolain/ComfyUI-Easy-Use /app/comfyui/custom_nodes/ComfyUI-Easy-Use && git -C /app/comfyui/custom_nodes/ComfyUI-Easy-Use checkout 8730ffd14044ee9392db3b192646266576bc67df && git -C /app/comfyui/custom_nodes/ComfyUI-Easy-Use submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-Easy-Use/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-Easy-Use/requirements.txt; fi

RUN git clone https://github.com/pixaroma/ComfyUI-Pixaroma /app/comfyui/custom_nodes/ComfyUI-Pixaroma && git -C /app/comfyui/custom_nodes/ComfyUI-Pixaroma checkout 7c464447ce2a17dff9489b79d4daf32e55b28f38 && git -C /app/comfyui/custom_nodes/ComfyUI-Pixaroma submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-Pixaroma/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-Pixaroma/requirements.txt; fi

RUN git clone https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE /app/comfyui/custom_nodes/ComfyUI-H3-Prompt-IDE && git -C /app/comfyui/custom_nodes/ComfyUI-H3-Prompt-IDE checkout 36f8a720e0b0775ecdb7bae41fceae3b513ff2cd && git -C /app/comfyui/custom_nodes/ComfyUI-H3-Prompt-IDE submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-H3-Prompt-IDE/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-H3-Prompt-IDE/requirements.txt; fi

RUN git clone https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder /app/comfyui/custom_nodes/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder && git -C /app/comfyui/custom_nodes/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder checkout 23038f5050acdcbb8e938bfd7c87a18d2cc84aab && git -C /app/comfyui/custom_nodes/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/requirements.txt; fi

RUN git clone https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod /app/comfyui/custom_nodes/ComfyUI-MiniMaxH3Mod && git -C /app/comfyui/custom_nodes/ComfyUI-MiniMaxH3Mod checkout 6131c1f62c612a074d6aac8435c26ffa19c00563 && git -C /app/comfyui/custom_nodes/ComfyUI-MiniMaxH3Mod submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-MiniMaxH3Mod/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-MiniMaxH3Mod/requirements.txt; fi

RUN git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler /app/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler && git -C /app/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler checkout 40316cf008b2fd8663263270669eb4da23f89d2c && git -C /app/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/requirements.txt; fi

RUN git clone https://github.com/GACLove/ComfyUI-VFI /app/comfyui/custom_nodes/ComfyUI-VFI && git -C /app/comfyui/custom_nodes/ComfyUI-VFI checkout 6176a430f12cd16003f4664c1e3c6af8e96cc3c6 && git -C /app/comfyui/custom_nodes/ComfyUI-VFI submodule update --init --recursive && if [ -f /app/comfyui/custom_nodes/ComfyUI-VFI/requirements.txt ]; then pip install --no-cache-dir -r /app/comfyui/custom_nodes/ComfyUI-VFI/requirements.txt; fi

COPY requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

COPY nodes.lock.json /app/worker/nodes.lock.json
COPY models_manifest.json /app/worker/models_manifest.json
COPY workflow/ /app/worker/workflow/
COPY src/ /app/worker/src/
COPY static/ /app/worker/static/

RUN mkdir -p /app/comfyui/output /app/comfyui/temp /app/comfyui/input /workspace && cp /app/worker/workflow/*.json /app/comfyui/user/default/workflows/ && chmod +x /app/worker/src/start.sh

WORKDIR /app/worker
EXPOSE 8000 8188
CMD ["/app/worker/src/start.sh"]
