FROM pavanalur3/deps-base-new:py311-cu128

WORKDIR /app

ENV COMFYUI_DIR="/app/comfyui"
ENV MODELS_DIR="/app/comfyui/models"
ENV WORKFLOW_DIR="/app/worker/workflow"
ENV WORKSPACE_DIR="/workspace"
ENV PYTHONPATH="/app/worker/src:${PYTHONPATH}"
ENV GIT_TERMINAL_PROMPT=0

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/comfyui /app/worker /workspace

RUN git clone --depth 1 --branch v0.37.0 https://github.com/Comfy-Org/ComfyUI /app/comfyui && pip install --no-cache-dir -r /app/comfyui/requirements.txt

RUN mkdir -p /app/comfyui/custom_nodes /app/comfyui/models /app/comfyui/input /app/comfyui/output /app/comfyui/temp /app/comfyui/user/default/workflows

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
