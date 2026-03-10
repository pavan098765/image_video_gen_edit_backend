#!/usr/bin/env python3
"""
Generate Dockerfile for FlashVSR Video Upscaler worker.
FROM flashvsr-pro:latest, COPY models + worker src, pip install deps, CMD uvicorn.
Do NOT override ENTRYPOINT - base image's entrypoint activates conda.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "FlashVSR-v1.1")
BASE_IMAGE = "pavanalur3/flashvsr-pro:latest"


def main():
    with open(OUT_DOCKERFILE, "w") as f:
        f.write(f"FROM {BASE_IMAGE}\n\n")

        # COPY models (base .dockerignore excludes models/)
        f.write("# Copy FlashVSR-v1.1 models (must exist after prep_model)\n")
        f.write("COPY models/FlashVSR-v1.1 /workspace/FlashVSR-Pro/models/FlashVSR-v1.1\n\n")

        # COPY worker src
        f.write("# Copy worker API\n")
        f.write("COPY src /app/worker/src\n\n")

        # pip install our minimal deps (runs in flashvsr env via base SHELL)
        f.write("# Install FastAPI / uvicorn for worker API\n")
        f.write("RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic tenacity requests httpx\n\n")

        f.write("ENV FLASHVSR_PRO_DIR=/workspace/FlashVSR-Pro\n")
        f.write("ENV PYTHONPATH=/app/worker/src\n")
        f.write("ENV WORKSPACE_DIR=/workspace\n\n")

        f.write("RUN mkdir -p /app/worker /workspace && touch /app/worker/uvicorn.log\n\n")
        f.write("WORKDIR /app/worker\n")
        f.write("EXPOSE 8000\n\n")

        # Do NOT set ENTRYPOINT - keep base's entrypoint.sh
        f.write('CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]\n')

    print(f"Wrote {OUT_DOCKERFILE}")


if __name__ == "__main__":
    main()
