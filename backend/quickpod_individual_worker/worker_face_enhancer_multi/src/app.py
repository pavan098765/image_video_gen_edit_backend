import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QPW_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
COMMON_SRC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if COMMON_SRC not in sys.path:
    sys.path.insert(0, COMMON_SRC)

try:
    from merge import main as _merge_main

    _merge_main()
except Exception as _merge_error:
    print(f"merge preflight failed: {_merge_error}")

from multi_model_runtime import MultiModelWorker  # type: ignore
from omni_config import (  # type: ignore
    OMNI_SHARED_BAKED_MODELS,
    get_effective_model_eta_seconds,
    get_effective_supported_models,
    omni_download_plan_excluding,
)
from inference_router import get_registry, preprocess

BAKED_MODELS = frozenset({"face_enhancer"})

worker = MultiModelWorker(
    model_registry=get_registry(),
    supported_models=get_effective_supported_models(),
    model_eta_seconds=get_effective_model_eta_seconds(),
    download_plan=omni_download_plan_excluding(BAKED_MODELS | OMNI_SHARED_BAKED_MODELS),
    preprocess_hook=preprocess,
)

app = worker.create_fastapi_app(title=os.getenv("MODEL_KEY", "face-enhancer-multi-worker"))
