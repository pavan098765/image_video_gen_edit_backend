import os
import sys
from typing import Any, Dict, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QPW_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
COMMON_SRC = os.path.join(QPW_ROOT, "worker_multi_common", "src")
if COMMON_SRC not in sys.path:
    sys.path.insert(0, COMMON_SRC)

from omni_config import build_omni_registry, omni_preprocess_hook  # type: ignore


def get_registry() -> Dict[str, Any]:
    return build_omni_registry(
        worker_root=os.path.dirname(os.path.dirname(THIS_DIR))
    )


def preprocess(
    model: str, mode: Optional[str], params: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    return omni_preprocess_hook(model, mode, params, job_id)
