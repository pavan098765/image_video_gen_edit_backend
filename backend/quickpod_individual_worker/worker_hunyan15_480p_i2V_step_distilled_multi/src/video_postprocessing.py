"""
Video post-processing bridge retained in multi-model workers.
"""
import importlib.util
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QPW_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
LEGACY_FILE = os.path.join(
    QPW_ROOT, "worker_hunyan15_480p_i2V_step_distilled", "src", "video_postprocessing.py"
)

_SPEC = importlib.util.spec_from_file_location("legacy_video_postprocessing_hunyuan_i2v", LEGACY_FILE)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)

for _name in dir(_MOD):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MOD, _name)

