"""
Video post-processing bridge retained in multi-model workers.
Prefers nested WAN I2V copy (present in slim Docker images); falls back to repo sibling.
"""
import importlib.util
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NESTED = os.path.join(
    THIS_DIR,
    "inference",
    "worker_wan_22_fp8_lightx2v_i2v",
    "src",
    "video_postprocessing.py",
)
_REPO_SIBLING = os.path.join(
    os.path.dirname(os.path.dirname(THIS_DIR)),
    "worker_wan_22_fp8_lightx2v_i2v",
    "src",
    "video_postprocessing.py",
)
LEGACY_FILE = _NESTED if os.path.isfile(_NESTED) else _REPO_SIBLING

_SPEC = importlib.util.spec_from_file_location(
    "legacy_video_postprocessing_wan_i2v", LEGACY_FILE
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)

for _name in dir(_MOD):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MOD, _name)

