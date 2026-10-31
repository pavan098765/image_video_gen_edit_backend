import re
import subprocess
from pathlib import Path

root = Path(".")
skip = {
    "backend/server_controller/src/config.env",
    "backend/server_controller/src/server_config.py",
    "backend/quickpod_worker/config.env",
    "backend/server_controller/src/serviceAccountKey.json",
    "_scan_secrets.py",
}
pats = {
    "hex_literal": re.compile(r'["\'][A-Fa-f0-9]{32,}["\']'),
}
raw = subprocess.check_output(["git", "ls-files", "-z"])
paths = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
for rel in paths:
    rel = rel.replace("\\", "/")
    if rel in skip:
        continue
    path = root / rel
    if not path.is_file() or path.stat().st_size > 2_000_000:
        continue
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        continue
    for name, pat in pats.items():
        for match in pat.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line = text.splitlines()[line_no - 1]
            if "os.getenv" in line or "os.environ" in line or "SECRETKEY" in match.group(0):
                continue
            if "nvapi-..." in match.group(0) or "sk-..." in match.group(0):
                continue
            print(f"{rel}:{line_no} {name} len={len(match.group(0))}")
