"""Scan workflow JSON and list all node types (class_type) for custom node mapping."""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        path = Path(__file__).resolve().parent.parent / "workflow" / "WAN2.2 T2V-I2V-T2I-S2V K3NK v2.5.4 SVI.json"
    else:
        path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    types = {}
    for n in data.get("nodes", []):
        t = n.get("type") or n.get("class_type")
        if t and not t.startswith("{"):  # skip uuid-like
            types[t] = types.get(t, 0) + 1
    out = path.parent / "workflow_node_types.txt"
    with open(out, "w", encoding="utf-8") as f:
        for t in sorted(types.keys()):
            f.write(f"{t}\n")
    print(f"Written {len(types)} node types to {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
