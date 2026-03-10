#!/usr/bin/env python3
"""
diagnose_lora_keys.py
=====================
Inspects LoRA and model parameter keys to diagnose "0 layers baked" issues.
Optionally dumps ALL keys to a text file you can share for further analysis.

What it produces:
  1. Console: format detection, prefix analysis, match test, suggested fix
  2. lora_key_dump.txt (--dump): ALL keys from every LoRA + model transformer,
     with shapes — safe to share, contains no weight values

Usage:
    # LoRA inspection only (fast, no model load):
    python diagnose_lora_keys.py \
        --lora /path/to/high.safetensors /path/to/low.safetensors

    # Full match test against model:
    python diagnose_lora_keys.py \
        --lora /path/to/high.safetensors /path/to/low.safetensors \
        --model-dir /path/to/diffusers/model

    # Same + dump ALL keys to text file for sharing:
    python diagnose_lora_keys.py \
        --lora /path/to/high.safetensors /path/to/low.safetensors \
        --model-dir /path/to/diffusers/model \
        --dump

    # Custom dump output path:
    python diagnose_lora_keys.py \
        --lora /path/to/lora.safetensors \
        --model-dir /path/to/model \
        --dump --dump-file /tmp/my_key_dump.txt

    # SVI example:
    python diagnose_lora_keys.py \
        --lora /work_space/models/loras/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro_diffusers.safetensors \
               /work_space/models/loras/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro_diffusers.safetensors \
        --model-dir /work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64 \
        --dump
"""

import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LORA_SUFFIXES = [
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
]

# Ordered longest-first so more-specific prefixes are tried first
CANDIDATE_STRIP_PREFIXES = [
    "base_model.model.transformer_2.",
    "base_model.model.transformer.",
    "base_model.model.",
    "transformer_2.",
    "transformer.",
    "diffusion_model.",
]


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
def strip_lora_suffix(key: str):
    for sfx in LORA_SUFFIXES:
        if key.endswith(sfx):
            return key[:-len(sfx)], sfx
    return key, ""


def detect_lora_format(keys: list) -> str:
    sample = keys[0] if keys else ""
    if sample.startswith("base_model.model."):
        return "PEFT-wrapped  (base_model.model. prefix)"
    if sample.startswith("transformer_2."):
        return "PEFT-diffusers-native  (transformer_2. prefix)"
    if sample.startswith("transformer."):
        return "PEFT-diffusers-native  (transformer. prefix)"
    if sample.startswith("diffusion_model."):
        return "ComfyUI  (diffusion_model. prefix)"
    if sample.startswith("lora_unet_"):
        return "Kohya/flat  (lora_unet_ prefix)"
    if sample.startswith("blocks.") or ".blocks." in sample:
        return "Diffusers-native  (blocks. prefix — already correct)"
    return f"Unknown  (sample: {sample[:80]})"


def prefix_breakdown(keys: list, top_n: int = 8) -> dict:
    groups = defaultdict(int)
    for k in keys:
        groups[k.split(".")[0]] += 1
    return dict(sorted(groups.items(), key=lambda x: -x[1])[:top_n])


def find_best_strip_prefix(base_keys: list, model_keys: set):
    """Try every candidate prefix and return the one with highest match count."""
    best_prefix, best_count = "", 0
    total = len(base_keys)
    for prefix in [""] + CANDIDATE_STRIP_PREFIXES:
        count = sum(
            1 for b in base_keys
            if (b[len(prefix):] if prefix and b.startswith(prefix) else b) + ".weight" in model_keys
        )
        if count > best_count:
            best_count, best_prefix = count, prefix
    return best_prefix, best_count, total


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------
class DumpWriter:
    """Writes to both console (optionally) and a dump file."""

    def __init__(self, dump_path: str = None):
        self.dump_path = dump_path
        self._fh = None
        if dump_path:
            self._fh = open(dump_path, "w", encoding="utf-8")
            self._fh.write(f"# lora_key_dump.txt — generated {datetime.now().isoformat()}\n")
            self._fh.write("# Contains key names and shapes only — no weight values.\n")
            self._fh.write("# Safe to share.\n\n")

    def write(self, text: str, console: bool = True):
        if console:
            print(text)
        if self._fh:
            self._fh.write(text + "\n")

    def section(self, title: str):
        line = "=" * 65
        self.write(f"\n{line}")
        self.write(title)
        self.write(line)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def dump_all_lora_keys(lora_path: str, sd: dict, dw: DumpWriter):
    """Write every key + shape from a LoRA to the dump file."""
    dw.section(f"ALL KEYS — {os.path.basename(lora_path)}  ({len(sd)} total)")
    dw.write(f"{'Key':<90}  Shape")
    dw.write("-" * 110)
    for k in sorted(sd.keys()):
        shape_str = str(list(sd[k].shape))
        dw.write(f"{k:<90}  {shape_str}")


def dump_all_model_keys(component_name: str, params: dict, dw: DumpWriter):
    """Write every parameter key + shape from a model component to the dump file."""
    dw.section(f"ALL KEYS — model.{component_name}  ({len(params)} params)")
    dw.write(f"{'Key':<90}  Shape  Dtype")
    dw.write("-" * 120)
    for k in sorted(params.keys()):
        p = params[k]
        shape_str = str(list(p.shape))
        dtype_str = str(p.dtype).replace("torch.", "")
        dw.write(f"{k:<90}  {shape_str:<20}  {dtype_str}")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_lora(lora_path: str, model_params: dict, dw: DumpWriter):
    from safetensors.torch import load_file

    dw.section(f"LoRA: {os.path.basename(lora_path)}")

    try:
        sd = load_file(lora_path, device="cpu")
    except Exception as e:
        dw.write(f"  ERROR loading file: {e}")
        return

    all_keys = list(sd.keys())
    dw.write(f"  Total keys   : {len(all_keys)}")
    dw.write(f"  Format       : {detect_lora_format(all_keys)}")
    dw.write(f"  Key prefixes : {prefix_breakdown(all_keys)}")

    # LoRA suffix inventory
    suffixes_found = set()
    for k in all_keys:
        for sfx in LORA_SUFFIXES:
            if k.endswith(sfx):
                suffixes_found.add(sfx)
    dw.write(f"  Suffixes     : {sorted(suffixes_found)}")

    # Sample keys
    dw.write(f"\n  Sample keys (first 15):")
    for k in all_keys[:15]:
        dw.write(f"    {k}  {list(sd[k].shape)}")

    # Collect base keys from down/A weights
    base_keys = []
    for k in all_keys:
        base, sfx = strip_lora_suffix(k)
        if sfx in (".lora_A.weight", ".lora_down.weight"):
            base_keys.append(base)

    dw.write(f"\n  Unique LoRA layers : {len(base_keys)}")

    if not base_keys:
        dw.write("\n  WARNING: No lora_A / lora_down keys found.")
        dw.write("  Non-standard suffixes present:")
        non_std = set()
        for k in all_keys:
            parts = k.rsplit(".", 1)
            if len(parts) == 2:
                non_std.add("." + parts[1])
        for s in sorted(non_std)[:20]:
            dw.write(f"    {s}")
        # Still dump all keys if dump enabled
        if dw.dump_path:
            dump_all_lora_keys(lora_path, sd, dw)
        return

    # Match test
    if model_params is not None:
        model_keys = set(model_params.keys())
        best_prefix, match_count, total = find_best_strip_prefix(base_keys, model_keys)
        pct = 100 * match_count / total if total else 0

        dw.write(f"\n  {'='*55}")
        dw.write(f"  MATCH TEST vs model ({len(model_keys)} params):")
        dw.write(f"  {'='*55}")
        dw.write(f"  Best strip prefix : '{best_prefix}'")
        dw.write(f"  Layers matched    : {match_count}/{total}  ({pct:.0f}%)")

        if match_count == 0:
            dw.write(f"\n  ✗ ZERO matches — LoRA may need full key remapping, not just prefix strip.")
            dw.write(f"    Check the full key dump (--dump) and share for analysis.")
        elif best_prefix == "" and match_count == total:
            dw.write(f"\n  ✓ Perfect match with NO prefix stripping needed.")
            dw.write(f"    The '0 layers baked' bug is in the pairs-building loop,")
            dw.write(f"    not the key lookup. Check that rfind('.lora_') finds the suffix.")
        elif pct >= 80:
            dw.write(f"\n  ✓ Good match — strip prefix '{best_prefix}' in inject_lora().")
        else:
            dw.write(f"\n  ⚠ Partial match ({pct:.0f}%) — may need remapping + prefix strip.")

        # Concrete examples
        dw.write(f"\n  Verification (first 8 base keys):")
        for base in base_keys[:8]:
            stripped = base[len(best_prefix):] if best_prefix and base.startswith(best_prefix) else base
            wkey = stripped + ".weight"
            status = "✓" if wkey in model_keys else "✗"
            dw.write(f"    {status}  {base}")
            dw.write(f"         → {wkey}")

        # Suggested fix
        dw.write(f"\n  SUGGESTED FIX for inject_lora() in quantize_sdnq_multi.py:")
        dw.write(f"  {'─'*55}")
        if best_prefix:
            dw.write(f"""
  Add this block in inject_lora(), AFTER the lora_sfx stripping loop
  and BEFORE any TRANSFORMER_KEYS_RENAME_DICT remapping:

      # Strip module-level prefix from PEFT/diffusers-native LoRA keys.
      # LoRA key prefix '{best_prefix}' must be removed because the model
      # passed to inject_lora() IS that submodule (not the full pipeline).
      _STRIP_PREFIXES = [
          "base_model.model.transformer_2.",
          "base_model.model.transformer.",
          "base_model.model.",
          "transformer_2.",
          "transformer.",
      ]
      for _pfx in _STRIP_PREFIXES:
          if k.startswith(_pfx):
              k = k[len(_pfx):]
              break
""")
        else:
            dw.write(f"""
  No prefix stripping needed — keys already match.
  The bug is elsewhere (likely pairs dict not being built correctly).
  Share the full key dump (--dump) for further analysis.
""")

    # Dump all keys to file (no model needed for this part)
    if dw.dump_path:
        dump_all_lora_keys(lora_path, sd, dw)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_params(model_dir: str, dw: DumpWriter) -> dict:
    dw.section(f"Model: {os.path.basename(model_dir)}")

    workspace = os.path.dirname(os.path.abspath(__file__))
    sdnq_src = os.path.join(workspace, "sdnq", "src")
    if os.path.exists(sdnq_src) and sdnq_src not in sys.path:
        sys.path.insert(0, sdnq_src)

    try:
        from sdnq import SDNQConfig
        dw.write("  ✓ SDNQ registered")
    except ImportError:
        dw.write("  ⚠ SDNQ not found (may cause errors with quantized models)")

    import diffusers
    dw.write("  Loading pipeline on CPU (may take ~30s)...")
    try:
        pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
            model_dir, torch_dtype="auto", low_cpu_mem_usage=True,
        )
    except Exception as e:
        dw.write(f"  ERROR: {e}")
        return {}

    all_params = {}
    for name in ["transformer", "transformer_2"]:
        component = getattr(pipe, name, None)
        if component is None:
            dw.write(f"  {name}: not present")
            continue
        params = dict(component.named_parameters())
        dw.write(f"  {name}: {len(params)} parameters")
        dw.write(f"  {name} key prefixes: {prefix_breakdown(list(params.keys()))}")
        dw.write(f"  {name} sample keys (first 10):")
        for k in list(params.keys())[:10]:
            p = params[k]
            dw.write(f"    {k}  {list(p.shape)}  {str(p.dtype).replace('torch.','')}")
        all_params.update(params)

        # Dump all model keys if dump requested
        if dw.dump_path:
            dump_all_model_keys(name, params, dw)

    dw.write(f"\n  Total params collected: {len(all_params)}")
    return all_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Diagnose LoRA↔model key mismatch. Use --dump to export all keys to a shareable file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lora", nargs="+", required=True, metavar="PATH",
        help="One or more LoRA .safetensors files to inspect.",
    )
    parser.add_argument(
        "--model-dir", metavar="DIR", default=None,
        help="Diffusers model directory (optional, enables match testing).",
    )
    parser.add_argument(
        "--dump", action="store_true",
        help="Dump ALL keys + shapes from every LoRA and the model to --dump-file.",
    )
    parser.add_argument(
        "--dump-file", metavar="PATH", default="lora_key_dump.txt",
        help="Output path for key dump (default: lora_key_dump.txt next to this script).",
    )
    args = parser.parse_args()

    # Resolve dump path relative to script location
    if args.dump and not os.path.isabs(args.dump_file):
        args.dump_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dump_file)

    dump_path = args.dump_file if args.dump else None
    dw = DumpWriter(dump_path)

    dw.write("\n" + "=" * 65)
    dw.write("Wan LoRA Key Diagnostics")
    dw.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dw.write("=" * 65)

    if dump_path:
        dw.write(f"\nKey dump will be saved to: {dump_path}")

    # Validate LoRA paths
    lora_paths = []
    for p in args.lora:
        if not os.path.exists(p):
            dw.write(f"WARNING: LoRA not found, skipping: {p}")
        else:
            lora_paths.append(p)

    if not lora_paths:
        dw.write("ERROR: No valid LoRA files found.")
        dw.close()
        sys.exit(1)

    # Load model params once
    model_params = None
    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            dw.write(f"ERROR: Model directory not found: {args.model_dir}")
            dw.close()
            sys.exit(1)
        model_params = load_model_params(args.model_dir, dw)

    # Analyze each LoRA
    for lora_path in lora_paths:
        analyze_lora(lora_path, model_params, dw)

    # Footer
    dw.write("\n" + "=" * 65)
    dw.write("Diagnosis complete.")
    if dump_path:
        dw.write(f"\nFull key dump saved to: {dump_path}")
        dw.write("Share this file for detailed analysis.")
    dw.write("=" * 65 + "\n")

    dw.close()


if __name__ == "__main__":
    main()