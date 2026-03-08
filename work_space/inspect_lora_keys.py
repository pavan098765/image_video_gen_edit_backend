#!/usr/bin/env python3
"""Inspect actual keys inside a safetensors LoRA file."""
import sys
from safetensors.torch import load_file

path = sys.argv[1]
sd = load_file(path, device="cpu")
keys = list(sd.keys())
print(f"Total keys: {len(keys)}")
print(f"First key:  {keys[0]}")
print(f"\nFirst 10 keys:")
for k in keys[:10]:
    print(f"  {k}")

# Detect format
sample = keys[0]
if sample.startswith("diffusion_model."):
    print("\n>>> FORMAT: ComfyUI — needs conversion")
elif sample.startswith("lora_unet_"):
    print("\n>>> FORMAT: Kohya/flat — needs conversion")
elif sample.startswith("blocks."):
    print("\n>>> FORMAT: Diffusers — already correct")
else:
    print(f"\n>>> FORMAT: UNKNOWN — prefix: {sample.split('.')[0]}")