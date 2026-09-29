# Provenance and third-party notice

## Companion workflow

This package accompanies the modified community workflow
`LTX-2.5-INT8-AllModes-Stage2-Speed-Quality-LoRA-Civitai-v1.0.json`, SHA-256
`1D31375A81453B5E3E99E262E46AF839613B0B7A7A6F0A98670ABF2611189EB6`.

The workflow was developed with reference to and incorporates elements of:

- repository: `https://github.com/Lightricks/ComfyUI-LTXVideo`
- commit: `ac4d99839020b983e956a8ab67ec38aec1b6e65a`
- file: `example_workflows/2.5/LTX-2.5_T2V_I2V_Two_Stage_Distilled.json`
- source Git blob: `aa576a3aa5453993a9e0d02f1ebc8807497d258a`
- source raw SHA-256: `C6359BA0151EA6BBD94C4E4E5125571521B5E4AFB1493C642B35D49228FD7787`

The modified workflow remains subject to the complete upstream source license
preserved verbatim at
`LICENSES/ComfyUI-LTXVideo-ac4d998-LTX-2-Community-2026-01-05.txt`.

Prominent workflow changes include a single T2V/I2V/FLF2V mode router,
explicit final canvas and timing controls, bounded portable image loading, an
INT8 ConvRot transformer path, selectable Stage 2 Sage/Sol or native attention,
optional quantized-base LoRA pass-through, official x2 latent upscaling,
guide crop/reapply paths, decode-tile presets, and a pre-decode unload barrier.

## Locally authored code

To the extent of rights owned by contributor `sjh6229`, the MIT License in
`LICENSES/MIT.txt` applies to:

- `custom_nodes/ltx25_smart_controls/__init__.py`
- this package's contributor-authored documentation and metadata

The MIT grant does not replace or modify the terms governing the modified
workflow, LTX model material, trademarks, or third-party dependencies.

## External materials not included

This ZIP does not include the workflow JSON, model weights, LoRAs, example
media, ComfyUI, ComfyUI-KJNodes, ComfyUI-SolAttn_triton, SageAttention, or
Triton. They retain their own licenses and security responsibilities.

The official LTX-2.x license text from LTX-2 commit
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` is preserved verbatim at
`LICENSES/LTX-2.x-fd4ded7-Community-2026-08-11.md` for notice and reference.
It does not replace the terms presented with the separately downloaded gated
artifact. Verify and accept the official terms before download or use:

- `https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md`
- `https://huggingface.co/Lightricks/LTX-2.5`

No trademark license or endorsement is implied. LTX, Lightricks, ComfyUI, and
third-party project names belong to their respective owners.
