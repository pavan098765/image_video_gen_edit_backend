# LTX-2.5 INT8 AllModes companion nodes

This ZIP contains the locally authored ComfyUI helper nodes required by the
`LTX-2.5 INT8 AllModes - Stage 2 Speed/Quality - Optional LoRA` workflow.

It does **not** contain model weights, LoRAs, example images, generated media,
credentials, or third-party custom-node source code.

## Matching workflow

- Civitai workflow file: `LTX-2.5-INT8-AllModes-Stage2-Speed-Quality-LoRA-Civitai-v1.0.json`
- SHA-256: `1D31375A81453B5E3E99E262E46AF839613B0B7A7A6F0A98670ABF2611189EB6`
- Workflow node contract: `LTX25AllModesControlsV2`

## Install

1. Close ComfyUI.
2. Extract this ZIP into the **ComfyUI root directory**. After extraction, this
   exact file must exist:

   `ComfyUI/custom_nodes/ltx25_smart_controls/__init__.py`

3. Install the external dependencies from the authoritative sources and pinned
   versions listed in `DEPENDENCIES.md`.
4. Place the five LTX-2.5 model files in the exact model folders listed there.
5. Restart ComfyUI, refresh the browser, then import the workflow JSON.

Do not search for a lookalike package based only on a missing node class name.
This ZIP is the required source for all `LTX25...` nodes used by the workflow.

## What the nodes do

- Route T2V, I2V, and FLF2V from one mode control.
- Validate final resolution, frame count, FPS, and decode-tile controls.
- Load only the guide images required by the selected mode.
- Reject guide images outside the ComfyUI input directory, files larger than
  256 MiB, images over 80 million decoded pixels, and animated/multi-frame files.
- Apply the official core LTX I2V/first-last guide operations.
- Select Stage 2 Speed or Quality attention without invalidating Stage 1.
- Optionally apply one ordinary LTX transformer LoRA through ComfyUI's
  quantized-base bypass path. `None` is a true pass-through.

## Stage 2 modes

- **Speed - Sage + Sol (12-16GB target):** approximate attention for both stages.
- **Quality - Native exact (24GB+ recommended):** native exact attention for
  Stage 2. The recommendation is not a guarantee that 1080p/15-second jobs will
  avoid model offload or run quickly on every 24GB card.

The VAE decode preset is independent of this choice and changes tiled decode
memory/speed only. A physical 12GB GPU has not been validated.

## Security notes

The bundled Python module performs no network requests, subprocess launches,
shell commands, render submissions, or file writes. It can read a user-selected
still image from ComfyUI's input directory and a user-selected LoRA from
ComfyUI's configured LoRA directory. LoRA loading uses ComfyUI's safe tensor
loader path (`safe_load=True`).

Custom nodes execute Python on the recipient's machine. Verify
`SHA256SUMS.txt` before installing this package. External dependencies are not
covered by this package's source review and retain their own security and
license terms.

## Compatibility and validation

Validated host values are recorded in `DEPENDENCIES.md`. Windows 11 with an
RTX 5080 16GB was the development host. The final package was statically
reviewed and packaged without starting ComfyUI, submitting `/prompt`, or
running a GPU render.

## Licenses and provenance

The locally authored helper code and package documentation are MIT-licensed to
the extent of the contributor's rights. The companion workflow incorporates
elements of Lightricks' official ComfyUI-LTXVideo workflow and remains subject
to the preserved upstream license. Model weights and third-party dependencies
are not bundled. Read `LICENSE`, `NOTICE.md`, and `LICENSES/` before
redistribution.

This is an independent community project and is not endorsed by Lightricks,
ComfyUI, KJNodes, SageAttention, or Sol-Attention.
