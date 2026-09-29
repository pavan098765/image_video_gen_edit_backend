# Security notes

## Package boundary

The ZIP allowlist contains one executable Python source file plus documentation,
license texts, a manifest, and checksums. It intentionally excludes model
weights, media, prompts, credentials, local paths, caches, compiled bytecode,
logs, package managers, installers, and third-party source trees.

## Expected local access

The helper module can:

- enumerate filenames in the configured ComfyUI input and LoRA directories for
  UI selectors;
- read only a mode-enabled still image contained in the real ComfyUI input
  directory after encoded-size, decoded-pixel, and single-frame checks;
- hash that selected image in bounded streaming chunks for ComfyUI cache
  invalidation; and
- load a user-selected LoRA through ComfyUI with `safe_load=True`.

It does not make network requests, spawn processes, invoke a shell, submit a
render, or write files.

## Recipient guidance

- Verify `SHA256SUMS.txt` before copying files into ComfyUI.
- Install external nodes only from the pinned authoritative sources in
  `DEPENDENCIES.md`.
- Review every custom node before installation; custom nodes execute Python in
  the ComfyUI process.
- Do not disable ComfyUI or repository secret scanning to install this package.
- Treat untrusted images and LoRAs as untrusted local inputs even with the
  included validation and safe-loading controls.

The published review covers only the files in this ZIP, not external nodes,
binary dependencies, model weights, the Civitai service, or the recipient's
ComfyUI deployment.
