"""Host-aware controls for the LTX-2.5 Smart Fast ComfyUI workflow."""

from __future__ import annotations

import hashlib
import math
import os
import warnings


MAX_SEED = (1 << 64) - 1
FPS = 24


def _snap(value: float, multiple: int = 64) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _dimensions(width: int, height: int, aspect_ratio: str) -> tuple[int, int]:
    """Keep the profile pixel budget while changing aspect ratio."""
    area = width * height
    ratios = {
        "16:9 Landscape": 16 / 9,
        "9:16 Portrait": 9 / 16,
        "1:1 Square": 1.0,
    }
    ratio = ratios[aspect_ratio]
    resolved_width = _snap(math.sqrt(area * ratio))
    resolved_height = _snap(math.sqrt(area / ratio))
    return resolved_width, resolved_height


def _detect_vram_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    except Exception:
        return None


def _resolve_host(requested: str) -> tuple[str, float | None]:
    detected = _detect_vram_gib()
    if requested in {
        "12GB Safe (experimental)",
        "12GB Decode Tiles (experimental)",
    }:
        return "12GB Safe", detected
    if requested == "16GB Fast":
        return "16GB Fast", detected
    if requested == "24GB+ Quality":
        return "24GB+ Quality", detected

    if detected is not None and detected >= 22:
        return "24GB+ Quality", detected
    if detected is not None and detected >= 14:
        return "16GB Fast", detected
    return "12GB Safe", detected


def _landscape_canvas(host: str, quality: str, duration: int) -> tuple[int, int]:
    quality_key = {
        "High Detail (Conv VAE)": "best",
        "Balanced": "balanced",
        "Draft": "draft",
    }[quality]

    if host == "24GB+ Quality":
        table = {
            "best": (1536, 896),
            "balanced": (1344, 768),
            "draft": (1152, 640),
        }
        if duration > 15:
            table = {
                "best": (1344, 768),
                "balanced": (1152, 640),
                "draft": (1024, 576),
            }
        return table[quality_key]

    if host == "16GB Fast":
        table = {
            "best": (1344, 768),
            "balanced": (1152, 640),
            "draft": (1024, 576),
        }
        if duration > 15:
            table = {
                "best": (1152, 640),
                "balanced": (1024, 576),
                "draft": (896, 512),
            }
        return table[quality_key]

    # 12GB is intentionally conservative and remains experimental until tested
    # on a physical 12GB card.
    if duration <= 5:
        table = {
            "best": (1024, 576),
            "balanced": (896, 512),
            "draft": (768, 448),
        }
    elif duration <= 10:
        table = {
            "best": (896, 512),
            "balanced": (832, 448),
            "draft": (768, 448),
        }
    elif duration <= 15:
        table = {
            "best": (896, 512),
            "balanced": (832, 448),
            "draft": (768, 448),
        }
    else:
        table = {
            "best": (832, 448),
            "balanced": (768, 448),
            "draft": (704, 384),
        }
    return table[quality_key]


def _decode_profile(host: str) -> tuple[int, int, int, int]:
    # 16/24GB values follow the official recommended table. The 12GB experiment
    # uses the official workflow's more conservative default temporal 64/8.
    if host == "24GB+ Quality":
        return 1024, 256, 128, 48
    if host == "16GB Fast":
        return 768, 192, 128, 32
    return 512, 64, 64, 8


VAE_DECODE_PRESETS = (
    "Auto (VRAM-based)",
    "Low VRAM (12GB target, experimental)",
    "Medium Tiles (16GB recommended)",
    "Large Tiles (24GB recommended)",
)


def _resolve_vae_decode_preset(requested: str) -> tuple[str, float | None, str]:
    """Resolve a decode-only preset while accepting the legacy profile labels."""

    detected = _detect_vram_gib()
    aliases = {
        "12GB Safe (experimental)": "12GB Safe",
        "12GB Decode Tiles (experimental)": "12GB Safe",
        VAE_DECODE_PRESETS[1]: "12GB Safe",
        "16GB Fast": "16GB Fast",
        VAE_DECODE_PRESETS[2]: "16GB Fast",
        "24GB+ Quality": "24GB+ Quality",
        VAE_DECODE_PRESETS[3]: "24GB+ Quality",
    }
    if requested in aliases:
        host = aliases[requested]
    elif requested in {"Auto (detect VRAM)", VAE_DECODE_PRESETS[0]}:
        if detected is not None and detected >= 22:
            host = "24GB+ Quality"
        elif detected is not None and detected >= 14:
            host = "16GB Fast"
        else:
            host = "12GB Safe"
    else:
        raise ValueError(f"Unsupported VAE decode preset: {requested}")

    label = {
        "12GB Safe": VAE_DECODE_PRESETS[1],
        "16GB Fast": VAE_DECODE_PRESETS[2],
        "24GB+ Quality": VAE_DECODE_PRESETS[3],
    }[host]
    return host, detected, label


class LTX25AutoProfile:
    """Resolve all technical canvas/decode values from a small creative control set."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gpu_profile": (
                    [
                        "Auto (detect VRAM)",
                        "16GB Fast",
                        "12GB Safe (experimental)",
                        "24GB+ Quality",
                    ],
                    {"default": "Auto (detect VRAM)"},
                ),
                "duration_seconds": (
                    "INT",
                    {"default": 15, "min": 3, "max": 20, "step": 1},
                ),
                "aspect_ratio": (
                    ["16:9 Landscape", "9:16 Portrait", "1:1 Square"],
                    {"default": "16:9 Landscape"},
                ),
                "quality": (
                    ["High Detail (Conv VAE)", "Balanced", "Draft"],
                    {"default": "High Detail (Conv VAE)"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "width",
        "height",
        "frames",
        "fps",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
        "stage1_seed",
        "stage2_seed",
        "profile_summary",
    )
    FUNCTION = "resolve"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "Automatically resolves canvas, frame count, tiled Conv VAE decode, and "
        "deterministic two-stage seeds for LTX-2.5. 12GB mode is experimental."
    )

    def resolve(
        self,
        gpu_profile: str,
        duration_seconds: int,
        aspect_ratio: str,
        quality: str,
        seed: int,
    ):
        host, detected = _resolve_host(gpu_profile)
        base_width, base_height = _landscape_canvas(host, quality, duration_seconds)
        width, height = _dimensions(base_width, base_height, aspect_ratio)
        frames = duration_seconds * FPS + 1
        tile_size, overlap, temporal_size, temporal_overlap = _decode_profile(host)
        stage1_seed = int(seed) & MAX_SEED
        stage2_seed = (stage1_seed + 1) & MAX_SEED
        detected_text = "unknown" if detected is None else f"{detected:.1f}GiB"
        support = "experimental" if host == "12GB Safe" else "host profile"
        summary = (
            f"{host} ({support}, detected={detected_text}) | {quality} | "
            f"{width}x{height} | {duration_seconds}s/{frames}f@{FPS} | "
            f"Conv VAE tiles {tile_size}/{overlap}/{temporal_size}/{temporal_overlap}"
        )
        return {
            "ui": {"text": [summary]},
            "result": (
                width,
                height,
                frames,
                float(FPS),
                tile_size,
                overlap,
                temporal_size,
                temporal_overlap,
                stage1_seed,
                stage2_seed,
                summary,
            ),
        }


class LTX25AllModesControls:
    """Resolve direct creator controls without silently changing their intent."""

    MODES = (
        "T2V (text only)",
        "I2V (first frame)",
        "FLF2V (first + last frame)",
    )
    TIMING_MODES = (
        "Duration drives frames",
        "Manual exact frames",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(cls.MODES), {"default": cls.MODES[1]}),
                "gpu_profile": (
                    [
                        "Auto (detect VRAM)",
                        "16GB Fast",
                        "12GB Decode Tiles (experimental)",
                        "24GB+ Quality",
                    ],
                    {"default": "Auto (detect VRAM)"},
                ),
                "final_width": (
                    "INT",
                    {"default": 1344, "min": 256, "max": 2048, "step": 64},
                ),
                "final_height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 2048, "step": 64},
                ),
                "timing_mode": (
                    list(cls.TIMING_MODES),
                    {"default": cls.TIMING_MODES[0]},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 30.0, "step": 1.0},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 15.0, "min": 1.0, "max": 30.0, "step": 0.5},
                ),
                "manual_frames": (
                    "INT",
                    {"default": 361, "min": 9, "max": 993, "step": 8},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
            }
        }


    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "final_width",
        "final_height",
        "stage1_width",
        "stage1_height",
        "frames",
        "fps",
        "effective_seconds",
        "need_first_frame",
        "need_last_frame",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
        "stage1_seed",
        "stage2_seed",
        "resolved_summary",
    )
    FUNCTION = "resolve"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "Single source of truth for T2V, I2V, and FLF2V mode, final canvas, "
        "duration/manual frame authority, decode tiles, and two-stage seeds."
    )

    @staticmethod
    def _require_multiple_of_64(name: str, value: int) -> int:
        value = int(value)
        if not 256 <= value <= 2048:
            raise ValueError(f"{name} must be between 256 and 2048, not {value}.")
        if value % 64 == 0:
            return value
        lower = max(256, (value // 64) * 64)
        upper = min(2048, lower + 64)
        raise ValueError(
            f"{name} must be a multiple of 64 for the half-resolution LTX stage. "
            f"Use {lower} or {upper}, not {value}."
        )

    @staticmethod
    def _manual_frames(value: int) -> int:
        value = int(value)
        if 9 <= value <= 993 and (value - 1) % 8 == 0:
            return value
        bounded = min(max(value, 9), 993)
        lower = max(9, ((bounded - 1) // 8) * 8 + 1)
        upper = min(993, lower if lower == value else lower + 8)
        raise ValueError(
            "manual_frames must be exactly 8*n+1 and at most 993 for LTX-2.5 audio. "
            f"Use {lower} or {upper}, not {value}."
        )

    def resolve(
        self,
        mode: str,
        gpu_profile: str,
        final_width: int,
        final_height: int,
        timing_mode: str,
        fps: float,
        duration_seconds: float,
        manual_frames: int,
        seed: int,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported LTX mode: {mode}")
        if timing_mode not in self.TIMING_MODES:
            raise ValueError(f"Unsupported timing mode: {timing_mode}")

        final_width = self._require_multiple_of_64("final_width", final_width)
        final_height = self._require_multiple_of_64("final_height", final_height)
        fps = float(fps)
        if not 1.0 <= fps <= 30.0:
            raise ValueError("fps must be between 1 and 30.")

        if timing_mode == self.TIMING_MODES[0]:
            duration_seconds = float(duration_seconds)
            if not 1.0 <= duration_seconds <= 30.0:
                raise ValueError("duration_seconds must be between 1 and 30.")
            frames = 1 + math.floor(fps * duration_seconds / 8.0) * 8
            if frames < 9:
                raise ValueError(
                    "The requested duration is too short for LTX-2.5. "
                    f"At {fps:g} fps, request at least {8.0 / fps:.3f} seconds."
                )
            if frames > 993:
                raise ValueError(
                    "The requested duration and FPS exceed the 993-frame LTX audio limit. "
                    "Reduce duration or FPS."
                )
            timing_summary = (
                f"Duration: requested {duration_seconds:g}s -> {frames}f "
                f"@{fps:g} = {(frames - 1) / fps:.3f}s"
            )
        else:
            frames = self._manual_frames(manual_frames)
            timing_summary = (
                f"Manual: {frames}f @{fps:g} = {(frames - 1) / fps:.3f}s "
                "(duration field ignored)"
            )

        host, detected = _resolve_host(gpu_profile)
        tile_size, overlap, temporal_size, temporal_overlap = _decode_profile(host)
        stage1_seed = int(seed) & MAX_SEED
        stage2_seed = (stage1_seed + 1) & MAX_SEED
        need_first = mode != self.MODES[0]
        need_last = mode == self.MODES[2]
        detected_text = "unknown" if detected is None else f"{detected:.1f}GiB"
        host_label = "12GB Decode Tiles" if host == "12GB Safe" else host
        support = "EXPERIMENTAL" if host == "12GB Safe" else "host profile"
        summary = (
            f"{mode} | FINAL {final_width}x{final_height} "
            f"(stage 1 {final_width // 2}x{final_height // 2}) | "
            f"{timing_summary} | {host_label} {support}, detected={detected_text} | "
            f"Conv VAE tiles {tile_size}/{overlap}/{temporal_size}/{temporal_overlap}. "
            "Canvas and timing are user-controlled and are not auto-clamped for VRAM."
        )
        return {
            "ui": {"text": [summary]},
            "result": (
                final_width,
                final_height,
                final_width // 2,
                final_height // 2,
                frames,
                fps,
                (frames - 1) / fps,
                need_first,
                need_last,
                tile_size,
                overlap,
                temporal_size,
                temporal_overlap,
                stage1_seed,
                stage2_seed,
                summary,
            ),
        }


class LTX25AllModesControlsV2(LTX25AllModesControls):
    """AllModes controls with an accurately named, decode-only VAE preset."""

    @classmethod
    def INPUT_TYPES(cls):
        legacy = super().INPUT_TYPES()["required"]
        required = {}
        for name, spec in legacy.items():
            if name == "gpu_profile":
                required["vae_decode_preset"] = (
                    list(VAE_DECODE_PRESETS),
                    {"default": VAE_DECODE_PRESETS[0]},
                )
            else:
                required[name] = spec
        return {"required": required}

    DESCRIPTION = (
        "Single source of truth for T2V, I2V, and FLF2V mode, final canvas, "
        "duration/manual frame authority, decode-only VAE tiling, and two-stage seeds. "
        "The VAE decode preset does not change generation quality."
    )

    def resolve(
        self,
        mode: str,
        vae_decode_preset: str,
        final_width: int,
        final_height: int,
        timing_mode: str,
        fps: float,
        duration_seconds: float,
        manual_frames: int,
        seed: int,
    ):
        host, detected, resolved_label = _resolve_vae_decode_preset(vae_decode_preset)
        legacy_label = {
            "12GB Safe": "12GB Decode Tiles (experimental)",
            "16GB Fast": "16GB Fast",
            "24GB+ Quality": "24GB+ Quality",
        }[host]
        output = super().resolve(
            mode,
            legacy_label,
            final_width,
            final_height,
            timing_mode,
            fps,
            duration_seconds,
            manual_frames,
            seed,
        )
        result = list(output["result"])
        old_summary = result[-1]
        summary_prefix = " | ".join(old_summary.split(" | ")[:3])
        detected_text = "unknown" if detected is None else f"{detected:.1f}GiB"
        selection = (
            f"Auto -> {resolved_label}"
            if vae_decode_preset in {"Auto (detect VRAM)", VAE_DECODE_PRESETS[0]}
            else resolved_label
        )
        tile_size, overlap, temporal_size, temporal_overlap = result[9:13]
        summary = (
            f"{summary_prefix} | VAE decode: {selection}, detected={detected_text} | "
            f"Conv VAE tiles {tile_size}/{overlap}/{temporal_size}/{temporal_overlap}. "
            "This preset changes decode tiling only, not model precision, sampling, or "
            "generation quality. Larger tiles can reduce tile count but use more VRAM. "
            "Canvas and timing remain user-controlled."
        )
        result[-1] = summary
        return {"ui": {"text": [summary]}, "result": tuple(result)}


class LTX25Stage2Mode:
    """Choose the Stage 2 attention path without invalidating Stage 1."""

    SPEED = "Speed — Sage + Sol (12–16GB target)"
    QUALITY = "Quality — Native exact (24GB+ recommended)"
    MODES = (SPEED, QUALITY)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage2_mode": (
                    list(cls.MODES),
                    {"default": cls.SPEED},
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("use_speed_attention",)
    FUNCTION = "resolve"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "Stage 2 generation path only. Speed reuses the Sage + Sol approximate "
        "attention path used by Stage 1. Quality uses native exact attention. "
        "24GB is a recommendation, not a no-offload guarantee; 1080p 15-second "
        "native attention can remain slow even on 24GB. This setting is independent "
        "of the VAE decode tile preset."
    )

    def resolve(self, stage2_mode: str):
        if stage2_mode not in self.MODES:
            raise ValueError(f"Unsupported Stage 2 mode: {stage2_mode}")
        return (stage2_mode == self.SPEED,)


class LTX25ModeImageLoader:
    """Portable mode-aware loader plus the official LTX image preparation."""

    # Empty is intentional: the Comfy frontend skips image preview requests for
    # falsey values while still presenting its native upload button.
    SENTINEL = ""
    MAX_SOURCE_BYTES = 256 * 1024 * 1024
    MAX_SOURCE_PIXELS = 80_000_000

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                # MODE is the sole authority. forceInput keeps this as a socket
                # and prevents a contradictory local IGNORED/USED widget.
                "enabled": ("BOOLEAN", {"forceInput": True}),
                "image": (
                    [cls.SENTINEL, *sorted(files)],
                    {"image_upload": True},
                ),
                "max_long_edge": (
                    "INT",
                    {"default": 1536, "min": 256, "max": 4096, "step": 64},
                ),
                "img_compression": (
                    "INT",
                    {"default": 18, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_or_none",)
    FUNCTION = "load"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "Portable image picker: disabled slots never validate or read a file. Enabled "
        "single-frame images pass bounded preflight checks before the official "
        "long-edge resize and LTXVPreprocess path."
    )

    @staticmethod
    def _validate_controls(max_long_edge, img_compression) -> tuple[int, int]:
        max_long_edge = int(max_long_edge)
        img_compression = int(img_compression)
        if not 256 <= max_long_edge <= 4096:
            raise ValueError("max_long_edge must be between 256 and 4096.")
        if not 0 <= img_compression <= 100:
            raise ValueError("img_compression must be between 0 and 100.")
        return max_long_edge, img_compression

    @classmethod
    def _resolve_input_path(cls, image: str) -> str:
        import folder_paths

        if not folder_paths.exists_annotated_filepath(image):
            raise ValueError(
                f"The selected image is not available on this PC: {image!r}. "
                "Choose or upload a replacement in this node."
            )
        path = os.path.realpath(folder_paths.get_annotated_filepath(image))
        input_root = os.path.realpath(folder_paths.get_input_directory())
        try:
            contained = os.path.commonpath((input_root, path)) == input_root
        except ValueError:
            contained = False
        if not contained:
            raise ValueError(
                "LTX mode images must be inside ComfyUI's input directory. "
                "Output/temp annotations and external paths are not accepted."
            )
        return path

    @classmethod
    def _preflight_path(cls, path: str) -> tuple[int, int, int]:
        from PIL import Image, UnidentifiedImageError

        try:
            encoded_bytes = os.path.getsize(path)
        except OSError as exc:
            raise ValueError("The selected image cannot be inspected safely.") from exc
        if encoded_bytes <= 0 or encoded_bytes > cls.MAX_SOURCE_BYTES:
            raise ValueError(
                "The selected image must be non-empty and no larger than 256 MiB."
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as source:
                    width, height = map(int, source.size)
                    frames = int(getattr(source, "n_frames", 1))
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise ValueError("The selected image exceeds the safe decoded-pixel limit.") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("The selected file is not a supported still image.") from exc

        if width <= 0 or height <= 0 or width * height > cls.MAX_SOURCE_PIXELS:
            raise ValueError(
                "The selected image must have positive dimensions and no more than "
                "80,000,000 decoded pixels."
            )
        if frames != 1:
            raise ValueError(
                "Animated or multi-frame images are not accepted as LTX guide frames. "
                "Export one still PNG, JPEG, or WebP frame first."
            )
        return encoded_bytes, width, height

    @classmethod
    def _preflight_image(cls, image: str) -> str:
        path = cls._resolve_input_path(image)
        cls._preflight_path(path)
        return path

    @staticmethod
    def _stream_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        # The selected filename may be stale on another PC while MODE disables
        # this slot. Runtime validation below is mode-aware; prompt validation
        # must therefore not reject the portable empty/stale filename first.
        return True

    @classmethod
    def IS_CHANGED(cls, enabled, image, max_long_edge, img_compression):
        max_long_edge, img_compression = cls._validate_controls(
            max_long_edge, img_compression
        )
        if not enabled:
            return f"disabled:{max_long_edge}:{img_compression}"
        if image == cls.SENTINEL:
            return f"missing:{max_long_edge}:{img_compression}"
        path = cls._preflight_image(image)
        return f"{cls._stream_sha256(path)}:{max_long_edge}:{img_compression}"

    def load(self, enabled, image, max_long_edge, img_compression):
        max_long_edge, img_compression = self._validate_controls(
            max_long_edge, img_compression
        )
        if not enabled:
            return (None,)
        if image == self.SENTINEL:
            raise ValueError(
                "The selected LTX mode requires this image. Choose or upload it in this node."
            )
        import nodes
        self._preflight_image(image)
        image, _mask = nodes.LoadImage().load_image(image)
        import comfy.utils
        from comfy_extras.nodes_lt import LTXVPreprocess

        height, width = int(image.shape[1]), int(image.shape[2])
        longest = max(height, width)
        if longest > max_long_edge:
            scale = float(max_long_edge) / float(longest)
            resized_width = max(2, int(round(width * scale / 2.0)) * 2)
            resized_height = max(2, int(round(height * scale / 2.0)) * 2)
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                resized_width,
                resized_height,
                "lanczos",
                "disabled",
            ).movedim(1, -1)

        output = LTXVPreprocess.execute(
            image=image,
            img_compression=img_compression,
        )
        return (output.result[0],)


class LTX25ApplyGuides:
    """Apply the official I2V or FLF2V path based on available mode images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent_with_guides")
    FUNCTION = "apply"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "FIRST only uses core LTXVImgToVideoInplace. FIRST+LAST uses core "
        "LTXVAddGuide at frame 0 and -1. Keep LTXVCropGuides after sampling."
    )

    def apply(
        self,
        positive,
        negative,
        vae,
        latent,
        strength,
        first_image=None,
        last_image=None,
    ):
        from comfy_extras.nodes_lt import LTXVAddGuide, LTXVImgToVideoInplace

        if last_image is None:
            if first_image is None:
                return (positive, negative, latent)
            output = LTXVImgToVideoInplace.execute(
                vae=vae,
                image=first_image,
                latent=latent,
                strength=float(strength),
                bypass=False,
            )
            return (positive, negative, output.result[0])

        if first_image is None:
            raise ValueError("FLF2V requires both FIRST and LAST images.")

        output = LTXVAddGuide.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            latent=latent,
            image=first_image,
            frame_idx=0,
            strength=float(strength),
        )
        positive, negative, latent = output.result

        output = LTXVAddGuide.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            latent=latent,
            image=last_image,
            frame_idx=-1,
            strength=float(strength),
        )
        positive, negative, latent = output.result

        return (positive, negative, latent)


class LTX25OptionalBypassLoRA:
    """Apply one ordinary LTX transformer LoRA without rewriting quantized weights."""

    NONE = "None (base INT8 model)"

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        loras = [name for name in folder_paths.get_filename_list("loras")]
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    [cls.NONE, *loras],
                    {"default": cls.NONE},
                ),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "LTX-2.5/Smart Workflow"
    DESCRIPTION = (
        "Optional MODEL-only LoRA for the prequantized LTX INT8 transformer. "
        "None is a true pass-through. Selected adapters use ComfyUI's bypass-LoRA "
        "path so the base ConvRot weights stay quantized. Ordinary LTX transformer "
        "LoRAs only; IC-LoRAs need their dedicated guide workflow."
    )

    def apply(self, model, lora_name: str, strength_model: float):
        strength_model = float(strength_model)
        if lora_name == self.NONE or strength_model == 0.0:
            return (model,)

        import comfy.sd
        import comfy.utils
        import folder_paths

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, _clip = comfy.sd.load_bypass_lora_for_models(
            model,
            None,
            lora,
            strength_model,
            0.0,
        )
        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "LTX25AutoProfile": LTX25AutoProfile,
    "LTX25AllModesControls": LTX25AllModesControls,
    "LTX25AllModesControlsV2": LTX25AllModesControlsV2,
    "LTX25Stage2Mode": LTX25Stage2Mode,
    "LTX25ModeImageLoader": LTX25ModeImageLoader,
    "LTX25ApplyGuides": LTX25ApplyGuides,
    "LTX25OptionalBypassLoRA": LTX25OptionalBypassLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25AutoProfile": "LTX-2.5 Smart Auto Profile (12/16/24GB)",
    "LTX25AllModesControls": "LTX-2.5 AllModes Controls",
    "LTX25AllModesControlsV2": "LTX-2.5 AllModes Controls V2 (VAE Decode Preset)",
    "LTX25Stage2Mode": "LTX-2.5 Stage 2 Speed / Quality Mode",
    "LTX25ModeImageLoader": "LTX-2.5 Mode Image Loader",
    "LTX25ApplyGuides": "LTX-2.5 Apply FIRST / LAST Guides",
    "LTX25OptionalBypassLoRA": "LTX-2.5 Optional INT8 LoRA",
}
