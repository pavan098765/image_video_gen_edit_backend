"""
Image upscaler bridge kept as part of multi-model worker structure.
"""
from PIL import Image


def upscale_image(image: Image.Image, scale_factor: int = 2) -> Image.Image:
    scale = max(1, int(scale_factor or 1))
    if scale == 1:
        return image
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.LANCZOS)

