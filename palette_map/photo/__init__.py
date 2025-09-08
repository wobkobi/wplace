# palette_map/photo/__init__.py
"""
Photo-mode API.

Provides:
  dither_photo(rgb, alpha, items, pal_lab, pal_lch, workers=..., progress=False)
    Map an RGBA image to the palette using error-diffusion dithering.
    Args:
      rgb: uint8 [H,W,3]
      alpha: uint8 [H,W]
      items: list[PaletteItem]
      pal_lab: Lab [P,3]
      pal_lch: Lch [P,3]
      workers: int, thread pool size
      progress: bool, print progress
    Returns:
      uint8 [H,W,3] mapped image.
"""

from .dither import dither_photo

__all__ = ["dither_photo"]
