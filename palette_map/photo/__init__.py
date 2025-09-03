# palette_map/photo/__init__.py
"""
Photo mode: error-diffusion dithering guided by Lab/LCh cost model.
"""

from .dither import dither_photo

__all__ = ["dither_photo"]
