# palette_map/bw/__init__.py
"""
Black & white (greys-only) remapping and dithering utilities.
"""

from .dither import dither_bw, run_bw, select_bw_indices

__all__ = ["select_bw_indices", "dither_bw", "run_bw"]
