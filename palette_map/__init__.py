# palette_map/__init__.py
"""
palette_map package.

Exports:
- run_pixel: refined pixel-mode mapper
- bw: black/white helpers (dither_bw, run_bw_auto, selectors)
- colour_convert, core_types, palette_lock, palette_data, colour_metrics
- PALETTE (if defined in palette_data)

Usage:
    from palette_map import run_pixel
    from palette_map.colour_convert import lab_to_lch, rgb_to_lab
"""

__version__ = "0.2.1"

# Core namespaces
from . import colour_convert 
from . import core_types 
from . import palette_lock 
from . import palette_data 
from . import colour_select


# Data
try:
    from .palette_data import PALETTE as PALETTE
except Exception:
    PALETTE = None

# Pixel mode entry (no stub; fail fast if missing)
from .pixel.run import run_pixel

__all__ = [
    "__version__",
    "colour_convert",
    "core_types",
    "palette_lock",
    "palette_data",
    "colour_select",
    "PALETTE",
    "run_pixel",
]
