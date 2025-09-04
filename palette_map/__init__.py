# palette_map/__init__.py
"""
palette_map package.

Exports:
- run_pixel: refined pixel-mode mapper
- analysis, color_convert, core_types, enforce, palette_data modules
- PALETTE (if defined in palette_data)

Usage:
    from palette_map import run_pixel
    from palette_map.color_convert import lab_to_lch, rgb_to_lab
"""

__version__ = "0.1.1"

# Core namespaces
from . import color_convert as color_convert
from . import core_types as core_types
from . import palette_lock as palette_lock
from . import palette_data as palette_data

# Data
try:
    from .palette_data import PALETTE as PALETTE  # type: ignore
except Exception:
    PALETTE = None  # type: ignore[assignment]

# Pixel mode entry (no stub; fail fast if missing)
from .pixel.run import run_pixel  # type: ignore

__all__ = [
    "__version__",
    "color_convert",
    "core_types",
    "palette_lock",
    "palette_data",
    "PALETTE",
    "run_pixel",
]
