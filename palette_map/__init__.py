# palette_map/__init__.py
"""
palette_map package.

Purpose:
  Utilities for remapping images to the wplace palette. See palette_map.py for CLI.

Public API:
  run_pixel: pixel-art recoloring entry point.
  colour_convert: color space transforms (rgb_to_lab, lab_to_lch, etc.).
  core_types: shared type aliases (U8Image, U8Mask, Lab, Lch, NameOf, PaletteItem).
  palette_lock: palette tier and gating helpers.
  palette_data: palette definitions and build_palette().
  colour_select: palette restriction helpers.
  PALETTE: optional raw palette constant if provided by palette_data.

Quick start:
  from palette_map import run_pixel
  from palette_map.colour_convert import rgb_to_lab, lab_to_lch
"""

__version__ = "0.2.1"

# Re-export namespaces for convenience.
from . import colour_convert
from . import core_types
from . import palette_lock
from . import palette_data
from . import colour_select
from . import utils

# Optional data constant.
try:
    from .palette_data import PALETTE as PALETTE
except Exception:
    PALETTE = None  # type: ignore[assignment]

# Pixel mode entry point. ImportError should surface immediately if missing.
from .pixel.run import run_pixel

__all__ = [
    "__version__",
    "colour_convert",
    "core_types",
    "palette_lock",
    "palette_data",
    "colour_select",
    "utils",
    "PALETTE",
    "run_pixel",
]
