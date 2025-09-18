# palette_map/__init__.py
"""
palette_map package.

Purpose:
  Utilities for remapping images to the wplace palette. See palette_map.py for CLI.

Public API:
  run_pixel     : pixel-art recolouring entry point.
  dither_photo  : photo-mode recolouring with error-diffusion.
  colour_convert: colour space transforms (rgb_to_lab, lab_to_lch, etc.).
  core_types    : shared type aliases (U8Image, U8Mask, Lab, Lch, NameOf, PaletteItem).
  palette_data  : palette definitions and build helpers.
  colour_select : palette restriction helpers.
  utils         : shared helpers (I/O, math, logging).
  PALETTE       : optional raw palette constant if provided by palette_data.

Quick start:
  from palette_map import run_pixel, dither_photo
  from palette_map.colour_convert import rgb_to_lab, lab_to_lch
"""

__version__ = "0.3.0"

# Re-export namespaces for convenience.
from . import colour_convert
from . import core_types
from . import palette_data
from . import colour_select
from . import utils
from . import pixel
from . import photo

# Optional data constant.
try:
    from .palette_data import PALETTE as PALETTE  # noqa: F401
except Exception:
    PALETTE = None  # type: ignore[assignment]

# Mode entry points. ImportError should surface immediately if missing.
from .pixel.run import run_pixel  # noqa: E402,F401
from .photo.dither import dither_photo  # noqa: E402,F401

__all__ = [
    "__version__",
    "colour_convert",
    "core_types",
    "palette_data",
    "colour_select",
    "utils",
    "pixel",
    "photo",
    "PALETTE",
    "run_pixel",
    "dither_photo",
]
