# palette_map/__init__.py
from __future__ import annotations

__version__ = "0.1.0"

# Core data & builders
from .palette_data import PALETTE, GREY_HEXES, build_palette

# Types
from .core_types import PaletteItem, SourceItem

# Colour conversion utilities
from .color_convert import srgb_to_lab_batch, lab_to_lch_batch
from .analysis import ciede2000_pair

# Image I/O
from .image_io import (
    load_image_rgba,
    save_image_rgba,
    resize_rgba_height,
    binarise_alpha,
    is_image_file,
)

# Mode helpers
from .mode import decide_auto_mode

# Enforcers (palette locking)
from .enforce import (
    lock_to_palette_by_uniques,
    lock_to_palette_per_pixel,
    is_palette_only,
)

# BW / Photo / Pixel entry points
from .bw import dither_bw
from .photo import dither_photo
from .pixel import run_pixel

__all__ = [
    "__version__",
    # data
    "PALETTE",
    "GREY_HEXES",
    "build_palette",
    # types
    "PaletteItem",
    "SourceItem",
    # color utils
    "srgb_to_lab_batch",
    "lab_to_lch_batch",
    "ciede2000_pair",
    # io
    "load_image_rgba",
    "save_image_rgba",
    "resize_rgba_height",
    "binarise_alpha",
    "is_image_file",
    # mode
    "decide_auto_mode",
    # enforce
    "lock_to_palette_by_uniques",
    "lock_to_palette_per_pixel",
    "is_palette_only",
    # algorithms
    "dither_bw",
    "dither_photo",
    "run_pixel",
]
