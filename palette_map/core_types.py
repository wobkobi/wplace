# palette_map/core_types.py
from __future__ import annotations

"""
Core type aliases, small value objects, and lightweight helpers.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# basic aliases
RGBTuple = Tuple[int, int, int]
HexStr = str

U8Image = NDArray[np.uint8]  # e.g., (H, W, 3)
U8Mask = NDArray[np.uint8]  # e.g., (H, W)
Lab = NDArray[np.float32]  # (..., 3) CIE Lab
Lch = NDArray[np.float32]  # (..., 3) CIE LCh

# collections and algorithm helpers
CandidatesRow = List[Tuple[float, int]]  # [(cost, palette_index), ...]
CostLookup = Dict[Tuple[int, int], float]  # (src_idx, pal_idx) -> cost
Assignment = Dict[int, int]  # src_idx -> pal_idx
RGBToRGBMap = Dict[RGBTuple, RGBTuple]  # exact RGB mapping
NameOf = Mapping[HexStr, str]  # "#rrggbb" -> human-readable name


@dataclass(frozen=True)
class PaletteItem:
    """Palette entry with precomputed Lab and LCh rows."""

    rgb: RGBTuple
    name: str
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


@dataclass(frozen=True)
class SourceItem:
    """Unique source colour with count and precomputed Lab and LCh rows."""

    rgb: RGBTuple
    count: int
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


@dataclass(frozen=True)
class HueRange:
    """Closed hue interval in degrees with wrap-around semantics."""

    lo: float  # [0, 360)
    hi: float  # [0, 360)

    def contains(self, h: float) -> bool:
        h = float(h) % 360.0
        lo = float(self.lo) % 360.0
        hi = float(self.hi) % 360.0
        if lo <= hi:
            return lo <= h <= hi
        return h >= lo or h <= hi


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def hue_diff_deg(a: float, b: float) -> float:
    """Minimal absolute difference between two hues in degrees (0..180]."""
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def hue_in(h: float, lo: float, hi: float) -> bool:
    """Convenience wrapper around HueRange.contains()."""
    return HueRange(lo, hi).contains(h)


def rgb_to_hex(rgb: RGBTuple) -> HexStr:
    """RGB tuple to lowercase hex string '#rrggbb'."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(s: str) -> RGBTuple:
    """Parse '#rgb' or '#rrggbb' (case-insensitive) into an RGB tuple."""
    s = s.strip().lower()
    if not s.startswith("#"):
        raise ValueError("hex must start with '#'")
    if len(s) == 4:
        r, g, b = s[1], s[2], s[3]
        s = f"#{r}{r}{g}{g}{b}{b}"
    if len(s) != 7:
        raise ValueError("hex must be '#rrggbb' or '#rgb'")
    return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))


def hex_list_to_u8_rgb(hex_list: Sequence[str]) -> U8Image:
    """
    Convert a sequence of hex strings ('#rrggbb' or 'rrggbb') to a (N,3) uint8 array.
    Uses hex_to_rgb for a single source of truth.
    """
    arr = np.empty((len(hex_list), 3), dtype=np.uint8)
    for i, hx in enumerate(hex_list):
        r, g, b = hex_to_rgb(hx if hx.startswith("#") else f"#{hx}")
        arr[i, 0] = r
        arr[i, 1] = g
        arr[i, 2] = b
    return arr


def ensure_rgb_tuple(x: Union[Sequence[int], NDArray[np.generic]]) -> RGBTuple:
    """
    Coerce a 3-length sequence or array to an (int, int, int) RGB tuple.
    Helpful when extracting values from NumPy rows.
    """
    if isinstance(x, np.ndarray):
        if x.size < 3:
            raise ValueError("array too small for RGB")
        return (int(x[..., 0]), int(x[..., 1]), int(x[..., 2]))
    if len(x) < 3:
        raise ValueError("sequence too small for RGB")
    return (int(x[0]), int(x[1]), int(x[2]))


def assert_u8_image(img: np.ndarray) -> U8Image:
    """Validate a uint8 (H,W,3 or 4) image and return it typed as U8Image."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] < 3:
        raise TypeError("expected uint8 (H,W,3/4) image")
    return img  # type: ignore[return-value]


def assert_u8_mask(mask: np.ndarray) -> U8Mask:
    """Validate a uint8 (H,W) mask and return it typed as U8Mask."""
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise TypeError("expected uint8 (H,W) mask")
    return mask  # type: ignore[return-value]


PixelMapper = Callable[
    [U8Image, U8Mask, NDArray[np.uint8], Lab, Lch, bool],
    Union[U8Image, Tuple[U8Image, Mapping[RGBTuple, str]]],
]

PhotoMapper = Callable[
    [U8Image, U8Mask, List[PaletteItem], Lab, Lch, int],
    U8Image,
]

BWMapper = Callable[
    [U8Image, U8Mask, NDArray[np.uint8], Lab, Lch, int],
    U8Image,
]


__all__ = [
    "RGBTuple",
    "HexStr",
    "U8Image",
    "U8Mask",
    "Lab",
    "Lch",
    "CandidatesRow",
    "CostLookup",
    "Assignment",
    "RGBToRGBMap",
    "NameOf",
    "PaletteItem",
    "SourceItem",
    "HueRange",
    "clamp",
    "hue_diff_deg",
    "hue_in",
    "rgb_to_hex",
    "hex_to_rgb",
    "hex_list_to_u8_rgb",
    "ensure_rgb_tuple",
    "assert_u8_image",
    "assert_u8_mask",
    "PixelMapper",
    "PhotoMapper",
    "BWMapper",
]
