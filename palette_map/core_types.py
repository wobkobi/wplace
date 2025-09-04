# palette_map/core_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

# ---- basic aliases
RGBTuple = Tuple[int, int, int]
HexStr = str

U8Image = NDArray[np.uint8]  # e.g., (H, W, 3)
U8Mask = NDArray[np.uint8]  # e.g., (H, W)
Lab = NDArray[np.float32]  # (..., 3) in CIE Lab
Lch = NDArray[np.float32]  # (..., 3) in CIE LCh

# ---- collections / algorithm aliases
CandidatesRow = List[Tuple[float, int]]  # [(cost, palette_index), ...]
CostLookup = Dict[Tuple[int, int], float]  # (src_idx, pal_idx) -> cost
Assignment = Dict[int, int]  # src_idx -> pal_idx
RGBToRGBMap = Dict[RGBTuple, RGBTuple]  # exact RGB mapping
NameOf = Mapping[HexStr, str]  # "#rrggbb" -> human name


# ---- core items
@dataclass(frozen=True)
class PaletteItem:
    """A palette entry with precomputed Lab/LCh."""

    rgb: RGBTuple
    name: str
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


@dataclass(frozen=True)
class SourceItem:
    """A unique source colour with count and precomputed Lab/LCh."""

    rgb: RGBTuple
    count: int
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


# ---- light utility value objects
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
        # wrap-around (e.g., 340..20)
        return h >= lo or h <= hi


# ---- helpers (tiny and dependency-free)
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def hue_diff_deg(a: float, b: float) -> float:
    """Minimal absolute difference between two hues in degrees (0..180]."""
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def hue_in(h: float, lo: float, hi: float) -> bool:
    """Convenience wrapper around HueRange.contains()."""
    return HueRange(lo, hi).contains(h)


def rgb_to_hex(rgb: RGBTuple) -> HexStr:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(s: str) -> RGBTuple:
    """Accepts '#rgb' or '#rrggbb' (case-insensitive)."""
    s = s.strip().lower()
    if not s.startswith("#"):
        raise ValueError("hex must start with '#'")
    if len(s) == 4:  # #rgb -> #rrggbb
        r, g, b = s[1], s[2], s[3]
        s = f"#{r}{r}{g}{g}{b}{b}"
    if len(s) != 7:
        raise ValueError("hex must be '#rrggbb' or '#rgb'")
    return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))


def ensure_rgb_tuple(x: Union[Sequence[int], NDArray[np.generic]]) -> RGBTuple:
    """
    Coerce a 3-length sequence/array to (int,int,int).
    Helpful for silencing type checkers when extracting from NumPy rows.
    """
    if isinstance(x, np.ndarray):
        if x.size < 3:
            raise ValueError("array too small for RGB")
        return (int(x[..., 0]), int(x[..., 1]), int(x[..., 2]))
    if len(x) < 3:
        raise ValueError("sequence too small for RGB")
    return (int(x[0]), int(x[1]), int(x[2]))


# Small guard/validators (no heavy logic; keep runtime cost near-zero)
def assert_u8_image(img: np.ndarray) -> U8Image:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] < 3:
        raise TypeError("expected uint8 (H,W,3/4) image")
    return img  # type: ignore[return-value]


def assert_u8_mask(mask: np.ndarray) -> U8Mask:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise TypeError("expected uint8 (H,W) mask")
    return mask  # type: ignore[return-value]


# ---- lightweight function type aliases (for clarity, optional to use)
PixelMapper = Callable[
    [U8Image, U8Mask, NDArray[np.uint8], Lab, Lch, bool],
    Union[U8Image, Tuple[U8Image, Mapping[RGBTuple, str]]],
]

PhotoMapper = Callable[
    [U8Image, U8Mask, NDArray[np.uint8], Lab, Lch, int],
    U8Image,
]

BWMapper = Callable[
    [U8Image, U8Mask, NDArray[np.uint8], Lab, Lch, int],
    U8Image,
]


__all__ = [
    # arrays & primitives
    "RGBTuple",
    "HexStr",
    "U8Image",
    "U8Mask",
    "Lab",
    "Lch",
    # core items
    "PaletteItem",
    "SourceItem",
    # alg aliases
    "CandidatesRow",
    "CostLookup",
    "Assignment",
    "RGBToRGBMap",
    "NameOf",
    # value objects
    "HueRange",
    # helpers
    "clamp",
    "hue_diff_deg",
    "hue_in",
    "rgb_to_hex",
    "hex_to_rgb",
    "ensure_rgb_tuple",
    "assert_u8_image",
    "assert_u8_mask",
    # callable types
    "PixelMapper",
    "PhotoMapper",
    "BWMapper",
]
