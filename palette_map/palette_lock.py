# palette_map/palette_lock.py
from __future__ import annotations

"""
Palette lock helpers.

Functions:
  palette_set(palette) -> set of RGB tuples
  is_palette_only(img_rgb, alpha, pal_set) -> bool
  lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat) -> U8Image
  lock_to_palette_per_pixel(out_rgb, alpha, palette) -> U8Image

Use cases:
  - palette_set + is_palette_only: quick guard to skip remapping
  - lock_to_palette_by_uniques: snap only off-palette uniques to nearest entries
  - lock_to_palette_per_pixel: force every visible pixel to nearest palette RGB
"""

from typing import Iterator, List, Optional, Set, Tuple, Dict

import numpy as np

from .colour_convert import rgb_to_lab
from .core_types import RGBTuple, PaletteItem, U8Image, U8Mask, Lab
from .utils import nearest_palette_indices_lab_distance


def palette_set(palette: List[PaletteItem]) -> Set[RGBTuple]:
    """Return a set of all RGB tuples present in the palette."""
    return {p.rgb for p in palette}


def _iter_visible_coords(alpha: U8Mask) -> Iterator[Tuple[int, int]]:
    """Yield (y, x) for all non-transparent pixels."""
    ys, xs = np.nonzero(alpha != 0)
    return zip(ys.tolist(), xs.tolist())


def is_palette_only(img_rgb: U8Image, alpha: U8Mask, pal_set: Set[RGBTuple]) -> bool:
    """True if every visible pixel is already in the palette."""
    for y, x in _iter_visible_coords(alpha):
        t: RGBTuple = (
            int(img_rgb[y, x, 0]),
            int(img_rgb[y, x, 1]),
            int(img_rgb[y, x, 2]),
        )
        if t not in pal_set:
            return False
    return True


def lock_to_palette_by_uniques(
    out_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],
    pal_lab_mat: Lab,
) -> U8Image:
    """
    Snap colours to nearest palette entries, but only for unique colours
    that are off-palette. Faster than per-pixel nearest while preserving
    any pixels that already match the palette exactly.

    Args:
      out_rgb: uint8 [H,W,3] image to be snapped
      alpha: uint8 [H,W] visibility mask
      palette: list of PaletteItem, used to emit RGB values
      pal_lab_mat: float32 [P,3] Lab rows for the same palette
    Returns:
      uint8 [H,W,3] image with off-palette uniques replaced by nearest palette RGB
    """
    mask = alpha != 0
    samples = out_rgb[mask]
    if samples.size == 0:
        return out_rgb

    # Unique RGBs among visible pixels
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = samples.view(dt).reshape(-1)
    uniq = np.unique(flat)
    uniq_rgb = np.stack([uniq["r"], uniq["g"], uniq["b"]], axis=1).astype(np.uint8)

    pal_set = palette_set(palette)
    off_mask = np.array(
        [(int(r), int(g), int(b)) not in pal_set for r, g, b in uniq_rgb.tolist()],
        dtype=bool,
    )
    if not np.any(off_mask):
        return out_rgb

    off_np = uniq_rgb[off_mask]
    off_lab = rgb_to_lab(off_np).reshape(-1, 3).astype(np.float32)

    # Map each offending unique RGB to its nearest palette RGB
    pal_rgb_arr = np.array([p.rgb for p in palette], dtype=np.uint8)
    nearest = nearest_palette_indices_lab_distance(off_lab, pal_lab_mat)  # [M]
    mapping: Dict[RGBTuple, RGBTuple] = {}
    for i in range(off_np.shape[0]):
        key_rgb: RGBTuple = (int(off_np[i, 0]), int(off_np[i, 1]), int(off_np[i, 2]))
        j = int(nearest[i])
        rep_rgb: RGBTuple = (
            int(pal_rgb_arr[j, 0]),
            int(pal_rgb_arr[j, 1]),
            int(pal_rgb_arr[j, 2]),
        )
        mapping[key_rgb] = rep_rgb

    # Apply mapping to visible pixels only
    out = out_rgb.copy()
    for y, x in _iter_visible_coords(alpha):
        k: RGBTuple = (int(out[y, x, 0]), int(out[y, x, 1]), int(out[y, x, 2]))
        rep: Optional[RGBTuple] = mapping.get(k)
        if rep is not None:
            out[y, x, 0] = np.uint8(rep[0])
            out[y, x, 1] = np.uint8(rep[1])
            out[y, x, 2] = np.uint8(rep[2])

    return out


def lock_to_palette_per_pixel(
    out_rgb: U8Image, alpha: U8Mask, palette: List[PaletteItem]
) -> U8Image:
    """
    Force every visible pixel to the nearest palette entry in RGB space.
    This is a strict lock and can collapse dithering. Use with care.

    Args:
      out_rgb: uint8 [H,W,3]
      alpha: uint8 [H,W]
      palette: list of PaletteItem
    Returns:
      uint8 [H,W,3] image
    """
    out = out_rgb.copy()
    mask = alpha != 0
    if not np.any(mask):
        return out

    pal_rgb = np.array([p.rgb for p in palette], dtype=np.int16)
    coords = np.argwhere(mask)
    chunk = 200_000

    for i in range(0, coords.shape[0], chunk):
        sl = coords[i : i + chunk]
        pts = out[sl[:, 0], sl[:, 1]].astype(np.int16)
        diff = pts[:, None, :] - pal_rgb[None, :, :]
        de2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(de2, axis=1)
        mapped = pal_rgb[idx].astype(np.uint8)
        out[sl[:, 0], sl[:, 1]] = mapped

    return out


__all__ = [
    "palette_set",
    "is_palette_only",
    "lock_to_palette_by_uniques",
    "lock_to_palette_per_pixel",
]
