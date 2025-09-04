# palette_map/palette_lock.py
from __future__ import annotations

from typing import Iterator, List, Optional, Set, Tuple, Dict

import numpy as np

from .color_convert import rgb_to_lab
from .core_types import RGBTuple, PaletteItem, U8Image, U8Mask, Lab


def palette_set(palette: List[PaletteItem]) -> Set[RGBTuple]:
    """Return a set of all RGB tuples present in the palette."""
    return {p.rgb for p in palette}


def _iter_visible_coords(alpha: U8Mask) -> Iterator[Tuple[int, int]]:
    """Yield (y, x) for all non-transparent pixels."""
    ys, xs = np.nonzero(alpha != 0)
    return zip(ys.tolist(), xs.tolist())


def is_palette_only(img_rgb: U8Image, alpha: U8Mask, pal_set: Set[RGBTuple]) -> bool:
    """True if every visible pixel is already in the palette."""
    h, w, _ = img_rgb.shape
    for y, x in _iter_visible_coords(alpha):
        t: RGBTuple = (
            int(img_rgb[y, x, 0]),
            int(img_rgb[y, x, 1]),
            int(img_rgb[y, x, 2]),
        )
        if t not in pal_set:
            return False
    return True


def count_off_palette_pixels(
    img_rgb: U8Image, alpha: U8Mask, palette: List[PaletteItem]
) -> int:
    """Count visible pixels whose RGB is not present in the palette."""
    pal = palette_set(palette)
    c = 0
    for y, x in _iter_visible_coords(alpha):
        t: RGBTuple = (
            int(img_rgb[y, x, 0]),
            int(img_rgb[y, x, 1]),
            int(img_rgb[y, x, 2]),
        )
        if t not in pal:
            c += 1
    return c


def lock_to_palette_by_uniques(
    out_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],
    pal_lab_mat: Lab,
) -> U8Image:
    """
    Snap colours to nearest palette entries, but only for unique colours
    that are off-palette (faster than per-pixel nearest).
    """
    H, W, _ = out_rgb.shape
    mask = alpha != 0
    samples = out_rgb[mask]
    if samples.size == 0:
        return out_rgb

    # Unique RGBs among visible pixels (fast via structured view)
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = samples.view(dt).reshape(-1)
    uniq = np.unique(flat)
    uniq_rgb = np.stack([uniq["r"], uniq["g"], uniq["b"]], axis=1).astype(np.uint8)

    # Filter to those not already in the palette
    pal_set = palette_set(palette)
    off_mask = np.array(
        [(int(r), int(g), int(b)) not in pal_set for r, g, b in uniq_rgb.tolist()],
        dtype=bool,
    )
    if not np.any(off_mask):
        return out_rgb

    off_np = uniq_rgb[off_mask]
    off_lab = rgb_to_lab(off_np).reshape(-1, 3).astype(np.float32)

    # Build mapping from offending RGB -> nearest palette RGB
    mapping: Dict[RGBTuple, RGBTuple] = {}
    pal_rgb_arr = np.array([p.rgb for p in palette], dtype=np.uint8)

    for i in range(off_lab.shape[0]):
        s_lab = off_lab[i]
        diff = pal_lab_mat - s_lab
        de2 = (diff * diff).sum(axis=1)
        best = int(np.argmin(de2))
        rep_rgb: RGBTuple = (
            int(pal_rgb_arr[best, 0]),
            int(pal_rgb_arr[best, 1]),
            int(pal_rgb_arr[best, 2]),
        )
        key_rgb: RGBTuple = (int(off_np[i, 0]), int(off_np[i, 1]), int(off_np[i, 2]))
        mapping[key_rgb] = rep_rgb

    # Apply mapping to visible pixels only
    out = out_rgb.copy()
    for y, x in _iter_visible_coords(alpha):
        key_rgb2: RGBTuple = (int(out[y, x, 0]), int(out[y, x, 1]), int(out[y, x, 2]))
        rep_opt: Optional[RGBTuple] = mapping.get(key_rgb2)
        if rep_opt is not None:
            out[y, x, 0] = np.uint8(rep_opt[0])
            out[y, x, 1] = np.uint8(rep_opt[1])
            out[y, x, 2] = np.uint8(rep_opt[2])

    return out


def lock_to_palette_per_pixel(
    out_rgb: U8Image, alpha: U8Mask, palette: List[PaletteItem]
) -> U8Image:
    """
    Force every visible pixel to the nearest palette entry in RGB space.
    This is a strict lock; it can remove dithering if used post-process.
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
    "count_off_palette_pixels",
]
