# palette_map/bw/dither.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..color_convert import rgb_to_lab
from ..core_types import (
    U8Image,
    U8Mask,
    Lab,
    Lch,
    PaletteItem,
    RGBTuple,
)
from ..palette_lock import (
    is_palette_only,
    lock_to_palette_by_uniques,
    lock_to_palette_per_pixel,
    palette_set,
)
from ..palette_data import GREY_HEXES, build_palette


# ---------- small threaded helper (precompute only) ---------------------------


def _rgb_to_lab_threaded(rgb: U8Image, workers: int) -> Lab:
    """Threaded RGB→Lab row-chunk conversion (returns float32 Lab)."""
    H = int(rgb.shape[0])
    if workers <= 1 or H < 256:
        return rgb_to_lab(rgb).astype(np.float32, copy=False)
    # split by rows
    parts = []
    step = (H + workers - 1) // workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for s in range(0, H, step):
            e = min(H, s + step)
            futs.append(ex.submit(rgb_to_lab, rgb[s:e]))
        for fu in futs:
            parts.append(fu.result().astype(np.float32, copy=False))
    return np.vstack(parts).astype(np.float32, copy=False)


# ---------- core utilities ----------------------------------------------------


def select_bw_indices(pal_lch_mat: Lch, chroma_max: float = 6.0) -> np.ndarray:
    """Return indices of near-neutral palette entries by chroma threshold."""
    chroma = pal_lch_mat[:, 1].astype(np.float32)
    idx = np.where(chroma <= float(chroma_max))[0]
    if idx.size == 0:
        lo = int(np.argmin(pal_lch_mat[:, 0]))
        hi = int(np.argmax(pal_lch_mat[:, 0]))
        idx = np.unique(np.array([lo, hi], dtype=int))
    return idx


def dither_bw(
    img_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    *,
    workers: int = 1,
    debug: bool = False,
) -> Tuple[U8Image, Optional[Dict]]:
    """
    Floyd–Steinberg diffusion in L* using only near-neutral palette entries.
    Returns (rgb_out, None) to match the wrapper's (image, meta) convention.
    """
    H, W, _ = img_rgb.shape
    out: U8Image = np.zeros_like(img_rgb, dtype=np.uint8)

    # Threaded RGB→Lab precompute
    lab_img = _rgb_to_lab_threaded(img_rgb, workers).reshape(H, W, 3)
    L_src = lab_img[..., 0].astype(np.float32, copy=False)

    grey_idx = select_bw_indices(pal_lch_mat, chroma_max=6.0)
    L_grey = pal_lch_mat[grey_idx, 0].astype(np.float32, copy=False)
    grey_rgb = np.array([palette[i].rgb for i in grey_idx], dtype=np.uint8)

    errL = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        serp_left = (y % 2) == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)
        nbrs = (
            ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
            if serp_left
            else ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        )
        for x in xs:
            if alpha[y, x] == 0:
                continue
            L_here = float(L_src[y, x] + errL[y, x])
            j = int(np.argmin(np.abs(L_grey - L_here)))
            out[y, x] = grey_rgb[j]

            eL = L_here - float(L_grey[j])
            for dx, dy, w in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * float(w)

    if debug:
        used = int(len(grey_idx))
        print(f"[debug] bw snap  workers={workers}  greys_used={used}", flush=True)

    return out, None


def run_bw(
    img_rgb: U8Image, alpha: U8Mask, *, debug: bool = False, workers: int = 1
) -> Tuple[U8Image, Dict[RGBTuple, str]]:
    """
    Snap any image to the built-in greyscale subpalette (no diffusion),
    enforcing exact palette colours across visible pixels.
    Returns (rgb_out, name_of_dict).
    """
    grey_pairs = [(hx, hx) for hx in GREY_HEXES]
    pal_items, name_of, pal_lab, pal_lch = build_palette(grey_pairs)

    out_rgb = lock_to_palette_by_uniques(
        img_rgb, alpha, pal_items, pal_lab.astype(np.float32, copy=False)
    )
    if not is_palette_only(out_rgb, alpha, palette_set(pal_items)):
        out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, pal_items)

    if debug:
        used = int(
            len(
                select_bw_indices(
                    pal_lch.astype(np.float32, copy=False), chroma_max=6.0
                )
            )
        )
        print(f"[debug] bw snap  workers={workers}  greys_used={used}", flush=True)

    return out_rgb, name_of


__all__ = ["select_bw_indices", "dither_bw", "run_bw"]
