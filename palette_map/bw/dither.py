# palette_map/bw/dither.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..color_convert import srgb_to_lab
from ..core_types import PaletteItem
from ..enforce import (
    is_palette_only,
    lock_to_palette_by_uniques,
    lock_to_palette_per_pixel,
    palette_set,
)
from ..palette_data import GREY_HEXES, build_palette

__all__ = ["select_bw_indices", "dither_bw", "run_bw"]


def select_bw_indices(pal_lch_mat: np.ndarray, chroma_max: float = 6.0) -> np.ndarray:
    chroma = pal_lch_mat[:, 1].astype(np.float32)
    idx = np.where(chroma <= float(chroma_max))[0]
    if idx.size == 0:
        lo = int(np.argmin(pal_lch_mat[:, 0]))
        hi = int(np.argmax(pal_lch_mat[:, 0]))
        idx = np.unique(np.array([lo, hi], dtype=int))
    return idx


def dither_bw(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
) -> np.ndarray:
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    lab_img = srgb_to_lab(img_rgb).reshape(H, W, 3)
    L_src = lab_img[..., 0].astype(np.float32)

    grey_idx = select_bw_indices(pal_lch_mat, chroma_max=6.0)
    L_grey = pal_lch_mat[grey_idx, 0].astype(np.float32)
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
            L_here = L_src[y, x] + errL[y, x]
            j = int(np.argmin(np.abs(L_grey - L_here)))
            out[y, x] = grey_rgb[j]
            eL = L_here - L_grey[j]
            for dx, dy, w in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

    return out


def run_bw(
    img_rgb: np.ndarray, alpha: np.ndarray, debug: bool = False
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], str]]:
    grey_pairs = [(hx, hx) for hx in GREY_HEXES]
    pal_items, name_of, pal_lab, pal_lch = build_palette(grey_pairs)

    out_rgb = lock_to_palette_by_uniques(
        img_rgb, alpha, pal_items, pal_lab.astype(np.float32)
    )
    if not is_palette_only(out_rgb, alpha, palette_set(pal_items)):
        out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, pal_items)

    if debug:
        used = len(select_bw_indices(pal_lch.astype(np.float32), chroma_max=6.0))
        print(f"[debug] bw snap  greys_used={used}")

    return out_rgb, name_of
