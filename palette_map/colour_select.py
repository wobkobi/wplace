# palette_map/colour_select.py
from __future__ import annotations

"""
Palette selection helpers.

Exports:
  resolve_limit_once(colours, limit_opt) -> Optional[int]
  neutral_indices(pal_lch) -> np.ndarray
  greyish_sources(src_rgb, src_lch) -> np.ndarray
  restrict_palette(colours, img_rgb, alpha, pal_rgb_full, pal_lab_full, pal_lch_full, name_of, limit_opt, debug=False)
    -> (pal_rgb, pal_lab, pal_lch, sel_idx)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .core_types import U8Image, U8Mask, Lab, Lch, NameOf
from .colour_convert import rgb_to_lab
from .utils import unique_visible, nearest_palette_indices_lab

NEUTRAL_C_MAX = 5.0
DROP_RATIO = 0.55
COVERAGE_TARGET = 0.96
MIN_K = 2


def resolve_limit_once(colours: str, limit_opt: Optional[int]) -> Optional[int]:
    """
    Return the user-provided limit (if >=2) or None.
    No interactive prompting here; caller decides.
    """
    if colours != "limited":
        return None
    if limit_opt is None:
        return None
    return int(limit_opt) if limit_opt >= MIN_K else MIN_K


def neutral_indices(pal_lch: Lch) -> np.ndarray:
    """
    Build a neutral pool from a palette LCh array.
    Prefer chroma <= 10. If none, use the lowest-chroma quartile.
    Always include darkest and brightest entries to span lightness.
    """
    C = pal_lch[:, 1].astype(np.float32, copy=False)
    idx = np.where(C <= 10.0)[0]
    if idx.size == 0:
        q = float(np.quantile(C, 0.25))
        idx = np.where(C <= q)[0]
    if idx.size == 0:
        lo = int(np.argmin(pal_lch[:, 0]))
        hi = int(np.argmax(pal_lch[:, 0]))
        return np.unique(np.array([lo, hi], dtype=int))
    lo = int(np.argmin(pal_lch[:, 0]))
    hi = int(np.argmax(pal_lch[:, 0]))
    return np.unique(np.concatenate([idx, np.array([lo, hi], dtype=int)])).astype(int)


def greyish_sources(src_rgb: U8Image, src_lch: Lch) -> np.ndarray:
    """
    Detect grey-ish source colours among unique RGB rows.

    Rules:
      small RGB channel spread (<= 20)
      chroma <= 22
      hue near 0 or 180 unless chroma <= 12
    Returns:
      boolean mask array with length U (unique count)
    """
    r = src_rgb[:, 0].astype(np.int16)
    g = src_rgb[:, 1].astype(np.int16)
    b = src_rgb[:, 2].astype(np.int16)
    max_delta = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    C = src_lch[:, 1]
    H = src_lch[:, 2]
    d0 = np.minimum(H, 360.0 - H)
    d180 = np.minimum(np.abs(H - 180.0), 360.0 - np.abs(H - 180.0))
    near_neutral_axis = (np.minimum(d0, d180) <= 18.0) | (C <= 12.0)
    return (max_delta <= 20) & (C <= 22.0) & near_neutral_axis


def _auto_k_from_hist(counts_per_palette: np.ndarray) -> int:
    """
    Decide K from a usage histogram:
      1) first big drop where next/prev <= DROP_RATIO
      2) else the smallest K that reaches COVERAGE_TARGET of total weight
    """
    if counts_per_palette.size == 0:
        return MIN_K

    c_sorted = np.sort(counts_per_palette)[::-1]
    for i in range(len(c_sorted) - 1):
        a, b = float(c_sorted[i]), float(c_sorted[i + 1])
        if a <= 0:
            break
        if b / a <= DROP_RATIO and (i + 1) >= MIN_K:
            return i + 1

    total = float(c_sorted.sum())
    if total <= 0:
        return MIN_K
    cum = np.cumsum(c_sorted, dtype=np.float64)
    k_cov = int(np.searchsorted(cum, COVERAGE_TARGET * total, side="left")) + 1
    return max(MIN_K, min(k_cov, len(c_sorted)))


def _pick_top_indices_by_usage(counts_per_palette: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k palette bins by usage."""
    if counts_per_palette.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(counts_per_palette)[::-1]
    k = max(MIN_K, min(k, order.size))
    return order[:k].astype(np.int32, copy=False)


def restrict_palette(
    colours: str,
    img_rgb: U8Image,
    alpha: U8Mask,
    pal_rgb_full: U8Image,
    pal_lab_full: Lab,
    pal_lch_full: Lch,
    name_of: NameOf,
    limit_opt: Optional[int],
    debug: bool = False,
) -> Tuple[U8Image, Lab, Lch, np.ndarray]:
    """
    Return a restricted palette view (pal_rgb, pal_lab, pal_lch, sel_idx).

    colours:
      "full": no restriction
      "bw": neutrals only by LCh chroma, ensure extremes if empty
      "limited": choose top-K by usage. If user provided --limit, treat as an
                 upper bound and allow lowering K when a big drop is detected.
                 If no limit provided, choose K automatically.
    """
    P = pal_rgb_full.shape[0]

    if colours == "full":
        sel = np.arange(P, dtype=np.int32)
        return (
            pal_rgb_full[sel],
            pal_lab_full[sel].astype(np.float32, copy=False),
            pal_lch_full[sel].astype(np.float32, copy=False),
            sel,
        )

    if colours == "bw":
        mask = pal_lch_full[:, 1] <= NEUTRAL_C_MAX
        idx = np.where(mask)[0]
        if idx.size == 0:
            lo = int(np.argmin(pal_lch_full[:, 0]))
            hi = int(np.argmax(pal_lch_full[:, 0]))
            idx = np.unique(np.array([lo, hi], dtype=np.int32))
        sel = idx.astype(np.int32, copy=False)
        if debug:
            print(f"[debug] colours=bw  neutrals={sel.size}", flush=True)
        return (
            pal_rgb_full[sel],
            pal_lab_full[sel].astype(np.float32, copy=False),
            pal_lch_full[sel].astype(np.float32, copy=False),
            sel,
        )

    uniq_rgb, uniq_counts = unique_visible(img_rgb, alpha)
    if uniq_rgb.shape[0] == 0:
        lo = int(np.argmin(pal_lch_full[:, 0]))
        hi = int(np.argmax(pal_lch_full[:, 0]))
        sel = np.unique(np.array([lo, hi], dtype=np.int32))
        return (
            pal_rgb_full[sel],
            pal_lab_full[sel].astype(np.float32, copy=False),
            pal_lch_full[sel].astype(np.float32, copy=False),
            sel,
        )

    src_lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    nearest = nearest_palette_indices_lab(src_lab, pal_lab_full)  # (U,)
    counts_per_palette = np.bincount(
        nearest, weights=uniq_counts.astype(np.float64), minlength=P
    )

    k_auto = _auto_k_from_hist(counts_per_palette)
    if debug:
        print(f"[debug] limited: k_auto={k_auto}", flush=True)

    if limit_opt is not None:
        k_final = min(int(limit_opt), k_auto)
        if debug and k_final != limit_opt:
            print(
                f"[debug] limited: user limit={limit_opt} -> using {k_final} due to big drop",
                flush=True,
            )
    else:
        k_final = k_auto

    sel = _pick_top_indices_by_usage(counts_per_palette, k_final)

    if debug:
        order = np.argsort(counts_per_palette)[::-1]
        top = order[: min(12, order.size)]
        total = counts_per_palette.sum() or 1.0
        print(f"[debug] limited: selected K={sel.size}", flush=True)
        for j in top:
            rgb = pal_rgb_full[j]
            hhex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            share = counts_per_palette[j] / total
            nm = name_of.get(hhex, "?")
            flag = "*" if j in sel.tolist() else " "
            print(f"  {flag} {hhex}  {nm}: share={share:.1%}", flush=True)

    return (
        pal_rgb_full[sel],
        pal_lab_full[sel].astype(np.float32, copy=False),
        pal_lch_full[sel].astype(np.float32, copy=False),
        sel,
    )


__all__ = [
    "restrict_palette",
    "resolve_limit_once",
    "neutral_indices",
    "greyish_sources",
]
