from __future__ import annotations

"""
Palette selection helpers.

Exports:
  resolve_limit_once(colours_mode, limit_opt) -> Optional[int]
  neutral_indices(pal_lch) -> np.ndarray
  greyish_sources(src_rgb, src_lch) -> np.ndarray
  restrict_palette(colours_mode, img_rgb, alpha, pal_rgb_full, pal_lab_full, pal_lch_full, name_of, limit_opt, debug=False)
    -> (pal_rgb, pal_lab, pal_lch, sel_idx)
"""

from typing import Optional, Tuple

import numpy as np

from .core_types import Lab, Lch, NameOf, U8Image, U8Mask
from .colour_convert import rgb_to_lab
from .utils import (
    unique_visible_rgb,
    nearest_palette_indices_lab_distance,
    debug_log,
    key_value_pairs_to_string,
)

# Heuristics / knobs
NEUTRAL_C_MAX = 5.0
DROP_RATIO = 0.55
COVERAGE_TARGET = 0.96
MIN_K = 2


def resolve_limit_once(colours_mode: str, limit_opt: Optional[int]) -> Optional[int]:
    """
    Return the user-provided limit (if >= MIN_K) or None.
    No interactive prompting here; caller decides.
    """
    if colours_mode != "limited":
        return None
    if limit_opt is None:
        return None
    return int(limit_opt) if int(limit_opt) >= MIN_K else MIN_K


def neutral_indices(pal_lch: Lch) -> np.ndarray:
    """
    Build a neutral pool from a palette LCh array.

    Preference:
      - primary: chroma <= 10
      - fallback: lowest-chroma quartile
      - always include darkest and brightest to span lightness
    """
    chroma = pal_lch[:, 1].astype(np.float32, copy=False)
    idx = np.where(chroma <= 10.0)[0]
    if idx.size == 0:
        q = float(np.quantile(chroma, 0.25))
        idx = np.where(chroma <= q)[0]

    lo = int(np.argmin(pal_lch[:, 0]))
    hi = int(np.argmax(pal_lch[:, 0]))

    if idx.size == 0:
        return np.unique(np.array([lo, hi], dtype=np.int32))

    return np.unique(np.concatenate([idx, np.array([lo, hi], dtype=np.int32)])).astype(
        np.int32, copy=False
    )


def greyish_sources(src_rgb: U8Image, src_lch: Lch) -> np.ndarray:
    """
    Detect grey-ish source colours among unique RGB rows.

    Rules:
      - small RGB channel spread (<= 20)
      - chroma <= 22
      - hue near 0 or 180 unless chroma <= 12
    Returns:
      boolean mask array with length U (unique count)
    """
    r = src_rgb[:, 0].astype(np.int16, copy=False)
    g = src_rgb[:, 1].astype(np.int16, copy=False)
    b = src_rgb[:, 2].astype(np.int16, copy=False)
    max_delta = np.maximum.reduce([np.abs(r - g), np.abs(g - b), np.abs(r - b)])

    chroma = src_lch[:, 1].astype(np.float32, copy=False)
    hue = src_lch[:, 2].astype(np.float32, copy=False)

    d0 = np.minimum(hue, 360.0 - hue)
    d180 = np.minimum(np.abs(hue - 180.0), 360.0 - np.abs(hue - 180.0))
    near_neutral_axis = (np.minimum(d0, d180) <= 18.0) | (chroma <= 12.0)

    return (max_delta <= 20) & (chroma <= 22.0) & near_neutral_axis


def _auto_k_from_hist(counts_per_palette: np.ndarray) -> int:
    """
    Decide K from a usage histogram:
      1) first big drop where next/prev <= DROP_RATIO
      2) else the smallest K that reaches COVERAGE_TARGET of total weight
    """
    if counts_per_palette.size == 0:
        return MIN_K

    c_sorted = np.sort(counts_per_palette)[::-1]
    # big drop rule
    for i in range(c_sorted.size - 1):
        a, b = float(c_sorted[i]), float(c_sorted[i + 1])
        if a <= 0.0:
            break
        if (b / a) <= DROP_RATIO and (i + 1) >= MIN_K:
            return i + 1

    # coverage rule
    total = float(c_sorted.sum())
    if total <= 0.0:
        return MIN_K
    cum = np.cumsum(c_sorted, dtype=np.float64)
    k_cov = int(np.searchsorted(cum, COVERAGE_TARGET * total, side="left")) + 1
    return max(MIN_K, min(k_cov, c_sorted.size))


def _pick_top_indices_by_usage(counts_per_palette: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k palette bins by usage."""
    if counts_per_palette.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(counts_per_palette)[::-1]
    k_eff = max(MIN_K, min(int(k), order.size))
    return order[:k_eff].astype(np.int32, copy=False)


def restrict_palette(
    colours_mode: str,
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

    colours_mode:
      - "full":     no restriction
      - "bw":       neutrals only by LCh chroma; ensure extremes if empty
      - "limited":  choose top-K by usage. If the user provided --limit, treat
                    it as an upper bound and allow lowering K when a big drop
                    is detected. If no limit provided, choose K automatically.
    """
    P = pal_rgb_full.shape[0]

    # Full palette
    if colours_mode == "full":
        sel = np.arange(P, dtype=np.int32)
        return (
            pal_rgb_full[sel],
            pal_lab_full[sel].astype(np.float32, copy=False),
            pal_lch_full[sel].astype(np.float32, copy=False),
            sel,
        )

    # Black & white (neutrals)
    if colours_mode == "bw":
        mask = pal_lch_full[:, 1] <= NEUTRAL_C_MAX
        idx = np.where(mask)[0]
        if idx.size == 0:
            lo = int(np.argmin(pal_lch_full[:, 0]))
            hi = int(np.argmax(pal_lch_full[:, 0]))
            idx = np.unique(np.array([lo, hi], dtype=np.int32))
        sel = idx.astype(np.int32, copy=False)
        if debug:
            debug_log(
                key_value_pairs_to_string(
                    [("Palette", "neutrals only"), ("Neutrals", sel.size)]
                )
            )
        return (
            pal_rgb_full[sel],
            pal_lab_full[sel].astype(np.float32, copy=False),
            pal_lch_full[sel].astype(np.float32, copy=False),
            sel,
        )

    # Limited palette (top-K by usage)
    uniq_rgb, uniq_counts = unique_visible_rgb(img_rgb, alpha)
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
    nearest = nearest_palette_indices_lab_distance(src_lab, pal_lab_full)  # (U,)
    counts_per_palette = np.bincount(
        nearest, weights=uniq_counts.astype(np.float64), minlength=P
    )

    k_auto = _auto_k_from_hist(counts_per_palette)
    if debug:
        debug_log(
            key_value_pairs_to_string(
                [("Palette", "limited (top-K)"), ("Auto K", k_auto)]
            )
        )

    if limit_opt is not None:
        k_final = min(int(limit_opt), k_auto)
        if debug:
            debug_log(
                key_value_pairs_to_string(
                    [
                        ("User limit", int(limit_opt)),
                        ("Using K", int(k_final)),
                        (
                            "Reason",
                            "big drop" if k_final != int(limit_opt) else "user cap",
                        ),
                    ]
                )
            )
    else:
        k_final = k_auto

    sel = _pick_top_indices_by_usage(counts_per_palette, k_final)

    if debug:
        order = np.argsort(counts_per_palette)[::-1]
        top = order[: min(12, order.size)]
        total = counts_per_palette.sum() or 1.0
        debug_log(key_value_pairs_to_string([("Selected K", int(sel.size))]))
        debug_log("Top palette by estimated usage:")
        sel_set = set(sel.tolist())
        for j in top:
            rgb = pal_rgb_full[j]
            hhex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            share = counts_per_palette[j] / total
            nm = name_of.get(hhex, "?")
            mark = "*" if j in sel_set else " "
            debug_log(f"  {mark} {hhex}  {nm}: share={share:.1%}")

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
