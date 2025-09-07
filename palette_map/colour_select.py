# palette_map/colour_select.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .core_types import U8Image, U8Mask, Lab, Lch, NameOf
from .colour_convert import rgb_to_lab

# ---- knobs (gentle; stable defaults) ----------------------------------------
NEUTRAL_C_MAX = 5.0  # <= this chroma => neutral/grey
DROP_RATIO = 0.55  # big drop if next bucket <= 55% of previous
COVERAGE_TARGET = 0.96  # fallback if no clear drop
MIN_K = 2

# -----------------------------------------------------------------------------


def resolve_limit_once(colours: str, limit_opt: Optional[int]) -> Optional[int]:
    """
    Return the user-provided limit (if >=2) or None.
    We *don't* force interactive input here; palette_map.py can pass None.
    """
    if colours != "limited":
        return None
    if limit_opt is None:
        return None
    return int(limit_opt) if limit_opt >= MIN_K else MIN_K


def _unique_visible(rgb: U8Image, alpha: U8Mask) -> Tuple[U8Image, np.ndarray]:
    vis = alpha > 0
    if not np.any(vis):
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    flat = rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return uniq.astype(np.uint8), counts.astype(np.int64)


def _nearest_palette_indices(src_lab: Lab, pal_lab: Lab) -> np.ndarray:
    # Euclidean in Lab for speed (ΔE2000 not needed for coarse histogram)
    diff = pal_lab[None, :, :] - src_lab[:, None, :]
    de2 = np.sum(diff * diff, axis=2)
    return np.argmin(de2, axis=1).astype(np.int32)


def _auto_k_from_hist(counts_per_palette: np.ndarray) -> int:
    """
    Decide K from palette-usage histogram:
      1) Look for the first big drop (next/prev <= DROP_RATIO).
      2) If none, pick the smallest K that achieves COVERAGE_TARGET.
    """
    if counts_per_palette.size == 0:
        return MIN_K

    c_sorted = np.sort(counts_per_palette)[::-1]
    # 1) big drop
    for i in range(len(c_sorted) - 1):
        a, b = float(c_sorted[i]), float(c_sorted[i + 1])
        if a <= 0:
            break
        if b / a <= DROP_RATIO and (i + 1) >= MIN_K:
            return i + 1

    # 2) coverage
    total = float(c_sorted.sum())
    if total <= 0:
        return MIN_K
    cum = np.cumsum(c_sorted, dtype=np.float64)
    k_cov = int(np.searchsorted(cum, COVERAGE_TARGET * total, side="left")) + 1
    return max(MIN_K, min(k_cov, len(c_sorted)))


def _pick_top_indices_by_usage(counts_per_palette: np.ndarray, k: int) -> np.ndarray:
    if counts_per_palette.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(counts_per_palette)[::-1]
    k = max(MIN_K, min(k, order.size))
    return order[:k].astype(np.int32, copy=False)


def _neutral_mask(pal_lch_full: Lch) -> np.ndarray:
    return pal_lch_full[:, 1] <= NEUTRAL_C_MAX


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
      - "full":    no restriction
      - "bw":      neutrals only (C* <= NEUTRAL_C_MAX), ensure extremes if empty
      - "limited": choose top-K by usage; if user supplied --limit, treat as an
                   *upper bound* and allow lowering K when a big drop is detected.
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
        mask = _neutral_mask(pal_lch_full)
        idx = np.where(mask)[0]
        if idx.size == 0:
            # Ensure darkest & brightest to span
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

    # ---- limited
    # Build a palette-usage histogram for *this* image
    uniq_rgb, uniq_counts = _unique_visible(img_rgb, alpha)
    if uniq_rgb.shape[0] == 0:
        # fallback: pick MIN_K darkest/brightest to have something
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
    nearest = _nearest_palette_indices(src_lab, pal_lab_full)  # (U,)
    # accumulate counts per palette index
    counts_per_palette = np.bincount(
        nearest, weights=uniq_counts.astype(np.float64), minlength=P
    )

    # Decide K
    k_auto = _auto_k_from_hist(counts_per_palette)
    if debug:
        print(f"[debug] limited: k_auto={k_auto}", flush=True)

    # If user provided --limit, treat it as an *upper bound*; allow going lower
    # when a big drop suggests fewer colours.
    if limit_opt is not None:
        k_final = min(int(limit_opt), k_auto)
        if debug and k_final != limit_opt:
            print(
                f"[debug] limited: user limit={limit_opt} → using {k_final} due to big drop",
                flush=True,
            )
    else:
        k_final = k_auto

    # Pick top-K by usage
    sel = _pick_top_indices_by_usage(counts_per_palette, k_final)

    if debug:
        # small report (top 12)
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


__all__ = ["restrict_palette", "resolve_limit_once"]
