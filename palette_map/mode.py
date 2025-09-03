# palette_map/mode.py
from __future__ import annotations
from typing import Literal

import numpy as np

from .analysis import unique_colors_and_counts

"""
Mode selection helpers.

Exports:
- decide_auto_mode(img_rgb, alpha, *, max_uniques=512, topk=16, share_thresh=0.80) -> Literal["pixel","photo"]
- effective_mode(requested, img_rgb, alpha) -> Literal["pixel","photo","bw"]

Notes:
- "auto" picks "pixel" when the image uses few unique colours or when the top-N
  colours dominate the visible pixels; otherwise "photo".
"""


Mode = Literal["auto", "pixel", "photo", "bw"]
ResolvedMode = Literal["pixel", "photo", "bw"]


def decide_auto_mode(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    max_uniques: int = 512,
    topk: int = 16,
    share_thresh: float = 0.80,
) -> Literal["pixel", "photo"]:
    """
    Heuristic:
      - If visible unique colours ≤ max_uniques OR the top `topk` colours cover
        ≥ `share_thresh` of visible pixels → "pixel".
      - Else → "photo".
    """
    items = unique_colors_and_counts(img_rgb, alpha)
    total = sum(n for _c, n in items)
    if total == 0:
        # degenerate: no visible pixels → harmless to use pixel
        return "pixel"

    n_uniques = len(items)
    k = min(topk, n_uniques)
    top_share = (sum(n for _c, n in items[:k]) / total) if k > 0 else 1.0

    return (
        "pixel" if (n_uniques <= max_uniques or top_share >= share_thresh) else "photo"
    )


def effective_mode(
    requested: Mode, img_rgb: np.ndarray, alpha: np.ndarray
) -> ResolvedMode:
    """
    Resolve a user-requested mode into a concrete one.
    - "bw" stays "bw"
    - "pixel" / "photo" stay as is
    - "auto" → decide_auto_mode(...)
    """
    if requested == "bw":
        return "bw"
    if requested in ("pixel", "photo"):
        return requested
    # requested == "auto"
    return decide_auto_mode(img_rgb, alpha)


__all__ = ["Mode", "ResolvedMode", "decide_auto_mode", "effective_mode"]
