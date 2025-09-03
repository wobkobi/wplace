# palette_map/pixel/run.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..analysis import ciede2000_pair
from ..color_convert import lab_to_lch_batch, srgb_to_lab_batch
from ..core_types import PaletteItem, SourceItem
from .candidates import build_candidate_rows
from .nudges import (
    rebalance_neutral_greys,
    spread_grey_collisions,
)

try:
    from .nudges import nudge_cool_darks_off_slate  # optional
except Exception:
    nudge_cool_darks_off_slate = None  # type: ignore[assignment]


def _unique_visible(
    rgb: np.ndarray, alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = alpha > 0
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    vis = rgb[mask].reshape(-1, 3)
    uniq, inv, counts = np.unique(vis, axis=0, return_inverse=True, return_counts=True)
    return uniq.astype(np.uint8), inv.astype(np.int64), counts.astype(np.int64)


def _fallback_nearest_all(
    assigned: Dict[int, int], lab_u: np.ndarray, pal_lab: np.ndarray
) -> None:
    U = lab_u.shape[0]
    P = pal_lab.shape[0]
    if len(assigned) == U:
        return
    for i in range(U):
        if i in assigned:
            continue
        s_lab = lab_u[i]
        dE = np.array(
            [ciede2000_pair(s_lab, pal_lab[j]) for j in range(P)], dtype=np.float32
        )
        assigned[i] = int(np.argmin(dE))


def run_pixel(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    debug: bool = False,
) -> Tuple[np.ndarray, Dict[int, str]]:
    H, W, _ = img_rgb.shape
    uniq, inv, counts = _unique_visible(img_rgb, alpha)
    U = uniq.shape[0]

    if debug:
        print(f"[debug] pixel  size_in={W}x{H}  size_eff={W}x{H}  uniques_visible={U}")

    if U == 0:
        return img_rgb.copy(), {}

    lab_u = srgb_to_lab_batch(uniq).reshape(-1, 3).astype(np.float32)
    lch_u = lab_to_lch_batch(lab_u).reshape(-1, 3).astype(np.float32)

    rows_by_i, cost_lu = build_candidate_rows(lab_u, lch_u, pal_lab, pal_lch, workers=0)

    assigned: Dict[int, int] = {}
    for i in range(U):
        rows = rows_by_i.get(i, [])
        if rows:
            assigned[i] = int(rows[0][1])

    _fallback_nearest_all(assigned, lab_u, pal_lab)

    rebalance_neutral_greys(lch_u, pal_lch, assigned, rows_by_i)
    spread_grey_collisions(lch_u, pal_lch, assigned, rows_by_i, counts)

    if nudge_cool_darks_off_slate is not None:
        sources: List[SourceItem] = [
            SourceItem(
                rgb=(int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2])),
                count=int(counts[i]),
                lab=lab_u[i],
                lch=lch_u[i],
            )
            for i in range(U)
        ]
        nudge_cool_darks_off_slate(sources, assigned, cost_lu, palette, pal_lch)

    if debug:
        taken = list(assigned.values())
        if taken:
            from collections import Counter

            c = Counter(taken).most_common(5)
            print(f"[debug] assign  uniques={U}  top_targets={c}")

    mapped_u = np.zeros((U, 3), dtype=np.uint8)
    for i, j in assigned.items():
        r, g, b = palette[j].rgb
        mapped_u[i, 0] = r
        mapped_u[i, 1] = g
        mapped_u[i, 2] = b

    out = img_rgb.copy()
    vis_mask = alpha > 0
    out_vals = mapped_u[inv]
    out[vis_mask] = out_vals.reshape(-1, 3)

    used_idx = sorted(set(assigned.values()))
    name_map: Dict[int, str] = {
        j: palette[j].name for j in used_idx if 0 <= j < len(palette)
    }
    return out, name_map


__all__ = ["run_pixel"]
