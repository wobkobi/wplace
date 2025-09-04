# palette_map/pixel/run.py
from __future__ import annotations

from typing import Dict, List, Tuple
import math
import time
import numpy as np

from palette_map.color_convert import rgb_to_lab, lab_to_lch
from palette_map.core_types import (
    U8Image,
    U8Mask,
    Lab,
    Lch,
    RGBTuple,
    SourceItem,
)

# ----------------------------
# Basic helpers (tiny baseline)
# ----------------------------


def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = [float(x) for x in lab1.tolist()]
    L2, a2, b2 = [float(x) for x in lab2.tolist()]
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - math.sqrt((Cm**7) / (Cm**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = (math.degrees(math.atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (math.degrees(math.atan2(b2, a2p)) + 360.0) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif dhp > 180.0:
        dhp -= 360.0
    elif dhp < -180.0:
        dhp += 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)
    Lpm = (L1 + L2) / 2.0
    Cpm = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hpm = 0.5 * (h1p + h2p)
    else:
        hpm = (
            0.5 * (h1p + h2p + 360.0)
            if (h1p + h2p) < 360.0
            else 0.5 * (h1p + h2p - 360.0)
        )
    T = (
        1
        - 0.17 * math.cos(math.radians(hpm - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hpm))
        + 0.32 * math.cos(math.radians(3.0 * hpm + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hpm - 63.0))
    )
    d_ro = 30.0 * math.exp(-(((hpm - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * math.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1.0 + (0.015 * (Lpm - 50.0) ** 2.0) / math.sqrt(20.0 + (Lpm - 50.0) ** 2.0)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc
    dE = math.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return float(dE)


def unique_colors_and_counts(rgb: U8Image, alpha: U8Mask) -> List[Tuple[RGBTuple, int]]:
    mask = alpha != 0
    if not mask.any():
        return []
    samples = rgb[mask]
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = samples.view(dt).reshape(-1)
    uniq, counts = np.unique(flat, return_counts=True)
    rs = uniq["r"].astype(int)
    gs = uniq["g"].astype(int)
    bs = uniq["b"].astype(int)
    items = [
        ((int(rs[i]), int(gs[i]), int(bs[i])), int(counts[i])) for i in range(len(uniq))
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def build_sources(rgb: U8Image, alpha: U8Mask) -> Tuple[List[SourceItem], Lab, Lch]:
    items = unique_colors_and_counts(rgb, alpha)
    if not items:
        return (
            [],
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    src_rgb = np.array([c for c, _ in items], dtype=np.uint8)
    src_lab = rgb_to_lab(src_rgb).reshape(-1, 3).astype(np.float32)
    src_lch = lab_to_lch(src_lab).reshape(-1, 3).astype(np.float32)
    sources: List[SourceItem] = []
    for i, ((r, g, b), n) in enumerate(items):
        sources.append(
            SourceItem(
                rgb=(int(r), int(g), int(b)),
                count=int(n),
                lab=src_lab[i].copy(),
                lch=src_lch[i].copy(),
            )
        )
    return sources, src_lab, src_lch


def _apply_mapping(
    img_rgb: U8Image, alpha: U8Mask, mapping: Dict[RGBTuple, RGBTuple]
) -> U8Image:
    out = img_rgb.copy()
    vis = alpha != 0
    for (sr, sg, sb), (tr, tg, tb) in mapping.items():
        m = (
            vis
            & (img_rgb[..., 0] == sr)
            & (img_rgb[..., 1] == sg)
            & (img_rgb[..., 2] == sb)
        )
        if m.any():
            out[m, 0] = np.uint8(tr)
            out[m, 1] = np.uint8(tg)
            out[m, 2] = np.uint8(tb)
    return out


# ----------------------------
# Soft-unique assignment
# ----------------------------


def _soft_unique_assign(
    sources: List[SourceItem],
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    *,
    unique_tol: float = 2.5,
    topk: int = 8,
) -> Tuple[Dict[int, int], List[List[int]], List[List[float]]]:
    """
    Returns:
      assigned: {src_index -> pal_index}
      cand_idx: per-src sorted palette indices by ascending dE
      cand_cost: per-src sorted dE list
    Strategy:
      - Order sources by descending pixel count.
      - For each source, prefer the nearest unused palette colour.
      - If it's taken, scan next best candidates for an UNUSED one whose dE
        is within `unique_tol` of the source's own nearest dE.
      - If none found, allow sharing (keep nearest even if used).
    """
    n_s = len(sources)
    n_p = int(pal_lab.shape[0])

    # Build per-source candidate lists (indices and costs)
    cand_idx: List[List[int]] = [[] for _ in range(n_s)]
    cand_cost: List[List[float]] = [[] for _ in range(n_s)]
    for i, s in enumerate(sources):
        costs = [ciede2000_pair(s.lab, pal_lab[j]) for j in range(n_p)]
        order = list(range(n_p))
        order.sort(key=lambda j: costs[j])
        cand_idx[i] = order
        cand_cost[i] = [costs[j] for j in order]

    assigned: Dict[int, int] = {}
    used: set[int] = set()

    order_src = list(range(n_s))
    order_src.sort(key=lambda i: -sources[i].count)

    for i in order_src:
        best_j = cand_idx[i][0]
        best_c = cand_cost[i][0]

        # nearest is unused → take it
        if best_j not in used:
            assigned[i] = best_j
            used.add(best_j)
            continue

        # try to find a distinct alternative within tolerance
        picked = None
        lim = min(topk, len(cand_idx[i]))
        for k in range(1, lim):
            j = cand_idx[i][k]
            if j in used:
                continue
            c = cand_cost[i][k]
            if c <= best_c + unique_tol:
                picked = j
                break

        if picked is not None:
            assigned[i] = picked
            used.add(picked)
        else:
            # fall back to sharing to avoid large drift
            assigned[i] = best_j

    return assigned, cand_idx, cand_cost


# ----------------------------
# Public entry point
# ----------------------------


def run_pixel(
    img_rgb: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,  # (P,3) uint8
    pal_lab: Lab,  # (P,3) float32
    pal_lch: Lch,  # (P,3) float32
    debug: bool = False,
) -> U8Image:
    """
    Baseline pixel mapper with soft-unique assignment:
      1) gather unique visible colours
      2) compute nearest palette by CIEDE2000
      3) try to avoid two sources mapping to the same palette colour unless
         that would increase error by > unique_tol dE
      4) apply per-unique mapping
    """
    t0 = time.perf_counter()

    sources, src_lab, src_lch = build_sources(img_rgb, alpha)
    if not sources:
        if debug:
            print("[debug] pixel  empty (no visible pixels)")
        return img_rgb

    assigned, cand_idx, cand_cost = _soft_unique_assign(
        sources, pal_rgb, pal_lab, pal_lch, unique_tol=2.5, topk=8
    )

    # Build mapping
    mapping: Dict[RGBTuple, RGBTuple] = {}
    for i, s in enumerate(sources):
        j = assigned[i]
        mapping[s.rgb] = (int(pal_rgb[j, 0]), int(pal_rgb[j, 1]), int(pal_rgb[j, 2]))

    out = _apply_mapping(img_rgb, alpha, mapping)

    if debug:
        print("[debug] per-unique mapping:")
        for i, s in enumerate(sources):
            j = assigned[i]
            sL, sC, sh = [float(x) for x in src_lch[i].tolist()]
            tL, tC, th = [float(x) for x in pal_lch[j].tolist()]
            dh = th - sh
            while dh > 180.0:
                dh -= 360.0
            while dh < -180.0:
                dh += 360.0
            s_hex = f"#{s.rgb[0]:02x}{s.rgb[1]:02x}{s.rgb[2]:02x}"
            t_hex = f"#{int(pal_rgb[j,0]):02x}{int(pal_rgb[j,1]):02x}{int(pal_rgb[j,2]):02x}"
            dE = (
                cand_cost[i][cand_idx[i].index(j)]
                if j in cand_idx[i]
                else ciede2000_pair(s.lab, pal_lab[j])
            )
            print(
                f"  src {s_hex:>7}  count={s.count:6d} -> {t_hex:>7}  "
                f"[dE={dE:5.2f}, dL={tL - sL:+.3f}, dC={tC - sC:+.3f}, dh={dh:.1f} deg]"
            )
        t1 = time.perf_counter()
        print(f"[debug] pixel mapping time={t1 - t0:.3f}s")

    return out
