#!/usr/bin/env python3
"""
palette_map_dual.py — closest-first + frequency-conflict + neighbour tone-pairing
(with soft, general guardrails; no hard-coded colour exceptions)

- pixel: OKLab/OKLCh cost with generic constraints, frequency-first assignment
         with peer awareness, neighbour separation, gradient sign preservation.
- photo: OKLab nearest with a mild "don't collapse vivid colours to grey" bias.

Alpha preserved. No upscaling.
Default output "<input_stem>_wplace.png".
Prints palette usage (hex, name, count).

Examples
  python palette_map_dual.py input.png
  python palette_map_dual.py input.png --mode pixel --height 512
  python palette_map_dual.py input.png --mode photo
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Set
import argparse
import numpy as np
from PIL import Image

# ---------- Palette (wheel → greyscale) ----------
PALETTE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("#ed1c24", "Red"),
    ("#d18078", "Peach"),
    ("#fa8072", "Light Red"),
    ("#9b5249", "Dark Peach"),
    ("#fab6a4", "Light Peach"),
    ("#e45c1a", "Dark Orange"),
    ("#684634", "Dark Brown"),
    ("#ffc5a5", "Light Beige"),
    ("#d18051", "Dark Beige"),
    ("#ff7f27", "Orange"),
    ("#7b6352", "Dark Tan"),
    ("#f8b277", "Beige"),
    ("#d6b594", "Light Tan"),
    ("#9c846b", "Tan"),
    ("#dba463", "Light Brown"),
    ("#95682a", "Brown"),
    ("#f6aa09", "Gold"),
    ("#9c8431", "Dark Goldenrod"),
    ("#6d643f", "Dark Stone"),
    ("#948c6b", "Stone"),
    ("#cdc59e", "Light Stone"),
    ("#c5ad31", "Goldenrod"),
    ("#f9dd3b", "Yellow"),
    ("#e8d45f", "Light Goldenrod"),
    ("#fffabc", "Light Yellow"),
    ("#4a6b3a", "Dark Olive"),
    ("#87ff5e", "Light Green"),
    ("#5a944a", "Olive"),
    ("#84c573", "Light Olive"),
    ("#13e67b", "Green"),
    ("#0eb968", "Dark Green"),
    ("#13e1be", "Light Teal"),
    ("#0c816e", "Dark Teal"),
    ("#bbfaf2", "Light Cyan"),
    ("#10aea6", "Teal"),
    ("#60f7f2", "Cyan"),
    ("#0f799f", "Dark Cyan"),
    ("#7dc7ff", "Light Blue"),
    ("#4093e4", "Blue"),
    ("#333941", "Dark Slate"),
    ("#28509e", "Dark Blue"),
    ("#6d758d", "Slate"),
    ("#99b1fb", "Light Indigo"),
    ("#b3b9d1", "Light Slate"),
    ("#b5aef1", "Light Slate Blue"),
    ("#7a71c4", "Slate Blue"),
    ("#4a4284", "Dark Slate Blue"),
    ("#6b50f6", "Indigo"),
    ("#4d31b8", "Dark Indigo"),
    ("#e09ff9", "Light Purple"),
    ("#780c99", "Dark Purple"),
    ("#aa38b9", "Purple"),
    ("#cb007a", "Dark Pink"),
    ("#ec1f80", "Pink"),
    ("#f38da9", "Light Pink"),
    ("#600018", "Deep Red"),
    ("#a50e1e", "Dark Red"),
    ("#000000", "Black"),
    ("#3c3c3c", "Dark Gray"),
    ("#787878", "Gray"),
    ("#aaaaaa", "Medium Gray"),
    ("#d2d2d2", "Light Gray"),
    ("#ffffff", "White"),
)
PALETTE_HEX: tuple[str, ...] = tuple(h for h, _ in PALETTE_ENTRIES)
HEX_TO_NAME: Dict[str, str] = {h.lower(): n for h, n in PALETTE_ENTRIES}


# ---------- Helpers ----------
def parse_hex(code: str) -> Tuple[int, int, int]:
    s = code[1:] if code.startswith("#") else code
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def build_palette_rgb() -> np.ndarray:
    cols = np.array([parse_hex(h) for h in PALETTE_HEX], dtype=np.uint8)
    # de-dup while preserving order
    _, idx = np.unique(cols.view([("", cols.dtype)] * cols.shape[1]), return_index=True)
    return cols[np.sort(idx)]


def _srgb_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    l = (
        0.4122214708 * lin[..., 0]
        + 0.5363325363 * lin[..., 1]
        + 0.0514459929 * lin[..., 2]
    )
    m = (
        0.2119034982 * lin[..., 0]
        + 0.6806995451 * lin[..., 1]
        + 0.1073969566 * lin[..., 2]
    )
    s = (
        0.0883024619 * lin[..., 0]
        + 0.2817188376 * lin[..., 1]
        + 0.6299787005 * lin[..., 2]
    )
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, A, B], axis=-1).astype(np.float32)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rgb_to_hex(row: np.ndarray) -> str:
    return f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}"


# ---------- Photo mode ----------
PHOTO_SRC_SAT_T = 0.06
PHOTO_PAL_GREY_T = 0.03
PHOTO_GREY_PENALTY = 1.7


def _nearest_indices_oklab_photo(
    src_lab: np.ndarray,
    pal_lab: np.ndarray,
    src_chroma: np.ndarray,
    pal_chroma: np.ndarray,
) -> np.ndarray:
    diff = src_lab[:, None, :] - pal_lab[None, :, :]
    dist2 = np.einsum("knc,knc->kn", diff, diff, optimize=True)
    mask = (src_chroma[:, None] > PHOTO_SRC_SAT_T) & (
        pal_chroma[None, :] < PHOTO_PAL_GREY_T
    )
    if mask.any():
        dist2 = np.where(mask, dist2 * (PHOTO_GREY_PENALTY**2), dist2)
    return np.argmin(dist2, axis=1)


def map_photo_nearest(rgb_flat: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    uniq_rgb, inv = np.unique(rgb_flat, axis=0, return_inverse=True)
    pal_lab = _srgb_to_oklab(palette_rgb)
    pal_C = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    src_lab = _srgb_to_oklab(uniq_rgb)
    src_C = np.hypot(src_lab[:, 1], src_lab[:, 2])
    idx = _nearest_indices_oklab_photo(src_lab, pal_lab, src_C, pal_C)
    return palette_rgb[idx][inv].astype(np.uint8)


# ---------- Pixel mode: general scoring ----------
# Base OKLCh weights
HUE_WEIGHT_BASE = 0.45
CHROMA_FLOOR = 0.02
CHROMA_RANGE = 0.12

# Anti-grey for saturated sources
SRC_SAT_T = 0.04
PAL_GREY_T = 0.05
GREY_PENALTY = 2.0

# White / near-white guard
WHITE_HARD_L = 0.97
WHITE_HARD_C = 0.015
NEARWHITE_L = 0.94
NEARWHITE_C = 0.020
NEARWHITE_BLOCK = 3.0
WHITE_BONUS = 0.85
VLGREY_BONUS = 0.92

# Neutral guard (keep greys neutral)
NEUTRAL_SRC_C_MAX = 0.06
NEUTRAL_L_MIN = 0.35
NEUTRAL_L_MAX = 0.92
WARM_HUE_CENTRE = np.deg2rad(35.0)  # orange/brown region
WARM_HUE_BW = np.deg2rad(55.0)
WARM_MIN_C = 0.04
NEUTRAL_TO_WARM_PEN = 3.0
NEUTRAL_TARGET_C_MAX = 0.10  # neutrals prefer palette with low chroma

# Pale-to-oversaturated clamp (generic, hue-agnostic)
PALE_C = 0.05
PALE_MAX_DELTA_C = 0.05
PALE_OVER_PEN = 1.6

# Hue jump limiter (generic)
HUE_JUMP_T = np.deg2rad(30.0)
HUE_JUMP_SLOPE = 0.5  # multiplies extra cost when |Δh| > threshold

# Chroma direction consistency
DELTA_C_TOL = 0.04
CHROMA_BIG_JUMP_PEN = 1.5

# Soft penalty for large lightness change (keeps mapping believable)
DELTA_L_SOFT_TOL = 0.06
DELTA_L_SOFT_WEIGHT = 0.35

# Candidate set building
GOOD_ENOUGH_RATIO = 1.06
MAX_CANDIDATES = 22
PEER_MIN_CANDS = 12  # for colours that touch many neighbours

# Neighbour separation
HUE_CLUSTER_BW = np.deg2rad(28.0)  # tone ladder width
ORDER_TOL_SRC = 0.003  # OKLab L tolerances
ORDER_TOL_TGT = 0.003


def compute_oklab_cost(src_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    pal_lab = _srgb_to_oklab(palette_rgb)
    src_lab = _srgb_to_oklab(src_rgb)

    pal_L = pal_lab[:, 0]
    pal_a = pal_lab[:, 1]
    pal_b = pal_lab[:, 2]
    pal_C = np.hypot(pal_a, pal_b)
    pal_h = np.arctan2(pal_b, pal_a)

    src_L = src_lab[:, 0]
    src_a = src_lab[:, 1]
    src_b = src_lab[:, 2]
    src_C = np.hypot(src_a, src_b)
    src_h = np.arctan2(src_b, src_a)

    # Base OKLCh distance
    dL2 = (src_L[:, None] - pal_L[None, :]) ** 2
    dC2 = (src_C[:, None] - pal_C[None, :]) ** 2
    dh = _wrap(src_h[:, None] - pal_h[None, :])
    hue_w = (
        HUE_WEIGHT_BASE
        * np.clip((src_C - CHROMA_FLOOR) / CHROMA_RANGE, 0.0, 1.0)[:, None]
    )
    cost = 0.6 * dL2 + 1.0 * dC2 + hue_w * (dh**2)

    # Saturated sources should not fall to grey
    to_grey = (src_C[:, None] > SRC_SAT_T) & (pal_C[None, :] < PAL_GREY_T)
    if to_grey.any():
        cost = np.where(to_grey, cost * (GREY_PENALTY**2), cost)

    # White hard snap and near-white guard
    # Find white index and very-light-grey mask once
    pal_is_white = (pal_C < 0.010) & (pal_L > 0.985)
    pal_is_vlgrey = (pal_C < 0.020) & (pal_L > 0.94)
    # Hard snap to pure white
    hard_white = (src_L >= WHITE_HARD_L) & (src_C <= WHITE_HARD_C)
    if hard_white.any():
        # make white cost minimal, others very large
        nonwhite = ~pal_is_white[None, :]
        cost = np.where(
            hard_white[:, None] & nonwhite, cost * (NEARWHITE_BLOCK**4), cost
        )
        cost = np.where(
            hard_white[:, None] & pal_is_white[None, :], cost * WHITE_BONUS, cost
        )
    # Near-white preference
    near_white = (src_L >= NEARWHITE_L) & (src_C <= NEARWHITE_C)
    if near_white.any():
        coloured = ~pal_is_vlgrey[None, :]
        cost = np.where(
            near_white[:, None] & coloured, cost * (NEARWHITE_BLOCK**2), cost
        )
        cost = np.where(
            near_white[:, None] & pal_is_white[None, :], cost * WHITE_BONUS, cost
        )
        cost = np.where(
            near_white[:, None] & pal_is_vlgrey[None, :], cost * VLGREY_BONUS, cost
        )

    # Neutral greys avoid warm browns/tans and prefer low-chroma targets
    src_is_neutral = (
        (src_C <= NEUTRAL_SRC_C_MAX)
        & (src_L >= NEUTRAL_L_MIN)
        & (src_L <= NEUTRAL_L_MAX)
    )
    if src_is_neutral.any():
        pal_is_warm = (np.abs(_wrap(pal_h - WARM_HUE_CENTRE)) <= WARM_HUE_BW) & (
            pal_C >= WARM_MIN_C
        )
        cost = np.where(
            src_is_neutral[:, None] & pal_is_warm[None, :],
            cost * (NEUTRAL_TO_WARM_PEN**2),
            cost,
        )
        pal_lowC = pal_C <= NEUTRAL_TARGET_C_MAX
        cost = np.where(src_is_neutral[:, None] & pal_lowC[None, :], cost * 0.90, cost)

    # Pale colours should not jump to very saturated targets
    src_is_pale = src_C <= PALE_C
    if src_is_pale.any():
        too_sat = pal_C[None, :] > (src_C[:, None] + PALE_MAX_DELTA_C)
        cost = np.where(src_is_pale[:, None] & too_sat, cost * (PALE_OVER_PEN**2), cost)

    # Hue jump limiter (for large hue swings on low-chroma sources)
    big_hue_jump = np.abs(dh) > HUE_JUMP_T
    low_chroma_src = src_C[:, None] < 0.12
    cost = np.where(
        big_hue_jump & low_chroma_src,
        cost * (1.0 + HUE_JUMP_SLOPE * (np.abs(dh) - HUE_JUMP_T)),
        cost,
    )

    # Chroma direction consistency (discourage large ΔC)
    deltaC = pal_C[None, :] - src_C[:, None]
    big_up = deltaC > DELTA_C_TOL
    big_dn = deltaC < -DELTA_C_TOL
    cost = np.where(
        src_is_pale[:, None] & big_up, cost * (CHROMA_BIG_JUMP_PEN**2), cost
    )
    cost = np.where(
        (src_C[:, None] > 0.12) & big_dn, cost * (CHROMA_BIG_JUMP_PEN**2), cost
    )

    # Soft penalty for large lightness change (hides extreme L jumps)
    deltaL = pal_L[None, :] - src_L[:, None]
    soft_L = np.maximum(0.0, np.abs(deltaL) - DELTA_L_SOFT_TOL)
    cost = cost + DELTA_L_SOFT_WEIGHT * (soft_L**2)

    return cost


def build_candidate_lists(
    cost: np.ndarray, min_count_for_peers: Set[int] | None = None
) -> List[np.ndarray]:
    """Keep candidates within GOOD_ENOUGH_RATIO of the best, limited to MAX_CANDIDATES.
    If min_count_for_peers provided, indices in that set will be extended later."""
    K = cost.shape[0]
    best = cost.min(axis=1)
    order = np.argsort(cost, axis=1)
    out: List[np.ndarray] = []
    for i in range(K):
        r = order[i]
        r = r[cost[i, r] <= best[i] * GOOD_ENOUGH_RATIO]
        if r.size == 0:
            r = order[i, :1]
        out.append(r[:MAX_CANDIDATES])
    return out


# ---------- Neighbour graph (8-neighbour) ----------
ADJ_MAX_UNIQUES = 40000


def _peer_sets_from_uimg(uimg: np.ndarray) -> List[Set[int]]:
    H, W = uimg.shape
    valid = uimg >= 0
    if not np.any(valid):
        return [set()]
    K = int(uimg[valid].max()) + 1
    if K <= 0:
        return [set()]
    if K > ADJ_MAX_UNIQUES:
        return [set() for _ in range(K)]

    counts: Dict[tuple[int, int], int] = {}
    deg = np.zeros(K, dtype=np.int64)

    def add_pairs(a: np.ndarray, b: np.ndarray) -> None:
        m = (a >= 0) & (b >= 0) & (a != b)
        if not np.any(m):
            return
        ia = a[m].astype(np.int32)
        ib = b[m].astype(np.int32)
        lo = np.minimum(ia, ib)
        hi = np.maximum(ia, ib)
        pairs = np.stack([lo, hi], axis=1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (i, j), c in zip(uniq, cnt):
            key = (int(i), int(j))
            counts[key] = counts.get(key, 0) + int(c)
            deg[int(i)] += int(c)
            deg[int(j)] += int(c)

    add_pairs(uimg[:, :-1], uimg[:, 1:])  # right
    add_pairs(uimg[:-1, :], uimg[1:, :])  # down
    add_pairs(uimg[:-1, :-1], uimg[1:, 1:])  # diag down-right
    add_pairs(uimg[:-1, 1:], uimg[1:, :-1])  # diag down-left

    peers: List[Set[int]] = [set() for _ in range(K)]
    for (i, j), c in counts.items():
        if c <= 0:
            continue
        # connect both directions
        peers[i].add(j)
        peers[j].add(i)
    return peers


# ---------- Frequency-first assignment with peer awareness ----------
def _topk_max_matching_iter(
    top_idx: np.ndarray, cands: List[np.ndarray], num_pal: int, peers: List[Set[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterative Kuhn-style augmenting paths (no recursion).
    Avoid assigning a palette already owned by a peer if an augmenting path exists.
    """
    owner = np.full(num_pal, -1, dtype=np.int32)
    chosen = np.full(len(cands), -1, dtype=np.int32)

    for i_val in top_idx:
        src = int(i_val)
        seen = np.zeros(num_pal, dtype=bool)
        parent = np.full(
            num_pal, -1, dtype=np.int32
        )  # remember which source led to palette
        queue: List[int] = []

        # seed with all allowed candidates not owned by peers
        for pj in cands[src]:
            j = int(pj)
            if owner[j] != -1 and owner[j] in peers[src]:
                continue
            if not seen[j]:
                seen[j] = True
                parent[j] = -2  # root marker
                queue.append(j)

        aug_end = -1
        while queue:
            j = queue.pop(0)
            if owner[j] == -1:
                aug_end = j
                break
            # explore the owner's alternative candidates
            other = int(owner[j])
            for pj2 in cands[other]:
                k = int(pj2)
                if owner[k] != -1 and owner[k] in peers[other]:
                    continue
                if not seen[k]:
                    seen[k] = True
                    parent[k] = j
                    queue.append(k)

        # build augmenting path if found
        if aug_end != -1:
            j = aug_end
            cur_src = src
            while True:
                prev_owner = owner[j]
                owner[j] = cur_src
                chosen[cur_src] = j
                if parent[j] == -2:
                    break
                j_prev = parent[j]
                cur_src = int(owner[j_prev])
                j = j_prev

    return owner, chosen


# ---------- Neighbour-aware greedy + post-passes ----------
def map_with_neighbours(
    uniq_rgb: np.ndarray,
    inv: np.ndarray,
    counts: np.ndarray,
    palette_rgb: np.ndarray,
    uimg: np.ndarray,  # 2D map of unique indices (-1 for transparent)
) -> np.ndarray:
    K, Np = uniq_rgb.shape[0], palette_rgb.shape[0]

    # Score all (source×palette)
    cost = compute_oklab_cost(uniq_rgb, palette_rgb)

    # Candidate lists
    cands = build_candidate_lists(cost)

    # Peer sets (8-neighbour)
    peers = _peer_sets_from_uimg(uimg)

    # Expand candidate lists for peer-heavy colours
    order_full = [np.argsort(cost[i]) for i in range(K)]
    for i in range(K):
        if peers[i] and cands[i].size < PEER_MIN_CANDS:
            cands[i] = order_full[i][: max(PEER_MIN_CANDS, cands[i].size)]

    # Frequency-first ordering
    order_src = np.argsort(-counts)
    topM = min(Np, K)
    top_idx = order_src[:topM]

    # Max-matching for the top block with peer awareness
    owner, chosen_top = _topk_max_matching_iter(top_idx, cands, Np, peers)

    used = set(int(j) for j in owner if int(j) != -1)
    choice = np.full(K, -1, dtype=np.int32)
    for i_val in top_idx:
        ii = int(i_val)
        cj = int(chosen_top[ii])
        if cj != -1:
            choice[ii] = cj

    # Greedy for the rest, avoid neighbour collisions where possible
    for i_val in order_src:
        ii = int(i_val)
        if choice[ii] != -1:
            continue
        row = cands[ii]
        if row.size == 0:
            row = np.argsort(cost[ii])[:1]

        peer_used: Set[int] = {int(choice[j]) for j in peers[ii] if 0 <= choice[j] < Np}

        # 1) first free non-neighbour palette
        picked = None
        for jv in row:
            j = int(jv)
            if j not in used and j not in peer_used:
                picked = j
                break
        if picked is not None:
            choice[ii] = picked
            used.add(picked)
            continue

        # 2) any non-neighbour (even if reused)
        for jv in row:
            j = int(jv)
            if j not in peer_used:
                choice[ii] = j
                used.add(j)
                break
        if choice[ii] != -1:
            continue

        # 3) any free
        for jv in row:
            j = int(jv)
            if j not in used:
                choice[ii] = j
                used.add(j)
                break
        if choice[ii] != -1:
            continue

        # 4) best
        choice[ii] = int(row[0])
        used.add(int(row[0]))

    # --- Post-pass A: split neighbour collisions with tone ladder ---
    pal_lab = _srgb_to_oklab(palette_rgb)
    pal_L = pal_lab[:, 0]
    pal_a = pal_lab[:, 1]
    pal_b = pal_lab[:, 2]
    pal_C = np.hypot(pal_a, pal_b)
    pal_h = np.arctan2(pal_b, pal_a)

    src_L = _srgb_to_oklab(uniq_rgb)[:, 0]
    src_C = np.hypot(_srgb_to_oklab(uniq_rgb)[:, 1], _srgb_to_oklab(uniq_rgb)[:, 2])

    for ii in range(K):
        pi = int(choice[ii])
        if pi < 0:
            continue
        for jj in peers[ii]:
            if jj <= ii:
                continue
            pj = int(choice[jj])
            if pj < 0:
                continue
            if pj != pi:
                continue  # no collision

            # Move the less frequent to a nearby tone in the same hue cluster if possible
            move = ii if counts[ii] <= counts[jj] else jj
            keep = jj if move == ii else ii
            keep_h = pal_h[int(choice[keep])]
            keep_L = pal_L[int(choice[keep])]
            # tone ladder: same-hue cluster
            cluster = np.where(np.abs(_wrap(pal_h - keep_h)) <= HUE_CLUSTER_BW)[0]
            # Prefer entries that keep local order: darker if source is darker, else lighter
            want_darker = src_L[move] < src_L[keep] - ORDER_TOL_SRC
            cand_list = cands[move]
            # iterate candidates in order, filter to cluster and not equal to keep
            new_pick = None
            for jv in cand_list:
                j = int(jv)
                if j == int(choice[keep]):
                    continue
                if j not in cluster:
                    continue
                # lightness order intention
                if want_darker and pal_L[j] > keep_L + ORDER_TOL_TGT:
                    continue
                if (not want_darker) and pal_L[j] < keep_L - ORDER_TOL_TGT:
                    continue
                new_pick = j
                break
            # fallback: nearest in cluster even if order not perfect
            if new_pick is None:
                for jv in cand_list:
                    j = int(jv)
                    if j == int(choice[keep]):
                        continue
                    if j in cluster:
                        new_pick = j
                        break
            if new_pick is not None:
                choice[move] = new_pick

    # --- Post-pass B: preserve neighbour gradient signs (L and C) ---
    for ii in range(K):
        pi = int(choice[ii])
        if pi < 0:
            continue
        for jj in peers[ii]:
            if jj <= ii:
                continue
            pj = int(choice[jj])
            if pj < 0:
                continue
            # Source relations
            dL_src = src_L[ii] - src_L[jj]
            dC_src = src_C[ii] - src_C[jj]
            # Target relations
            dL_tgt = pal_L[pi] - pal_L[pj]
            dC_tgt = pal_C[pi] - pal_C[pj]

            # If signs flip materially, try to nudge the less frequent to restore sign
            need_fix_L = (abs(dL_src) > ORDER_TOL_SRC) and (
                np.sign(dL_src) != np.sign(dL_tgt)
            )
            need_fix_C = (abs(dC_src) > 0.01) and (np.sign(dC_src) != np.sign(dC_tgt))

            if need_fix_L or need_fix_C:
                move = ii if counts[ii] <= counts[jj] else jj
                keep = jj if move == ii else ii
                target_L = pal_L[int(choice[keep])] + (
                    ORDER_TOL_TGT * (-1 if dL_src > 0 else 1)
                )
                target_C = pal_C[int(choice[keep])] + (0.01 * (-1 if dC_src > 0 else 1))

                # Try to find a candidate for 'move' that restores the sign(s)
                new_pick = None
                for jv in cands[move]:
                    j = int(jv)
                    if j == int(choice[keep]):
                        continue
                    okL = True
                    okC = True
                    if need_fix_L:
                        okL = (dL_src > 0 and pal_L[j] > target_L - ORDER_TOL_TGT) or (
                            dL_src < 0 and pal_L[j] < target_L + ORDER_TOL_TGT
                        )
                    if need_fix_C:
                        okC = (dC_src > 0 and pal_C[j] > target_C - 0.005) or (
                            dC_src < 0 and pal_C[j] < target_C + 0.005
                        )
                    if okL and okC:
                        new_pick = j
                        break
                if new_pick is not None:
                    choice[move] = new_pick

    return palette_rgb[choice][inv].astype(np.uint8)


# ---------- IO / reporting ----------
def downscale_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if target_h <= 0:
        return img
    w, h = img.width, img.height
    if target_h >= h:
        return img
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), resample=Image.Resampling.BOX)


def print_usage_report(img: Image.Image) -> None:
    arr = np.array(img, dtype=np.uint8)
    if img.mode == "RGBA":
        mask = arr[:, :, 3] > 0
        if not mask.any():
            print("Colours used: none (fully transparent)")
            return
        cols = arr[:, :, :3][mask].reshape(-1, 3)
    else:
        cols = arr.reshape(-1, 3)

    uniq, counts = np.unique(cols, axis=0, return_counts=True)

    def _hx(row: np.ndarray) -> str:
        return f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}".lower()

    items = [
        (int(cnt), _hx(rgb), HEX_TO_NAME.get(_hx(rgb), "Unknown"))
        for rgb, cnt in zip(uniq, counts)
    ]
    items.sort(key=lambda t: t[0], reverse=True)
    print("Colours used:")
    for cnt, hx, name in items:
        print(f"{hx}  {name}: {cnt}")


# ---------- Pipeline ----------
def process(
    input_path: str,
    output_path: str | None,
    target_height: int,
    mode: str,
    auto_photo_threshold: int,
) -> None:
    palette_rgb = build_palette_rgb()

    im = Image.open(input_path).convert("RGBA")
    im = downscale_to_height(im, target_height)
    arr = np.array(im, dtype=np.uint8)
    H, W = arr.shape[:2]

    rgb_all = arr[:, :, :3].reshape(-1, 3)
    alpha = arr[:, :, 3].reshape(-1)
    vis_mask = alpha > 0

    chosen_mode = mode
    if mode == "auto":
        uniq_vis_est = (
            np.unique(rgb_all[vis_mask], axis=0).shape[0] if vis_mask.any() else 0
        )
        chosen_mode = "photo" if uniq_vis_est > auto_photo_threshold else "pixel"

    rgb_out = rgb_all.copy()
    if vis_mask.any():
        if chosen_mode == "pixel":
            # uniques for visible pixels
            uniq_rgb, inv, counts = np.unique(
                rgb_all[vis_mask], axis=0, return_inverse=True, return_counts=True
            )

            # 2D map of unique indices (-1 elsewhere) for neighbour discovery
            uimg = np.full((H, W), -1, dtype=np.int32)
            flat_vis = np.flatnonzero(vis_mask.reshape(H * W))
            ys = flat_vis // W
            xs = flat_vis % W
            uimg[ys, xs] = inv

            mapped = map_with_neighbours(uniq_rgb, inv, counts, palette_rgb, uimg)
        else:
            mapped = map_photo_nearest(rgb_all[vis_mask], palette_rgb)

        rgb_out[vis_mask] = mapped

    out = np.dstack([rgb_out.reshape(H, W, 3), alpha.reshape(H, W)]).astype(np.uint8)
    out_img = Image.fromarray(out, mode="RGBA")

    if not output_path:
        out_path = Path(input_path).with_name(f"{Path(input_path).stem}_wplace.png")
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", optimize=False)

    print(f"Mode: {chosen_mode}")
    print(f"Wrote {out_path} | size={W}x{H} | palette_size={len(PALETTE_ENTRIES)}")
    print_usage_report(out_img)


# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(
        description="OKLab palette mapper: closest-first, frequency conflict resolution, neighbour tone-pairing, and generic guardrails."
    )
    p.add_argument("input", type=str, help="Input image file")
    p.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output PNG path; default '<input_stem>_wplace.png'",
    )
    p.add_argument(
        "--height", type=int, default=0, help="Downscale height; 0 keeps size"
    )
    p.add_argument(
        "--mode",
        choices=["auto", "pixel", "photo"],
        default="auto",
        help="Mapping mode",
    )
    p.add_argument(
        "--auto-threshold",
        type=int,
        default=4096,
        help="Unique visible colour threshold for auto photo mode",
    )
    args = p.parse_args()

    process(args.input, args.output, args.height, args.mode, args.auto_threshold)


if __name__ == "__main__":
    main()
