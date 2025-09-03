# palette_map/pixel/nudges.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..analysis import ciede2000_pair, hue_diff_deg
from ..constants import (
    BLUE_BAND_MAX,
    BLUE_BAND_MIN,
    BLUE_KEEP_RATIO,
    COOL_LIGHT_MIN_L,
    DARK_NEUTRAL_TO_SLATE_TOL,
    GREY_SRC_CUTOFF,
    NEUTRAL_C_MAX,
    NEUTRAL_REASSIGN_TOL,
    SLATE_HUE_MAX,
    SLATE_HUE_MIN,
    WARM_DARK_L_MAX,
    WARM_DARK_MIN_SC,
    WARM_HUE_MAX,
    WARM_HUE_MIN,
    WARM_TARGET_MIN_C,
)
from ..core_types import PaletteItem, SourceItem


def rebalance_neutral_greys(
    lch_u: np.ndarray,
    pal_lch: np.ndarray,
    assigned: Dict[int, int],
    rows_by_i: Dict[int, List[Tuple[float, int]]],
) -> None:
    GAP = 1.6
    for i, j0 in list(assigned.items()):
        sC = float(lch_u[i, 1])
        if sC < 8.0:
            continue
        if float(pal_lch[j0, 1]) > NEUTRAL_C_MAX:
            continue
        rows = rows_by_i.get(i, [])
        if not rows:
            continue
        base = rows[0][0]
        for cost_alt, j_alt in rows[1:4]:
            if float(pal_lch[j_alt, 1]) > NEUTRAL_C_MAX and (cost_alt - base) <= GAP:
                assigned[i] = int(j_alt)
                break


def spread_grey_collisions(
    lch_u: np.ndarray,
    pal_lch: np.ndarray,
    assigned: Dict[int, int],
    rows_by_i: Dict[int, List[Tuple[float, int]]],
    counts: np.ndarray,
) -> None:
    COST_TOL = 0.9
    neutral_js = [j for j in range(pal_lch.shape[0]) if pal_lch[j, 1] <= NEUTRAL_C_MAX]
    if len(neutral_js) <= 1:
        return
    by_target: Dict[int, List[int]] = {}
    for i, j in assigned.items():
        by_target.setdefault(j, []).append(i)
    for j, idxs in by_target.items():
        if pal_lch[j, 1] > NEUTRAL_C_MAX or len(idxs) <= 1:
            continue
        tL = float(pal_lch[j, 0])
        for i in sorted(idxs, key=lambda k: -int(counts[k])):
            rows = rows_by_i.get(i, [])
            if not rows:
                continue
            base_cost = rows[0][0]
            sL = float(lch_u[i, 0])
            for cost, j_alt in rows[:6]:
                if j_alt == j or pal_lch[j_alt, 1] > NEUTRAL_C_MAX:
                    continue
                tL_alt = float(pal_lch[j_alt, 0])
                if (
                    abs(sL - tL_alt) + 0.1 < abs(sL - tL)
                    and (cost - base_cost) <= COST_TOL
                ):
                    assigned[i] = int(j_alt)
                    break


def nudge_warm_darks_off_slate(
    sources: List[SourceItem],
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    pal_items: List[PaletteItem],
    pal_lch: np.ndarray,
) -> None:
    n = pal_lch.shape[0]
    neutral = {j for j in range(n) if float(pal_lch[j, 1]) <= NEUTRAL_C_MAX}
    slate = {
        j for j in range(n) if SLATE_HUE_MIN <= float(pal_lch[j, 2]) <= SLATE_HUE_MAX
    }
    warm_targets = [
        j
        for j in range(n)
        if (WARM_HUE_MIN <= float(pal_lch[j, 2]) <= WARM_HUE_MAX)
        and (float(pal_lch[j, 1]) > NEUTRAL_C_MAX)
    ]
    if not warm_targets:
        return

    def pair_cost(i: int, j: int) -> float:
        return cost_lu.get(
            (i, j), float(ciede2000_pair(sources[i].lab, pal_items[j].lab))
        )

    for i, s in enumerate(sources):
        sL, sC, sh = map(float, s.lch)
        if not (
            WARM_HUE_MIN <= sh <= WARM_HUE_MAX
            and sL <= WARM_DARK_L_MAX
            and sC >= WARM_DARK_MIN_SC
        ):
            continue
        pj = assigned.get(i)
        if pj is None or (pj not in neutral and pj not in slate):
            continue

        base = pair_cost(i, pj)
        cand = sorted(
            warm_targets,
            key=lambda j: (
                abs(float(pal_lch[j, 0]) - sL),
                hue_diff_deg(sh, float(pal_lch[j, 2])),
            ),
        )[:8]

        for g in cand:
            if float(pal_lch[g, 1]) < max(WARM_TARGET_MIN_C, NEUTRAL_C_MAX + 0.1):
                continue
            alt = pair_cost(i, g)
            if alt <= base + NEUTRAL_REASSIGN_TOL:
                assigned[i] = int(g)
                cost_lu[(i, int(g))] = float(alt)
                break


def nudge_cool_lights_off_neutral_and_slate(
    sources: List[SourceItem],
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    pal_items: List[PaletteItem],
    pal_lch: np.ndarray,
) -> None:
    n = pal_lch.shape[0]
    neutral = {j for j in range(n) if float(pal_lch[j, 1]) <= NEUTRAL_C_MAX}
    slate = {
        j for j in range(n) if SLATE_HUE_MIN <= float(pal_lch[j, 2]) <= SLATE_HUE_MAX
    }
    blue_targets = [
        j
        for j in range(n)
        if BLUE_BAND_MIN <= float(pal_lch[j, 2]) <= BLUE_BAND_MAX
        and float(pal_lch[j, 1]) > NEUTRAL_C_MAX
    ]
    if not blue_targets:
        return

    def pair_cost(i: int, j: int) -> float:
        return cost_lu.get(
            (i, j), float(ciede2000_pair(sources[i].lab, pal_items[j].lab))
        )

    for i, s in enumerate(sources):
        sL, sC, sh = map(float, s.lch)
        if not (
            sL >= COOL_LIGHT_MIN_L
            and sC > GREY_SRC_CUTOFF
            and BLUE_BAND_MIN <= sh <= BLUE_BAND_MAX
        ):
            continue

        pj = assigned.get(i)
        if pj is None:
            continue

        on_neutral = pj in neutral
        on_low_slate = (pj in slate) and (
            float(pal_lch[pj, 1]) < max(NEUTRAL_C_MAX + 0.1, sC * BLUE_KEEP_RATIO)
        )
        if not (on_neutral or on_low_slate):
            continue

        base = pair_cost(i, pj)
        cand = sorted(
            blue_targets,
            key=lambda j: (
                abs(float(pal_lch[j, 0]) - sL),
                hue_diff_deg(sh, float(pal_lch[j, 2])),
            ),
        )[:8]

        for g in cand:
            if float(pal_lch[g, 1]) < max(
                NEUTRAL_C_MAX + 0.1, sC * BLUE_KEEP_RATIO * 0.9
            ):
                continue
            alt = pair_cost(i, g)
            if alt <= base + NEUTRAL_REASSIGN_TOL:
                assigned[i] = int(g)
                cost_lu[(i, int(g))] = float(alt)
                break


def nudge_dark_neutrals_to_slate(
    sources: List[SourceItem],
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    pal_items: List[PaletteItem],
    pal_lch: np.ndarray,
) -> None:
    n = pal_lch.shape[0]
    neutral_set = {j for j in range(n) if float(pal_lch[j, 1]) <= NEUTRAL_C_MAX}
    slate_low = [
        j
        for j in range(n)
        if (SLATE_HUE_MIN <= float(pal_lch[j, 2]) <= SLATE_HUE_MAX)
        and (float(pal_lch[j, 1]) <= 8.0)
    ]
    if not slate_low:
        return

    def pair_cost(i: int, j: int) -> float:
        return cost_lu.get(
            (i, j), float(ciede2000_pair(sources[i].lab, pal_items[j].lab))
        )

    for i, s in enumerate(sources):
        sL, sC, _ = map(float, s.lch)
        if sC > GREY_SRC_CUTOFF or sL > 55.0:
            continue
        pj = assigned.get(i)
        if pj is None or pj not in neutral_set:
            continue

        base = pair_cost(i, pj)
        cands = sorted(slate_low, key=lambda g: abs(float(pal_lch[g, 0]) - sL))[:6]
        for g in cands:
            alt = pair_cost(i, g)
            if alt <= base + DARK_NEUTRAL_TO_SLATE_TOL:
                assigned[i] = int(g)
                cost_lu[(i, int(g))] = float(alt)
                break


def nudge_cool_darks_off_slate(
    sources: List[SourceItem],
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    pal_items: List[PaletteItem],
    pal_lch: np.ndarray,
) -> None:
    n = pal_lch.shape[0]
    neutral_set = {j for j in range(n) if float(pal_lch[j, 1]) <= NEUTRAL_C_MAX}
    slatey_lowC = {
        j
        for j in range(n)
        if (SLATE_HUE_MIN <= float(pal_lch[j, 2]) <= SLATE_HUE_MAX)
        and (float(pal_lch[j, 1]) <= 18.0)
    }
    good_pool = [
        j
        for j in range(n)
        if (170.0 <= float(pal_lch[j, 2]) <= 235.0) and (float(pal_lch[j, 1]) >= 10.0)
    ]
    if not good_pool:
        return

    def pair_cost(i: int, j: Optional[int]) -> float:
        if j is None:
            return float("inf")
        return cost_lu.get(
            (i, j), float(ciede2000_pair(sources[i].lab, pal_items[j].lab))
        )

    for i, s in enumerate(sources):
        sL, sC, sh = map(float, s.lch)
        if not (180.0 <= sh <= 220.0 and sL <= 55.0 and sC >= 6.0):
            continue
        pj = assigned.get(i)
        if pj is None or (pj not in neutral_set and pj not in slatey_lowC):
            continue

        base = pair_cost(i, pj)
        cands = sorted(
            good_pool,
            key=lambda j: (
                abs(float(pal_lch[j, 0]) - sL),
                hue_diff_deg(sh, float(pal_lch[j, 2])),
                -float(pal_lch[j, 1]),
            ),
        )
        cands = [j for j in cands if float(pal_lch[j, 0]) <= sL + 4.0]
        if not cands:
            continue

        TOL = 22.0
        for g in cands[:8]:
            if float(pal_lch[g, 1]) < max(10.0, 0.70 * sC):
                continue
            alt = pair_cost(i, g)
            if alt <= base + TOL:
                assigned[i] = int(g)
                cost_lu[(i, int(g))] = float(alt)
                break


def keep_blue_teal_postfix(
    src_lab: np.ndarray,
    src_lch: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    tol: float = 8.0,
) -> None:
    n_pal = int(pal_lch.shape[0])
    neutral = {j for j in range(n_pal) if float(pal_lch[j, 1]) <= NEUTRAL_C_MAX}
    slate = {
        j
        for j in range(n_pal)
        if SLATE_HUE_MIN <= float(pal_lch[j, 2]) <= SLATE_HUE_MAX
    }
    blue_targets = [
        j
        for j in range(n_pal)
        if (BLUE_BAND_MIN <= float(pal_lch[j, 2]) <= BLUE_BAND_MAX)
        and (float(pal_lch[j, 1]) > NEUTRAL_C_MAX)
    ]
    if not blue_targets:
        return

    def pair_cost(ii: int, jj: int) -> float:
        return cost_lu.get((ii, jj), float(ciede2000_pair(src_lab[ii], pal_lab[jj])))

    for i, pj in list(assigned.items()):
        sL, sC, sh = map(float, src_lch[i])
        cool = BLUE_BAND_MIN <= sh <= BLUE_BAND_MAX
        if not (cool and sC > 4.0):
            continue

        on_neutral = pj in neutral
        on_low_slate = (pj in slate) and (
            float(pal_lch[pj, 1]) < max(NEUTRAL_C_MAX + 0.1, sC * BLUE_KEEP_RATIO)
        )
        if not (on_neutral or on_low_slate):
            continue

        base = pair_cost(i, pj)
        candidates = sorted(
            blue_targets,
            key=lambda g: (
                abs(float(pal_lch[g, 0]) - sL),
                hue_diff_deg(sh, float(pal_lch[g, 2])),
            ),
        )[:8]

        for g in candidates:
            if float(pal_lch[g, 1]) < max(
                NEUTRAL_C_MAX + 0.1, sC * BLUE_KEEP_RATIO * 0.9
            ):
                continue
            alt = pair_cost(i, g)
            if alt <= base + tol:
                assigned[i] = int(g)
                cost_lu[(i, int(g))] = float(alt)
                break


__all__ = [
    "rebalance_neutral_greys",
    "spread_grey_collisions",
    "nudge_warm_darks_off_slate",
    "nudge_cool_lights_off_neutral_and_slate",
    "nudge_dark_neutrals_to_slate",
    "nudge_cool_darks_off_slate",
    "keep_blue_teal_postfix",
]
