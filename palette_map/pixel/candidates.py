# palette_map/pixel/candidates.py
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

from ..analysis import ciede2000_pair, hue_diff_deg


def candidate_row_for_source(
    i: int,
    s_lab_row: np.ndarray,
    s_lch_row: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
) -> Tuple[int, List[Tuple[float, int]]]:
    NEUTRAL_C_MAX = 3.0
    BLUE_MIN, BLUE_MAX = 190.0, 240.0
    COOL_SLATE_MIN, COOL_SLATE_MAX = 200.0, 225.0
    SAT_C = 18.0

    sL, sC, sh = float(s_lch_row[0]), float(s_lch_row[1]), float(s_lch_row[2])
    saturated = sC >= SAT_C
    hue_w = min(1.0, max(0.0, sC / 25.0))

    rows: List[Tuple[float, int]] = []
    P = pal_lab.shape[0]

    for j in range(P):
        tL = float(pal_lch[j, 0])
        tC = float(pal_lch[j, 1])
        th = float(pal_lch[j, 2])

        de = float(ciede2000_pair(s_lab_row, pal_lab[j]))
        dh = hue_diff_deg(sh, th)

        de += 0.08 * hue_w * (dh / 30.0)

        if sL < 35.0 and tL > sL + 10.0:
            de += 0.4 * (tL - (sL + 10.0))

        if tC <= NEUTRAL_C_MAX and sC >= 10.0:
            de += 6.0 if saturated else 3.0

        grow_allow = 4.0 if sC < 12.0 else 2.0
        if tC > sC + grow_allow:
            de += (0.06 if sC < 12.0 else 0.12) * (tC - (sC + grow_allow))

        if BLUE_MIN <= sh <= BLUE_MAX:
            if tC <= NEUTRAL_C_MAX:
                de += 8.0
            if sC >= 10.0 and tC < 0.5 * sC:
                de += 2.0
            if sh < 220.0 and 230.0 <= th <= 270.0:
                de += 2.0
            if 195.0 <= th <= 235.0 and (tC >= max(6.0, 0.6 * sC)):
                de -= 0.8

        if COOL_SLATE_MIN <= sh <= COOL_SLATE_MAX and sC >= 8.0 and sL >= 55.0:
            if tC <= NEUTRAL_C_MAX:
                de += 5.0
            if 200.0 <= th <= 235.0 and tC <= 12.0:
                de -= 0.6

        if sL >= 65.0 and tL + 3.0 < sL:
            de += 0.15 * (sL - (tL + 3.0))

        rows.append((de, j))

    rows.sort(key=lambda t: t[0])
    return i, rows

def build_candidate_rows(
    lab_u: np.ndarray,
    lch_u: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    workers: int = 0,
) -> Tuple[Dict[int, List[Tuple[float, int]]], Dict[Tuple[int, int], float]]:
    U = lab_u.shape[0]
    rows_by_i: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(U)}
    cost_lu: Dict[Tuple[int, int], float] = {}

    if workers > 1 and U >= 32:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(candidate_row_for_source, i, lab_u[i], lch_u[i], pal_lab, pal_lch)
                for i in range(U)
            ]
            for fu in as_completed(futs):
                i2, rows = fu.result()
                rows_by_i[i2] = rows
                for c, j in rows:
                    cost_lu[(i2, j)] = c
    else:
        for i in range(U):
            _, rows = candidate_row_for_source(i, lab_u[i], lch_u[i], pal_lab, pal_lch)
            rows_by_i[i] = rows
            for c, j in rows:
                cost_lu[(i, j)] = c

    return rows_by_i, cost_lu

__all__ = ["candidate_row_for_source", "build_candidate_rows"]
