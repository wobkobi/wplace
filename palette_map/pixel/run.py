# palette_map/pixel/run.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from palette_map.core_types import U8Image, U8Mask, Lab, Lch, hue_diff_deg
from palette_map.color_convert import rgb_to_lab, lab_to_lch


# ----------------------------- ΔE2000 -----------------------------------------


def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - math.sqrt((Cm**7) / (Cm**7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    h1p = (
        0.0
        if (a1p == 0.0 and b1 == 0.0)
        else (math.degrees(math.atan2(b1, a1p)) % 360.0)
    )
    h2p = (
        0.0
        if (a2p == 0.0 and b2 == 0.0)
        else (math.degrees(math.atan2(b2, a2p)) % 360.0)
    )

    dLp = L2 - L1
    dCp = C2p - C1p

    dh = h2p - h1p
    if (C1p * C2p) == 0.0:
        dhp = 0.0
    elif dh > 180.0:
        dhp = dh - 360.0
    elif dh < -180.0:
        dhp = dh + 360.0
    else:
        dhp = dh
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(0.5 * dhp))

    Lpm = 0.5 * (L1 + L2)
    Cpm = 0.5 * (C1p + C2p)

    if (C1p * C2p) == 0.0:
        hpm = h1p + h2p
    else:
        d = abs(h1p - h2p)
        if d <= 180.0:
            hpm = 0.5 * (h1p + h2p)
        else:
            hpm = 0.5 * (h1p + h2p + (360.0 if (h1p + h2p) < 360.0 else -360.0))

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hpm - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hpm))
        + 0.32 * math.cos(math.radians(3.0 * hpm + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hpm - 63.0))
    )
    d_ro = 30.0 * math.exp(-(((hpm - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * math.sqrt((Cpm**7) / (Cpm**7 + 25.0**7))

    Sl = 1.0 + (0.015 * ((Lpm - 50.0) ** 2.0)) / math.sqrt(20.0 + (Lpm - 50.0) ** 2.0)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc

    return float(
        math.sqrt(
            (dLp / Sl) ** 2.0
            + (dCp / Sc) ** 2.0
            + (dHp / Sh) ** 2.0
            + Rt * (dCp / Sc) * (dHp / Sh)
        )
    )


# ------------------------------ helpers ----------------------------------------


def _unique_visible(rgb: U8Image, alpha: U8Mask) -> Tuple[np.ndarray, np.ndarray]:
    vis = alpha > 0
    if not np.any(vis):
        return np.empty((0, 3), np.uint8), np.empty((0,), np.int64)
    flat = rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return uniq.astype(np.uint8), counts.astype(np.int64)


def _apply_mapping(
    img_rgb: U8Image,
    alpha: U8Mask,
    src_uniq: np.ndarray,
    tgt_rgb: np.ndarray,
) -> U8Image:
    out = img_rgb.copy()
    vis = alpha > 0
    if not np.any(vis):
        return out

    flat_vis = img_rgb[vis].reshape(-1, 3).astype(np.uint32, copy=False)
    keys_vis = (flat_vis[:, 0] << 16) | (flat_vis[:, 1] << 8) | flat_vis[:, 2]
    src_keys = (
        (src_uniq[:, 0].astype(np.uint32) << 16)
        | (src_uniq[:, 1].astype(np.uint32) << 8)
        | src_uniq[:, 2].astype(np.uint32)
    )
    key_to_idx: Dict[int, int] = {int(k): i for i, k in enumerate(src_keys.tolist())}
    idxs = np.fromiter(
        (key_to_idx[int(k)] for k in keys_vis.tolist()),
        count=keys_vis.size,
        dtype=np.int32,
    )
    mapped = tgt_rgb[idxs]
    out[vis] = mapped.reshape((-1, 3)).astype(np.uint8, copy=False)
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    pos = q * float(cw[-1])
    idx = int(np.searchsorted(cw, pos, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _de_matrix(src_lab: Lab, pal_lab: Lab) -> np.ndarray:
    S, P = src_lab.shape[0], pal_lab.shape[0]
    de = np.empty((S, P), dtype=np.float32)
    for i in range(S):
        for j in range(P):
            de[i, j] = ciede2000_pair(src_lab[i], pal_lab[j])
    return de


def _build_costs(
    src_lch: Lch, pal_lch: Lch, base_de: np.ndarray, mode: str
) -> np.ndarray:
    S, P = base_de.shape
    costs = base_de.astype(np.float32).copy()

    sL = src_lch[:, 0][:, None]  # (S,1)
    sC = src_lch[:, 1][:, None]  # (S,1)
    tL = pal_lch[:, 0][None, :]  # (1,P)
    tC = pal_lch[:, 1][None, :]  # (1,P)

    if mode == "light":
        costs += 0.03 * np.abs(tL - sL)

    elif mode == "hue":
        # pairwise hue penalty, scaled by source chroma so neutrals aren’t over-penalized
        dh = np.empty((S, P), dtype=np.float32)
        for i in range(S):
            hi = float(src_lch[i, 2])
            for j in range(P):
                dh[i, j] = hue_diff_deg(hi, float(pal_lch[j, 2]))
        hue_w = np.minimum(1.0, sC / 30.0)
        costs += 0.12 * hue_w * (dh / 10.0)

    elif mode == "chroma":
        # discourage mapping to lower chroma than source
        costs += 0.04 * np.maximum(0.0, sC - tC)

    elif mode == "neutral":
        # strongly prefer neutral→neutral, lightly tie-break by lightness
        neutral_src = (sC <= 8.0).astype(np.float32)
        neutral_tgt = (tC <= 12.0).astype(np.float32)
        costs += (
            0.30 * (1.0 - neutral_tgt) * neutral_src
        )  # penalty if pushing neutral src to chroma tgt
        costs -= 1.20 * (neutral_src * neutral_tgt)  # bonus for neutral→neutral
        costs += 0.02 * neutral_src * np.abs(tL - sL)  # tie-break by L closeness

    return costs


# -------- soft-unique + de-share that protects neutrals from blending ----------


def _assign_soft_unique(
    costs: np.ndarray,
    base_de: np.ndarray,
    src_lch: Lch,
    pal_lch: Lch,
    counts: np.ndarray,
) -> np.ndarray:
    S, _ = costs.shape
    order = np.argsort(-counts)  # large areas first
    assigned = -np.ones(S, dtype=np.int32)
    used: set[int] = set()

    sC = src_lch[:, 1]
    tC_all = pal_lch[:, 1]

    SRC_NEUTRAL = 10.0
    TGT_NEUTRAL = 12.0
    UNIQUE_TOL = 1.25
    UNIQUE_TOL_NEUTRAL = 2.75  # allow a bigger ΔE rise to keep neutrals distinct

    for i in order:
        row = costs[i]
        ranked = np.argsort(row)
        j_best = int(ranked[0])

        j_unused = next((int(j) for j in ranked if j not in used), None)
        if j_unused is None:
            assigned[i] = j_best
            used.add(j_best)
            continue

        de_best = float(base_de[i, j_best])
        de_unused = float(base_de[i, j_unused])

        src_is_neutral = float(sC[i]) <= SRC_NEUTRAL
        best_is_neutral = float(tC_all[j_best]) <= TGT_NEUTRAL
        unused_is_chroma = float(tC_all[j_unused]) > TGT_NEUTRAL

        tol = UNIQUE_TOL_NEUTRAL if src_is_neutral else UNIQUE_TOL
        take_unused = (de_unused <= de_best + tol) and not (
            src_is_neutral and best_is_neutral and unused_is_chroma
        )

        if take_unused:
            assigned[i] = j_unused
            used.add(j_unused)
        else:
            assigned[i] = j_best
            used.add(j_best)

    return assigned


def _break_shares_neutral_aware(
    assigned: np.ndarray,
    base_de: np.ndarray,
    src_lch: Lch,
    pal_lch: Lch,
    counts: np.ndarray,
) -> np.ndarray:
    S = assigned.size
    used = set(int(j) for j in assigned.tolist())
    groups: Dict[int, List[int]] = {}
    for i, j in enumerate(assigned.tolist()):
        groups.setdefault(int(j), []).append(i)

    SRC_NEUTRAL = 10.0
    TGT_NEUTRAL = 12.0
    MAX_DE_INCREASE_NEUTRAL = 3.0  # more generous for neutrals
    MAX_DE_INCREASE_CHROMA = 1.5

    chip_weight: Dict[int, int] = {}
    for j, idxs in groups.items():
        chip_weight[j] = int(np.sum(counts[idxs])) if len(idxs) > 1 else 0

    for chip, idxs in sorted(groups.items(), key=lambda kv: -chip_weight[kv[0]]):
        if len(idxs) < 2:
            continue
        idxs_sorted = sorted(idxs, key=lambda i: float(base_de[i, chip]))
        keep = idxs_sorted[0]
        for i in idxs_sorted[1:]:
            cur_de = float(base_de[i, chip])
            src_neutral = float(src_lch[i, 1]) <= SRC_NEUTRAL
            tol = MAX_DE_INCREASE_NEUTRAL if src_neutral else MAX_DE_INCREASE_CHROMA

            # search for best unused alternative under tolerance
            for j_alt in np.argsort(base_de[i]).tolist():
                if j_alt == chip or j_alt in used:
                    continue
                tgt_neutral = float(pal_lch[j_alt, 1]) <= TGT_NEUTRAL
                if src_neutral and not tgt_neutral:
                    continue  # keep neutrals on neutral chips
                new_de = float(base_de[i, j_alt])
                if new_de <= cur_de + tol:
                    assigned[i] = int(j_alt)
                    used.add(int(j_alt))
                    break  # moved
            # else keep sharing

    return assigned


def _deshare_neutrals_global(
    assigned: np.ndarray,
    base_de: np.ndarray,
    src_lch: Lch,
    pal_lch: Lch,
    counts: np.ndarray,
) -> np.ndarray:
    """Global pass: move neutral sources to unused neutral chips if ΔE rise is small."""
    S = assigned.size
    used = set(int(j) for j in assigned.tolist())
    SRC_NEUTRAL = 10.0
    TGT_NEUTRAL = 12.0
    TOL = 3.0

    # build candidate moves (delta, large count first negative, i, j_alt)
    moves: List[Tuple[float, int, int, int]] = []
    for i in range(S):
        if float(src_lch[i, 1]) > SRC_NEUTRAL:
            continue
        j_cur = int(assigned[i])
        cur_de = float(base_de[i, j_cur])
        # if already unique neutral, skip
        if float(pal_lch[j_cur, 1]) <= TGT_NEUTRAL and (
            assigned.tolist().count(j_cur) == 1
        ):
            continue
        # best unused neutral
        for j_alt in np.argsort(base_de[i]).tolist():
            if float(pal_lch[j_alt, 1]) > TGT_NEUTRAL:
                continue
            if j_alt in used:
                continue
            new_de = float(base_de[i, j_alt])
            delta = new_de - cur_de
            if delta <= TOL:
                moves.append((delta, -int(counts[i]), i, int(j_alt)))
                break

    moves.sort()
    for _, _, i, j in moves:
        if j in used:
            continue
        assigned[i] = j
        used.add(j)
    return assigned


# ---------------------- single run, ensemble, and selection --------------------


def _run_variant(
    mode: str,
    base_de: np.ndarray,
    src_lch: Lch,
    pal_lch: Lch,
    counts: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, int]:
    costs = _build_costs(src_lch, pal_lch, base_de, mode)
    assigned = _assign_soft_unique(costs, base_de, src_lch, pal_lch, counts)
    assigned = _break_shares_neutral_aware(assigned, base_de, src_lch, pal_lch, counts)
    assigned = _deshare_neutrals_global(assigned, base_de, src_lch, pal_lch, counts)

    S = base_de.shape[0]
    de = np.array([base_de[i, int(assigned[i])] for i in range(S)], dtype=np.float32)
    mean_de = float(np.average(de, weights=counts))
    p90 = _weighted_quantile(de, counts, 0.90)
    mxx = float(np.max(de)) if S else 0.0

    hist: Dict[int, int] = {}
    for j in assigned.tolist():
        hist[j] = hist.get(j, 0) + 1
    shares = sum(1 for v in hist.values() if v > 1)
    return assigned, mean_de, p90, mxx, shares


def _ensemble_assign(
    assigned_runs: List[np.ndarray],
    base_de: np.ndarray,
    counts: np.ndarray,
    src_lch: Lch,
    pal_lch: Lch,
) -> Tuple[np.ndarray, float, float, float, int]:
    S = base_de.shape[0]
    cand: List[List[int]] = [list({int(r[i]) for r in assigned_runs}) for i in range(S)]
    for i in range(S):
        if not cand[i]:
            cand[i] = [int(np.argmin(base_de[i]))]

    order = np.argsort(-counts)
    assigned = -np.ones(S, dtype=np.int32)
    used: set[int] = set()

    for i in order:
        de_vals = sorted(
            ((j, float(base_de[i, j])) for j in cand[i]), key=lambda x: x[1]
        )
        j_pick = next((j for j, _ in de_vals if j not in used), None)
        if j_pick is None:
            j_pick = de_vals[0][0]  # share for now
        assigned[i] = j_pick
        used.add(j_pick)

    assigned = _break_shares_neutral_aware(assigned, base_de, src_lch, pal_lch, counts)

    de = np.array([base_de[i, int(assigned[i])] for i in range(S)], dtype=np.float32)
    mean_de = float(np.average(de, weights=counts))
    p90 = _weighted_quantile(de, counts, 0.90)
    mxx = float(np.max(de)) if S else 0.0
    hist: Dict[int, int] = {}
    for j in assigned.tolist():
        hist[j] = hist.get(j, 0) + 1
    shares = sum(1 for v in hist.values() if v > 1)
    return assigned, mean_de, p90, mxx, shares


def _pick_final(
    single_runs: List[Tuple[str, np.ndarray, float, float, float, int]],
    ensemble: Tuple[np.ndarray, float, float, float, int],
) -> Tuple[str, np.ndarray, float, float, float, int]:
    best_idx = min(range(len(single_runs)), key=lambda k: single_runs[k][2])
    best = single_runs[best_idx]
    ens_assigned, ens_mean, ens_p90, ens_max, ens_shares = ensemble
    _, _, b_mean, b_p90, b_max, b_shares = best
    if (ens_mean <= b_mean) or (ens_p90 + 0.5 < b_p90) or (ens_max + 1.0 < b_max):
        return ("ensemble", ens_assigned, ens_mean, ens_p90, ens_max, ens_shares)
    return (
        f"single:{single_runs[best_idx][0]}",
        best[1],
        b_mean,
        b_p90,
        b_max,
        b_shares,
    )


# ----------------------------------- main --------------------------------------


def run_pixel(
    rgb_in: U8Image,
    alpha: U8Mask,
    pal_rgb: np.ndarray,
    pal_lab: Lab,
    pal_lch: Lch,
    debug: bool = False,
) -> U8Image:
    uniq_rgb, counts = _unique_visible(rgb_in, alpha)
    if uniq_rgb.size == 0:
        return rgb_in

    src_lab: Lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    src_lch: Lch = lab_to_lch(src_lab.astype(np.float32))
    base_de = _de_matrix(src_lab, pal_lab)

    modes = ["base", "light", "hue", "chroma", "neutral"]
    single_runs: List[Tuple[str, np.ndarray, float, float, float, int]] = []
    assigned_runs: List[np.ndarray] = []

    for m in modes:
        a, mean_de, p90, mxx, shares = _run_variant(
            m, base_de, src_lch, pal_lch, counts
        )
        single_runs.append((m, a, mean_de, p90, mxx, shares))
        assigned_runs.append(a)
        if debug:
            print(
                f"[debug] run {m:7s}  mean dE={mean_de:.3f}  p90={p90:.2f}  max={mxx:.2f}  shares={shares}",
                flush=True,
            )

    ens = _ensemble_assign(assigned_runs, base_de, counts, src_lch, pal_lch)
    tag, assigned, f_mean, f_p90, f_max, f_shares = _pick_final(single_runs, ens)

    if debug:
        print(
            f"[debug] chosen {tag}  mean dE={f_mean:.3f}  p90={f_p90:.2f}  max={f_max:.2f}  shares={f_shares}",
            flush=True,
        )
        de_chosen = np.array(
            [base_de[i, int(assigned[i])] for i in range(base_de.shape[0])],
            dtype=np.float32,
        )
        order = np.argsort(-de_chosen)
        top = order[: min(12, order.size)]
        print("[debug] important mappings:", flush=True)
        for i in top:
            j = int(assigned[i])
            sL, sC, sh = [float(x) for x in src_lch[i]]
            tL, tC, th = [float(x) for x in pal_lch[j]]
            dh = hue_diff_deg(sh, th)
            sr = uniq_rgb[i]
            tr = pal_rgb[j]
            print(
                f"  src #{sr[0]:02x}{sr[1]:02x}{sr[2]:02x}  "
                f"count={int(counts[i]):4d} -> "
                f"#{tr[0]:02x}{tr[1]:02x}{tr[2]:02x}  "
                f"[dE={de_chosen[i]:5.2f}, dL={tL-sL:+6.3f}, dC={tC-sC:+6.3f}, dh={dh:4.1f} deg]",
                flush=True,
            )

    tgt = np.zeros_like(uniq_rgb, dtype=np.uint8)
    for i in range(uniq_rgb.shape[0]):
        tgt[i] = pal_rgb[int(assigned[i])]
    return _apply_mapping(rgb_in, alpha, uniq_rgb, tgt)
