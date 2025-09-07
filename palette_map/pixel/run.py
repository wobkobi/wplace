# palette_map/pixel/run.py
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from palette_map.core_types import U8Image, U8Mask, Lab, Lch
from palette_map.colour_convert import rgb_to_lab, lab_to_lch, delta_e2000_vec


# ----------------- utilities --------------------------------------------------


def _unique_visible_with_inverse(
    rgb: U8Image, alpha: U8Mask
) -> Tuple[U8Image, np.ndarray, np.ndarray]:
    vis = alpha > 0
    if not np.any(vis):
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    flat = rgb[vis].reshape(-1, 3)
    uniq, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    return (
        uniq.astype(np.uint8, copy=False),
        counts.astype(np.int64, copy=False),
        inv.astype(np.int64, copy=False),
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return float(v[-1])
    pos = q * cw[-1]
    idx = np.searchsorted(cw, pos, side="left")
    return float(v[min(idx, len(v) - 1)])


def _prefilter_topk(s_lab: np.ndarray, pal_lab: Lab, k: int) -> np.ndarray:
    diff = pal_lab - s_lab
    de2 = np.sum(diff * diff, axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def _hue_dist_deg(h1: float, h2: np.ndarray) -> np.ndarray:
    dh = np.abs(h2 - h1)
    return np.where(dh > 180.0, 360.0 - dh, dh)


# ----------------- neutral / grey-ish detection ------------------------------


def _neutral_indices(pal_lch: Lch) -> np.ndarray:
    """
    Build a neutral pool:
      - Prefer chroma <= 10 (tight), else fall back to the lowest-chroma quartile.
      - Always include darkest & brightest entries (for coverage).
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


def _greyish_sources(src_rgb: U8Image, src_lch: Lch) -> np.ndarray:
    """
    Tighter grey-ish gate:
      - Small RGB channel spread (<= 20)
      - Low-ish chroma (C <= 22)
      - AND hue near a neutral axis (|h| or |h-180| <= 18°) unless C<=12
    """
    r = src_rgb[:, 0].astype(np.int16)
    g = src_rgb[:, 1].astype(np.int16)
    b = src_rgb[:, 2].astype(np.int16)
    max_delta = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    C = src_lch[:, 1]
    H = src_lch[:, 2]
    # distance to neutral axes (0°, 180°)
    d0 = np.minimum(H, 360.0 - H)
    d180 = np.minimum(np.abs(H - 180.0), 360.0 - np.abs(H - 180.0))
    near_neutral_axis = (np.minimum(d0, d180) <= 18.0) | (C <= 12.0)
    return (max_delta <= 20) & (C <= 22.0) & near_neutral_axis


# ----------------- multi-run (baseline behaviour) ----------------------------


def _score_one_run(
    src_rgb: U8Image,
    src_lab: Lab,
    src_lch: Lch,
    pal_lab: Lab,
    pal_lch: Lch,
    *,
    flavor: str,
    topk: int,
) -> np.ndarray:
    P = pal_lab.shape[0]
    Ns = src_lab.shape[0]
    out = np.empty((Ns,), dtype=np.int32)

    for i in range(Ns):
        sL = float(src_lch[i, 0])
        sC = float(src_lch[i, 1])
        sh = float(src_lch[i, 2])

        idxs = _prefilter_topk(src_lab[i], pal_lab, min(topk, P))
        cand_lab = pal_lab[idxs]
        cand_lch = pal_lch[idxs]

        dE = delta_e2000_vec(src_lab[i], cand_lab).astype(np.float32, copy=False)
        cost = dE.copy()

        cL = cand_lch[:, 0]
        cC = cand_lch[:, 1]
        cH = cand_lch[:, 2]

        if flavor == "light":
            cost += 0.15 * np.abs(cL - sL).astype(np.float32)
        elif flavor == "hue":
            hue_w = min(1.0, sC / 40.0)
            dh = (_hue_dist_deg(sh, cH) / 180.0).astype(np.float32)
            cost += 0.08 * hue_w * dh
        elif flavor == "chroma":
            cost += 0.06 * np.abs(cC - sC).astype(np.float32)
        elif flavor == "neutral":
            if sC < 12.0:
                cost += 0.6 * np.maximum(0.0, cC - sC).astype(np.float32)
                cost += (cC > 16.0).astype(np.float32) * 3.0

        out[i] = int(idxs[int(np.argmin(cost))])

    return out


def _ensemble_pick(
    src_lab: Lab,
    src_lch: Lch,
    pal_lab: Lab,
    pal_lch: Lch,
    choices: List[np.ndarray],
) -> np.ndarray:
    Ns = src_lab.shape[0]
    out = np.empty((Ns,), dtype=np.int32)
    tH_all = pal_lch[:, 2]

    for i in range(Ns):
        cand_js = np.array([c[i] for c in choices], dtype=np.int32)
        cand_labs = pal_lab[cand_js]
        dEs = np.array(
            [
                delta_e2000_vec(src_lab[i], cand_labs[j : j + 1])[0]
                for j in range(cand_labs.shape[0])
            ],
            dtype=np.float32,
        )
        best = int(np.argmin(dEs))
        j_best = int(cand_js[best])
        de_best = float(dEs[best])

        near = dEs <= (de_best + 0.10)
        if np.count_nonzero(near) > 1:
            sH = float(src_lch[i, 2])
            js = cand_js[near]
            dh = np.abs(_hue_dist_deg(sH, tH_all[js]))
            j_best = int(js[int(np.argmin(dh))])

        out[i] = j_best

    return out


# ----------------- unique neutral assignment for grey-ish --------------------


def _greedy_unique_greys(
    grey_idx: np.ndarray,
    neutral_idx: np.ndarray,
    src_lab: Lab,
    src_lch: Lch,
    counts: np.ndarray,
    pal_lab: Lab,
    pal_lch: Lch,
) -> Dict[int, int]:
    """
    Assign grey-ish sources to distinct neutral palette entries *when safe*.

    Gates:
      - L* proximity: consider only unused neutrals with |ΔL| <= 22
      - ΔE proximity: force uniqueness only if an unused neutral is within +2.0 ΔE of
        the *best neutral* for that source.
      - Chromatic escape hatch: if the *global best* (over the whole palette) beats the
        best neutral by > 0.75 ΔE, use the global best instead (don't force a neutral).
    """
    mapping: Dict[int, int] = {}
    if grey_idx.size == 0 or neutral_idx.size == 0:
        return mapping

    order = np.argsort(-counts[grey_idx])
    taken: set[int] = set()
    palL = pal_lch[:, 0].astype(np.float32, copy=False)

    for k in order:
        i = int(grey_idx[k])

        # --- compare best-neutral vs best-overall
        de_neu_all = delta_e2000_vec(src_lab[i], pal_lab[neutral_idx]).astype(
            np.float32, copy=False
        )
        r_neu_best = int(np.argmin(de_neu_all))
        j_neu_best = int(neutral_idx[r_neu_best])
        de_neu_best = float(de_neu_all[r_neu_best])

        de_full_all = delta_e2000_vec(src_lab[i], pal_lab).astype(
            np.float32, copy=False
        )
        j_full_best = int(np.argmin(de_full_all))
        de_full_best = float(de_full_all[j_full_best])

        # if a chromatic (or any) palette entry is clearly better than any neutral, prefer it
        if de_full_best + 0.75 < de_neu_best:
            mapping[i] = j_full_best
            # don't mark as taken: we didn't consume a neutral
            continue

        # otherwise, try to give each grey a unique *neutral* when it doesn't hurt
        # respect L* proximity and uniqueness
        Ls = float(src_lch[i, 0])
        dL_neu_all = np.abs(palL[neutral_idx] - Ls)
        mask_unused = np.array(
            [int(neutral_idx[r]) not in taken for r in range(neutral_idx.size)],
            dtype=bool,
        )
        allowed = np.where(mask_unused & (dL_neu_all <= 22.0))[0]

        if allowed.size > 0:
            r_allowed = allowed[int(np.argmin(de_neu_all[allowed]))]
            j_allowed = int(neutral_idx[r_allowed])
            de_allowed = float(de_neu_all[r_allowed])

            # only force uniqueness if it's close to the best-neutral
            if de_allowed <= (de_neu_best + 2.0):
                mapping[i] = j_allowed
                taken.add(j_allowed)
                continue

        # fallback: use the best-neutral (even if reusing)
        mapping[i] = j_neu_best
        taken.add(j_neu_best)

    return mapping


# ----------------- gentle whole-palette decongestion -------------------------


def _light_decongest(
    chosen_idx: np.ndarray,
    src_lab: Lab,
    counts: np.ndarray,
    pal_lab: Lab,
) -> np.ndarray:
    Ns = chosen_idx.size
    out = chosen_idx.copy()
    used: Dict[int, int] = {}
    for j in out:
        used[int(j)] = used.get(int(j), 0) + 1

    budgets = np.empty((Ns,), dtype=np.float32)
    for i in range(Ns):
        j0 = int(out[i])
        de_best = float(delta_e2000_vec(src_lab[i], pal_lab[j0 : j0 + 1])[0])
        budgets[i] = min(0.8, max(0.35, 0.45 + 0.05 * de_best))

    order = np.argsort(-counts)
    all_js = np.arange(pal_lab.shape[0], dtype=np.int32)
    for i in order:
        j0 = int(out[i])
        if used.get(j0, 0) <= 1:
            continue
        de_all = delta_e2000_vec(src_lab[i], pal_lab).astype(np.float32, copy=False)
        de0 = float(de_all[j0])
        mask = de_all <= (de0 + budgets[i])
        cand = all_js[mask]
        cand = np.array([j for j in cand if used.get(int(j), 0) == 0], dtype=np.int32)
        if cand.size == 0:
            continue
        j_new = int(cand[int(np.argmin(de_all[cand]))])
        out[i] = j_new
        used[j0] -= 1
        used[j_new] = 1

    return out


# ----------------- public entry ----------------------------------------------


def run_pixel(
    img_rgb: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    *,
    debug: bool = False,
    workers: int = 1,  # signature compatibility
) -> U8Image:
    uniq_rgb, counts, inv = _unique_visible_with_inverse(img_rgb, alpha)
    if uniq_rgb.shape[0] == 0:
        return img_rgb.copy()

    s_lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    s_lch = lab_to_lch(s_lab)

    P = int(pal_lab.shape[0])
    topk = 16 if P >= 32 else max(8, P // 2)

    # multi-run flavours
    flavors = ["base", "light", "hue", "chroma", "neutral"]
    per_run_idx: List[np.ndarray] = []
    stats = []
    for fl in flavors:
        idx = _score_one_run(
            uniq_rgb,
            s_lab,
            s_lch,
            pal_lab.astype(np.float32),
            pal_lch.astype(np.float32),
            flavor=fl,
            topk=topk,
        )
        per_run_idx.append(idx)
        labs = pal_lab[idx]
        dE = np.array(
            [
                delta_e2000_vec(s_lab[i], labs[i : i + 1])[0]
                for i in range(s_lab.shape[0])
            ],
            dtype=np.float32,
        )
        mean_dE = float(np.average(dE, weights=counts))
        p90 = _weighted_percentile(dE, counts, 0.90)
        mx = float(dE.max()) if dE.size else 0.0
        shares = int(uniq_rgb.shape[0] - np.unique(idx).size)
        stats.append((fl, mean_dE, p90, mx, shares))

    chosen = _ensemble_pick(
        s_lab,
        s_lch,
        pal_lab.astype(np.float32),
        pal_lch.astype(np.float32),
        per_run_idx,
    )

    # --- Stage 1: safe unique assignment for grey-ish vs neutrals -------------
    grey_mask = _greyish_sources(uniq_rgb, s_lch)
    g_idx = np.where(grey_mask)[0]
    neutrals = _neutral_indices(pal_lch.astype(np.float32))
    g_map = _greedy_unique_greys(
        g_idx,
        neutrals,
        s_lab,
        s_lch,
        counts,
        pal_lab.astype(np.float32),
        pal_lch.astype(np.float32),
    )

    chosen2 = chosen.copy()
    for si, pj in g_map.items():
        chosen2[int(si)] = int(pj)

    # --- Stage 2: gentle decongestion across the rest -------------------------
    chosen3 = _light_decongest(chosen2, s_lab, counts, pal_lab.astype(np.float32))

    # ---- debug prints --------------------------------------------------------
    if debug:
        for fl, md, p90, mx, sh in stats:
            print(
                f"[debug] run {fl:<12} mean dE={md:.3f}  p90={p90:.2f}  max={mx:.2f}  shares={sh}",
                flush=True,
            )

        labs_fin = pal_lab[chosen3]
        dE_fin = np.array(
            [
                delta_e2000_vec(s_lab[i], labs_fin[i : i + 1])[0]
                for i in range(s_lab.shape[0])
            ],
            dtype=np.float32,
        )
        md = float(np.average(dE_fin, weights=counts))
        p90 = _weighted_percentile(dE_fin, counts, 0.90)
        mx = float(dE_fin.max()) if dE_fin.size else 0.0
        sh = int(uniq_rgb.shape[0] - np.unique(chosen3).size)
        print(
            f"[debug] chosen ensemble  mean dE={md:.3f}  p90={p90:.2f}  max={mx:.2f}  shares={sh}",
            flush=True,
        )

        # “important mappings” (top 12 by ΔE)
        t_lab = pal_lab[chosen3]
        t_lch = lab_to_lch(t_lab.astype(np.float32))
        order = np.argsort(
            -np.array(
                [
                    delta_e2000_vec(s_lab[i], t_lab[i : i + 1])[0]
                    for i in range(s_lab.shape[0])
                ]
            )
        )[:12]
        if order.size > 0:
            print("[debug] important mappings:", flush=True)
            for i in order:
                sr = uniq_rgb[i]
                tr = pal_rgb[chosen3[i]]
                sL, sC, shh = float(s_lch[i, 0]), float(s_lch[i, 1]), float(s_lch[i, 2])
                tL, tC, th = float(t_lch[i, 0]), float(t_lch[i, 1]), float(t_lch[i, 2])
                dh = th - shh
                while dh > 180.0:
                    dh -= 360.0
                while dh < -180.0:
                    dh += 360.0
                print(
                    f"  src #{sr[0]:02x}{sr[1]:02x}{sr[2]:02x}  count={int(counts[i]):5d} "
                    f"-> #{tr[0]:02x}{tr[1]:02x}{tr[2]:02x}  "
                    f"[dE={delta_e2000_vec(s_lab[i], t_lab[i:i+1])[0]:5.2f}, "
                    f"dL={tL - sL:+6.3f}, dC={tC - sC:+6.3f}, dh={dh:4.1f} deg]",
                    flush=True,
                )

        if g_idx.size:
            # how many actually landed on neutral entries?
            neutral_set = set(
                int(j) for j in _neutral_indices(pal_lch.astype(np.float32))
            )
            assigned_neutral = sum(
                1 for si, pj in g_map.items() if int(pj) in neutral_set
            )
            print(
                f"[debug] grey unique assign: {assigned_neutral}/{g_idx.size} greys mapped to distinct neutrals",
                flush=True,
            )

    # ---- materialise per-pixel ----------------------------------------------
    out = img_rgb.copy()
    vis = alpha > 0
    if np.any(vis):
        pal_map_rgb = pal_rgb[chosen3]
        out[vis] = pal_map_rgb[inv].astype(np.uint8, copy=False)
    return out
