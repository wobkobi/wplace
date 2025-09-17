# palette_map/pixel/run.py
from __future__ import annotations

"""
Pixel-mode mapper.

Maps unique visible colours to palette entries using several scoring flavours,
then ensembles the picks, assigns greys to distinct neutral entries when safe,
and lightly decongests overused palette slots.
"""

from typing import List, Tuple, Dict
import numpy as np

from palette_map.core_types import U8Image, U8Mask, Lab, Lch
from palette_map.colour_convert import rgb_to_lab, lab_to_lch, delta_e2000_vec
from palette_map.utils import (
    weighted_percentile,
    topk_indices_by_lab_distance as prefilter_topk,
    hue_distance_degrees,
    debug_log,
)
from palette_map.colour_select import neutral_indices, greyish_sources


def _unique_visible_colours_with_inverse(
    rgb: U8Image, alpha: U8Mask
) -> Tuple[U8Image, np.ndarray, np.ndarray]:
    """
    Unique visible RGB rows with counts and inverse index.

    Returns:
      unique_rgb: uint8 [U,3]
      counts: int64 [U]
      inverse_idx: int64 [Nvis], where unique_rgb[inverse_idx] reconstructs flattened visible pixels
    """
    visible_mask = alpha > 0
    if not np.any(visible_mask):
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    flat_visible = rgb[visible_mask].reshape(-1, 3)
    unique_rgb, inverse_idx, counts = np.unique(
        flat_visible, axis=0, return_inverse=True, return_counts=True
    )
    return (
        unique_rgb.astype(np.uint8, copy=False),
        counts.astype(np.int64, copy=False),
        inverse_idx.astype(np.int64, copy=False),
    )


def _score_one_flavour(
    src_rgb: U8Image,
    src_lab: Lab,
    src_lch: Lch,
    pal_lab: Lab,
    pal_lch: Lch,
    *,
    flavour: str,
    topk: int,
) -> np.ndarray:
    """
    Score a single run flavour and pick best palette index per unique source.

    Flavours:
      base     : pure dE2000
      light    : add small lightness proximity
      hue      : add hue proximity weighted by chroma
      chroma   : add chroma proximity
      neutral  : discourage high-chroma targets when source is near-neutral
    """
    num_palette = pal_lab.shape[0]
    num_src = src_lab.shape[0]
    out = np.empty((num_src,), dtype=np.int32)

    for i in range(num_src):
        src_L = float(src_lch[i, 0])
        src_C = float(src_lch[i, 1])
        src_H = float(src_lch[i, 2])

        candidate_indices = prefilter_topk(src_lab[i], pal_lab, min(topk, num_palette))
        candidate_lab = pal_lab[candidate_indices]
        candidate_lch = pal_lch[candidate_indices]

        de_vec = delta_e2000_vec(src_lab[i], candidate_lab).astype(
            np.float32, copy=False
        )
        cost_vec = de_vec.copy()

        tgt_L = candidate_lch[:, 0]
        tgt_C = candidate_lch[:, 1]
        tgt_H = candidate_lch[:, 2]

        if flavour == "light":
            cost_vec += 0.15 * np.abs(tgt_L - src_L).astype(np.float32)
        elif flavour == "hue":
            hue_w = min(1.0, src_C / 40.0)
            d_h = (hue_distance_degrees(src_H, tgt_H) / 180.0).astype(np.float32)
            cost_vec += 0.08 * hue_w * d_h
        elif flavour == "chroma":
            cost_vec += 0.06 * np.abs(tgt_C - src_C).astype(np.float32)
        elif flavour == "neutral":
            if src_C < 12.0:
                cost_vec += 0.6 * np.maximum(0.0, tgt_C - src_C).astype(np.float32)
                cost_vec += (tgt_C > 16.0).astype(np.float32) * 3.0

        out[i] = int(candidate_indices[int(np.argmin(cost_vec))])

    return out


def _ensemble_select(
    src_lab: Lab,
    src_lch: Lch,
    pal_lab: Lab,
    pal_lch: Lch,
    choices: List[np.ndarray],
) -> np.ndarray:
    """
    Ensemble several candidate picks per source:
      1) choose the lowest dE2000 across flavours
      2) if ties within +0.10, choose the one closest in hue
    """
    num_src = src_lab.shape[0]
    out = np.empty((num_src,), dtype=np.int32)
    pal_H_all = pal_lch[:, 2]

    for i in range(num_src):
        cand_indices = np.array([c[i] for c in choices], dtype=np.int32)
        cand_labs = pal_lab[cand_indices]
        de_vec = np.array(
            [
                delta_e2000_vec(src_lab[i], cand_labs[j : j + 1])[0]
                for j in range(cand_labs.shape[0])
            ],
            dtype=np.float32,
        )
        best_rel = int(np.argmin(de_vec))
        best_idx = int(cand_indices[best_rel])
        best_de = float(de_vec[best_rel])

        within_tie = de_vec <= (best_de + 0.10)
        if np.count_nonzero(within_tie) > 1:
            src_H = float(src_lch[i, 2])
            tie_indices = cand_indices[within_tie]
            d_h = np.abs(hue_distance_degrees(src_H, pal_H_all[tie_indices]))
            best_idx = int(tie_indices[int(np.argmin(d_h))])

        out[i] = best_idx

    return out


def _assign_unique_greys(
    grey_src_indices: np.ndarray,
    neutral_palette_indices: np.ndarray,
    src_lab: Lab,
    src_lch: Lch,
    counts: np.ndarray,
    pal_lab: Lab,
    pal_lch: Lch,
) -> Dict[int, int]:
    """
    Assign grey-ish sources to distinct neutral palette entries when safe.

    Gates:
      L* proximity: consider only unused neutrals with |dL| <= 22
      dE proximity: force uniqueness only if an unused neutral is within +2.0 dE
                    of the best-neutral for that source
      Escape hatch: if global best beats best-neutral by > 0.75 dE, use global best
    """
    mapping: Dict[int, int] = {}
    if grey_src_indices.size == 0 or neutral_palette_indices.size == 0:
        return mapping

    order = np.argsort(-counts[grey_src_indices])
    taken: set[int] = set()
    pal_L_all = pal_lch[:, 0].astype(np.float32, copy=False)

    for k in order:
        i = int(grey_src_indices[k])

        de_to_neutrals = delta_e2000_vec(
            src_lab[i], pal_lab[neutral_palette_indices]
        ).astype(np.float32, copy=False)
        rel_best = int(np.argmin(de_to_neutrals))
        best_neutral_idx = int(neutral_palette_indices[rel_best])
        best_neutral_de = float(de_to_neutrals[rel_best])

        de_full = delta_e2000_vec(src_lab[i], pal_lab).astype(np.float32, copy=False)
        best_full_idx = int(np.argmin(de_full))
        best_full_de = float(de_full[best_full_idx])

        if best_full_de + 0.75 < best_neutral_de:
            mapping[i] = best_full_idx
            continue

        src_L = float(src_lch[i, 0])
        dL_neutrals = np.abs(pal_L_all[neutral_palette_indices] - src_L)
        mask_unused = np.array(
            [
                int(neutral_palette_indices[r]) not in taken
                for r in range(neutral_palette_indices.size)
            ],
            dtype=bool,
        )
        allowed = np.where(mask_unused & (dL_neutrals <= 22.0))[0]

        if allowed.size > 0:
            rel_allowed = allowed[int(np.argmin(de_to_neutrals[allowed]))]
            allowed_idx = int(neutral_palette_indices[rel_allowed])
            allowed_de = float(de_to_neutrals[rel_allowed])
            if allowed_de <= (best_neutral_de + 2.0):
                mapping[i] = allowed_idx
                taken.add(allowed_idx)
                continue

        mapping[i] = best_neutral_idx
        taken.add(best_neutral_idx)

    return mapping


def _lightly_decongest_palette(
    chosen_indices: np.ndarray,
    src_lab: Lab,
    counts: np.ndarray,
    pal_lab: Lab,
) -> np.ndarray:
    """
    Spread overused palette slots when alternatives are close enough.

    For each unique colour, compute a small tolerance based on its best dE.
    If its current palette slot is overused and there exists an unused slot
    within that tolerance, reassign to the closest such slot.
    """
    num_src = chosen_indices.size
    out = chosen_indices.copy()
    usage_counts: Dict[int, int] = {}
    for j in out:
        usage_counts[int(j)] = usage_counts.get(int(j), 0) + 1

    budgets = np.empty((num_src,), dtype=np.float32)
    for i in range(num_src):
        j0 = int(out[i])
        best_de = float(delta_e2000_vec(src_lab[i], pal_lab[j0 : j0 + 1])[0])
        budgets[i] = min(0.8, max(0.35, 0.45 + 0.05 * best_de))

    order = np.argsort(-counts)
    all_indices = np.arange(pal_lab.shape[0], dtype=np.int32)
    for i in order:
        j0 = int(out[i])
        if usage_counts.get(j0, 0) <= 1:
            continue
        de_all = delta_e2000_vec(src_lab[i], pal_lab).astype(np.float32, copy=False)
        de0 = float(de_all[j0])
        mask = de_all <= (de0 + budgets[i])
        candidates = all_indices[mask]
        candidates = np.array(
            [j for j in candidates if usage_counts.get(int(j), 0) == 0], dtype=np.int32
        )
        if candidates.size == 0:
            continue
        j_new = int(candidates[int(np.argmin(de_all[candidates]))])
        out[i] = j_new
        usage_counts[j0] -= 1
        usage_counts[j_new] = 1

    return out


def run_pixel(
    img_rgb: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    *,
    debug: bool = False,
    workers: int = 1,  # kept for API parity
) -> U8Image:
    """
    Pixel-mode mapping entry.

    Steps:
      1) unique visible colours
      2) multiple scoring flavours
      3) ensemble selection
      4) unique grey assignment to neutrals
      5) light decongestion
      6) materialise per-pixel
    """
    unique_rgb, counts, inverse_idx = _unique_visible_colours_with_inverse(
        img_rgb, alpha
    )
    if unique_rgb.shape[0] == 0:
        return img_rgb.copy()

    src_lab = rgb_to_lab(unique_rgb.astype(np.float32))
    src_lch = lab_to_lch(src_lab)

    num_palette = int(pal_lab.shape[0])
    topk = 16 if num_palette >= 32 else max(8, num_palette // 2)

    flavours = ["base", "light", "hue", "chroma", "neutral"]
    per_flavour_indices: List[np.ndarray] = []
    flavour_stats = []
    for fl in flavours:
        idx = _score_one_flavour(
            unique_rgb,
            src_lab,
            src_lch,
            pal_lab.astype(np.float32),
            pal_lch.astype(np.float32),
            flavour=fl,
            topk=topk,
        )
        per_flavour_indices.append(idx)
        labs = pal_lab[idx]
        de_vec = np.array(
            [
                delta_e2000_vec(src_lab[i], labs[i : i + 1])[0]
                for i in range(src_lab.shape[0])
            ],
            dtype=np.float32,
        )
        mean_de = float(np.average(de_vec, weights=counts))
        p90 = weighted_percentile(de_vec, counts, 0.90)
        mx = float(de_vec.max()) if de_vec.size else 0.0
        shares = int(unique_rgb.shape[0] - np.unique(idx).size)
        flavour_stats.append((fl, mean_de, p90, mx, shares))

    chosen_indices = _ensemble_select(
        src_lab,
        src_lch,
        pal_lab.astype(np.float32),
        pal_lch.astype(np.float32),
        per_flavour_indices,
    )

    grey_mask = greyish_sources(unique_rgb, src_lch)  # returns a boolean mask
    grey_indices = np.where(grey_mask)[0]
    neutral_indices_list = neutral_indices(pal_lch.astype(np.float32))
    grey_assign_map = _assign_unique_greys(
        grey_indices,
        neutral_indices_list,
        src_lab,
        src_lch,
        counts,
        pal_lab.astype(np.float32),
        pal_lch.astype(np.float32),
    )

    chosen_after_grey = chosen_indices.copy()
    for src_i, pal_j in grey_assign_map.items():
        chosen_after_grey[int(src_i)] = int(pal_j)

    chosen_after_decongest = _lightly_decongest_palette(
        chosen_after_grey, src_lab, counts, pal_lab.astype(np.float32)
    )

    if debug:
        debug_log("pixel stats per flavour:")
        debug_log("  flavour        mean dE    p90     max   shares")
        for fl, md, p90, mx, sh in flavour_stats:
            debug_log(f"  {fl:<12}  {md:7.3f}  {p90:7.2f}  {mx:7.2f}  {sh:6d}")

        labs_final = pal_lab[chosen_after_decongest]
        de_vec_final = np.array(
            [
                delta_e2000_vec(src_lab[i], labs_final[i : i + 1])[0]
                for i in range(src_lab.shape[0])
            ],
            dtype=np.float32,
        )
        md = float(np.average(de_vec_final, weights=counts))
        p90 = weighted_percentile(de_vec_final, counts, 0.90)
        mx = float(de_vec_final.max()) if de_vec_final.size else 0.0
        sh = int(unique_rgb.shape[0] - np.unique(chosen_after_decongest).size)
        debug_log("pixel ensemble result:")
        debug_log(f"  mean dE={md:.3f}  p90={p90:.2f}  max={mx:.2f}  shares={sh}")

        tgt_lab = pal_lab[chosen_after_decongest]
        tgt_lch = lab_to_lch(tgt_lab.astype(np.float32))
        worst_first = np.argsort(
            -np.array(
                [
                    delta_e2000_vec(src_lab[i], tgt_lab[i : i + 1])[0]
                    for i in range(src_lab.shape[0])
                ]
            )
        )[:12]
        if worst_first.size > 0:
            debug_log("important mappings (worst-first):")
            for i in worst_first:
                src_rgb = unique_rgb[i]
                tgt_rgb = pal_rgb[chosen_after_decongest[i]]
                sL, sC, sH = (
                    float(src_lch[i, 0]),
                    float(src_lch[i, 1]),
                    float(src_lch[i, 2]),
                )
                tL, tC, tH = (
                    float(tgt_lch[i, 0]),
                    float(tgt_lch[i, 1]),
                    float(tgt_lch[i, 2]),
                )
                dH = tH - sH
                while dH > 180.0:
                    dH -= 360.0
                while dH < -180.0:
                    dH += 360.0
                debug_log(
                    f"  src #{src_rgb[0]:02x}{src_rgb[1]:02x}{src_rgb[2]:02x}  count={int(counts[i]):5d} "
                    f"-> #{tgt_rgb[0]:02x}{tgt_rgb[1]:02x}{tgt_rgb[2]:02x}  "
                    f"[dE={delta_e2000_vec(src_lab[i], tgt_lab[i:i+1])[0]:5.2f}, "
                    f"dL={tL - sL:+6.3f}, dC={tC - sC:+6.3f}, dH={dH:4.1f} deg]"
                )

        if grey_indices.size:
            neutral_set = set(
                int(j) for j in neutral_indices(pal_lch.astype(np.float32))
            )
            assigned_neutral = sum(
                1 for _si, pj in grey_assign_map.items() if int(pj) in neutral_set
            )
            debug_log(
                f"grey unique assign: {assigned_neutral}/{grey_indices.size} "
                f"greys mapped to distinct neutrals"
            )

    out = img_rgb.copy()
    visible_mask = alpha > 0
    if np.any(visible_mask):
        pal_map_rgb = pal_rgb[chosen_after_decongest]
        out[visible_mask] = pal_map_rgb[inverse_idx].astype(np.uint8, copy=False)
    return out
