# palette_map/photo/dither.py
from __future__ import annotations

"""
Textured photo dither.

Dither photos and gradients into the fixed palette using error diffusion in Lab-L
with palette-aware costs. Optionally does 2-colour micro-mix using a tiled Bayer 8x8
threshold to approximate the average colour more closely.

This version includes lightweight profiling. Set profile=True in kwargs to print a
timing and counters summary at the end.
"""

from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import math
import time
import numpy as np

from palette_map.colour_convert import delta_e2000_vec, rgb_to_lab_threaded
from palette_map.core_types import U8Image, U8Mask, Lab, Lch, PaletteItem
from palette_map.utils import (
    box_blur_2d,
    fmt_eta,
    prefilter_topk_lab as prefilter_topk,
    progress_line,
)

# Tunables

# Size of the box blur in pixels for local lightness. Higher = smoother, slower.
BLUR_RADIUS = 2
# How quickly local gradients reduce texture strength. Higher = more smoothing in busy areas.
GRAD_K = 5.0
# How many palette candidates to test per pixel after prefiltering.
PHOTO_TOPK = 12
# Limits how much lightness error gets diffused. Stops streaks and ringing.
EL_CLAMP = 4.0

# Treat a pixel as near neutral if its chroma is below this.
NEUTRAL_EST_C = 6.0
# Only estimate neutral a/b bias from pixels with lightness inside this range.
NEUTRAL_EST_LMIN = 25.0
NEUTRAL_EST_LMAX = 85.0
# Strength of the neutral a/b bias correction. 0 disables it.
AB_BIAS_GAIN = 0.5

# Base weight for keeping local contrast similar to the source.
W_LOCAL_BASE = 1.0
# Allow a little contrast loss before adding a penalty.
CONTRAST_MARGIN = 2.0
# How much hue mismatch matters. Scales up with source chroma.
W_HUE = 0.4
# How much chroma mismatch matters.
W_CHROMA = 0.02
# Below this lightness, treat pixels as shadow and relax matching slightly.
SHADOW_L = 40.0
# Extra hue penalty in shadows.
W_SHADOW_HUE = 0.10
# Extra penalty for picking a target that is even more chromatic in shadows.
W_SHADOW_CHROMA_UP = 0.05
# If the source chroma is below this, prefer neutral targets a bit more.
CHROMA_SRC_MIN = 8.0
# Targets with chroma at or below this are considered neutral.
NEUTRAL_C_MAX = 5.0

# If chroma is below this, treat as very close to neutral.
NEAR_NEUTRAL_C = 2.5
# Lightness cut for near black.
NEAR_BLACK_L = 10.0
# Lightness cut for near white.
NEAR_WHITE_L = 92.0
# Small nudges that protect metals and near-neutrals from bad picks.
W_NEUTRAL_GUARD = 0.05
W_LIGHTNESS_LOCK = 0.03

# Quantisation steps for the tiny decision cache. Bigger steps = more hits, less precision.
Q_L = 1.0
Q_AB = 2.0
Q_LLOC = 1.0
Q_WLOC = 0.1

# Two-colour micro mix for fine texture and closer averages.
# Turn off if you want pure single-colour dithering.
MIX_ENABLE = True
# Choose the partner from this many top candidates.
MIX_TOPK = 6
# Only mix if the two-colour average reduces dE by at least this much.
MIX_MIN_DE_GAIN = 0.15
# If the source is near-neutral, only mix neutral colours.
MIX_LOCK_NEUTRALS = True

# 8x8 Bayer thresholds in [0..64); normalised to [0,1)
_BAYER8 = (
    np.array(
        [
            [0, 48, 12, 60, 3, 51, 15, 63],
            [32, 16, 44, 28, 35, 19, 47, 31],
            [8, 56, 4, 52, 11, 59, 7, 55],
            [40, 24, 36, 20, 43, 27, 39, 23],
            [2, 50, 14, 62, 1, 49, 13, 61],
            [34, 18, 46, 30, 33, 17, 45, 29],
            [10, 58, 6, 54, 9, 57, 5, 53],
            [42, 26, 38, 22, 41, 25, 37, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
)

# Threaded helpers for local ops


def _split_rows_with_halo(
    h: int, parts: int, halo: int
) -> List[Tuple[int, int, int, int]]:
    parts = max(1, int(parts))
    step = (h + parts - 1) // parts
    out: List[Tuple[int, int, int, int]] = []
    s = 0
    while s < h:
        e = min(s + step, h)
        s_pad = max(0, s - halo)
        e_pad = min(h, e + halo)
        out.append((s, e, s_pad, e_pad))
        s = e
    return out


def box_blur_2d_threaded(arr: np.ndarray, radius: int, workers: int) -> np.ndarray:
    # Hot when radius > 0 on large images. Threads keep exact output by using a halo.
    if radius <= 0 or workers <= 1 or arr.shape[0] < 256:
        return box_blur_2d(arr, radius)
    H, W = arr.shape
    chunks = _split_rows_with_halo(H, workers, radius)
    out = np.empty((H, W), dtype=np.float32)

    def run_one(ch):
        s, e, s_pad, e_pad = ch
        sub = box_blur_2d(arr[s_pad:e_pad], radius)
        return (s, e, sub[(s - s_pad) : (s - s_pad) + (e - s)])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s, e, sub_core in ex.map(run_one, chunks):
            out[s:e] = sub_core
    return out


def grad_mag_threaded(L: np.ndarray, workers: int) -> np.ndarray:
    # Small blur of |grad|. Threads help on tall images.
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    g = gx + gy
    return box_blur_2d_threaded(g, 1, workers)


# Neutral bias estimate


def estimate_ab_bias(lab_img: Lab, alpha: U8Mask) -> Tuple[float, float]:
    """
    Estimate a and b neutral bias from visible near-neutral pixels to reduce
    colour cast in highlights and midtones.
    """
    L = lab_img[..., 0]
    a = lab_img[..., 1]
    b = lab_img[..., 2]
    C = np.hypot(a, b)
    mask = (
        (alpha != 0)
        & (C <= NEUTRAL_EST_C)
        & (L >= NEUTRAL_EST_LMIN)
        & (L <= NEUTRAL_EST_LMAX)
    )
    if not mask.any():
        return (0.0, 0.0)
    return (AB_BIAS_GAIN * float(a[mask].mean()), AB_BIAS_GAIN * float(b[mask].mean()))


# Cost model


def photo_cost_components(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    t_lch: Lch,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute palette-aware cost terms for a single source colour against target LCh rows.
    Returns (cost_vector, target_chroma_vector).
    """
    sL, sC, sh = float(s_lch[0]), float(s_lch[1]), float(s_lch[2])
    tL, tC, th = t_lch[:, 0], t_lch[:, 1], t_lch[:, 2]
    cost = np.zeros_like(tL, dtype=np.float32)

    # Local contrast preservation
    cost += W_LOCAL_BASE * w_local_eff * np.abs((tL - L_local) - (sL - L_local))

    d_s = abs(sL - L_local)
    d_t = np.abs(tL - L_local)
    cost += np.where(
        d_t + CONTRAST_MARGIN < d_s, (d_s - d_t - CONTRAST_MARGIN) * 1.0, 0.0
    )

    hue_weight = min(1.0, sC / 40.0)
    hue_dist = np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0
    cost += W_HUE * hue_weight * hue_dist
    cost += W_CHROMA * np.abs(tC - sC)

    if sL < SHADOW_L:
        shadow = (SHADOW_L - sL) / SHADOW_L
        chroma_up = np.maximum(0.0, tC - sC)
        cost += shadow * (W_SHADOW_HUE * hue_dist + W_SHADOW_CHROMA_UP * chroma_up)

    if sC >= CHROMA_SRC_MIN:
        cost += np.where(
            tC <= NEUTRAL_C_MAX, 0.10 + 0.02 * np.maximum(0.0, sC - tC), 0.0
        )

    return cost, tC


# Candidate pick


def _nearest_idx_photo(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    near_extreme_neutral: bool,
    k: int,
    *,
    pal_is_neutral: np.ndarray,
    prof: Dict[str, float] | None = None,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick nearest palette index for one source Lab using a combined cost:
    dE2000 + palette-aware components, with a fast top-k prefilter.

    Hot path: delta_e2000_vec over the top-k candidates.
    """
    t0 = time.perf_counter()
    idxs = prefilter_topk(s_lab, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]
    dE = delta_e2000_vec(s_lab, cand_lab)
    comp, _ = photo_cost_components(s_lab, s_lch, L_local, w_local_eff, cand_lch)
    cost = dE + comp

    if s_lch[1] <= NEUTRAL_EST_C:
        cost += W_NEUTRAL_GUARD * np.where(cand_lch[:, 1] <= NEUTRAL_C_MAX, 1.0, 0.0)
        cost += W_LIGHTNESS_LOCK * np.abs(
            (cand_lch[:, 0] - L_local) - (s_lch[0] - L_local)
        )
    if near_extreme_neutral:
        cost = np.where(cand_lch[:, 1] > NEUTRAL_C_MAX, cost + 1e6, cost)

    j_rel = int(np.argmin(cost))

    if prof is not None:
        prof["t_nearest"] += time.perf_counter() - t0
        prof["calls_nearest"] += 1
        prof["evals_nearest"] += int(cand_lab.shape[0])

    return int(idxs[j_rel]), idxs, cost, dE


# Public entry


def dither_photo(
    img_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    workers: int = 1,
    **kwargs,
) -> U8Image:
    """
    Serpentine error diffusion in Lab-L with palette-aware scoring.
    Adds optional 2-colour micro-mix (Bayer 8x8) for closer average colour.

    Set profile=True in kwargs to see a timing summary and counters.

    Hot spots to watch:
      - RGB to Lab conversion on large images
      - Local blur and gradient precompute
      - Inner pixel loop: candidate pick and optional micro-mix
    """
    progress_enabled = bool(kwargs.get("progress", True))
    profile = bool(kwargs.get("profile", False))

    # Simple profiler store
    prof: Dict[str, float] = {
        "t_total": 0.0,
        "t_rgb2lab": 0.0,
        "t_locals": 0.0,
        "t_bias": 0.0,
        "t_loop": 0.0,
        "t_nearest": 0.0,
        "calls_nearest": 0.0,
        "evals_nearest": 0.0,
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "mix_evals": 0.0,
        "mix_accepted": 0.0,
    }

    def _p(pct: float, eta_s: float, final: bool = False) -> None:
        if not progress_enabled:
            return
        progress_line(f"[photo] {int(pct):3d}% (ETA {fmt_eta(eta_s)})", final=final)

    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    # Adaptive top-k by image size
    pixels = H * W
    if pixels <= 800_000:
        topk_global = int(kwargs.get("topk", PHOTO_TOPK))
    elif pixels <= 2_500_000:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 10)
    else:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 8)

    t_total0 = time.perf_counter()

    # RGB to Lab (threaded, deterministic)
    t0 = time.perf_counter()
    lab_img = rgb_to_lab_threaded(img_rgb, workers).reshape(H, W, 3)
    prof["t_rgb2lab"] = time.perf_counter() - t0

    L_src = lab_img[..., 0].astype(np.float32, copy=False)

    # Precompute locals (threaded blur and grad)
    t0 = time.perf_counter()
    if BLUR_RADIUS > 0:
        L_loc = box_blur_2d_threaded(L_src, BLUR_RADIUS, workers)
    else:
        L_loc = L_src.astype(np.float32, copy=False)
    G = grad_mag_threaded(L_src, workers)
    prof["t_locals"] = time.perf_counter() - t0

    # Neutral bias estimate
    t0 = time.perf_counter()
    a_bias, b_bias = estimate_ab_bias(lab_img, alpha)
    prof["t_bias"] = time.perf_counter() - t0

    errL = np.zeros((H, W), dtype=np.float32)
    pal_is_neutral = pal_lch_mat[:, 1] <= NEUTRAL_C_MAX

    # Decision cache
    pick_cache: Dict[Tuple[int, int, int, int, int, int], int] = {}

    # ETA tracking
    start = time.perf_counter()
    last_row_t = start
    rows_done = 0
    ewma_row_ms = None
    EWMA_ALPHA = 0.15
    pct_step_rows = max(1, H // 100)
    min_interval = 0.25
    last_progress = start

    # Main diffusion (hot)
    t_loop0 = time.perf_counter()
    for y in range(H):
        serp_left = (y % 2) == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)
        neigh = (
            ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
            if serp_left
            else ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        )

        for x in xs:
            if alpha[y, x] == 0:
                continue

            # Local weight from gradient (fast)
            wloc = max(0.2, 1.0 - (G[y, x] / GRAD_K))

            # Source with error and neutral bias applied
            sL = float(lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(lab_img[y, x, 1]) - a_bias
            sb = float(lab_img[y, x, 2]) - b_bias
            C = float(math.hypot(sa, sb))
            h = (math.degrees(math.atan2(sb, sa)) + 360.0) % 360.0

            near_extreme = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            # Open search slightly for highly saturated pixels
            topk_eff = topk_global + (4 if C >= 24.0 else 0)

            # Tiny decision cache key
            key = (
                int(round(sL / Q_L)),
                int(round(sa / Q_AB)),
                int(round(sb / Q_AB)),
                int(round(L_loc[y, x] / Q_LLOC)),
                int(round(wloc / Q_WLOC)),
                1 if near_extreme else 0,
            )
            j_pick = pick_cache.get(key, -1)

            # Hot path: candidate pick and optional micro-mix
            if j_pick >= 0 and not MIX_ENABLE:
                prof["cache_hits"] += 1
            else:
                prof["cache_misses"] += 1
                s_lab = np.array([sL, sa, sb], dtype=np.float32)
                s_lch = np.array([sL, C, h], dtype=np.float32)

                j_best, idxs, cost, dE = _nearest_idx_photo(
                    s_lab,
                    s_lch,
                    float(L_loc[y, x]),
                    W_LOCAL_BASE * wloc,
                    pal_lab_mat,
                    pal_lch_mat,
                    near_extreme_neutral=near_extreme,
                    k=topk_eff,
                    pal_is_neutral=pal_is_neutral,
                    prof=prof if profile else None,
                )
                j_pick = j_best

                # Optional micro-mix (costly when enabled)
                if MIX_ENABLE:
                    prof["mix_evals"] += 1
                    rel_sorted = np.argsort(cost)[: min(MIX_TOPK, cost.size)]
                    rel_sorted = [r for r in rel_sorted if idxs[r] != j_best]
                    if rel_sorted:
                        j_alt_rel = rel_sorted[0]
                        j_alt = int(idxs[j_alt_rel])
                        if (not MIX_LOCK_NEUTRALS) or (
                            pal_lch_mat[j_best, 1] <= NEUTRAL_C_MAX
                            and pal_lch_mat[j_alt, 1] <= NEUTRAL_C_MAX
                        ):
                            p0 = pal_lab_mat[j_best]
                            p1 = pal_lab_mat[j_alt]
                            d = p0 - p1
                            denom = float(np.dot(d, d)) + 1e-12
                            w = float(np.dot(s_lab - p1, d) / denom)
                            w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w
                            mix_lab = (w * p0 + (1.0 - w) * p1).astype(np.float32)
                            de_mix = float(delta_e2000_vec(s_lab, mix_lab[None, :])[0])
                            de_best = float(delta_e2000_vec(s_lab, p0[None, :])[0])
                            if de_mix + 1e-6 < de_best - MIX_MIN_DE_GAIN:
                                t = float(_BAYER8[y & 7, x & 7])
                                j_pick = j_best if t < w else j_alt
                                prof["mix_accepted"] += 1

                pick_cache[key] = j_pick

            # Write RGB
            r, g, b = palette[j_pick].rgb
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b

            # Diffuse L error (fast)
            tgt_L = float(pal_lch_mat[j_pick, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            if eL != 0.0:
                if serp_left:
                    if y + 0 < H and x + 1 < W and alpha[y + 0, x + 1] != 0:
                        errL[y + 0, x + 1] += eL * (7 / 16)
                    if y + 1 < H and x - 1 >= 0 and alpha[y + 1, x - 1] != 0:
                        errL[y + 1, x - 1] += eL * (3 / 16)
                    if y + 1 < H and x + 0 < W and alpha[y + 1, x + 0] != 0:
                        errL[y + 1, x + 0] += eL * (5 / 16)
                    if y + 1 < H and x + 1 < W and alpha[y + 1, x + 1] != 0:
                        errL[y + 1, x + 1] += eL * (1 / 16)
                else:
                    if y + 0 < H and x - 1 >= 0 and alpha[y + 0, x - 1] != 0:
                        errL[y + 0, x - 1] += eL * (7 / 16)
                    if y + 1 < H and x + 1 < W and alpha[y + 1, x + 1] != 0:
                        errL[y + 1, x + 1] += eL * (3 / 16)
                    if y + 1 < H and x + 0 < W and alpha[y + 1, x + 0] != 0:
                        errL[y + 1, x + 0] += eL * (5 / 16)
                    if y + 1 < H and x - 1 >= 0 and alpha[y + 1, x - 1] != 0:
                        errL[y + 1, x - 1] += eL * (1 / 16)

        # Row timing and ETA
        rows_done += 1
        now = time.perf_counter()
        row_dt = now - last_row_t
        last_row_t = now
        if ewma_row_ms is None:
            ewma_row_ms = row_dt
        else:
            ewma_row_ms = (1.0 - EWMA_ALPHA) * ewma_row_ms + EWMA_ALPHA * row_dt
        if (rows_done % pct_step_rows == 0) or (now - last_progress >= min_interval):
            rows_left = H - rows_done
            eta = max(0.0, rows_left * float(ewma_row_ms))
            _p(100.0 * rows_done / H, eta, final=False)
            last_progress = now

    prof["t_loop"] = time.perf_counter() - t_loop0
    prof["t_total"] = time.perf_counter() - t_total0

    _p(100.0, 0.0, final=True)

    # Optional profiling summary
    if profile:
        total = max(1e-9, prof["t_total"])

        def pct(t: float) -> str:
            return f"{100.0 * t / total:5.1f}%"

        print("[profile] photo mode timing:", flush=True)
        print(
            f"  total         {prof['t_total']:.3f}s  {pct(prof['t_total'])}",
            flush=True,
        )
        print(
            f"  rgb->lab      {prof['t_rgb2lab']:.3f}s  {pct(prof['t_rgb2lab'])}",
            flush=True,
        )
        print(
            f"  locals        {prof['t_locals']:.3f}s  {pct(prof['t_locals'])}",
            flush=True,
        )
        print(
            f"  neutral bias  {prof['t_bias']:.3f}s  {pct(prof['t_bias'])}", flush=True
        )
        print(
            f"  loop          {prof['t_loop']:.3f}s  {pct(prof['t_loop'])}", flush=True
        )
        print(
            f"    nearest     {prof['t_nearest']:.3f}s  {pct(prof['t_nearest'])}",
            flush=True,
        )
        print(f"    nearest calls     {int(prof['calls_nearest'])}", flush=True)
        print(
            f"    nearest evals     {int(prof['evals_nearest'])} (sum of top-k)",
            flush=True,
        )
        print(f"    cache hits        {int(prof['cache_hits'])}", flush=True)
        print(f"    cache misses      {int(prof['cache_misses'])}", flush=True)
        if MIX_ENABLE:
            print(f"    mix evals         {int(prof['mix_evals'])}", flush=True)
            print(f"    mix accepted      {int(prof['mix_accepted'])}", flush=True)

    return out
