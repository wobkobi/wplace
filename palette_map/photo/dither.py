# palette_map/photo/dither.py
# --- Textured photo dither (drop-in) -----------------------------------------
from __future__ import annotations

from typing import List, Tuple, Dict
import math
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from palette_map.colour_convert import delta_e2000_vec, rgb_to_lab
from palette_map.core_types import (
    U8Image,
    U8Mask,
    Lab,
    Lch,
    PaletteItem,
)

# ---- Tunables ---------------------------------------------------------------
BLUR_RADIUS = 2
GRAD_K = 5.0
PHOTO_TOPK = 12
EL_CLAMP = 4.0

NEUTRAL_EST_C = 6.0
NEUTRAL_EST_LMIN = 25.0
NEUTRAL_EST_LMAX = 85.0
AB_BIAS_GAIN = 0.5

W_LOCAL_BASE = 1.0
CONTRAST_MARGIN = 2.0
W_HUE = 0.4
W_CHROMA = 0.02
SHADOW_L = 40.0
W_SHADOW_HUE = 0.10
W_SHADOW_CHROMA_UP = 0.05
CHROMA_SRC_MIN = 8.0
NEUTRAL_C_MAX = 5.0

NEAR_NEUTRAL_C = 2.5
NEAR_BLACK_L = 10.0
NEAR_WHITE_L = 92.0

# gentle guards for metals/near-neutrals
W_NEUTRAL_GUARD = 0.05
W_LIGHTNESS_LOCK = 0.03

# tiny decision cache quantization
Q_L = 1.0
Q_AB = 2.0
Q_LLOC = 1.0
Q_WLOC = 0.1

# --- micro-mix (2-colour texturing) ------------------------------------------
MIX_ENABLE = True  # set False to disable mixing entirely
MIX_TOPK = 6  # choose pair from the best K candidates
MIX_MIN_DE_GAIN = 0.15  # require at least this ΔE2000 gain vs single best
MIX_LOCK_NEUTRALS = True  # if near_extreme_neutral, force pair to neutrals too

# 8×8 Bayer thresholds in [0..64); we normalize to [0,1)
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


# ---------- threaded precompute ----------------------------------------------


def _split_rows(h: int, parts: int) -> List[Tuple[int, int]]:
    parts = max(1, int(parts))
    step = (h + parts - 1) // parts
    return [(i, min(i + step, h)) for i in range(0, h, step)]


def _rgb_to_lab_threaded(rgb: U8Image, workers: int) -> Lab:
    H = int(rgb.shape[0])
    if workers <= 1 or H < 256:
        return rgb_to_lab(rgb).astype(np.float32, copy=False)
    chunks = _split_rows(H, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(rgb_to_lab, rgb[s:e]) for s, e in chunks]
        parts = [f.result().astype(np.float32, copy=False) for f in futs]
    return np.vstack(parts).astype(np.float32, copy=False)


# ----------------------- image space utilities --------------------------------


def box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    H, W = arr.shape
    arr = arr.astype(np.float32, copy=False)
    ii = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    ii_pad = np.zeros((H + 1, W + 1), dtype=np.float32)
    ii_pad[1:, 1:] = ii
    ys = np.arange(H, dtype=np.int32)[:, None]
    xs = np.arange(W, dtype=np.int32)[None, :]
    y1 = np.clip(ys - radius, 0, H - 1)
    y2 = np.clip(ys + radius, 0, H - 1)
    x1 = np.clip(xs - radius, 0, W - 1)
    x2 = np.clip(xs + radius, 0, W - 1)
    s = (
        ii_pad[y2 + 1, x2 + 1]
        - ii_pad[y1, x2 + 1]
        - ii_pad[y2 + 1, x1]
        + ii_pad[y1, x1]
    )
    area = (y2 - y1 + 1) * (x2 - x1 + 1)
    return (s / area).astype(np.float32, copy=False)


def grad_mag(L: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    return box_blur_2d(gx + gy, 1)


def estimate_ab_bias(lab_img: Lab, alpha: U8Mask) -> Tuple[float, float]:
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


# ----------------------- palette distance utilities ---------------------------


def prefilter_topk(s_lab: np.ndarray, pal_lab_mat: Lab, k: int) -> np.ndarray:
    diff = pal_lab_mat - s_lab
    de2 = (diff * diff).sum(axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def photo_cost_components(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    t_lch: Lch,
) -> Tuple[np.ndarray, np.ndarray]:
    sL, sC, sh = float(s_lch[0]), float(s_lch[1]), float(s_lch[2])
    tL, tC, th = t_lch[:, 0], t_lch[:, 1], t_lch[:, 2]
    cost = np.zeros_like(tL, dtype=np.float32)

    # local contrast preservation
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
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    idxs = prefilter_topk(s_lab, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]
    dE = delta_e2000_vec(s_lab, cand_lab)
    comp, tC = photo_cost_components(s_lab, s_lch, L_local, w_local_eff, cand_lch)
    cost = dE + comp

    # gentle neutral preference near neutral sources
    if s_lch[1] <= NEUTRAL_EST_C:
        cost += W_NEUTRAL_GUARD * np.where(cand_lch[:, 1] <= NEUTRAL_C_MAX, 1.0, 0.0)
        cost += W_LIGHTNESS_LOCK * np.abs(
            (cand_lch[:, 0] - L_local) - (s_lch[0] - L_local)
        )

    if near_extreme_neutral:
        cost = np.where(cand_lch[:, 1] > NEUTRAL_C_MAX, cost + 1e6, cost)

    j_rel = int(np.argmin(cost))
    return int(idxs[j_rel]), idxs, cost, dE


# ----------------------- progress helpers -------------------------------------


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    s = int(round(seconds))
    if s >= 3600:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h}h {m}m"
    if s >= 60:
        m = s // 60
        r = s % 60
        return f"{m}m {r}s"
    return f"{s}s"


def _progress(msg: str, final: bool = False) -> None:
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()
    if final:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ----------------------- public entry -----------------------------------------


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
    Serpentine error-diffusion in Lab-L with palette-aware scoring.
    Adds optional 2-colour micro-mix (Bayer 8×8) for closer average colour.
    Progress: single-line % + ETA (disable with progress=False).
    """
    progress_enabled = bool(kwargs.get("progress", True))

    def _p(pct: float, eta_s: float, final: bool = False) -> None:
        if not progress_enabled:
            return
        msg = f"[photo] {int(pct):3d}% (ETA {_fmt_eta(eta_s)})"
        sys.stdout.write("\r\033[K" + msg)
        sys.stdout.flush()
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()

    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    # adaptive top-k by image size
    pixels = H * W
    if pixels <= 800_000:
        topk_global = int(kwargs.get("topk", PHOTO_TOPK))
    elif pixels <= 2_500_000:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 10)
    else:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 8)

    # RGB→Lab (threaded)
    lab_img = _rgb_to_lab_threaded(img_rgb, workers).reshape(H, W, 3)
    L_src = lab_img[..., 0].astype(np.float32, copy=False)

    # precompute locals
    L_loc = box_blur_2d(L_src, BLUR_RADIUS)
    G = grad_mag(L_src)
    a_bias, b_bias = estimate_ab_bias(lab_img, alpha)
    errL = np.zeros((H, W), dtype=np.float32)

    pal_is_neutral = pal_lch_mat[:, 1] <= NEUTRAL_C_MAX

    # tiny decision cache
    pick_cache: Dict[Tuple[int, int, int, int, int, int], int] = {}

    # --- ETA: EWMA of row time (stable) --------------------------------------
    start = time.perf_counter()
    last_row_t = start
    rows_done = 0
    ewma_row_ms = None
    EWMA_ALPHA = 0.15  # inertia (lower = smoother)

    pct_step_rows = max(1, H // 100)
    min_interval = 0.25
    last_progress = start

    # main diffusion
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

            # gradient -> local weight
            wloc = max(0.2, 1.0 - (G[y, x] / GRAD_K))

            # source Lab with error & neutral bias
            sL = float(lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(lab_img[y, x, 1]) - a_bias
            sb = float(lab_img[y, x, 2]) - b_bias
            C = float(math.hypot(sa, sb))
            h = (math.degrees(math.atan2(sb, sa)) + 360.0) % 360.0

            near_extreme = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            # adaptive top-k: open up a bit for high-chroma
            topk_eff = topk_global + (4 if C >= 24.0 else 0)

            # decision cache
            key = (
                int(round(sL / Q_L)),
                int(round(sa / Q_AB)),
                int(round(sb / Q_AB)),
                int(round(L_loc[y, x] / Q_LLOC)),
                int(round(wloc / Q_WLOC)),
                1 if near_extreme else 0,
            )
            j_pick = pick_cache.get(key, -1)

            if j_pick < 0 or MIX_ENABLE:
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
                )
                j_pick = j_best

                # --- 2-colour micro-mix (optional) --------------------------
                # Find an alternate partner to reduce ΔE2000 on average.
                if MIX_ENABLE:
                    # choose partner among top MIX_TOPK distinct candidates
                    rel_sorted = np.argsort(cost)[: min(MIX_TOPK, cost.size)]
                    rel_sorted = [r for r in rel_sorted if idxs[r] != j_best]
                    if rel_sorted:
                        j_alt_rel = rel_sorted[0]
                        j_alt = int(idxs[j_alt_rel])

                        # if near-extreme, optionally force both to neutral
                        if (not MIX_LOCK_NEUTRALS) or (
                            pal_lch_mat[j_best, 1] <= NEUTRAL_C_MAX
                            and pal_lch_mat[j_alt, 1] <= NEUTRAL_C_MAX
                        ):
                            p0 = pal_lab_mat[j_best]
                            p1 = pal_lab_mat[j_alt]
                            d = p0 - p1
                            denom = float(np.dot(d, d)) + 1e-12
                            # fraction of p0 to best match s_lab in least squares
                            w = float(np.dot(s_lab - p1, d) / denom)
                            w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w

                            mix_lab = (w * p0 + (1.0 - w) * p1).astype(np.float32)
                            # evaluate ΔE of the *average*
                            de_mix = float(delta_e2000_vec(s_lab, mix_lab[None, :])[0])
                            de_best = float(delta_e2000_vec(s_lab, p0[None, :])[0])

                            if de_mix + 1e-6 < de_best - MIX_MIN_DE_GAIN:
                                # stochastic choice via Bayer threshold (tile)
                                t = float(_BAYER8[y & 7, x & 7])
                                # coverage: use p0 when t < w
                                j_pick = j_best if t < w else j_alt
                            # else: keep single best

                pick_cache[key] = j_pick

            # write RGB
            r, g, b = palette[j_pick].rgb
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b

            # diffuse L error
            tgt_L = float(pal_lch_mat[j_pick, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            for dx, dy, w in neigh:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

        # --- ETA update (EWMA) ----------------------------------------------
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

    _p(100.0, 0.0, final=True)
    return out
