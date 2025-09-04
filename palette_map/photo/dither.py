# palette_map/photo/dither.py
# --- Textured photo dither (drop-in) -----------------------------------------
# Compatible with your wrappers. Accepts a palette list whose items have `.rgb`,
# and Lab/LCh matrices aligned to that palette order. Threaded precompute.
from __future__ import annotations

from typing import List, Tuple
import math
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from palette_map.color_convert import rgb_to_lab
from palette_map.core_types import (
    U8Image,
    U8Mask,
    Lab,
    Lch,
    PaletteItem,
)

# ---- Tunables (safe defaults; adjust later if you want) ----
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


# ---------- small threaded helpers (precompute only; diffusion stays serial) --


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


def compute_L_image(rgb: U8Image) -> np.ndarray:
    # Lab L channel from sRGB
    return rgb_to_lab(rgb)[..., 0].astype(np.float32, copy=False)


def box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    H, W = arr.shape
    arr = arr.astype(np.float32, copy=False)
    ii = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    ii_pad = np.zeros((H + 1, W + 1), dtype=np.float32)
    ii_pad[1:, 1:] = ii
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
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


def ciede2000_vec(s: np.ndarray, cand: Lab) -> np.ndarray:
    L1, a1, b1 = s[0], s[1], s[2]
    L2, a2, b2 = cand[:, 0], cand[:, 1], cand[:, 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt((Cm**7) / (Cm**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = (np.degrees(np.atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.atan2(b2, a2p)) + 360.0) % 360.0
    dLp = L2 - L1
    dhp = h2p - h1p
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lpm = 0.5 * (L1 + L2)
    Cpm = 0.5 * (C1p + C2p)
    hpm = np.where(
        (C1p * C2p) == 0,
        h1p + h2p,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            0.5 * (h1p + h2p),
            np.where(
                (h1p + h2p) < 360.0,
                0.5 * (h1p + h2p + 360.0),
                0.5 * (h1p + h2p - 360.0),
            ),
        ),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hpm - 30))
        + 0.24 * np.cos(np.radians(2 * hpm))
        + 0.32 * np.cos(np.radians(3 * hpm + 6))
        - 0.20 * np.cos(np.radians(4 * hpm - 63))
    )
    d_ro = 30.0 * np.exp(-(((hpm - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1.0 + (0.015 * ((Lpm - 50.0) ** 2)) / np.sqrt(20.0 + (Lpm - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2.0 * d_ro)) * Rc
    dCp = C2p - C1p
    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    ).astype(np.float32, copy=False)


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

    # preserve relative local contrast to blurred L
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


def nearest_palette_idx_photo_lab_prefilter_vec(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    near_extreme_neutral: bool,
    k: int,
) -> int:
    idxs = prefilter_topk(s_lab, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]
    dE = ciede2000_vec(s_lab, cand_lab)
    comp_base, tC = photo_cost_components(s_lab, s_lch, L_local, w_local_eff, cand_lch)
    cost = dE + comp_base
    if near_extreme_neutral:
        cost = np.where(tC > NEUTRAL_C_MAX, cost + 1e6, cost)
    return int(idxs[int(np.argmin(cost))])


# ----------------------- helpers for progress / ETA ---------------------------


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
    # \r = return to line start, \033[K = clear to end of line
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()
    if final:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ----------------------- public entry (threaded precompute) -------------------


def dither_photo(
    img_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],  # items with .rgb
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    *,
    workers: int = 1,
    **kwargs,
) -> U8Image:
    """
    Serpentine error-diffusion in Lab L with palette-aware scoring.
    Uses more colours where it lowers perceptual error.

    Threading: precomputations (RGB→Lab) are parallelized across row-chunks.
    The diffusion loop itself remains serial (Floyd–Steinberg dependency).

    Progress: prints single-line % + ETA (always on; disable with progress=False).
    """
    # ---- progress helper (single-line overwrite + ETA) -----------------------
    import sys, time

    progress_enabled = bool(kwargs.get("progress", True))

    def _progress(pct: float, eta_s: float, final: bool = False) -> None:
        if not progress_enabled:
            return
        mm = int(eta_s // 60)
        ss = int(eta_s % 60)
        msg = f"[photo] {int(pct):3d}% (ETA {mm:02d}:{ss:02d})"
        sys.stdout.write("\r\033[K" + msg)
        sys.stdout.flush()
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # ---- image setup ---------------------------------------------------------
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    # Auto-tune prefilter size (middle ground speed/quality)
    # ~<0.8MP → 12, 0.8–2.5MP → 10, >2.5MP → 8
    pixels = H * W
    if pixels <= 800_000:
        topk_eff = int(kwargs.get("topk", PHOTO_TOPK))  # default 12
    elif pixels <= 2_500_000:
        topk_eff = min(int(kwargs.get("topk", PHOTO_TOPK)), 10)
    else:
        topk_eff = min(int(kwargs.get("topk", PHOTO_TOPK)), 8)

    # Threaded RGB→Lab (row chunks)
    t0 = time.perf_counter()
    lab_img = _rgb_to_lab_threaded(img_rgb, workers).reshape(H, W, 3)
    L_src = lab_img[..., 0].astype(np.float32, copy=False)

    # Vector precompute
    L_loc = box_blur_2d(L_src, BLUR_RADIUS)
    G = grad_mag(L_src)
    a_bias, b_bias = estimate_ab_bias(lab_img, alpha)

    errL = np.zeros((H, W), dtype=np.float32)

    # Progress timing
    start = time.perf_counter()
    last_tick = start
    rows_done = 0

    # Update frequency: every ~1% or 0.25s, whichever is later
    pct_step_rows = max(1, H // 100)
    min_interval = 0.25

    # ---- main diffusion ------------------------------------------------------
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

            # Local-contrast weight from gradient
            wloc = max(0.2, 1.0 - (G[y, x] / GRAD_K))

            # Source Lab with error + neutral bias compensation
            sL = float(lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(lab_img[y, x, 1]) - a_bias
            sb = float(lab_img[y, x, 2]) - b_bias
            s_lab_adj = np.array([sL, sa, sb], dtype=np.float32)

            # To LCh (fast, local)
            C = float(math.hypot(sa, sb))
            h = (math.degrees(math.atan2(sb, sa)) + 360.0) % 360.0
            s_lch_adj = np.array([sL, C, h], dtype=np.float32)

            # Extreme near-neutral pixels near black/white: restrict to neutrals
            near_extreme = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            # Prefilter by Lab Euclidean (top-k), then score with dE2000 + photo costs
            idxs = prefilter_topk(s_lab_adj, pal_lab_mat, topk_eff)
            cand_lab = pal_lab_mat[idxs]
            cand_lch = pal_lch_mat[idxs]
            dE = ciede2000_vec(s_lab_adj, cand_lab)
            comp, tC = photo_cost_components(
                s_lab_adj, s_lch_adj, float(L_loc[y, x]), W_LOCAL_BASE * wloc, cand_lch
            )
            cost = dE + comp
            if near_extreme:
                cost = np.where(tC > NEUTRAL_C_MAX, cost + 1e6, cost)
            j = int(idxs[int(np.argmin(cost))])

            # Write RGB
            r, g, b = palette[j].rgb
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b

            # Diffuse L error (clamped)
            tgt_L = float(pal_lch_mat[j, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            for dx, dy, w in neigh:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

        # progress update (row-level)
        rows_done += 1
        if (rows_done % pct_step_rows == 0) or (time.perf_counter() - last_tick >= min_interval):
            now = time.perf_counter()
            elapsed = now - start
            rows_left = max(0, H - rows_done)
            rps = rows_done / elapsed if elapsed > 0 else 0.0
            eta = rows_left / rps if rps > 0 else 0.0
            _progress(100.0 * rows_done / H, eta, final=False)
            last_tick = now

    # final progress line
    total_eta = 0.0
    _progress(100.0, total_eta, final=True)
    return out

