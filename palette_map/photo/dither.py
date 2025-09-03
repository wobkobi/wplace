# palette_map/photo/dither.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..analysis import srgb_to_lab  # wrappers live in analysis
from ..core_types import PaletteItem

# =========================
# Photo-mode tunables
# =========================
W_LOCAL_BASE = 0.18
GRAD_K = 12.0
W_HUE = 0.06
W_CHROMA = 0.05
BLUR_RADIUS = 2
W_CONTRAST = 0.15
CONTRAST_MARGIN = 0.8
PHOTO_NEUTRAL_C_MAX = 5.0
NEUTRAL_EST_C = 8.0
NEUTRAL_EST_LMIN = 20.0
NEUTRAL_EST_LMAX = 80.0
AB_BIAS_GAIN = 0.8
NEAR_NEUTRAL_C = 4.0
NEAR_BLACK_L = 32.0
NEAR_WHITE_L = 86.0
SHADOW_L = 38.0
W_SHADOW_HUE = 0.12
W_SHADOW_CHROMA_UP = 0.10
EL_CLAMP = 6.0
PHOTO_TOPK = 6


def _box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    H, W = arr.shape
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
    return (s / area).astype(np.float32)


def _compute_L_image(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0

    def lin(u: np.ndarray) -> np.ndarray:
        return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

    arr_lin = lin(arr)
    Y = (
        arr_lin[..., 0] * 0.2126729
        + arr_lin[..., 1] * 0.7151522
        + arr_lin[..., 2] * 0.0721750
    )
    fy = np.where(Y > 0.008856, np.cbrt(Y), 7.787 * Y + 16.0 / 116.0)
    L = 116.0 * fy - 16.0
    return L.astype(np.float32)


def _grad_mag(L: np.ndarray) -> np.ndarray:
    H, W = L.shape
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    g = gx + gy
    return _box_blur_2d(g, 1)


def _estimate_ab_bias(lab_img: np.ndarray, alpha: np.ndarray) -> Tuple[float, float]:
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
    a_mean = float(a[mask].mean())
    b_mean = float(b[mask].mean())
    return (AB_BIAS_GAIN * a_mean, AB_BIAS_GAIN * b_mean)


def _prefilter_topk(s_lab: np.ndarray, pal_lab_mat: np.ndarray, k: int) -> np.ndarray:
    diff = pal_lab_mat - s_lab
    de2 = (diff * diff).sum(axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def _ciede2000_vec(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    L1, a1, b1 = s[0], s[1], s[2]
    L2, a2, b2 = cand[:, 0], cand[:, 1], cand[:, 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt((Cm**7) / (Cm**7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = (np.degrees(np.atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.atan2(b2, a2p)) + 360.0) % 360.0
    dLp = L2 - L1
    dhp = h2p - h1p
    dhp = np.where((C1p * C2p) == 0.0, 0.0, dhp)
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lpm = 0.5 * (L1 + L2)
    Cpm = 0.5 * (C1p + C2p)
    hpm = np.where(
        (C1p * C2p) == 0.0,
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
        1.0
        - 0.17 * np.cos(np.radians(hpm - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hpm))
        + 0.32 * np.cos(np.radians(3.0 * hpm + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hpm - 63.0))
    )
    d_ro = 30.0 * np.exp(-(((hpm - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * np.sqrt((Cpm**7) / (Cpm**7 + 25.0**7))
    Sl = 1.0 + (0.015 * (Lpm - 50.0) ** 2.0) / np.sqrt(20.0 + (Lpm - 50.0) ** 2.0)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2.0 * d_ro)) * Rc
    dCp = Cpm - C1p
    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def _photo_cost_components(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    t_lch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sL, sC, sh = float(s_lch[0]), float(s_lch[1]), float(s_lch[2])
    tL = t_lch[:, 0]
    tC = t_lch[:, 1]
    th = t_lch[:, 2]

    cost = np.zeros_like(tL, dtype=np.float32)

    cost += W_LOCAL_BASE * w_local_eff * np.abs((tL - L_local) - (sL - L_local))

    d_s = abs(sL - L_local)
    d_t = np.abs(tL - L_local)
    cost += np.where(
        d_t + CONTRAST_MARGIN < d_s, W_CONTRAST * (d_s - d_t - CONTRAST_MARGIN), 0.0
    )

    hue_weight = min(1.0, sC / 40.0)
    cost += (
        W_HUE
        * hue_weight
        * (np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0)
    )
    cost += W_CHROMA * np.abs(tC - sC)

    if sL < SHADOW_L:
        shadow_factor = (SHADOW_L - sL) / SHADOW_L
        hue_dist = np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0
        chroma_up = np.maximum(0.0, tC - sC)
        cost += shadow_factor * (
            W_SHADOW_HUE * hue_dist + W_SHADOW_CHROMA_UP * chroma_up
        )

    if sC >= 10.0:
        cost += np.where(
            tC <= PHOTO_NEUTRAL_C_MAX, 10.0 + 0.02 * np.maximum(0.0, sC - tC), 0.0
        )

    return cost, tC


def _prefilter_top_pick(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
    near_extreme_neutral: bool,
    k: int,
) -> int:
    idxs = _prefilter_topk(s_lab, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]
    dE = _ciede2000_vec(s_lab, cand_lab)
    comp_base, tC = _photo_cost_components(s_lab, s_lch, L_local, w_local_eff, cand_lch)
    cost = dE + comp_base
    if near_extreme_neutral:
        cost = np.where(tC > PHOTO_NEUTRAL_C_MAX, cost + 1e6, cost)
    return int(idxs[int(np.argmin(cost))])


def dither_photo(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: list[PaletteItem],
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
) -> np.ndarray:
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    lab_img = srgb_to_lab(img_rgb).reshape(H, W, 3)
    L_src = lab_img[..., 0]
    L_loc = _box_blur_2d(L_src, BLUR_RADIUS)
    G = _grad_mag(L_src)
    ab_bias = _estimate_ab_bias(lab_img, alpha)

    errL = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        serp_left = (y % 2) == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)
        neighbours = (
            ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
            if serp_left
            else ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        )
        for x in xs:
            if alpha[y, x] == 0:
                continue

            wloc = max(0.2, 1.0 - (G[y, x] / GRAD_K))

            sL = float(lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(lab_img[y, x, 1]) - ab_bias[0]
            sb = float(lab_img[y, x, 2]) - ab_bias[1]
            s_lab_adj = np.array([sL, sa, sb], dtype=np.float32)

            C = float(np.hypot(s_lab_adj[1], s_lab_adj[2]))
            h = (np.degrees(np.arctan2(s_lab_adj[2], s_lab_adj[1])) + 360.0) % 360.0
            s_lch_adj = np.array([s_lab_adj[0], C, h], dtype=np.float32)

            near_extreme_neutral = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            j = _prefilter_top_pick(
                s_lab_adj,
                s_lch_adj,
                float(L_loc[y, x]),
                W_LOCAL_BASE * wloc,
                pal_lab_mat,
                pal_lch_mat,
                near_extreme_neutral,
                PHOTO_TOPK,
            )

            r, g, b = palette[j].rgb
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b

            tgt_L = float(pal_lch_mat[j, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            for dx, dy, w in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

    return out
