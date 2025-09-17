# palette_map/colour_convert.py
from __future__ import annotations

"""
Colour conversions and metrics (D65).

Exports:
  rgb_to_linear(srgb)
  rgb_to_lab(rgb)
  lab_to_lch(lab)
  delta_e2000_pair(lab1, lab2)
  delta_e2000_vec(src_lab, cand_lab)
  rgb_to_lab_threaded(rgb, workers)

Compat aliases:
  ciede2000_pair === delta_e2000_pair
  ciede2000_vec  === delta_e2000_vec
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .core_types import Lab, Lch  # NDArray[np.float32]


# sRGB to linear


def rgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB (non-linear 0..1) to linear RGB (0..1). Vectorised.
    Args:
      srgb: array[...,3] in 0..1 (float)
    Returns:
      float32 array[...,3]
    """
    srgb_f = srgb.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        linear = np.where(
            srgb_f <= 0.04045, srgb_f / 12.92, ((srgb_f + 0.055) / 1.055) ** 2.4
        )
    return linear.astype(np.float32, copy=False)


# sRGB to Lab (D65)


def rgb_to_lab(rgb: np.ndarray) -> Lab:
    """
    sRGB to CIE Lab (D65).
    Accepts uint8 [0..255] or float [0..1]. Preserves shape (...,3). Returns float32.
    """
    rgb_f = rgb.astype(np.float32, copy=False)
    if rgb_f.max() > 1.0:
        rgb_f = rgb_f / 255.0

    r_lin = rgb_to_linear(rgb_f[..., 0])
    g_lin = rgb_to_linear(rgb_f[..., 1])
    b_lin = rgb_to_linear(rgb_f[..., 2])

    # Linear RGB -> XYZ (D65)
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    # Reference white (D65)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn
    e, k = 216.0 / 24389.0, 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.where(t > e, np.cbrt(t), (k * t + 16.0) / 116.0).astype(
                np.float32, copy=False
            )

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    out = np.empty(rgb_f.shape, dtype=np.float32)
    out[..., 0] = L
    out[..., 1] = a
    out[..., 2] = b
    return out


# Lab to LCh


def lab_to_lch(lab: Lab) -> Lch:
    """
    Lab[...,3] to LCh[...,3] (degrees in [0,360)).
    Returns float32 with shape preserved.
    """
    orig_shape = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32, copy=False)
    L = flat[:, 0]
    a = flat[:, 1]
    b = flat[:, 2]
    C = np.hypot(a, b)
    h = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    lch = np.stack([L, C, h], axis=1).astype(np.float32, copy=False)
    return lch.reshape(orig_shape)


# CIEDE2000


def delta_e2000_pair(
    lab1: Sequence[float] | NDArray[np.floating],
    lab2: Sequence[float] | NDArray[np.floating],
) -> float:
    """
    CIEDE2000 distance between two Lab colours.
    Scalar reference implementation.
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - math.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def _hue(a_val: float, b_val: float) -> float:
        if a_val == 0.0 and b_val == 0.0:
            return 0.0
        ang = math.degrees(math.atan2(b_val, a_val))
        return ang + 360.0 if ang < 0.0 else ang

    h1p = _hue(a1p, b1)
    h2p = _hue(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p * C2p == 0.0:
        dhp = 0.0
    elif dhp > 180.0:
        dhp -= 360.0
    elif dhp < -180.0:
        dhp += 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    L_bar = 0.5 * (L1 + L2)
    C_bar_p = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        h_bar_p = h1p + h2p
    else:
        h_sum = h1p + h2p
        h_diff = abs(h1p - h2p)
        if h_diff <= 180.0:
            h_bar_p = 0.5 * h_sum
        elif h_sum < 360.0:
            h_bar_p = 0.5 * (h_sum + 360.0)
        else:
            h_bar_p = 0.5 * (h_sum - 360.0)

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2.0))
    R_c = 2.0 * math.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7))

    S_l = 1.0 + (0.015 * ((L_bar - 50.0) ** 2.0)) / math.sqrt(
        20.0 + ((L_bar - 50.0) ** 2.0)
    )
    S_c = 1.0 + 0.045 * C_bar_p
    S_h = 1.0 + 0.015 * C_bar_p * T
    R_t = -math.sin(math.radians(2.0 * d_theta)) * R_c

    kL = kC = kH = 1.0
    dE = math.sqrt(
        (dLp / (kL * S_l)) ** 2
        + (dCp / (kC * S_c)) ** 2
        + (dHp / (kH * S_h)) ** 2
        + R_t * (dCp / (kC * S_c)) * (dHp / (kH * S_h))
    )
    return float(dE)


def delta_e2000_vec(src_lab: Lab, cand_lab: Lab) -> NDArray[np.float32]:
    """
    Row-wise CIEDE2000 for one source Lab vs many candidate Labs.
    Uses the scalar routine per row for consistent results.

    Args:
      src_lab: Lab [3] or [1,3]
      cand_lab: Lab [N,3]
    Returns:
      float32 array [N]
    """
    s = np.asarray(src_lab, dtype=np.float32)
    cands = np.asarray(cand_lab, dtype=np.float32)
    out = np.empty((cands.shape[0],), dtype=np.float32)
    for i in range(cands.shape[0]):
        out[i] = delta_e2000_pair(s, cands[i])
    return out


# Threaded helpers


def _split_rows(height: int, parts: int) -> list[tuple[int, int]]:
    """Partition height into ~parts contiguous [start, end) row spans."""
    parts = max(1, int(parts))
    step = (height + parts - 1) // parts
    return [(i, min(i + step, height)) for i in range(0, height, step)]


def rgb_to_lab_threaded(rgb: np.ndarray, workers: int) -> Lab:
    """
    Threaded RGB->Lab conversion by splitting rows.

    Args:
      rgb: uint8 or float array [H,W,3]
      workers: number of threads; if <=1 or H<256, runs single-threaded
    Returns:
      Lab float32 array [H,W,3]
    """
    height = int(rgb.shape[0])
    if workers <= 1 or height < 256:
        return rgb_to_lab(rgb).astype(np.float32, copy=False)

    chunks = _split_rows(height, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(rgb_to_lab, rgb[s:e]) for s, e in chunks]
        parts = [f.result().astype(np.float32, copy=False) for f in futures]
    return np.vstack(parts).astype(np.float32, copy=False)


# Compat aliases

ciede2000_pair = delta_e2000_pair
ciede2000_vec = delta_e2000_vec


__all__ = [
    "rgb_to_linear",
    "rgb_to_lab",
    "lab_to_lch",
    "delta_e2000_pair",
    "delta_e2000_vec",
    "rgb_to_lab_threaded",
    "ciede2000_pair",
    "ciede2000_vec",
]
