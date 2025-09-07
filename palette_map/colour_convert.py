# palette_map/colour_convert.py
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .core_types import Lab, Lch  # NDArray[np.float32]

"""
Colour conversions and metrics (D65).

Exports:
- rgb_to_linear(u)
- rgb_to_lab(rgb)
- lab_to_lch(lab)
- delta_e2000_pair(lab1, lab2)
- delta_e2000_vec(src_lab, cand_lab)

Compat aliases:
- ciede2000_pair === delta_e2000_pair
- ciede2000_vec  === delta_e2000_vec
"""


# ---------------- sRGB → linear ----------------------------------------------


def rgb_to_linear(u: np.ndarray) -> np.ndarray:
    """
    sRGB (nonlinear 0..1) → linear RGB (0..1). Vectorized. Returns float32.
    Accepts any shape (..., 3).
    """
    u = u.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        out = np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
    return out.astype(np.float32, copy=False)


# ---------------- sRGB → Lab (D65) -------------------------------------------


def rgb_to_lab(rgb: np.ndarray) -> Lab:
    """
    sRGB → CIE Lab (D65). Accepts uint8 [0..255] or float32/float64 [0..1].
    Preserves input shape (..., 3). Returns float32.
    """
    arr = rgb.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # sRGB → linear
    rl = rgb_to_linear(arr[..., 0])
    gl = rgb_to_linear(arr[..., 1])
    bl = rgb_to_linear(arr[..., 2])

    # linear RGB → XYZ (D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # XYZ → Lab (D65 white)
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

    out = np.empty(arr.shape, dtype=np.float32)
    out[..., 0] = L
    out[..., 1] = a
    out[..., 2] = b
    return out


# ---------------- Lab → LCh ---------------------------------------------------


def lab_to_lch(lab: Lab) -> Lch:
    """
    Lab(float32)[...,3] → LCh(float32)[...,3].
    L*=L, C*=sqrt(a^2+b^2), h°=atan2(b,a) in degrees [0,360).
    Shape is preserved.
    """
    orig = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32, copy=False)
    L = flat[:, 0]
    a = flat[:, 1]
    b = flat[:, 2]
    C = np.hypot(a, b)
    h = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    lch = np.stack([L, C, h], axis=1).astype(np.float32, copy=False)
    return lch.reshape(orig)


# ---------------- CIEDE2000 (shared) -----------------------------------------


def delta_e2000_pair(
    lab1: Sequence[float] | NDArray[np.floating],
    lab2: Sequence[float] | NDArray[np.floating],
) -> float:
    """
    CIEDE2000 distance between two Lab colours.
    Scalar implementation (reference-consistent).
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - math.sqrt((Cbar**7) / (Cbar**7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def _h(a: float, b: float) -> float:
        if a == 0.0 and b == 0.0:
            return 0.0
        ang = math.degrees(math.atan2(b, a))
        return ang + 360.0 if ang < 0.0 else ang

    h1p = _h(a1p, b1)
    h2p = _h(a2p, b2)

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

    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        hbarp = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff <= 180.0:
            hbarp = 0.5 * hsum
        elif hsum < 360.0:
            hbarp = 0.5 * (hsum + 360.0)
        else:
            hbarp = 0.5 * (hsum - 360.0)

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hbarp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hbarp))
        + 0.32 * math.cos(math.radians(3.0 * hbarp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hbarp - 63.0))
    )

    dTheta = 30.0 * math.exp(-(((hbarp - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * math.sqrt((Cbarp**7) / (Cbarp**7 + 25.0**7))

    Sl = 1.0 + (0.015 * ((Lbar - 50.0) ** 2.0)) / math.sqrt(
        20.0 + ((Lbar - 50.0) ** 2.0)
    )
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -math.sin(math.radians(2.0 * dTheta)) * Rc

    kL = kC = kH = 1.0
    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def delta_e2000_vec(s: Lab, cand: Lab) -> NDArray[np.float32]:
    """
    Row-wise ΔE for one source Lab vs many candidate Labs.
    Uses the exact scalar routine per row to keep results identical
    across modules (good for small k in top-k searches).
    """
    s = np.asarray(s, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    out = np.empty((cand.shape[0],), dtype=np.float32)
    for i in range(cand.shape[0]):
        out[i] = delta_e2000_pair(s, cand[i])
    return out


# ---- Compat aliases ----------------------------------------------------------

ciede2000_pair = delta_e2000_pair
ciede2000_vec = delta_e2000_vec


__all__ = [
    "rgb_to_linear",
    "rgb_to_lab",
    "lab_to_lch",
    "delta_e2000_pair",
    "delta_e2000_vec",
    "ciede2000_pair",
    "ciede2000_vec",
]
