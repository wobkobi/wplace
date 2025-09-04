# palette_map/color_convert.py
from __future__ import annotations

import numpy as np
from .core_types import Lab, Lch

"""
sRGB ↔ Lab/LCh (D65). Vectorized NumPy implementations.

Exports:
- rgb_to_linear(u)
- rgb_to_lab(rgb)
- lab_to_lch_batch(lab)
- lab_to_lch(lab)
- srgb_to_lab(rgb)            # legacy shim → rgb_to_lab
- srgb_to_lab_batch(rgb)      # legacy shim → rgb_to_lab
"""


def rgb_to_linear(u: np.ndarray) -> np.ndarray:
    """
    sRGB (nonlinear 0..1) → linear RGB (0..1). Vectorized. Returns float32.
    Accepts any shape (..., 3).
    """
    u = u.astype(np.float32, copy=False)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(
        np.float32, copy=False
    )


def rgb_to_lab(rgb: np.ndarray) -> Lab:
    """
    sRGB → CIE Lab (D65). Accepts uint8 [0..255] or float32/float64 [0..1].
    Preserves input shape (..., 3). Returns float32.
    """
    arr = rgb.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # sRGB → linear
    rl, gl, bl = (
        rgb_to_linear(arr[..., 0]),
        rgb_to_linear(arr[..., 1]),
        rgb_to_linear(arr[..., 2]),
    )

    # linear RGB → XYZ (D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # XYZ → Lab (D65 white)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn
    e, k = 216.0 / 24389.0, 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
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
    return out  # type: ignore[return-value]


def lab_to_lch_batch(lab: Lab) -> Lch:
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
    return lch.reshape(orig)  # type: ignore[return-value]


def lab_to_lch(lab: Lab) -> Lch:
    """Convenience wrapper; identical to lab_to_lch_batch for any shape."""
    return lab_to_lch_batch(lab)


__all__ = [
    "rgb_to_linear",
    "rgb_to_lab",
    "lab_to_lch_batch",
    "lab_to_lch",
]
