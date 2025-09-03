# palette_map/color_convert.py
from __future__ import annotations

import numpy as np

from .core_types import Lab, Lch

"""
sRGB ↔ Lab/LCh conversions (D65) with vectorized NumPy implementations.

Exports:
- _srgb_to_linear(u)
- srgb_to_lab_batch(rgb_u8)
- lab_to_lch_batch(lab)
- srgb_to_lab(rgb_u8)
- lab_to_lch(lab)
"""


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """
    sRGB (nonlinear 0..1) → linear RGB (0..1), vectorized.
    Accepts float array in [0,1]; returns float32.
    """
    u = u.astype(np.float32)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def srgb_to_lab_batch(rgb_u8: np.ndarray) -> Lab:
    """
    Vectorized sRGB(uint8)[...,3] → Lab(float32)[...,3] under D65.
    Works for (H,W,3) or (N,3).
    """
    orig_shape = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0

    rl = _srgb_to_linear(flat[:, 0])
    gl = _srgb_to_linear(flat[:, 1])
    bl = _srgb_to_linear(flat[:, 2])

    # sRGB to XYZ (D65)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Reference white (D65)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0).astype(
            np.float32
        )

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = np.stack([L, a, b], axis=1).astype(np.float32)
    return lab.reshape(orig_shape)


def lab_to_lch_batch(lab: Lab) -> Lch:
    """
    Vectorized Lab(float32)[...,3] → LCh(float32)[...,3].
    L*=L, C*=sqrt(a^2+b^2), h°=atan2(b,a) in degrees [0,360).
    """
    orig_shape = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32)
    L = flat[:, 0]
    a = flat[:, 1]
    b = flat[:, 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a))
    h = (h + 360.0) % 360.0
    lch = np.stack([L, C, h], axis=1).astype(np.float32)
    return lch.reshape(orig_shape)


def srgb_to_lab(rgb_u8: np.ndarray) -> Lab:
    """Convenience wrapper; identical to srgb_to_lab_batch for any shape."""
    return srgb_to_lab_batch(rgb_u8)


def lab_to_lch(lab: Lab) -> Lch:
    """Convenience wrapper; identical to lab_to_lch_batch for any shape."""
    return lab_to_lch_batch(lab)


__all__ = [
    "_srgb_to_linear",
    "srgb_to_lab_batch",
    "lab_to_lch_batch",
    "srgb_to_lab",
    "lab_to_lch",
]
