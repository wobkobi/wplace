# palette_map/utils.py
from __future__ import annotations

"""
Shared utilities for palette_map.

Includes small math helpers, image IO, palette lookups, progress formatting,
and lightweight image-space ops used by both pixel and photo modes.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from .core_types import U8Image, U8Mask, Lab, Lch, NameOf
from .colour_convert import rgb_to_lab


def fmt_secs(s: float) -> str:
    """Format seconds into 'ms' or 'Xm Ys' compact strings."""
    if s < 60.0:
        return f"{s*1000:.1f}ms" if s < 1.0 else f"{s:.3f}s"
    m = int(s // 60)
    return f"{m}m {s - 60*m:.1f}s"


def fmt_eta(seconds: float | None) -> str:
    """Format ETA seconds into 'Hh Mm', 'Mm Ss', or '--:--'."""
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

def _fmt_total_compact(s: float) -> str:
    # minutes + seconds if >=60s; seconds with decimals if <60s; ms if <1s
    if s >= 60.0:
        m = int(s // 60)
        r = int(round(s - 60 * m))
        return f"{m}m {r}s"
    if s >= 1.0:
        return f"{s:.1f}s"
    return f"{s * 1000.0:.1f}ms"

def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile for q in [0,1]."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw.size == 0:
        return float("nan")
    if cw[-1] == 0:
        return float(v[-1])
    pos = q * cw[-1]
    idx = np.searchsorted(cw, pos, side="left")
    return float(v[min(idx, len(v) - 1)])


def unique_visible(rgb: U8Image, alpha: U8Mask) -> Tuple[U8Image, np.ndarray]:
    """Return (unique RGB rows among alpha>0, counts)."""
    vis = alpha > 0
    if not np.any(vis):
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    flat = rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return uniq.astype(np.uint8), counts.astype(np.int64)


def nearest_palette_indices_lab(src_lab: Lab, pal_lab: Lab) -> np.ndarray:
    """For each source Lab row pick nearest palette row by Euclidean distance."""
    diff = pal_lab[None, :, :] - src_lab[:, None, :]
    de2 = np.sum(diff * diff, axis=2)
    return np.argmin(de2, axis=1).astype(np.int32)


def prefilter_topk_lab(s_lab: np.ndarray, pal_lab: Lab, k: int) -> np.ndarray:
    """Indices of k nearest palette rows by Euclidean distance in Lab."""
    diff = pal_lab - s_lab
    de2 = (diff * diff).sum(axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def hue_dist_deg_vec(h1: float, h2: np.ndarray) -> np.ndarray:
    """Minimal absolute hue distance in degrees for vector h2."""
    dh = np.abs(h2 - h1)
    return np.where(dh > 180.0, 360.0 - dh, dh)


def pil_resample(name: str) -> int:
    """Map a string to a Pillow resampling filter enum."""
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    return Image.Resampling.BICUBIC


def load_rgba(path: Path) -> Tuple[U8Image, U8Mask]:
    """Load an image with Pillow, convert to RGBA, return (rgb, alpha)."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    return arr[..., :3], arr[..., 3]


def save_png_rgba(path: Path, rgb: U8Image, alpha: U8Mask) -> None:
    """Save RGB and alpha arrays as a PNG file."""
    out = np.concatenate([rgb, alpha[..., None]], axis=-1)
    Image.fromarray(out).save(path)


def colour_usage_report(
    mapped_rgb: U8Image, alpha: U8Mask, name_of: NameOf
) -> List[Tuple[str, str, int]]:
    """
    Compute a simple colour usage report for visible pixels.

    Returns a list of (hex, name, count) sorted by count desc.
    """
    vis = alpha > 0
    if not np.any(vis):
        return []
    flat = mapped_rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    out: List[Tuple[str, str, int]] = []
    for rgb, c in sorted(zip(uniq, counts), key=lambda x: -int(x[1])):
        h = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        out.append((h, name_of.get(h, "?"), int(c)))
    return out


def split_rows(h: int, parts: int) -> List[Tuple[int, int]]:
    """Partition range [0,h) into ~parts contiguous [start,end) row spans."""
    parts = max(1, int(parts))
    step = (h + parts - 1) // parts
    return [(i, min(i + step, h)) for i in range(0, h, step)]


def box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """2D box blur using an integral image. Returns float32 array."""
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
    """L1 gradient magnitude with a small blur. Returns float32 array."""
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    return box_blur_2d(gx + gy, 1)


def progress_line(msg: str, final: bool = False) -> None:
    """Print a single-line progress message that overwrites previous output."""
    import sys as _sys

    _sys.stdout.write("\r\033[K" + msg)
    _sys.stdout.flush()
    if final:
        _sys.stdout.write("\n")
        _sys.stdout.flush()


def map_nearest_rgb_lab(
    rgb_in: U8Image, alpha: U8Mask, pal_rgb: U8Image, pal_lab: Lab
) -> U8Image:
    """Nearest-neighbour palette map using Lab distance for visible pixels only."""
    src_lab: Lab = rgb_to_lab(rgb_in.astype(np.float32))
    flat = src_lab.reshape(-1, 3)
    vis = alpha.reshape(-1) > 0
    idx = np.zeros(flat.shape[0], dtype=np.int32)
    if np.any(vis):
        idx_vis = nearest_palette_indices_lab(flat[vis], pal_lab.astype(np.float32))
        idx[vis] = idx_vis
    return pal_rgb[idx].reshape(rgb_in.shape).astype(np.uint8)


__all__ = [
    "fmt_secs",
    "fmt_eta",
    "weighted_percentile",
    "unique_visible",
    "nearest_palette_indices_lab",
    "prefilter_topk_lab",
    "hue_dist_deg_vec",
    "pil_resample",
    "load_rgba",
    "save_png_rgba",
    "colour_usage_report",
    "split_rows",
    "box_blur_2d",
    "grad_mag",
    "progress_line",
    "map_nearest_rgb_lab",
]
