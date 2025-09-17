# palette_map/utils.py
from __future__ import annotations

"""
Shared utilities for palette_map.

Includes small maths helpers, image I/O, palette lookups, progress formatting,
lightweight image-space operations used by both pixel and photo modes, and tidy logging.
"""

from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
from PIL import Image

from .core_types import U8Image, U8Mask, Lab, NameOf
from .colour_convert import rgb_to_lab


#  Time / size formatting


def format_seconds_compact(seconds: float) -> str:
    """Human-friendly seconds: '<ms>ms', '<s>s', or 'Mm Ss'."""
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.3f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds - 60 * minutes:.1f}s"


def format_eta(seconds: float | None) -> str:
    """Format an ETA in seconds as 'Hh Mm', 'Mm Ss', 'Ss', or '--:--' for unknown."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    if total >= 3600:
        hours = total // 3600
        minutes = (total % 3600) // 60
        return f"{hours}h {minutes}m"
    if total >= 60:
        minutes = total // 60
        rem = total % 60
        return f"{minutes}m {rem}s"
    return f"{total}s"


def format_total_duration_compact(seconds: float) -> str:
    """Compact total duration: 'Mm Ss', 'Ss.s', or 'ms'."""
    if seconds >= 60.0:
        minutes = int(seconds // 60)
        rem = int(round(seconds - 60 * minutes))
        return f"{minutes}m {rem}s"
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000.0:.1f}ms"


# Small stats / colour helpers


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Weighted percentile for quantile in [0,1], robust to scalar / 0-D inputs."""
    values_flat = np.asarray(values, dtype=np.float64).ravel()
    weights_flat = np.asarray(weights, dtype=np.float64).ravel()

    if values_flat.size == 0:
        return float("nan")

    # Align shapes
    if weights_flat.size == 1 and values_flat.size > 1:
        weights_flat = np.full_like(
            values_flat, float(weights_flat[0]), dtype=np.float64
        )
    elif weights_flat.size != values_flat.size:
        try:
            weights_flat = np.broadcast_to(weights_flat, values_flat.shape).astype(
                np.float64, copy=False
            )
        except ValueError:
            # Fallback: equal weights
            weights_flat = np.ones_like(values_flat, dtype=np.float64)

    order = np.argsort(values_flat)
    values_sorted = values_flat[order]
    weights_sorted = np.maximum(weights_flat[order], 0.0)

    cum_weights = np.cumsum(weights_sorted)
    if cum_weights.size == 0:
        return float(values_sorted[-1])
    total_weight = float(cum_weights[-1])
    if total_weight <= 0.0:
        return float(values_sorted[-1])

    q = float(np.clip(quantile, 0.0, 1.0))
    target = q * total_weight
    idx = int(np.searchsorted(cum_weights, target, side="left"))
    idx = min(idx, values_sorted.size - 1)
    return float(values_sorted[idx])


def unique_visible_rgb(
    image_rgb: U8Image, alpha_mask: U8Mask
) -> Tuple[U8Image, np.ndarray]:
    """Return (unique RGB rows among alpha>0, counts)."""
    visible_mask = alpha_mask > 0
    if not np.any(visible_mask):
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    flat_rgb = image_rgb[visible_mask].reshape(-1, 3)
    uniques, counts = np.unique(flat_rgb, axis=0, return_counts=True)
    return uniques.astype(np.uint8, copy=False), counts.astype(np.int64, copy=False)


def nearest_palette_indices_lab_distance(src_lab: Lab, pal_lab: Lab) -> np.ndarray:
    """For each source Lab row, pick nearest palette row by Euclidean distance."""
    diff = pal_lab[None, :, :] - src_lab[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1).astype(np.int32)


def topk_indices_by_lab_distance(
    source_lab: np.ndarray, palette_lab: Lab, k: int
) -> np.ndarray:
    """Indices of the k nearest palette rows by Euclidean distance in Lab."""
    diff = palette_lab - source_lab
    dist2 = (diff * diff).sum(axis=1)
    if k >= dist2.shape[0]:
        return np.argsort(dist2)
    partial = np.argpartition(dist2, k)[:k]
    return partial[np.argsort(dist2[partial])]


def hue_distance_degrees(
    source_hue_deg: float, target_hues_deg: np.ndarray
) -> np.ndarray:
    """Minimal absolute hue distance in degrees for vector 'target_hues_deg'."""
    delta = np.abs(target_hues_deg - source_hue_deg)
    return np.where(delta > 180.0, 360.0 - delta, delta)


# I/O / image helpers


def pillow_resample_from_name(name: str) -> int:
    """Map a string to a Pillow resampling filter enum."""
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    return Image.Resampling.BICUBIC  # default


def load_image_rgba(path: Path) -> Tuple[U8Image, U8Mask]:
    """Load an image with Pillow, convert to RGBA, return (rgb, alpha)."""
    pil_img = Image.open(path).convert("RGBA")
    arr = np.array(pil_img, dtype=np.uint8)
    return arr[..., :3], arr[..., 3]


def save_png_rgba(path: Path, rgb: U8Image, alpha: U8Mask) -> None:
    """Save RGB and alpha arrays as a PNG file."""
    out = np.concatenate([rgb, alpha[..., None]], axis=-1)
    Image.fromarray(out).save(path)


def colour_usage_report(
    mapped_rgb: U8Image, alpha_mask: U8Mask, name_of: NameOf
) -> List[Tuple[str, str, int]]:
    """
    Compute a simple colour usage report for visible pixels.

    Returns a list of (hex, name, count) sorted by count descending.
    """
    visible_mask = alpha_mask > 0
    if not np.any(visible_mask):
        return []
    flat = mapped_rgb[visible_mask].reshape(-1, 3)
    uniques, counts = np.unique(flat, axis=0, return_counts=True)
    report: List[Tuple[str, str, int]] = []
    for rgb_row, count in sorted(zip(uniques, counts), key=lambda x: -int(x[1])):
        hex_str = f"#{int(rgb_row[0]):02x}{int(rgb_row[1]):02x}{int(rgb_row[2]):02x}"
        report.append((hex_str, name_of.get(hex_str, "?"), int(count)))
    return report


def split_rows_into_parts(height: int, parts: int) -> List[Tuple[int, int]]:
    """Partition range [0, height) into ~parts contiguous [start, end) row spans."""
    parts = max(1, int(parts))
    step = (height + parts - 1) // parts
    return [(start, min(start + step, height)) for start in range(0, height, step)]


# Lightweight image-space ops


def box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """2D box blur using an integral image. Returns float32 array."""
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    height, width = arr.shape
    src = arr.astype(np.float32, copy=False)
    integ = np.cumsum(np.cumsum(src, axis=0), axis=1)
    integ_pad = np.zeros((height + 1, width + 1), dtype=np.float32)
    integ_pad[1:, 1:] = integ
    ys = np.arange(height, dtype=np.int32)[:, None]
    xs = np.arange(width, dtype=np.int32)[None, :]
    y1 = np.clip(ys - radius, 0, height - 1)
    y2 = np.clip(ys + radius, 0, height - 1)
    x1 = np.clip(xs - radius, 0, width - 1)
    x2 = np.clip(xs + radius, 0, width - 1)
    region_sum = (
        integ_pad[y2 + 1, x2 + 1]
        - integ_pad[y1, x2 + 1]
        - integ_pad[y2 + 1, x1]
        + integ_pad[y1, x1]
    )
    area = (y2 - y1 + 1) * (x2 - x1 + 1)
    return (region_sum / area).astype(np.float32, copy=False)


def gradient_magnitude_l1(lightness: np.ndarray) -> np.ndarray:
    """L1 gradient magnitude with a small blur. Returns float32 array."""
    gx = np.zeros_like(lightness, dtype=np.float32)
    gy = np.zeros_like(lightness, dtype=np.float32)
    gx[:, 1:] = np.abs(lightness[:, 1:] - lightness[:, :-1])
    gy[1:, :] = np.abs(lightness[1:, :] - lightness[:-1, :])
    return box_blur_2d(gx + gy, 1)


#  CLI / progress logging


def print_progress_line(message: str, final: bool = False) -> None:
    """Print a single-line progress message that overwrites previous output."""
    import sys as _sys

    _sys.stdout.write("\r\033[K" + message)
    _sys.stdout.flush()
    if final:
        _sys.stdout.write("\n")
        _sys.stdout.flush()


def enable_line_buffered_stdout() -> None:
    """
    Enable line-buffered stdout when supported.
    Helps live progress printing in terminals that expose .reconfigure().
    """
    import sys

    reconfig = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig):
        try:
            reconfig(line_buffering=True, write_through=True)
        except Exception:
            pass


# Pretty logging


def format_bool_on_off(value: Any) -> str:
    """Pretty boolean: 'on'/'off' for bools; str(value) otherwise."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def format_number_compact(value: Any) -> str:
    """Pretty number: 1_234 style for ints; compact for floats; passthrough otherwise."""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text
    return str(value)


def format_percentage(x: float, decimals: int = 1) -> str:
    """Format percentage; accepts 0..1 or 0..100 inputs."""
    val = x * 100.0 if 0.0 <= x <= 1.0 else x
    return f"{val:.{decimals}f}%"


def key_value_pairs_to_string(
    pairs: Iterable[Tuple[str, Any]], sep: str = "  ", eq: str = ": "
) -> str:
    """
    Format (name, value) pairs as 'Name: value' blocks separated by sep.
    Uses format_bool_on_off / format_number_compact for readability.
    """
    out: List[str] = []
    for name, value in pairs:
        display = (
            format_bool_on_off(value)
            if isinstance(value, bool)
            else format_number_compact(value)
        )
        out.append(f"{name}{eq}{display}")
    return sep.join(out)


def print_config_line(
    section: str, pairs: Iterable[Tuple[str, Any]], debug: bool
) -> None:
    """
    Emit a single human-readable config line, e.g.:
      [photo] Workers: 8  Blocks: on  Processes: 6  Threads/Proc: 1  Top-K: 16
    Routes to debug() when debug=True, else to log().
    """
    line = f"[{section}] {key_value_pairs_to_string(pairs)}"
    (debug_log if debug else log)(line)


def print_banner(title: str) -> None:
    """Section banner."""
    print(f"\n=== {title} ===", flush=True)


def log(message: str) -> None:
    """Plain log line."""
    print(message, flush=True)


def debug_log(message: str) -> None:
    """Debug log line."""
    print(f"[debug] {message}", flush=True)


def warn(message: str) -> None:
    """Warning log line."""
    print(f"[warn] {message}", flush=True)


def error(message: str) -> None:
    """Error log line to stderr."""
    import sys

    print(f"[error] {message}", file=sys.stderr, flush=True)


#  Nearest-map fallback


def map_nearest_in_lab_space(
    rgb_in: U8Image, alpha_mask: U8Mask, pal_rgb: U8Image, pal_lab: Lab
) -> U8Image:
    """Nearest-neighbour palette map using Lab distance for visible pixels only."""
    src_lab: Lab = rgb_to_lab(rgb_in.astype(np.float32))
    flat_lab = src_lab.reshape(-1, 3)
    visible_flat = alpha_mask.reshape(-1) > 0
    indices = np.zeros(flat_lab.shape[0], dtype=np.int32)
    if np.any(visible_flat):
        nearest_idx = nearest_palette_indices_lab_distance(
            flat_lab[visible_flat], pal_lab.astype(np.float32)
        )
        indices[visible_flat] = nearest_idx
    return pal_rgb[indices].reshape(rgb_in.shape).astype(np.uint8)


__all__ = [
    # formatting (new)
    "format_seconds_compact",
    "format_eta",
    "format_total_duration_compact",
    # pretty formatting helpers (new)
    "format_bool_on_off",
    "format_number_compact",
    "format_percentage",
    # stats / colour helpers (new)
    "weighted_percentile",
    "unique_visible_rgb",
    "nearest_palette_indices_lab_distance",
    "topk_indices_by_lab_distance",
    "hue_distance_degrees",
    # I/O helpers (new)
    "pillow_resample_from_name",
    "load_image_rgba",
    "save_png_rgba",
    "colour_usage_report",
    "split_rows_into_parts",
    # image-space ops
    "box_blur_2d",
    "gradient_magnitude_l1",
    # logging / progress (new)
    "print_progress_line",
    "enable_line_buffered_stdout",
    "print_config_line",
    "print_banner",
    "log",
    "debug_log",
    "warn",
    "error",
    # fallback mapper (new + alias)
    "map_nearest_in_lab_space",
]
