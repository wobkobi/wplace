# palette_map/analysis.py
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from PIL import ImageCms
except Exception:  # pragma: no cover
    ImageCms = None  # type: ignore[assignment]

from .core_types import (
    Lab,
    Lch,
    PaletteItem,
    RGBTuple,
    SourceItem,
    U8Image,
    U8Mask,
)

# =========
# Hex / RGB
# =========
def hex_to_rgb(h: str) -> RGBTuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = list(rgb)
    return f"#{r:02x}{g:02x}{b:02x}"

# ======================
# sRGB → Lab / Lab → LCh
# ======================
def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def srgb_to_lab_batch(rgb_u8: np.ndarray) -> Lab:
    orig = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    rl = _srgb_to_linear(flat[:, 0])
    gl = _srgb_to_linear(flat[:, 1])
    bl = _srgb_to_linear(flat[:, 2])
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
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
    return lab.reshape(orig)


def lab_to_lch_batch(lab: Lab) -> Lch:
    orig = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32)
    L = flat[:, 0]
    a = flat[:, 1]
    b = flat[:, 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a))
    h = (h + 360.0) % 360.0
    lch = np.stack([L, C, h], axis=1).astype(np.float32)
    return lch.reshape(orig)


def srgb_to_lab(rgb_u8: np.ndarray) -> Lab:
    return srgb_to_lab_batch(rgb_u8)


def lab_to_lch(lab: Lab) -> Lch:
    return lab_to_lch_batch(lab)

# ==================
# DeltaE & hue utils
# ==================
def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = [float(x) for x in lab1.tolist()]
    L2, a2, b2 = [float(x) for x in lab2.tolist()]
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - math.sqrt((Cm**7) / (Cm**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = (math.degrees(math.atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (math.degrees(math.atan2(b2, a2p)) + 360.0) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif dhp > 180.0:
        dhp -= 360.0
    elif dhp < -180.0:
        dhp += 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)
    Lpm = (L1 + L2) / 2.0
    Cpm = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hpm = 0.5 * (h1p + h2p)
    else:
        hpm = (
            0.5 * (h1p + h2p + 360.0)
            if (h1p + h2p) < 360.0
            else 0.5 * (h1p + h2p - 360.0)
        )
    T = (
        1
        - 0.17 * math.cos(math.radians(hpm - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hpm))
        + 0.32 * math.cos(math.radians(3.0 * hpm + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hpm - 63.0))
    )
    d_ro = 30.0 * math.exp(-(((hpm - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * math.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1.0 + (0.015 * (Lpm - 50.0) ** 2.0) / math.sqrt(20.0 + (Lpm - 50.0) ** 2.0)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc
    dE = math.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return float(dE)


def hue_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d

# =========
# Image IO
# =========
def binarise_alpha(alpha_or_img: U8Mask | Image.Image, threshold: int = 127) -> U8Mask:
    if isinstance(alpha_or_img, Image.Image):
        im = alpha_or_img
        if im.mode == "RGBA":
            a = np.array(im.split()[-1], dtype=np.uint8)
        elif im.mode == "L":
            a = np.array(im, dtype=np.uint8)
        else:
            a = np.array(im.convert("L"), dtype=np.uint8)
    else:
        a = np.asarray(alpha_or_img, dtype=np.uint8)

    out = np.zeros_like(a, dtype=np.uint8)
    out[a >= np.uint8(threshold)] = 255
    return out


def load_image_rgba(path: Path) -> Tuple[U8Image, U8Mask]:
    with Image.open(path) as im0:
        im1 = ImageOps.exif_transpose(im0)
        icc = im1.info.get("icc_profile")
        if icc and ImageCms is not None:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(
                    im1,
                    src_prof,
                    dst_prof,
                    renderingIntent=ImageCms.INTENT_PERCEPTUAL,
                    outputMode="RGBA",
                )
                if im is None:
                    im = im1.convert("RGBA")
            except Exception:
                im = im1.convert("RGBA")
        else:
            im = im1.convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    a = binarise_alpha(arr[..., 3])
    return rgb, a


def save_image_rgba(path: Path, rgb: U8Image, alpha: U8Mask) -> Path:
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = binarise_alpha(alpha)
    Image.fromarray(out, mode="RGBA").save(path)
    return path


def resize_rgba_height(
    rgb: U8Image, alpha: U8Mask, dst_h: Optional[int], resample: Image.Resampling
) -> Tuple[U8Image, U8Mask]:
    H0, W0, _ = rgb.shape
    if dst_h is None or dst_h <= 0 or dst_h >= H0:
        return rgb, binarise_alpha(alpha)
    dst_w = int(round(W0 * (dst_h / float(H0))))
    rgba = np.zeros((H0, W0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    im = Image.fromarray(rgba, mode="RGBA")
    im2 = im.resize((dst_w, dst_h), resample=resample)
    arr = np.array(im2, dtype=np.uint8)
    return arr[..., :3], binarise_alpha(arr[..., 3])


def is_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.seek(0)
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False

# ========================
# Palette / source builders
# ========================
def build_palette(
    hex_name_pairs: List[Tuple[str, str]],
) -> Tuple[List[PaletteItem], Dict[RGBTuple, str], Lab, Lch]:
    rgbs = np.array([hex_to_rgb(hx) for hx, _ in hex_name_pairs], dtype=np.uint8)
    labs = srgb_to_lab(rgbs).reshape(-1, 3)
    lchs = lab_to_lch(labs).reshape(-1, 3)

    items: List[PaletteItem] = []
    name_of: Dict[RGBTuple, str] = {}
    for i, (_hx, name) in enumerate(hex_name_pairs):
        rgb_t: RGBTuple = (int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]))
        items.append(
            PaletteItem(rgb=rgb_t, name=name, lab=labs[i].copy(), lch=lchs[i].copy())
        )
        name_of[rgb_t] = name
    return items, name_of, labs, lchs


def unique_colors_and_counts(rgb: U8Image, alpha: U8Mask) -> List[Tuple[RGBTuple, int]]:
    mask = alpha != 0
    if not mask.any():
        return []
    samples = rgb[mask]
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = samples.view(dt).reshape(-1)
    uniq, counts = np.unique(flat, return_counts=True)
    rs = uniq["r"].astype(int)
    gs = uniq["g"].astype(int)
    bs = uniq["b"].astype(int)
    items = [
        ((int(rs[i]), int(gs[i]), int(bs[i])), int(counts[i])) for i in range(len(uniq))
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def build_sources(rgb: U8Image, alpha: U8Mask) -> Tuple[List[SourceItem], Lab, Lch]:
    items = unique_colors_and_counts(rgb, alpha)
    if not items:
        return (
            [],
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    src_rgb = np.array([c for c, _ in items], dtype=np.uint8)
    src_lab = srgb_to_lab(src_rgb).reshape(-1, 3)
    src_lch = lab_to_lch(src_lab).reshape(-1, 3)
    sources: List[SourceItem] = []
    for i, ((r, g, b), n) in enumerate(items):
        sources.append(
            SourceItem(
                rgb=(int(r), int(g), int(b)),
                count=int(n),
                lab=src_lab[i].copy(),
                lch=src_lch[i].copy(),
            )
        )
    return sources, src_lab, src_lch

# =================
# Misc. conveniences
# =================
def prefilter_topk(s_lab: np.ndarray, pal_lab_mat: np.ndarray, k: int) -> np.ndarray:
    diff = pal_lab_mat - s_lab
    de2 = (diff * diff).sum(axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def decide_auto_mode(img_rgb: U8Image, alpha: U8Mask) -> str:
    items = unique_colors_and_counts(img_rgb, alpha)
    total = sum(n for _c, n in items)
    n_uniques = len(items)
    topk = sum(n for _c, n in items[: min(16, n_uniques)])
    share = (topk / total) if total > 0 else 1.0
    return "pixel" if (n_uniques <= 512 or share >= 0.80) else "photo"

# =================
# CLI helper output
# =================
def print_color_usage(
    out_rgb: U8Image, alpha: U8Mask, name_of: Dict[RGBTuple, str], top: int = 16
) -> None:
    """Print per-unique usage for visible pixels, showing hex codes."""
    mask = alpha != 0
    if not np.any(mask):
        print("No visible pixels")
        return
    vis = out_rgb[mask].reshape(-1, 3)
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = vis.view(dt).reshape(-1)
    uniq, counts = np.unique(flat, return_counts=True)
    total = int(counts.sum())

    rows: List[Tuple[str, int, float]] = []
    for i in range(len(uniq)):
        rgb = (int(uniq["r"][i]), int(uniq["g"][i]), int(uniq["b"][i]))
        n = int(counts[i])
        pct = 100.0 * n / max(1, total)
        rows.append((rgb_to_hex(rgb), n, pct))

    rows.sort(key=lambda t: -t[1])

    print("[debug] per-unique mapping (hex):")
    for hx, n, pct in rows[:top]:
        print(f"  {hx}  count={n:6d}  {pct:5.1f}%")



__all__ = [
    # hex/rgb
    "hex_to_rgb",
    "rgb_to_hex",
    # conversions
    "srgb_to_lab",
    "lab_to_lch",
    "srgb_to_lab_batch",
    "lab_to_lch_batch",
    # deltaE / hue
    "ciede2000_pair",
    "hue_diff_deg",
    # image io
    "binarise_alpha",
    "load_image_rgba",
    "save_image_rgba",
    "resize_rgba_height",
    "is_image_file",
    # palette/source builders
    "build_palette",
    "unique_colors_and_counts",
    "build_sources",
    # misc
    "prefilter_topk",
    "decide_auto_mode",
    # cli helper
    "print_color_usage",
]
