#!/usr/bin/env python3
"""
palette_map.py

A small, dependable image remapper that snaps images to a fixed palette.

Key points:
- Supports four modes: --mode {auto, pixel, photo, bw}
  * auto  : Chooses "pixel" for low/medium colour complexity, otherwise "photo".
  * pixel : For pixel art / UI sprites. No dithering, exact palette mapping.
            Includes a post-pass that reduces multiple different neutral greys
            collapsing into the same grey (the "grey blending" issue).
  * photo : Lightweight error-diffusion dithering guided by local lightness,
            followed by a strict palette lock. Good for photos/paintings.
  * bw    : Black & white mode (actually greyscale ramp only). Uses only:
            #000000, #3c3c3c, #787878, #aaaaaa, #d2d2d2, #ffffff

- Resizing: optional downscale by a target height. Never upscales.
  Uses BOX for "pixel" and "bw" (keeps crisp edges), LANCZOS for "photo".

- Alpha: hard-thresholded to 0 or 255 to avoid semi-transparent surprises.

- Performance:
  * Candidate lists for "pixel" mode are precomputed (optionally multi-process).
  * No recursion in the assignment step — avoids recursion errors.
  * The "photo" mode uses simple vectorised maths and a compact dither.

- Debug: --debug prints tidy and consistent lines with dE/dL/dC/dh numbers,
  plus colour usage at the end.

"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

# User Palette

# The palette is written as (hex, name) pairs. Order matters a little for the
# grey rebalancing pass (we detect the explicit grey ramp by hex values).
PALETTE: List[Tuple[str, str]] = [
    ("#ed1c24", "Red"),
    ("#d18078", "Peach"),
    ("#fa8072", "Light Red"),
    ("#9b5249", "Dark Peach"),
    ("#fab6a4", "Light Peach"),
    ("#e45c1a", "Dark Orange"),
    ("#684634", "Dark Brown"),
    ("#ffc5a5", "Light Beige"),
    ("#d18051", "Dark Beige"),
    ("#ff7f27", "Orange"),
    ("#7b6352", "Dark Tan"),
    ("#f8b277", "Beige"),
    ("#d6b594", "Light Tan"),
    ("#9c846b", "Tan"),
    ("#dba463", "Light Brown"),
    ("#95682a", "Brown"),
    ("#f6aa09", "Gold"),
    ("#9c8431", "Dark Goldenrod"),
    ("#6d643f", "Dark Stone"),
    ("#948c6b", "Stone"),
    ("#cdc59e", "Light Stone"),
    ("#c5ad31", "Goldenrod"),
    ("#f9dd3b", "Yellow"),
    ("#e8d45f", "Light Goldenrod"),
    ("#fffabc", "Light Yellow"),
    ("#4a6b3a", "Dark Olive"),
    ("#87ff5e", "Light Green"),
    ("#5a944a", "Olive"),
    ("#84c573", "Light Olive"),
    ("#13e67b", "Green"),
    ("#0eb968", "Dark Green"),
    ("#13e1be", "Light Teal"),
    ("#0c816e", "Dark Teal"),
    ("#bbfaf2", "Light Cyan"),
    ("#10aea6", "Teal"),
    ("#60f7f2", "Cyan"),
    ("#0f799f", "Dark Cyan"),
    ("#7dc7ff", "Light Blue"),
    ("#4093e4", "Blue"),
    ("#333941", "Dark Slate"),
    ("#28509e", "Dark Blue"),
    ("#6d758d", "Slate"),
    ("#99b1fb", "Light Indigo"),
    ("#b3b9d1", "Light Slate"),
    ("#b5aef1", "Light Slate Blue"),
    ("#7a71c4", "Slate Blue"),
    ("#4a4284", "Dark Slate Blue"),
    ("#6b50f6", "Indigo"),
    ("#4d31b8", "Dark Indigo"),
    ("#e09ff9", "Light Purple"),
    ("#780c99", "Dark Purple"),
    ("#aa38b9", "Purple"),
    ("#cb007a", "Dark Pink"),
    ("#ec1f80", "Pink"),
    ("#f38da9", "Light Pink"),
    ("#600018", "Deep Red"),
    ("#a50e1e", "Dark Red"),
    ("#000000", "Black"),
    ("#3c3c3c", "Dark Gray"),
    ("#787878", "Gray"),
    ("#aaaaaa", "Medium Gray"),
    ("#d2d2d2", "Light Gray"),
    ("#ffffff", "White"),
]

# Sub-palette for bw mode (greys only)
GREY_HEXES = ["#000000", "#3c3c3c", "#787878", "#aaaaaa", "#d2d2d2", "#ffffff"]


# Tunables and Thresholds

# Photo mode tuning (kept compact but commented)
W_LOCAL_BASE = 0.18  # weight for staying near local lightness
GRAD_K = 12.0  # gradient divisor (bigger = less sensitive)
W_HUE = 0.06  # penalty for hue drift (soft)
W_CHROMA = 0.05  # penalty for chroma change
W_WASH = 0.65  # reserved (not used now; kept for clarity)
L_WASH_THRESH = 4.0  # reserved (not used now)
W_DESAT = 0.05  # reserved (not used now)
C_DESAT_THRESH = 10.0  # reserved (not used now)
BLUR_RADIUS = 2  # smoothing radius for local L
W_CONTRAST = 0.15  # favour preserving contrast near edges
CONTRAST_MARGIN = 0.8  # small buffer so we do not jitter
NEUTRAL_C_MAX = 6.0  # we consider palette chroma <= this as neutral
CHROMA_SRC_MIN = 10.0  # source chroma to start punishing neutral targets
NEUTRAL_PENALTY = 8.0  # penalty applied if saturated source goes to neutral
NEUTRAL_EST_C = 8.0  # for estimating ab bias
NEUTRAL_EST_LMIN = 20.0
NEUTRAL_EST_LMAX = 80.0
AB_BIAS_GAIN = 0.8  # fraction of neutral cloud mean ab to subtract
NEAR_NEUTRAL_C = 4.0  # near-neutral source chroma
NEAR_BLACK_L = 32.0  # treat as near black
NEAR_WHITE_L = 86.0  # treat as near white
SHADOW_L = 38.0  # "shadow zone" pivot
W_SHADOW_HUE = 0.12  # encourage cool shifts in shadows (small)
W_SHADOW_CHROMA_UP = 0.10  # allow chroma to rise in shadows (small)
EL_CLAMP = 6.0  # clamp error diffusion (lightness)
Q_L = 0.5
Q_A = 1.0
Q_B = 1.0
Q_LOC = 1.0
Q_W = 0.05  # reserved
PHOTO_TOPK = 6  # small prefilter to accelerate per-pixel search

# Pixel mode tuning
SWAP_MARGIN = 1.0  # willingness to swap holder for lower cost
W_HUE_PX = 0.35  # hue drift penalty weight
HUE_PX_SCALE = 30.0  # normaliser for hue penalty (degrees)
PIXEL_SAT_C = 20.0  # if source chroma >= this, we treat it as "saturated"
H_RING = 25.0  # soft ring around source hue; outside gets penalty
W_HUE_RING = 0.6  # penalty strength for leaving the ring
L_DARK = 45.0  # if source L is below this, do not brighten too much
L_DARK_MAX_UP = 10.0  # how much a dark colour can brighten (absolute L)
W_L_DARK = 0.30  # penalty for breaking the above
W_BRIGHT_PX = 0.12  # mild preference to stay bright rather than dim
PREF_HUE_DEG = 35.0  # "preferred" hue gate for chromatic sources
PREF_MIN_TC_ABS = 8.0  # minimum target chroma for preferred gate
PREF_MIN_TC_REL = 0.50  # relative to source chroma
PREF_ENFORCE_C = 14.0  # hard-skip non-preferred if source chroma >= this

# Neutral rebalancing (reduces grey blending in pixel mode)
NEUTRAL_SRC_C_MAX = 8.0  # treat source as neutral if chroma <= this
NEUTRAL_REASSIGN_TOL = 2.5  # cost slack allowed to move to a different grey
NEUTRAL_L_SEP = 4.0  # if neutral sources differ this much in L, avoid same grey

# Alpha threshold: convert to 0 or 255
ALPHA_THRESH = 128

# Colour Utilities


def hex_to_rgb_u8(hx: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) uint8 tuple."""
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def rgb_to_hex(rgb: Iterable[int]) -> str:
    """Convert (r, g, b) to '#rrggbb'."""
    r, g, b = list(rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """Vectorised sRGB to linear RGB."""
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def srgb_to_lab_batch(rgb_u8: np.ndarray) -> np.ndarray:
    """Vectorised sRGB (uint8) -> Lab (float32). Assumes D65/2deg."""
    orig_shape = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    rl = _srgb_to_linear(flat[:, 0])
    gl = _srgb_to_linear(flat[:, 1])
    bl = _srgb_to_linear(flat[:, 2])
    # Linear RGB to XYZ (D65)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    # Normalise by reference white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    # f(t) piecewise for Lab
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    lab = np.stack([L, a, b], axis=1).astype(np.float32)
    return lab.reshape(orig_shape)


def lab_to_lch_batch(lab: np.ndarray) -> np.ndarray:
    """Vectorised Lab -> LCH (float32). Hue in [0, 360)."""
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


def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Scalar CIEDE2000 between two Lab colours (debug/spot use)."""
    L1, a1, b1 = lab1.tolist()
    L2, a2, b2 = lab2.tolist()
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1 - math.sqrt((Cm**7) / (Cm**7 + 25**7)))
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
    Rc = 2.0 * math.sqrt((Cpm**7.0) / (Cpm**7.0 + 25.0**7.0))
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
    """Shortest circular distance between two hues (degrees)."""
    d = abs(a - b)
    return d if d <= 180.0 else 360.0 - d


# Image IO


def binarise_alpha(alpha: np.ndarray, thresh: int = ALPHA_THRESH) -> np.ndarray:
    """Convert alpha to hard 0/255 by threshold."""
    return np.where(alpha.astype(np.int16) >= int(thresh), 255, 0).astype(np.uint8)


def load_image_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image as RGBA numpy arrays (rgb uint8, alpha uint8 binarised)."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = binarise_alpha(arr[..., 3])
    return rgb, alpha


def save_image_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> Path:
    """Save (rgb, alpha) as a PNG. Forces .png extension."""
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = binarise_alpha(alpha)
    Image.fromarray(out).save(path)
    return path


def resize_rgba_height(
    rgb: np.ndarray, alpha: np.ndarray, dst_h: int, resample: Image.Resampling
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downscale by a requested height. Never upscale.
    We keep alpha as sharp 0/255 after resizing.
    """
    H0, W0, _ = rgb.shape
    if dst_h is None or dst_h <= 0 or dst_h >= H0:
        return rgb, binarise_alpha(alpha)
    dst_w = int(round(W0 * (dst_h / float(H0))))
    rgba = np.zeros((H0, W0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    im = Image.fromarray(rgba)
    im2 = im.resize((dst_w, dst_h), resample=resample)
    arr = np.array(im2, dtype=np.uint8)
    rgb2 = arr[..., :3]
    a2 = binarise_alpha(arr[..., 3])
    return rgb2, a2


# Data Structures for Colours


@dataclass(frozen=True)
class PaletteItem:
    """A single palette entry with cached Lab/LCH."""

    rgb: Tuple[int, int, int]
    name: str
    lab: np.ndarray
    lch: np.ndarray


@dataclass(frozen=True)
class SourceItem:
    """A unique colour from the source image with count and Lab/LCH."""

    rgb: Tuple[int, int, int]
    count: int
    lab: np.ndarray
    lch: np.ndarray


# Build Palette and Source Lists


def build_palette(
    hex_name_pairs: List[Tuple[str, str]] = PALETTE,
) -> Tuple[List[PaletteItem], Dict[Tuple[int, int, int], str], np.ndarray, np.ndarray]:
    """
    Build palette items + name lookup + Lab/LCH matrices.
    Returns:
      - list of PaletteItem
      - dict from (r,g,b) -> human name
      - pal_lab: (N,3) float32
      - pal_lch: (N,3) float32
    """
    rgbs = np.array([hex_to_rgb_u8(hx) for hx, _ in hex_name_pairs], dtype=np.uint8)
    labs = srgb_to_lab_batch(rgbs).reshape(-1, 3)
    lchs = lab_to_lch_batch(labs).reshape(-1, 3)
    items: List[PaletteItem] = []
    name_of: Dict[Tuple[int, int, int], str] = {}
    for i, (_hx, name) in enumerate(hex_name_pairs):
        rgb_tuple = (int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]))
        items.append(
            PaletteItem(
                rgb=rgb_tuple, name=name, lab=labs[i].copy(), lch=lchs[i].copy()
            )
        )
        name_of[rgb_tuple] = name
    return items, name_of, labs, lchs


def unique_colors_and_counts(
    rgb: np.ndarray, alpha: np.ndarray
) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Collect unique visible RGB colours with counts, sorted by count desc.
    Uses a packed dtype trick for speed.
    """
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


def build_sources(
    rgb: np.ndarray, alpha: np.ndarray
) -> Tuple[List[SourceItem], np.ndarray, np.ndarray]:
    """
    Turn unique colour list into SourceItem objects and stacked Lab/LCH arrays.
    """
    items = unique_colors_and_counts(rgb, alpha)
    if not items:
        return (
            [],
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    src_rgb = np.array([c for c, _ in items], dtype=np.uint8)
    src_lab = srgb_to_lab_batch(src_rgb).reshape(-1, 3)
    src_lch = lab_to_lch_batch(src_lab).reshape(-1, 3)
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


# Pixel Mode: Candidates


def _candidate_row_for_source(
    i: int,
    s_lab_row: np.ndarray,
    s_lch_row: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
) -> Tuple[int, List[Tuple[float, int]]]:
    """
    Build a sorted candidate list (cost, palette_index) for one source colour.
    Cost mixes CIEDE2000 with gentle constraints to keep hue/chroma/lightness sane.
    """
    sL = float(s_lch_row[0])
    sC = float(s_lch_row[1])
    sh = float(s_lch_row[2])

    rows: List[Tuple[float, int]] = []
    hue_weight = min(1.0, max(0.0, sC / 25.0))  # only care about hue if chromatic

    for j in range(pal_lab.shape[0]):
        # Base distance in Lab (perceptual-ish)
        de = ciede2000_pair(s_lab_row, pal_lab[j])

        # Add a soft hue penalty so bright blues do not drift to cyan/purple
        dh = hue_diff_deg(sh, float(pal_lch[j, 2]))
        de += W_HUE_PX * hue_weight * (dh / HUE_PX_SCALE)

        # If the source is clearly saturated, strongly discourage big hue changes
        if sC >= PIXEL_SAT_C and dh > H_RING:
            de += W_HUE_RING * ((dh - H_RING) / H_RING)

        # Very dark source should not jump way up in lightness
        tL = float(pal_lch[j, 0])
        if sL < L_DARK and tL > sL + L_DARK_MAX_UP:
            de += W_L_DARK * (tL - (sL + L_DARK_MAX_UP))

        # Prefer to stay same or brighter; getting darker is mildly penalised
        if tL < sL:
            de += W_BRIGHT_PX * (sL - tL)

        # Avoid dumping saturated sources into neutral greys
        tC = float(pal_lch[j, 1])
        if sC >= PIXEL_SAT_C and tC <= NEUTRAL_C_MAX:
            de += NEUTRAL_PENALTY

        rows.append((float(de), j))

    rows.sort(key=lambda t: t[0])
    return i, rows


def build_candidates(
    src_lab: np.ndarray,
    src_lch: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    sources: List[SourceItem],
    workers: int,
) -> Tuple[List[List[Tuple[float, int]]], Dict[Tuple[int, int], float]]:
    """
    Build candidate lists for all sources.
    If we have at least ~32 sources and >1 CPU, use a process pool.
    Returns:
      cand[i] -> sorted list of (cost, palette_index)
      cost_lu[(i,j)] -> cost
    """
    n_s = src_lab.shape[0]
    cand: List[List[Tuple[float, int]]] = [None] * n_s  # type: ignore
    cost_lu: Dict[Tuple[int, int], float] = {}
    if workers > 1 and n_s >= 32:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _candidate_row_for_source,
                    i,
                    src_lab[i],
                    src_lch[i],
                    pal_lab,
                    pal_lch,
                )
                for i in range(n_s)
            ]
            for fu in as_completed(futures):
                i, rows = fu.result()
                cand[i] = rows
                for cost, j in rows:
                    cost_lu[(i, j)] = cost
    else:
        for i in range(n_s):
            _, rows = _candidate_row_for_source(
                i, src_lab[i], src_lch[i], pal_lab, pal_lch
            )
            cand[i] = rows
            for cost, j in rows:
                cost_lu[(i, j)] = cost
    return cand, cost_lu


def _preferred_gate(s_lch_row: np.ndarray, t_lch_row: np.ndarray) -> bool:
    """
    Decide if a candidate is 'preferred' for a chromatic source.
    Conditions:
      - If source chroma is low, everything is fine.
      - Otherwise, hue within PREF_HUE_DEG and target has enough chroma.
    """
    sC = float(s_lch_row[1])
    if sC < 12.0:
        return True
    sh = float(s_lch_row[2])
    th = float(t_lch_row[2])
    tC = float(t_lch_row[1])
    if hue_diff_deg(sh, th) > PREF_HUE_DEG:
        return False
    min_tc = max(PREF_MIN_TC_ABS, PREF_MIN_TC_REL * sC)
    if tC < min_tc:
        return False
    return True


# Pixel Mode: Non-recursive Assignment


def assign_unique_iterative(
    sources: List[SourceItem],
    cand: List[List[Tuple[float, int]]],
    cost_lu: Dict[Tuple[int, int], float],
    n_pal: int,
    src_lch: np.ndarray,
    pal_lch: np.ndarray,
) -> Dict[int, int]:
    """
    Greedy, non-recursive assignment that avoids recursion errors.
    - Sort sources by frequency (descending).
    - Walk each source's candidate list; chromatic sources skip non-preferred.
    - If a palette slot is contested, keep the cheaper claimant and nudge the
      loser forward. A small hue advantage can also steal a slot.
    - Bound the number of passes to avoid infinite ping-ponging.
    Returns: dict mapping source_index -> palette_index
    """
    order = list(range(len(sources)))
    order.sort(key=lambda i: -sources[i].count)

    assigned: Dict[int, int] = {}
    owner_of: Dict[int, int] = {}  # palette_index -> source_index
    next_pos: Dict[int, int] = {i: 0 for i in range(len(sources))}
    max_loops = n_pal * 8 + len(sources) * 3

    def first_preferred_index(i: int) -> int:
        """Find the first candidate that passes the preferred gate (within 12)."""
        limit = min(12, len(cand[i]))
        for pos in range(limit):
            _cost, pj = cand[i][pos]
            if _preferred_gate(src_lch[i], pal_lch[pj]):
                return pos
        return 0

    # Initial seeding: chromatic sources start at their first "preferred" slot
    for i in order:
        sC = float(src_lch[i, 1])
        next_pos[i] = first_preferred_index(i) if sC >= PREF_ENFORCE_C else 0

    loops = 0
    queue = list(order)
    while queue and loops < max_loops:
        loops += 1
        i = queue.pop(0)
        pos = next_pos[i]
        tried = 0
        while pos < len(cand[i]) and tried < (n_pal + 4):
            tried += 1
            cost_i, pj = cand[i][pos]
            # Hard skip non-preferred if source is sufficiently chromatic
            if float(src_lch[i, 1]) >= PREF_ENFORCE_C and not _preferred_gate(
                src_lch[i], pal_lch[pj]
            ):
                pos += 1
                continue
            holder = owner_of.get(pj)
            if holder is None:
                owner_of[pj] = i
                assigned[i] = pj
                next_pos[i] = pos
                break
            if holder == i:
                assigned[i] = pj
                next_pos[i] = pos
                break
            # Compare costs
            c_holder = cost_lu[(holder, pj)]
            if cost_i + SWAP_MARGIN < c_holder:
                owner_of[pj] = i
                assigned[i] = pj
                next_pos[i] = pos
                # Previous holder nudged forward
                next_pos[holder] = next_pos.get(holder, 0) + 1
                queue.append(holder)
                break
            # Small hue advantage rule: if costs are close, let the nearer-in-hue take it
            th = float(pal_lch[pj, 2])
            d_i = hue_diff_deg(float(src_lch[i, 2]), th)
            d_h = hue_diff_deg(float(src_lch[holder, 2]), th)
            if cost_i <= c_holder + 2.0 and d_i + 6.0 < d_h:
                owner_of[pj] = i
                assigned[i] = pj
                next_pos[i] = pos
                next_pos[holder] = next_pos.get(holder, 0) + 1
                queue.append(holder)
                break
            pos += 1
        else:
            # Could not find a free/stealable candidate; just pick best
            if len(cand[i]) > 0:
                assigned[i] = cand[i][0][1]
                next_pos[i] = 0
    return assigned


# Pixel Mode: Neutral Grey Rebalancing


def rebalance_neutral_greys(
    sources: List[SourceItem],
    assigned: Dict[int, int],
    cost_lu: Dict[Tuple[int, int], float],
    palette_items: List[PaletteItem],
    pal_lch: np.ndarray,
) -> None:
    """
    Reduce the "grey blending" issue where two different near-neutral sources
    collapse to the same grey even if their lightness is quite different.

    Strategy:
    - Identify near-neutral sources by chroma (<= NEUTRAL_SRC_C_MAX).
    - Walk them in increasing L. If two land on the same grey but their L
      differs by >= NEUTRAL_L_SEP, attempt to move the latter to a more
      appropriate grey (closer L) if the cost is within NEUTRAL_REASSIGN_TOL.
    - We only consider greys in GREY_HEXES.
    """
    grey_indices = np.where(pal_lch[:, 1] <= NEUTRAL_C_MAX)[0].tolist()
    if not grey_indices:
        # Fallback: at least keep darkest and lightest as anchors
        lo = int(np.argmin(pal_lch[:, 0]))
        hi = int(np.argmax(pal_lch[:, 0]))
        grey_indices = list(dict.fromkeys([lo, hi]))

    neutral_src_idx = [
        i for i, s in enumerate(sources) if float(s.lch[1]) <= NEUTRAL_SRC_C_MAX
    ]
    if len(neutral_src_idx) <= 1:
        return

    neutral_src_idx.sort(key=lambda i: float(sources[i].lch[0]))
    used_by: Dict[int, int] = {}  # grey palette index -> first claimant source index

    for i in neutral_src_idx:
        pj = assigned.get(i)
        if pj is None or pj not in grey_indices:
            continue
        sL = float(sources[i].lch[0])
        # First claimant keeps the grey
        if pj not in used_by:
            used_by[pj] = i
            continue

        # Second claimant: if lightness differs enough, try another grey
        prev_i = used_by[pj]
        prevL = float(sources[prev_i].lch[0])
        if abs(sL - prevL) < NEUTRAL_L_SEP:
            continue  # L too close; sharing is fine

        # Consider other greys ordered by closeness in L
        choices = [(abs(float(pal_lch[g, 0]) - sL), g) for g in grey_indices if g != pj]
        choices.sort(key=lambda t: t[0])

        base_cost = cost_lu.get(
            (i, pj), ciede2000_pair(sources[i].lab, palette_items[pj].lab)
        )
        for _dl, g in choices:
            alt_cost = cost_lu.get(
                (i, g), ciede2000_pair(sources[i].lab, palette_items[g].lab)
            )
            if alt_cost <= base_cost + NEUTRAL_REASSIGN_TOL and g not in used_by:
                assigned[i] = g
                used_by[g] = i
                break
        # If nothing acceptable, leave as is


# Pixel Mode: Apply Mapping and Debug


def apply_mapping(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    mapping: Dict[Tuple[int, int, int], Tuple[int, int, int]],
) -> np.ndarray:
    """Apply an exact RGB mapping only to visible pixels."""
    out = img_rgb.copy()
    h, w, _ = out.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            key: Tuple[int, int, int] = (
                int(out[y, x, 0]),
                int(out[y, x, 1]),
                int(out[y, x, 2]),
            )
            rep = mapping.get(key)
            if rep is not None:
                out[y, x, 0] = np.uint8(rep[0])
                out[y, x, 1] = np.uint8(rep[1])
                out[y, x, 2] = np.uint8(rep[2])
    return out


def run_pixel(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette_items: List[PaletteItem],
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    debug: bool,
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], str]]:
    """
    Main pipeline for pixel mode:
      1) Build unique source list and Lab/LCH
      2) Build candidate lists (optionally multi-process)
      3) Greedy, non-recursive assignment with collision handling
      4) Neutral grey rebalancing pass
      5) Apply mapping
      6) Print verbose debug if requested
    """
    t0 = time.perf_counter()
    sources, src_lab, src_lch = build_sources(img_rgb, alpha)
    cpu = os.cpu_count() or 1
    workers = max(1, min(32, cpu - 1))

    if len(sources) == 0:
        out_rgb = img_rgb
        if debug:
            print("[debug] pixel  empty (no visible pixels)")
        return out_rgb, {}

    cand, cost_lu = build_candidates(
        src_lab, src_lch, pal_lab, pal_lch, sources, workers
    )
    assigned = assign_unique_iterative(
        sources, cand, cost_lu, len(palette_items), src_lch, pal_lch
    )

    # Neutral grey rebalancing to avoid multiple different greys collapsing
    rebalance_neutral_greys(sources, assigned, cost_lu, palette_items, pal_lch)

    # Build RGB mapping and apply
    mapping: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    for i, s in enumerate(sources):
        # Fallback to best candidate if somehow unassigned
        pj = assigned.get(i, cand[i][0][1]) if len(cand[i]) else assigned.get(i, None)
        if pj is None:
            continue
        mapping[s.rgb] = palette_items[pj].rgb

    out_rgb = apply_mapping(img_rgb, alpha, mapping)

    t1 = time.perf_counter()
    if debug:
        uniqs = unique_colors_and_counts(img_rgb, alpha)
        total = max(1, sum(n for _c, n in uniqs))
        top16 = sum(n for _c, n in uniqs[: min(16, len(uniqs))])
        top16_share = top16 / total
        print(
            f"[debug] pixel  size_in={img_rgb.shape[1]}x{img_rgb.shape[0]}  size_eff={img_rgb.shape[1]}x{img_rgb.shape[0]}  workers={workers}  time={t1 - t0:.3f}s"
        )
        print(f"[debug] uniques_visible={len(uniqs)}  top16_share={top16_share:.3f}")
        print("[debug] per-unique mapping:")
        for i, s in enumerate(sources):
            pj = assigned.get(i, cand[i][0][1]) if len(cand[i]) else assigned.get(i, -1)
            if pj is None or pj < 0:
                continue
            de = ciede2000_pair(s.lab, pal_lab[pj])
            sL, sC, sh = s.lch.tolist()
            tL, tC, th = pal_lch[pj].tolist()
            d_h = hue_diff_deg(float(sh), float(th))
            print(
                f"  src {rgb_to_hex(s.rgb):>7}  count={s.count:6d} -> {rgb_to_hex(palette_items[pj].rgb):>7}  "
                f"[dE={de:5.2f}, dL={tL-sL:+.3f}, dC={tC-sC:+.3f}, dh={d_h:.1f} deg]"
            )
    return out_rgb, {p.rgb: p.name for p in palette_items}


# Photo Mode Helpers


def compute_L_image(rgb: np.ndarray) -> np.ndarray:
    """Return Lab L channel for an RGB image (float32)."""
    arr = rgb.astype(np.float32) / 255.0
    arr_lin = _srgb_to_linear(arr)
    Y = (
        arr_lin[..., 0] * 0.2126729
        + arr_lin[..., 1] * 0.7151522
        + arr_lin[..., 2] * 0.0721750
    )
    fy = np.where(Y > 0.008856, np.cbrt(Y), 7.787 * Y + 16.0 / 116.0)
    L = 116.0 * fy - 16.0
    return L.astype(np.float32)


def box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast box blur using summed area table. Radius 0 returns the original."""
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


def grad_mag(L: np.ndarray) -> np.ndarray:
    """Simple Manhattan gradient magnitude; blurred a little to reduce noise."""
    H, W = L.shape
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    g = gx + gy
    return box_blur_2d(g, 1)


def estimate_ab_bias(lab_img: np.ndarray, alpha: np.ndarray) -> Tuple[float, float]:
    """
    Estimate a small a/b bias from near-neutral mid-tones so we can subtract
    it before palette matching. Helps avoid a mild colour cast in neutrals.
    """
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


def prefilter_topk(s_lab: np.ndarray, pal_lab_mat: np.ndarray, k: int) -> np.ndarray:
    """Quick top-k by squared Euclidean in Lab for pre-filtering candidates."""
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
    t_lch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (cost_component, target_chroma) for 'photo' candidate scoring.
    Encourages: preserve local lightness pattern, modest hue/chroma change,
    slightly cooler shadows, and not dumping saturated colours into neutrals.
    """
    sL = s_lch[0]
    sC = s_lch[1]
    sh = s_lch[2]
    tL = t_lch[:, 0]
    tC = t_lch[:, 1]
    th = t_lch[:, 2]
    cost = np.zeros_like(tL, dtype=np.float32)

    # Stay near the local lightness (relative to the local background)
    cost += W_LOCAL_BASE * w_local_eff * np.abs((tL - L_local) - (sL - L_local))

    # Preserve contrast a bit (if target is strictly "inside" the local delta)
    d_s = abs(sL - L_local)
    d_t = np.abs(tL - L_local)
    cost += np.where(
        d_t + CONTRAST_MARGIN < d_s, W_CONTRAST * (d_s - d_t - CONTRAST_MARGIN), 0.0
    )

    # Hue and chroma changes should be gentle unless justified
    hue_weight = min(1.0, sC / 40.0)
    cost += (
        W_HUE
        * hue_weight
        * (np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0)
    )
    cost += W_CHROMA * np.abs(tC - sC)

    # In deeper shadows we allow a tiny nudge towards cooler / more chroma
    if sL < 40.0:
        shadow_factor = (SHADOW_L - sL) / SHADOW_L
        hue_dist = np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0
        chroma_up = np.maximum(0.0, tC - sC)
        cost += shadow_factor * (
            W_SHADOW_HUE * hue_dist + W_SHADOW_CHROMA_UP * chroma_up
        )

    # If the source is colourful, avoid collapsing into neutral greys
    if sC >= CHROMA_SRC_MIN:
        cost += np.where(
            tC <= NEUTRAL_C_MAX, NEUTRAL_PENALTY + 0.02 * np.maximum(0.0, sC - tC), 0.0
        )

    return cost, tC


def ciede2000_vec(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 for a single sample vs many candidates (photo mode)."""
    L1 = s[0]
    a1 = s[1]
    b1 = s[2]
    L2 = cand[:, 0]
    a2 = cand[:, 1]
    b2 = cand[:, 2]
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
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    Cpm = 0.5 * (C1p + C2p)
    h1p = (np.degrees(np.atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.atan2(b2, a2p)) + 360.0) % 360.0
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
        - 0.17 * np.cos(np.radians(hpm - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hpm))
        + 0.32 * np.cos(np.radians(3.0 * hpm + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hpm - 63.0))
    )
    d_ro = 30.0 * np.exp(-(((hpm - 275.0) / 25.0) ** 2.0))
    Rc = 2.0 * np.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1.0 + (0.015 * (Lpm - 50.0) ** 2) / np.sqrt(20.0 + (Lpm - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cpm
    Sh = 1.0 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2.0 * d_ro)) * Rc
    dCp = C2p - C1p
    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def nearest_palette_idx_photo_lab_prefilter_vec(
    s_lab: np.ndarray,
    s_lch: np.ndarray,
    L_local: float,
    w_local_eff: float,
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
    near_extreme_neutral: bool,
    k: int,
) -> int:
    """Choose the best palette index for a pixel (photo mode), using a top-k prefilter."""
    idxs = prefilter_topk(s_lab, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]
    dE = ciede2000_vec(s_lab, cand_lab)
    comp_base, tC = photo_cost_components(s_lab, s_lch, L_local, w_local_eff, cand_lch)
    cost = dE + comp_base
    if near_extreme_neutral:
        # Near black or white and near-neutral chroma: prefer neutral targets
        cost = np.where(tC > NEUTRAL_C_MAX, cost + 1e6, cost)
    return int(idxs[int(np.argmin(cost))])


# MODES


def dither_photo(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
) -> np.ndarray:
    """
    Lightweight serpentine error-diffusion that pushes only Lab L.
    After dithering, we enforce the palette strictly.
    """
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    lab_img = srgb_to_lab_batch(img_rgb).reshape(H, W, 3)
    L_src = lab_img[..., 0]
    L_loc = box_blur_2d(L_src, BLUR_RADIUS)
    G = grad_mag(L_src)
    ab_bias = estimate_ab_bias(lab_img, alpha)

    errL = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        serp_left = y % 2 == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)
        # Floyd–Steinberg weights (mirrored on odd rows)
        neighbours: Tuple[Tuple[int, int, float], ...] = (
            ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
            if serp_left
            else ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        )
        for x in xs:
            if alpha[y, x] == 0:
                continue

            # The sharper the local edge, the less we push towards L_local
            wloc = max(0.2, 1.0 - (G[y, x] / GRAD_K))

            # Apply accumulated L error, remove neutral a/b bias
            sL = float(lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(lab_img[y, x, 1]) - ab_bias[0]
            sb = float(lab_img[y, x, 2]) - ab_bias[1]
            s_lab_adj = np.array([sL, sa, sb], dtype=np.float32)

            # Build LCH for cost components
            C = float(math.hypot(s_lab_adj[1], s_lab_adj[2]))
            h = (math.degrees(math.atan2(s_lab_adj[2], s_lab_adj[1])) + 360.0) % 360.0
            s_lch_adj = np.array([s_lab_adj[0], C, h], dtype=np.float32)

            near_extreme_neutral = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            j = nearest_palette_idx_photo_lab_prefilter_vec(
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

            # Diffuse only the L error
            tgt_L = float(pal_lch_mat[j, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            for dx, dy, w in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

    return out


def select_bw_indices(pal_lch_mat: np.ndarray, chroma_max: float = 6.0) -> np.ndarray:
    """
    Pick all near-neutral palette entries (greys) purely by chroma.
    chroma_max is the LCH C threshold below which we treat a colour as grey.
    Falls back to the darkest and lightest palette colours if none qualify.
    """
    grey_idx = np.where(pal_lch_mat[:, 1] <= float(chroma_max))[0]
    if grey_idx.size == 0:
        # Fallback: ensure we have at least two anchors (dark + light)
        lo = int(np.argmin(pal_lch_mat[:, 0]))  # lowest L*
        hi = int(np.argmax(pal_lch_mat[:, 0]))  # highest L*
        grey_idx = np.unique(np.array([lo, hi], dtype=int))
    return grey_idx


def dither_bw(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: list,
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
) -> np.ndarray:
    """
    Black/white (greys-allowed) mode:
    - Detect greys in the palette by low chroma (no predefined list).
    - Quantise/dither using L* only, but output exact palette RGBs.
    Uses Floyd–Steinberg error diffusion on L* to preserve detail.
    """
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    # Work in Lab once for speed; we only need L* per pixel here.
    lab_img = srgb_to_lab_batch(img_rgb).reshape(H, W, 3)
    L_src = lab_img[..., 0].astype(np.float32)

    # Build grey candidate set from the palette
    grey_idx = select_bw_indices(pal_lch_mat, chroma_max=6.0)
    L_grey = pal_lch_mat[grey_idx, 0].astype(np.float32)
    grey_rgb = np.array([palette[i].rgb for i in grey_idx], dtype=np.uint8)

    # Error buffer for L*
    errL = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        serp_left = y % 2 == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)
        nbrs = (
            ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
            if serp_left
            else ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        )
        for x in xs:
            if alpha[y, x] == 0:
                continue

            # Current luminance after propagated error
            L_here = L_src[y, x] + errL[y, x]

            # Choose grey with nearest L*
            j = int(np.argmin(np.abs(L_grey - L_here)))
            out[y, x] = grey_rgb[j]

            # Diffuse the L* error to neighbours (favour NZ spelling for neighbour)
            eL = L_here - L_grey[j]
            for dx, dy, w in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

    return out


# Palette Enforcement Utilities


def _palette_set(palette: List[PaletteItem]) -> set[Tuple[int, int, int]]:
    """Return a set of (r,g,b) tuples from palette items."""
    return {p.rgb for p in palette}


def is_palette_only(
    img_rgb: np.ndarray, alpha: np.ndarray, pal_set: set[Tuple[int, int, int]]
) -> bool:
    """Check if every visible pixel is already a palette colour."""
    h, w, _ = img_rgb.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            t = (int(img_rgb[y, x, 0]), int(img_rgb[y, x, 1]), int(img_rgb[y, x, 2]))
            if t not in pal_set:
                return False
    return True


def lock_to_palette_by_uniques(
    out_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
    pal_lab_mat: np.ndarray,
) -> np.ndarray:
    """
    Fast enforcement: build a mapping for all unique off-palette colours
    by nearest in Lab, then apply it to the whole image.
    """
    H, W, _ = out_rgb.shape
    pal_set = _palette_set(palette)
    mask = alpha != 0
    samples = out_rgb[mask]
    if samples.size == 0:
        return out_rgb
    # Unique scan using a packed dtype for speed
    dt = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    flat = samples.view(dt).reshape(-1)
    uniq = np.unique(flat)
    rs = uniq["r"].astype(int)
    gs = uniq["g"].astype(int)
    bs = uniq["b"].astype(int)
    off: List[Tuple[int, int, int]] = []
    for i in range(len(uniq)):
        t = (int(rs[i]), int(gs[i]), int(bs[i]))
        if t not in pal_set:
            off.append(t)
    if not off:
        return out_rgb

    off_np = np.array(off, dtype=np.uint8)
    off_lab = srgb_to_lab_batch(off_np).reshape(-1, 3)
    mapping: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    pal_rgb_arr: np.ndarray = np.array([p.rgb for p in palette], dtype=np.uint8)

    for i in range(off_lab.shape[0]):
        s_lab = off_lab[i]
        diff = pal_lab_mat - s_lab
        de2 = (diff * diff).sum(axis=1)
        best = int(np.argmin(de2))
        r = int(pal_rgb_arr[best, 0])
        g = int(pal_rgb_arr[best, 1])
        b = int(pal_rgb_arr[best, 2])
        key: Tuple[int, int, int] = (
            int(off_np[i, 0]),
            int(off_np[i, 1]),
            int(off_np[i, 2]),
        )
        mapping[key] = (r, g, b)

    out = out_rgb.copy()
    for y in range(H):
        for x in range(W):
            if alpha[y, x] == 0:
                continue
            t = (int(out[y, x, 0]), int(out[y, x, 1]), int(out[y, x, 2]))
            rep = mapping.get(t)
            if rep is not None:
                out[y, x, 0], out[y, x, 1], out[y, x, 2] = rep
    return out


def lock_to_palette_per_pixel(
    out_rgb: np.ndarray, alpha: np.ndarray, palette: List[PaletteItem]
) -> np.ndarray:
    """
    Slow but bullet-proof fallback: map each visible pixel to nearest palette
    colour by squared distance in RGB.
    """
    H, W, _ = out_rgb.shape
    out = out_rgb.copy()
    mask = alpha != 0
    if not mask.any():
        return out
    pal_rgb = np.array([p.rgb for p in palette], dtype=np.int16)
    coords = np.argwhere(mask)
    chunk = 200_000
    for i in range(0, coords.shape[0], chunk):
        sl = coords[i : i + chunk]
        pts = out[sl[:, 0], sl[:, 1]].astype(np.int16)
        diff = pts[:, None, :] - pal_rgb[None, :, :]
        de2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(de2, axis=1)
        mapped = pal_rgb[idx].astype(np.uint8)
        out[sl[:, 0], sl[:, 1]] = mapped
    return out


# Reporting Helpers


def print_color_usage(
    out_rgb: np.ndarray, alpha: np.ndarray, name_of: Dict[Tuple[int, int, int], str]
) -> None:
    """Print a neat colour usage report sorted by frequency."""
    h, w, _ = out_rgb.shape
    cnt: Counter[Tuple[int, int, int]] = Counter()
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            key: Tuple[int, int, int] = (
                int(out_rgb[y, x, 0]),
                int(out_rgb[y, x, 1]),
                int(out_rgb[y, x, 2]),
            )
            cnt[key] += 1
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    print("Colours used:")
    for (r, g, b), n in items:
        hx = rgb_to_hex((r, g, b))
        nm = name_of.get((r, g, b), "")
        label = f"  {hx}  {nm}: {n}" if nm else f"  {hx}: {n}"
        print(label)


def count_off_palette_pixels(
    img_rgb: np.ndarray, alpha: np.ndarray, palette: List[PaletteItem]
) -> int:
    """Count visible pixels not in the palette (sanity check for photo mode)."""
    pal_set = _palette_set(palette)
    h, w, _ = img_rgb.shape
    c = 0
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            t = (int(img_rgb[y, x, 0]), int(img_rgb[y, x, 1]), int(img_rgb[y, x, 2]))
            if t not in pal_set:
                c += 1
    return c


# Mode Selection (auto)


def decide_auto_mode(img_rgb: np.ndarray, alpha: np.ndarray) -> str:
    """
    Simple heuristic:
      - If unique colours are modest and the top16 cover >= 80% of visible
        pixels, assume pixel-art and choose 'pixel'.
      - Otherwise choose 'photo'.
    """
    items = unique_colors_and_counts(img_rgb, alpha)
    total = sum(n for _c, n in items)
    n_uniques = len(items)
    topk = sum(n for _c, n in items[: min(16, n_uniques)])
    share = (topk / total) if total > 0 else 1.0
    return "pixel" if (n_uniques <= 512 or share >= 0.80) else "photo"


# BW (Greyscale) Mode


def run_bw(
    img_rgb: np.ndarray, alpha: np.ndarray, debug: bool
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], str]]:
    """
    Black & white mode (really: grey ramp only).
    Restrict the palette to GREY_HEXES and lock the image to it.
    """
    grey_pairs = [(hx, hx) for hx in GREY_HEXES]  # reuse hex as name for tidy reporting
    pal_items, name_of, pal_lab, _pal_lch = build_palette(grey_pairs)
    pal_lab_mat = pal_lab.astype(np.float32)

    # Quick unique-based lock, then a per-pixel sweep only if required
    out_rgb = lock_to_palette_by_uniques(img_rgb, alpha, pal_items, pal_lab_mat)
    if not is_palette_only(out_rgb, alpha, _palette_set(pal_items)):
        out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, pal_items)
    return out_rgb, name_of


# Main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Palette remapper. Args: --mode {auto,pixel,photo,bw}, --height N, --debug"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "--output", "-o", help="Output path (.png forced). Defaults to *_wplace.png"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pixel", "photo", "bw"],
        default="auto",
        help="auto | pixel | photo | bw (greys only)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Process height. Downscale only; never upscale.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debug printing")
    args = parser.parse_args()

    t0 = time.perf_counter()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_wplace.png")
    )

    # Load input
    img_rgb, alpha = load_image_rgba(in_path)
    H0, W0, _ = img_rgb.shape
    if args.debug:
        a0_255 = int((alpha == 255).sum())
        a0_0 = int((alpha == 0).sum())
        print(f"[debug] load   {W0}x{H0}  alpha(255)={a0_255}  alpha(0)={a0_0}")

    # Build full palette once (pixel/photo use it; bw builds its own)
    palette, name_of_full, pal_lab, pal_lch = build_palette()
    pal_lab_mat = pal_lab.astype(np.float32)
    pal_lch_mat = pal_lch.astype(np.float32)
    pal_set_full = _palette_set(palette)

    # Decide mode if auto
    mode_eff = args.mode if args.mode != "auto" else decide_auto_mode(img_rgb, alpha)

    # Resize policy
    resample = (
        Image.Resampling.BOX
        if mode_eff in ("pixel", "bw")
        else Image.Resampling.LANCZOS
    )
    img_rgb, alpha = resize_rgba_height(img_rgb, alpha, args.height, resample)
    H1, W1, _ = img_rgb.shape
    if args.debug:
        a1_255 = int((alpha == 255).sum())
        a1_0 = int((alpha == 0).sum())
        print(f"[debug] size   {W1}x{H1}  alpha(255)={a1_255}  alpha(0)={a1_0}")
        print(f"[debug] mode   {mode_eff}")

    # Run selected mode
    if mode_eff == "photo":
        t_photo0 = time.perf_counter()
        out_rgb = dither_photo(img_rgb, alpha, palette, pal_lab_mat, pal_lch_mat)
        # Enforce palette strictly
        ok0 = is_palette_only(out_rgb, alpha, pal_set_full)
        if not ok0:
            out_rgb = lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat)
        ok1 = is_palette_only(out_rgb, alpha, pal_set_full)
        if not ok1:
            out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, palette)
        ok2 = is_palette_only(out_rgb, alpha, pal_set_full)
        t_photo1 = time.perf_counter()

        out_path = save_image_rgba(out_path, out_rgb, alpha)
        if args.debug:
            off = 0 if ok2 else count_off_palette_pixels(out_rgb, alpha, palette)
            print(
                f"[debug] photo  time={t_photo1 - t_photo0:.3f}s  palette_ok={ok2}  off_pixels={off}"
            )
        print("Mode: photo")
        print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
        print_color_usage(out_rgb, alpha, name_of_full)
        if args.debug:
            print(f"[debug] total  {time.perf_counter() - t0:.3f}s")
        return

    if mode_eff == "bw":
        t0_bw = time.perf_counter()
        out_rgb = dither_bw(img_rgb, alpha, palette, pal_lab_mat, pal_lch_mat)
        out_path = save_image_rgba(out_path, out_rgb, alpha)
        if args.debug:
            print(
                f"[debug] bw     time={time.perf_counter() - t0_bw:.3f}s  greys_used={len(select_bw_indices(pal_lch_mat))}"
            )
        print("Mode: bw")
        print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
        print_color_usage(out_rgb, alpha, name_of_full)
        if args.debug:
            print(f"[debug] total  {time.perf_counter() - t0:.3f}s")
        return

    # pixel
    out_rgb, _name_map = run_pixel(
        img_rgb, alpha, palette, pal_lab, pal_lch, args.debug
    )
    # One more pass to ensure output is strictly within the palette
    out_rgb = lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat)
    out_path = save_image_rgba(out_path, out_rgb, alpha)

    print("Mode: pixel")
    print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
    print_color_usage(out_rgb, alpha, name_of_full)
    if args.debug:
        print(f"[debug] total  {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
