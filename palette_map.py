#!/usr/bin/env python3
"""
palette_map.py

Palette-constrained image remapper.

Primary purpose:
- Read an input image (any format Pillow can open).
- Optionally downscale by TARGET HEIGHT ONLY (never upscale).
- Hard-threshold alpha to 0/255 (fully transparent or fully opaque).
- Map all visible pixels to a fixed colour palette.
- Two rendering strategies:
    * pixel : best for pixel art / low unique-colour images.
    * photo : preserves photo-like detail while still mapping to the palette.
- Auto mode picks between the two.

CLI:
    palette_map.py INPUT [--output OUT.png]
                         [--mode {auto,pixel,photo}]
                         [--height H]
                         [--debug]

Notes:
- Only three arguments: mode, height, debug (plus optional output path).
- Scaling uses height only. If requested height >= image height, no scaling occurs.
- Alpha is binarised with a hard threshold (no partial transparency in output).
- The output is always PNG (so alpha is preserved cleanly).
- Debug prints are ASCII only (no special characters).
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

# ===================== USER PALETTE (hex, name) =====================
# Order matters only for reporting. All matching is distance-based.

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
    ("#3c3c3c", "Dark Grey"),
    ("#787878", "Grey"),
    ("#aaaaaa", "Medium Grey"),
    ("#d2d2d2", "Light Grey"),
    ("#ffffff", "White"),
]


# ===================== TUNING (photo mode) =====================
# These weights/cutoffs balance fidelity vs palette correctness.

W_LOCAL_BASE = 0.18  # Weight for keeping local contrast consistent
GRAD_K = 12.0  # Edge magnitude scale for local weighting
W_HUE = 0.06  # Weight for hue consistency
W_CHROMA = 0.05  # Weight for matching chroma (saturation)
W_WASH = 0.65  # (reserved, not used directly; kept for clarity)
L_WASH_THRESH = 4.0  # (reserved)
W_DESAT = 0.05  # (reserved)
C_DESAT_THRESH = 10.0  # (reserved)
BLUR_RADIUS = 2  # Radius for local-L blur used in contrast guidance
W_CONTRAST = 0.15  # Penalty if target flattens local contrast too much
CONTRAST_MARGIN = 0.8  # Allow small contrast differences before penalising

NEUTRAL_C_MAX = 6.0  # Chroma threshold considered neutral (greys/near-greys)
CHROMA_SRC_MIN = 10.0  # If source is more saturated than this, avoid mapping to neutral
NEUTRAL_PENALTY = 8.0  # Penalty for sending saturated colours to neutral targets
NEUTRAL_EST_C = 8.0  # For global a/b bias estimation, consider colours with C <= this
NEUTRAL_EST_LMIN = 20.0  # L range for neutral estimation
NEUTRAL_EST_LMAX = 80.0
AB_BIAS_GAIN = 0.8  # How strongly to subtract global a/b drift

NEAR_NEUTRAL_C = 4.0  # Very low chroma considered near-neutral
NEAR_BLACK_L = 32.0  # Very dark L considered "near black"
NEAR_WHITE_L = 86.0  # Very bright L considered "near white"

SHADOW_L = 38.0  # Below this, allow some hue/chroma boosting for legibility
W_SHADOW_HUE = 0.12
W_SHADOW_CHROMA_UP = 0.10

EL_CLAMP = 6.0  # Clamp L error diffusion to avoid ringing
Q_L = 0.5
Q_A = 1.0
Q_B = 1.0
Q_LOC = 1.0
Q_W = 0.05  # (reserved)
PHOTO_TOPK = 6  # Photo mode: prefilter to this many nearest candidates

ALPHA_THRESH = 128  # Hard threshold alpha; below -> 0, above/equal -> 255


# ===================== PIXEL mode stability =====================
# These guide "unique-colour to palette" assignment under conflicts.

SWAP_MARGIN = 1.0  # How much better a candidate must be to evict a holder
W_HUE_PX = 0.35  # Add hue distance to CIEDE2000 to prefer closer hues
HUE_PX_SCALE = 30.0  # Hue penalty scale (deg)
PIXEL_SAT_C = 20.0  # If source chroma >= this, avoid mapping to neutrals
H_RING = 25.0  # Larger hue mismatch ring penalty threshold
W_HUE_RING = 0.6  # Penalty for hue mismatches beyond H_RING
L_DARK = 45.0  # For dark sources, avoid jumping too bright
L_DARK_MAX_UP = 10.0  # Allow some brightening for darks
W_L_DARK = 0.30  # Penalty for brightening dark colours too much

H_REEVAL = 28.0  # If chosen hue is far, try to re-evaluate
COST_TOL_REEVAL = 6.0  # Allow some cost slack when re-evaluating

# Prefer brighter targets; avoid mapping saturated sources to near-neutrals.
W_BRIGHT_PX = 0.12
PREF_HUE_DEG = 35.0
PREF_MIN_TC_ABS = 8.0
PREF_MIN_TC_REL = 0.50
PREF_ENFORCE_C = 14.0  # If source chroma >= this, hard-skip non-preferred candidates


# ===================== Colour maths =====================


def hex_to_rgb_u8(hx: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' to 8-bit RGB tuple."""
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def rgb_to_hex(rgb: Iterable[int]) -> str:
    """Convert (r,g,b) to '#rrggbb'."""
    r, g, b = list(rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """sRGB to linear RGB."""
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def srgb_to_lab_batch(rgb_u8: np.ndarray) -> np.ndarray:
    """Vectorised sRGB -> CIE Lab (D65). Accepts (...,3) uint8, returns same shape float32."""
    orig_shape = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    rl = _srgb_to_linear(flat[:, 0])
    gl = _srgb_to_linear(flat[:, 1])
    bl = _srgb_to_linear(flat[:, 2])

    # XYZ transform (sRGB D65)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalise by white point (D65)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

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
    """Vectorised Lab -> LCh. Accepts (...,3) float32, returns same shape float32."""
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
    """CIEDE2000 colour difference between two Lab triplets (float32 arrays of shape (3,))."""
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


# ===================== IO + resizing =====================


def binarize_alpha(alpha: np.ndarray, thresh: int = ALPHA_THRESH) -> np.ndarray:
    """Hard-threshold alpha to 0 or 255."""
    return np.where(alpha.astype(np.int16) >= int(thresh), 255, 0).astype(np.uint8)


def load_image_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image, convert to RGBA, return (rgb, alpha) uint8 arrays."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = binarize_alpha(arr[..., 3])
    return rgb, alpha


def save_image_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> Path:
    """Save an RGBA PNG (force .png if needed)."""
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = binarize_alpha(alpha)
    Image.fromarray(out).save(path)
    return path


def resize_rgba_height(
    rgb: np.ndarray,
    alpha: np.ndarray,
    dst_h: int,
    resample: Image.Resampling,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downscale by height only. If dst_h is None or >= original height, return input unchanged.
    Uses a 4-channel pass, then re-binarises alpha to keep it crisp.
    """
    H0, W0, _ = rgb.shape
    if dst_h is None or dst_h <= 0 or dst_h >= H0:
        return rgb, binarize_alpha(alpha)

    dst_w = int(round(W0 * (dst_h / float(H0))))
    rgba = np.zeros((H0, W0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    im = Image.fromarray(rgba)
    im2 = im.resize((dst_w, dst_h), resample=resample)
    arr = np.array(im2, dtype=np.uint8)
    rgb2 = arr[..., :3]
    a2 = binarize_alpha(arr[..., 3])
    return rgb2, a2


# ===================== Structures =====================


@dataclass(frozen=True)
class PaletteItem:
    """A colour in the palette, with multiple representations for fast matching."""

    rgb: Tuple[int, int, int]
    name: str
    lab: np.ndarray
    lch: np.ndarray


@dataclass(frozen=True)
class SourceItem:
    """A unique source colour visible in the image, with count and Lab/LCh."""

    rgb: Tuple[int, int, int]
    count: int
    lab: np.ndarray
    lch: np.ndarray


# ===================== Build sets =====================


def build_palette() -> (
    Tuple[List[PaletteItem], Dict[Tuple[int, int, int], str], np.ndarray, np.ndarray]
):
    """Build palette items and matrices for Lab/LCh."""
    rgbs = np.array([hex_to_rgb_u8(hx) for hx, _ in PALETTE], dtype=np.uint8)
    labs = srgb_to_lab_batch(rgbs).reshape(-1, 3)
    lchs = lab_to_lch_batch(labs).reshape(-1, 3)
    items: List[PaletteItem] = []
    name_of: Dict[Tuple[int, int, int], str] = {}
    for i, (_hx, name) in enumerate(PALETTE):
        rgb_t = (int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]))
        items.append(
            PaletteItem(rgb=rgb_t, name=name, lab=labs[i].copy(), lch=lchs[i].copy())
        )
        name_of[rgb_t] = name
    return items, name_of, labs, lchs


def unique_colors_and_counts(
    rgb: np.ndarray, alpha: np.ndarray
) -> List[Tuple[Tuple[int, int, int], int]]:
    """Return visible unique colours and their counts, sorted by count desc."""
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
    """Build list of visible unique source colours and their Lab/LCh arrays."""
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


# ===================== Hue helpers =====================


def hue_diff_deg(a: float, b: float) -> float:
    """Absolute circular hue difference in degrees (0..180)."""
    d = abs(a - b)
    return d if d <= 180.0 else 360.0 - d


# ===================== Pixel mode =====================


def _candidate_row_for_source(
    src_index: int,
    src_lab_row: np.ndarray,
    src_lch_row: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
) -> Tuple[int, List[Tuple[float, int]]]:
    """
    For a single source unique colour, compute a cost for every palette colour.
    Cost is CIEDE2000 + pixel-mode regularisers (hue, darkening, neutral-avoid).
    Returns a sorted candidate list: [(cost, palette_idx), ...] asc by cost.
    """
    sL = float(src_lch_row[0])
    sC = float(src_lch_row[1])
    sh = float(src_lch_row[2])

    rows: List[Tuple[float, int]] = []
    hue_weight = min(1.0, max(0.0, sC / 25.0))  # less hue penalty if near-neutral

    for j in range(pal_lab.shape[0]):
        # Base distance
        dE = ciede2000_pair(src_lab_row, pal_lab[j])

        # Encourage closer hue if the source has chroma
        dh = hue_diff_deg(sh, float(pal_lch[j, 2]))
        dE += W_HUE_PX * hue_weight * (dh / HUE_PX_SCALE)

        # Extra penalty for large hue mismatches on saturated sources
        if sC >= PIXEL_SAT_C and dh > H_RING:
            dE += W_HUE_RING * ((dh - H_RING) / H_RING)

        # Avoid jumping too bright for dark sources
        tL = float(pal_lch[j, 0])
        if sL < L_DARK and tL > sL + L_DARK_MAX_UP:
            dE += W_L_DARK * (tL - (sL + L_DARK_MAX_UP))

        # Prefer brighter (not darker) targets a little
        if tL < sL:
            dE += W_BRIGHT_PX * (sL - tL)

        # If the source is saturated, avoid mapping to near-neutral targets
        tC = float(pal_lch[j, 1])
        if sC >= PIXEL_SAT_C and tC <= NEUTRAL_C_MAX:
            dE += NEUTRAL_PENALTY

        rows.append((float(dE), j))

    rows.sort(key=lambda t: t[0])
    return src_index, rows


def build_candidates(
    src_lab: np.ndarray,
    src_lch: np.ndarray,
    pal_lab: np.ndarray,
    pal_lch: np.ndarray,
    sources: List[SourceItem],
    workers: int,
) -> Tuple[List[List[Tuple[float, int]]], Dict[Tuple[int, int], float]]:
    """
    For every source unique colour, precompute a sorted candidate list of palette matches.
    Optionally parallelised across processes for speed on large unique sets.
    Returns:
        candidates: list of lists [(cost, palette_idx), ...] for each source index
        cost_lookup: dict keyed by (src_idx, pal_idx) -> cost
    """
    n_src = src_lab.shape[0]
    candidates: List[List[Tuple[float, int]]] = [None] * n_src  # type: ignore
    cost_lookup: Dict[Tuple[int, int], float] = {}

    if workers > 1 and n_src >= 32:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _candidate_row_for_source,
                    i,
                    src_lab[i],
                    src_lch[i],
                    pal_lab,
                    pal_lch,
                )
                for i in range(n_src)
            ]
            for fu in as_completed(futs):
                i, rows = fu.result()
                candidates[i] = rows
                for cost, j in rows:
                    cost_lookup[(i, j)] = cost
    else:
        for i in range(n_src):
            _, rows = _candidate_row_for_source(
                i, src_lab[i], src_lch[i], pal_lab, pal_lch
            )
            candidates[i] = rows
            for cost, j in rows:
                cost_lookup[(i, j)] = cost

    return candidates, cost_lookup


def _preferred_gate(src_lch_row: np.ndarray, tgt_lch_row: np.ndarray) -> bool:
    """
    Return True if the target passes preference gates for saturated sources:
    - Hue within PREF_HUE_DEG
    - Target chroma at least a fraction of source chroma
    """
    sC = float(src_lch_row[1])
    if sC < 12.0:
        return True
    sh = float(src_lch_row[2])
    th = float(tgt_lch_row[2])
    tC = float(tgt_lch_row[1])
    if hue_diff_deg(sh, th) > PREF_HUE_DEG:
        return False
    min_tc = max(PREF_MIN_TC_ABS, PREF_MIN_TC_REL * sC)
    if tC < min_tc:
        return False
    return True


def assign_unique(
    sources: List[SourceItem],
    candidates: List[List[Tuple[float, int]]],
    cost_lookup: Dict[Tuple[int, int], float],
    n_pal: int,
    src_lch: np.ndarray,
    pal_lch: np.ndarray,
) -> Dict[int, int]:
    """
    Assign each visible unique source colour to one palette colour.
    - Process sources in descending frequency.
    - Each palette colour may be used by many sources; this assigner just finds
      the best mapping for each source, resolving conflicts with a simple
      iterative "evict and retry" loop (non-recursive to avoid deep recursion).

    Returns: dict source_index -> palette_index
    """
    order = list(range(len(sources)))
    order.sort(key=lambda i: -sources[i].count)

    taken_by: Dict[int, int] = {}  # palette idx -> the source idx currently holding it
    assigned: Dict[int, int] = {}  # source idx  -> palette idx
    cursor: Dict[int, int] = {i: 0 for i in range(len(sources))}

    def first_preferred_index(i: int) -> int | None:
        rows = candidates[i]
        for pos, (_c, pj) in enumerate(rows[: min(12, len(rows))]):
            if _preferred_gate(src_lch[i], pal_lch[pj]):
                return pos
        return None

    # Simple queue of work items; if a source is evicted, requeue it.
    todo: List[int] = order[:]
    tries: Dict[int, int] = {i: 0 for i in range(len(sources))}
    MAX_TRIES = n_pal + 8  # Termination guard to prevent infinite ping-pong

    while todo:
        i = todo.pop(0)
        rows = candidates[i]
        if not rows:
            continue

        # Start at preferred if available, otherwise continue from last cursor
        pref_pos = first_preferred_index(i)
        start_pos = pref_pos if pref_pos is not None else cursor.get(i, 0)
        pos = max(0, min(start_pos, len(rows) - 1))

        src_hue = float(src_lch[i, 2])
        src_chroma = float(src_lch[i, 1])

        placed = False
        while pos < len(rows) and tries[i] < MAX_TRIES:
            tries[i] += 1
            cost_i, pal_idx = rows[pos]

            # If source is fairly saturated, skip non-preferred targets
            if src_chroma >= PREF_ENFORCE_C and not _preferred_gate(
                src_lch[i], pal_lch[pal_idx]
            ):
                pos += 1
                continue

            holder = taken_by.get(pal_idx)
            if holder is None or holder == i:
                taken_by[pal_idx] = i
                assigned[i] = pal_idx
                cursor[i] = pos
                placed = True
                break

            # Compare with existing holder
            current_cost = cost_lookup[(holder, pal_idx)]
            better = (cost_i + SWAP_MARGIN) < current_cost

            tgt_hue = float(pal_lch[pal_idx, 2])
            d_i = hue_diff_deg(src_hue, tgt_hue)
            d_h = hue_diff_deg(float(src_lch[holder, 2]), tgt_hue)
            tie_hue = (
                (not better) and (cost_i <= current_cost + 2.0) and (d_i + 6.0 < d_h)
            )

            if better or tie_hue:
                # Evict holder, give palette colour to i
                taken_by[pal_idx] = i
                assigned[i] = pal_idx
                cursor[i] = pos

                # Advance holder past this pal_idx and requeue if it still has options
                hpos = next(
                    (
                        k
                        for k, (_c2, jx) in enumerate(candidates[holder])
                        if jx == pal_idx
                    ),
                    cursor.get(holder, 0),
                )
                cursor[holder] = hpos + 1
                if (
                    cursor[holder] < len(candidates[holder])
                    and tries[holder] < MAX_TRIES
                ):
                    todo.append(holder)
                placed = True
                break

            pos += 1

        if not placed:
            # Fallback to very first candidate to guarantee assignment
            assigned[i] = rows[0][1]
            cursor[i] = 0

    # Optional hue re-evaluation: if hue is far, try a close-hue candidate if cost is close
    for i, pal_idx in list(assigned.items()):
        src_hue = float(src_lch[i, 2])
        tgt_hue = float(pal_lch[pal_idx, 2])
        dh = hue_diff_deg(src_hue, tgt_hue)
        base_cost = cost_lookup[(i, pal_idx)]
        if dh <= H_REEVAL:
            continue
        for cand_cost, cand_idx in candidates[i]:
            cand_hue = float(pal_lch[cand_idx, 2])
            dhk = hue_diff_deg(src_hue, cand_hue)
            if dhk <= H_REEVAL and cand_cost <= base_cost + COST_TOL_REEVAL:
                src_L = float(src_lch[i, 0])
                tgt_L = float(pal_lch[cand_idx, 0])
                if src_L >= L_DARK or tgt_L <= src_L + L_DARK_MAX_UP:
                    assigned[i] = cand_idx
                    break

    return assigned


def apply_mapping(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    mapping: Dict[Tuple[int, int, int], Tuple[int, int, int]],
) -> np.ndarray:
    """Replace colours per mapping dict for visible pixels, leaving others and alpha as-is."""
    out = img_rgb.copy()
    h, w, _ = out.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            key = (int(out[y, x, 0]), int(out[y, x, 1]), int(out[y, x, 2]))
            replacement = mapping.get(key)
            target = replacement if replacement is not None else key
            out[y, x, 0] = np.uint8(target[0])
            out[y, x, 1] = np.uint8(target[1])
            out[y, x, 2] = np.uint8(target[2])
    return out


# ===================== Photo helpers =====================


def compute_L_image(rgb: np.ndarray) -> np.ndarray:
    """Compute CIE L (per pixel) from sRGB image."""
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
    """Fast box blur using summed-area table (integral image)."""
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


def grad_mag(L_img: np.ndarray) -> np.ndarray:
    """Simple 3x3-ish gradient magnitude (Manhattan) blurred a bit to stabilise."""
    H, W = L_img.shape
    gx = np.zeros_like(L_img, dtype=np.float32)
    gy = np.zeros_like(L_img, dtype=np.float32)
    gx[:, 1:] = np.abs(L_img[:, 1:] - L_img[:, :-1])
    gy[1:, :] = np.abs(L_img[1:, :] - L_img[:-1, :])
    g = gx + gy
    return box_blur_2d(g, 1)


def estimate_ab_bias(lab_img: np.ndarray, alpha: np.ndarray) -> Tuple[float, float]:
    """
    Estimate global colour drift in a/b by averaging near-neutral pixels.
    This helps keep greys neutral in photo mode.
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


def prefilter_topk(
    src_lab_row: np.ndarray, pal_lab_mat: np.ndarray, k: int
) -> np.ndarray:
    """
    Quick prefilter: pick top-k candidates by squared Euclidean distance in Lab.
    This speeds up the more expensive cost calculation.
    """
    diff = pal_lab_mat - src_lab_row
    de2 = (diff * diff).sum(axis=1)
    if k >= de2.shape[0]:
        return np.argsort(de2)
    idx = np.argpartition(de2, k)[:k]
    return idx[np.argsort(de2[idx])]


def photo_cost_components(
    src_lab_row: np.ndarray,
    src_lch_row: np.ndarray,
    L_local: float,
    w_local_eff: float,
    tgt_lch_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute additional costs for photo mode for a set of target palette LCh rows:
    - Keep local contrast around L_local.
    - Soft hue and chroma matching.
    - In shadows, allow a little hue shift and chroma increase for legibility.
    - Penalise sending saturated sources to neutral targets.
    Returns (base_cost_array, target_chroma_array).
    """
    sL = src_lch_row[0]
    sC = src_lch_row[1]
    sh = src_lch_row[2]
    tL = tgt_lch_rows[:, 0]
    tC = tgt_lch_rows[:, 1]
    th = tgt_lch_rows[:, 2]

    base = np.zeros_like(tL, dtype=np.float32)

    # Keep local contrast around L_local
    base += W_LOCAL_BASE * w_local_eff * np.abs((tL - L_local) - (sL - L_local))

    # Do not flatten contrast too much (allow small margin)
    d_s = abs(sL - L_local)
    d_t = np.abs(tL - L_local)
    base += np.where(
        d_t + CONTRAST_MARGIN < d_s, W_CONTRAST * (d_s - d_t - CONTRAST_MARGIN), 0.0
    )

    # Hue guidance (weigh less when source is near-neutral)
    hue_w = min(1.0, sC / 40.0)
    base += (
        W_HUE * hue_w * (np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0)
    )

    # Chroma guidance
    base += W_CHROMA * np.abs(tC - sC)

    # Shadow handling: allow some hue shift and chroma bump to keep details readable
    if sL < 40.0:
        shadow_factor = (SHADOW_L - sL) / SHADOW_L
        hue_dist = np.minimum(np.abs(th - sh), 360.0 - np.abs(th - sh)) / 180.0
        chroma_up = np.maximum(0.0, tC - sC)
        base += shadow_factor * (
            W_SHADOW_HUE * hue_dist + W_SHADOW_CHROMA_UP * chroma_up
        )

    # If source is saturated, avoid mapping to neutral targets
    if sC >= CHROMA_SRC_MIN:
        base += np.where(
            tC <= NEUTRAL_C_MAX, NEUTRAL_PENALTY + 0.02 * np.maximum(0.0, sC - tC), 0.0
        )

    return base, tC


def ciede2000_vec(src_lab_row: np.ndarray, cand_lab_rows: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 between a single src_lab_row and many candidate rows."""
    L1 = src_lab_row[0]
    a1 = src_lab_row[1]
    b1 = src_lab_row[2]
    L2 = cand_lab_rows[:, 0]
    a2 = cand_lab_rows[:, 1]
    b2 = cand_lab_rows[:, 2]

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
    src_lab_row: np.ndarray,
    src_lch_row: np.ndarray,
    L_local: float,
    w_local_eff: float,
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
    near_extreme_neutral: bool,
    k: int,
) -> int:
    """
    Photo-mode: choose best palette index for one pixel.
    Steps:
      1) Prefilter to k nearest by Lab Euclidean distance (fast).
      2) Compute CIEDE2000 + photo costs on that subset.
      3) If pixel is near extreme neutral, force choosing from neutral palette colours.
    """
    idxs = prefilter_topk(src_lab_row, pal_lab_mat, k)
    cand_lab = pal_lab_mat[idxs]
    cand_lch = pal_lch_mat[idxs]

    dE = ciede2000_vec(src_lab_row, cand_lab)
    comp_base, tC = photo_cost_components(
        src_lab_row, src_lch_row, L_local, w_local_eff, cand_lch
    )
    cost = dE + comp_base

    if near_extreme_neutral:
        # If the source pixel is near black or white and nearly neutral,
        # force neutral outputs (others get a huge penalty).
        cost = np.where(tC > NEUTRAL_C_MAX, cost + 1e6, cost)

    return int(idxs[int(np.argmin(cost))])


# ===================== Photo mode =====================


def dither_photo(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
    pal_lab_mat: np.ndarray,
    pal_lch_mat: np.ndarray,
) -> np.ndarray:
    """
    Photo-like rendering: per-pixel palette selection guided by local contrast,
    with a small Floyd-Steinberg error diffusion on L only to reduce banding.

    Steps:
      - Convert to Lab and compute local L reference and simple gradient magnitude.
      - Estimate global a/b bias from near-neutrals and subtract to keep greys clean.
      - For each visible pixel:
          * Adjusted Lab by removing a/b bias.
          * Compute LCh for source pixel.
          * Choose best palette via prefiltered CIEDE2000 + photo costs.
          * Place chosen RGB.
          * Diffuse a small portion of L error to neighbours.
    """
    H, W, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    src_lab_img = srgb_to_lab_batch(img_rgb).reshape(H, W, 3)
    L_src = src_lab_img[..., 0]
    L_local = box_blur_2d(L_src, BLUR_RADIUS)
    grad = grad_mag(L_src)
    ab_bias = estimate_ab_bias(src_lab_img, alpha)

    errL = np.zeros((H, W), dtype=np.float32)  # L error buffer for diffusion

    for y in range(H):
        serp_left = y % 2 == 0
        xs = range(W) if serp_left else range(W - 1, -1, -1)

        # Classic Floyd-Steinberg neighbours (serpentine)
        if serp_left:
            neighbors: Tuple[Tuple[int, int, float], ...] = (
                (1, 0, 7 / 16),
                (-1, 1, 3 / 16),
                (0, 1, 5 / 16),
                (1, 1, 1 / 16),
            )
        else:
            neighbors = (
                (-1, 0, 7 / 16),
                (1, 1, 3 / 16),
                (0, 1, 5 / 16),
                (-1, 1, 1 / 16),
            )

        for x in xs:
            if alpha[y, x] == 0:
                continue

            # Weight local-contrast constraint less at edges
            wloc = max(0.2, 1.0 - (grad[y, x] / GRAD_K))

            # Apply L error diffusion and subtract global a/b bias
            sL = float(src_lab_img[y, x, 0]) + float(errL[y, x])
            sa = float(src_lab_img[y, x, 1]) - ab_bias[0]
            sb = float(src_lab_img[y, x, 2]) - ab_bias[1]
            src_lab_adj = np.array([sL, sa, sb], dtype=np.float32)

            # Compute source LCh
            C = float(math.hypot(src_lab_adj[1], src_lab_adj[2]))
            h = (
                math.degrees(math.atan2(src_lab_adj[2], src_lab_adj[1])) + 360.0
            ) % 360.0
            src_lch_adj = np.array([src_lab_adj[0], C, h], dtype=np.float32)

            # Special casing for near-perfect neutrals near extremes
            near_extreme_neutral = (C <= NEAR_NEUTRAL_C) and (
                sL <= NEAR_BLACK_L or sL >= NEAR_WHITE_L
            )

            # Pick best palette colour
            j = nearest_palette_idx_photo_lab_prefilter_vec(
                src_lab_adj,
                src_lch_adj,
                float(L_local[y, x]),
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

            # Diffuse small portion of L error to neighbours
            tgt_L = float(pal_lch_mat[j, 0])
            eL = max(-EL_CLAMP, min(EL_CLAMP, sL - tgt_L))
            for dx, dy, w in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] != 0:
                    errL[ny, nx] += eL * w

    return out


# ===================== Palette enforcement =====================


def _palette_set(palette: List[PaletteItem]) -> set[Tuple[int, int, int]]:
    return {p.rgb for p in palette}


def is_palette_only(
    img_rgb: np.ndarray, alpha: np.ndarray, pal_set: set[Tuple[int, int, int]]
) -> bool:
    """Return True if all visible pixels are in the palette set."""
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
    If any colours are not in the palette (should not happen often), replace them
    by finding the nearest palette colour in Lab using a unique-colour pass.
    """
    H, W, _ = out_rgb.shape
    pal_set = _palette_set(palette)
    mask = alpha != 0
    samples = out_rgb[mask]
    if samples.size == 0:
        return out_rgb

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

    # Simple nearest in Lab for each off-palette unique
    for i in range(off_lab.shape[0]):
        s_lab = off_lab[i]
        diff = pal_lab_mat - s_lab
        de2 = (diff * diff).sum(axis=1)
        best = int(np.argmin(de2))
        r = int(pal_rgb_arr[best, 0])
        g = int(pal_rgb_arr[best, 1])
        b = int(pal_rgb_arr[best, 2])
        key = (int(off_np[i, 0]), int(off_np[i, 1]), int(off_np[i, 2]))
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
    out_rgb: np.ndarray,
    alpha: np.ndarray,
    palette: List[PaletteItem],
) -> np.ndarray:
    """
    Final safety pass: per-pixel nearest in RGB to the palette.
    Chunked to keep memory reasonable.
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


# ===================== Reporting =====================


def print_color_usage(
    out_rgb: np.ndarray, alpha: np.ndarray, name_of: Dict[Tuple[int, int, int], str]
) -> None:
    """Print a count of how many times each palette colour appears in the visible output."""
    h, w, _ = out_rgb.shape
    cnt: Counter[Tuple[int, int, int]] = Counter()
    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0:
                continue
            key = (int(out_rgb[y, x, 0]), int(out_rgb[y, x, 1]), int(out_rgb[y, x, 2]))
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
    """Count how many visible pixels are not exact palette colours."""
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


# ===================== Auto mode heuristic =====================


def decide_auto_mode(img_rgb: np.ndarray, alpha: np.ndarray) -> str:
    """
    Pick 'pixel' if the image already looks like pixel art:
      - few unique colours, or
      - top-16 colours dominate the visible pixels.
    Otherwise pick 'photo'.
    """
    items = unique_colors_and_counts(img_rgb, alpha)
    total = sum(n for _c, n in items)
    n_uniques = len(items)
    topk = sum(n for _c, n in items[: min(16, n_uniques)])
    share = (topk / total) if total > 0 else 1.0
    return "pixel" if (n_uniques <= 512 or share >= 0.80) else "photo"


# ===================== Main =====================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Palette remapper. Args: --mode {auto,pixel,photo}, --height N, --debug"
    )
    parser.add_argument("input", help="Input image")
    parser.add_argument("--output", "-o", help="Output path (.png forced)")
    parser.add_argument(
        "--mode",
        choices=["auto", "pixel", "photo"],
        default="auto",
        help="Render strategy: 'pixel' for pixel art, 'photo' for photos, 'auto' to choose.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Process height in pixels. Downscale only; never upscale. Omit to keep original.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = parser.parse_args()

    t0 = time.perf_counter()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_wplace.png")
    )

    # Load and report base size and alpha stats
    img_rgb, alpha = load_image_rgba(in_path)
    H0, W0, _ = img_rgb.shape
    if args.debug:
        a255 = int((alpha == 255).sum())
        a0 = int((alpha == 0).sum())
        print(f"[debug] load   {W0}x{H0}  alpha(255)={a255}  alpha(0)={a0}")

    # Build palette structures
    palette, name_of, pal_lab, pal_lch = build_palette()
    pal_lab_mat = pal_lab.astype(np.float32)
    pal_lch_mat = pal_lch.astype(np.float32)
    pal_set = _palette_set(palette)

    # Decide mode (auto) and resize strategy
    mode_eff = args.mode if args.mode != "auto" else decide_auto_mode(img_rgb, alpha)
    resample = Image.Resampling.BOX if mode_eff == "pixel" else Image.Resampling.LANCZOS

    # Downscale by height only (never upscale)
    img_rgb, alpha = resize_rgba_height(img_rgb, alpha, args.height, resample)
    H1, W1, _ = img_rgb.shape
    if args.debug:
        a255 = int((alpha == 255).sum())
        a0 = int((alpha == 0).sum())
        print(f"[debug] size   {W1}x{H1}  alpha(255)={a255}  alpha(0)={a0}")
        print(f"[debug] mode   {mode_eff}")

    # Worker count for candidate building (pixel mode)
    cpu = os.cpu_count() or 1
    workers = max(1, min(32, cpu - 1))

    if mode_eff == "photo":
        # Photo pipeline
        t_photo0 = time.perf_counter()
        out_rgb = dither_photo(img_rgb, alpha, palette, pal_lab_mat, pal_lch_mat)

        # Ensure the result is strictly in-palette (should already be, but double-lock)
        ok0 = is_palette_only(out_rgb, alpha, pal_set)
        if not ok0:
            out_rgb = lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat)
        ok1 = is_palette_only(out_rgb, alpha, pal_set)
        if not ok1:
            out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, palette)
        ok2 = is_palette_only(out_rgb, alpha, pal_set)
        t_photo1 = time.perf_counter()

        # Save and report
        out_path = save_image_rgba(out_path, out_rgb, alpha)
        if args.debug:
            off = 0 if ok2 else count_off_palette_pixels(out_rgb, alpha, palette)
            print(
                f"[debug] photo  time={t_photo1 - t_photo0:.3f}s  palette_ok={ok2}  off_pixels={off}"
            )
        print("Mode: photo")
        print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
        print_color_usage(out_rgb, alpha, name_of)
        if args.debug:
            print(f"[debug] total  {time.perf_counter() - t0:.3f}s")
        return

    # Pixel pipeline
    t_pix0 = time.perf_counter()

    # Unique colours visible
    sources, src_lab, src_lch = build_sources(img_rgb, alpha)
    if len(sources) == 0:
        out_rgb = img_rgb
        candidates: List[List[Tuple[float, int]]] = []
        cost_lookup: Dict[Tuple[int, int], float] = {}
        assigned: Dict[int, int] = {}
    else:
        # Precompute candidate lists (possibly parallel)
        candidates, cost_lookup = build_candidates(
            src_lab, src_lch, pal_lab, pal_lch, sources, workers
        )
        # Assign each unique to a palette colour
        assigned = assign_unique(
            sources, candidates, cost_lookup, len(palette), src_lch, pal_lch
        )

        # Build mapping dict (source_rgb -> chosen palette rgb)
        mapping: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        for i, s in enumerate(sources):
            pal_idx = assigned.get(i, candidates[i][0][1])
            mapping[s.rgb] = palette[pal_idx].rgb

        # Apply mapping to the image
        out_rgb = apply_mapping(img_rgb, alpha, mapping)

    # Safety: lock to palette by uniques (rarely needed, fast)
    out_rgb = lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat)
    t_pix1 = time.perf_counter()

    # Save and report
    out_path = save_image_rgba(out_path, out_rgb, alpha)

    if args.debug:
        uniqs = unique_colors_and_counts(img_rgb, alpha)
        top16_share = sum(n for _c, n in uniqs[: min(16, len(uniqs))]) / max(
            1, sum(n for _c, n in uniqs)
        )
        print(
            f"[debug] pixel  size_in={W0}x{H0}  size_eff={W1}x{H1}  workers={workers}  time={t_pix1 - t_pix0:.3f}s"
        )
        print(f"[debug] uniques_visible={len(uniqs)}  top16_share={top16_share:.3f}")
        if len(sources) > 0:
            print("[debug] per-unique mapping:")
            for i, s in enumerate(sources):
                have_row = (i < len(candidates)) and len(candidates[i]) > 0
                pal_idx = (
                    assigned.get(i, candidates[i][0][1])
                    if have_row
                    else assigned.get(i, -1)
                )
                if pal_idx is None or pal_idx < 0:
                    continue
                dE = ciede2000_pair(s.lab, pal_lab[pal_idx])
                sL, sC, sh = s.lch.tolist()
                tL, tC, th = pal_lch[pal_idx].tolist()
                dh = hue_diff_deg(float(sh), float(th))
                print(
                    f"  src {rgb_to_hex(s.rgb):>7}  count={s.count:6d} -> {rgb_to_hex(palette[pal_idx].rgb):>7}  "
                    f"[dE={dE:5.2f}, dL={tL - sL:+.3f}, dC={tC - sC:+.3f}, dh={dh:.1f} deg]"
                )

    print("Mode: pixel")
    print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
    print_color_usage(out_rgb, alpha, name_of)

    if args.debug:
        print(f"[debug] total  {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
