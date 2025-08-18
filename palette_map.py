#!/usr/bin/env python3
"""
palette_map_dual.py — pixel/photo modes with lightness + neutrality guards

Modes
-----
- pixel : Global OKLab/OKLCh distance with:
          • stronger anti-brown for neutrals,
          • near-white clamp (prevents “light cyan as white”),
          • lightness guards (stops light colours falling to deep ones),
          • gentle green→olive preference over “stone/grey” for genuinely green sources,
          • global top-colour matching + greedy fill with strict re-use rules.
- photo : Classic OKLab nearest with a mild anti-grey bias for saturated sources.

Notes
-----
• Alpha is preserved.
• Downsize only (no upscaling) if --height is provided.
• Output defaults to "<input_stem>_wplace.png".
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List
import argparse
import numpy as np
from PIL import Image


# =============================================================================
#                               PALETTE
# =============================================================================
# Single source of truth: (hex, name, tier)
PALETTE_ENTRIES: tuple[tuple[str, str, str], ...] = (
    # Free
    ("#000000", "Black", "Free"),
    ("#3c3c3c", "Dark Gray", "Free"),
    ("#787878", "Gray", "Free"),
    ("#d2d2d2", "Light Gray", "Free"),
    ("#ffffff", "White", "Free"),
    ("#600018", "Deep Red", "Free"),
    ("#ed1c24", "Red", "Free"),
    ("#ff7f27", "Orange", "Free"),
    ("#f6aa09", "Gold", "Free"),
    ("#f9dd3b", "Yellow", "Free"),
    ("#fffabc", "Light Yellow", "Free"),
    ("#0eb968", "Dark Green", "Free"),
    ("#13e67b", "Green", "Free"),
    ("#87ff5e", "Light Green", "Free"),
    ("#0c816e", "Dark Teal", "Free"),
    ("#10aea6", "Teal", "Free"),
    ("#13e1be", "Light Teal", "Free"),
    ("#28509e", "Dark Blue", "Free"),
    ("#4093e4", "Blue", "Free"),
    ("#60f7f2", "Cyan", "Free"),
    ("#6b50f6", "Indigo", "Free"),
    ("#99b1fb", "Light Indigo", "Free"),
    ("#780c99", "Dark Purple", "Free"),
    ("#aa38b9", "Purple", "Free"),
    ("#e09ff9", "Light Purple", "Free"),
    ("#cb007a", "Dark Pink", "Free"),
    ("#ec1f80", "Pink", "Free"),
    ("#f38da9", "Light Pink", "Free"),
    ("#684634", "Dark Brown", "Free"),
    ("#95682a", "Brown", "Free"),
    ("#f8b277", "Beige", "Free"),
    # Premium
    ("#aaaaaa", "Medium Gray", "Premium"),
    ("#a50e1e", "Dark Red", "Premium"),
    ("#fa8072", "Light Red", "Premium"),
    ("#e45c1a", "Dark Orange", "Premium"),
    ("#9c8431", "Dark Goldenrod", "Premium"),
    ("#c5ad31", "Goldenrod", "Premium"),
    ("#e8d45f", "Light Goldenrod", "Premium"),
    ("#4a6b3a", "Dark Olive", "Premium"),
    ("#5a944a", "Olive", "Premium"),
    ("#84c573", "Light Olive", "Premium"),
    ("#0f799f", "Dark Cyan", "Premium"),
    ("#bbfaf2", "Light Cyan", "Premium"),
    ("#7dc7ff", "Light Blue", "Premium"),
    ("#4d31b8", "Dark Indigo", "Premium"),
    ("#4a4284", "Dark Slate Blue", "Premium"),
    ("#7a71c4", "Slate Blue", "Premium"),
    ("#b5aef1", "Light Slate Blue", "Premium"),
    ("#9b5249", "Dark Peach", "Premium"),
    ("#d18078", "Peach", "Premium"),
    ("#fab6a4", "Light Peach", "Premium"),
    ("#dba463", "Light Brown", "Premium"),
    ("#7b6352", "Dark Tan", "Premium"),
    ("#9c846b", "Tan", "Premium"),
    ("#d6b594", "Light Tan", "Premium"),
    ("#d18051", "Dark Beige", "Premium"),
    ("#ffc5a5", "Light Beige", "Premium"),
    ("#6d643f", "Dark Stone", "Premium"),
    ("#948c6b", "Stone", "Premium"),
    ("#cdc59e", "Light Stone", "Premium"),
    ("#333941", "Dark Slate", "Premium"),
    ("#6d758d", "Slate", "Premium"),
    ("#b3b9d1", "Light Slate", "Premium"),
)

# Derived views (names & tiers are used for the usage report)
PALETTE_HEX: tuple[str, ...] = tuple(h for h, _, _ in PALETTE_ENTRIES)
HEX_TO_NAME: Dict[str, str] = {h.lower(): n for h, n, _ in PALETTE_ENTRIES}
HEX_TO_TIER: Dict[str, str] = {h.lower(): t for h, _, t in PALETTE_ENTRIES}


# =============================================================================
#                               COLOR UTILS
# =============================================================================
def parse_hex(code: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (R, G, B) as ints 0..255"""
    s = code[1:] if code.startswith("#") else code
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def build_palette() -> np.ndarray:
    """Return unique RGB rows from PALETTE_ENTRIES as uint8 array [N,3]."""
    cols = np.array([parse_hex(h) for h in PALETTE_HEX], dtype=np.uint8)
    # De-dup while preserving order
    _, idx = np.unique(cols.view([("", cols.dtype)] * cols.shape[1]), return_index=True)
    return cols[np.sort(idx)]


def _srgb_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> OKLab float32 array of same leading shape [...,3]."""
    rgb = rgb_u8.astype(np.float32) / 255.0

    # sRGB -> linear
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    # Linear RGB -> LMS
    l = (
        0.4122214708 * lin[..., 0]
        + 0.5363325363 * lin[..., 1]
        + 0.0514459929 * lin[..., 2]
    )
    m = (
        0.2119034982 * lin[..., 0]
        + 0.6806995451 * lin[..., 1]
        + 0.1073969566 * lin[..., 2]
    )
    s = (
        0.0883024619 * lin[..., 0]
        + 0.2817188376 * lin[..., 1]
        + 0.6299787005 * lin[..., 2]
    )

    # LMS -> OKLab
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, A, B], axis=-1).astype(np.float32)


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
#                              PHOTO MODE
# =============================================================================
# Classic nearest (OKLab) with a mild anti-grey bias for saturated sources.
PHOTO_SRC_SAT_T: float = 0.06  # source considered saturated if C* > this
PHOTO_PAL_GREY_T: float = 0.03  # palette considered grey if C* < this
PHOTO_GREY_PENALTY: float = 1.7  # penalise mapping saturated colours to greys


def _nearest_indices_oklab_photo(
    src_lab: np.ndarray,
    pal_lab: np.ndarray,
    src_chroma: np.ndarray,
    pal_chroma: np.ndarray,
) -> np.ndarray:
    """Return nearest palette index per source in OKLab with anti-grey penalty."""
    diff = src_lab[:, None, :] - pal_lab[None, :, :]
    d2 = np.einsum("knc,knc->kn", diff, diff, optimize=True)
    mask = (src_chroma[:, None] > PHOTO_SRC_SAT_T) & (
        pal_chroma[None, :] < PHOTO_PAL_GREY_T
    )
    if mask.any():
        d2 = np.where(mask, d2 * (PHOTO_GREY_PENALTY**2), d2)
    return np.argmin(d2, axis=1)


def palette_map_array_photo(rgb_flat: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Vectorised nearest-colour mapping in OKLab (photo mode)."""
    unique_rgb, inverse = np.unique(rgb_flat, axis=0, return_inverse=True)
    pal_lab = _srgb_to_oklab(palette)
    pal_chroma = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    src_lab = _srgb_to_oklab(unique_rgb)
    src_chroma = np.hypot(src_lab[:, 1], src_lab[:, 2])
    idx = _nearest_indices_oklab_photo(src_lab, pal_lab, src_chroma, pal_chroma)
    return palette[idx][inverse].astype(np.uint8)


# =============================================================================
#                              PIXEL MODE
# =============================================================================
# Base OKLCh weights
HUE_WEIGHT_BASE = 0.45
CHROMA_FLOOR = 0.02
CHROMA_RANGE = 0.12

# Anti-grey for saturated sources (pixel mode)
SRC_SAT_T = 0.04
PAL_GREY_T = 0.05
GREY_PENALTY = 2.0

# Near-white guard for chromatic→white (legacy piece; kept)
WHITE_L_T = 0.92
WHITE_PENALTY = 3.0

# Stronger brown guard (reduce greys drifting warm)
BROWN_HUE_C = np.deg2rad(35.0)  # around orange
BROWN_HUE_BW = np.deg2rad(50.0)  # widened from 35°
BROWN_CHROMA_MIN = 0.05  # lowered to trigger sooner
BROWN_PENALTY = 6.0  # increased from 4.0

# Lightness guards to keep light colours from falling to deep choices
LIGHT_DROP_L_T = 0.06
LIGHT_SRC_T = 0.70
HUE_CLOSE_T = np.deg2rad(28.0)
CLOSE_CHROMA_T = 0.05
LIGHT_DROP_PEN = 7.0

# Soft bias to avoid going darker when the source is already light
LIGHT_SOFT_W = 0.25
LIGHT_SOFT_SRC = 0.60
LIGHT_SOFT_SPAN = 0.35

# Candidate & sharing controls (kept conservative for pixel art)
GOOD_ENOUGH_RATIO = 1.06
MAX_CANDIDATES = 24
SHARE_GUARD_RATIO = 1.08
STRICT_REUSE_RATIO = 1.10

# Near-white clamp (stop “light cyan used as white”; prefer white/light greys)
NEAR_WHITE_L_T = 0.94
NEAR_WHITE_C_T = 0.02
NEAR_WHITE_CYAN_PEN = 3.0  # mild, multiplicative (applied squared)
NEAR_WHITE_WHITE_BONUS = 0.85  # <1 pulls white-ish
NEAR_WHITE_VLG_BONUS = 0.92  # <1 pulls very light greys
CYAN_HUE_C = np.deg2rad(200.0)
CYAN_HUE_BW = np.deg2rad(60.0)

# Green vs “stone/grey” (nudge real greens toward greens/olives instead of stones)
GREEN_SRC_C_T = 0.06
GREEN_HUE_C = np.deg2rad(130.0)
GREEN_HUE_BW = np.deg2rad(55.0)
STONE_HUE_C = np.deg2rad(75.0)
STONE_HUE_BW = np.deg2rad(30.0)
STONE_C_MAX = 0.14
GREEN_TO_STONE_PEN = 1.2  # mild (applied squared)
OLIVE_HUE_C = np.deg2rad(105.0)
OLIVE_HUE_BW = np.deg2rad(30.0)
OLIVE_MIN_C = 0.10
OLIVE_BONUS = 0.95  # <1 pulls olive/green

# Neutral-ish threshold reused in a few places
NEUTRAL_SRC_T = 0.06  # “neutral-ish” if source C* < this

# Low-chroma green rescue (for cases like #3a553b)
LOWC_GREEN_C_MIN = 0.03
LOWC_GREEN_C_MAX = 0.06
LOWC_GREEN_HUE_BW = np.deg2rad(45.0)
LOWC_GREEN_BONUS = 0.96  # gentle pull to olive/green
LOWC_GREEN_STONE_PEN = 1.3  # gentle push away from stone/grey

# Cyan (blue-cyan) vs Teal preference for *light* cyans
CYAN_SRC_C_T = 0.05  # source must have some chroma
CYAN_SRC_L_T = 0.65  # only nudge lighter cyans
TEAL_HUE_C = np.deg2rad(170.0)  # teal/green-cyan center
TEAL_HUE_BW = np.deg2rad(25.0)
BLUECYAN_HUE_C = np.deg2rad(205.0)  # preferred target band
BLUECYAN_HUE_BW = np.deg2rad(35.0)
CYAN_TO_TEAL_PEN = 1.35  # squared → gentle push away from teal
CYAN_TO_BLUE_BONUS = 0.92  # <1 pulls toward blue-cyan
CYAN_NOT_DARKER_DL = 0.02  # don’t pull if target is noticeably darker


def compute_d2(src_rgb: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    """
    Build a distance table D^2[src, pal] in OKLab/OKLCh space,
    with a few domain-specific guards and soft biases layered on top.

    The base term is:
        0.8 * (ΔL)^2 + 1.0 * (ΔC)^2 + w_hue(srcC) * (Δh)^2
    where w_hue grows with source chroma (so hue matters more when colour is saturated).
    """
    # Convert to OKLab (float32)
    pal = _srgb_to_oklab(pal_rgb)
    src = _srgb_to_oklab(src_rgb)

    # Palette OKLab → L*, a*, b*, C*, h
    pal_L, pal_A, pal_B = pal[:, 0], pal[:, 1], pal[:, 2]
    pal_C = np.hypot(pal_A, pal_B)
    pal_h = np.arctan2(pal_B, pal_A)

    # Source OKLab → L*, a*, b*, C*, h
    src_L, src_A, src_B = src[:, 0], src[:, 1], src[:, 2]
    src_C = np.hypot(src_A, src_B)
    src_h = np.arctan2(src_B, src_A)

    # Base OKLCh deltas
    dL2 = (src_L[:, None] - pal_L[None, :]) ** 2
    dC2 = (src_C[:, None] - pal_C[None, :]) ** 2
    dh = _wrap(src_h[:, None] - pal_h[None, :])
    hue_w = (
        HUE_WEIGHT_BASE
        * np.clip((src_C - CHROMA_FLOOR) / CHROMA_RANGE, 0.0, 1.0)[:, None]
    )

    # Base distance (kept close to your original balance)
    d2 = 0.8 * dL2 + 1.0 * dC2 + hue_w * (dh**2)

    # Saturated sources: avoid mapping to very low-chroma palette colours
    pen_grey = (src_C[:, None] > SRC_SAT_T) & (pal_C[None, :] < PAL_GREY_T)
    if pen_grey.any():
        d2 = np.where(pen_grey, d2 * (GREY_PENALTY**2), d2)

    # Chromatic sources should not become near-white in the palette
    pen_white = (
        (src_C[:, None] > 0.04)
        & (pal_C[None, :] < PAL_GREY_T)
        & (pal_L[None, :] > WHITE_L_T)
    )
    if pen_white.any():
        d2 = np.where(pen_white, d2 * (WHITE_PENALTY**2), d2)

    # Near-white refinement: stop “light cyan” being used as white, nudge to white/light grey
    near_white = (src_L > NEAR_WHITE_L_T) & (src_C < NEAR_WHITE_C_T)
    if np.any(near_white):
        cyan_band = np.abs(_wrap(pal_h - CYAN_HUE_C)) <= CYAN_HUE_BW
        d2 = np.where(
            near_white[:, None] & cyan_band[None, :], d2 * (NEAR_WHITE_CYAN_PEN**2), d2
        )
        whiteish = (pal_C < 0.015) & (pal_L > 0.97)
        vlg = (pal_C < 0.030) & (pal_L > 0.92)  # very light grey
        d2 = np.where(
            near_white[:, None] & whiteish[None, :], d2 * NEAR_WHITE_WHITE_BONUS, d2
        )
        d2 = np.where(near_white[:, None] & vlg[None, :], d2 * NEAR_WHITE_VLG_BONUS, d2)

    # Brown guard for neutral-ish sources: avoid warm browns for near-neutrals
    neutral_src = src_C < NEUTRAL_SRC_T
    brown_band = (np.abs(_wrap(pal_h - BROWN_HUE_C)) <= BROWN_HUE_BW) & (
        pal_C >= BROWN_CHROMA_MIN
    )
    if np.any(brown_band) and np.any(neutral_src):
        d2 = np.where(
            neutral_src[:, None] & brown_band[None, :], d2 * (BROWN_PENALTY**2), d2
        )

    # Soft bias for light colours: prefer not to go darker if otherwise similar
    soft_w = np.clip((src_L - LIGHT_SOFT_SRC) / max(LIGHT_SOFT_SPAN, 1e-6), 0.0, 1.0)[
        :, None
    ]
    delta_L_neg = np.clip(src_L[:, None] - pal_L[None, :], 0.0, 1.0)
    d2 = d2 + LIGHT_SOFT_W * (soft_w * (delta_L_neg**2))

    # Hard light→deep guard when hue and chroma are otherwise close
    light_src = src_L > LIGHT_SRC_T
    too_dark = pal_L[None, :] < (src_L[:, None] - LIGHT_DROP_L_T)
    hue_close = np.abs(_wrap(src_h[:, None] - pal_h[None, :])) <= HUE_CLOSE_T
    chroma_close = np.abs(src_C[:, None] - pal_C[None, :]) <= CLOSE_CHROMA_T
    forbid_mask = light_src[:, None] & too_dark & hue_close & chroma_close
    if np.any(forbid_mask):
        d2 = np.where(forbid_mask, d2 * (LIGHT_DROP_PEN**2), d2)

    # Green vs stone: gently prefer olive/green over stone when the source is clearly green
    greenish_src = (src_C >= GREEN_SRC_C_T) & (
        np.abs(_wrap(src_h - GREEN_HUE_C)) <= GREEN_HUE_BW
    )
    if np.any(greenish_src):
        close_L = np.abs(src_L[:, None] - pal_L[None, :]) <= 0.10
        stone_band = (np.abs(_wrap(pal_h - STONE_HUE_C)) <= STONE_HUE_BW) | (
            pal_C < STONE_C_MAX
        )
        d2 = np.where(
            greenish_src[:, None] & stone_band[None, :] & close_L,
            d2 * (GREEN_TO_STONE_PEN**2),
            d2,
        )
        olive_band = (np.abs(_wrap(pal_h - OLIVE_HUE_C)) <= OLIVE_HUE_BW) & (
            pal_C >= OLIVE_MIN_C
        )
        green_band = np.abs(_wrap(pal_h - GREEN_HUE_C)) <= GREEN_HUE_BW
        pull = greenish_src[:, None] & (olive_band | green_band)[None, :] & close_L
        d2 = np.where(pull, d2 * OLIVE_BONUS, d2)

    # Low-chroma green rescue (e.g. #3a553b): pull toward olive/green, push away from stone/grey
    lowc_green = (
        (src_C >= LOWC_GREEN_C_MIN)
        & (src_C <= LOWC_GREEN_C_MAX)
        & (np.abs(_wrap(src_h - GREEN_HUE_C)) <= LOWC_GREEN_HUE_BW)
    )
    if np.any(lowc_green):
        close_L = np.abs(src_L[:, None] - pal_L[None, :]) <= 0.12
        olive_band = (np.abs(_wrap(pal_h - OLIVE_HUE_C)) <= OLIVE_HUE_BW) & (
            pal_C >= 0.08
        )
        green_band = np.abs(_wrap(pal_h - GREEN_HUE_C)) <= GREEN_HUE_BW
        stone_band = (np.abs(_wrap(pal_h - STONE_HUE_C)) <= STONE_HUE_BW) | (
            pal_C < 0.10
        )
        d2 = np.where(
            lowc_green[:, None] & (olive_band | green_band)[None, :] & close_L,
            d2 * LOWC_GREEN_BONUS,
            d2,
        )
        d2 = np.where(
            lowc_green[:, None] & stone_band[None, :] & close_L,
            d2 * (LOWC_GREEN_STONE_PEN**2),
            d2,
        )

    # Light-cyan preference: keep light cyans in blue-cyan rather than teal/green-cyan
    cyan_light_src = (
        (src_C >= CYAN_SRC_C_T)
        & (src_L >= CYAN_SRC_L_T)
        & (np.abs(_wrap(src_h - CYAN_HUE_C)) <= CYAN_HUE_BW)
    )
    if np.any(cyan_light_src):
        teal_band = np.abs(_wrap(pal_h - TEAL_HUE_C)) <= TEAL_HUE_BW
        bluecyan_band = np.abs(_wrap(pal_h - BLUECYAN_HUE_C)) <= BLUECYAN_HUE_BW
        not_darker = pal_L[None, :] >= (src_L[:, None] - CYAN_NOT_DARKER_DL)

        # Push away from teal (green-leaning) when the source is light cyan
        d2 = np.where(
            cyan_light_src[:, None] & teal_band[None, :],
            d2 * (CYAN_TO_TEAL_PEN**2),
            d2,
        )
        # Mild pull toward blue-cyan choices that aren’t darker than the source
        d2 = np.where(
            cyan_light_src[:, None] & bluecyan_band[None, :] & not_darker,
            d2 * CYAN_TO_BLUE_BONUS,
            d2,
        )

    return d2


# =============================================================================
#                     CANDIDATE SELECTION & ASSIGNMENT
# =============================================================================
def _build_cands(d2: np.ndarray) -> List[np.ndarray]:
    """
    For each unique source colour (row in d2), keep the palette indices whose
    distance is within GOOD_ENOUGH_RATIO of the row’s best. Also cap the count
    to MAX_CANDIDATES to keep matching fast and stable.
    """
    K = d2.shape[0]
    best = d2.min(axis=1)
    order = np.argsort(d2, axis=1)
    out: List[np.ndarray] = []
    for i in range(K):
        r = order[i]
        r = r[d2[i, r] <= best[i] * GOOD_ENOUGH_RATIO]
        if r.size == 0:
            r = order[i, :1]
        out.append(r[:MAX_CANDIDATES])
    return out


def _max_match(
    top_idx: np.ndarray, cands: List[np.ndarray], N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight maximum bipartite matching for the top M frequent source colours
    so they each try to claim distinct palette entries first.
    Returns:
        owner[N]  : which source row owns palette index j (or -1)
        chosen[K] : chosen palette index per source (or -1)
    """
    owner = np.full(N, -1, dtype=np.int32)
    chosen = np.full(len(cands), -1, dtype=np.int32)

    def dfs(i: int, seen: np.ndarray) -> bool:
        for j in cands[i]:
            jj = int(j)
            if seen[jj]:
                continue
            seen[jj] = True
            oi = int(owner[jj])
            if oi == -1 or dfs(oi, seen):
                owner[jj] = int(i)
                chosen[int(i)] = jj
                return True
        return False

    for i_val in top_idx:
        ii = int(i_val)
        seen = np.zeros(N, dtype=bool)
        dfs(ii, seen)

    return owner, chosen


def map_visible_pixels_global(
    rgb_vis_flat: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """
    Global mapping for visible (non-transparent) pixels:
      1) unique colours -> distance table -> candidate lists,
      2) match most frequent uniques to distinct palette colours,
      3) greedy fill with strict “don’t reuse if close alt exists”.
    """
    # Collapse to unique visible RGBs (saves work & stabilizes results)
    uniq, inv, counts = np.unique(
        rgb_vis_flat, axis=0, return_inverse=True, return_counts=True
    )
    K, N = uniq.shape[0], palette.shape[0]

    # Compute distances & build candidate lists
    d2 = compute_d2(uniq, palette)
    cands = _build_cands(d2)

    # Most frequent first (these compete for unique palette slots)
    order_src = np.argsort(-counts)
    topM = min(N, K)
    top_idx = order_src[:topM]

    # Try to give distinct palette entries to the top colours
    owner, chosen = _max_match(top_idx, cands, N)
    used = set(int(j) for j in owner if int(j) != -1)

    # Initialize per-unique choice
    choice = np.full(K, -1, dtype=np.int32)

    # Assign matched top colours
    for i_val in top_idx:
        ii = int(i_val)
        cj = int(chosen[ii])
        if cj != -1:
            choice[ii] = cj
            used.add(cj)

    # Greedy pass for the rest with strict reuse avoidance
    for i_val in order_src:
        ii = int(i_val)
        if choice[ii] != -1:
            continue

        row = cands[ii]
        if row.size == 0:
            row = np.argsort(d2[ii])[:1]

        best = int(row[0])
        db = float(d2[ii, best])

        # If the best palette colour is taken, try a close free alternative
        if best in used:
            alt = None
            for jv in row[1:]:
                j = int(jv)
                if j not in used and d2[ii, j] <= db * SHARE_GUARD_RATIO:
                    alt = j
                    break
            if alt is not None:
                choice[ii] = alt
                used.add(alt)
                continue

        # Avoid any used slot if a free alternative is within STRICT_REUSE_RATIO
        alt2 = None
        for jv in row:
            j = int(jv)
            if j in used:
                continue
            if d2[ii, j] <= db * STRICT_REUSE_RATIO:
                alt2 = j
                break
        if alt2 is not None:
            choice[ii] = alt2
            used.add(alt2)
            continue

        # Fall back to the best (may reuse)
        choice[ii] = best
        used.add(best)

    # Expand choices back to full image order
    return palette[choice][inv].astype(np.uint8)


# =============================================================================
#                               IO / PIPELINE
# =============================================================================
def scale_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Downsize to target height using BOX; never upscale."""
    if target_h <= 0:
        return img
    w, h = img.width, img.height
    if target_h >= h:
        return img
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), resample=Image.Resampling.BOX)


def report_usage(img: Image.Image) -> None:
    """Print colours used with names and tiers (no hex), sorted by count desc."""
    arr = np.array(img, dtype=np.uint8)
    if img.mode == "RGBA":
        mask = arr[:, :, 3] > 0
        if not mask.any():
            print("Colours used: none (fully transparent)")
            return
        cols = arr[:, :, :3][mask].reshape(-1, 3)
    else:
        cols = arr.reshape(-1, 3)

    uniq, counts = np.unique(cols, axis=0, return_counts=True)

    def _rgb_to_hex(row: np.ndarray) -> str:
        return f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}"

    items: List[tuple[int, str, str]] = []
    for rgbv, cnt in zip(uniq, counts):
        hx = _rgb_to_hex(rgbv).lower()
        name = HEX_TO_NAME.get(hx, "Unknown")
        tier = HEX_TO_TIER.get(hx, "Premium")
        items.append((cnt, name, tier))
    items.sort(key=lambda t: t[0], reverse=True)

    print("Colours used:")
    for cnt, name, tier in items:
        print(f"{name} [{tier}]: {cnt}")


def process(
    input_path: str, output_path: str | None, target_height: int, mode: str
) -> None:
    """Main processing pipeline: load → downsize → map → save + usage report."""
    palette = build_palette()

    # Load & convert (preserve transparency; handles PNG P/tRNS as well)
    im_in = Image.open(input_path)
    im_rgba = im_in.convert("RGBA")
    im_small = scale_to_height(im_rgba, target_height)

    arr = np.array(im_small, dtype=np.uint8)
    h, w = arr.shape[:2]

    # Flatten to RGB + keep alpha for visibility mask
    rgb = arr[:, :, :3].reshape(-1, 3)
    a = arr[:, :, 3].reshape(-1)
    vis_mask = a > 0  # only map visible pixels

    # Mode selection (simple heuristic for “photo-like” images)
    chosen_mode = mode
    if mode == "auto":
        uniq_vis = np.unique(rgb[vis_mask], axis=0).shape[0] if vis_mask.any() else 0
        chosen_mode = "photo" if uniq_vis > 4096 else "pixel"

    # Map visible pixels
    rgb_out = rgb.copy()
    if vis_mask.any():
        if chosen_mode == "pixel":
            mapped = map_visible_pixels_global(rgb[vis_mask], palette)
        else:
            mapped = palette_map_array_photo(rgb[vis_mask], palette)
        rgb_out[vis_mask] = mapped

    # Re-pack RGBA and save
    out = np.dstack([rgb_out.reshape(h, w, 3), a.reshape(h, w)]).astype(np.uint8)
    out_img = Image.fromarray(out)

    # Output path
    if output_path is None or output_path == "":
        out_path = Path(input_path).with_name(f"{Path(input_path).stem}_wplace.png")
    else:
        out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", optimize=False)

    # Logging
    print(f"Mode: {chosen_mode}")
    print(f"Wrote {out_path} | size={w}x{h} | palette_size={len(PALETTE_ENTRIES)}\n")
    report_usage(out_img)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dual-mode palette mapper with lightness/neutrality guards and strict global uniqueness."
    )
    ap.add_argument("input", type=str)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path; default '<input_stem>_wplace.png'",
    )
    ap.add_argument(
        "--height", type=int, default=512, help="Output height (downsize only)"
    )
    ap.add_argument("--mode", choices=["auto", "pixel", "photo"], default="auto")
    args = ap.parse_args()
    process(args.input, args.output, args.height, args.mode)


if __name__ == "__main__":
    main()
