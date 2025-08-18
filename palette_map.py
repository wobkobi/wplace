#!/usr/bin/env python3
"""
palette_map_dual.py  — pixel/photo modes with lightness + neutrality guards

Modes
-----
- pixel : Global OKLab/OKLCh distance with:
          - stronger anti-brown for neutrals,
          - near-white clamp (prevents "light cyan as white"),
          - lightness guards (stops light colours falling to deep ones),
          - gentle green->olive preference over "stone/grey" for genuinely green sources,
          - global top-colour matching + greedy fill with strict re-use rules.
- photo : Classic OKLab nearest with a mild anti-grey bias for saturated sources.

Notes
-----
- Alpha is preserved.
- Downsize only (no upscaling) if --height is provided.
- Output defaults to "<input_stem>_wplace.png".
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
import argparse
import numpy as np
from PIL import Image


# ---------- Single source of truth: (hex, name, tier) ----------
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


# ---------------- Utils ----------------
def parse_hex(code: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (R, G, B) as ints 0..255"""
    s = code[1:] if code.startswith("#") else code
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def build_palette() -> np.ndarray:
    """Return unique RGB rows from PALETTE_ENTRIES as uint8 array [N,3]."""
    cols = np.array([parse_hex(h) for h in PALETTE_HEX], dtype=np.uint8)
    # de-dup while preserving order
    _, idx = np.unique(cols.view([("", cols.dtype)] * cols.shape[1]), return_index=True)
    return cols[np.sort(idx)]


def _srgb_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> OKLab float32 array of same leading shape [...,3]."""
    rgb = rgb_u8.astype(np.float32) / 255.0

    # sRGB -> linear
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    # LMS
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

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, A, B], axis=-1).astype(np.float32)


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ======== PHOTO MODE (classic nearest with chroma-aware anti-grey) ========
# Higher -> fewer sources counted as "saturated"; Lower -> more sources counted as saturated.
PHOTO_SRC_SAT_T: float = 0.06
# Higher -> fewer palette colours treated as grey; Lower -> more palette colours treated as grey.
PHOTO_PAL_GREY_T: float = 0.03
# Higher -> stronger push away from grey targets for saturated sources; Lower -> weaker push.
PHOTO_GREY_PENALTY: float = 1.7


def _nearest_indices_oklab_photo(
    src_lab: np.ndarray,
    pal_lab: np.ndarray,
    src_chroma: np.ndarray,
    pal_chroma: np.ndarray,
) -> np.ndarray:
    """Return nearest palette index per source colour (photo mode)."""
    delta = src_lab[:, None, :] - pal_lab[None, :, :]
    dist_sq = np.einsum("knc,knc->kn", delta, delta, optimize=True)

    saturated_to_grey = (src_chroma[:, None] > PHOTO_SRC_SAT_T) & (
        pal_chroma[None, :] < PHOTO_PAL_GREY_T
    )
    if saturated_to_grey.any():
        dist_sq = np.where(
            saturated_to_grey, dist_sq * (PHOTO_GREY_PENALTY**2), dist_sq
        )

    nearest_idx = np.argmin(dist_sq, axis=1)
    return nearest_idx


def palette_map_array_photo(rgb_flat: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Vectorised nearest-colour mapping in OKLab (photo mode)."""
    unique_rgb, inverse_idx = np.unique(rgb_flat, axis=0, return_inverse=True)
    pal_lab = _srgb_to_oklab(palette)
    pal_chroma = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    src_lab = _srgb_to_oklab(unique_rgb)
    src_chroma = np.hypot(src_lab[:, 1], src_lab[:, 2])
    idx = _nearest_indices_oklab_photo(src_lab, pal_lab, src_chroma, pal_chroma)
    return palette[idx][inverse_idx].astype(np.uint8)


# ======== PIXEL MODE (global distance + matching + guards) ========
# Base OKLCh weights
# Higher -> hue differences matter more; Lower -> hue differences matter less.
HUE_WEIGHT_BASE = 0.45
# Higher -> more sources considered above floor (slightly increases hue weight overall); Lower -> more sources near zero hue weight.
CHROMA_FLOOR = 0.02
# Higher -> slower ramp-up of hue weight as chroma increases; Lower -> faster ramp-up.
CHROMA_RANGE = 0.12

# Anti-grey for saturated sources (pixel mode)
# Higher -> fewer sources counted as saturated; Lower -> more sources counted as saturated.
SRC_SAT_T = 0.04
# Higher -> fewer palette colours treated as grey; Lower -> more palette colours treated as grey.
PAL_GREY_T = 0.05
# Higher -> stronger push away from grey targets; Lower -> weaker push.
GREY_PENALTY = 2.0

# Near-white guard for chromatic->white (legacy piece; kept)
# Higher -> harder to map chromatic colours to very light greys/white; Lower -> easier.
WHITE_L_T = 0.92
# Higher -> stronger block of chromatic->white mapping; Lower -> weaker block.
WHITE_PENALTY = 3.0

# Stronger brown guard (reduce greys drifting warm)
# Center of "brown/orange" band (radians). Shift to aim the band.
BROWN_HUE_C = np.deg2rad(35.0)
# Higher -> wider band of hues considered brown; Lower -> narrower.
BROWN_HUE_BW = np.deg2rad(50.0)
# Higher -> only more saturated browns get penalized; Lower -> even weak browns get penalized.
BROWN_CHROMA_MIN = 0.05
# Higher -> stronger push away from warm browns for neutral sources; Lower -> weaker push.
BROWN_PENALTY = 6.0

# Lightness guards to keep light colours from falling to deep choices
# Higher -> larger allowed drop in L before penalty; Lower -> earlier penalty when mapping darker.
LIGHT_DROP_L_T = 0.06
# Higher -> only very light sources get guarded; Lower -> more mid-lights get guarded.
LIGHT_SRC_T = 0.70
# Higher -> requires closer hue match to trigger "same family" guard; Lower -> looser hue matching.
HUE_CLOSE_T = np.deg2rad(28.0)
# Higher -> requires closer chroma match to trigger guard; Lower -> looser chroma matching.
CLOSE_CHROMA_T = 0.05
# Higher -> stronger block when a light colour would fall to a deep one; Lower -> weaker block.
LIGHT_DROP_PEN = 7.0

# Soft bias against going darker (smooth, continuous)
# Higher -> stronger soft penalty on going darker; Lower -> weaker.
LIGHT_SOFT_W = 0.25
# Higher -> the soft penalty kicks in only for brighter sources; Lower -> starts affecting midtones.
LIGHT_SOFT_SRC = 0.60
# Higher -> wider ramp region for the soft penalty; Lower -> narrower ramp.
LIGHT_SOFT_SPAN = 0.35

# Candidate & sharing controls (kept conservative for pixel art)
# Higher -> keep more near-best palette candidates; Lower -> keep fewer (more decisive).
GOOD_ENOUGH_RATIO = 1.06
# Higher -> consider more candidates per source (slower, more flexible); Lower -> fewer (faster, stricter).
MAX_CANDIDATES = 24
# Higher -> easier to choose a different unused palette when best is taken; Lower -> harder (more reuse).
SHARE_GUARD_RATIO = 1.08
# Higher -> stricter about avoiding reuse when a free alternative is close; Lower -> reuse more often.
STRICT_REUSE_RATIO = 1.10

# Near-white clamp (stop "light cyan used as white" and favour white/light greys)
# Higher -> only very bright pixels are treated as near-white; Lower -> more pixels treated as near-white.
NEAR_WHITE_L_T = 0.94
# Higher -> only extremely neutral pixels engage the near-white rules; Lower -> slightly tinted pixels also engage.
NEAR_WHITE_C_T = 0.02
# Higher -> stronger push away from cyan-tinted whites; Lower -> weaker push.
NEAR_WHITE_CYAN_PEN = 3.0
# Lower than 1.0 pulls more strongly toward white; set closer to 1.0 to weaken pull.
NEAR_WHITE_WHITE_BONUS = 0.85
# Lower than 1.0 pulls more strongly toward very light grey; closer to 1.0 weakens it.
NEAR_WHITE_VLG_BONUS = 0.92
# Cyan band center and width used in near-white clamp.
CYAN_HUE_C = np.deg2rad(200.0)
# Higher -> wider cyan band considered "bad" for near-white; Lower -> narrower.
CYAN_HUE_BW = np.deg2rad(60.0)

# Green vs "stone/grey" (nudge real greens toward greens/olives instead of stones)
# Higher -> requires more saturated green sources to enable these tweaks; Lower -> triggers more often.
GREEN_SRC_C_T = 0.06
# Green center and band. Shift/resize to redefine what counts as "green".
GREEN_HUE_C = np.deg2rad(130.0)
GREEN_HUE_BW = np.deg2rad(55.0)
# Stone center and band. Adjust to catch more or fewer stone-like hues.
STONE_HUE_C = np.deg2rad(75.0)
STONE_HUE_BW = np.deg2rad(30.0)
# Higher -> only very low-chroma targets count as stone-ish; Lower -> more targets flagged as stone-ish.
STONE_C_MAX = 0.14
# Higher -> stronger push away from stone/grey when the source is green; Lower -> weaker.
GREEN_TO_STONE_PEN = 1.2
# Olive center and band. Adjust to pull greens more toward olive range.
OLIVE_HUE_C = np.deg2rad(105.0)
OLIVE_HUE_BW = np.deg2rad(30.0)
# Higher -> only more saturated olives are favored; Lower -> weaker olives also favored.
OLIVE_MIN_C = 0.10
# Below 1.0 pulls more strongly toward olive/green; closer to 1.0 weakens pull.
OLIVE_BONUS = 0.95

# Neutral threshold used in a few places
# Higher -> more pixels count as "neutral-ish"; Lower -> fewer.
NEUTRAL_SRC_T = 0.06

# Low-chroma green rescue (for cases like #3a553b)
# Increase to include slightly stronger chroma greens; decrease to require very low chroma.
LOWC_GREEN_C_MIN = 0.03
# Increase to include higher chroma greens; decrease to restrict to very muted greens.
LOWC_GREEN_C_MAX = 0.06
# Higher -> broader hue band for "low-chroma green"; Lower -> tighter.
LOWC_GREEN_HUE_BW = np.deg2rad(45.0)
# Below 1.0 pulls toward olive/green more; closer to 1.0 reduces the pull.
LOWC_GREEN_BONUS = 0.96
# Higher -> stronger push away from stone/grey for low-chroma greens; Lower -> weaker.
LOWC_GREEN_STONE_PEN = 1.3

# Cyan (blue-cyan) vs Teal preference for light cyans
# Higher -> require more chroma to trigger cyan vs teal logic; Lower -> trigger more often.
CYAN_SRC_C_T = 0.05
# Higher -> only lighter cyans get adjusted; Lower -> mid-light cyans can be adjusted too.
CYAN_SRC_L_T = 0.65
# Center and width of cyan detection band for the light-cyan rule.
CYAN_HUE_BAND_C = np.deg2rad(200.0)
# Higher -> detect a wider range of cyan-like sources; Lower -> narrower detection.
CYAN_HUE_BAND_BW = np.deg2rad(35.0)
# Teal target band center and width.
TEAL_HUE_C = np.deg2rad(170.0)
# Higher -> more target hues count as teal; Lower -> fewer.
TEAL_HUE_BW = np.deg2rad(25.0)
# Blue-cyan preferred band center and width.
BLUECYAN_HUE_C = np.deg2rad(205.0)
# Higher -> more target hues count as blue-cyan; Lower -> fewer.
BLUECYAN_HUE_BW = np.deg2rad(35.0)
# Higher -> stronger push away from teal for light cyans; Lower -> weaker.
CYAN_TO_TEAL_PEN = 1.35
# Below 1.0 pulls more toward blue-cyan; closer to 1.0 weakens the pull.
CYAN_TO_BLUE_BONUS = 0.92
# Higher -> allow slightly darker blue-cyan targets; Lower -> require almost no darkening.
CYAN_NOT_DARKER_DL = 0.02


def compute_cost_matrix(src_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """
    Build the OKLab-based cost matrix:
      cost[row=source_color, col=palette_color] = distance with guards/biases.
    """
    # Convert to OKLab
    palette_lab = _srgb_to_oklab(palette_rgb)
    source_lab = _srgb_to_oklab(src_rgb)

    # Palette components
    palette_L, palette_A, palette_B = (
        palette_lab[:, 0],
        palette_lab[:, 1],
        palette_lab[:, 2],
    )
    palette_C = np.hypot(palette_A, palette_B)
    palette_hue = np.arctan2(palette_B, palette_A)

    # Source components
    source_L, source_A, source_B = source_lab[:, 0], source_lab[:, 1], source_lab[:, 2]
    source_C = np.hypot(source_A, source_B)
    source_hue = np.arctan2(source_B, source_A)

    # Base OKLCh distance
    delta_L_sq = (source_L[:, None] - palette_L[None, :]) ** 2
    delta_C_sq = (source_C[:, None] - palette_C[None, :]) ** 2
    delta_hue_wrapped = _wrap(source_hue[:, None] - palette_hue[None, :])
    hue_weight = (
        HUE_WEIGHT_BASE
        * np.clip((source_C - CHROMA_FLOOR) / CHROMA_RANGE, 0.0, 1.0)[:, None]
    )

    cost = 0.8 * delta_L_sq + 1.0 * delta_C_sq + hue_weight * (delta_hue_wrapped**2)

    # Anti-grey for saturated sources
    saturated_to_grey = (source_C[:, None] > SRC_SAT_T) & (
        palette_C[None, :] < PAL_GREY_T
    )
    if saturated_to_grey.any():
        cost = np.where(saturated_to_grey, cost * (GREY_PENALTY**2), cost)

    # Block chromatic -> white
    chroma_to_white = (
        (source_C[:, None] > 0.04)
        & (palette_C[None, :] < PAL_GREY_T)
        & (palette_L[None, :] > WHITE_L_T)
    )
    if chroma_to_white.any():
        cost = np.where(chroma_to_white, cost * (WHITE_PENALTY**2), cost)

    # Near-white discourages cyan-ish substitutions; pulls toward white/v. light grey
    near_white_src = (source_L > NEAR_WHITE_L_T) & (source_C < NEAR_WHITE_C_T)
    if np.any(near_white_src):
        palette_cyan_band = np.abs(_wrap(palette_hue - CYAN_HUE_C)) <= CYAN_HUE_BW
        cost = np.where(
            near_white_src[:, None] & palette_cyan_band[None, :],
            cost * (NEAR_WHITE_CYAN_PEN**2),
            cost,
        )

        palette_white_like = (palette_C < 0.015) & (palette_L > 0.97)
        palette_vlight_grey = (palette_C < 0.030) & (palette_L > 0.92)

        cost = np.where(
            near_white_src[:, None] & palette_white_like[None, :],
            cost * NEAR_WHITE_WHITE_BONUS,
            cost,
        )
        cost = np.where(
            near_white_src[:, None] & palette_vlight_grey[None, :],
            cost * NEAR_WHITE_VLG_BONUS,
            cost,
        )

    # Brown guard for truly neutral sources
    neutral_src_mask = source_C < NEUTRAL_SRC_T
    palette_brown_band = (np.abs(_wrap(palette_hue - BROWN_HUE_C)) <= BROWN_HUE_BW) & (
        palette_C >= BROWN_CHROMA_MIN
    )
    if np.any(palette_brown_band) and np.any(neutral_src_mask):
        cost = np.where(
            neutral_src_mask[:, None] & palette_brown_band[None, :],
            cost * (BROWN_PENALTY**2),
            cost,
        )

    # Lightness soft bias (do not go darker if source is light) + hard light->deep guard
    soft_lightness_weight = np.clip(
        (source_L - LIGHT_SOFT_SRC) / max(LIGHT_SOFT_SPAN, 1e-6), 0.0, 1.0
    )[:, None]
    darker_than_source = np.clip(source_L[:, None] - palette_L[None, :], 0.0, 1.0)
    cost = cost + LIGHT_SOFT_W * (soft_lightness_weight * (darker_than_source**2))

    source_is_light = source_L > LIGHT_SRC_T
    palette_is_too_dark = palette_L[None, :] < (source_L[:, None] - LIGHT_DROP_L_T)
    hue_close_mask = (
        np.abs(_wrap(source_hue[:, None] - palette_hue[None, :])) <= HUE_CLOSE_T
    )
    chroma_close_mask = np.abs(source_C[:, None] - palette_C[None, :]) <= CLOSE_CHROMA_T
    forbid_light_to_deep = (
        source_is_light[:, None]
        & palette_is_too_dark
        & hue_close_mask
        & chroma_close_mask
    )
    if np.any(forbid_light_to_deep):
        cost = np.where(forbid_light_to_deep, cost * (LIGHT_DROP_PEN**2), cost)

    # Greens: discourage mapping to stone/grey; mild pull to olive/green
    source_is_clearly_green = (source_C >= GREEN_SRC_C_T) & (
        np.abs(_wrap(source_hue - GREEN_HUE_C)) <= GREEN_HUE_BW
    )
    if np.any(source_is_clearly_green):
        lightness_close_mask = np.abs(source_L[:, None] - palette_L[None, :]) <= 0.10
        palette_stone_band = (
            np.abs(_wrap(palette_hue - STONE_HUE_C)) <= STONE_HUE_BW
        ) | (palette_C < STONE_C_MAX)
        cost = np.where(
            source_is_clearly_green[:, None]
            & palette_stone_band[None, :]
            & lightness_close_mask,
            cost * (GREEN_TO_STONE_PEN**2),
            cost,
        )

        palette_olive_band = (
            np.abs(_wrap(palette_hue - OLIVE_HUE_C)) <= OLIVE_HUE_BW
        ) & (palette_C >= OLIVE_MIN_C)
        palette_green_band = np.abs(_wrap(palette_hue - GREEN_HUE_C)) <= GREEN_HUE_BW
        green_pull_mask = (
            source_is_clearly_green[:, None]
            & (palette_olive_band | palette_green_band)[None, :]
            & lightness_close_mask
        )
        cost = np.where(green_pull_mask, cost * OLIVE_BONUS, cost)

    # Low-chroma green rescue (e.g., #3a553b-like)
    source_is_lowC_green = (
        (source_C >= LOWC_GREEN_C_MIN)
        & (source_C <= LOWC_GREEN_C_MAX)
        & (np.abs(_wrap(source_hue - GREEN_HUE_C)) <= LOWC_GREEN_HUE_BW)
    )
    if np.any(source_is_lowC_green):
        lightness_close_mask = np.abs(source_L[:, None] - palette_L[None, :]) <= 0.12
        palette_olive_band = (
            np.abs(_wrap(palette_hue - OLIVE_HUE_C)) <= OLIVE_HUE_BW
        ) & (palette_C >= 0.08)
        palette_green_band = np.abs(_wrap(palette_hue - GREEN_HUE_C)) <= GREEN_HUE_BW
        palette_stoneish = (
            np.abs(_wrap(palette_hue - STONE_HUE_C)) <= STONE_HUE_BW
        ) | (palette_C < 0.10)

        cost = np.where(
            source_is_lowC_green[:, None]
            & (palette_olive_band | palette_green_band)[None, :]
            & lightness_close_mask,
            cost * LOWC_GREEN_BONUS,
            cost,
        )
        cost = np.where(
            source_is_lowC_green[:, None]
            & palette_stoneish[None, :]
            & lightness_close_mask,
            cost * (LOWC_GREEN_STONE_PEN**2),
            cost,
        )

    # Light cyans: prefer blue-cyan over teal if not darker
    source_is_light_cyan = (
        (source_C >= CYAN_SRC_C_T)
        & (source_L >= CYAN_SRC_L_T)
        & (np.abs(_wrap(source_hue - CYAN_HUE_BAND_C)) <= CYAN_HUE_BAND_BW)
    )
    if np.any(source_is_light_cyan):
        palette_teal_band = np.abs(_wrap(palette_hue - TEAL_HUE_C)) <= TEAL_HUE_BW
        palette_bluecyan_band = (
            np.abs(_wrap(palette_hue - BLUECYAN_HUE_C)) <= BLUECYAN_HUE_BW
        )
        target_not_darker = palette_L[None, :] >= (
            source_L[:, None] - CYAN_NOT_DARKER_DL
        )

        cost = np.where(
            source_is_light_cyan[:, None] & palette_teal_band[None, :],
            cost * (CYAN_TO_TEAL_PEN**2),
            cost,
        )
        cost = np.where(
            source_is_light_cyan[:, None]
            & palette_bluecyan_band[None, :]
            & target_not_darker,
            cost * CYAN_TO_BLUE_BONUS,
            cost,
        )

    return cost


def build_candidate_lists(cost: np.ndarray) -> List[np.ndarray]:
    """
    Per unique source colour, keep palette candidates within GOOD_ENOUGH_RATIO of the best,
    up to MAX_CANDIDATES per row. Returns list of 1D index arrays.
    """
    num_src = cost.shape[0]
    best_per_row = cost.min(axis=1)
    palette_order = np.argsort(cost, axis=1)
    candidate_lists: List[np.ndarray] = []
    for i in range(num_src):
        row_indices = palette_order[i]
        row_indices = row_indices[
            cost[i, row_indices] <= best_per_row[i] * GOOD_ENOUGH_RATIO
        ]
        if row_indices.size == 0:
            row_indices = palette_order[i, :1]
        candidate_lists.append(row_indices[:MAX_CANDIDATES])
    return candidate_lists


def match_top_sources_to_unique_palette(
    top_src_indices: np.ndarray, candidate_lists: List[np.ndarray], num_palette: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight maximum bipartite matching for the most frequent source colours.

    Returns:
      palette_owner: length num_palette, value is source index that owns that palette,
                     or -1 if unowned.
      chosen_palette_for_source: length num_sources (len(candidate_lists)),
                     value is palette index chosen for that source, or -1 if none.
    """
    palette_owner = np.full(num_palette, -1, dtype=np.int32)
    chosen_palette_for_source = np.full(len(candidate_lists), -1, dtype=np.int32)

    def try_assign(src_i: int, seen_palette: np.ndarray) -> bool:
        for pal_j in candidate_lists[src_i]:
            pal_j = int(pal_j)
            if seen_palette[pal_j]:
                continue
            seen_palette[pal_j] = True
            current_owner = int(palette_owner[pal_j])
            if current_owner == -1 or try_assign(current_owner, seen_palette):
                palette_owner[pal_j] = int(src_i)
                chosen_palette_for_source[int(src_i)] = pal_j
                return True
        return False

    for src_i_val in top_src_indices:
        src_i = int(src_i_val)
        seen = np.zeros(num_palette, dtype=bool)
        try_assign(src_i, seen)

    return palette_owner, chosen_palette_for_source


def map_visible_pixels_global(
    rgb_vis_flat: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """
    Global mapping for visible (non-transparent) pixels:
      1) unique colours -> cost table -> candidate lists,
      2) match most frequent uniques to distinct palette colours,
      3) greedy fill with strict "do not reuse if close alt exists".
    """
    unique_colors, inverse_index, counts = np.unique(
        rgb_vis_flat, axis=0, return_inverse=True, return_counts=True
    )
    num_unique_colors, num_palette_colors = unique_colors.shape[0], palette.shape[0]

    cost = compute_cost_matrix(unique_colors, palette)
    candidate_lists = build_candidate_lists(cost)

    # Process most frequent unique colours first
    src_order_by_count = np.argsort(-counts)
    top_match_count = min(num_palette_colors, num_unique_colors)
    top_src_indices = src_order_by_count[:top_match_count]

    palette_owner, chosen_palette_for_source = match_top_sources_to_unique_palette(
        top_src_indices, candidate_lists, num_palette_colors
    )
    used_palette_indices = set(int(j) for j in palette_owner if int(j) != -1)

    chosen_palette_index_for_unique = np.full(num_unique_colors, -1, dtype=np.int32)

    # Assign matched top colours
    for src_i_val in top_src_indices:
        src_i = int(src_i_val)
        pal_j = int(chosen_palette_for_source[src_i])
        if pal_j != -1:
            chosen_palette_index_for_unique[src_i] = pal_j
            used_palette_indices.add(pal_j)

    # Greedy for remaining with stricter no-reuse and "good enough" guards
    for src_i_val in src_order_by_count:
        src_i = int(src_i_val)
        if chosen_palette_index_for_unique[src_i] != -1:
            continue

        candidates_for_i = candidate_lists[src_i]
        if candidates_for_i.size == 0:
            candidates_for_i = np.argsort(cost[src_i])[:1]

        best_idx = int(candidates_for_i[0])
        best_cost = float(cost[src_i, best_idx])

        # If best already used, try a close free alternative (SHARE_GUARD_RATIO)
        if best_idx in used_palette_indices:
            near_free_alt = None
            for pal_j in candidates_for_i[1:]:
                pal_j = int(pal_j)
                if (
                    pal_j not in used_palette_indices
                    and cost[src_i, pal_j] <= best_cost * SHARE_GUARD_RATIO
                ):
                    near_free_alt = pal_j
                    break
            if near_free_alt is not None:
                chosen_palette_index_for_unique[src_i] = near_free_alt
                used_palette_indices.add(near_free_alt)
                continue

        # Avoid reusing any used palette if a free alt is within STRICT_REUSE_RATIO
        free_alt = None
        for pal_j in candidates_for_i:
            pal_j = int(pal_j)
            if pal_j in used_palette_indices:
                continue
            if cost[src_i, pal_j] <= best_cost * STRICT_REUSE_RATIO:
                free_alt = pal_j
                break
        if free_alt is not None:
            chosen_palette_index_for_unique[src_i] = free_alt
            used_palette_indices.add(free_alt)
            continue

        # Fall back to best (may reuse)
        chosen_palette_index_for_unique[src_i] = best_idx
        used_palette_indices.add(best_idx)

    return palette[chosen_palette_index_for_unique][inverse_index].astype(np.uint8)


# ------------- Shared helpers -------------
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


# ------------- Pipeline -------------
def process(
    input_path: str, output_path: str | None, target_height: int, mode: str
) -> None:
    palette = build_palette()

    # Load & convert (preserve transparency; handles PNG P/tRNS as well)
    im_in = Image.open(input_path)
    im_rgba = im_in.convert("RGBA")
    im_small = scale_to_height(im_rgba, target_height)

    arr = np.array(im_small, dtype=np.uint8)
    h, w = arr.shape[:2]

    rgb = arr[:, :, :3].reshape(-1, 3)
    a = arr[:, :, 3].reshape(-1)
    vis_mask = a > 0

    # Mode selection
    chosen_mode = mode
    if mode == "auto":
        # Heuristic: many unique visible colours -> photo; otherwise pixel
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

    out = np.dstack([rgb_out.reshape(h, w, 3), a.reshape(h, w)]).astype(np.uint8)
    out_img = Image.fromarray(out)

    # Output path
    if output_path is None or output_path == "":
        out_path = Path(input_path).with_name(f"{Path(input_path).stem}_wplace.png")
    else:
        out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", optimize=False)
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
