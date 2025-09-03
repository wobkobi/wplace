# palette_map/constants.py
"""
Global palette and tunables used across the project.

- PALETTE, GREY_HEXES
- Photo mode constants (PHOTO_*)
- Pixel mode constants (blue/teal protection, grey spreading, warm/cool nudges, etc.)
"""
from __future__ import annotations

from typing import List, Tuple

# =========================
# User Palette (hex, name)
# =========================
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

GREY_HEXES = ["#000000", "#3c3c3c", "#787878", "#aaaaaa", "#d2d2d2", "#ffffff"]

# ==================
# Photo mode (PHOTO)
# ==================
W_LOCAL_BASE: float = 0.18
GRAD_K: float = 12.0
W_HUE: float = 0.06
W_CHROMA: float = 0.05
W_WASH: float = 0.65
L_WASH_THRESH: float = 4.0
W_DESAT: float = 0.05
C_DESAT_THRESH: float = 10.0
BLUR_RADIUS: int = 2
W_CONTRAST: float = 0.15
CONTRAST_MARGIN: float = 0.8
PHOTO_NEUTRAL_C_MAX: float = 5.0
NEUTRAL_EST_C: float = 8.0
NEUTRAL_EST_LMIN: float = 20.0
NEUTRAL_EST_LMAX: float = 80.0
AB_BIAS_GAIN: float = 0.8
NEAR_NEUTRAL_C: float = 4.0
NEAR_BLACK_L: float = 32.0
NEAR_WHITE_L: float = 86.0
SHADOW_L: float = 38.0
W_SHADOW_HUE: float = 0.12
W_SHADOW_CHROMA_UP: float = 0.10
EL_CLAMP: float = 6.0
Q_L: float = 0.5
Q_A: float = 1.0
Q_B: float = 1.0
Q_LOC: float = 1.0
Q_W: float = 0.05
PHOTO_TOPK: int = 6

# =======================
# Pixel mode (PIXEL_* )
# =======================
W_HUE_PX: float = 1.0
HUE_PX_SCALE: float = 10.0
L_DARK: float = 35.0
L_DARK_MAX_UP: float = 10.0
W_L_DARK: float = 0.6
L_UP_SOFT_MAX: float = 12.0
W_L_UP_SOFT: float = 0.25
W_BRIGHT_PX: float = 0.12

WHITE_AVOID_UNLESS_L: float = 92.0
WHITE_AVOID_MIN_SC: float = 2.0
WHITE_AVOID_PEN: float = 6.0

CHROMA_FLOOR_ABS: float = 6.0
CHROMA_FLOOR_REL: float = 0.45
W_CHROMA_FLOOR: float = 0.6
CHROMA_SRC_MIN: float = 3.0
C_STRETCH_THRESH: float = 14.0
W_CHROMA_STRETCH_LOW: float = 0.25
W_CHROMA_STRETCH: float = 0.5

NEUTRAL_C_MAX: float = 3.5
NEUTRAL_SRC_C_MAX: float = 4.5
NO_COLLAPSE_MIN_SC: float = 7.5
NO_COLLAPSE_TO_PURE_NEUTRAL_C: float = 1.5
NEUTRAL_PENALTY: float = 12.0
SAT_TO_NEUTRAL_MULT: float = 1.2
NEUTRAL_BONUS_C: float = 1.2
NEUTRAL_BONUS: float = 0.8
NEUTRAL_REASSIGN_TOL: float = 2.5

GREY_SRC_CUTOFF: float = 5.0
MUTED_TGT_MAXC: float = 14.0
PEN_MUTED_TO_VIVID: float = 5.0
BPC_HUE_MIN: float = 200.0
BPC_HUE_MAX: float = 300.0
BPC_LOW_C_CAP: float = 6.0
PEN_MUTED_TO_BPC: float = 4.5
WARM_LOW_C_CAP: float = 6.0
PEN_MUTED_TO_WARM: float = 4.0

PIXEL_SAT_C: float = 22.0
BLUE_BAND_MIN: float = 190.0
BLUE_BAND_MAX: float = 235.0
W_BLUE_KEEP_NEUTRAL: float = 14.0
W_BLUE_KEEP_LOW_C: float = 6.0
H_RING: float = 28.0
W_HUE_RING: float = 6.0

SLATE_HUE_MIN: float = 190.0
SLATE_HUE_MAX: float = 240.0
SLATE_BONUS_C_MAX: float = 8.0
SLATE_BONUS: float = 3.0

COOL_SLATE_HUE_MIN: float = 200.0
COOL_SLATE_HUE_MAX: float = 250.0
COOL_SLATE_MIN_SC: float = 6.0
COOL_SLATE_SLATE_BONUS_C_MAX: float = 10.0
COOL_SLATE_SLATE_BONUS: float = 3.0
COOL_SLATE_NEUTRAL_PEN: float = 9.0

WARM_HUE_MIN: float = 20.0
WARM_HUE_MAX: float = 70.0
WARM_DARK_L_MAX: float = 52.0
WARM_DARK_MIN_SC: float = 5.5
PEN_WARM_DARK_TO_NEUTRAL: float = 12.0
PEN_WARM_DARK_TO_SLATE: float = 7.0
WARM_DARK_DH_SOFT: float = 18.0
W_WARM_DARK_DH: float = 0.18
WARM_TARGET_MIN_C: float = 8.5
W_CHROMA_WARM_DARK: float = 2.2

GREY_SPREAD_EXTRA_ALLOW: float = 8.0
GREY_SPREAD_WEIGHT_L: float = 1.2
LOWCHROMA_ALT_MAXC: float = 12.0
COOL_LOWCHROMA_TARGET_C_MIN: float = 4.0
COOL_NEUTRAL_BLOCK_SC_MIN: float = 6.0
COOL_LIGHT_REASSIGN_TOL: float = 10.0
DARK_NEUTRAL_TO_SLATE_TOL: float = 7.0
WARM_DARK_SPREAD_TOL: float = 12.0
BLUE_ATTRACT: float = 3.8
BLUE_TO_SLATE_PEN: float = 7.0
GREY_SPREAD_MAX_L_JUMP: float = 8.0
GREY_SPREAD_COST_DELTA: float = 1.5

TEAL_BAND_MIN: float = 175.0
TEAL_BAND_MAX: float = 205.0
BLUE_PROTECT_MIN_C: float = 4.0
BLUE_KEEP_RATIO: float = 0.55
BLUE_KEEP_MIN_RATIO: float = BLUE_KEEP_RATIO
PEN_BLUE_TO_NEUTRAL: float = 80.0
PEN_BLUE_TO_SLATE: float = 40.0
W_BLUE_CHROMA_FLOOR: float = 1.8
W_CHROMA_FLOOR_BLUE_MULT: float = 2.0
COOL_LIGHT_MIN_L: float = 58.0

__all__ = [
    "PALETTE",
    "GREY_HEXES",
    "W_LOCAL_BASE",
    "GRAD_K",
    "W_HUE",
    "W_CHROMA",
    "W_WASH",
    "L_WASH_THRESH",
    "W_DESAT",
    "C_DESAT_THRESH",
    "BLUR_RADIUS",
    "W_CONTRAST",
    "CONTRAST_MARGIN",
    "PHOTO_NEUTRAL_C_MAX",
    "NEUTRAL_EST_C",
    "NEUTRAL_EST_LMIN",
    "NEUTRAL_EST_LMAX",
    "AB_BIAS_GAIN",
    "NEAR_NEUTRAL_C",
    "NEAR_BLACK_L",
    "NEAR_WHITE_L",
    "SHADOW_L",
    "W_SHADOW_HUE",
    "W_SHADOW_CHROMA_UP",
    "EL_CLAMP",
    "Q_L",
    "Q_A",
    "Q_B",
    "Q_LOC",
    "Q_W",
    "PHOTO_TOPK",
    "W_HUE_PX",
    "HUE_PX_SCALE",
    "L_DARK",
    "L_DARK_MAX_UP",
    "W_L_DARK",
    "L_UP_SOFT_MAX",
    "W_L_UP_SOFT",
    "W_BRIGHT_PX",
    "WHITE_AVOID_UNLESS_L",
    "WHITE_AVOID_MIN_SC",
    "WHITE_AVOID_PEN",
    "CHROMA_FLOOR_ABS",
    "CHROMA_FLOOR_REL",
    "W_CHROMA_FLOOR",
    "CHROMA_SRC_MIN",
    "C_STRETCH_THRESH",
    "W_CHROMA_STRETCH_LOW",
    "W_CHROMA_STRETCH",
    "NEUTRAL_C_MAX",
    "NEUTRAL_SRC_C_MAX",
    "NO_COLLAPSE_MIN_SC",
    "NO_COLLAPSE_TO_PURE_NEUTRAL_C",
    "NEUTRAL_PENALTY",
    "SAT_TO_NEUTRAL_MULT",
    "NEUTRAL_BONUS_C",
    "NEUTRAL_BONUS",
    "NEUTRAL_REASSIGN_TOL",
    "GREY_SRC_CUTOFF",
    "MUTED_TGT_MAXC",
    "PEN_MUTED_TO_VIVID",
    "BPC_HUE_MIN",
    "BPC_HUE_MAX",
    "BPC_LOW_C_CAP",
    "PEN_MUTED_TO_BPC",
    "WARM_LOW_C_CAP",
    "PEN_MUTED_TO_WARM",
    "PIXEL_SAT_C",
    "BLUE_BAND_MIN",
    "BLUE_BAND_MAX",
    "W_BLUE_KEEP_NEUTRAL",
    "W_BLUE_KEEP_LOW_C",
    "H_RING",
    "W_HUE_RING",
    "SLATE_HUE_MIN",
    "SLATE_HUE_MAX",
    "SLATE_BONUS_C_MAX",
    "SLATE_BONUS",
    "COOL_SLATE_HUE_MIN",
    "COOL_SLATE_HUE_MAX",
    "COOL_SLATE_MIN_SC",
    "COOL_SLATE_SLATE_BONUS_C_MAX",
    "COOL_SLATE_SLATE_BONUS",
    "COOL_SLATE_NEUTRAL_PEN",
    "WARM_HUE_MIN",
    "WARM_HUE_MAX",
    "WARM_DARK_L_MAX",
    "WARM_DARK_MIN_SC",
    "PEN_WARM_DARK_TO_NEUTRAL",
    "PEN_WARM_DARK_TO_SLATE",
    "WARM_DARK_DH_SOFT",
    "W_WARM_DARK_DH",
    "WARM_TARGET_MIN_C",
    "W_CHROMA_WARM_DARK",
    "GREY_SPREAD_EXTRA_ALLOW",
    "GREY_SPREAD_WEIGHT_L",
    "LOWCHROMA_ALT_MAXC",
    "COOL_LOWCHROMA_TARGET_C_MIN",
    "COOL_NEUTRAL_BLOCK_SC_MIN",
    "COOL_LIGHT_REASSIGN_TOL",
    "DARK_NEUTRAL_TO_SLATE_TOL",
    "WARM_DARK_SPREAD_TOL",
    "BLUE_ATTRACT",
    "BLUE_TO_SLATE_PEN",
    "GREY_SPREAD_MAX_L_JUMP",
    "GREY_SPREAD_COST_DELTA",
    "TEAL_BAND_MIN",
    "TEAL_BAND_MAX",
    "BLUE_PROTECT_MIN_C",
    "BLUE_KEEP_RATIO",
    "BLUE_KEEP_MIN_RATIO",
    "PEN_BLUE_TO_NEUTRAL",
    "PEN_BLUE_TO_SLATE",
    "W_BLUE_CHROMA_FLOOR",
    "W_CHROMA_FLOOR_BLUE_MULT",
    "COOL_LIGHT_MIN_L",
]
