# palette_map/palette_data.py
from __future__ import annotations

"""
Palette definitions and builders.

Exports:
  PALETTE: list[tuple[str, str]]  # [(hex, name), ...]
  build_palette(hex_name_pairs=PALETTE)
    -> (items: list[PaletteItem],
        name_of: dict[RGBTuple, str],
        pal_lab: Lab,
        pal_lch: Lch)
"""

from typing import Dict, List, Tuple

import numpy as np

from .core_types import PaletteItem, Lab, Lch, RGBTuple, hex_to_rgb
from .colour_convert import rgb_to_lab, lab_to_lch


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


def build_palette(
    hex_name_pairs: List[Tuple[str, str]] = PALETTE,
) -> Tuple[List[PaletteItem], Dict[RGBTuple, str], Lab, Lch]:
    """
    Convert a list of (hex, name) into:
      items: list[PaletteItem] with rgb, name, lab, lch
      name_of: dict mapping RGBTuple -> name
      pal_lab: float32 array [P,3]
      pal_lch: float32 array [P,3]
    """
    rgbs_u8 = np.array([hex_to_rgb(hx) for hx, _ in hex_name_pairs], dtype=np.uint8)

    pal_lab: Lab = (
        rgb_to_lab(rgbs_u8.astype(np.float32))
        .reshape(-1, 3)
        .astype(np.float32, copy=False)
    )
    pal_lch: Lch = lab_to_lch(pal_lab).reshape(-1, 3).astype(np.float32, copy=False)

    items: List[PaletteItem] = []
    name_of: Dict[RGBTuple, str] = {}

    for i, (_hx, name) in enumerate(hex_name_pairs):
        rgb_tuple: RGBTuple = (
            int(rgbs_u8[i, 0]),
            int(rgbs_u8[i, 1]),
            int(rgbs_u8[i, 2]),
        )
        items.append(
            PaletteItem(
                rgb=rgb_tuple,
                name=name,
                lab=pal_lab[i].copy(),
                lch=pal_lch[i].copy(),
            )
        )
        name_of[rgb_tuple] = name

    return items, name_of, pal_lab, pal_lch


__all__ = ["PALETTE", "build_palette"]
