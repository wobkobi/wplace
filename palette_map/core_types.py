# palette_map/core_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# ---- basic aliases
RGBTuple = Tuple[int, int, int]

U8Image = NDArray[np.uint8]  # e.g., (H, W, 3)
U8Mask = NDArray[np.uint8]  # e.g., (H, W)
Lab = NDArray[np.float32]  # (..., 3) in CIE Lab
Lch = NDArray[np.float32]  # (..., 3) in CIE LCh

# ---- candidate / assignment structures
Cost = float
Candidate = Tuple[Cost, int]  # (cost, palette_index)
CandidateRow = List[Candidate]
CandidateTable = List[CandidateRow]

Assignment = Dict[int, int]  # source_index -> palette_index
CostLookup = Dict[Tuple[int, int], float]  # (source_index, palette_index) -> cost
NameMap = Dict[RGBTuple, str]


# ---- core items
@dataclass(frozen=True)
class PaletteItem:
    """A palette entry with precomputed Lab/LCh."""

    rgb: RGBTuple
    name: str
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


@dataclass(frozen=True)
class SourceItem:
    """A unique source colour with count and precomputed Lab/LCh."""

    rgb: RGBTuple
    count: int
    lab: Lab  # shape (3,)
    lch: Lch  # shape (3,)


__all__ = [
    "RGBTuple",
    "U8Image",
    "U8Mask",
    "Lab",
    "Lch",
    "Cost",
    "Candidate",
    "CandidateRow",
    "CandidateTable",
    "Assignment",
    "CostLookup",
    "NameMap",
    "PaletteItem",
    "SourceItem",
]
