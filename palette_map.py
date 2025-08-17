#!/usr/bin/env python3
"""
palette_map_dual.py  — stricter uniqueness

Modes:
- pixel : Global OKLab/OKLCh distance + matching for top colours, then greedy with
          stricter no-reuse policy for the rest.
- photo : Classic OKLab nearest with mild anti-grey bias.

Alpha preserved. No upscaling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
import argparse
import numpy as np
from PIL import Image

# ---------- Palette ----------
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
PALETTE_HEX: tuple[str, ...] = tuple(h for h, _, _ in PALETTE_ENTRIES)
HEX_TO_NAME: Dict[str, str] = {h.lower(): n for h, n, _ in PALETTE_ENTRIES}
HEX_TO_TIER: Dict[str, str] = {h.lower(): t for h, _, t in PALETTE_ENTRIES}


# ---------- Utils ----------
def parse_hex(code: str) -> Tuple[int, int, int]:
    s = code[1:] if code.startswith("#") else code
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def build_palette() -> np.ndarray:
    cols = np.array([parse_hex(h) for h in PALETTE_HEX], dtype=np.uint8)
    _, idx = np.unique(cols.view([("", cols.dtype)] * cols.shape[1]), return_index=True)
    return cols[np.sort(idx)]


def _srgb_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
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


# ===== Photo mode (nearest with mild anti-grey) =====
PHOTO_SRC_SAT_T = 0.06
PHOTO_PAL_GREY_T = 0.03
PHOTO_GREY_PENALTY = 1.7


def _nearest_indices_oklab_photo(
    src_lab: np.ndarray,
    pal_lab: np.ndarray,
    src_chroma: np.ndarray,
    pal_chroma: np.ndarray,
) -> np.ndarray:
    diff = src_lab[:, None, :] - pal_lab[None, :, :]
    d2 = np.einsum("knc,knc->kn", diff, diff, optimize=True)
    mask = (src_chroma[:, None] > PHOTO_SRC_SAT_T) & (
        pal_chroma[None, :] < PHOTO_PAL_GREY_T
    )
    if mask.any():
        d2 = np.where(mask, d2 * (PHOTO_GREY_PENALTY**2), d2)
    return np.argmin(d2, axis=1)


def palette_map_array_photo(rgb_flat: np.ndarray, palette: np.ndarray) -> np.ndarray:
    unique_rgb, inv = np.unique(rgb_flat, axis=0, return_inverse=True)
    pal_lab = _srgb_to_oklab(palette)
    pal_C = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    src_lab = _srgb_to_oklab(unique_rgb)
    src_C = np.hypot(src_lab[:, 1], src_lab[:, 2])
    idx = _nearest_indices_oklab_photo(src_lab, pal_lab, src_C, pal_C)
    return palette[idx][inv].astype(np.uint8)


# ===== Pixel mode (global matching + stricter no-reuse) =====
HUE_WEIGHT_BASE = 0.45
CHROMA_FLOOR = 0.02
CHROMA_RANGE = 0.12
SRC_SAT_T = 0.04
PAL_GREY_T = 0.05
GREY_PENALTY = 2.0
WHITE_L_T = 0.92
WHITE_PENALTY = 3.0

GOOD_ENOUGH_RATIO = 1.05  # stricter candidate sets
MAX_CANDIDATES = 20
SHARE_GUARD_RATIO = 1.10  # if best is owned by a top colour, need alt within 10%
STRICT_REUSE_RATIO = 1.12  # avoid reusing any already-used slot if free alt within 12%


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_d2(src_rgb: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    pal = _srgb_to_oklab(pal_rgb)
    src = _srgb_to_oklab(src_rgb)
    pal_L, pal_C, pal_h = (
        pal[:, 0],
        np.hypot(pal[:, 1], pal[:, 2]),
        np.arctan2(pal[:, 2], pal[:, 1]),
    )
    src_L, src_C, src_h = (
        src[:, 0],
        np.hypot(src[:, 1], src[:, 2]),
        np.arctan2(src[:, 2], src[:, 1]),
    )

    dL2 = (src_L[:, None] - pal_L[None, :]) ** 2
    dC2 = (src_C[:, None] - pal_C[None, :]) ** 2
    dh = _wrap(src_h[:, None] - pal_h[None, :])
    hue_w = (
        HUE_WEIGHT_BASE
        * np.clip((src_C - CHROMA_FLOOR) / CHROMA_RANGE, 0.0, 1.0)[:, None]
    )
    d2 = 0.6 * dL2 + 1.0 * dC2 + hue_w * (dh**2)

    pen_grey = (src_C[:, None] > SRC_SAT_T) & (pal_C[None, :] < PAL_GREY_T)
    if pen_grey.any():
        d2 = np.where(pen_grey, d2 * (GREY_PENALTY**2), d2)
    pen_white = (
        (src_C[:, None] > 0.04)
        & (pal_C[None, :] < PAL_GREY_T)
        & (pal_L[None, :] > WHITE_L_T)
    )
    if pen_white.any():
        d2 = np.where(pen_white, d2 * (WHITE_PENALTY**2), d2)
    return d2


def _build_cands(d2: np.ndarray) -> List[np.ndarray]:
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
    uniq, inv, counts = np.unique(
        rgb_vis_flat, axis=0, return_inverse=True, return_counts=True
    )
    K, N = uniq.shape[0], palette.shape[0]
    d2 = compute_d2(uniq, palette)
    cands = _build_cands(d2)

    order_src = np.argsort(-counts)
    topM = min(N, K)
    top_idx = order_src[:topM]

    owner, chosen = _max_match(top_idx, cands, N)
    used = set(int(j) for j in owner if int(j) != -1)

    choice = np.full(K, -1, dtype=np.int32)

    # Assign matched top colours
    for i_val in top_idx:
        ii = int(i_val)
        cj = int(chosen[ii])
        if cj != -1:
            choice[ii] = cj

    # Greedy for remaining with stricter no-reuse
    for i_val in order_src:
        ii = int(i_val)
        if choice[ii] != -1:  # already set by matching
            continue

        row = cands[ii]  # cands is a List[np.ndarray]
        if row.size == 0:
            row = np.argsort(d2[ii])[:1]

        best = int(row[0])
        db = float(d2[ii, best])

        # If best owned by a top colour, try close alternative
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

        # Avoid reusing any used slot if a free alt is within STRICT_REUSE_RATIO
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

        # Fall back to best (may reuse)
        choice[ii] = best
        used.add(best)

    return palette[choice][inv].astype(np.uint8)


# ---------- Shared ----------
def scale_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if target_h <= 0:
        return img
    w, h = img.width, img.height
    if target_h >= h:
        return img
    nw = max(1, round(w * target_h / h))
    return img.resize((nw, target_h), resample=Image.Resampling.BOX)


def report_usage(img: Image.Image) -> None:
    arr = np.array(img, dtype=np.uint8)
    if img.mode == "RGBA":
        m = arr[:, :, 3] > 0
        if not m.any():
            print("Colours used: none (fully transparent)")
            return
        cols = arr[:, :, :3][m].reshape(-1, 3)
    else:
        cols = arr.reshape(-1, 3)
    uniq, cnt = np.unique(cols, axis=0, return_counts=True)

    def _hx(r: np.ndarray) -> str:
        return f"#{int(r[0]):02x}{int(r[1]):02x}{int(r[2]):02x}"

    items = []
    for c, n in zip(uniq, cnt):
        h = _hx(c).lower()
        items.append((n, HEX_TO_NAME.get(h, "Unknown"), HEX_TO_TIER.get(h, "Premium")))
    items.sort(key=lambda t: t[0], reverse=True)
    print("Colours used:")
    for n, name, tier in items:
        print(f"{name} [{tier}]: {n}")


# ---------- Pipeline ----------
def process(
    input_path: str,
    output_path: str,
    target_height: int,
    mode: str,
    auto_threshold: int,
) -> None:
    palette = build_palette()
    im = Image.open(input_path).convert("RGBA")
    im = scale_to_height(im, target_height)
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3].reshape(-1, 3)
    a = arr[:, :, 3].reshape(-1)
    vis = a > 0

    chosen = mode
    if mode == "auto":
        uniq_vis = np.unique(rgb[vis], axis=0).shape[0] if vis.any() else 0
        chosen = "photo" if uniq_vis > auto_threshold else "pixel"

    rgb_out = rgb.copy()
    if vis.any():
        mapped = (
            map_visible_pixels_global(rgb[vis], palette)
            if chosen == "pixel"
            else palette_map_array_photo(rgb[vis], palette)
        )
        rgb_out[vis] = mapped

    out = np.dstack([rgb_out.reshape(h, w, 3), a.reshape(h, w)]).astype(np.uint8)
    Image.fromarray(out).save(output_path, format="PNG", optimize=False)
    print(f"Mode: {chosen}")
    print(f"Wrote {output_path} | size={w}x{h} | palette_size={len(PALETTE_ENTRIES)}\n")
    report_usage(Image.fromarray(out))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dual-mode palette mapper with stricter uniqueness."
    )
    ap.add_argument("input", type=str)
    ap.add_argument("output", type=str)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--mode", choices=["auto", "pixel", "photo"], default="auto")
    ap.add_argument("--auto-threshold", type=int, default=4096)
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process(args.input, args.output, args.height, args.mode, args.auto_threshold)


if __name__ == "__main__":
    main()
