#!/usr/bin/env python3
# palette_map_fixed.py
# Map every pixel to the nearest colour from a fixed palette, then enforce membership.

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import argparse
import numpy as np
from PIL import Image

# Fixed palette (hex). Order preserved. Duplicates removed automatically.
PALETTE_HEX: tuple[str, ...] = (
    "#000000",
    "#3c3c3c",
    "#787878",
    "#aaaaaa",
    "#d2d2d2",
    "#ffffff",
    "#600018",
    "#a50e1e",
    "#ed1c24",
    "#fa8072",
    "#e45c1a",
    "#ff7f27",
    "#f6aa09",
    "#f9dd3b",
    "#fffabc",
    "#9c8431",
    "#c5ad31",
    "#e8d45f",
    "#4a6b3a",
    "#5a944a",
    "#84c573",
    "#0eb968",
    "#13e67b",
    "#87ff5e",
    "#0c816e",
    "#10aea6",
    "#13e1be",
    "#0f799f",
    "#60f7f2",
    "#bbfaf2",
    "#28509e",
    "#4093e4",
    "#7dc7ff",
    "#4d31b8",
    "#6b50f6",
    "#99b1fb",
    "#4a4284",
    "#7a71c4",
    "#b5aef1",
    "#780c99",
    "#aa38b9",
    "#e09ff9",
    "#cb007a",
    "#ec1f80",
    "#f38da9",
    "#9b5249",
    "#d18078",
    "#fab6a4",
    "#684634",
    "#95682a",
    "#dba463",
    "#7b6352",
    "#9c846b",
    "#d6b594",
    "#d18051",
    "#f8b277",
    "#ffc5a5",
    "#6d643f",
    "#948c6b",
    "#cdc59e",
    "#333941",
    "#6d758d",
    "#b3b9d1",
)


def parse_hex(code: str) -> Tuple[int, int, int]:
    s = code[1:] if code.startswith("#") else code
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def build_palette() -> np.ndarray:
    cols = np.array([parse_hex(h) for h in PALETTE_HEX], dtype=np.uint8)  # (N,3)
    # Deduplicate while preserving order
    _, idx = np.unique(cols.view([("", cols.dtype)] * cols.shape[1]), return_index=True)
    return cols[np.sort(idx)]


def map_unique_colours_to_palette(
    unique_rgb: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """
    unique_rgb: (K,3) uint8 unique colours from the image
    palette:    (N,3) uint8 fixed palette
    Returns indices (K,) of the nearest palette colour for each unique colour.
    Exhaustive comparison over all palette colours.
    """
    urgb32 = unique_rgb.astype(np.int32)
    pal32 = palette.astype(np.int32)
    diff = urgb32[:, None, :] - pal32[None, :, :]  # (K,N,3)
    d2 = np.sum(diff * diff, axis=2, dtype=np.int64)  # (K,N)
    return np.argmin(d2, axis=1)


def palette_map_array(rgb_flat: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map flat (H*W,3) rgb to nearest palette colours using a unique-colour table."""
    unique_rgb, inverse = np.unique(rgb_flat, axis=0, return_inverse=True)
    nearest_idx = map_unique_colours_to_palette(unique_rgb, palette)
    return palette[nearest_idx][inverse].astype(np.uint8)


def remove_partial_alpha(alpha_flat: np.ndarray) -> np.ndarray:
    """Binary alpha: >0 -> 255, else 0."""
    return np.where(alpha_flat > 0, 255, 0).astype(np.uint8)


def scale_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Downsize to target_h preserving aspect ratio. No upscaling."""
    if target_h <= 0:
        return img
    w, h = img.width, img.height
    if target_h >= h:  # no upscale
        return img
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), resample=Image.Resampling.BOX)


def process(input_path: str, output_path: str, target_height: int) -> None:
    palette = build_palette()

    im0 = Image.open(input_path)
    has_alpha = im0.mode in ("LA", "RGBA")
    im0 = im0.convert("RGBA") if has_alpha else im0.convert("RGB")

    arr0 = np.array(im0, dtype=np.uint8)
    h0, w0 = arr0.shape[0], arr0.shape[1]

    if has_alpha:
        rgb0 = arr0[:, :, :3].reshape(-1, 3)
        a0 = remove_partial_alpha(arr0[:, :, 3].reshape(-1))
    else:
        rgb0 = arr0.reshape(-1, 3)

    # Pass 1: snap to palette
    rgb1 = palette_map_array(rgb0, palette)
    if has_alpha:
        im1 = Image.fromarray(np.dstack([rgb1.reshape(h0, w0, 3), a0.reshape(h0, w0)]))
    else:
        im1 = Image.fromarray(rgb1.reshape(h0, w0, 3))

    # Resize to desired pixel-art height
    im2 = scale_to_height(im1, target_height)

    # Pass 2: snap again after resampling
    arr2 = np.array(im2, dtype=np.uint8)
    h2, w2 = arr2.shape[0], arr2.shape[1]
    if has_alpha:
        rgb2 = arr2[:, :, :3].reshape(-1, 3)
        a2 = remove_partial_alpha(arr2[:, :, 3].reshape(-1))
    else:
        rgb2 = arr2.reshape(-1, 3)

    rgb3 = palette_map_array(rgb2, palette)
    if has_alpha:
        out = np.dstack([rgb3.reshape(h2, w2, 3), a2.reshape(h2, w2)]).astype(np.uint8)
    else:
        out = rgb3.reshape(h2, w2, 3).astype(np.uint8)

    Image.fromarray(out).save(output_path, format="PNG", optimize=False)
    print(f"Wrote {output_path} | size={w2}x{h2} | palette_size={len(palette)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Palette map → downscale to target height → palette map again."
    )
    ap.add_argument("input", type=str)
    ap.add_argument("output", type=str)
    ap.add_argument("--height", type=int, default=512, help="Output height in pixels")
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process(args.input, args.output, args.height)


if __name__ == "__main__":
    main()
