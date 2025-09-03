#!/usr/bin/env python3
"""
palette_map.py — CLI entry-point for colour_shiftr.

Usage:
  python palette_map.py <input_path> [--output OUT] [--mode auto|pixel|photo|bw] [--height H] [--debug]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Package imports
from palette_map import (  # io; palette enforcing; mode
    PALETTE,
    build_palette,
    decide_auto_mode,
    dither_bw,
    dither_photo,
    is_image_file,
    is_palette_only,
    load_image_rgba,
    lock_to_palette_by_uniques,
    lock_to_palette_per_pixel,
    resize_rgba_height,
    run_pixel,
    save_image_rgba,
)
from palette_map.analysis import print_color_usage
from PIL import Image, UnidentifiedImageError


def _palette_set(palette) -> set[Tuple[int, int, int]]:
    return {p.rgb for p in palette}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap images to a fixed palette (pixel / photo / bw)."
    )
    parser.add_argument("input", help="Input image path (file or directory)")
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (.png forced) when processing a single file. Defaults to *_wplace.png",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug printing",
    )
    args = parser.parse_args()

    # Build full palette once
    palette, name_of_full, pal_lab, pal_lch = build_palette(PALETTE)
    pal_lab_mat = pal_lab.astype(np.float32)
    pal_lch_mat = pal_lch.astype(np.float32)
    pal_set_full = _palette_set(palette)

    def process_one(
        in_path: Path, height: Optional[int], allow_output_arg: bool
    ) -> None:
        if not is_image_file(in_path):
            print(f"[error] {in_path} is not a valid image")
            return

        t0 = time.perf_counter()

        out_path = (
            Path(args.output)
            if (allow_output_arg and args.output)
            else in_path.with_name(in_path.stem + "_wplace.png")
        )
        # avoid in-place overwrite
        if in_path.resolve(strict=False) == out_path.resolve(strict=False):
            out_path = in_path.with_name(in_path.stem + "_wplace.png")

        # Load
        img_rgb, alpha = load_image_rgba(in_path)
        H0, W0, _ = img_rgb.shape
        if args.debug:
            a0_255 = int((alpha == 255).sum())
            a0_0 = int((alpha == 0).sum())
            print(f"[debug] load   {W0}x{H0}  alpha(255)={a0_255}  alpha(0)={a0_0}")

        # Decide mode
        mode_eff = (
            args.mode if args.mode != "auto" else decide_auto_mode(img_rgb, alpha)
        )

        # Resize (only downscale)
        resample = (
            Image.Resampling.NEAREST
            if mode_eff in ("pixel", "bw")
            else Image.Resampling.LANCZOS
        )
        img_rgb, alpha = resize_rgba_height(img_rgb, alpha, height, resample)
        H1, W1, _ = img_rgb.shape
        if args.debug:
            a1_255 = int((alpha == 255).sum())
            a1_0 = int((alpha == 0).sum())
            print(f"[debug] size   {W1}x{H1}  alpha(255)={a1_255}  alpha(0)={a1_0}")
            print(f"[debug] mode   {mode_eff}")

        # Run chosen mode
        if mode_eff == "photo":
            t_photo0 = time.perf_counter()
            out_rgb = dither_photo(img_rgb, alpha, palette, pal_lab_mat, pal_lch_mat)

            # enforce palette strictly
            ok0 = is_palette_only(out_rgb, alpha, pal_set_full)
            if not ok0:
                out_rgb = lock_to_palette_by_uniques(
                    out_rgb, alpha, palette, pal_lab_mat
                )
            ok1 = is_palette_only(out_rgb, alpha, pal_set_full)
            if not ok1:
                out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, palette)
            ok2 = is_palette_only(out_rgb, alpha, pal_set_full)
            t_photo1 = time.perf_counter()

            out_path = save_image_rgba(out_path, out_rgb, alpha)
            if args.debug:
                off = 0 if ok2 else -1  # only compute when needed
                if not ok2:  # lazy count to avoid extra pass unless debugging
                    # light count: scan only visible pixels
                    mask = alpha != 0
                    s = set(tuple(px) for px in out_rgb[mask].reshape(-1, 3))
                    off = sum(1 for c in s if c not in pal_set_full)
                print(
                    f"[debug] photo  time={t_photo1 - t_photo0:.3f}s  palette_ok={ok2}  off_pixels≈{off}"
                )
            print("Mode: photo")
            print(
                f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}"
            )
            print_color_usage(out_rgb, alpha, name_of_full)
            if args.debug:
                print(f"[debug] total  {time.perf_counter() - t0:.3f}s")
            return

        if mode_eff == "bw":
            t_bw0 = time.perf_counter()
            out_rgb = dither_bw(img_rgb, alpha, palette, pal_lab_mat, pal_lch_mat)
            out_path = save_image_rgba(out_path, out_rgb, alpha)
            if args.debug:
                print(f"[debug] bw     time={time.perf_counter() - t_bw0:.3f}s")
            print("Mode: bw")
            print(
                f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}"
            )
            print_color_usage(out_rgb, alpha, name_of_full)
            if args.debug:
                print(f"[debug] total  {time.perf_counter() - t0:.3f}s")
            return

        # pixel
        out_rgb, _ = run_pixel(
            img_rgb, alpha, palette, pal_lab, pal_lch, debug=args.debug
        )

        # hard enforce palette (some pixel paths may already be exact)
        if not is_palette_only(out_rgb, alpha, pal_set_full):
            out_rgb = lock_to_palette_by_uniques(out_rgb, alpha, palette, pal_lab_mat)
        if not is_palette_only(out_rgb, alpha, pal_set_full):
            out_rgb = lock_to_palette_per_pixel(out_rgb, alpha, palette)

        out_path = save_image_rgba(out_path, out_rgb, alpha)
        print("Mode: pixel")
        print(f"Wrote {out_path.name} | size={W1}x{H1} | palette_size={len(PALETTE)}")
        print_color_usage(out_rgb, alpha, name_of_full)
        if args.debug:
            print(f"[debug] total  {time.perf_counter() - t0:.3f}s")

    in_path = Path(args.input)

    if in_path.is_dir():
        ALLOWED_EXT = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
            ".gif",
        }
        for f in sorted(in_path.iterdir()):
            if not f.is_file():
                continue
            name_l = f.name.lower()
            if name_l.endswith("_wplace.png"):
                if args.debug:
                    print(f"[debug] skip already-processed: {f.name}")
                continue
            if f.suffix.lower() not in ALLOWED_EXT:
                if args.debug:
                    print(f"[debug] skip by ext: {f.name}")
                continue
            # quick validity check (avoid exceptions)
            try:
                with Image.open(f) as im:
                    im.seek(0)
            except (UnidentifiedImageError, OSError):
                if args.debug:
                    print(f"[debug] skip non-image: {f.name}")
                continue

            try:
                process_one(f, None, allow_output_arg=False)
            except Exception as e:
                print(f"[error] {f.name}: {e}")
        return

    process_one(in_path, args.height, allow_output_arg=True)


if __name__ == "__main__":
    main()
