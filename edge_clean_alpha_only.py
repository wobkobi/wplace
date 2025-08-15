#!/usr/bin/env python3
# edge_clean_alpha_transparent_fill.py
# Clean jagged edges by editing alpha only. RGB is untouched.
# Policies let you control whether transparency can be created or removed.

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageFilter


def binarise(a: Image.Image, threshold: int) -> Image.Image:
    arr = np.array(a, dtype=np.uint8)
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def clean_alpha(
    alpha: Image.Image, kernel: int, close_iters: int, open_iters: int, use_mode: bool
) -> Image.Image:
    if kernel % 2 == 0 or kernel < 3:
        raise ValueError("kernel must be an odd integer ≥ 3")
    a = alpha
    for _ in range(close_iters):
        a = a.filter(ImageFilter.MaxFilter(kernel)).filter(
            ImageFilter.MinFilter(kernel)
        )  # closing
    for _ in range(open_iters):
        a = a.filter(ImageFilter.MinFilter(kernel)).filter(
            ImageFilter.MaxFilter(kernel)
        )  # opening
    if use_mode:
        a = a.filter(ImageFilter.ModeFilter(kernel))
    return a


def combine_alpha(
    original: Image.Image, cleaned: Image.Image, threshold: int, policy: str
) -> Image.Image:
    a_in = np.array(original, dtype=np.uint8)
    a_bin = np.array(binarise(cleaned, threshold), dtype=np.uint8)

    if policy == "preserve-transparent":
        # Never turn fully transparent pixels opaque
        a_out = np.where(a_in == 0, 0, a_bin)
    elif policy == "grow-only":
        # Can add opacity, never delete it
        a_out = np.maximum(a_in, a_bin)
    elif policy == "shrink-only":
        # Can delete opacity, never add it
        a_out = np.minimum(a_in, a_bin)
    elif policy == "free":
        a_out = a_bin
    else:
        raise ValueError("invalid policy")

    # Final binary clamp
    a_out = np.where(a_out > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(a_out, mode="L")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smooth PNG edges by modifying alpha only; RGB unchanged."
    )
    ap.add_argument("input", type=str)
    ap.add_argument("output", type=str)
    ap.add_argument(
        "--kernel",
        type=int,
        default=5,
        help="Odd neighbourhood size (≥3). Larger = smoother.",
    )
    ap.add_argument(
        "--close",
        type=int,
        default=2,
        help="Closing iterations (fill pits, round corners).",
    )
    ap.add_argument(
        "--open", type=int, default=0, help="Opening iterations (shave spikes)."
    )
    ap.add_argument(
        "--mode", action="store_true", help="Apply ModeFilter for extra smoothing."
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Alpha > threshold becomes 255, else 0.",
    )
    ap.add_argument(
        "--policy",
        choices=["preserve-transparent", "grow-only", "shrink-only", "free"],
        default="preserve-transparent",
        help=(
            "preserve-transparent: never make new opaque pixels (avoid black fills). "
            "grow-only: only add opacity. shrink-only: only remove opacity. free: allow both."
        ),
    )
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGBA")
    r, g, b, a = img.split()

    a_clean = clean_alpha(a, args.kernel, args.close, args.open, args.mode)
    a_out = combine_alpha(a, a_clean, args.threshold, args.policy)

    out = Image.merge("RGBA", (r, g, b, a_out))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output, format="PNG", optimize=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
