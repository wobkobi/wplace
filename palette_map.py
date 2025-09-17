#!/usr/bin/env python3
"""
palette_map.py
Recolour RGBA images to the wplace palette. Supports "pixel" and "photo" strategies.

Usage:
  python palette_map.py INPUT [OUTPUT] --mode [auto|pixel|photo] --height H --resample [nearest|bilinear|bicubic|lanczos] --debug

Modes:
  pixel : Global OKLab/OKLCh matching tuned for pixel art with neutral and lightness guards.
  photo : Dithered remap for photos and gradients using error diffusion.
  auto  : Choose pixel for small images and photo otherwise.

Input:
  Any Pillow-readable image. Alpha is preserved. Only non-zero alpha pixels are recoloured.

Output:
  PNG by default. If OUTPUT is omitted, writes <stem>_wplace.png next to INPUT.

Notes:
  Palette data and colour transforms come from palette_map.palette_data and colour_* modules.
  Shared IO/math helpers come from palette_map.utils.
  CPU bound. ThreadPoolExecutor is used where safe.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

# mappers
from palette_map.pixel.run import (
    run_pixel,
)  # (rgb, a, pal_rgb, pal_lab, pal_lch, debug)->rgb
from palette_map.photo.dither import (
    dither_photo,
)  # (rgb, a, items, pal_lab, pal_lch, workers, ...)->rgb

# palette + helpers
from palette_map.palette_data import construct_palette
from palette_map.core_types import PaletteItem, U8Image, U8Mask, Lab, Lch, NameOf
from palette_map.colour_convert import rgb_to_lab
from palette_map.colour_select import restrict_palette
from palette_map.utils import (
    # formatting
    format_total_duration_compact,
    format_seconds_compact,
    # IO / ops
    load_image_rgba,
    save_png_rgba,
    pillow_resample_from_name,
    unique_visible_rgb,
    weighted_percentile,
    nearest_palette_indices_lab_distance,
    map_nearest_in_lab_space,
    colour_usage_report,
    # pretty logging
    print_banner,
    log,
    debug_log,
    print_config_line,
    key_value_pairs_to_string,
    enable_line_buffered_stdout,
)

# CLI args & small helpers


def _default_workers() -> int:
    """Leave a few cores free for the system; returns a sensible worker count."""
    n = os.cpu_count() or 4
    reserve = 1 if n <= 6 else 2 if n <= 12 else 3 if n <= 18 else 4
    return max(1, n - reserve)


def parse_cli_args() -> argparse.Namespace:
    """
    Parse CLI arguments for palette recolouring.

    Returns:
      argparse.Namespace with:
        src: Path to image or folder
        outdir: optional Path for outputs
        mode: "auto" | "pixel" | "photo"
        height: optional int max output height
        resample: resize filter name
        limit: None | -1 (auto-limited) | K (cap to top-K colours)
        jobs: parallel file workers
        workers: internal threads for heavy steps
        debug: bool for verbose mapping details
    """
    parser = argparse.ArgumentParser(
        prog="palette_map",
        description="Recolour image(s) to the project palette with tidy, readable output.",
    )
    parser.add_argument("src", type=Path, help="Input image or folder")
    parser.add_argument(
        "--outdir", type=Path, default=None, help="Output directory (optional)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pixel", "photo"],
        default="auto",
        help="Mapping mode.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize so height<=H. Omit for no resize.",
    )
    parser.add_argument(
        "--resample",
        choices=["auto", "nearest", "bilinear", "bicubic", "lanczos"],
        default="auto",
        help='Scaling filter. "auto" => nearest for pixel, lanczos for photo.',
    )
    parser.add_argument(
        "--limit",
        nargs="?",
        const=-1,
        type=int,
        help="Restrict to top-K palette colours. Omit value for auto K; omit flag for full palette.",
    )
    parser.add_argument(
        "--jobs", type=int, default=2, help="Files processed in parallel"
    )
    parser.add_argument(
        "--workers", type=int, default=_default_workers(), help="Internal workers"
    )
    parser.add_argument("--debug", action="store_true", help="Verbose mapping details")
    return parser.parse_args()


def _build_palette_views() -> Tuple[U8Image, Lab, Lch, List[PaletteItem], NameOf]:
    """
    Build palette arrays and name lookup for mappers.

    Returns:
      pal_rgb: uint8 [P,3]
      pal_lab: float32 [P,3]
      pal_lch: float32 [P,3]
      items: List[PaletteItem]
      name_of_hex: dict "#rrggbb" -> name
    """
    items, name_of_rgb, pal_lab, pal_lch = construct_palette()
    pal_rgb: U8Image = np.array([pi.rgb for pi in items], dtype=np.uint8)
    name_of_hex: NameOf = {
        f"#{r:02x}{g:02x}{b:02x}": items[i].name
        for i, (r, g, b) in enumerate(pal_rgb.tolist())
    }
    return (
        pal_rgb,
        pal_lab.astype(np.float32, copy=False),
        pal_lch.astype(np.float32, copy=False),
        items,
        name_of_hex,
    )


def _debug_photo_stats(
    unique_rgb: U8Image,
    counts: np.ndarray,
    pal_rgb: U8Image,
    pal_lab: Lab,
    name_of_hex: NameOf,
    top_k: int = 20,
) -> None:
    """
    Print per-mapped palette usage and dE stats for photo mode.

    Shows weighted mean and percentiles of dE2000 between unique source colours
    and their nearest palette entries, plus the top palette colours by share.
    """
    if unique_rgb.shape[0] == 0:
        debug_log("per-mapped palette: (no visible pixels)")
        return
    src_lab: Lab = rgb_to_lab(unique_rgb.astype(np.float32))
    nearest_idx = nearest_palette_indices_lab_distance(
        src_lab, pal_lab.astype(np.float32)
    )
    diff = pal_lab[nearest_idx] - src_lab
    de = np.sqrt(np.sum(diff * diff, axis=1))
    de_mean = float(np.average(de, weights=counts))
    de_p50 = weighted_percentile(de, counts, 0.50)
    de_p90 = weighted_percentile(de, counts, 0.90)
    de_p99 = weighted_percentile(de, counts, 0.99)

    usage: Dict[int, int] = {}
    for j, c in zip(nearest_idx.tolist(), counts.tolist()):
        usage[j] = usage.get(j, 0) + c

    total_px = int(counts.sum())
    mapped_colours = len(usage)

    debug_log(
        key_value_pairs_to_string(
            [
                ("Unique colours", int(unique_rgb.shape[0])),
                ("Mapped colours", int(mapped_colours)),
                ("dE mean", de_mean),
                ("dE p50", de_p50),
                ("dE p90", de_p90),
                ("dE p99", de_p99),
            ]
        )
    )
    debug_log(f"per-mapped palette (top {top_k}):")
    top = sorted(usage.items(), key=lambda kv: -kv[1])[:top_k]
    for j, c in top:
        rgb = pal_rgb[j]
        hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        share = c / total_px if total_px else 0.0
        name = name_of_hex.get(hex_code, "?")
        debug_log(f"  -> {hex_code}  {name}: pixels={c:,}  share={share:.1%}")


def decide_auto_mode(img_rgb: U8Image, alpha: U8Mask, pal_lab: Lab) -> str:
    """
    Pick "pixel" for few uniques or concentrated colours, else "photo".

    Heuristic:
      - Count unique visible RGBs and their concentration in the top 16.
      - Quick coarse map of uniques to palette bins in Lab to gauge spread.
    """
    unique_rgb, counts = unique_visible_rgb(img_rgb, alpha)
    n_uniques = int(unique_rgb.shape[0])
    if n_uniques == 0:
        return "pixel"

    total = int(counts.sum()) or 1
    k_raw = min(16, n_uniques)
    topk_share = (np.sort(counts)[-k_raw:].sum() / total) if k_raw > 0 else 1.0

    src_lab = rgb_to_lab(unique_rgb.astype(np.float32))
    idx = nearest_palette_indices_lab_distance(src_lab, pal_lab.astype(np.float32))
    _ = np.bincount(idx, weights=counts.astype(np.float64), minlength=pal_lab.shape[0])

    if (n_uniques <= 512) or (topk_share >= 0.80):
        return "pixel"
    return "photo"


# Per-file processing


def _process_single_image(
    src_path: Path,
    out_path: Optional[Path],
    mode: str,
    height_cap: Optional[int],
    resample_name: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    palette_items: List[PaletteItem],
    name_of_hex: NameOf,
    limit_arg: Optional[int],
) -> None:
    """
    Process a single image path end-to-end:
      load -> optional resize -> palette restrict -> map -> save -> report.
    """
    t_start = time.perf_counter()
    if out_path is None:
        out_path = src_path.with_name(f"{src_path.stem}_wplace.png")

    print_banner(src_path.name)

    rgb_in, alpha = load_image_rgba(src_path)
    height0, width0 = rgb_in.shape[0], rgb_in.shape[1]

    # Debug: report original size and alpha
    if debug:
        alpha_full0 = int(np.count_nonzero(alpha == 255))
        alpha_zero0 = int(np.count_nonzero(alpha == 0))
        debug_log(
            key_value_pairs_to_string(
                [
                    ("Loaded", f"{width0}x{height0}"),
                    ("Alpha=255", alpha_full0),
                    ("Alpha=0", alpha_zero0),
                ]
            )
        )

    # Mode selection
    mode_effective = mode
    if mode == "auto":
        mode_effective = decide_auto_mode(rgb_in, alpha, pal_lab)
        debug_log(f"auto picked mode: {mode_effective}")

    # Optional resize
    t_resize0 = time.perf_counter()
    if height_cap is not None and height0 > height_cap:
        new_h = int(height_cap)
        new_w = int(round(width0 * (new_h / height0)))
        resample_eff = (
            resample_name
            if resample_name != "auto"
            else ("nearest" if mode_effective == "pixel" else "lanczos")
        )
        res_enum = pillow_resample_from_name(resample_eff)
        rgb_in = np.array(
            Image.fromarray(rgb_in).resize((new_w, new_h), res_enum), dtype=np.uint8
        )
        alpha = np.array(
            Image.fromarray(alpha).resize((new_w, new_h), res_enum), dtype=np.uint8
        )

    # Debug: report effective size and alpha after resize, and mode
    t_resize1 = time.perf_counter()
    height, width = rgb_in.shape[0], rgb_in.shape[1]

    if debug:
        alpha_full1 = int(np.count_nonzero(alpha == 255))
        alpha_zero1 = int(np.count_nonzero(alpha == 0))
        if (width, height) != (width0, height0):
            debug_log(
                key_value_pairs_to_string(
                    [
                        ("Resized", f"{width}x{height}"),
                        ("Alpha=255", alpha_full1),
                        ("Alpha=0", alpha_zero1),
                    ]
                )
            )
        debug_log(f"mode: {mode_effective}")

    # Palette restriction
    if limit_arg is None:
        colours_mode = "full"
        limit_opt: Optional[int] = None
    else:
        colours_mode = "limited"
        limit_opt = None if int(limit_arg) < 0 else int(limit_arg)

    pal_rgb_sel, pal_lab_sel, pal_lch_sel, sel_idx = restrict_palette(
        colours_mode,
        rgb_in,
        alpha,
        pal_rgb,
        pal_lab,
        pal_lch,
        name_of=name_of_hex,
        limit_opt=limit_opt,
        debug=debug,
    )
    t_after_restrict = time.perf_counter()

    # Pre-map debug
    unique_rgb, counts = unique_visible_rgb(rgb_in, alpha)
    if debug:
        if mode_effective == "pixel":
            top_16 = counts[np.argsort(-counts)[:16]].sum() if len(counts) else 0
            total = counts.sum() if len(counts) else 1
            share = (top_16 / total) if total else 0.0
            debug_log(
                key_value_pairs_to_string(
                    [
                        ("Pre-map size", f"{width}x{height}"),
                        ("Workers", workers),
                        (
                            "Prep time",
                            format_seconds_compact(t_after_restrict - t_resize1),
                        ),
                        ("Visible uniques", int(unique_rgb.shape[0])),
                        ("Top16 share", f"{share:.3f}"),
                    ]
                )
            )
        else:
            _debug_photo_stats(
                unique_rgb, counts, pal_rgb_sel, pal_lab_sel, name_of_hex, top_k=10
            )

    # Map
    try:
        if mode_effective == "photo":
            items_sel = [palette_items[i] for i in sel_idx.tolist()]

            # MP config: mirror photo dither defaults (single source of truth lives in dither.py)
            ncpu = os.cpu_count() or 4
            reserve = 1 if ncpu <= 6 else 2 if ncpu <= 12 else 3 if ncpu <= 18 else 4
            mp_procs = max(1, ncpu - reserve)
            mp_threads = 1
            mp_block_rows = 256
            mp_overlap_rows = 24
            topk = 16

            # Human-friendly one-liner
            print_config_line(
                "photo",
                [
                    ("Workers", workers),
                    ("Blocks", True),
                    ("Processes", mp_procs),
                    ("Threads/Proc", mp_threads),
                    ("Block rows", mp_block_rows),
                    ("Overlap rows", mp_overlap_rows),
                    ("Top-K", topk),
                ],
                debug=debug,
            )

            mapped = dither_photo(
                rgb_in,
                alpha,
                items_sel,
                pal_lab_sel,
                pal_lch_sel,
                workers=workers,
                mp_blocks=True,
                mp_procs=mp_procs,
                mp_threads=mp_threads,
                mp_block_rows=mp_block_rows,
                mp_overlap_rows=mp_overlap_rows,
                topk=topk,
                progress=True,  # show ETA in photo mode
                profile=debug,  # detailed per-stage timing when --debug
            )
        else:
            if debug:
                print_config_line("pixel", [("Workers", workers)], debug=True)
            mapped = run_pixel(
                rgb_in, alpha, pal_rgb_sel, pal_lab_sel, pal_lch_sel, debug=debug
            )

        if isinstance(mapped, tuple):
            mapped = mapped[0]
        if mapped is None:
            raise ValueError("mapper returned None")
        if mapped.dtype != np.uint8:
            mapped = mapped.astype(np.uint8)
        if (
            (mapped.ndim != 3)
            or (mapped.shape[-1] not in (3, 4))
            or (mapped.shape[:2] != rgb_in.shape[:2])
        ):
            raise ValueError(f"invalid mapper output shape {mapped.shape}")
        if mapped.shape[-1] == 4:
            mapped = mapped[..., :3]
        mapped = np.ascontiguousarray(mapped)
    except Exception as e:
        if debug:
            debug_log(f"mapper failed ({e}); falling back to nearest")
        mapped = map_nearest_in_lab_space(rgb_in, alpha, pal_rgb_sel, pal_lab_sel)
    t_after_map = time.perf_counter()

    # Save
    save_png_rgba(out_path, mapped, alpha)
    t_after_save = time.perf_counter()

    # Report
    log(f"Mode: {mode_effective}")
    log(
        f"Wrote {out_path.name} | size={width}x{height} | palette_size={pal_rgb_sel.shape[0]}"
    )
    log("Colours used:")
    for hex_code, name, count in colour_usage_report(mapped, alpha, name_of_hex):
        log(f"  {hex_code}  {name}: {count:,}")

    total_pixels = int((alpha > 0).sum())
    log(f"Total pixels: {total_pixels:,}")

    if debug:
        map_secs = t_after_map - t_after_restrict
        if map_secs > 0:
            rate_mpx_s = (total_pixels / map_secs) / 1e6
            mpx = total_pixels / 1e6
            debug_log(
                f"throughput {rate_mpx_s:.2f} MPx/s  ({mpx:.2f} MPx in {format_seconds_compact(map_secs)})"
            )

    if debug:
        debug_log(
            f"Total {format_total_duration_compact(t_after_save - t_start)}  "
            f"(load={format_seconds_compact(t_after_restrict - t_start)}, resize={format_seconds_compact(t_resize1 - t_resize0)}, "
            f"mode={format_seconds_compact(t_after_map - t_after_restrict)}, save={format_seconds_compact(t_after_save - t_after_map)})"
        )
    else:
        log(f"Total time {format_total_duration_compact(t_after_save - t_start)}")


def _process_one_captured(
    path: Path,
    mode: str,
    height_cap: Optional[int],
    resample_name: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    palette_items: List[PaletteItem],
    name_of_hex: NameOf,
    limit_arg: Optional[int],
    outdir: Optional[Path],
) -> str:
    """
    Process a single file with stdout capture.

    Useful for concurrent execution where output should be printed in order.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        if path.stem.endswith("_wplace"):
            print_banner(path.name)
            debug_log("skipped output artifact (_wplace)")
            return buf.getvalue()
        dst = (outdir / f"{path.stem}_wplace.png") if outdir else None
        _process_single_image(
            path,
            dst,
            mode,
            height_cap,
            resample_name,
            debug,
            workers,
            pal_rgb,
            pal_lab,
            pal_lch,
            palette_items,
            name_of_hex,
            limit_arg,
        )
    return buf.getvalue()


def _process_one_live(
    path: Path,
    mode: str,
    height_cap: Optional[int],
    resample_name: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    palette_items: List[PaletteItem],
    name_of_hex: NameOf,
    limit_arg: Optional[int],
    outdir: Optional[Path],
) -> None:
    """Process a single file and stream logs to stdout."""
    if path.stem.endswith("_wplace"):
        print_banner(path.name)
        debug_log("skipped output artifact (_wplace)")
        return
    dst = (outdir / f"{path.stem}_wplace.png") if outdir else None
    _process_single_image(
        path,
        dst,
        mode,
        height_cap,
        resample_name,
        debug,
        workers,
        pal_rgb,
        pal_lab,
        pal_lch,
        palette_items,
        name_of_hex,
        limit_arg,
    )


# Entry point


def main() -> None:
    """
    CLI entry point.

    Handles single file or folder. In folder mode supports --jobs parallelism
    while preserving readable output ordering.
    """
    enable_line_buffered_stdout()
    args = parse_cli_args()

    cpu_cores = os.cpu_count() or 1
    # Always show a concise run configuration up-front.
    print_config_line(
        "run",
        [("CPU cores", cpu_cores), ("Workers", args.workers), ("Jobs", args.jobs)],
        debug=False,
    )
    if args.debug:
        debug_log(
            key_value_pairs_to_string(
                [
                    ("Mode", args.mode),
                    ("Height cap", args.height or "-"),
                    ("Resample", args.resample),
                ]
            )
        )

    src = args.src
    if not src.exists():
        print(f"error: not found: {src}", file=sys.stderr, flush=True)
        sys.exit(2)

    pal_rgb, pal_lab, pal_lch, palette_items, name_of_hex = _build_palette_views()

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if src.is_dir():
        all_entries = list(src.iterdir())
        files = [
            p
            for p in all_entries
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.stem.endswith("_wplace")
        ]
        files.sort(key=lambda p: p.name.lower())
        if args.debug:
            debug_log(
                key_value_pairs_to_string(
                    [
                        ("Folder entries", len(all_entries)),
                        ("Images", len(files)),
                        ("Jobs", args.jobs),
                        ("Workers", args.workers),
                    ]
                )
            )

        if args.jobs == 1:
            for p in files:
                _process_one_live(
                    p,
                    args.mode,
                    args.height,
                    args.resample,
                    args.debug,
                    args.workers,
                    pal_rgb,
                    pal_lab,
                    pal_lch,
                    palette_items,
                    name_of_hex,
                    args.limit,
                    args.outdir,
                )
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futures = [
                    ex.submit(
                        _process_one_captured,
                        p,
                        args.mode,
                        args.height,
                        args.resample,
                        args.debug,
                        args.workers,
                        pal_rgb,
                        pal_lab,
                        pal_lch,
                        palette_items,
                        name_of_hex,
                        args.limit,
                        args.outdir,
                    )
                    for p in files
                ]
                blocks = [f.result() for f in futures]
            print("".join(blocks), end="", flush=True)
    else:
        _process_one_live(
            src,
            args.mode,
            args.height,
            args.resample,
            args.debug,
            args.workers,
            pal_rgb,
            pal_lab,
            pal_lch,
            palette_items,
            name_of_hex,
            args.limit,
            args.outdir,
        )


if __name__ == "__main__":
    main()
