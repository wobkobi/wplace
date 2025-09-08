#!/usr/bin/env python3
"""
palette_map.py
Recolor RGBA images to the wplace palette. Supports "pixel" and "photo" strategies.

Usage:
  python palette_map.py INPUT [OUTPUT] --mode [auto|pixel|photo] --height H --resample [nearest|bilinear|bicubic|lanczos] --debug

Modes:
  pixel : Global OKLab/OKLCh matching tuned for pixel art with neutral and lightness guards.
  photo : Dithered remap for photos and gradients using error diffusion.
  auto  : Choose pixel for small images and photo otherwise.

Input:
  Any Pillow-readable image. Alpha is preserved. Only nonzero alpha pixels are recolored.

Output:
  PNG by default. If OUTPUT is omitted, writes <stem>_wplace.png next to INPUT.

Notes:
  Palette data and color transforms come from palette_map.palette_data and colour_* modules.
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
from palette_map.palette_data import build_palette
from palette_map.core_types import PaletteItem, U8Image, U8Mask, Lab, Lch, NameOf
from palette_map.colour_convert import rgb_to_lab, lab_to_lch
from palette_map.colour_select import restrict_palette
from palette_map.utils import (
    fmt_secs,
    fmt_eta,
    load_rgba,
    save_png_rgba,
    pil_resample,
    unique_visible,
    weighted_percentile,
    nearest_palette_indices_lab,
    map_nearest_rgb_lab,
    colour_usage_report,
)


def _enable_line_buffered_stdout() -> None:
    """
    Enable line-buffered stdout when supported.

    Helps live progress printing in terminals that expose .reconfigure().
    Silently ignored if unsupported.
    """
    f = getattr(sys.stdout, "reconfigure", None)
    if callable(f):
        try:
            f(line_buffering=True, write_through=True)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for palette recoloring.

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
    p = argparse.ArgumentParser(
        prog="palette_map",
        description="Recolour image(s) to the project palette with ordered debug output.",
    )
    p.add_argument("src", type=Path, help="Input image or folder")
    p.add_argument(
        "--outdir", type=Path, default=None, help="Output directory (optional)"
    )
    p.add_argument(
        "--mode",
        choices=["auto", "pixel", "photo"],
        default="auto",
        help="Mapping mode.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize so height<=H. Omit for no resize.",
    )
    p.add_argument(
        "--resample",
        choices=["auto", "nearest", "bilinear", "bicubic", "lanczos"],
        default="auto",
        help='Scaling filter. "auto" => nearest for pixel, lanczos for photo.',
    )
    p.add_argument(
        "--limit",
        nargs="?",
        const=-1,
        type=int,
        help="Restrict to top-K palette colours. Omit value for auto K; omit flag for full palette.",
    )
    p.add_argument("--jobs", type=int, default=1, help="Files processed in parallel")
    p.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4, help="Internal workers"
    )
    p.add_argument("--debug", action="store_true", help="Verbose mapping details")
    return p.parse_args()


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
    items, name_of_rgb, pal_lab, pal_lch = build_palette()
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


def _print_debug_photo(
    uniq_rgb: U8Image,
    counts: np.ndarray,
    pal_rgb: U8Image,
    pal_lab: Lab,
    name_of: NameOf,
    top_k: int = 20,
) -> None:
    """
    Print per-mapped palette usage and dE stats for photo mode.

    Shows weighted mean and percentiles of dE2000 between unique source colours
    and their nearest palette entries, plus the top palette colours by share.
    """
    if uniq_rgb.shape[0] == 0:
        print("[debug] per-mapped palette: (no visible pixels)", flush=True)
        return
    src_lab: Lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    idx = nearest_palette_indices_lab(src_lab, pal_lab.astype(np.float32))
    diff = pal_lab[idx] - src_lab
    de = np.sqrt(np.sum(diff * diff, axis=1))
    de_mean = float(np.average(de, weights=counts))
    de_p50 = weighted_percentile(de, counts, 0.50)
    de_p90 = weighted_percentile(de, counts, 0.90)
    de_p99 = weighted_percentile(de, counts, 0.99)
    agg: Dict[int, int] = {}
    for j, c in zip(idx.tolist(), counts.tolist()):
        agg[j] = agg.get(j, 0) + c
    total = int(counts.sum())
    mapped_colours = len(agg)
    print(
        f"[debug] uniques_visible={uniq_rgb.shape[0]}  mapped_colours={mapped_colours}  "
        f"dE_mean={de_mean:.2f}  dE_p50={de_p50:.2f}  dE_p90={de_p90:.2f}  dE_p99={de_p99:.2f}",
        flush=True,
    )
    print(f"[debug] per-mapped palette (top {top_k}):", flush=True)
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k]
    for j, c in top:
        rgb = pal_rgb[j]
        hhex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        share = c / total if total else 0.0
        name = name_of.get(hhex, "?")
        print(f"  -> {hhex}  {name}: pixels={c}  share={share:.1%}", flush=True)


def decide_auto_mode(img_rgb: U8Image, alpha: U8Mask, pal_lab: Lab) -> str:
    """
    Pick "pixel" for few uniques or concentrated colours, else "photo".

    Heuristic:
      - Count unique visible RGBs and their concentration in the top 16.
      - Quick coarse map of uniques to palette bins in Lab to gauge spread.
    """
    uniq, counts = unique_visible(img_rgb, alpha)
    n_uniques = int(uniq.shape[0])
    if n_uniques == 0:
        return "pixel"

    total = int(counts.sum()) or 1
    k_raw = min(16, n_uniques)
    topk_raw_share = (np.sort(counts)[-k_raw:].sum() / total) if k_raw > 0 else 1.0

    src_lab = rgb_to_lab(uniq.astype(np.float32))
    idx = nearest_palette_indices_lab(src_lab, pal_lab.astype(np.float32))
    _ = np.bincount(idx, weights=counts.astype(np.float64), minlength=pal_lab.shape[0])

    if (n_uniques <= 512) or (topk_raw_share >= 0.80):
        return "pixel"
    return "photo"


def _process_one(
    src_path: Path,
    out_path: Optional[Path],
    mode: str,
    height: Optional[int],
    resample_sel: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    items: List[PaletteItem],
    name_of_hex: NameOf,
    limit_arg: Optional[int],
) -> None:
    """
    Process a single image path end-to-end:
      load -> optional resize -> palette restrict -> map -> save -> report.
    """
    t0 = time.perf_counter()
    if out_path is None:
        out_path = src_path.with_name(f"{src_path.stem}_wplace.png")

    print(f"\n=== {src_path.name} ===", flush=True)

    rgb_in, alpha = load_rgba(src_path)
    h0, w0 = rgb_in.shape[0], rgb_in.shape[1]

    eff_mode = mode
    if mode == "auto":
        eff_mode = decide_auto_mode(rgb_in, alpha, pal_lab)
        if debug:
            print(f"[debug] auto picked mode = {eff_mode}", flush=True)

    t_r0 = time.perf_counter()
    if height is not None and h0 > height:
        new_h = height
        new_w = int(round(w0 * (new_h / h0)))
        res_name = (
            resample_sel
            if resample_sel != "auto"
            else ("nearest" if eff_mode == "pixel" else "lanczos")
        )
        res = pil_resample(res_name)
        rgb_in = np.array(
            Image.fromarray(rgb_in).resize((new_w, new_h), res), dtype=np.uint8
        )
        alpha = np.array(
            Image.fromarray(alpha).resize((new_w, new_h), res), dtype=np.uint8
        )

    t_r1 = time.perf_counter()
    h, w = rgb_in.shape[0], rgb_in.shape[1]

    if debug:
        a_full0 = int(np.count_nonzero(alpha == 255))
        a_zero0 = int(np.count_nonzero(alpha == 0))
        print(
            f"[debug] load   {w0}x{h0}  alpha(255)={a_full0}  alpha(0)={a_zero0}",
            flush=True,
        )
        print(
            f"[debug] size   {w}x{h}  alpha(255)={a_full0}  alpha(0)={a_zero0}",
            flush=True,
        )
        print(f"[debug] mode   {eff_mode}", flush=True)

    if limit_arg is None:
        colours_mode = "full"
        limit_opt = None
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

    t1 = time.perf_counter()

    uniq_rgb, counts = unique_visible(rgb_in, alpha)
    if debug:
        print(
            f"[debug] pixel size_in={w0}x{h0}  size_eff={w}x{h}  workers={workers}  time={fmt_secs(t1 - t_r1)}",
            flush=True,
        )
        if eff_mode == "pixel":
            top16 = counts[np.argsort(-counts)[:16]].sum() if len(counts) else 0
            total = counts.sum() if len(counts) else 1
            share = (top16 / total) if total else 0.0
            print(
                f"[debug] uniques_visible={uniq_rgb.shape[0]}  top16_share={share:.3f}",
                flush=True,
            )
        else:
            _print_debug_photo(
                uniq_rgb, counts, pal_rgb_sel, pal_lab_sel, name_of_hex, top_k=20
            )

    try:
        if eff_mode == "photo":
            items_sel = [items[i] for i in sel_idx.tolist()]
            mapped = dither_photo(
                rgb_in,
                alpha,
                items_sel,
                pal_lab_sel,
                pal_lch_sel,
                workers=workers,
                progress=debug,
            )
        else:
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
            print(f"[debug] mapper failed ({e}); falling back to nearest", flush=True)
        mapped = map_nearest_rgb_lab(rgb_in, alpha, pal_rgb_sel, pal_lab_sel)

    t2 = time.perf_counter()

    save_png_rgba(out_path, mapped, alpha)
    t3 = time.perf_counter()

    print(f"Mode: {eff_mode}", flush=True)
    print(
        f"Wrote {out_path.name} | size={w}x{h} | palette_size={pal_rgb_sel.shape[0]}",
        flush=True,
    )
    print("Colours used:", flush=True)
    for hhex, nm, cnt in colour_usage_report(mapped, alpha, name_of_hex):
        print(f"  {hhex}  {nm}: {cnt}", flush=True)
    print(
        f"[debug] total  {fmt_secs(t3 - t0)}  "
        f"(load={fmt_secs(t1 - t0)}, resize={fmt_secs(t_r1 - t_r0)}, "
        f"mode={fmt_secs(t2 - t1)}, save={fmt_secs(t3 - t2)})",
        flush=True,
    )


def _process_one_captured(
    path: Path,
    mode: str,
    height: Optional[int],
    resample: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    items: List[PaletteItem],
    name_of: NameOf,
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
            print(f"\n=== {path.name} ===")
            print("[debug] skipped output artifact (_wplace)")
            return buf.getvalue()
        dst = (outdir / f"{path.stem}_wplace.png") if outdir else None
        _process_one(
            path,
            dst,
            mode,
            height,
            resample,
            debug,
            workers,
            pal_rgb,
            pal_lab,
            pal_lch,
            items,
            name_of,
            limit_arg,
        )
    return buf.getvalue()


def _process_one_live(
    path: Path,
    mode: str,
    height: Optional[int],
    resample_sel: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    items: List[PaletteItem],
    name_of_hex: NameOf,
    limit_arg: Optional[int],
    outdir: Optional[Path],
) -> None:
    """
    Process a single file and stream logs to stdout.
    """
    if path.stem.endswith("_wplace"):
        print(f"\n=== {path.name} ===", flush=True)
        print("[debug] skipped output artifact (_wplace)", flush=True)
        return
    dst = (outdir / f"{path.stem}_wplace.png") if outdir else None
    _process_one(
        path,
        dst,
        mode,
        height,
        resample_sel,
        debug,
        workers,
        pal_rgb,
        pal_lab,
        pal_lch,
        items,
        name_of_hex,
        limit_arg,
    )


def main() -> None:
    """
    CLI entry point.

    Handles single file or folder. In folder mode supports --jobs parallelism
    while preserving readable output ordering.
    """
    _enable_line_buffered_stdout()
    args = parse_args()

    src = args.src
    if not src.exists():
        print(f"error: not found: {src}", file=sys.stderr, flush=True)
        sys.exit(2)

    pal_rgb, pal_lab, pal_lch, items, name_of_hex = _build_palette_views()

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
        print(
            f"[debug] folder scan  entries={len(all_entries)}  images={len(files)}  jobs={args.jobs}  workers={args.workers}",
            flush=True,
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
                    items,
                    name_of_hex,
                    args.limit,
                    args.outdir,
                )
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = [
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
                        items,
                        name_of_hex,
                        args.limit,
                        args.outdir,
                    )
                    for p in files
                ]
                blocks = [f.result() for f in futs]
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
            items,
            name_of_hex,
            args.limit,
            args.outdir,
        )


if __name__ == "__main__":
    main()
