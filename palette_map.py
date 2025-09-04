#!/usr/bin/env python3
# palette_map.py — ordered debug blocks; skip *_wplace; mode=auto|pixel|photo|bw
from __future__ import annotations

import argparse
import inspect
import io
import os
import sys
import time
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace

import numpy as np
from PIL import Image, UnidentifiedImageError

from palette_map import run_pixel
from palette_map.core_types import (
    U8Image,
    U8Mask,
    Lab,
    Lch,
    NameOf,
)
from palette_map.color_convert import lab_to_lch, rgb_to_lab


# ---------------- I/O + buffering helpers ----------------


def _enable_line_buffered_stdout() -> None:
    """Try to make stdout line-buffered so progress prints appear continuously."""
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------- Palette helpers ----------------


def _get_palette_items_from_pkg():
    try:
        from palette_map.palette_data import build_palette  # type: ignore

        res = build_palette()
        if isinstance(res, tuple):
            for el in res:
                if isinstance(el, list):
                    return el
            return res[0]
        return res
    except Exception:
        return None


def _call_dither_dynamic(
    fn: Any,
    rgb_in: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    workers: int,
    debug: bool,
) -> U8Image:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(rgb_in, alpha, pal_rgb, pal_lab, pal_lch)

    args = [rgb_in, alpha]
    kwargs = {}

    items = _get_palette_items_from_pkg()
    if items is None or (
        hasattr(pal_lab, "shape") and len(items) != int(pal_lab.shape[0])
    ):
        items = [
            SimpleNamespace(rgb=(int(r), int(g), int(b)))
            for r, g, b in pal_rgb.tolist()
        ]

    if "palette" in sig.parameters:
        kwargs["palette"] = items
    elif "palette_items" in sig.parameters:
        kwargs["palette_items"] = items

    if "pal_lab_mat" in sig.parameters:
        kwargs["pal_lab_mat"] = pal_lab
    elif "pal_lab" in sig.parameters:
        kwargs["pal_lab"] = pal_lab

    if "pal_lch_mat" in sig.parameters:
        kwargs["pal_lch_mat"] = pal_lch
    elif "pal_lch" in sig.parameters:
        kwargs["pal_lch"] = pal_lch

    if "workers" in sig.parameters:
        kwargs["workers"] = workers
    if "debug" in sig.parameters:
        kwargs["debug"] = debug

    return fn(*args, **kwargs)


def _call_photo_dither(
    rgb_in: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    workers: int,
    debug: bool,
) -> U8Image:
    try:
        from palette_map.photo.dither import dither_photo  # type: ignore
    except Exception:
        return _map_nearest(rgb_in, alpha, pal_rgb, pal_lab)
    return _call_dither_dynamic(
        dither_photo, rgb_in, alpha, pal_rgb, pal_lab, pal_lch, workers, debug
    )


def _call_bw_dither(
    rgb_in: U8Image,
    alpha: U8Mask,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    workers: int,
    debug: bool,
) -> U8Image:
    try:
        from palette_map.bw.dither import dither_bw  # type: ignore
    except Exception:
        gray = np.dot(rgb_in.astype(np.float32), [0.2126, 0.7152, 0.0722])
        rgb_mono = np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)
        return _map_nearest(rgb_mono, alpha, pal_rgb, pal_lab)
    return _call_dither_dynamic(
        dither_bw, rgb_in, alpha, pal_rgb, pal_lab, pal_lch, workers, debug
    )


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
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
        choices=["auto", "pixel", "photo", "bw"],
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
        help='Scaling filter. "auto" => nearest for pixel, lanczos for photo/bw.',
    )
    p.add_argument("--jobs", type=int, default=1, help="Files processed in parallel")
    p.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4, help="Internal workers"
    )
    p.add_argument("--debug", action="store_true", help="Verbose mapping details")
    return p.parse_args()


# ---------------- Palette ----------------


def _hex_to_rgb_array(hex_list: List[str]) -> U8Image:
    arr = np.zeros((len(hex_list), 3), dtype=np.uint8)
    for i, hx in enumerate(hex_list):
        s = hx[1:] if hx.startswith("#") else hx
        arr[i] = [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)]
    return arr


def load_palette() -> Tuple[U8Image, List[str], NameOf]:
    from palette_map import PALETTE

    pal_hex: List[str] = [h for h, _ in PALETTE]
    pal_rgb: U8Image = _hex_to_rgb_array(pal_hex)
    name_of: NameOf = {h: n for h, n in PALETTE}
    return pal_rgb, pal_hex, name_of


# ---------------- Mapping ----------------


def nearest_palette_indices(src_lab: Lab, pal_lab: Lab) -> np.ndarray:
    diff = pal_lab[None, :, :] - src_lab[:, None, :]
    de2 = np.sum(diff * diff, axis=2)
    return np.argmin(de2, axis=1).astype(np.int32)


def _map_nearest(
    rgb_in: U8Image, alpha: U8Mask, pal_rgb: U8Image, pal_lab: Lab
) -> U8Image:
    src_lab: Lab = rgb_to_lab(rgb_in.astype(np.float32))
    flat = src_lab.reshape(-1, 3)
    vis = alpha.reshape(-1) > 0
    idx = np.zeros(flat.shape[0], dtype=np.int32)
    if np.any(vis):
        idx_vis = nearest_palette_indices(flat[vis], pal_lab.astype(np.float32))
        idx[vis] = idx_vis
    return pal_rgb[idx].reshape(rgb_in.shape).astype(np.uint8)


# ---------------- I/O ----------------


def _pil_resample(name: str) -> int:
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    return Image.Resampling.BICUBIC


def _load_rgba(path: Path) -> Tuple[U8Image, U8Mask]:
    im = Image.open(path)
    try:
        im = im.convert("RGBA")
    except UnidentifiedImageError:
        im = im.convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    return arr[..., :3], arr[..., 3]


def _save_png_rgb_a(path: Path, rgb: U8Image, alpha: U8Mask) -> None:
    out = np.concatenate([rgb, alpha[..., None]], axis=-1)
    Image.fromarray(out).save(path)


# ---------------- Utils ----------------


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.3f}s"
    m = int(s // 60)
    return f"{m}m {s - 60*m:.1f}s"


def _unique_visible(rgb: U8Image, alpha: U8Mask) -> Tuple[U8Image, np.ndarray]:
    vis = alpha > 0
    if not np.any(vis):
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    flat = rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return uniq.astype(np.uint8), counts.astype(np.int64)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return float(v[-1])
    pos = q * cw[-1]
    idx = np.searchsorted(cw, pos, side="left")
    return float(v[min(idx, len(v) - 1)])


def _is_grayscaleish(
    rgb: U8Image, alpha: U8Mask, tol: int = 3, frac: float = 0.95
) -> bool:
    vis = alpha > 0
    if not np.any(vis):
        return False
    r, g, b = [rgb[..., i].astype(np.int16) for i in (0, 1, 2)]
    d1 = np.abs(r - g)[vis]
    d2 = np.abs(g - b)[vis]
    ok = np.logical_and(d1 <= tol, d2 <= tol)
    return (np.count_nonzero(ok) / ok.size) >= frac


def decide_auto_mode(img_rgb: U8Image, alpha: U8Mask) -> str:
    """
    Pixel vs Photo heuristic (old behavior):
      - If unique colours <= 512 OR the top-16 colours cover >= 80% of visible
        pixels, choose 'pixel'.
      - Otherwise choose 'photo'.
    """
    uniq, counts = _unique_visible(img_rgb, alpha)
    n_uniques = int(uniq.shape[0])
    if n_uniques == 0:
        return "pixel"

    total = int(counts.sum())
    if total == 0:
        return "pixel"

    k = min(16, n_uniques)
    # sum of largest-k counts
    topk = int(np.sort(counts)[-k:].sum()) if k > 0 else 0
    share = (topk / total) if total > 0 else 1.0

    return "pixel" if (n_uniques <= 512 or share >= 0.80) else "photo"


def _auto_pick_mode(rgb: U8Image, alpha: U8Mask) -> str:
    # keep BW detection first
    if _is_grayscaleish(rgb, alpha, tol=3, frac=0.95):
        return "bw"
    return decide_auto_mode(rgb, alpha)


def _print_per_unique(
    uniq_rgb: U8Image,
    counts: np.ndarray,
    pal_rgb: U8Image,
    pal_lab: Lab,
    pal_lch: Lch,
    name_of: NameOf,
) -> None:
    if uniq_rgb.shape[0] == 0:
        print("[debug] per-unique mapping: (no visible pixels)", flush=True)
        return
    src_lab: Lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    src_lch: Lch = lab_to_lch(src_lab)
    idx = nearest_palette_indices(src_lab, pal_lab)
    order = np.argsort(-counts)
    print("[debug] per-unique mapping:", flush=True)
    for i in order:
        s_hex = f"#{uniq_rgb[i,0]:02x}{uniq_rgb[i,1]:02x}{uniq_rgb[i,2]:02x}"
        j = int(idx[i])
        t_rgb = pal_rgb[j]
        t_lab = pal_lab[j]
        t_lch = pal_lch[j]
        dE = float(np.linalg.norm(t_lab - src_lab[i]))
        dL = float(t_lch[0] - src_lch[i, 0])
        dC = float(t_lch[1] - src_lch[i, 1])
        dh = float(t_lch[2] - src_lch[i, 2])
        while dh > 180.0:
            dh -= 360.0
        while dh < -180.0:
            dh += 360.0
        t_hex = f"#{t_rgb[0]:02x}{t_rgb[1]:02x}{t_rgb[2]:02x}"
        print(
            f"  src {s_hex}  count={counts[i]:5d} -> {t_hex}  "
            f"[dE={dE:5.2f}, dL={dL:+6.3f}, dC={dC:+6.3f}, dh={dh:4.1f} deg]",
            flush=True,
        )


def _print_debug_photo(
    uniq_rgb: U8Image,
    counts: np.ndarray,
    pal_rgb: U8Image,
    pal_lab: Lab,
    name_of: NameOf,
    top_k: int = 20,
) -> None:
    if uniq_rgb.shape[0] == 0:
        print("[debug] per-mapped palette: (no visible pixels)", flush=True)
        return
    src_lab: Lab = rgb_to_lab(uniq_rgb.astype(np.float32))
    idx = nearest_palette_indices(src_lab, pal_lab)
    diff = pal_lab[idx] - src_lab
    de = np.sqrt(np.sum(diff * diff, axis=1))
    de_mean = float(np.average(de, weights=counts))
    de_p50 = _weighted_percentile(de, counts, 0.50)
    de_p90 = _weighted_percentile(de, counts, 0.90)
    de_p99 = _weighted_percentile(de, counts, 0.99)
    agg_counts: Dict[int, int] = {}
    for j, c in zip(idx.tolist(), counts.tolist()):
        agg_counts[j] = agg_counts.get(j, 0) + c
    total = int(counts.sum())
    mapped_colours = len(agg_counts)
    print(
        f"[debug] uniques_visible={uniq_rgb.shape[0]}  mapped_colours={mapped_colours}  "
        f"dE_mean={de_mean:.2f}  dE_p50={de_p50:.2f}  dE_p90={de_p90:.2f}  dE_p99={de_p99:.2f}",
        flush=True,
    )
    print(f"[debug] per-mapped palette (top {top_k}):", flush=True)
    top = sorted(agg_counts.items(), key=lambda kv: -kv[1])[:top_k]
    for j, c in top:
        rgb = pal_rgb[j]
        hhex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        share = c / total if total else 0.0
        name = name_of.get(hhex, "?")
        print(f"  -> {hhex}  {name}: pixels={c}  share={share:.1%}", flush=True)


def _colour_usage_report(
    mapped_rgb: U8Image,
    alpha: U8Mask,
    pal_hex: List[str],
    name_of: NameOf,
) -> List[Tuple[str, str, int]]:
    vis = alpha > 0
    if not np.any(vis):
        return []
    flat = mapped_rgb[vis].reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    out: List[Tuple[str, str, int]] = []
    for rgb, c in sorted(zip(uniq, counts), key=lambda x: -int(x[1])):
        h = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        out.append((h, name_of.get(h, "?"), int(c)))
    return out


# ---------------- One file ----------------


def _process_one(
    src_path: Path,
    out_path: Path | None,
    mode: str,
    height: int | None,
    resample_sel: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_hex: List[str],
    name_of: NameOf,
    pal_lab: Lab,
    pal_lch: Lch,
) -> None:
    t0 = time.perf_counter()
    if out_path is None:
        out_path = src_path.with_name(f"{src_path.stem}_wplace.png")

    rgb_in, alpha = _load_rgba(src_path)

    eff_mode = mode
    if mode == "auto":
        eff_mode = _auto_pick_mode(rgb_in, alpha)
        if debug:
            print(f"[debug] auto picked mode = {eff_mode}", flush=True)

    resample_eff = (
        resample_sel
        if resample_sel != "auto"
        else ("nearest" if eff_mode == "pixel" else "lanczos")
    )

    if eff_mode == "bw":
        gray = np.dot(rgb_in.astype(np.float32), [0.2126, 0.7152, 0.0722])
        rgb_in = np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)

    h0, w0 = rgb_in.shape[0], rgb_in.shape[1]
    if debug:
        a_full = int(np.count_nonzero(alpha == 255))
        a_zero = int(np.count_nonzero(alpha == 0))
        print(
            f"[debug] load   {w0}x{h0}  alpha(255)={a_full}  alpha(0)={a_zero}",
            flush=True,
        )

    t_r0 = time.perf_counter()
    if height is not None and h0 > height:
        new_h = height
        new_w = int(round(w0 * (new_h / h0)))
        res = _pil_resample(resample_eff)
        rgb_in = np.array(
            Image.fromarray(rgb_in, mode="RGB").resize((new_w, new_h), res),
            dtype=np.uint8,
        )
        alpha = np.array(
            Image.fromarray(alpha, mode="L").resize((new_w, new_h), res), dtype=np.uint8
        )
    t_r1 = time.perf_counter()

    h, w = rgb_in.shape[0], rgb_in.shape[1]
    if debug:
        a_full = int(np.count_nonzero(alpha == 255))
        a_zero = int(np.count_nonzero(alpha == 0))
        print(
            f"[debug] size   {w}x{h}  alpha(255)={a_full}  alpha(0)={a_zero}",
            flush=True,
        )
        print(f"[debug] mode   {eff_mode}", flush=True)

    t_u0 = time.perf_counter()
    uniq_rgb, counts = _unique_visible(rgb_in, alpha)
    t_u1 = time.perf_counter()
    if debug:
        print(
            f"[debug] pixel size_in={w0}x{h0}  size_eff={w}x{h}  workers={workers}  time={_fmt_secs(t_u1 - t_u0)}",
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
            _print_debug_photo(uniq_rgb, counts, pal_rgb, pal_lab, name_of, top_k=20)

    t1 = time.perf_counter()
    try:
        if eff_mode == "photo":
            mapped = _call_photo_dither(
                rgb_in, alpha, pal_rgb, pal_lab, pal_lch, workers, debug
            )
        elif eff_mode == "bw":
            mapped = _call_bw_dither(
                rgb_in, alpha, pal_rgb, pal_lab, pal_lch, workers, debug
            )
        else:
            mapped = run_pixel(rgb_in, alpha, pal_rgb, pal_lab, pal_lch, debug=debug)

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
        mapped = _map_nearest(rgb_in, alpha, pal_rgb, pal_lab)
    t2 = time.perf_counter()

    _save_png_rgb_a(out_path, mapped, alpha)
    t3 = time.perf_counter()

    print(f"Mode: {eff_mode}", flush=True)
    print(
        f"Wrote {out_path.name} | size={w}x{h} | palette_size={len(pal_hex)}",
        flush=True,
    )
    print("Colours used:", flush=True)
    for hhex, nm, cnt in _colour_usage_report(mapped, alpha, pal_hex, name_of):
        print(f"  {hhex}  {nm}: {cnt}", flush=True)
    print(
        f"[debug] total  {_fmt_secs(t3 - t0)}  "
        f"(load={_fmt_secs(t1 - t0)}, resize={_fmt_secs(t_r1 - t_r0)}, "
        f"mode={_fmt_secs(t2 - t1)}, save={_fmt_secs(t3 - t2)})",
        flush=True,
    )


# ----- streaming vs captured printing ----------------------------------------


def _process_one_captured(
    p: Path,
    mode: str,
    height: int | None,
    resample_sel: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_hex: List[str],
    name_of: NameOf,
    pal_lab: Lab,
    pal_lch: Lch,
    outdir: Path | None,
) -> str:
    """Capture output for parallel jobs to avoid interleaving."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"\n=== {p.name} ===")
        if p.stem.endswith("_wplace"):
            print("[debug] skipped output artifact (_wplace)")
            return buf.getvalue()
        dst = (outdir / f"{p.stem}_wplace.png") if outdir else None
        _process_one(
            src_path=p,
            out_path=dst,
            mode=mode,
            height=height,
            resample_sel=resample_sel,
            debug=debug,
            workers=workers,
            pal_rgb=pal_rgb,
            pal_hex=pal_hex,
            name_of=name_of,
            pal_lab=pal_lab,
            pal_lch=pal_lch,
        )
    return buf.getvalue()


def _process_one_live(
    p: Path,
    mode: str,
    height: int | None,
    resample_sel: str,
    debug: bool,
    workers: int,
    pal_rgb: U8Image,
    pal_hex: List[str],
    name_of: NameOf,
    pal_lab: Lab,
    pal_lch: Lch,
    outdir: Path | None,
) -> None:
    """Stream output live (no buffering)."""
    print(f"\n=== {p.name} ===", flush=True)
    if p.stem.endswith("_wplace"):
        print("[debug] skipped output artifact (_wplace)", flush=True)
        return
    dst = (outdir / f"{p.stem}_wplace.png") if outdir else None
    _process_one(
        src_path=p,
        out_path=dst,
        mode=mode,
        height=height,
        resample_sel=resample_sel,
        debug=debug,
        workers=workers,
        pal_rgb=pal_rgb,
        pal_hex=pal_hex,
        name_of=name_of,
        pal_lab=pal_lab,
        pal_lch=pal_lch,
    )


# ---------------- Main ----------------


def main() -> None:
    _enable_line_buffered_stdout()
    args = parse_args()

    src = args.src
    if not src.exists():
        print(f"error: not found: {src}", file=sys.stderr, flush=True)
        sys.exit(2)

    pal_rgb, pal_hex, name_of = load_palette()
    pal_lab: Lab = rgb_to_lab(pal_rgb.astype(np.float32))
    pal_lch: Lch = lab_to_lch(pal_lab.astype(np.float32))

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
                    args.resample,  # effective choice decided per-file (after auto-pick)
                    args.debug,
                    args.workers,
                    pal_rgb,
                    pal_hex,
                    name_of,
                    pal_lab,
                    pal_lch,
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
                        pal_hex,
                        name_of,
                        pal_lab,
                        pal_lch,
                        args.outdir,
                    )
                    for p in files
                ]
                blocks = [f.result() for f in futs]
            print("".join(blocks), end="", flush=True)
    else:
        # Single file => always stream prints live
        _process_one_live(
            src,
            args.mode,
            args.height,
            args.resample,
            args.debug,
            args.workers,
            pal_rgb,
            pal_hex,
            name_of,
            pal_lab,
            pal_lch,
            args.outdir,
        )


if __name__ == "__main__":
    main()
