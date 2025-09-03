#!/usr/bin/env python3
"""
ow_mosaic.py - Pixel-accurate mosaic from an order file and a CSV of box sizes.

Spec
- Order file: each line is a row, comma-separated names. Case-insensitive match to CSV.
- CSV header: file,height,width (spaces ok). Units are pixels.
- Transparent rectangles with a border in INK. Borders never double-thicken (mask union).
- Global alignment: columns align across all rows. Uniform gutters and outer margins.
- Even spacing: a single GAP value is used for horizontal/vertical gutters and outer margins.
- Future-proof option: --auto-gap-pct scales GAP from the current data (shortest cell edge).
- Each rectangle centred horizontally and vertically in its cell.
- Labels = filename stem. One UNIFORM font size that fits the smallest box’s label.
- 1 px outer border. Strict name-set match. Writes a JSON manifest.

Usage
  python ow_mosaic.py --order "OW Order.txt" --csv "overwatch/ow.csv" --out "mosaic.png"
  # fixed 4 px spacing
  python ow_mosaic.py --gap 4
  # auto spacing = 3% of the shortest cell edge (min column width or row height)
  python ow_mosaic.py --auto-gap-pct 3
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# --------------------------- shared colour ---------------------------
# Change this single variable to set BOTH text and border colours (RGBA).
INK: Tuple[int, int, int, int] = (0, 0, 0, 255)

# --------------------------- data ---------------------------


@dataclass(frozen=True)
class BoxSpec:
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class PlacedBox:
    name: str
    width: int
    height: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive


# --------------------------- helpers ---------------------------


def norm_name(s: str) -> str:
    return s.strip().lower()


def label_from_name(s: str) -> str:
    stem = Path(s).stem
    return stem if stem else s


def read_order(order_path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in order_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        names = [t.strip() for t in line.split(",") if t.strip()]
        if names:
            rows.append(names)
    if not rows:
        raise SystemExit("Order file has no rows.")
    return rows


def read_csv_sizes(csv_path: Path) -> Dict[str, BoxSpec]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, skipinitialspace=True)
        wanted = {"file", "height", "width"}
        headers = {h.strip().lower() for h in rdr.fieldnames or []}
        if not wanted.issubset(headers):
            raise SystemExit(
                f"CSV header must include file,height,width. Found: {headers}"
            )
        seen: Dict[str, BoxSpec] = {}
        for row in rdr:
            raw_name = str(row.get("file", "")).strip()
            if raw_name == "":
                raise SystemExit("CSV row with empty 'file' field.")
            try:
                h = int(str(row.get("height", "")).strip())
                w = int(str(row.get("width", "")).strip())
            except ValueError:
                raise SystemExit(
                    f"Non-integer height/width for '{raw_name}'."
                ) from None
            key = norm_name(raw_name)
            if key in seen:
                raise SystemExit(
                    f"Duplicate name in CSV after normalising: '{raw_name}'."
                )
            seen[key] = BoxSpec(name=raw_name, width=w, height=h)
    if not seen:
        raise SystemExit("CSV contained no rows.")
    return seen


def compute_global_grid(
    order_rows: List[List[str]], specs: Dict[str, BoxSpec]
) -> Tuple[List[int], List[int]]:
    col_count = max(len(r) for r in order_rows)
    col_widths: List[int] = []
    for c in range(col_count):
        max_w = 0
        for r in order_rows:
            if c < len(r):
                key = norm_name(r[c])
                max_w = max(max_w, specs[key].width)
        col_widths.append(max_w)
    row_heights: List[int] = []
    for r in order_rows:
        max_h = 0
        for name in r:
            key = norm_name(name)
            max_h = max(max_h, specs[key].height)
        row_heights.append(max_h)
    return col_widths, row_heights


def resolve_gap(
    col_widths: List[int], row_heights: List[int], gap: int, auto_gap_pct: float
) -> int:
    if auto_gap_pct <= 0:
        return max(0, int(gap))
    if not col_widths or not row_heights:
        return max(0, int(gap))
    shortest_cell = min(min(col_widths), min(row_heights))
    auto_gap = int(round(shortest_cell * (auto_gap_pct / 100.0)))
    return max(1, auto_gap)


def try_font_paths(user_path: Optional[str]):
    candidates: List[str] = []
    if user_path:
        candidates.append(user_path)
    candidates += [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, 16), p
        except OSError:
            continue
    return ImageFont.load_default(), None


def best_font_size_for_rect(
    font_file: str, text: str, rect_w: int, rect_h: int, padding: int = 4
) -> int:
    lo, hi = 1, max(8, min(rect_w, rect_h))
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_file, mid)
        tw, th = f.getbbox(text)[2], f.getbbox(text)[3]
        if tw <= rect_w - 2 * padding and th <= rect_h - 2 * padding:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# --------------------------- main render ---------------------------


def render(
    order_path: Path,
    csv_path: Path,
    out_path: Path,
    manifest_path: Optional[Path],
    font_path: Optional[str],
    outer_border: int,
    rect_border: int,
    draw_grid_boundaries: bool,
    gap: int,
    auto_gap_pct: float,
) -> None:
    order_rows = read_order(order_path)
    specs = read_csv_sizes(csv_path)

    # strict name set match
    order_names = [norm_name(n) for row in order_rows for n in row]
    order_set = set(order_names)
    csv_set = set(specs.keys())
    if order_set != csv_set:
        missing = sorted(order_set - csv_set)
        extra = sorted(csv_set - order_set)
        msg = []
        if missing:
            msg.append(f"Missing in CSV: {missing}")
        if extra:
            msg.append(f"Extra in CSV: {extra}")
        raise SystemExit("Name mismatch. " + " ".join(msg))

    # global grid
    col_widths, row_heights = compute_global_grid(order_rows, specs)
    col_count, row_count = len(col_widths), len(row_heights)

    # future-proof uniform gap
    gap_px = resolve_gap(col_widths, row_heights, gap, auto_gap_pct)

    # content size with uniform gutters
    content_w = sum(col_widths) + gap_px * max(0, col_count - 1)
    content_h = sum(row_heights) + gap_px * max(0, row_count - 1)

    # canvas size with equal outer margins = gap_px
    canvas_w = content_w + 2 * (outer_border + gap_px)
    canvas_h = content_h + 2 * (outer_border + gap_px)

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # border mask to avoid double-thick overlaps
    border_mask = Image.new("L", (canvas_w, canvas_h), 0)
    bdraw = ImageDraw.Draw(border_mask)

    # outer border once
    if outer_border > 0:
        bdraw.rectangle(
            [0, 0, canvas_w - 1, canvas_h - 1], outline=255, width=outer_border
        )

    # precompute column and row starts
    x_starts: List[int] = []
    x_cursor = outer_border + gap_px
    for c in range(col_count):
        x_starts.append(x_cursor)
        x_cursor += col_widths[c] + (gap_px if c < col_count - 1 else 0)

    y_starts: List[int] = []
    y_cursor = outer_border + gap_px
    for r in range(row_count):
        y_starts.append(y_cursor)
        y_cursor += row_heights[r] + (gap_px if r < row_count - 1 else 0)

    # place boxes centred inside each cell
    placed: List[PlacedBox] = []
    for r_idx, row in enumerate(order_rows):
        row_h = row_heights[r_idx]
        for c_idx in range(col_count):
            if c_idx >= len(row):
                continue  # empty cell
            name = row[c_idx]
            spec = specs[norm_name(name)]
            col_w = col_widths[c_idx]
            x0_cell = x_starts[c_idx]
            y0_cell = y_starts[r_idx]
            x0 = x0_cell + (col_w - spec.width) // 2
            y0 = y0_cell + (row_h - spec.height) // 2
            x1 = x0 + spec.width - 1
            y1 = y0 + spec.height - 1

            placed.append(
                PlacedBox(
                    name=spec.name,
                    width=spec.width,
                    height=spec.height,
                    row=r_idx,
                    col=c_idx,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )
            if rect_border > 0:
                bdraw.rectangle([x0, y0, x1, y1], outline=255, width=rect_border)

    # optional debug cell boundaries
    if draw_grid_boundaries:
        # vertical boundaries
        xx = outer_border + gap_px
        for c in range(col_count - 1):
            xx += col_widths[c]
            bdraw.line(
                [
                    (xx, outer_border + gap_px),
                    (xx, canvas_h - outer_border - gap_px - 1),
                ],
                fill=128,
                width=1,
            )
            xx += gap_px
        # horizontal boundaries
        yy = outer_border + gap_px
        for r in range(row_count - 1):
            yy += row_heights[r]
            bdraw.line(
                [
                    (outer_border + gap_px, yy),
                    (canvas_w - outer_border - gap_px - 1, yy),
                ],
                fill=128,
                width=1,
            )
            yy += gap_px

    # apply all borders at once
    img.paste(INK, (0, 0), border_mask)

    # labels: uniform size that fits the smallest box’s label
    base_font, font_file = try_font_paths(font_path)
    if font_file:
        sizes: List[int] = []
        for pb in placed:
            text = label_from_name(pb.name)
            sizes.append(
                best_font_size_for_rect(font_file, text, pb.width, pb.height, padding=4)
            )
        uniform_size = max(1, min(sizes)) if sizes else 12
        font = ImageFont.truetype(font_file, uniform_size)
    else:
        font = base_font  # bitmap fallback

    for pb in placed:
        text = label_from_name(pb.name)
        tbx = draw.textbbox((0, 0), text, font=font)
        tw, th = tbx[2] - tbx[0], tbx[3] - tbx[1]
        tx = pb.x0 + (pb.width - tw) // 2
        ty = pb.y0 + (pb.height - th) // 2
        draw.text((tx, ty), text, fill=INK, font=font)

    # save image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    # save manifest
    manifest = {
        "order_file": str(order_path),
        "csv_file": str(csv_path),
        "image_file": str(out_path),
        "outer_border": outer_border,
        "rect_border": rect_border,
        "gap_px": gap_px,
        "auto_gap_pct": auto_gap_pct,
        "col_widths": col_widths,
        "row_heights": row_heights,
        "content_width": content_w,
        "content_height": content_h,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "placed": [
            {
                "name": pb.name,
                "row": pb.row,
                "col": pb.col,
                "width": pb.width,
                "height": pb.height,
                "x0": pb.x0,
                "y0": pb.y0,
                "x1": pb.x1,
                "y1": pb.y1,
            }
            for pb in placed
        ],
    }
    if manifest_path is None:
        manifest_path = out_path.with_suffix(".json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)


# --------------------------- cli ---------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a pixel-accurate mosaic PNG from an order file and a CSV of box sizes."
    )
    p.add_argument(
        "--order",
        type=Path,
        default=Path("OW Order.txt"),
        help="Path to order text file.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("overwatch/ow.csv"),
        help="Path to CSV with file,height,width.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("mosaic.png"), help="Output PNG path."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path. Defaults to <out>.json",
    )
    p.add_argument(
        "--font",
        type=str,
        default=None,
        help="TTF font path. Tries common system fonts if omitted.",
    )
    p.add_argument(
        "--outer-border", type=int, default=1, help="Outer border thickness in pixels."
    )
    p.add_argument(
        "--border",
        type=int,
        default=1,
        help="Per-rectangle border thickness in pixels.",
    )
    p.add_argument(
        "--gap",
        type=int,
        default=3,
        help="Uniform gutters and outer margins in pixels.",
    )
    p.add_argument(
        "--auto-gap-pct",
        type=float,
        default=0.0,
        help="If >0, gap = this percent of the shortest cell edge.",
    )
    p.add_argument(
        "--grid-boundaries", action="store_true", help="Draw debug cell boundaries."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    render(
        order_path=args.order,
        csv_path=args.csv,
        out_path=args.out,
        manifest_path=args.manifest,
        font_path=args.font,
        outer_border=args.outer_border,
        rect_border=args.border,
        draw_grid_boundaries=args.grid_boundaries,
        gap=args.gap,
        auto_gap_pct=args.auto_gap_pct,
    )


if __name__ == "__main__":
    main()
