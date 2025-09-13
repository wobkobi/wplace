# wplace palette mapper

I could not find a color remapper I liked, so I wrote one. It has two modes. One for pixel art. One for photos and painterly stuff. They use different logic. Play with both.

You can cap how many different colors the output uses, like with black and white images only using black and white. You can also scale the image. Pixel and photo modes prefer different resamplers. You can change which one is used if you want to get technical.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# auto mode chooses pixel for small images, photo otherwise
python palette_map.py input.png

# write to a folder
python palette_map.py input.png --outdir out/

# force a mode
python palette_map.py input.png --mode pixel
python palette_map.py input.png --mode photo
```

Output is `<name>_wplace.png` unless you set `--outdir`.

## Modes

### pixel

Fast global matching in OKLab and OKLCh with a few guards for lightness and neutrals. Good for sprites, icons, UI, and flat colors.

### photo

Dithered remap with error diffusion in Lab-L. Has a tiny 2 color micro mix that helps gradients and skin. Good for photos, soft shading, and noisy sources.

## Limit how many colors are used

This caps the palette for the current image. Use this when you want a tighter look.

```bash
# pick a good K automatically from the image
python palette_map.py input.png --limit

# hard cap to K colors
python palette_map.py input.png --limit 8
```

Tip: `--limit` works in both modes. The code picks the top colors the image actually uses, not just the first K in the palette.

## Scaling

You can resize by height. Width follows to keep the aspect.

```bash
# shrink so height <= 256
python palette_map.py input.png --height 256
```

### Resampler

```bash
# auto picks nearest for pixel, lanczos for photo
python palette_map.py input.png --resample auto

# or set it yourself
python palette_map.py input.png --resample nearest
python palette_map.py input.png --resample bilinear
python palette_map.py input.png --resample bicubic
python palette_map.py input.png --resample lanczos
```

## Debug and color report

Use `--debug` to see what the mapper is doing and a per color usage summary.

```bash
python palette_map.py input.png --debug
```

You will see the mode, palette size, a small stats table, and a list like:

```text
=== input.png ===
Mode: pixel
Wrote input_wplace.png | size=178x100 | palette_size=63
Colours used:
  #000000  Black: 9495
  #3c3c3c  Dark Gray: 110
  #787878  Gray: 38
  #aaaaaa  Medium Gray: 20
  #333941  Dark Slate: 5
Total pixels: 9668
total  197.5ms  (load=61.6ms, resize=2.8ms, mode=134.1ms, save=1.8ms)
```

## Performance

- `--jobs` runs multiple input files in parallel.
- `--workers` sets internal worker threads for heavy steps.

Example:

```bash
python palette_map.py images/ --jobs 4 --workers 8
```

## Notes

- Alpha is preserved. Only pixels with alpha > 0 are recolored.
- Output is PNG.
- The mapper will fall back to a simple nearest match if something fails.
- Photo does take a while for bigger photos but it is worth it.

### Why two modes

Pixel art wants crisp edges and controlled lightness. Photos want smooth tone and less banding. The code uses different scoring for each, so try both and pick what looks right.

### I refined this, so it is a bit opinionated

Naming is simple. Defaults are safe. If you want to push it, tweak `palette_map/photo/dither.py` and `palette_map/pixel/run.py`. The comments point at the important knobs. I tested this with the files I am working on.

---

## CLI arguments index

All flags are optional unless noted.

- `SRC` Path to an input image or a folder of images. Required.

- `--outdir PATH` Directory to write outputs. Default is alongside the input file.

- `--mode {auto,pixel,photo}` Mapping mode. Default `auto` (small images -> pixel, else photo).

- `--height H` Resize so the output height is at most `H` pixels. Width scales to keep aspect. Omit to keep original size.

- `--resample {auto,nearest,bilinear,bicubic,lanczos}` Filter for resizing. Default `auto` (`nearest` for pixel, `lanczos` for photo).

- `--limit [K]` Limit the working palette for this image.

  - Use `--limit` with no number to auto-pick a good K from the image.
  - Use `--limit K` to hard-cap to K colors (minimum 2). Works in both modes.

- `--jobs N` Number of input files to process in parallel. Default `2`.

- `--workers N` Internal worker threads for heavy steps. Default is your CPU count.

- `--debug` Print detailed mapping stats and a per-color usage report.

Examples:

```bash
python palette_map.py input.png --mode photo --limit 12 --height 512 --resample lanczos
python palette_map.py images/ --jobs 4 --workers 8 --debug
```

## Future plans

- I might turn this into a website if i feel like it
- Actually use this to paint instead of refine this
- um idk
