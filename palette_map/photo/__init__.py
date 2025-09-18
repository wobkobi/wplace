# palette_map/photo/__init__.py
"""
Photo-mode API.

Provides:
  dither_photo(rgb, alpha, items, pal_lab, pal_lch, workers=1, **kwargs)
    Map an RGBA image to the palette using error-diffusion dithering.

    Args:
      rgb       : uint8 [H,W,3]
      alpha     : uint8 [H,W]
      items     : list[PaletteItem]
      pal_lab   : Lab [P,3]
      pal_lch   : Lch [P,3]
      workers   : int, threads for local ops (RGB→Lab, blur) in single-process path

      Kwargs:
        progress       : bool, print smoothed percent+ETA (default True)
        profile        : bool, print timing/counters summary (default False)

        # Optional overlapped block multiprocessing (quality-safe):
        mp_blocks      : bool, enable MP block mode (default False)
        mp_procs       : int|None, process count; default uses available CPUs minus a small reserve
        mp_threads     : int, threads per process for local ops (default 1)
        mp_block_rows  : int, core block height; defaults scale with image size/CPUs
        mp_overlap_rows: int, overlap rows per seam (default 24)

        # Palette candidate control:
        topk           : int, candidate palette prefilter size (auto-scaled by image size)

    Returns:
      uint8 [H,W,3] mapped image. Alpha is preserved by callers.

    Notes:
      - Serpentine Floyd-Steinberg diffusion in Lab-L.
      - Deterministic blue-noise micro-mix for fine texture.
      - Palette-aware scoring in Lab/LCh with local-contrast and shadow terms.
"""

from .dither import dither_photo

__all__ = ["dither_photo"]
