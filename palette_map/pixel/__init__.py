# palette_map/pixel/__init__.py
"""
Pixel-mode API.

Provides:
  run_pixel(img_rgb, alpha, pal_rgb, pal_lab, pal_lch, *, debug=False, workers=1) -> U8Image
    Map an RGBA image to the palette using pixel-art heuristics.

    Args:
      img_rgb : uint8 [H,W,3]
      alpha   : uint8 [H,W]
      pal_rgb : uint8 [P,3]
      pal_lab : float32 [P,3]
      pal_lch : float32 [P,3]
      debug   : bool, print selection and mismatch stats
      workers : int, kept for API parity (not used for parallelism)

    Returns:
      uint8 [H,W,3] mapped image. Alpha is preserved by caller.

    Notes:
      - Ensembles several scoring flavours (base/light/hue/chroma/neutral).
      - Assigns grey-ish sources to distinct neutral entries when safe.
      - Light decongestion spreads overused palette slots within tolerance.
"""

from .run import run_pixel

__all__ = ["run_pixel"]
