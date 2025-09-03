# palette_map/image_io.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

"""
Image I/O helpers (RGBA in sRGB), alpha binarization, and resize utilities.
"""

try:
    from PIL import ImageCms  # ICC conversion if profile present
except Exception:  # pragma: no cover
    ImageCms = None  # type: ignore[assignment]

def binarise_alpha(
    alpha_or_img: np.ndarray | Image.Image, threshold: int = 127
) -> np.ndarray:
    if isinstance(alpha_or_img, Image.Image):
        im = alpha_or_img
        if im.mode == "RGBA":
            a = np.array(im.split()[-1], dtype=np.uint8)
        elif im.mode == "L":
            a = np.array(im, dtype=np.uint8)
        else:
            a = np.array(im.convert("L"), dtype=np.uint8)
    else:
        a = np.asarray(alpha_or_img, dtype=np.uint8)

    mask = a >= np.uint8(threshold)
    out = np.zeros_like(a, dtype=np.uint8)
    out[mask] = 255
    return out


def _convert_to_srgb_rgba(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    icc_bytes = im.info.get("icc_profile")

    if icc_bytes and ImageCms is not None:
        try:
            src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            dst_prof = ImageCms.createProfile("sRGB")
            im2 = ImageCms.profileToProfile(
                im,
                src_prof,
                dst_prof,
                renderingIntent=ImageCms.INTENT_PERCEPTUAL,
                outputMode="RGBA",
            )
            if im2 is None:
                return im.convert("RGBA")
            return im2
        except Exception:
            return im.convert("RGBA")

    return im.convert("RGBA")


def load_image_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im0:
        im = _convert_to_srgb_rgba(im0)
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = binarise_alpha(arr[..., 3])
    return rgb, alpha


def save_image_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> Path:
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    H, W, _ = rgb.shape
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = binarise_alpha(alpha)
    Image.fromarray(out, mode="RGBA").save(path)
    return path


def resize_rgba_height(
    rgb: np.ndarray,
    alpha: np.ndarray,
    dst_h: Optional[int],
    resample: Image.Resampling,
) -> Tuple[np.ndarray, np.ndarray]:
    H0, W0, _ = rgb.shape
    if dst_h is None or dst_h <= 0 or dst_h >= H0:
        return rgb, binarise_alpha(alpha)

    dst_w = int(round(W0 * (dst_h / float(H0))))
    rgba = np.zeros((H0, W0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    im = Image.fromarray(rgba, mode="RGBA")
    im2 = im.resize((dst_w, dst_h), resample=resample)
    arr = np.array(im2, dtype=np.uint8)
    rgb2 = arr[..., :3]
    a2 = binarise_alpha(arr[..., 3])
    return rgb2, a2


def is_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.seek(0)
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False


__all__ = [
    "binarise_alpha",
    "load_image_rgba",
    "save_image_rgba",
    "resize_rgba_height",
    "is_image_file",
]
