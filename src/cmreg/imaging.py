"""Image IO, kept in one place so every consumer decodes identically.

A second decoder with its own scaling or colour convention is how two metrics computed on
"the same image" end up incomparable -- the reasoning is
``../Thermal-To-Optical-Translation/src/t2o/data/dataset.py``'s, applied here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class ImagingError(OSError):
    """Raised when an image cannot be read."""


def read_shape(path: Path | str) -> tuple[int, int]:
    """``(height, width)`` without decoding the pixel data.

    Generating ground truth needs only the geometry, and for a 2,500-pair set the difference
    between reading a header and decoding a JPEG is minutes.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError as exc:
        raise ImagingError(f"could not read image header: {path} ({exc})") from exc
    return height, width


def read_gray(path: Path | str) -> NDArray[np.uint8]:
    """Decode as single-channel uint8, indexed ``[y, x]``."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise ImagingError(f"could not read image: {path} ({exc})") from exc
