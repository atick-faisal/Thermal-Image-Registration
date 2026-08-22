"""The homography warp model: the shared ``apply_warp`` every downstream consumer uses.

TASKS.md P3-4 asks for homography, affine, similarity, TPS and homography + residual flow
behind one interface. Only homography exists today; the rest land with P3-4. What matters
now is that there is exactly **one** implementation of "apply this transform to this image"
and one of "map these points" -- a second one with its own coordinate convention is how a
benchmark ends up comparing methods that were never measured the same way.

Convention, fixed here and relied on everywhere: ``H`` maps **source pixel coordinates to
target pixel coordinates**, points are ``(x, y)`` in pixel units with the pixel *centre* at
integer coordinates, and image arrays are indexed ``[y, x]``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
# Exactly what OpenCV's `MatLike` covers. `NDArray[np.generic]` is wider (it admits
# object and string dtypes) and fails type-checking at every cv2 call site.
ImageArray = NDArray[np.integer[Any] | np.floating[Any]]

# Below this the homography is not usefully invertible and every derived quantity -- warped
# corners, dense fields, reprojection errors -- is numerical noise.
_MIN_DETERMINANT = 1e-8


class WarpError(ValueError):
    """Raised for a degenerate or malformed transform."""


def check_homography(h: FloatArray) -> FloatArray:
    """Validate shape and invertibility, returning ``h`` as float64.

    Fails here rather than letting a singular matrix propagate: ``cv2.warpPerspective`` on a
    degenerate ``H`` returns a black image rather than raising, which reads downstream as a
    catastrophic registration failure instead of a bad input.
    """
    matrix = np.asarray(h, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise WarpError(f"homography must be 3x3, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise WarpError("homography contains non-finite entries")
    if abs(float(np.linalg.det(matrix))) < _MIN_DETERMINANT:
        raise WarpError(f"homography is singular (|det| < {_MIN_DETERMINANT})")
    return matrix


def warp_points(points: NDArray[np.floating], h: FloatArray) -> FloatArray:
    """Map ``(N, 2)`` pixel coordinates through ``h``."""
    matrix = check_homography(h)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    # cv2 is typed as returning integer-or-floating; the dtype is float64 here by
    # construction, and saying so keeps every caller's annotations honest.
    return np.asarray(cv2.perspectiveTransform(pts, matrix), dtype=np.float64).reshape(-1, 2)


def corners(shape: tuple[int, int]) -> FloatArray:
    """The four image corners in ``(x, y)``, clockwise from the top-left.

    ``(width, height)`` rather than ``(width - 1, height - 1)``: the corners are the outer
    edges of the sampling grid, which is the convention ``cv2.warpPerspective`` uses when it
    maps a ``(w, h)`` destination size.
    """
    height, width = shape
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def apply_warp(
    image: ImageArray,
    h: FloatArray,
    out_shape: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageArray:
    """Warp ``image`` (indexed ``[y, x]``) through ``h`` into an ``out_shape`` canvas."""
    matrix = check_homography(h)
    height, width = image.shape[:2] if out_shape is None else out_shape
    return cv2.warpPerspective(image, matrix, (width, height), flags=interpolation)
