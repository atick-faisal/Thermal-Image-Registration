"""Warp models (TASKS.md P3-4).

The three parametric ones are implemented (P3-4a); TPS and homography + residual flow are the
dense-field half and land with P3-4b.
"""

from __future__ import annotations

from cmreg.warp.homography import (
    FloatArray,
    ImageArray,
    WarpError,
    apply_warp,
    check_homography,
    corners,
    warp_points,
)
from cmreg.warp.models import (
    DEGREES_OF_FREEDOM,
    MIN_CORRESPONDENCES,
    MIN_INLIERS,
    fit_least_squares,
    lift,
)

__all__ = [
    "DEGREES_OF_FREEDOM",
    "MIN_CORRESPONDENCES",
    "MIN_INLIERS",
    "FloatArray",
    "ImageArray",
    "WarpError",
    "apply_warp",
    "check_homography",
    "corners",
    "fit_least_squares",
    "lift",
    "warp_points",
]
