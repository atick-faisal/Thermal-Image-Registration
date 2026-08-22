"""Robust estimators (TASKS.md P3-3): USAC_MAGSAC, RANSAC, LMEDS, PROSAC. pydegensac is
outstanding -- it is a separate wheel and lands with the P3-10 sweep."""

from __future__ import annotations

from cmreg.estimate.robust import (
    MIN_CORRESPONDENCES,
    Estimate,
    EstimateError,
    estimate_homography,
    symmetric_reprojection_error,
)

__all__ = [
    "MIN_CORRESPONDENCES",
    "Estimate",
    "EstimateError",
    "estimate_homography",
    "symmetric_reprojection_error",
]
