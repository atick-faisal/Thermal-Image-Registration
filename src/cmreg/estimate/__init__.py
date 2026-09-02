"""Robust estimation (TASKS.md P3-3): USAC_MAGSAC, RANSAC, LMEDS, PROSAC, across the three
parametric warp models of P3-4a. pydegensac is outstanding -- it is a separate wheel."""

from __future__ import annotations

from cmreg.estimate.robust import (
    SUPPORTED_ESTIMATORS,
    UNSUPPORTED_REASON,
    Estimate,
    EstimateError,
    estimate_warp,
    supports,
    symmetric_reprojection_error,
)
from cmreg.estimate.select import (
    NEEDS_CONFIDENCE_REASON,
    needs_confidence,
    selected_indices,
)

__all__ = [
    "NEEDS_CONFIDENCE_REASON",
    "SUPPORTED_ESTIMATORS",
    "UNSUPPORTED_REASON",
    "Estimate",
    "EstimateError",
    "estimate_warp",
    "needs_confidence",
    "selected_indices",
    "supports",
    "symmetric_reprojection_error",
]
