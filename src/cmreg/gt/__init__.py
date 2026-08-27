"""Ground truth. Tier 1 (synthetic warps) today; Tier 2 (box transfer) and Tier 3 (the
manually annotated gold set) land with TASKS.md P2-8 and P2-10."""

from __future__ import annotations

from cmreg.gt.calibration import (
    CORNER_NAMES,
    CalibrationError,
    ResidualCalibration,
    load_calibration,
    write_calibration,
)
from cmreg.gt.synthetic import (
    DenseGT,
    dense_displacement,
    generator,
    overlap_ratio,
    sample_homography,
    warp_seed,
)

__all__ = [
    "CORNER_NAMES",
    "CalibrationError",
    "DenseGT",
    "ResidualCalibration",
    "dense_displacement",
    "generator",
    "load_calibration",
    "overlap_ratio",
    "sample_homography",
    "warp_seed",
    "write_calibration",
]
