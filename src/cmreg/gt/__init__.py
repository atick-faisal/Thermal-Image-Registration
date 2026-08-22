"""Ground truth. Tier 1 (synthetic warps) today; Tier 2 (box transfer) and Tier 3 (the
manually annotated gold set) land with TASKS.md P2-8 and P2-10."""

from __future__ import annotations

from cmreg.gt.synthetic import (
    DenseGT,
    dense_displacement,
    generator,
    overlap_ratio,
    sample_homography,
    warp_seed,
)

__all__ = [
    "DenseGT",
    "dense_displacement",
    "generator",
    "overlap_ratio",
    "sample_homography",
    "warp_seed",
]
