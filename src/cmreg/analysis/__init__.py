"""Aggregation and statistics (TASKS.md P8-1/P8-2). PLAN.md §15E names the sign-flip
permutation test and seeded bootstrap CI to port.

First occupant is the residual decomposition of TASKS.md P1-1b, which is Phase-1 work living
here because it reads the results store rather than producing it.
"""

from __future__ import annotations

from cmreg.analysis.residual import (
    AnalysisError,
    ResidualStructure,
    by_matcher,
    consensus_homography,
    render,
    residual_structure,
)

__all__ = [
    "AnalysisError",
    "ResidualStructure",
    "by_matcher",
    "consensus_homography",
    "render",
    "residual_structure",
]
