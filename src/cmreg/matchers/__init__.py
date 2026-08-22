"""The `Matcher` Protocol and its adapters (TASKS.md P0-3). OpenCV SIFT/ORB/AKAZE today;
`vismatch` is wrapped here next and always called with `skip_ransac=True`, and RIFT/LNIFT are
written fresh against the same Protocol. PLAN.md §15A."""

from __future__ import annotations

from cmreg.matchers.base import (
    MatcherError,
    MatchResult,
    available,
    empty_result,
    get_matcher,
    register,
)

__all__ = [
    "MatchResult",
    "MatcherError",
    "available",
    "empty_result",
    "get_matcher",
    "register",
]
