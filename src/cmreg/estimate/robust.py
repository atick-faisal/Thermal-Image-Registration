"""Robust homography estimation (TASKS.md P3-3), owned rather than borrowed.

PLAN.md §15A records why this layer is ours and not the matcher harness's: ``vismatch``'s
``BaseMatcher.compute_ransac`` calls

    cv2.findHomography(src, dst, cv2.USAC_MAGSAC, thresh, ransac_conf, ransac_iters)

and OpenCV's fifth positional parameter is ``mask``, not ``confidence`` -- so its configured
confidence never reaches the solver and silently stays at the 0.995 default while its
iteration cap lands in the right slot by accident. **Every call here is by keyword.** The
estimator sweep of P3-10 is meaningless if the knob being swept is not the knob being read.

Failure is a first-class outcome. A pair with three matches, or a solver that returns
``None``, or a matrix that is singular, produces an ``Estimate`` with ``h is None`` and a
stated reason -- never an exception, and never a silently dropped row (TASKS.md X-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.config import EstimateConfig, Estimator
from cmreg.warp import WarpError, check_homography, warp_points

FloatArray = NDArray[np.float64]

# A homography has 8 degrees of freedom; four point correspondences is the exact minimum.
MIN_CORRESPONDENCES = 4

_METHOD_FLAGS: dict[Estimator, int] = {
    Estimator.MAGSAC: cv2.USAC_MAGSAC,
    Estimator.RANSAC: cv2.RANSAC,
    Estimator.LMEDS: cv2.LMEDS,
    Estimator.PROSAC: cv2.USAC_PROSAC,
}


class EstimateError(ValueError):
    """Raised when an estimator is asked for something it cannot do -- PROSAC without
    confidences, say. Distinct from a *failed* estimate, which is a value, not an exception."""


@dataclass(frozen=True, slots=True)
class Estimate:
    """The outcome of one robust fit."""

    # None when the fit failed. Callers branch on this, not on `n_inliers`.
    h: FloatArray | None
    # (N,) over the input correspondences. All-False when the fit failed.
    inlier_mask: NDArray[np.bool_]
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    # Mean symmetric reprojection error over the inliers, in pixels. PLAN.md §6.4: gameable
    # -- a degenerate 4-point fit scores near zero and a smaller threshold lowers it
    # artificially -- so it is reported and never led with.
    reproj_err: float
    # None on success. Set to a short, greppable token on failure.
    failure_reason: str | None

    @property
    def failed(self) -> bool:
        return self.h is None


def _failed(reason: str, n_matches: int) -> Estimate:
    return Estimate(
        h=None,
        inlier_mask=np.zeros(n_matches, dtype=bool),
        n_matches=n_matches,
        n_inliers=0,
        inlier_ratio=0.0,
        reproj_err=float("nan"),
        failure_reason=reason,
    )


def estimate_homography(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    config: EstimateConfig,
    confidence: NDArray[np.floating] | None = None,
) -> Estimate:
    """Fit the homography mapping ``src`` onto ``dst``.

    ``confidence`` is the per-match score. PROSAC requires it -- it draws its minimal samples
    in descending score order, so feeding it an arbitrary order degrades it to plain RANSAC
    while still being reported as PROSAC. That silent degradation would put a wrong row in the
    P3-10 table, so it raises instead.
    """
    source = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise EstimateError(f"correspondence counts differ: {source.shape} vs {target.shape}")

    n_matches = len(source)
    if n_matches < MIN_CORRESPONDENCES:
        return _failed("too_few_matches", n_matches)

    if config.method is Estimator.PROSAC:
        if confidence is None:
            raise EstimateError(
                "PROSAC needs per-match confidences to order its samples; this matcher "
                "provides none. Choose another estimator or a matcher that scores its matches."
            )
        order = np.argsort(np.asarray(confidence, dtype=np.float64))[::-1]
        source, target = source[order], target[order]
    else:
        order = np.arange(n_matches)

    matrix, mask = cv2.findHomography(
        srcPoints=source,
        dstPoints=target,
        method=_METHOD_FLAGS[config.method],
        # LMEDS minimises the median residual instead of thresholding, so this never reaches
        # its solve -- but OpenCV still applies it to the returned `mask`, so it moves
        # `n_inliers` and can trip the four-inlier gate below on a fit that was perfectly good
        # (measured on opencv 5.0.0, TASKS.md P3-10). An LMEDS threshold row is flat in `h` and
        # not in the counts.
        ransacReprojThreshold=config.threshold_px,
        maxIters=config.max_iters,
        confidence=config.confidence,
    )
    if matrix is None or mask is None:
        return _failed("solver_returned_none", n_matches)

    try:
        # cv2 is typed as returning integer-or-floating; the solver's output is float64 by
        # construction, and saying so keeps every downstream annotation honest.
        matrix = check_homography(np.asarray(matrix, dtype=np.float64))
    except WarpError:
        # A singular fit is what a degenerate (collinear or coincident) inlier set produces.
        # `cv2.warpPerspective` would return an all-black image for it, which reads downstream
        # as a catastrophic registration failure instead of a bad estimate.
        return _failed("degenerate_homography", n_matches)

    sorted_mask = np.asarray(mask, dtype=bool).ravel()
    n_inliers = int(sorted_mask.sum())
    if n_inliers < MIN_CORRESPONDENCES:
        return _failed("too_few_inliers", n_matches)

    # Undo the PROSAC reordering so `inlier_mask` indexes the caller's correspondences.
    inlier_mask = np.zeros(n_matches, dtype=bool)
    inlier_mask[order[sorted_mask]] = True

    return Estimate(
        h=matrix,
        inlier_mask=inlier_mask,
        n_matches=n_matches,
        n_inliers=n_inliers,
        inlier_ratio=n_inliers / n_matches,
        reproj_err=symmetric_reprojection_error(source[sorted_mask], target[sorted_mask], matrix),
        failure_reason=None,
    )


def symmetric_reprojection_error(
    src: NDArray[np.floating], dst: NDArray[np.floating], h: FloatArray
) -> float:
    """Mean of the forward and backward reprojection distances.

    Symmetric rather than forward-only: a homography that compresses one image into a corner
    has a tiny forward error and an enormous backward one, and reporting only the forward
    direction rewards exactly that degenerate fit.
    """
    matrix = check_homography(h)
    forward = np.linalg.norm(warp_points(src, matrix) - dst, axis=1)
    backward = np.linalg.norm(warp_points(dst, np.linalg.inv(matrix)) - src, axis=1)
    return float((forward.mean() + backward.mean()) / 2.0)
