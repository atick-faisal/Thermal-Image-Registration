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

**Three warp models, one fit path** (TASKS.md P3-4a). The model is a parameter, not a second
implementation: each of the three dispatches to its OpenCV solver, lifts the result to 3x3 and
then runs the *identical* validity check, inlier gate and reprojection-error tail. Anything
that gave the restricted models their own tail would be comparing rows measured differently,
which is the failure ``warp/homography.py`` exists to prevent one layer down.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.config import EstimateConfig, Estimator, WarpModel
from cmreg.warp import (
    MIN_CORRESPONDENCES,
    MIN_INLIERS,
    ImageArray,
    WarpError,
    check_homography,
    lift,
    warp_points,
)

FloatArray = NDArray[np.float64]

_METHOD_FLAGS: dict[Estimator, int] = {
    Estimator.MAGSAC: cv2.USAC_MAGSAC,
    Estimator.RANSAC: cv2.RANSAC,
    Estimator.LMEDS: cv2.LMEDS,
    Estimator.PROSAC: cv2.USAC_PROSAC,
}

# Which estimators can actually fit which model. **Measured against opencv 5.0.0, not derived**
# -- there is no principled reason a preemptive-RANSAC variant cannot fit a 4-DoF model, and the
# restriction is an implementation gap in `estimateAffinePartial2D`, which accepts only RANSAC
# and LMEDS and raises
#
#     ptsetreg.cpp:1155: error: (-5:Bad argument) Unknown or unsupported robust estimation
#     method in function 'estimateAffinePartial2D'
#
# for every USAC method including the anchor's MAGSAC (TASKS.md P3-4a F39). `estimateAffine2D`
# has no such gap. The consequence lands on stage E rather than here: the anchor estimator
# cannot be held fixed across the warp axis, so that stage needs a control column under an
# estimator all three models admit.
#
# Asserted by `tests/test_warp_models.py` rather than only documented, so a future OpenCV that
# closes the gap fails the suite instead of leaving a stale exclusion standing in the paper.
_USAC_FREE = frozenset({Estimator.RANSAC, Estimator.LMEDS})
SUPPORTED_ESTIMATORS: dict[WarpModel, frozenset[Estimator]] = {
    WarpModel.HOMOGRAPHY: frozenset(Estimator),
    WarpModel.AFFINE: frozenset(Estimator),
    WarpModel.SIMILARITY: _USAC_FREE,
}

# The token a row carries when this run's (model, estimator) is one of the holes above. Greppable
# and distinct from `estimator_needs_confidence`, because the two have different causes: that one
# is a property of the *matcher*, this one of the estimator/model pair alone.
UNSUPPORTED_REASON = "estimator_unsupported_for_warp"


class EstimateError(ValueError):
    """Raised when an estimator is asked for something it cannot do -- PROSAC without
    confidences, say. Distinct from a *failed* estimate, which is a value, not an exception."""


def supports(model: WarpModel, method: Estimator) -> bool:
    """Whether ``method`` can fit ``model`` at all. See :data:`SUPPORTED_ESTIMATORS`."""
    return method in SUPPORTED_ESTIMATORS[model]


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


def estimate_warp(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    config: EstimateConfig,
    confidence: NDArray[np.floating] | None = None,
) -> Estimate:
    """Fit ``config.warp_model`` mapping ``src`` onto ``dst``, robustly.

    Named for the model rather than for the homography (it was ``estimate_homography`` before
    TASKS.md P3-4a): the returned matrix is 3x3 whichever model was fitted, and a name promising
    a homography would be a lie on two of the three.

    ``confidence`` is the per-match score. PROSAC requires it -- it draws its minimal samples
    in descending score order, so feeding it an arbitrary order degrades it to plain RANSAC
    while still being reported as PROSAC. That silent degradation would put a wrong row in the
    P3-10 table, so it raises instead.

    An estimator that cannot fit the requested model raises for the same reason: it fails
    identically on all 300 pairs of a single-variant run, so saying so once at the first pair is
    strictly better than 300 identical rows. Inside a *sweep* that is the wrong trade and the
    runner catches it beforehand (``eval/runner.py::_unsupported``), because aborting there would
    discard the variants that ran fine after the matching they share had been paid for.
    """
    source = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise EstimateError(f"correspondence counts differ: {source.shape} vs {target.shape}")

    model = config.warp_model
    if not supports(model, config.method):
        raise EstimateError(
            f"{config.method.value} cannot fit a {model.value} warp: OpenCV supports only "
            f"{sorted(e.value for e in SUPPORTED_ESTIMATORS[model])} for it. Choose another "
            f"estimator or another warp model (TASKS.md P3-4a)."
        )

    n_matches = len(source)
    if n_matches < MIN_CORRESPONDENCES[model]:
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

    matrix, mask = _fit(source, target, config)
    if matrix is None or mask is None:
        # Below the model's minimum `cv2.estimateAffine2D` returns `None` rather than raising
        # (opencv 5.0.0), so this is a real path and not only a defensive one.
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
    # `MIN_INLIERS`, not this model's own minimum: `warp/models.py` records why the gate is one
    # number across the axis rather than three.
    if n_inliers < MIN_INLIERS:
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


def _fit(
    source: FloatArray, target: FloatArray, config: EstimateConfig
) -> tuple[FloatArray | None, ImageArray | None]:
    """Run the model's OpenCV solver and return its transform as 3x3, plus the inlier mask.

    Every call is **by keyword**: PLAN.md §15A records the upstream harness losing its
    confidence value to `cv2.findHomography`'s fifth positional slot being `mask`. The two
    affine entry points have a different signature again, which is precisely why none of the
    three is called positionally here.
    """
    common = {
        "method": _METHOD_FLAGS[config.method],
        # LMEDS minimises the median residual instead of thresholding, so this never reaches
        # its solve -- but OpenCV still applies it to the returned `mask`, so it moves
        # `n_inliers` and can trip the inlier gate on a fit that was perfectly good (measured on
        # opencv 5.0.0, TASKS.md P3-10). An LMEDS threshold row is flat in `h` and not in the
        # counts.
        "ransacReprojThreshold": config.threshold_px,
        "maxIters": config.max_iters,
        "confidence": config.confidence,
    }
    if config.warp_model is WarpModel.HOMOGRAPHY:
        matrix, mask = cv2.findHomography(srcPoints=source, dstPoints=target, **common)
        return (None if matrix is None else np.asarray(matrix, dtype=np.float64)), mask

    solver = (
        cv2.estimateAffine2D
        if config.warp_model is WarpModel.AFFINE
        else cv2.estimateAffinePartial2D
    )
    affine, mask = solver(source, target, **common)
    # Lifted here rather than by the caller, so `estimate_warp`'s validity check, inlier gate and
    # reprojection error are the one code path all three models pass through. The `asarray` is
    # the same narrowing every other cv2 call site in this package performs: OpenCV is typed as
    # returning integer-or-floating, and the solver's output is float64 by construction.
    if affine is None:
        return None, mask
    return lift(np.asarray(affine, dtype=np.float64)), mask


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
