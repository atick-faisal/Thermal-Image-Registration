"""The parametric warp models (TASKS.md P3-4a): homography, affine, similarity.

PLAN.md §4.1 lists five warp models; these are the three that are **3x3 matrices**, and that
is the whole reason they are one task and TPS + residual flow are another. A matrix model
passes unchanged through every consumer the project already has -- ``dense_displacement``,
``corner_error``, the image-diagonal saturation, ``PairRow.h``'s nine floats and P2-12's
``R . inv(H_gt)`` composition. A dense-field model breaks each of them, so it is P3-4b.

What lives here is the *representation*: how many parameters each model has, what the minimum
correspondence count is, how a 2x3 becomes homogeneous, and the non-robust least-squares fit the
model floor is measured with (``warp/floor.py``). The **robust** fit -- and the fact that not
every estimator can perform it -- lives in ``estimate/robust.py``, which owns the ``Estimate``
value and the failure taxonomy.

Convention inherited from ``warp/homography.py``: matrices map source pixels to target pixels,
points are ``(x, y)``, arrays are indexed ``[y, x]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cmreg.config import WarpModel
from cmreg.warp.homography import FloatArray, WarpError, check_homography

# What the model can express, which is what a stage-E row is actually comparing. A homography
# has 8 (the 3x3 is defined up to scale), an affine 6, a similarity 4 -- rotation, uniform
# scale and translation, with no shear and no aspect change.
DEGREES_OF_FREEDOM: dict[WarpModel, int] = {
    WarpModel.HOMOGRAPHY: 8,
    WarpModel.AFFINE: 6,
    WarpModel.SIMILARITY: 4,
}

# The exact minimum for each solver, measured against opencv 5.0.0 rather than derived: below
# it `cv2.estimateAffine2D` returns `None` instead of raising, which `estimate/robust.py` reads
# as a failed fit. Each point contributes two equations, so this is `ceil(dof / 2)`.
MIN_CORRESPONDENCES: dict[WarpModel, int] = {
    WarpModel.HOMOGRAPHY: 4,
    WarpModel.AFFINE: 3,
    WarpModel.SIMILARITY: 2,
}

# The inlier count below which a fit is recorded as failed, **for every model alike**.
#
# It is deliberately not `MIN_CORRESPONDENCES[model]`. A 2-inlier similarity is exactly
# determined: it interpolates its own two points with zero residual and says nothing about the
# image. Gating each model at its own minimum would hand the restricted models extra
# "successes" on precisely the pairs stage E exists to discriminate -- the hard ones -- and
# `reg/success_rate_*` and `reg/failure_rate` would then be comparing populations selected by
# different rules. One gate across the axis costs the restricted models a handful of degenerate
# wins and buys a column that means the same thing in every row.
MIN_INLIERS = 4


def lift(matrix: NDArray[np.floating]) -> FloatArray:
    """Embed a 2x3 affine matrix in the 3x3 homogeneous form everything downstream expects.

    The one place this conversion happens. A second one with its own convention is exactly what
    ``warp/homography.py``'s docstring warns about: a benchmark comparing methods that were
    never measured the same way.
    """
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3):
        raise WarpError(f"affine matrix must be 2x3, got {affine.shape}")
    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2] = affine
    return check_homography(homogeneous)


def fit_least_squares(
    src: NDArray[np.floating], dst: NDArray[np.floating], model: WarpModel
) -> FloatArray:
    """The non-robust best fit of ``model`` to every correspondence given.

    Used to measure what a model *can* achieve, never to score a matcher -- the benchmark path
    is ``estimate/robust.py``. Written in numpy rather than through OpenCV because
    ``cv2.estimateAffine2D`` has no least-squares mode at all: ``method=0`` raises
    ``ptsetreg.cpp:1155 (-5) Unknown or unsupported robust estimation method`` (opencv 5.0.0),
    so its only fits are robust ones, and a floor measured with a robust estimator would be the
    floor of that estimator rather than of the model.

    Both restricted models are **linear in their parameters**, which is what makes this exact:

    * affine, 6 parameters, solving ``[x y 1] . theta = x'`` and the same for ``y'``;
    * similarity, 4 parameters ``(a, b, tx, ty)`` with ``x' = a x - b y + tx`` and
      ``y' = b x + a y + ty`` -- the ``a``/``b`` sharing *is* the constraint that removes shear
      and aspect change, expressed in the design matrix rather than imposed afterwards.
    """
    source = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise WarpError(f"correspondence counts differ: {source.shape} vs {target.shape}")
    if len(source) < MIN_CORRESPONDENCES[model]:
        raise WarpError(
            f"{model.value} needs at least {MIN_CORRESPONDENCES[model]} correspondences, "
            f"got {len(source)}"
        )

    if model is WarpModel.HOMOGRAPHY:
        return _fit_homography_dlt(source, target)

    x, y = source[:, 0], source[:, 1]
    ones, zeros = np.ones_like(x), np.zeros_like(x)
    if model is WarpModel.AFFINE:
        design = np.zeros((2 * len(source), 6), dtype=np.float64)
        design[0::2, :3] = np.stack([x, y, ones], axis=1)
        design[1::2, 3:] = np.stack([x, y, ones], axis=1)
    else:
        design = np.empty((2 * len(source), 4), dtype=np.float64)
        design[0::2] = np.stack([x, -y, ones, zeros], axis=1)
        design[1::2] = np.stack([y, x, zeros, ones], axis=1)

    solution, *_ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
    if model is WarpModel.AFFINE:
        return lift(solution.reshape(2, 3))
    a, b, tx, ty = solution
    return lift(np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64))


def _fit_homography_dlt(source: FloatArray, target: FloatArray) -> FloatArray:
    """Least-squares homography by the DLT, for symmetry with the two restricted models.

    ``cv2.findHomography(method=0)`` would do this, and is not used for one reason: the floor
    it is measured for must come from the same code path for all three models, or a
    "homography's floor is zero" check is testing OpenCV rather than this module.
    """
    n = len(source)
    design = np.zeros((2 * n, 9), dtype=np.float64)
    x, y = source[:, 0], source[:, 1]
    u, v = target[:, 0], target[:, 1]
    ones, zeros = np.ones(n), np.zeros(n)
    design[0::2] = np.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], axis=1)
    design[1::2] = np.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], axis=1)
    # The solution is the null vector, i.e. the right-singular vector of the smallest value.
    _, _, vt = np.linalg.svd(design)
    matrix = vt[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < np.finfo(np.float64).eps:
        raise WarpError("degenerate DLT solution: the fitted homography has h22 = 0")
    return check_homography(matrix / matrix[2, 2])
