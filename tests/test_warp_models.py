"""The parametric warp models (TASKS.md P3-4a): homography, affine, similarity.

Three things are pinned here, and only the first is routine.

1. Each model recovers its own transform, cleanly and under 40% outliers -- the shape
   ``tests/test_estimate.py`` established for P3-3.
2. **A restricted model is actually restricted.** Fitting a sheared correspondence set under
   ``similarity`` must *fail* to recover it, and the matrix that comes back must satisfy the
   similarity constraints. Nothing else in the suite would notice a dispatch that quietly fitted
   a homography for every model: every table would look plausible and every number in it would
   be wrong. This is the analogue of stage D's LMEDS integrity check.
3. **The (model, estimator) capability gap is asserted, not just documented.** It is a property
   of the installed OpenCV, so a build that closes it must fail this suite rather than leave a
   stale exclusion standing in the paper.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cmreg.config import EstimateConfig, Estimator, GTConfig, WarpModel
from cmreg.estimate import SUPPORTED_ESTIMATORS, EstimateError, estimate_warp, supports
from cmreg.gt import generator, sample_homography
from cmreg.metrics import corner_error, model_floor
from cmreg.warp import (
    DEGREES_OF_FREEDOM,
    MIN_CORRESPONDENCES,
    WarpError,
    fit_least_squares,
    lift,
    warp_points,
)

SHAPE = (200, 240)
N_POINTS = 200
OUTLIER_FRACTION = 0.4

# A 6-DoF affine with real shear (the 0.2 term) and a non-uniform scale, so it is outside the
# similarity family by a margin no estimator can absorb -- which is what makes it a falsifier.
AFFINE = np.array([[1.10, 0.20, 30.0], [-0.05, 0.95, -12.0], [0.0, 0.0, 1.0]])
# 4 DoF: a rotation-and-uniform-scale (0.9, -0.3 / 0.3, 0.9) plus a translation.
SIMILARITY = np.array([[0.90, -0.30, 40.0], [0.30, 0.90, -20.0], [0.0, 0.0, 1.0]])


def _points(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, [SHAPE[1], SHAPE[0]], size=(N_POINTS, 2))


def _contaminate(dst: np.ndarray, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """40% of the targets displaced by up to 60 px, and marked low-confidence."""
    rng = np.random.default_rng(seed)
    index = rng.choice(len(dst), int(len(dst) * OUTLIER_FRACTION), replace=False)
    noisy = dst.copy()
    noisy[index] += rng.uniform(-60.0, 60.0, size=(len(index), 2))
    scores = np.full(len(dst), 0.9)
    scores[index] = 0.3
    return noisy, scores


def _truth(model: WarpModel) -> np.ndarray:
    if model is WarpModel.HOMOGRAPHY:
        return sample_homography(GTConfig(), generator(0, 0), SHAPE)
    return AFFINE if model is WarpModel.AFFINE else SIMILARITY


def _anchor(model: WarpModel) -> Estimator:
    """An estimator this model admits. MAGSAC where possible, RANSAC for similarity."""
    return Estimator.MAGSAC if supports(model, Estimator.MAGSAC) else Estimator.RANSAC


# --- 1. each model recovers its own transform ----------------------------------------------


@pytest.mark.parametrize("model", list(WarpModel))
def test_clean_correspondences_recover_the_model_exactly(model: WarpModel) -> None:
    truth = _truth(model)
    src = _points()
    config = EstimateConfig(warp_model=model, method=_anchor(model))
    estimate = estimate_warp(src, warp_points(src, truth), config)
    assert estimate.h is not None
    assert corner_error(estimate.h, truth, SHAPE) < 1e-3


@pytest.mark.parametrize("model", list(WarpModel))
def test_forty_percent_outliers_still_recover_the_model(model: WarpModel) -> None:
    truth = _truth(model)
    src = _points()
    noisy, scores = _contaminate(warp_points(src, truth))
    config = EstimateConfig(warp_model=model, method=_anchor(model))
    estimate = estimate_warp(src, noisy, config, scores)
    assert estimate.h is not None
    assert corner_error(estimate.h, truth, SHAPE) < 1.0


@pytest.mark.parametrize("model", list(WarpModel))
def test_least_squares_recovers_the_model_from_clean_points(model: WarpModel) -> None:
    truth = _truth(model)
    src = _points()
    fitted = fit_least_squares(src, warp_points(src, truth), model)
    assert corner_error(fitted, truth, SHAPE) < 1e-6


# --- 2. a restricted model is actually restricted --------------------------------------------


def test_a_similarity_fit_cannot_recover_a_sheared_affine() -> None:
    """The falsification. If this passes with a small error, the dispatch is not restricting.

    `AFFINE` carries shear and non-uniform scale, neither of which a 4-DoF similarity can
    express, so the *correct* behaviour is a large residual. Asserting a floor rather than a
    ceiling is the point: every other test in this file would still pass if `similarity`
    silently fitted a homography.
    """
    src = _points()
    dst = warp_points(src, AFFINE)
    similarity = estimate_warp(
        src, dst, EstimateConfig(warp_model=WarpModel.SIMILARITY, method=Estimator.RANSAC)
    )
    affine = estimate_warp(
        src, dst, EstimateConfig(warp_model=WarpModel.AFFINE, method=Estimator.MAGSAC)
    )

    assert affine.h is not None and corner_error(affine.h, AFFINE, SHAPE) < 1e-3
    assert similarity.h is not None
    assert corner_error(similarity.h, AFFINE, SHAPE) > 1.0


@pytest.mark.parametrize("method", [Estimator.RANSAC, Estimator.LMEDS])
def test_a_similarity_fit_obeys_the_similarity_constraints(method: Estimator) -> None:
    """``a00 == a11`` and ``a01 == -a10``: no shear, no aspect change, and the bottom row exact.

    Checked on the *matrix* rather than on an error, because a fit can be numerically close to
    a similarity while being parameterised as something wider -- and it is the parameterisation
    that stage E is comparing.
    """
    src = _points()
    dst, _ = _contaminate(warp_points(src, SIMILARITY))
    estimate = estimate_warp(
        src, dst, EstimateConfig(warp_model=WarpModel.SIMILARITY, method=method)
    )
    assert estimate.h is not None
    h = estimate.h
    assert h[0, 0] == pytest.approx(h[1, 1], abs=1e-9)
    assert h[0, 1] == pytest.approx(-h[1, 0], abs=1e-9)
    assert h[2].tolist() == [0.0, 0.0, 1.0]


def test_an_affine_fit_leaves_the_projective_row_exact() -> None:
    """An affine has no perspective term, so the bottom row must be exactly ``[0, 0, 1]``.

    Not approximately: it comes from `lift`, which sets it, rather than from a solver.
    """
    src = _points()
    dst, _ = _contaminate(warp_points(src, AFFINE))
    estimate = estimate_warp(src, dst, EstimateConfig(warp_model=WarpModel.AFFINE))
    assert estimate.h is not None
    assert estimate.h[2].tolist() == [0.0, 0.0, 1.0]


def test_degrees_of_freedom_are_ordered_and_minimums_follow_them() -> None:
    """Each point gives two equations, so the minimum is ``ceil(dof / 2)``."""
    for model, dof in DEGREES_OF_FREEDOM.items():
        assert MIN_CORRESPONDENCES[model] == -(-dof // 2)
    assert DEGREES_OF_FREEDOM[WarpModel.HOMOGRAPHY] > DEGREES_OF_FREEDOM[WarpModel.AFFINE]
    assert DEGREES_OF_FREEDOM[WarpModel.AFFINE] > DEGREES_OF_FREEDOM[WarpModel.SIMILARITY]


# --- 3. the capability gap is a measurement --------------------------------------------------


@pytest.mark.parametrize("model", list(WarpModel))
@pytest.mark.parametrize("method", list(Estimator))
def test_the_capability_table_matches_what_opencv_actually_does(
    model: WarpModel, method: Estimator
) -> None:
    """`SUPPORTED_ESTIMATORS` is asserted against the installed OpenCV, not trusted.

    The gap is an implementation limit in `estimateAffinePartial2D`, not a mathematical one, so
    it can close in a future release -- and a stale exclusion in the paper is worse than none.
    """
    src = _points()
    dst = warp_points(src, _truth(model))
    scores = np.linspace(1.0, 0.5, len(src))
    config = EstimateConfig(warp_model=model, method=method)
    if supports(model, method):
        assert not estimate_warp(src, dst, config, scores).failed
        return
    with pytest.raises(EstimateError, match="cannot fit"):
        estimate_warp(src, dst, config, scores)


def test_only_the_similarity_model_has_a_capability_gap() -> None:
    """States the measured shape of the table, so a silent widening of it fails here."""
    assert SUPPORTED_ESTIMATORS[WarpModel.HOMOGRAPHY] == frozenset(Estimator)
    assert SUPPORTED_ESTIMATORS[WarpModel.AFFINE] == frozenset(Estimator)
    assert SUPPORTED_ESTIMATORS[WarpModel.SIMILARITY] == frozenset(
        {Estimator.RANSAC, Estimator.LMEDS}
    )


def test_opencv_partial_affine_still_rejects_usac_directly() -> None:
    """The upstream behaviour the table encodes, exercised without going through our layer.

    `ptsetreg.cpp:1155: (-5) Unknown or unsupported robust estimation method` (opencv 5.0.0).
    """
    src = _points()
    dst = warp_points(src, SIMILARITY)
    with pytest.raises(cv2.error, match=r"[Uu]nsupported"):
        cv2.estimateAffinePartial2D(src, dst, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0)


# --- the model floor -------------------------------------------------------------------------


def test_the_homography_floor_is_zero() -> None:
    """The model contains the truth, so its floor is exactly zero.

    The check that the floor is fitting the *requested* model: an implementation that used a
    restricted fit throughout would give this a nonzero value.
    """
    truth = sample_homography(GTConfig(), generator(0, 0), SHAPE)
    assert model_floor(truth, SHAPE, WarpModel.HOMOGRAPHY) == pytest.approx(0.0, abs=1e-9)


def test_an_affine_truth_floors_affine_at_zero_and_similarity_above_it() -> None:
    """A truth inside the affine family costs affine nothing and similarity its shear."""
    assert model_floor(AFFINE, SHAPE, WarpModel.AFFINE) == pytest.approx(0.0, abs=1e-9)
    assert model_floor(AFFINE, SHAPE, WarpModel.SIMILARITY) > 1.0
    assert model_floor(SIMILARITY, SHAPE, WarpModel.SIMILARITY) == pytest.approx(0.0, abs=1e-9)


def test_the_floor_is_a_lower_bound_on_what_an_estimator_achieves() -> None:
    """The claim the floor makes: no fit of that model scores below it.

    Exercised against a real robust fit on clean correspondences of a *projective* truth, which
    is the case stage E actually runs -- the estimator has nothing to blame but the model.
    """
    truth = sample_homography(GTConfig(), generator(0, 3), SHAPE)
    src = _points()
    dst = warp_points(src, truth)
    for model in (WarpModel.AFFINE, WarpModel.SIMILARITY):
        floor = model_floor(truth, SHAPE, model)
        estimate = estimate_warp(src, dst, EstimateConfig(warp_model=model, method=_anchor(model)))
        assert estimate.h is not None
        assert corner_error(estimate.h, truth, SHAPE) >= floor - 1e-9


def test_lift_rejects_a_matrix_that_is_not_two_by_three() -> None:
    with pytest.raises(WarpError, match="2x3"):
        lift(np.eye(3))
