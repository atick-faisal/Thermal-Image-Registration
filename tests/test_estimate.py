"""Robust estimation: every estimator must recover a known homography, cleanly and under
outliers, and every failure mode must be a value rather than an exception."""

from __future__ import annotations

import numpy as np
import pytest

from cmreg.config import EstimateConfig, Estimator, GTConfig
from cmreg.estimate import EstimateError, estimate_homography, symmetric_reprojection_error
from cmreg.gt import generator, sample_homography
from cmreg.metrics import corner_error
from cmreg.warp import warp_points

SHAPE = (200, 240)
N_POINTS = 200
OUTLIER_FRACTION = 0.4
ESTIMATORS = list(Estimator)


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    return sample_homography(GTConfig(), generator(0, 0), SHAPE)


@pytest.fixture(scope="module")
def correspondences(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    src = rng.uniform(0.0, [SHAPE[1], SHAPE[0]], size=(N_POINTS, 2))
    return src, warp_points(src, truth), rng.uniform(0.5, 1.0, size=N_POINTS)


@pytest.fixture(scope="module")
def contaminated(
    correspondences: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """40% of the correspondences displaced by up to 60 px, and marked low-confidence.

    The confidences are perturbed alongside so PROSAC is exercised on a *useful* ordering --
    an estimator whose whole premise is that scores predict correctness cannot be tested on
    scores that do not.
    """
    src, dst, confidence = correspondences
    rng = np.random.default_rng(2)
    index = rng.choice(N_POINTS, int(N_POINTS * OUTLIER_FRACTION), replace=False)
    noisy = dst.copy()
    noisy[index] += rng.uniform(-60.0, 60.0, size=(len(index), 2))
    scores = confidence.copy()
    scores[index] *= 0.3
    return src, noisy, scores


@pytest.mark.parametrize("method", ESTIMATORS)
def test_clean_correspondences_recover_the_homography_exactly(
    method: Estimator,
    truth: np.ndarray,
    correspondences: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    src, dst, confidence = correspondences
    estimate = estimate_homography(src, dst, EstimateConfig(method=method), confidence)
    assert not estimate.failed
    assert estimate.h is not None
    assert corner_error(estimate.h, truth, SHAPE) < 1e-3


@pytest.mark.parametrize("method", ESTIMATORS)
def test_forty_percent_outliers_are_rejected(
    method: Estimator,
    truth: np.ndarray,
    contaminated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    src, dst, confidence = contaminated
    estimate = estimate_homography(src, dst, EstimateConfig(method=method), confidence)
    assert not estimate.failed
    assert estimate.h is not None
    assert corner_error(estimate.h, truth, SHAPE) < 1.0
    # The 60% clean correspondences, and no more than a stray outlier or two.
    assert 0.55 <= estimate.inlier_ratio <= 0.65


@pytest.mark.parametrize("method", ESTIMATORS)
def test_the_inlier_mask_indexes_the_callers_correspondences(
    method: Estimator,
    contaminated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """PROSAC reorders its inputs by confidence; the mask it returns is in *that* order. If
    the reordering is not undone, `inlier_mask` silently points at the wrong matches -- which
    nothing downstream would notice until a per-match analysis in Phase 6."""
    src, dst, confidence = contaminated
    estimate = estimate_homography(src, dst, EstimateConfig(method=method), confidence)
    assert estimate.h is not None
    assert estimate.inlier_mask.shape == (N_POINTS,)
    residual = np.linalg.norm(warp_points(src, estimate.h) - dst, axis=1)
    assert residual[estimate.inlier_mask].max() < 10.0
    assert residual[~estimate.inlier_mask].mean() > residual[estimate.inlier_mask].mean()


def test_too_few_matches_is_a_value_not_an_exception() -> None:
    estimate = estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)), EstimateConfig())
    assert estimate.failed
    assert estimate.failure_reason == "too_few_matches"
    assert estimate.inlier_mask.shape == (3,)
    assert np.isnan(estimate.reproj_err)


def test_degenerate_correspondences_fail_rather_than_returning_a_singular_matrix() -> None:
    """All four points coincident. `cv2.warpPerspective` on a singular H returns a black
    image rather than raising, which reads downstream as a catastrophic registration failure
    instead of a bad estimate."""
    points = np.zeros((8, 2))
    estimate = estimate_homography(points, points, EstimateConfig())
    assert estimate.failed


def test_prosac_without_confidences_refuses_to_run(
    correspondences: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Silently degrading to RANSAC would put a row labelled PROSAC in the P3-10 table that
    was not produced by PROSAC."""
    src, dst, _ = correspondences
    with pytest.raises(EstimateError, match="PROSAC"):
        estimate_homography(src, dst, EstimateConfig(method=Estimator.PROSAC))


def test_mismatched_correspondence_counts_are_rejected() -> None:
    with pytest.raises(EstimateError, match="counts differ"):
        estimate_homography(np.zeros((5, 2)), np.zeros((6, 2)), EstimateConfig())


def test_reprojection_error_is_symmetric(truth: np.ndarray) -> None:
    """Forward-only would reward a homography that compresses one image into a corner: tiny
    forward error, enormous backward one."""
    rng = np.random.default_rng(4)
    src = rng.uniform(0.0, [SHAPE[1], SHAPE[0]], size=(20, 2))
    dst = warp_points(src, truth)
    assert symmetric_reprojection_error(src, dst, truth) < 1e-6
    forward = symmetric_reprojection_error(src, dst, truth)
    backward = symmetric_reprojection_error(dst, src, np.linalg.inv(truth))
    assert forward == pytest.approx(backward, abs=1e-9)


def test_confidence_reaches_the_solver() -> None:
    """The reason this layer is ours: PLAN.md §15A records the upstream harness passing its
    confidence into `cv2.findHomography`'s `mask` slot, so the value never arrived. Passing an
    out-of-range confidence must therefore be rejected by the config, not ignored."""
    from cmreg.config import ConfigError

    with pytest.raises((ValueError, ConfigError)):
        EstimateConfig(confidence=1.5)
