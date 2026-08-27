"""TASKS.md P2-4 -- the non-negotiable validation of the synthetic warp engine.

Phases 4, 5 and 6 all consume what ``cmreg.gt.synthetic`` produces. If it is subtly wrong,
all three train and evaluate against corrupted ground truth and the failure surfaces months
later. These tests recover a sampled homography from the dense field it induced and assert
the round trip is numerical noise.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cmreg.config import GTConfig
from cmreg.gt import (
    DenseGT,
    dense_displacement,
    generator,
    overlap_ratio,
    sample_homography,
    warp_seed,
)
from cmreg.warp import WarpError, apply_warp, check_homography, corners, warp_points

SHAPE = (48, 64)  # (height, width), non-square


def _identity_field(shape: tuple[int, int]) -> DenseGT:
    return dense_displacement(np.eye(3), shape)


def test_identity_homography_induces_zero_flow() -> None:
    gt = _identity_field(SHAPE)
    assert gt.flow.shape == (2, *SHAPE)
    assert np.allclose(gt.flow, 0.0)
    assert gt.valid.all()


def test_pure_translation_flow_is_exactly_the_translation() -> None:
    dx, dy = 3.0, -2.0
    homography = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    gt = dense_displacement(homography, SHAPE)
    assert np.allclose(gt.flow[0], dx)
    assert np.allclose(gt.flow[1], dy)
    # A shift of (3, -2) leaves the first 3 columns and last 2 rows of the source mapping
    # outside the target canvas.
    expected = (SHAPE[1] - abs(dx)) * (SHAPE[0] - abs(dy)) / (SHAPE[0] * SHAPE[1])
    assert overlap_ratio(gt) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("index", range(20))
def test_sampled_homography_is_recovered_from_its_own_dense_field(index: int) -> None:
    """The P2-4 round trip: H -> dense field -> H, over 20 independent samples."""
    homography = sample_homography(GTConfig(), generator(0, index), SHAPE)
    gt = dense_displacement(homography, SHAPE)

    height, width = SHAPE
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    target = source + gt.flow.reshape(2, -1).T

    # Every correspondence is exact by construction, so a least-squares fit over all of them
    # must return the original matrix -- no RANSAC, nothing to be robust to.
    fitted, _ = cv2.findHomography(source, target, 0)
    assert fitted is not None
    recovered = np.asarray(fitted, dtype=np.float64)

    # Homographies are equal up to scale; compare where it is observable, on the corners.
    reference = corners(SHAPE)
    displacement = np.linalg.norm(
        warp_points(reference, recovered) - warp_points(reference, homography), axis=1
    )
    assert displacement.max() < 1e-4


@pytest.mark.parametrize("index", range(10))
def test_sampled_homography_respects_the_configured_ranges(index: int) -> None:
    """A warp that leaves nothing overlapping is not a hard case, it is a broken sample."""
    gt = dense_displacement(sample_homography(GTConfig(), generator(7, index), SHAPE), SHAPE)
    assert overlap_ratio(gt) > 0.25


def test_warping_an_image_agrees_with_the_dense_field() -> None:
    """The field and ``apply_warp`` must describe the same transform.

    Checked on a single bright pixel rather than on image statistics: interpolation makes any
    aggregate comparison approximate, whereas a delta lands where the field says it lands.
    """
    homography = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    image = np.zeros(SHAPE, dtype=np.uint8)
    image[10, 20] = 255

    warped = apply_warp(image, homography)
    gt = dense_displacement(homography, SHAPE)
    dx, dy = gt.flow[:, 10, 20]

    assert warped[int(10 + dy), int(20 + dx)] == 255


def test_seed_and_index_determine_the_warp() -> None:
    reference = sample_homography(GTConfig(), generator(0, 3), SHAPE)
    assert np.array_equal(sample_homography(GTConfig(), generator(0, 3), SHAPE), reference)
    assert not np.allclose(sample_homography(GTConfig(), generator(1, 3), SHAPE), reference)
    assert not np.allclose(sample_homography(GTConfig(), generator(0, 4), SHAPE), reference)


def test_seed_and_index_do_not_collide() -> None:
    """``seed + index`` would make (0, 1) and (1, 0) the same run; the stride must not."""
    assert warp_seed(0, 1) != warp_seed(1, 0)
    assert len({warp_seed(s, i) for s in range(8) for i in range(8)}) == 64


def test_singular_homography_is_rejected() -> None:
    with pytest.raises(WarpError, match="singular"):
        dense_displacement(np.zeros((3, 3)), SHAPE)


def test_the_invertibility_check_is_symmetric_under_inversion() -> None:
    """The P3-7 server crash, pinned.

    The floor on ``|det|`` was one-sided, and ``det(inv(H)) == 1 / det(H)``: a grossly
    expanding fit was accepted while its inverse -- which every symmetric error term needs --
    was rejected, so ``symmetric_reprojection_error`` raised 50 pairs into a 300-pair run.
    The guarantee is an equivalence, not blanket acceptance: whatever passes, its inverse
    passes, and whatever is refused has its inverse refused too.
    """

    def accepts(matrix: np.ndarray) -> bool:
        try:
            check_homography(matrix)
        except WarpError:
            return False
        return True

    expanding = np.diag([1e5, 1e5, 1.0])
    assert not accepts(expanding), "an expanding fit whose inverse is noise is unusable"
    for matrix in (expanding, sample_homography(GTConfig(), generator(0, 0), SHAPE)):
        assert accepts(matrix) == accepts(np.linalg.inv(matrix))


def test_a_large_translation_is_not_mistaken_for_a_degenerate_one() -> None:
    """The guard is on the determinant, not the condition number.

    A translation of 1e7 px is absurd but exactly invertible, and its condition number is
    ~1e14 -- the homography's entries mix pixel and dimensionless units, so conditioning
    rejects legitimate transforms. ``det == 1`` says the right thing.
    """
    check_homography(np.array([[1.0, 0.0, 1e7], [0.0, 1.0, -1e7], [0.0, 0.0, 1.0]]))
