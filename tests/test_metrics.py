"""Metrics against analytically-known answers (TASKS.md P0-6).

Not ceremony: an off-by-one in the corner convention or a silently-dropped invalid mask
produces numbers that look entirely plausible, and would be discovered only when a reviewer
asks why two papers report different MACE for the same method.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmreg.metrics import (
    MetricError,
    RegistrationMetrics,
    auc,
    auc_key,
    corner_error,
    endpoint_error,
    success_rate,
    success_rate_key,
)
from cmreg.metrics.schema import EPE_MEAN, MACE

SHAPE = (48, 64)


def _constant_flow(dx: float, dy: float) -> np.ndarray:
    field = np.zeros((2, *SHAPE), dtype=np.float64)
    field[0] = dx
    field[1] = dy
    return field


def test_identical_flows_score_zero() -> None:
    field = _constant_flow(3.0, -1.0)
    result = endpoint_error(field, field)
    assert result.mean == 0.0
    assert result.median == 0.0
    assert result.count == SHAPE[0] * SHAPE[1]


def test_constant_offset_epe_is_exactly_the_hypotenuse() -> None:
    result = endpoint_error(_constant_flow(3.0, 4.0), _constant_flow(0.0, 0.0))
    assert result.mean == pytest.approx(5.0)
    assert result.median == pytest.approx(5.0)


def test_invalid_pixels_are_excluded_not_scored_as_zero() -> None:
    predicted = _constant_flow(0.0, 0.0)
    truth = _constant_flow(0.0, 0.0)
    predicted[0, :24, :] = 10.0  # half the image is wrong

    valid = np.ones(SHAPE, dtype=bool)
    valid[:24, :] = False

    assert endpoint_error(predicted, truth).mean == pytest.approx(5.0)
    assert endpoint_error(predicted, truth, valid).mean == 0.0


def test_epe_over_no_valid_pixels_raises_rather_than_returning_nan() -> None:
    field = _constant_flow(0.0, 0.0)
    with pytest.raises(MetricError, match="zero ground-truth overlap"):
        endpoint_error(field, field, np.zeros(SHAPE, dtype=bool))


def test_corner_error_of_a_pure_translation_is_the_translation() -> None:
    truth = np.eye(3)
    predicted = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    assert corner_error(predicted, truth, SHAPE) == pytest.approx(5.0)


def test_corner_error_ignores_the_scale_of_the_matrices() -> None:
    truth = np.array([[1.2, 0.1, 3.0], [-0.05, 0.9, 4.0], [1e-4, 2e-4, 1.0]])
    assert corner_error(truth * 7.0, truth, SHAPE) == pytest.approx(0.0, abs=1e-9)


def test_auc_of_a_constant_error_has_a_closed_form() -> None:
    """With every error equal to e, AUC at threshold t is exactly max(0, 1 - e/t)."""
    errors = np.full(50, 2.0)
    assert auc(errors, 5.0) == pytest.approx(1.0 - 2.0 / 5.0)
    assert auc(errors, 10.0) == pytest.approx(1.0 - 2.0 / 10.0)
    assert auc(errors, 2.0) == pytest.approx(0.0)
    assert auc(errors, 1.0) == pytest.approx(0.0)


def test_auc_is_one_only_for_perfect_registration() -> None:
    assert auc(np.zeros(10), 3.0) == pytest.approx(1.0)


def test_success_rate_counts_errors_at_the_threshold_as_successes() -> None:
    errors = np.array([1.0, 3.0, 3.0, 9.0])
    assert success_rate(errors, 3.0) == pytest.approx(0.75)
    assert success_rate(errors, 0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_thresholds_are_rejected(bad: float) -> None:
    with pytest.raises(MetricError, match="positive"):
        auc([1.0], bad)
    with pytest.raises(MetricError, match="positive"):
        success_rate([1.0], bad)


def test_metric_keys_match_the_frozen_schema() -> None:
    """TASKS.md X-5 audits against these names; a rename here is a migration, not an edit."""
    assert auc_key(3.0) == "reg/auc_3px"
    assert auc_key(3) == "reg/auc_3px"
    assert success_rate_key(5.0) == "reg/success_rate_5px"

    flat = RegistrationMetrics(
        epe_mean=1.0,
        epe_median=0.5,
        mace=2.0,
        auc={3.0: 0.9, 5.0: 0.95},
        success_rate={3.0: 0.8},
        failure_rate=0.25,
        n_pairs=4,
    ).to_dict()
    assert set(flat) == {
        EPE_MEAN,
        "reg/epe_median",
        MACE,
        "reg/failure_rate",
        "reg/n_pairs",
        "reg/auc_3px",
        "reg/auc_5px",
        "reg/success_rate_3px",
    }
    # `n_pairs` is a count carried in a float-valued dict, because W&B's `log` takes one
    # mapping and splitting it by type would put the run's size somewhere other than beside
    # the numbers it qualifies.
    assert flat["reg/n_pairs"] == 4.0


def test_saturation_bounds_a_catastrophic_pair(shape: tuple[int, int] = (480, 640)) -> None:
    """A projective map's horizon line can pass through the frame, sending nearby pixels
    arbitrarily far. Measured on MSRS val, one such SIFT fit in 361 pairs drove the dataset
    `reg/epe_mean` to 5.6e8 px while the median stayed at 570 px."""
    from cmreg.metrics import diagonal, endpoint_error

    bound = diagonal(shape)
    assert bound == pytest.approx(800.0, abs=1.0)

    truth = np.zeros((2, 4, 4), dtype=np.float64)
    predicted = np.zeros_like(truth)
    predicted[0, 0, 0] = 1e9
    predicted[0, 1, 1] = np.inf

    unbounded = endpoint_error(predicted, truth)
    assert unbounded.mean > 1e7
    assert unbounded.n_saturated == 0

    saturated = endpoint_error(predicted, truth, saturate_at=bound)
    assert saturated.n_saturated == 2
    assert saturated.mean == pytest.approx(2 * bound / 16)


def test_saturation_cannot_change_a_sub_threshold_comparison() -> None:
    """Every threshold in PLAN.md §6.1 is 10 px or less, so the bound only ever compresses
    errors that were already failures."""
    from cmreg.metrics import diagonal, endpoint_error

    truth = np.zeros((2, 4, 4), dtype=np.float64)
    predicted = np.full_like(truth, 2.0)
    plain = endpoint_error(predicted, truth)
    clipped = endpoint_error(predicted, truth, saturate_at=diagonal((480, 640)))
    assert clipped.mean == pytest.approx(plain.mean)
    assert clipped.n_saturated == 0


def test_saturated_corner_error_is_bounded_by_the_diagonal() -> None:
    from cmreg.metrics import corner_error, diagonal

    shape = (480, 640)
    truth = np.eye(3)
    # A homography whose horizon line (w = 1 - 0.0015624x = 0, i.e. x = 640.04) sits a
    # fraction of a pixel outside the right edge: the two corners there map near-infinitely
    # far. Note `check_homography` accepts it -- its determinant is exactly 1 -- which is the
    # point: this is a legitimate projective map, not a numerically singular one.
    degenerate = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.0015624, 0.0, 1.0]])
    assert corner_error(degenerate, truth, shape) > 1e4
    assert corner_error(degenerate, truth, shape, saturate=True) <= diagonal(shape)
