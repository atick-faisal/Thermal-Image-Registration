"""GT-based registration accuracy: EPE, corner error / MACE, AUC, success rate.

PLAN.md §6.1. These are the headline numbers, and §6.4 requires them to be GT-based rather
than mutual-information-based -- the legacy method optimises MI, so leading with MI would be
circular.

Every function here is unit-tested against a case whose answer is known analytically
(``tests/test_metrics.py``). That is not ceremony: an off-by-one in the corner convention or
a silently-dropped invalid mask produces numbers that look entirely plausible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cmreg.warp import corners, warp_points

FloatArray = NDArray[np.float64]


class MetricError(ValueError):
    """Raised when a metric is asked for on inputs that cannot support it."""


@dataclass(frozen=True, slots=True)
class EndPointError:
    """Per-pixel displacement error, summarised.

    Mean *and* median, because they disagree in the way this project cares about: a pair that
    is pixel-perfect at the centre and drifts 30 px at the edges (PLAN.md §6.2) has a modest
    median and a large mean, and reporting only one of them hides exactly that structure.
    """

    mean: float
    median: float
    count: int


def endpoint_error(
    predicted: NDArray[np.floating],
    truth: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> EndPointError:
    """Euclidean per-pixel error between two ``(2, H, W)`` displacement fields.

    ``valid`` excludes pixels with no ground truth -- those that map outside the target
    canvas. Scoring them would penalise a method for content it was never shown.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64)
    if pred.shape != true.shape:
        raise MetricError(f"flow shapes differ: {pred.shape} vs {true.shape}")
    if pred.ndim != 3 or pred.shape[0] != 2:
        raise MetricError(f"expected a (2, H, W) flow field, got {pred.shape}")

    distance = np.linalg.norm(pred - true, axis=0)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != distance.shape:
            raise MetricError(f"valid mask {mask.shape} does not match flow {distance.shape}")
        distance = distance[mask]

    if distance.size == 0:
        raise MetricError("no valid pixels to score; the pair has zero ground-truth overlap")
    return EndPointError(
        mean=float(distance.mean()), median=float(np.median(distance)), count=int(distance.size)
    )


def corner_error(predicted: FloatArray, truth: FloatArray, shape: tuple[int, int]) -> float:
    """Mean distance between the four image corners under ``predicted`` and under ``truth``.

    This is the per-pair term; averaging it over a dataset is MACE. It is the standard
    homography metric precisely because it is invariant to how the two matrices are scaled,
    which a direct matrix comparison is not.
    """
    reference = corners(shape)
    return float(
        np.linalg.norm(
            warp_points(reference, predicted) - warp_points(reference, truth), axis=1
        ).mean()
    )


def auc(errors: Sequence[float] | NDArray[np.floating], threshold: float) -> float:
    """Area under the cumulative error curve up to ``threshold``, normalised to [0, 1].

    Computed as the exact integral of the empirical CDF::

        AUC = (1 / (n * t)) * sum_i max(0, t - e_i)

    which follows from the CDF being a step function: each error ``e_i < t`` contributes
    ``(t - e_i) / n`` to the integral and nothing above ``t``.

    Note this is the *exact* step integral, not the piecewise-linear ``np.trapz`` variant
    used in the SuperGlue/LoFTR line of work; the two differ by O(1/n) and only the exact
    form has a closed-form answer to test against.
    """
    if threshold <= 0.0:
        raise MetricError(f"threshold must be positive, got {threshold}")
    values = np.asarray(errors, dtype=np.float64).ravel()
    if values.size == 0:
        raise MetricError("cannot compute AUC over an empty error set")
    return float(np.maximum(0.0, threshold - values).sum() / (values.size * threshold))


def success_rate(errors: Sequence[float] | NDArray[np.floating], threshold: float) -> float:
    """Fraction of pairs whose error is at or below ``threshold``."""
    if threshold <= 0.0:
        raise MetricError(f"threshold must be positive, got {threshold}")
    values = np.asarray(errors, dtype=np.float64).ravel()
    if values.size == 0:
        raise MetricError("cannot compute a success rate over an empty error set")
    return float((values <= threshold).mean())
