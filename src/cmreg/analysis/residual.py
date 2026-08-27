"""Is a dataset's residual misalignment systematic or random? (TASKS.md P1-1b)

TASKS.md P1-1a measured, with a zero-magnitude Tier-1 warp, that MSRS, FLIR-aligned and LLVIP
each carry a 4-6 px residual cross-modal misalignment and DroneVehicle ~59 px, and that the
number is a property of the *data* rather than of the matcher -- three matchers spanning three
decades of cost agreed to within 4% on FLIR. It deliberately did not claim more than that:

    it does not separate "the two cameras are misaligned" from "the two modalities cannot be
    localised to better than this against each other".

A corner error cannot separate them, because it says how far a fit was and never in which
direction. The residual homographies do:

* a **fixed rig miscalibration** is systematic -- every pair in a dataset shares one offset, so
  the per-pair residuals cluster tightly around a common consensus;
* a **cross-modal localisation limit** is random -- the residuals scatter around identity with
  no consistent bias.

So this module decomposes a run's residuals into a consensus part and a scatter part. The
reading is stated once, here, and carried into the paper's protocol section:

    scatter << magnitude  =>  systematic  =>  a rig offset. Correctable, and P3-1 may legitimately
                              compose it into the Tier-1 ground truth.
    scatter ~= magnitude  =>  random      =>  the modality gap. P3-1 must instead raise its
                              thresholds above the floor, or report Tier-1 error as relative to
                              the dataset's own alignment.

**Only meaningful on an identity-warp run** (``experiments/p1_alignment_audit.yaml``). There
``H_gt = I``, so the homography the estimator recovers *is* the pair's own residual ``R_i``.
Under a non-zero warp each row's ``h`` is ``H_gt^-1 . R_i`` and the decomposition below would be
measuring the sampled warps instead. :func:`residual_structure` cannot detect this from the rows
alone -- the config snapshot beside them is what records it -- so it is a documented
precondition, not a check.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cmreg.metrics import corner_error
from cmreg.results.store import PairRow
from cmreg.warp import corners, warp_points

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

_RULE_WIDTH = 72
_KEY_WIDTH = 26
# Corner order matches `warp/homography.py::corners`: clockwise from the top-left.
_CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


class AnalysisError(ValueError):
    """Raised when a group of rows cannot support a residual decomposition."""


@dataclass(frozen=True, slots=True)
class ResidualStructure:
    """One (dataset, matcher) group's residual, split into its systematic and random parts.

    All three magnitudes are four-corner errors in pixels, so they are directly comparable
    with ``reg/mace`` and with every threshold in PLAN.md §6.1.
    """

    dataset: str
    matcher: str
    # The shape `corner_shift` is expressed in. A corner field without it is unusable -- the
    # same four displacements mean a different warp at a different resolution -- and
    # `_residuals` already refuses a group whose rows disagree about it.
    height: int
    width: int
    # Successful rows only. A failed fit has no residual to decompose; the count is reported
    # so a decomposition over a handful of survivors is visibly that (TASKS.md X-4).
    n_pairs: int
    n_failed: int
    # Mean ``corner_error(R_i, I)``. Reproduces the run's own `reg/mace`, and is the total the
    # two parts below decompose.
    magnitude: float
    # Median ``corner_error(R_i, I)``: the *typical* pair's residual. Reported beside the mean
    # because the two diverge exactly where the mean stops meaning anything. Measured on
    # DroneVehicle val, the median residual is 4.7 px and the mean is 78.0 -- 36 of 50 pairs sit
    # near 5 px and 14 are catastrophic fits the estimator still called successes. Reading the
    # mean alone there says "77 px of per-pair variation", which is not what the data shows.
    magnitude_median: float
    # ``corner_error(R_bar, I)`` for the consensus residual: the part every pair shares.
    consensus: float
    # Mean ``corner_error(R_i, R_bar)``: the part that differs pair to pair.
    scatter: float
    # Median ``corner_error(R_i, R_bar)``: the *typical* pair's departure from the consensus.
    # The one scatter statistic that survives a matcher swap. Measured on FLIR val, roma scatters
    # 6.56 px about its consensus and superpoint-lightglue 24.32 -- but splg's mean is 2.8x its
    # own median, so those two numbers are not comparing the same thing and the difference is
    # splg's failure tail, not a difference in the data. Comparing matchers is the entire point
    # of P1-1d, and a mean cannot do it when one matcher fails gross and the other does not.
    scatter_median: float
    # 1 - scatter/magnitude. 1 means wholly systematic, 0 means wholly random. Not a fraction
    # of variance -- these are pixel means, not squares -- so it is read as a ratio and nothing
    # more.
    systematic_fraction: float
    # The consensus displacement of each image corner, in pixels. This is what says whether the
    # offset is a pure translation (four equal vectors) or carries scale/rotation, which is the
    # difference between a fixed baseline and a mounting error.
    corner_shift: tuple[tuple[float, float], ...]
    # The consensus read as physics rather than as eight numbers. Measured on FLIR-aligned val,
    # `rotation_deg` is -1.18: the split shipped as pre-registered carries a fixed camera roll.
    # On MSRS it is ~0 with `scale` at 0.987/0.983, a ~1.5% field-of-view contraction instead.
    # Reported here rather than derived downstream because the printed block is how a result
    # reaches the Mac from the training server, and a corner table cannot be eyeballed into a
    # rotation.
    rotation_deg: float
    # Singular values of the affine part: the two principal stretch factors. Equal and below 1
    # is an FOV contraction; unequal is an aspect/shear mismatch.
    scale: tuple[float, float]
    # Translation at the image *centre*, in pixels. At the origin -- a corner -- a translation is
    # inseparable from the rotation and scale about the centre, so measuring it there would
    # report a shift that is not one.
    centre_shift: tuple[float, float]


def _residuals(rows: Sequence[PairRow]) -> tuple[FloatArray, tuple[int, int]]:
    """The successful rows' homographies as ``(N, 3, 3)``, plus the shared image shape."""
    usable = [row for row in rows if row.success and row.h is not None]
    if not usable:
        raise AnalysisError("no successful rows: nothing to decompose")

    shapes = {(row.height, row.width) for row in usable}
    if len(shapes) != 1:
        raise AnalysisError(
            f"rows do not share one image shape ({sorted(shapes)}); group by (height, width) "
            "before decomposing -- a corner error is only comparable at a fixed image size"
        )
    height, width = next(iter(shapes))
    if height is None or width is None:  # pragma: no cover - a successful row always has one
        raise AnalysisError("successful rows carry no image shape; written by an older schema?")

    matrices = np.array([np.asarray(row.h, dtype=np.float64).reshape(3, 3) for row in usable])
    return matrices, (height, width)


def consensus_homography(matrices: FloatArray, shape: tuple[int, int]) -> FloatArray:
    """The residual every pair shares, built geometrically rather than by averaging matrices.

    Homography matrices do not average: they are defined up to scale and their entries mix
    units, so an element-wise mean of two sensible warps is not itself sensible. Averaging the
    *corner displacements* is well-defined, and refitting from those four correspondences
    returns to a homography exactly.

    The **median** rather than the mean, because P1-1a produced exactly the distribution that
    punishes a mean: a projective fit whose horizon line crosses the frame sends corners
    arbitrarily far, and one such pair in 361 drove a dataset mean to 5.6e8 px
    (``metrics/registration.py::diagonal``).
    """
    import cv2

    reference = corners(shape)
    displacements = np.array([warp_points(reference, m) - reference for m in matrices])
    consensus = reference + np.median(displacements, axis=0)
    return np.asarray(
        cv2.getPerspectiveTransform(reference.astype(np.float32), consensus.astype(np.float32)),
        dtype=np.float64,
    )


def residual_structure(rows: Sequence[PairRow]) -> ResidualStructure:
    """Decompose one group's residual. See the module docstring for the precondition."""
    if not rows:
        raise AnalysisError("cannot decompose zero rows")

    matrices, shape = _residuals(rows)
    identity = np.eye(3, dtype=np.float64)
    consensus = consensus_homography(matrices, shape)

    # `saturate=True` throughout, for the reason `metrics.diagonal` gives: a single
    # near-degenerate fit would otherwise be the only thing either mean reports.
    absolute = np.array(
        [corner_error(m, identity, shape, saturate=True) for m in matrices], dtype=np.float64
    )
    magnitude = float(absolute.mean())
    relative = np.array(
        [corner_error(m, consensus, shape, saturate=True) for m in matrices], dtype=np.float64
    )
    scatter = float(relative.mean())
    reference = corners(shape)
    shift = warp_points(reference, consensus) - reference
    rotation_deg, scale, centre_shift = _decompose(consensus, shape)

    return ResidualStructure(
        dataset=rows[0].dataset,
        matcher=rows[0].matcher,
        height=shape[0],
        width=shape[1],
        n_pairs=len(matrices),
        n_failed=len(rows) - len(matrices),
        magnitude=magnitude,
        magnitude_median=float(np.median(absolute)),
        consensus=corner_error(consensus, identity, shape, saturate=True),
        scatter=scatter,
        scatter_median=float(np.median(relative)),
        # Guarded rather than assumed positive: a run in which every pair registered perfectly
        # has magnitude 0, and the ratio is then 0/0 rather than a finding.
        systematic_fraction=float("nan") if magnitude == 0.0 else 1.0 - scatter / magnitude,
        corner_shift=tuple((float(dx), float(dy)) for dx, dy in shift),
        rotation_deg=rotation_deg,
        scale=scale,
        centre_shift=centre_shift,
    )


def _decompose(
    consensus: FloatArray, shape: tuple[int, int]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Split the consensus into rotation, principal scales and a centre translation.

    The affine part is polar-decomposed via its SVD: ``A = (U Vt) (V S Vt)`` splits it into a
    rotation and a symmetric stretch, which is the decomposition that survives the two being
    combined in either order. The projective row is deliberately not reported -- on all four
    audited datasets it is O(1e-5) and describes nothing physical at this magnitude.
    """
    normalised = consensus / consensus[2, 2]
    u, singular, vt = np.linalg.svd(normalised[:2, :2])
    rotation = u @ vt
    height, width = shape
    centre = np.array([[width / 2.0, height / 2.0]], dtype=np.float64)
    shifted = warp_points(centre, normalised) - centre
    return (
        float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        (float(singular[0]), float(singular[1])),
        (float(shifted[0, 0]), float(shifted[0, 1])),
    )


def by_matcher(rows: Sequence[PairRow]) -> tuple[ResidualStructure, ...]:
    """One decomposition per matcher, in first-appearance order.

    First-appearance rather than sorted, matching ``cli.py::_run_report``: it is the order the
    run produced them in, so a re-render is diffable against a pasted one.
    """
    names = list(dict.fromkeys(row.matcher for row in rows))
    return tuple(residual_structure([r for r in rows if r.matcher == name]) for name in names)


def render(structure: ResidualStructure) -> str:
    """The copy-pasteable console block, same channel discipline as ``results/report.py``."""
    rule = "-" * _RULE_WIDTH
    verdict = _verdict(structure)
    lines = ["=== CMREG RESIDUAL STRUCTURE ===", rule]
    lines.append(f"{'dataset':<{_KEY_WIDTH}}{structure.dataset}")
    lines.append(f"{'matcher':<{_KEY_WIDTH}}{structure.matcher}")
    lines.append(f"{'pairs':<{_KEY_WIDTH}}{structure.n_pairs} ({structure.n_failed} failed)")
    lines.append(rule)
    lines.append(f"{'magnitude px':<{_KEY_WIDTH}}{structure.magnitude:.4f}")
    lines.append(f"{'  median px':<{_KEY_WIDTH}}{structure.magnitude_median:.4f}   (typical pair)")
    lines.append(f"{'  consensus px':<{_KEY_WIDTH}}{structure.consensus:.4f}   (systematic)")
    lines.append(f"{'  scatter px':<{_KEY_WIDTH}}{structure.scatter:.4f}   (random)")
    lines.append(
        f"{'  scatter median px':<{_KEY_WIDTH}}{structure.scatter_median:.4f}   (typical pair)"
    )
    lines.append(f"{'systematic_fraction':<{_KEY_WIDTH}}{structure.systematic_fraction:.4f}")
    lines.append(rule)
    for name, (dx, dy) in zip(_CORNER_NAMES, structure.corner_shift, strict=True):
        lines.append(f"{'  shift ' + name:<{_KEY_WIDTH}}{dx:+8.3f}, {dy:+8.3f}")
    lines.append(rule)
    lines.append(f"{'consensus rotation deg':<{_KEY_WIDTH}}{structure.rotation_deg:+.4f}")
    lines.append(
        f"{'consensus scale':<{_KEY_WIDTH}}{structure.scale[0]:.5f} / {structure.scale[1]:.5f}"
    )
    lines.append(
        f"{'consensus centre px':<{_KEY_WIDTH}}"
        f"{structure.centre_shift[0]:+.3f}, {structure.centre_shift[1]:+.3f}"
    )
    lines.append(rule)
    lines.append(f"{'reading':<{_KEY_WIDTH}}{verdict}")
    lines.append("=== END ===")
    return "\n".join(lines)


# Two thirds of the magnitude in one part or the other. Deliberately coarse: this is a label on
# a printed block to orient a reader, and the three numbers above it are what the paper reports.
_SYSTEMATIC_AT = 0.67
_RANDOM_AT = 0.33
# `magnitude` and `scatter` are both means, so `systematic_fraction` inherits whatever a heavy
# tail does to them. Past this ratio of mean to median the tail *is* the mean and the fraction
# describes the outliers rather than the dataset, so the label abstains and points at the two
# raw numbers instead. 2.0 rather than something tuned: at twice the median, half the reported
# magnitude is coming from the minority of pairs.
_TAIL_AT = 2.0


def _verdict(structure: ResidualStructure) -> str:
    fraction = structure.systematic_fraction
    if not np.isfinite(fraction):
        return "no residual to decompose"
    if structure.magnitude > _TAIL_AT * structure.magnitude_median > 0.0:
        return (
            "tail-dominated -- a minority of gross fits carries the mean; "
            "read the median and consensus, not the fraction"
        )
    if fraction >= _SYSTEMATIC_AT:
        return "systematic -- a shared offset, i.e. rig miscalibration"
    if fraction <= _RANDOM_AT:
        return "random -- per-pair, i.e. the cross-modal localisation limit"
    return "mixed -- report both parts, do not label it"


@dataclass(frozen=True, slots=True)
class MatcherConsensus:
    """One dataset's residual as several matchers jointly see it (TASKS.md P1-1d).

    :func:`residual_structure` takes a median across *pairs* to give one matcher's ``R_bar``.
    This is that construction applied one level up, across *matchers*, and it exists because
    P1-1d measured that a single leg is not a calibration: roma's consensus field sat 3.44 px
    from superpoint-lightglue's and 2.50 px from eloftr's, which is 84% of the 4.09 px per-pair
    scatter the constant is supposed to be small against. The three-way median is 1.23 px from
    its furthest leg -- 30% of that scatter -- so what the extra legs buy is a **measured**
    uncertainty in place of one known only to be non-zero.
    """

    dataset: str
    matchers: tuple[str, ...]
    height: int
    width: int
    corner_shift: tuple[tuple[float, float], ...]
    # Each leg's mean corner distance from the published median, in the order of `matchers`.
    leg_distance_px: tuple[float, ...]
    # The pairs the *thinnest* leg contributed, not the total: the constant is only as well
    # evidenced as its least-supported witness, and summing would flatter a three-leg estimate
    # by counting the same 300 pairs three times.
    n_pairs: int

    @property
    def spread_px(self) -> float:
        return float(np.mean(self.leg_distance_px))

    @property
    def worst_case_px(self) -> float:
        return float(np.max(self.leg_distance_px))


def across_matchers(structures: Sequence[ResidualStructure]) -> MatcherConsensus:
    """Combine several matchers' consensus fields into the one published constant.

    Works on the corner *shifts* rather than the matrices, for the reason
    :func:`consensus_homography` gives, and the distances need no refit: both fields displace
    the same four reference corners, so a leg's ``corner_error`` against the median is exactly
    the mean norm of the difference of their shifts.
    """
    if not structures:
        raise AnalysisError("cannot combine zero matchers")
    datasets = {s.dataset for s in structures}
    if len(datasets) != 1:
        raise AnalysisError(f"cannot combine matchers across datasets: {sorted(datasets)}")
    shapes = {(s.height, s.width) for s in structures}
    if len(shapes) != 1:
        raise AnalysisError(f"cannot combine matchers across shapes: {sorted(shapes)}")
    if len(structures) == 1:
        # Permitted, because a one-leg field is the only thing a single-matcher run can offer
        # and it is still the right shape to inspect. Loud, because publishing it as a
        # calibration would ship that matcher's bias as a dataset constant.
        logger.warning(
            "consensus over a single matcher (%s): this is that matcher's bias, not a "
            "calibration -- P1-1d measured legs up to 3.44 px apart",
            structures[0].matcher,
        )

    shifts = np.array([s.corner_shift for s in structures], dtype=np.float64)
    median = np.median(shifts, axis=0)
    distances = tuple(float(np.linalg.norm(shift - median, axis=1).mean()) for shift in shifts)
    height, width = shapes.pop()
    return MatcherConsensus(
        dataset=structures[0].dataset,
        matchers=tuple(s.matcher for s in structures),
        height=height,
        width=width,
        corner_shift=tuple((float(dx), float(dy)) for dx, dy in median),
        leg_distance_px=distances,
        n_pairs=min(s.n_pairs for s in structures),
    )
