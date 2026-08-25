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
    # 1 - scatter/magnitude. 1 means wholly systematic, 0 means wholly random. Not a fraction
    # of variance -- these are pixel means, not squares -- so it is read as a ratio and nothing
    # more.
    systematic_fraction: float
    # The consensus displacement of each image corner, in pixels. This is what says whether the
    # offset is a pure translation (four equal vectors) or carries scale/rotation, which is the
    # difference between a fixed baseline and a mounting error.
    corner_shift: tuple[tuple[float, float], ...]


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
    scatter = float(np.mean([corner_error(m, consensus, shape, saturate=True) for m in matrices]))
    reference = corners(shape)
    shift = warp_points(reference, consensus) - reference

    return ResidualStructure(
        dataset=rows[0].dataset,
        matcher=rows[0].matcher,
        n_pairs=len(matrices),
        n_failed=len(rows) - len(matrices),
        magnitude=magnitude,
        magnitude_median=float(np.median(absolute)),
        consensus=corner_error(consensus, identity, shape, saturate=True),
        scatter=scatter,
        # Guarded rather than assumed positive: a run in which every pair registered perfectly
        # has magnitude 0, and the ratio is then 0/0 rather than a finding.
        systematic_fraction=float("nan") if magnitude == 0.0 else 1.0 - scatter / magnitude,
        corner_shift=tuple((float(dx), float(dy)) for dx, dy in shift),
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
    lines.append(f"{'systematic_fraction':<{_KEY_WIDTH}}{structure.systematic_fraction:.4f}")
    lines.append(rule)
    for name, (dx, dy) in zip(_CORNER_NAMES, structure.corner_shift, strict=True):
        lines.append(f"{'  shift ' + name:<{_KEY_WIDTH}}{dx:+8.3f}, {dy:+8.3f}")
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
