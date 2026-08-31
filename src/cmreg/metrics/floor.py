"""The model floor: what a restricted warp model can achieve against a projective truth.

Tier-1 ground truth is a **full projective homography** (`gt/synthetic.py::sample_homography`:
rotation +/-30 deg, log-uniform scale, perspective jitter, translation). A 6-DoF affine cannot
represent its perspective term and a 4-DoF similarity cannot represent that or its shear, so
neither can reach zero error however good the matcher is. The residual they are left with is a
**floor**, in exactly the sense `experiments/GRID.md` §3's dataset residual `R` is one: a
systematic term under every number in that row, present before any method is measured.

Stage E (TASKS.md P3-11) is unreadable without it. "Affine scores 9 px and homography 6 px" is
a statement about the models only once the affine floor is known; if that floor is 8.5 px the
row is reporting the ground truth's own perspective content and nothing about registration.

**The floor is fitted to the four image corners, deliberately.** `metrics.corner_error` is
mean corner displacement, so fitting the corners minimises the very quantity being reported and
the result is a *strict lower bound*: no fit of that model, from any matcher, on any
correspondence set, can score below it. A fit over correspondences scattered through the image
-- which is what a matcher actually supplies -- can only do worse. Quoting the bound rather than
a plausible average is what makes "the model cannot beat this" a claim rather than an estimate.

Under ``metrics/`` and not ``warp/`` because it *is* a metric: it is a corner error, computed
through the same `corner_error` every benchmark row uses, and `metrics` already sits above
`warp` in the import order (putting it below would make the two packages mutually recursive).
"""

from __future__ import annotations

from cmreg.config import WarpModel
from cmreg.metrics.registration import corner_error
from cmreg.warp import FloatArray, check_homography, corners, fit_least_squares, warp_points


def model_floor(truth: FloatArray, shape: tuple[int, int], model: WarpModel) -> float:
    """The smallest mean corner error any ``model`` fit can have against ``truth``.

    Zero for :attr:`WarpModel.HOMOGRAPHY` by construction -- the model contains the truth --
    which is what the unit test asserts, since a floor implementation that quietly used a
    restricted fit for every model would look entirely plausible otherwise.
    """
    matrix = check_homography(truth)
    reference = corners(shape)
    best = fit_least_squares(reference, warp_points(reference, matrix), model)
    return corner_error(best, matrix, shape)
