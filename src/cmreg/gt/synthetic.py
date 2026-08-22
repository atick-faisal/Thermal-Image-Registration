"""Tier-1 ground truth: known synthetic homographies and the dense fields they induce.

TASKS.md marks P2-4 -- validating this module analytically -- as **the single
highest-leverage item in the project**. Phases 4, 5 and 6 all train and evaluate against
what this file produces; if it is subtly wrong, all three are wrong and the failure surfaces
months later as an inexplicable result. Hence ``tests/test_synthetic_gt.py``, which recovers
a sampled homography from its own dense field and asserts the round-trip error is numerical
noise. Nothing may consume this module without that test passing.

**What gets persisted is the homography, not the field.** A dense field for one 640x512 pair
is 2 x 640 x 512 float32 = 2.6 MB; the 2,500-pair public set alone would be ~6.5 GB of
perfectly redundant data, since ``dense_displacement`` reconstructs it exactly from ``H`` and
the shape. GT records therefore carry ``(stem, height, width, H)`` and the field is derived
on demand.

Convention inherited from ``cmreg.warp.homography``: ``H`` maps source pixel coordinates to
target pixel coordinates, points are ``(x, y)``, arrays are indexed ``[y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.config import GTConfig
from cmreg.warp import check_homography, corners

FloatArray = NDArray[np.float64]

# Coprime with anything a seed is likely to be, so `(seed, index)` pairs do not collide the
# way `seed + index` does -- there, seed 0 / index 1 and seed 1 / index 0 are the same run.
_INDEX_STRIDE = 1_000_003


def warp_seed(seed: int, index: int) -> int:
    """Derive a per-pair seed from the experiment seed and the pair's index.

    Ported from ``../Thermal-To-Optical-Translation/src/t2o/data/dataset.py``, whose
    docstring records the bug this shape prevents: augmentation drawn from the *ambient*
    RNG came from the global stream at ``workers=0`` but from torch's per-worker derivation
    at ``workers>0``, so raising the worker count for speed silently changed the reported
    numbers. A field keyed on ``(seed, index)`` alone cannot do that -- pair *i* gets the
    same warp however the work was scheduled.
    """
    return (seed * _INDEX_STRIDE + index) % 2**32


def generator(seed: int, index: int) -> np.random.Generator:
    """The explicit ``Generator`` for one pair. Never ``np.random.*`` module functions."""
    return np.random.default_rng(warp_seed(seed, index))


@dataclass(frozen=True, slots=True)
class DenseGT:
    """The dense supervision one synthetic pair provides."""

    # (2, H, W): flow[:, y, x] is the (dx, dy) that carries source pixel (x, y) to its
    # location in the target. This is what an EPE is computed against.
    flow: NDArray[np.float32]
    # (H, W): False where the source pixel maps outside the target canvas. Scoring those
    # pixels would penalise a method for content it was never shown -- the partial-overlap
    # case of PLAN.md §5 is exactly this, made explicit.
    valid: NDArray[np.bool_]

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape


def sample_homography(
    config: GTConfig, rng: np.random.Generator, shape: tuple[int, int]
) -> FloatArray:
    """Sample a random homography over the ranges declared in ``config``.

    Built as *similarity, then perspective*: a rotation/scale/translation about the image
    centre, composed with a homography fitted to independently jittered corners. Sampling the
    nine matrix entries directly would be simpler and useless -- the ranges in PLAN.md §5 are
    stated in degrees and fractions of image size, and only this construction honours them.
    """
    height, width = shape
    centre_x, centre_y = width / 2.0, height / 2.0

    angle = float(rng.uniform(-config.rotation_deg, config.rotation_deg))
    # Log-uniform, so 0.8x and 1.25x are equally likely. Uniform sampling over [0.8, 1.25]
    # would put 55% of the mass above 1.0 and quietly bias the benchmark toward magnification.
    scale = float(np.exp(rng.uniform(np.log(config.scale_min), np.log(config.scale_max))))

    similarity = np.eye(3, dtype=np.float64)
    similarity[:2] = cv2.getRotationMatrix2D((centre_x, centre_y), angle, scale)
    similarity[0, 2] += float(rng.uniform(-config.translation, config.translation)) * width
    similarity[1, 2] += float(rng.uniform(-config.translation, config.translation)) * height

    source = corners(shape)
    jitter = rng.uniform(-config.perspective, config.perspective, size=(4, 2))
    target = source + jitter * np.array([width, height], dtype=np.float64)
    perspective = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    return check_homography(perspective @ similarity)


def dense_displacement(h: FloatArray, shape: tuple[int, int]) -> DenseGT:
    """The exact per-pixel displacement ``h`` induces over a ``shape`` source image.

    Computed on the full pixel grid in closed form rather than by sampling or by inverting a
    warped image, so the result carries no interpolation error and the round-trip test in
    P2-4 measures the construction rather than a resampler.
    """
    matrix = check_homography(h)
    height, width = shape

    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij"
    )
    ones = np.ones_like(xs)
    homogeneous = np.stack([xs, ys, ones], axis=0).reshape(3, -1)

    mapped = matrix @ homogeneous
    w = mapped[2]
    # Points on or behind the horizon line (w -> 0) have no finite image. Marking them
    # invalid rather than letting them become huge finite numbers keeps a single such pixel
    # from dominating a mean EPE.
    finite = np.abs(w) > np.finfo(np.float64).eps
    safe_w = np.where(finite, w, 1.0)
    target_x = mapped[0] / safe_w
    target_y = mapped[1] / safe_w

    flow = np.stack([target_x - homogeneous[0], target_y - homogeneous[1]], axis=0)
    inside = (target_x >= 0.0) & (target_x < width) & (target_y >= 0.0) & (target_y < height)

    return DenseGT(
        flow=flow.reshape(2, height, width).astype(np.float32),
        valid=(finite & inside).reshape(height, width),
    )


def overlap_ratio(gt: DenseGT) -> float:
    """Fraction of source pixels that land inside the target canvas.

    The controllable version of this is TASKS.md P2-2; reported here so that every Tier-1
    pair carries the number even before the partial-overlap generator exists.
    """
    return float(gt.valid.mean())
