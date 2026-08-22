"""Named preprocessing recipes and thermal upsampling (TASKS.md P3-2).

Ports the catalogue PLAN.md §15D indexes -- ``normalize_percentile``, ``normalize_clahe``,
``invert_image`` from ``/Users/ai/Python/vismatch/phase2_preprocessing.py`` and
``apply_sobel_gradient`` from ``phase2b_advanced_preprocessing.py`` -- stripped of their
``print``/matplotlib scaffolding and given one uniform signature.

Two things this module is responsible for that the originals were not:

**One percentile convention.** PLAN.md §15D records the same normalisation appearing at
0.3/99.75 in the production path and 2/98 in the batch pipeline. Rather than inherit one of
them, the bounds are config fields, so a run states which it used.

**Scale is returned, never assumed.** Upsampling changes pixel coordinates, so a keypoint
found in an upsampled image is not in native pixels. :class:`Preprocessed` carries the factor
and the runner maps coordinates back before estimation. Drop that and every upsampled run
reports errors inflated by the factor -- a P3-9 ablation that looks catastrophic for a reason
that has nothing to do with interpolation.

Every recipe is ``uint8 [H, W] -> uint8 [H, W]``, so recipes compose and a variant can be
swapped without any consumer knowing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.config import Interpolation, PreprocessConfig, Variant

GrayImage = NDArray[np.uint8]


class PreprocessError(ValueError):
    """Raised for an unknown recipe or an input that is not a uint8 grayscale image."""


_INTERPOLATION_FLAGS: dict[Interpolation, int] = {
    Interpolation.NEAREST: cv2.INTER_NEAREST,
    Interpolation.BILINEAR: cv2.INTER_LINEAR,
    Interpolation.BICUBIC: cv2.INTER_CUBIC,
    Interpolation.LANCZOS: cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True, slots=True)
class Preprocessed:
    """A preprocessed image and the coordinate scale it was produced at."""

    image: GrayImage
    # Multiply a native-pixel coordinate by this to get its coordinate in `image`. 1.0 for
    # every recipe; only upsampling moves it.
    scale: float

    def to_native(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map ``(N, 2)`` coordinates in this image back to native pixels.

        Note this inverts the *sampling grid*, not the pixel centres: ``cv2.resize`` places
        output pixel ``j`` at input coordinate ``(j + 0.5) / scale - 0.5``, and using the
        naive ``j / scale`` instead leaves a half-pixel bias that grows with the factor --
        small enough to survive review, large enough to shift a 3 px AUC.
        """
        pts = np.asarray(points, dtype=np.float64)
        if self.scale == 1.0:
            return pts
        return (pts + 0.5) / self.scale - 0.5


def _check_gray(image: NDArray[np.generic]) -> GrayImage:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise PreprocessError(f"expected a uint8 image, got {array.dtype}")
    if array.ndim != 2:
        raise PreprocessError(f"expected a single-channel [H, W] image, got shape {array.shape}")
    # A no-op at runtime (the dtype is uint8 by the check above); it is what carries that
    # fact into the type system, so every recipe's `-> GrayImage` means something.
    return array.astype(np.uint8, copy=False)


def invert(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    """The polarity fix of PLAN.md §15B: hot objects are often dark in the visible image.

    ``config`` is unused, as it is for every parameterless recipe; the uniform signature is
    what lets the registry be a plain dict instead of a dispatch table with special cases.
    """
    del config
    return np.asarray(255 - image, dtype=np.uint8)


def clahe(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    """Contrast-limited adaptive histogram equalisation."""
    operator = cv2.createCLAHE(
        clipLimit=config.clahe_clip, tileGridSize=(config.clahe_tile, config.clahe_tile)
    )
    return np.asarray(operator.apply(image), dtype=np.uint8)


def percentile(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    """Clip to the configured percentile bounds and rescale to the full uint8 range.

    A degenerate window (a constant image, or bounds that coincide) returns mid-grey rather
    than dividing by zero: a flat image genuinely carries no matchable structure, and a NaN
    field would propagate into the metrics as a silently dropped pair.
    """
    low = float(np.percentile(image, config.percentile_low))
    high = float(np.percentile(image, config.percentile_high))
    if high <= low:
        return np.full_like(image, 128)
    scaled = (image.astype(np.float64) - low) / (high - low) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def gradient(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    """Sobel gradient magnitude, min-max rescaled to uint8.

    The modality-invariant representation of the classical cross-modal line (RIFT, CFOG,
    HOPC all build on gradient structure rather than intensity), included so the benchmark
    has a handcrafted structural front-end without yet paying for phase congruency.
    """
    del config
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    peak = float(magnitude.max())
    if peak <= 0.0:
        return np.zeros_like(image)
    return np.asarray(magnitude / peak * 255.0, dtype=np.uint8)


def _identity(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    del config
    return image


def _clahe_invert(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    return invert(clahe(image, config), config)


def _percentile_invert(image: GrayImage, config: PreprocessConfig) -> GrayImage:
    return invert(percentile(image, config), config)


_RECIPES: dict[Variant, Callable[[GrayImage, PreprocessConfig], GrayImage]] = {
    Variant.NONE: _identity,
    Variant.INVERT: invert,
    Variant.CLAHE: clahe,
    Variant.CLAHE_INVERT: _clahe_invert,
    Variant.PERCENTILE: percentile,
    Variant.PERCENTILE_INVERT: _percentile_invert,
    Variant.GRADIENT: gradient,
}


def apply_variant(image: GrayImage, variant: Variant, config: PreprocessConfig) -> GrayImage:
    """Run one named recipe. Geometry-preserving by construction -- no recipe resizes."""
    array = _check_gray(image)
    recipe = _RECIPES.get(variant)
    if recipe is None:  # pragma: no cover - Variant is a closed StrEnum
        raise PreprocessError(f"unknown variant {variant!r}; known: {sorted(_RECIPES)}")
    return recipe(array, config)


def upsample(image: GrayImage, factor: int, interpolation: Interpolation) -> GrayImage:
    """Enlarge by an integer factor. ``factor == 1`` returns the input untouched."""
    array = _check_gray(image)
    if factor == 1:
        return array
    if factor < 1:
        raise PreprocessError(f"upsample factor must be >= 1, got {factor}")
    height, width = array.shape
    return np.asarray(
        cv2.resize(
            array,
            (width * factor, height * factor),
            interpolation=_INTERPOLATION_FLAGS[interpolation],
        ),
        dtype=np.uint8,
    )


def preprocess_reference(image: GrayImage, config: PreprocessConfig) -> Preprocessed:
    """The reference (unwarped) side. Never upsampled: it defines the coordinate frame every
    metric is reported in, and resampling it would move the frame itself."""
    return Preprocessed(image=apply_variant(image, config.reference, config), scale=1.0)


def preprocess_moving(image: GrayImage, config: PreprocessConfig) -> Preprocessed:
    """The moving (warped) side, including the P3-9 upsampling axis.

    Order matters: the recipe runs at native resolution and *then* the result is enlarged.
    Upsampling first would have CLAHE tiles and percentile bounds computed over interpolated
    pixels, making the ablation measure the interaction rather than the factor.
    """
    variant = apply_variant(image, config.moving, config)
    return Preprocessed(
        image=upsample(variant, config.moving_upsample, config.moving_interpolation),
        scale=float(config.moving_upsample),
    )
