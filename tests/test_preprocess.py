"""Preprocessing recipes and the scale plumbing that keeps metrics in native pixels."""

from __future__ import annotations

import numpy as np
import pytest

from cmreg.config import Interpolation, PreprocessConfig, Variant
from cmreg.preprocess import (
    PreprocessError,
    apply_variant,
    invert,
    kernel_for,
    percentile,
    preprocess_moving,
    preprocess_reference,
    rescale,
)

SHAPE = (48, 64)  # non-square, so a transposed index is visible


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, SHAPE, dtype=np.uint8)


@pytest.fixture
def config() -> PreprocessConfig:
    return PreprocessConfig()


@pytest.mark.parametrize("variant", list(Variant))
def test_every_recipe_preserves_geometry_and_dtype(
    variant: Variant, image: np.ndarray, config: PreprocessConfig
) -> None:
    """No recipe may resize: only upsampling is allowed to move coordinates."""
    out = apply_variant(image, variant, config)
    assert out.shape == SHAPE
    assert out.dtype == np.uint8


def test_invert_is_an_involution(image: np.ndarray, config: PreprocessConfig) -> None:
    assert np.array_equal(invert(invert(image, config), config), image)


def test_percentile_stretches_a_narrow_band_to_the_full_range(
    config: PreprocessConfig,
) -> None:
    """A ramp confined to [100, 110] must come back spanning [0, 255].

    A *two-outlier* image would not test this: at 2/98 both bounds land on the constant
    background, the window collapses, and the degenerate branch below is what runs instead.
    """
    ramp = np.tile(np.linspace(100, 110, SHAPE[1]), (SHAPE[0], 1)).astype(np.uint8)
    out = percentile(ramp, config)
    assert out.min() == 0
    assert out.max() == 255


def test_percentile_on_a_constant_image_is_mid_grey(config: PreprocessConfig) -> None:
    """A flat image has no percentile window; mid-grey beats a division by zero, whose NaNs
    would propagate into the metrics as a silently dropped pair."""
    out = percentile(np.full(SHAPE, 42, dtype=np.uint8), config)
    assert np.all(out == 128)


def test_gradient_of_a_constant_image_is_zero(config: PreprocessConfig) -> None:
    assert np.all(apply_variant(np.full(SHAPE, 42, dtype=np.uint8), Variant.GRADIENT, config) == 0)


def test_non_uint8_input_is_rejected(config: PreprocessConfig) -> None:
    with pytest.raises(PreprocessError, match="uint8"):
        apply_variant(
            np.zeros(SHAPE, dtype=np.float32),  # pyright: ignore[reportArgumentType]
            Variant.NONE,
            config,
        )


def test_colour_input_is_rejected(config: PreprocessConfig) -> None:
    with pytest.raises(PreprocessError, match="single-channel"):
        apply_variant(np.zeros((*SHAPE, 3), dtype=np.uint8), Variant.NONE, config)


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_rescale_scales_both_axes(image: np.ndarray, factor: int) -> None:
    out = rescale(image, factor, Interpolation.BICUBIC)
    assert out.image.shape == (SHAPE[0] * factor, SHAPE[1] * factor)
    assert out.scale == factor


def test_rescale_by_one_is_the_identity(image: np.ndarray) -> None:
    assert np.array_equal(rescale(image, 1.0, Interpolation.NEAREST).image, image)


@pytest.mark.parametrize("factor", [0.75, 0.5, 0.25])
def test_rescale_shrinks_both_axes(image: np.ndarray, factor: float) -> None:
    """P3-12a's direction. 48x64 divides evenly by all three, as 512x640 does."""
    out = rescale(image, factor, Interpolation.BICUBIC)
    assert out.image.shape == (round(SHAPE[0] * factor), round(SHAPE[1] * factor))
    assert out.scale == factor


def test_a_factor_that_does_not_preserve_aspect_is_refused(image: np.ndarray) -> None:
    """`to_native` inverts a *single* scale, so an x/y mismatch is a systematic sub-pixel bias
    it cannot represent -- and one that grows with the mismatch and looks like a worse matcher.
    0.3 of 48x64 rounds to 14x19: x0.29167 by x0.29688."""
    with pytest.raises(PreprocessError, match="anisotropic"):
        rescale(image, 0.3, Interpolation.BICUBIC)


def test_a_factor_that_shrinks_below_one_pixel_is_refused(image: np.ndarray) -> None:
    with pytest.raises(PreprocessError, match="below one pixel"):
        rescale(image, 0.001, Interpolation.BICUBIC)


def test_the_downscale_kernel_is_area_whatever_was_configured() -> None:
    """Not a preference. Every other kernel point-samples, so a downscale under bicubic aliases
    structure above the new Nyquist limit into false gradients and the resolution axis would be
    measuring aliasing. Above 1x `INTER_AREA` degenerates to nearest, which is why the direction
    picks rather than the caller."""
    import cv2

    for configured in Interpolation:
        assert kernel_for(0.5, configured) == ("area", cv2.INTER_AREA)
    assert kernel_for(1.0, Interpolation.BICUBIC) == ("bicubic", cv2.INTER_CUBIC)
    assert kernel_for(2.0, Interpolation.LANCZOS) == ("lanczos", cv2.INTER_LANCZOS4)


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_to_native_inverts_the_sampling_grid(image: np.ndarray, factor: int) -> None:
    """The round trip a keypoint makes: native -> upsampled -> native.

    Uses the sampling-grid convention ``(j + 0.5) / s - 0.5`` rather than ``j / s``. The naive
    form passes at s = 1 and leaves a half-pixel bias at every other factor -- small enough to
    survive review, large enough to move a 3 px AUC.
    """
    moving = preprocess_moving(image, PreprocessConfig(moving_upsample=factor))
    native = np.array([[0.0, 0.0], [10.0, 20.0], [63.0, 47.0]])
    upsampled = (native + 0.5) * factor - 0.5
    assert np.allclose(moving.to_native(upsampled), native)


def test_the_reference_side_ignores_the_moving_upsample_axis(image: np.ndarray) -> None:
    """P3-9's factor is the moving side's alone by definition; applying it to both would erase
    the very asymmetry stage C measured (F25)."""
    reference = preprocess_reference(image, PreprocessConfig(moving_upsample=4))
    assert reference.scale == 1.0
    assert reference.image.shape == SHAPE


def test_the_reference_side_does_follow_the_resolution_axis(image: np.ndarray) -> None:
    """P3-12a, and the half of stage F's precondition that did not exist before it: an
    asymmetric resolution sweep would re-measure P3-9's scale mismatch rather than resolution.
    This test fails against the pre-P3-12a `preprocess_reference`, which returned scale 1.0
    unconditionally."""
    reference = preprocess_reference(image, PreprocessConfig(input_scale=0.5))
    assert reference.scale == 0.5
    assert reference.image.shape == (SHAPE[0] // 2, SHAPE[1] // 2)


def test_both_sides_are_resized_by_the_same_factor(image: np.ndarray) -> None:
    """The property that makes the axis resolution rather than scale mismatch."""
    config = PreprocessConfig(input_scale=0.5)
    assert (
        preprocess_reference(image, config).image.shape
        == preprocess_moving(image, config).image.shape
    )


def test_the_two_resize_axes_compose_into_one_resample(image: np.ndarray) -> None:
    """Not a downscale followed by an upscale of the same image, which would lose detail this
    axis exists to measure the loss of and charge the loss to resolution."""
    config = PreprocessConfig(moving=Variant.NONE, input_scale=0.5, moving_upsample=2)
    moving = preprocess_moving(image, config)
    assert moving.scale == 1.0
    assert np.array_equal(moving.image, image)


def test_the_default_resolution_level_leaves_the_moving_side_bit_identical(
    image: np.ndarray,
) -> None:
    """P3-12a is additive: at `input_scale` 1.0 with an integer upsample the output is exactly
    what every stage A-E run produced, which is what keeps those runs reproducible."""
    config = PreprocessConfig(moving=Variant.PERCENTILE, moving_upsample=2)
    expected = rescale(percentile(image, config), 2, config.moving_interpolation)
    assert np.array_equal(preprocess_moving(image, config).image, expected.image)
    assert preprocess_moving(image, config).scale == 2.0


def test_moving_recipe_runs_before_resizing(image: np.ndarray) -> None:
    """Order matters: resizing first would have percentile bounds computed over interpolated
    pixels, so the P3-9 ablation would measure the interaction rather than the factor."""
    config = PreprocessConfig(moving=Variant.PERCENTILE, moving_upsample=2)
    expected = rescale(percentile(image, config), 2, config.moving_interpolation)
    assert np.array_equal(preprocess_moving(image, config).image, expected.image)


@pytest.mark.parametrize("factor", [0.25, 0.5, 0.75, 2.0])
def test_to_native_inverts_a_symmetric_resize(image: np.ndarray, factor: float) -> None:
    """The same round trip as above, on the axis that resizes both sides. Half a pixel is the
    whole content of the test: `j / s` would pass at s = 1 and fail here."""
    moving = preprocess_moving(image, PreprocessConfig(input_scale=factor))
    native = np.array([[0.0, 0.0], [10.0, 20.0], [63.0, 47.0]])
    resized = (native + 0.5) * factor - 0.5
    assert np.allclose(moving.to_native(resized), native)
