"""Preprocessing recipes and the scale plumbing that keeps metrics in native pixels."""

from __future__ import annotations

import numpy as np
import pytest

from cmreg.config import Interpolation, PreprocessConfig, Variant
from cmreg.preprocess import (
    PreprocessError,
    apply_variant,
    invert,
    percentile,
    preprocess_moving,
    preprocess_reference,
    upsample,
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
def test_upsample_scales_both_axes(image: np.ndarray, factor: int) -> None:
    out = upsample(image, factor, Interpolation.BICUBIC)
    assert out.shape == (SHAPE[0] * factor, SHAPE[1] * factor)


def test_upsample_by_one_is_the_identity(image: np.ndarray) -> None:
    assert np.array_equal(upsample(image, 1, Interpolation.NEAREST), image)


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


def test_reference_side_is_never_upsampled(image: np.ndarray) -> None:
    """It defines the frame every metric is reported in; resampling it moves the frame."""
    reference = preprocess_reference(image, PreprocessConfig(moving_upsample=4))
    assert reference.scale == 1.0
    assert reference.image.shape == SHAPE


def test_moving_recipe_runs_before_upsampling(image: np.ndarray) -> None:
    """Order matters: upsampling first would have percentile bounds computed over interpolated
    pixels, so the P3-9 ablation would measure the interaction rather than the factor."""
    config = PreprocessConfig(moving=Variant.PERCENTILE, moving_upsample=2)
    expected = upsample(percentile(image, config), 2, config.moving_interpolation)
    assert np.array_equal(preprocess_moving(image, config).image, expected)
