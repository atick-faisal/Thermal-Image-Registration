"""The `Matcher` Protocol, its result type, and the registry."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from cmreg.config import MatchConfig
from cmreg.matchers import MatcherError, MatchResult, available, empty_result, get_matcher, register
from cmreg.matchers.vismatch_backend import VISMATCH_MATCHERS
from tests.conftest import textured_image

MATCHERS = ("sift", "orb")
HAS_VISMATCH = importlib.util.find_spec("vismatch") is not None


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return textured_image(np.random.default_rng(3))


def test_registry_lists_the_builtin_matchers() -> None:
    assert set(MATCHERS).issubset(available())


def test_unknown_matcher_names_the_alternatives() -> None:
    with pytest.raises(MatcherError, match="available:"):
        get_matcher("no-such-matcher", MatchConfig())


@pytest.mark.skipif(not HAS_VISMATCH, reason="the `matchers` extra is not installed")
def test_the_vismatch_arm_is_registered() -> None:
    """Registration is gated on `find_spec`, so `available()` never advertises a name that
    `get_matcher` would then refuse -- a sweep must fail at startup, not six hours in."""
    assert set(VISMATCH_MATCHERS).issubset(available())


@pytest.mark.skipif(HAS_VISMATCH, reason="the `matchers` extra is installed")
def test_the_vismatch_arm_is_absent_without_the_extra() -> None:
    assert not set(VISMATCH_MATCHERS) & set(available())


def test_registering_a_duplicate_name_is_refused() -> None:
    """Two matchers under one name would produce rows that join together in the results store
    and silently average two different methods."""
    with pytest.raises(MatcherError, match="already registered"):
        register("sift", lambda config, device: get_matcher("orb", config, device))


@pytest.mark.parametrize("name", MATCHERS)
def test_matching_an_image_against_itself_finds_correspondences(
    name: str, image: np.ndarray
) -> None:
    result = get_matcher(name, MatchConfig())(image, image)
    assert len(result) > 50
    assert result.kpts0.shape == result.kpts1.shape
    # Identical inputs: every correspondence must be a fixed point.
    assert np.allclose(result.kpts0, result.kpts1)


@pytest.mark.parametrize("name", MATCHERS)
def test_a_featureless_image_yields_an_empty_result_not_an_exception(name: str) -> None:
    """A cross-modal pair a detector finds nothing in is a measurement, not an error."""
    flat = np.full((64, 64), 128, dtype=np.uint8)
    result = get_matcher(name, MatchConfig())(flat, flat)
    assert len(result) == 0
    assert result.kpts0.shape == (0, 2)


@pytest.mark.parametrize("name", MATCHERS)
def test_matching_is_deterministic(name: str, image: np.ndarray) -> None:
    """Brute force rather than FLANN precisely so this holds -- FLANN's kd-tree is approximate
    and internally seeded, which would make every fixed-seed assertion in the suite flaky."""
    other = textured_image(np.random.default_rng(11))
    first = get_matcher(name, MatchConfig())(image, other)
    second = get_matcher(name, MatchConfig())(image, other)
    assert np.array_equal(first.kpts0, second.kpts0)
    assert np.array_equal(first.kpts1, second.kpts1)


@pytest.mark.parametrize("name", MATCHERS)
def test_confidence_is_plumbed(name: str, image: np.ndarray) -> None:
    """PROSAC needs a per-match score and refuses to run without one. PLAN.md §15G recorded
    vismatch 1.2.0 dropping it; 1.3.1 returns it, and the OpenCV arm has always had it."""
    result = get_matcher(name, MatchConfig())(image, textured_image(np.random.default_rng(11)))
    assert result.confidence is not None
    assert result.confidence.shape == (len(result),)
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))


@pytest.mark.parametrize("name", MATCHERS)
def test_a_detector_that_finds_nothing_reports_zero_not_null(name: str) -> None:
    """`0` and `None` are different measurements: `0` is "looked and found nothing", `None` is
    "this method has no detection stage". Collapsing them makes the column meaningless."""
    flat = np.full((64, 64), 128, dtype=np.uint8)
    result = get_matcher(name, MatchConfig())(flat, flat)
    assert result.n_detected0 == 0
    assert result.n_detected1 == 0


def test_empty_result_uses_two_column_arrays() -> None:
    """Shape (0,) would pass `len() == 0` and then fail inside numpy at the first indexing."""
    result = empty_result(0, 0, 1.0, 2.0)
    assert result.kpts0.shape == (0, 2)
    assert result.kpts1.shape == (0, 2)


def test_mismatched_keypoint_counts_are_rejected() -> None:
    with pytest.raises(MatcherError, match="counts differ"):
        MatchResult(
            kpts0=np.zeros((3, 2)),
            kpts1=np.zeros((4, 2)),
            confidence=None,
            n_detected0=3,
            n_detected1=4,
            extract_ms=0.0,
            match_ms=0.0,
        )


def test_confidence_length_must_match_the_correspondences() -> None:
    with pytest.raises(MatcherError, match="confidence shape"):
        MatchResult(
            kpts0=np.zeros((3, 2)),
            kpts1=np.zeros((3, 2)),
            confidence=np.zeros(2),
            n_detected0=3,
            n_detected1=3,
            extract_ms=0.0,
            match_ms=0.0,
        )
