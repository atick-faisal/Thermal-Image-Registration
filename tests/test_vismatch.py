"""End-to-end smoke tests for the `vismatch` arm (TASKS.md P0-2).

Every test here is `slow` and downloads weights, so the pre-push gate (`pytest -m "not slow"`)
deselects the lot and the README's "nothing in the suite touches the network" rule survives.
Run them deliberately: `uv run pytest -m slow -k vismatch`.

What they are for: the adapter's one real risk is a **coordinate-convention error**. Every
`vismatch` wrapper resizes internally and maps its keypoints back through
`BaseMatcher.rescale_coords`; if that mapping is off, or if our `(H, W) uint8 -> (3, H, W)
float` conversion transposes an axis, the matcher still returns plausible-looking
correspondences and every metric downstream is quietly wrong. The check is therefore not "did
it return matches" but "does the estimated homography equal the one we applied" -- the same
shape as `tests/test_runner.py`'s same-modality pin.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from cmreg.config import EstimateConfig, MatchConfig
from cmreg.estimate import estimate_warp
from cmreg.matchers import get_matcher
from cmreg.metrics import corner_error
from cmreg.warp import apply_warp
from tests.conftest import textured_image

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("vismatch") is None,
        reason="the `matchers` extra is not installed",
    ),
]

# The cheapest learned matcher in the arm: a few MB of weights, CPU-fast, and detector-based,
# so it also exercises the `all_kpts` path a dense matcher leaves empty. Enough to prove the
# adapter; the per-backend triage that proves the *other* seventeen load is a separate script.
SMOKE_MATCHER = "xfeat"

SHAPE = (384, 512)


def _known_homography() -> np.ndarray:
    """A modest rotation + scale + translation, well inside GTConfig's sampling range."""
    angle = np.deg2rad(8.0)
    scale = 1.06
    center = np.array([SHAPE[1] / 2.0, SHAPE[0] / 2.0])
    rotation = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    offset = center - rotation @ center + np.array([9.0, -6.0])
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :2] = rotation
    homography[:2, 2] = offset
    return homography


def test_the_adapter_recovers_a_known_homography() -> None:
    """If this passes, keypoints are in the caller's pixel coordinates and the axes are right."""
    reference = textured_image(np.random.default_rng(5), SHAPE)
    homography = _known_homography()
    moved = np.asarray(apply_warp(reference, homography, out_shape=SHAPE), np.uint8)

    result = get_matcher(SMOKE_MATCHER, MatchConfig())(reference, moved)
    assert len(result) > 50, f"{SMOKE_MATCHER} found {len(result)} matches on a same-modality pair"

    # src is the moved image, dst is the reference: the same direction `eval/runner.py`
    # estimates in, so a sign error here is a sign error there.
    estimate = estimate_warp(result.kpts1, result.kpts0, EstimateConfig(), result.confidence)
    assert estimate.h is not None
    assert corner_error(estimate.h, np.linalg.inv(homography), SHAPE) < 2.0


def test_detection_counts_and_confidence_are_reported_honestly() -> None:
    """`n_detected` is null only when vismatch cannot distinguish zero from absent; xfeat is a
    detector, so it must report a real count."""
    image = textured_image(np.random.default_rng(7), SHAPE)
    result = get_matcher(SMOKE_MATCHER, MatchConfig())(image, image)

    assert result.n_detected0 is not None and result.n_detected0 > 0
    assert result.n_detected1 is not None and result.n_detected1 > 0
    # vismatch's forward() is monolithic; the whole cost lands in match_ms (module docstring).
    assert result.extract_ms == 0.0
    assert result.match_ms > 0.0


def test_ransac_stays_ours() -> None:
    """`skip_ransac` must be forced on: `vismatch/base_matcher.py::compute_ransac` passes its
    confidence into `cv2.findHomography`'s `mask` slot, and a benchmark cannot compare
    estimators it does not own."""
    matcher = get_matcher(SMOKE_MATCHER, MatchConfig())
    matcher(
        textured_image(np.random.default_rng(9), SHAPE),
        textured_image(np.random.default_rng(9), SHAPE),
    )
    assert matcher._load().skip_ransac is True  # type: ignore[attr-defined]
