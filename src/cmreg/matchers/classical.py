"""OpenCV detect-describe-match baselines: SIFT and ORB.

The classical arm PLAN.md §3 says reviewers expect, and -- more immediately -- the CPU-only
floor that lets the evaluation cell be validated end to end without weights, a GPU, or a
network. They are also the honest control for the P3-8 generality claim: if inverting the
optical grayscale helps SIFT as much as it helps RoMa, the effect is about polarity and not
about learned features.

Ratio-tested brute force rather than FLANN: FLANN's kd-tree is approximate and seeded
internally, so two runs of the same config would disagree by a few matches and every
"identical seed, identical result" test in the suite would become flaky. Brute force is
exact, deterministic, and at ``max_keypoints`` of a few thousand it is not the bottleneck.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.config import MatchConfig
from cmreg.matchers.base import FloatArray, MatchResult, empty_result, register
from cmreg.preprocess import GrayImage

logger = logging.getLogger(__name__)

# SIFT descriptors are float32 and compared with L2; ORB's are binary and compared with
# Hamming. Using the wrong norm does not raise -- it returns confident nonsense.
_SIFT_NORM = cv2.NORM_L2
_ORB_NORM = cv2.NORM_HAMMING


class _OpenCVMatcher:
    """Shared detect -> describe -> ratio-test body for a cv2 ``Feature2D``."""

    def __init__(self, name: str, detector: cv2.Feature2D, norm: int, config: MatchConfig) -> None:
        self._name = name
        self._detector = detector
        self._norm = norm
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, image0: GrayImage, image1: GrayImage) -> MatchResult:
        start = time.perf_counter()
        kpts0, desc0 = self._detector.detectAndCompute(image0, None)
        kpts1, desc1 = self._detector.detectAndCompute(image1, None)
        extract_ms = (time.perf_counter() - start) * 1e3

        n0, n1 = len(kpts0), len(kpts1)
        # `detectAndCompute` returns `None` descriptors for a featureless image, and two
        # descriptors are the minimum knnMatch(k=2) can be asked for.
        if desc0 is None or desc1 is None or n0 < 2 or n1 < 2:
            logger.debug("%s: too few keypoints (%d / %d)", self._name, n0, n1)
            return empty_result(n0, n1, extract_ms, 0.0)

        start = time.perf_counter()
        pairs = cv2.BFMatcher(self._norm, crossCheck=False).knnMatch(desc0, desc1, k=2)
        points0, points1, scores = _ratio_test(pairs, kpts0, kpts1, self._config.ratio_test)
        match_ms = (time.perf_counter() - start) * 1e3

        return MatchResult(
            kpts0=points0,
            kpts1=points1,
            confidence=scores,
            n_detected0=n0,
            n_detected1=n1,
            extract_ms=extract_ms,
            match_ms=match_ms,
        )


def _ratio_test(
    pairs: Sequence[Sequence[cv2.DMatch]],
    kpts0: Sequence[cv2.KeyPoint],
    kpts1: Sequence[cv2.KeyPoint],
    ratio: float,
) -> tuple[FloatArray, FloatArray, NDArray[np.float64]]:
    """Lowe's ratio test, with the ratio itself carried out as a per-match confidence.

    ``1 - d_best / d_second`` rather than the raw ratio: PROSAC and every sampling step want
    "higher is better" in [0, 1], and defining that once here keeps each consumer from
    inventing its own sign convention.
    """
    points0: list[tuple[float, float]] = []
    points1: list[tuple[float, float]] = []
    scores: list[float] = []
    for candidates in pairs:
        # knnMatch can return fewer than k candidates for a query near the end of the set.
        if len(candidates) < 2:
            continue
        best, second = candidates[0], candidates[1]
        if second.distance <= 0.0 or best.distance >= ratio * second.distance:
            continue
        # `.pt` is typed as a bare float sequence; unpacking pins the two-element shape the
        # `(N, 2)` array below depends on.
        x0, y0 = kpts0[best.queryIdx].pt
        x1, y1 = kpts1[best.trainIdx].pt
        points0.append((x0, y0))
        points1.append((x1, y1))
        scores.append(1.0 - best.distance / second.distance)

    if not points0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.asarray(points0, dtype=np.float64),
        np.asarray(points1, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    )


# `device` is part of the `MatcherFactory` signature and ignored here: OpenCV's detectors run
# on the CPU regardless. Accepted rather than special-cased so the registry has one calling
# convention -- a factory table with two shapes is a `TypeError` waiting for whichever matcher
# is added next.
def _sift(config: MatchConfig, device: str) -> _OpenCVMatcher:
    del device
    return _OpenCVMatcher(
        "sift", cv2.SIFT.create(nfeatures=config.max_keypoints), _SIFT_NORM, config
    )


def _orb(config: MatchConfig, device: str) -> _OpenCVMatcher:
    del device
    return _OpenCVMatcher("orb", cv2.ORB.create(nfeatures=config.max_keypoints), _ORB_NORM, config)


# SIFT and ORB are the only detectors left in the main module: opencv-python 5.0 dropped
# AKAZE, KAZE and BRISK from `cv2` (verified on 5.0.0 -- `dir(cv2)` has none of them), so a
# wider classical arm needs opencv-contrib. Not worth a second OpenCV wheel for a baseline
# nobody in the cross-modal literature reports; RIFT and LNIFT are the classical entries that
# matter here and they are net-new anyway (PLAN.md §15G).
register("sift", _sift)
register("orb", _orb)
