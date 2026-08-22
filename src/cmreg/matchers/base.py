"""The ``Matcher`` Protocol, its result type, and the registry (TASKS.md P0-3).

Every matcher -- OpenCV SIFT today, RoMa and MINIMA-RoMa through ``vismatch`` next, RIFT and
LNIFT written fresh after that -- reaches the evaluation cell through this one interface,
returning putative correspondences and *nothing else*. Robust estimation and warping stay
ours: PLAN.md §15A records that ``vismatch``'s own ``compute_ransac`` passes ``ransac_conf``
into ``cv2.findHomography``'s ``mask`` slot, so its confidence silently stays at OpenCV's
0.995 default. A benchmark cannot compare estimators it does not own.

``confidence`` is optional but plumbed from the start. PLAN.md §15G was written against
``vismatch`` 1.2.0, where every dense wrapper dropped RoMa's ``certainty`` on the floor;
**1.3.1 added it** as a seventh ``_forward`` return value and a ``matched_confidences`` key
(``vismatch/base_matcher.py``), so the signal PROSAC needs -- and the signal that samples dense
matches down to a tractable count -- now arrives through the adapter rather than needing a
bypass. The handcrafted backends (``sift-nn``, ``orb-nn``) still return ``None``, which is why
this stays optional rather than becoming required.

**The registry imports lazily, per backend.** Copied from ``vismatch/__init__.py``'s pattern
(PLAN.md §15A): a 17-matcher benchmark has to stay importable on a machine where 15 of the
backends are not installed, which is every machine at some point.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from cmreg.config import MatchConfig
from cmreg.preprocess import GrayImage

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class MatcherError(RuntimeError):
    """Raised for an unknown matcher name or a backend that failed to load."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Putative correspondences between two images, in each image's own pixel coordinates.

    ``kpts0[i]`` corresponds to ``kpts1[i]``; both are ``(N, 2)`` in ``(x, y)``. An empty
    result is legitimate -- a cross-modal pair a detector finds nothing in is a measurement,
    not an exception -- so consumers check ``len``, and the estimator reports the failure.
    """

    kpts0: FloatArray
    kpts1: FloatArray
    # Per-match score in [0, 1], higher is better. `None` when the backend exposes none;
    # PROSAC then refuses to run rather than silently degrading to RANSAC on an arbitrary order.
    confidence: FloatArray | None
    # Detected-keypoint counts before matching. Distinguishes "found nothing" from "found
    # plenty and matched none", which are different failures with different fixes. `None` means
    # the backend does not report a count: a dense matcher has no detection stage at all, and
    # `vismatch`'s dict flattens that case and a genuine zero into the same empty array, so
    # `0` is reserved for "looked and found nothing" and the ambiguous case is null.
    n_detected0: int | None
    n_detected1: int | None
    # Stage-separated timing (PLAN.md §6.5). Kept here rather than measured by the runner
    # because only the matcher knows where detection ends and matching begins.
    extract_ms: float
    match_ms: float

    def __post_init__(self) -> None:
        if self.kpts0.shape != self.kpts1.shape:
            raise MatcherError(f"keypoint counts differ: {self.kpts0.shape} vs {self.kpts1.shape}")
        if self.kpts0.ndim != 2 or self.kpts0.shape[1] != 2:
            raise MatcherError(f"expected (N, 2) keypoints, got {self.kpts0.shape}")
        if self.confidence is not None and self.confidence.shape != (len(self.kpts0),):
            raise MatcherError(
                f"confidence shape {self.confidence.shape} does not match {len(self.kpts0)} matches"
            )

    def __len__(self) -> int:
        return int(self.kpts0.shape[0])


def empty_result(
    n_detected0: int | None, n_detected1: int | None, extract_ms: float, match_ms: float
):
    """A zero-match result. One constructor so the empty arrays always have shape ``(0, 2)``
    rather than ``(0,)`` -- the latter passes ``len() == 0`` and then fails inside numpy."""
    return MatchResult(
        kpts0=np.empty((0, 2), dtype=np.float64),
        kpts1=np.empty((0, 2), dtype=np.float64),
        confidence=np.empty((0,), dtype=np.float64),
        n_detected0=n_detected0,
        n_detected1=n_detected1,
        extract_ms=extract_ms,
        match_ms=match_ms,
    )


@runtime_checkable
class Matcher(Protocol):
    """Anything that turns two grayscale images into putative correspondences."""

    @property
    def name(self) -> str:
        """The registry name. Stamped onto every results row, so it is the join key."""
        ...

    def __call__(self, image0: GrayImage, image1: GrayImage) -> MatchResult: ...


# `(config, device)`, not `(config)`: the device is invocational and must stay out of
# `config_hash()` (`cmreg/device.py` explains why), so it travels as an argument instead.
MatcherFactory = Callable[[MatchConfig, str], Matcher]
_REGISTRY: dict[str, MatcherFactory] = {}


def register(name: str, factory: MatcherFactory) -> None:
    """Add a factory under ``name``. Re-registering is an error: two matchers sharing a name
    would produce results rows that join together and average two different methods."""
    if name in _REGISTRY:
        raise MatcherError(f"matcher {name!r} is already registered")
    _REGISTRY[name] = factory


def available() -> tuple[str, ...]:
    """Every registered matcher name, sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get_matcher(name: str, config: MatchConfig, device: str = "cpu") -> Matcher:
    """Build a matcher by registry name on an already-resolved device.

    ``device`` is a concrete torch device string -- resolve ``"auto"`` through
    ``cmreg.device.resolve_device`` before calling. Defaulting it to ``"cpu"`` keeps the
    OpenCV arm and the test suite callable in one argument.
    """
    _load_builtins()
    factory = _REGISTRY.get(name)
    if factory is None:
        raise MatcherError(f"unknown matcher {name!r}; available: {', '.join(available())}")
    return factory(config, device)


_BUILTINS_LOADED = False


def _load_builtins() -> None:
    """Import the backends that ship with the package.

    Each backend is imported inside its own ``try``, so an unavailable one removes itself from
    the registry instead of taking every other matcher down with it. OpenCV is a hard
    dependency and cannot fail; ``vismatch`` is the optional ``matchers`` extra and routinely
    will -- on a machine synced without it, and on any machine where one of its heavy
    transitive imports breaks. A 20-matcher benchmark has to stay importable either way.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from cmreg.matchers import classical  # noqa: F401  (imported for its registrations)

    try:
        from cmreg.matchers import vismatch_backend  # noqa: F401  (imported to register)
    except ImportError:  # pragma: no cover - exercised only on an unsynced machine
        logger.warning(
            "the vismatch backends are unavailable; only %s will run. "
            "Install them with `uv sync --extra cpu --extra matchers`.",
            ", ".join(sorted(_REGISTRY)),
        )
