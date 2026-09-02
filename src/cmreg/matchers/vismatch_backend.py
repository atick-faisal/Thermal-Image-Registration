"""The learned-matcher arm, adapted from ``vismatch`` (TASKS.md P0-2, PLAN.md §15A).

``vismatch`` (gmberton's *Image Matching Models*) wraps ~70 matchers behind one interface and
covers PLAN.md §4.1's list except RIFT and LNIFT. We depend on it rather than vendoring
seventeen upstream repos, but only for *correspondences*:

* **``skip_ransac`` is forced on.** ``vismatch/base_matcher.py::compute_ransac`` passes
  ``ransac_conf`` into ``cv2.findHomography``'s fifth positional slot, which is ``mask``, not
  ``confidence`` -- so its confidence silently stays at OpenCV's 0.995 default. Beyond that
  bug, a benchmark cannot compare estimators it does not own (``cmreg/estimate/robust.py``).
* **Coordinates come back in the caller's pixels.** Every wrapper resizes internally and maps
  its keypoints back through ``BaseMatcher.rescale_coords`` before returning, and 1.3.1 also
  drops matches that land outside either image. ``tests/test_vismatch.py`` pins this
  end-to-end rather than trusting it.

Two things changed under us since PLAN.md was written, both in our favour:

1. **Confidence is no longer dropped.** §15G recorded that every dense wrapper discarded
   RoMa's ``certainty``; ``vismatch`` 1.3.1 returns it as a seventh ``_forward`` value and a
   ``matched_confidences`` key. PROSAC therefore works against the learned arm.
2. **MPS is supported** for torch >= 2.11, and ``RomaMatcher`` now pins float32 off CUDA --
   1.2.0's fp16 default overflowed DINOv2's activations to NaN on MPS.

**``max_keypoints`` is not universal.** It reaches ``get_matcher``'s ``max_num_keypoints``,
which xfeat, the LightGlue family and the handcrafted pair honour. The detector-free entries
do not: ``minima-roma`` returned 10 000 matches under a 2 048 budget because
``vismatch/im_models/minima.py`` calls ``self.matcher.sample(warp, certainty)`` with no
``num=``, falling back to RoMa's default, and ``matchanything.py`` has no keypoint budget at
all -- its match count comes from its own coarse grid. Not silently capped here: *which*
matches to drop (top-k by confidence? uniform?) is a methodological choice that belongs to
P3-12's match-count ablation, and truncating in the adapter would bury it.

**Resolved by P3-12c, and not by plumbing a budget through here.** The axis moved *downstream*
instead: ``estimate.max_matches`` caps the correspondences fed to the fit
(``cmreg/estimate/select.py``), which is defined for every backend and costs one solver call
rather than one match pass, where a per-backend budget would have been inert for exactly the
three entries above and would have measured matcher cost rather than correspondence count. This
paragraph stays as the record of why ``max_keypoints`` means different things across the
registry -- ``vismatch`` upstream is unchanged (im_models/minima.py, the missing ``num=``;
https://github.com/gmberton/vismatch).

What is *not* adapted: stage-separated timing. ``vismatch``'s ``forward()`` is one call, and
for a dense matcher extraction and matching are not separable even in principle -- there is no
detection stage. The whole cost is reported as ``match_ms`` with ``extract_ms`` at zero, so
``time/total_ms`` is the comparable runtime column and ``time/extract_ms`` means something
only for the OpenCV arm (PLAN.md §6.5, TASKS.md P3-14).
"""

from __future__ import annotations

import importlib.util
import logging
import time
from typing import Any

import numpy as np

from cmreg.config import MatchConfig
from cmreg.matchers.base import MatcherError, MatchResult, register
from cmreg.preprocess import GrayImage

logger = logging.getLogger(__name__)

# PLAN.md §4.1's matcher list, under vismatch's own spellings. Registered verbatim so a results
# row's `matcher` column is a name that can be looked up upstream; `sift-nn`/`orb-nn` do not
# collide with our OpenCV `sift`/`orb`, and keeping both is the honest comparison of two SIFT
# implementations rather than a claim that they are interchangeable.
#
# Failure mode of adding a name here carelessly: `available()` advertises a matcher whose
# weights are a 3 GB download or whose backend needs a CUDA extension, and a sweep discovers it
# six hours in. Everything below is triaged in TASKS.md P0-2 before it enters a config.
VISMATCH_MATCHERS: tuple[str, ...] = (
    # Dense / detector-free -- the arm the project is actually about.
    "roma",
    "romav2",
    "tiny-roma",
    "minima-roma",
    "minima-loftr",
    "minima-xoftr",
    "matchanything-roma",
    "matchanything-eloftr",
    "xoftr",
    "eloftr",
    "loftr",
    "gim-dkm",
    # Detector + learned matcher.
    "superpoint-lightglue",
    "disk-lightglue",
    "aliked-lightglue",
    "xfeat",
    # vismatch's own handcrafted baselines, kept beside ours as an implementation control.
    "sift-nn",
    "orb-nn",
)


def _to_rgb_tensor(image: GrayImage) -> Any:
    """``(H, W)`` uint8 -> ``(3, H, W)`` float in ``[0, 1]``, channel-replicated.

    ``vismatch/utils.py::to_tensor_image`` asserts exactly that shape and range. Replicating
    the single channel is not a loss: our preprocessing front-end (``preprocess/variants.py``)
    has already reduced both modalities to one channel by design, and the sibling
    implementations PLAN.md §15B records feed RoMa's DINOv2 the same way.
    """
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(torch.float32).div_(255.0)
    return tensor.unsqueeze(0).expand(3, -1, -1)


def _detected(all_kpts: np.ndarray) -> int | None:
    """Detection count, or ``None`` when ``vismatch`` cannot distinguish zero from absent.

    ``BaseMatcher.get_empty_array_if_none`` turns a dense matcher's ``all_kpts0=None`` into the
    same ``(0, 2)`` array a detector that found nothing would produce. Reporting `0` for both
    would put "RoMa has no detection stage" and "SuperPoint fired on nothing" in one column, so
    the ambiguous case is null and only a positive count is asserted.
    """
    return len(all_kpts) or None


class _VismatchMatcher:
    """One ``vismatch`` model behind our ``Matcher`` Protocol.

    The model is built on first call rather than in the factory: ``cmreg matchers`` and config
    validation both construct every configured matcher, and a factory that downloaded weights
    would make listing the registry a multi-gigabyte operation.
    """

    __slots__ = ("_config", "_device", "_model", "_name")

    def __init__(self, name: str, config: MatchConfig, device: str) -> None:
        self._name = name
        self._config = config
        self._device = device
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from vismatch import get_matcher as vismatch_get_matcher
        from vismatch.utils import disable_xformers

        # PLAN.md §15A: xformers is CUDA-only, and a DINOv2-backed model crashes on CPU/MPS
        # when it happens to be installed. RomaMatcher calls this itself; DeDoDe-Kornia and
        # anything else DINOv2-backed may not, so it is called unconditionally off CUDA.
        if not self._device.startswith("cuda"):
            disable_xformers()

        logger.info("loading vismatch matcher %r on %s", self._name, self._device)
        try:
            model = vismatch_get_matcher(
                self._name, device=self._device, max_num_keypoints=self._config.max_keypoints
            )
        except Exception as exc:
            # Deliberately broad: upstream raises bare `Exception`, `RuntimeError`,
            # `AssertionError` and `ImportError` depending on which backend failed. They all
            # mean the same thing here, and the P0-2 triage table wants the reason, not the
            # type -- catching narrowly would let one backend's novel exception abort a sweep.
            raise MatcherError(f"vismatch could not build {self._name!r}: {exc}") from exc

        # Estimation and warping are ours. Set after construction because `BaseMatcher.__init__`
        # hardcodes it to False and no constructor argument exposes it.
        model.skip_ransac = True
        self._model = model
        return model

    def __call__(self, image0: GrayImage, image1: GrayImage) -> MatchResult:
        model = self._load()
        tensor0, tensor1 = _to_rgb_tensor(image0), _to_rgb_tensor(image1)

        start = time.perf_counter()
        output = model(tensor0, tensor1)
        match_ms = (time.perf_counter() - start) * 1e3

        confidence = output["matched_confidences"]
        return MatchResult(
            kpts0=np.asarray(output["matched_kpts0"], dtype=np.float64),
            kpts1=np.asarray(output["matched_kpts1"], dtype=np.float64),
            confidence=None if confidence is None else np.asarray(confidence, dtype=np.float64),
            n_detected0=_detected(output["all_kpts0"]),
            n_detected1=_detected(output["all_kpts1"]),
            # See the module docstring: vismatch's forward() is monolithic, and a dense matcher
            # has no extraction stage to separate out in the first place.
            extract_ms=0.0,
            match_ms=match_ms,
        )


def _factory(name: str):
    def build(config: MatchConfig, device: str) -> _VismatchMatcher:
        return _VismatchMatcher(name, config, device)

    return build


def _register_all() -> None:
    """Register every name in :data:`VISMATCH_MATCHERS`, if ``vismatch`` is importable.

    Gated on ``find_spec`` rather than on an import: the check is cheap, and it means
    ``available()`` never advertises a name that ``get_matcher`` would then refuse. A machine
    synced without the ``matchers`` extra simply reports the OpenCV arm.
    """
    if importlib.util.find_spec("vismatch") is None:
        logger.debug("vismatch is not installed; the learned matcher arm is unavailable")
        return
    for name in VISMATCH_MATCHERS:
        register(name, _factory(name))


_register_all()
